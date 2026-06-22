"""Rotation refinement for the CLiMB submission (v006).

Why: RPE_rot uses ONLY the trajectory rotations while ATE uses only the
centres, so rotations can be replaced without touching ATE. The DA3 chain's
rotations are regression outputs whose per-step error (~1-2 deg) random-walks
to 5-7 deg at delta=40; epipolar rotations from ALIKED+LightGlue matches are
2x more accurate per edge, and LONG edges (7 nodes = ~42 frames) measure the
delta=40 quantity directly instead of accumulating it. Robust SO(3) averaging
over {feature edges (weight = inliers, Huber, 2-round rejection)} +
{DA3 edges (constant weight = the safety net where matching fails)} cut
rot@40 from 5.26 to 3.46 deg on the real 4-clip CV with ATE bit-identical.
"""
import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
import time
import numpy as np
import cv2


def photo_gain(K, hw, mu, D, cap=4.0):
    """Per-pixel gain that undoes the light's cos^mu angular falloff.

    The endoscope's light is rigidly attached to the camera, so irradiance goes
    as sigma * cos^mu(theta) / d^2 with theta the angle of the viewing ray from
    the light's principal direction. d needs depth and is not available here,
    but theta depends only on the intrinsics -- a fixed 2D map. Removing it
    means a surface point keeps its brightness as it drifts from the centre to
    the periphery, which is the illumination-only appearance change that stops
    a descriptor from matching itself across frames.

    Normalised to unit mean so overall exposure (and thus the detector's
    thresholds) is preserved: the centre is darkened as much as the rim is
    lifted, rather than pushing highlights into saturation.
    """
    h, w = hw
    u = (np.arange(w, dtype=np.float64) - K[0, 2]) / K[0, 0]
    v = (np.arange(h, dtype=np.float64) - K[1, 2]) / K[1, 1]
    U, V = np.meshgrid(u, v)
    r = np.stack([U, V, np.ones_like(U)], -1)
    r /= np.linalg.norm(r, axis=-1, keepdims=True)
    d = np.asarray(D, np.float64)
    d /= np.linalg.norm(d)
    c = np.clip(r @ d, 1e-6, 1.0)
    g = np.minimum(c ** (-mu), cap)
    return (g / g.mean()).astype(np.float32)


def photo_correct(im, gain, gamma=2.2):
    """Apply the gain in linear radiance, not on the gamma-encoded pixels."""
    x = (im.astype(np.float32) / 255.0) ** gamma
    x *= gain[..., None] if x.ndim == 3 else gain
    return (np.clip(x, 0.0, 1.0) ** (1.0 / gamma) * 255.0).astype(np.uint8)


def _despec(im, thr=245, dil=5, feather=11, sigma=6.0):
    """Suppress specular highlights before keypoint extraction.

    The endoscope light is co-located with the camera, so highlights are locked
    to the CAMERA frame: they stay put in the image while the world rotates
    under them. ALIKED sees them as the strongest blobs in an otherwise
    texture-poor mucosa, LightGlue happily matches highlight-to-highlight, and
    those correspondences vote for "no rotation". Where real texture is
    plentiful (Seq_003_a, 5.07 surviving edges/node) MAGSAC rejects them; where
    it is not (Seq_001_c 1.11, Seq_003_b 1.57) they are the majority and the
    edge is either dropped or wrong -- which is exactly the pattern we measured.

    Blending a blurred copy under a feathered mask (rather than cropping or
    inpainting) keeps the keypoint COUNT fixed at top-k, so the batched
    LightGlue path -- the thing that made v007 fit the runtime budget at all --
    still applies.
    """
    m = (im.max(2) >= thr).astype(np.uint8)
    if m.mean() < 0.002:
        return im
    m = cv2.dilate(m, np.ones((dil, dil), np.uint8))
    w = cv2.GaussianBlur(m.astype(np.float32), (feather, feather), 0)[..., None]
    return (im * (1.0 - w) + cv2.GaussianBlur(im, (0, 0), sigma) * w).astype(np.uint8)


SOLVER = os.environ.get("DA3_ROT_SOLVER", "essential")   # essential | rot2pt
# Where the rotavg tail actually goes. GPU work shrinks ~4x on the eval GPU
# while pure-CPU work (MAGSAC, recoverPose) does not, so the split decides
# what is worth optimising.
PROF = int(os.environ.get("DA3_ROT_PROF", "0"))
# MAGSAC + recoverPose are pure CPU and release the GIL, so running them in a
# side pool lets node k's robust fit overlap node k+1's ALIKED/LightGlue on the
# GPU instead of serialising behind it. This matters only on a fast GPU -- which
# is exactly the eval host, where the GPU half shrinks ~4x and the CPU half does
# not. 0 disables the pool (sequential, previous behaviour).
ROT_POOL = int(os.environ.get("DA3_ROT_POOL", "4"))
TRUNC_RATIO = float(os.environ.get("DA3_ROT_TRUNC", "0.75"))
DETECT_THR = float(os.environ.get("DA3_ROT_DTHR", "0.0"))
PHOTO_ON = int(os.environ.get("DA3_ROT_PHOTO", "0"))
_T = {"extract": 0.0, "match": 0.0, "match1": 0.0, "ransac": 0.0,
      "n_edge": 0, "n_batched": 0, "n_single": 0}
GATE = int(os.environ.get("DA3_ROT_GATE", "0"))
GATE_SOFT = float(os.environ.get("DA3_ROT_GATE_SOFT", 1.0))
GATE_HARD = float(os.environ.get("DA3_ROT_GATE_HARD", 1.6))


def _bearings(p, K):
    """Pixels -> unit rays in the camera frame."""
    x = (p[:, 0] - K[0, 2]) / K[0, 0]
    y = (p[:, 1] - K[1, 2]) / K[1, 1]
    v = np.stack([x, y, np.ones_like(x)], 1)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _kabsch(f0, f1, w=None):
    """R minimising sum w_i ||f1_i - R f0_i||^2 (Wahba / Kabsch on unit vectors)."""
    M = (f1 * (w[:, None] if w is not None else 1.0)).T @ f0
    U, _, Vt = np.linalg.svd(M)
    return U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt


def _rot_only(f0, f1, thr_deg=2.5, iters=64, min_inl=15, rng=None):
    """Rotation-only fit: 2-point RANSAC on bearings, then Huber IRLS.

    Why not the essential matrix: under the forward motion that dominates
    colonoscopy the epipole sits at the image centre, parallax angles collapse,
    and the 5-point problem is degenerate -- which is exactly where our measured
    error lives (RPE decomposed onto the camera axes puts 1.2-4.4x more error on
    pitch/yaw, perpendicular to the optical axis, than on roll about it).
    Fitting rotation alone sidesteps that: a radial parallax field is
    rotationally symmetric, so forward translation adds noise but no systematic
    rotation. Two correspondences suffice instead of five, so this is also
    cheaper than the MAGSAC 5-point search it replaces -- the binding constraint
    is the runtime budget (T <= 0.0198 s/frame, 0.0028 spare).
    """
    n = len(f0)
    if n < min_inl:
        return None, 0
    rng = rng or np.random.default_rng(0)
    cthr = np.cos(np.radians(thr_deg))
    best_R, best_n = None, 0
    for _ in range(iters):
        i, j = rng.choice(n, 2, replace=False)
        if abs(float(f0[i] @ f0[j])) > 0.9995:      # degenerate: near-parallel
            continue
        R = _kabsch(f0[[i, j]], f1[[i, j]])
        c = np.einsum("ij,ij->i", f1, (R @ f0.T).T)
        k = int((c > cthr).sum())
        if k > best_n:
            best_R, best_n = R, k
    if best_R is None or best_n < min_inl:
        return None, 0
    R = best_R
    for _ in range(6):                               # Huber IRLS on the angle
        ang = np.arccos(np.clip(np.einsum("ij,ij->i", f1, (R @ f0.T).T), -1, 1))
        d = np.radians(thr_deg)
        w = np.where(ang <= d, 1.0, d / np.maximum(ang, 1e-9))
        w[ang > 6 * d] = 0.0
        if w.sum() < min_inl:
            break
        R = _kabsch(f0, f1, w)
    ang = np.arccos(np.clip(np.einsum("ij,ij->i", f1, (R @ f0.T).T), -1, 1))
    return R, int((ang <= np.radians(thr_deg)).sum())


def _motion_gate(p0, p1):
    """Circular std of the flow direction (Track2Map, arXiv:2607.08408).

    R = |mean(exp(i*theta))| over the flow directions; sigma = sqrt(-2 ln R).
    Low sigma means one coherent motion field = camera-dominated, so the rigid
    rotation is meaningful; high sigma means the field is incoherent, which in
    colonoscopy is tissue deformation rather than camera motion. Costs one
    arctan2 and a mean.
    """
    d = p1 - p0
    m = np.linalg.norm(d, axis=1) > 1e-6
    if m.sum() < 8:
        return 0.0
    th = np.arctan2(d[m, 1], d[m, 0])
    Rlen = float(np.hypot(np.cos(th).mean(), np.sin(th).mean()))
    return float(np.sqrt(max(0.0, -2.0 * np.log(max(Rlen, 1e-12)))))


def _proj_so3(M):
    U, _, Vt = np.linalg.svd(M)
    return U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt


def _geo(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def _solve(R_init, edges, iters=30, huber_deg=2.0):
    """Robust rotation averaging (Gauss-Seidel + Huber), vectorised per node.

    This runs after the last frame, so it is a serial tail that no amount of GPU
    speed shortens: profiled at 4.5 s on a 975-frame clip, 23% of that clip's
    entire 19.3 s runtime budget.

    The arithmetic below is unchanged -- same sweep order, same Huber weights,
    same SVD projection, same stopping rule -- only the per-node work is now a
    handful of batched numpy calls instead of a Python loop over incident edges.
    That matters because the alternatives are not equivalent: a vectorised
    Jacobi sweep converges to a visibly worse point on these near-chain graphs
    (one hop per sweep), and a tangent-space Gauss-Newton converges to a BETTER
    robust cost but a different solution -- it moved the fused rotations 16-24
    deg off the DA3 chain against Gauss-Seidel's 5-8 deg, because 30 early-
    stopped sweeps are themselves acting as the regulariser this objective is
    tuned around.
    """
    n = len(R_init)
    R = np.stack([np.asarray(r, np.float64) for r in R_init])
    # Incident edges per node, pre-stacked so the sweep touches only arrays.
    nb = [[] for _ in range(n)]
    rel = [[] for _ in range(n)]
    wts = [[] for _ in range(n)]
    for (a, b, Rab, w) in edges:
        nb[a].append(b); rel[a].append(Rab);           wts[a].append(w)
        nb[b].append(a); rel[b].append(Rab.T.copy());  wts[b].append(w)
    act = [i for i in range(n) if nb[i]]
    NB = {i: np.asarray(nb[i], np.int64) for i in act}
    RL = {i: np.stack(rel[i]).astype(np.float64) for i in act}
    WT = {i: np.asarray(wts[i], np.float64) for i in act}
    for _ in range(iters):
        delta = 0.0
        for i in act:
            P = R[NB[i]] @ RL[i]                                   # (d,3,3)
            c = np.clip((np.einsum("kij,ij->k", P, R[i]) - 1.0) * 0.5, -1.0, 1.0)
            res = np.degrees(np.arccos(c))
            hw = np.where(res <= huber_deg, 1.0,
                          huber_deg / np.maximum(res, 1e-9))
            M = np.tensordot(WT[i] * hw, P, axes=(0, 0))
            U, _, Vt = np.linalg.svd(M)
            Rn = U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt
            d = np.degrees(np.arccos(
                np.clip((np.einsum("ij,ij->", R[i], Rn) - 1.0) * 0.5, -1.0, 1.0)))
            if d > delta:
                delta = d
            R[i] = Rn
        if delta < 0.01:
            break
    return [R[i] for i in range(n)]


def refine(extr_g, rot_cache, K_half, log=print,  # noqa: C901
           kp=None, steps=None, min_inl=15, w_da3=None):
    """extr_g: (n,4,4) w2c full trajectory. rot_cache: {frame_idx: encoded jpg}.
    K_half: pinhole K of the cached (rectified+cropped, half-res) frames.
    Returns extr_g with rotations replaced (centres untouched)."""
    _t0 = time.perf_counter()
    kp = kp or int(os.environ.get("DA3_ROT_KP", 768))
    if w_da3 is None:
        w_da3 = float(os.environ.get("DA3_ROT_WDA3", "30"))
    steps = steps or tuple(int(x) for x in os.environ.get("DA3_ROT_STEPS", "1,4,7").split(","))
    try:
        import torch
        from lightglue import ALIKED, LightGlue
        from lightglue.utils import rbd
    except Exception as exc:
        log(f"  rotavg: lightglue unavailable ({type(exc).__name__}: {exc}); skipping")
        return extr_g

    nodes = sorted(rot_cache)
    if len(nodes) < 3:
        return extr_g
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fp16 = dev == "cuda" and int(os.environ.get("DA3_ROT_FP16", "1"))
    ext = ALIKED(max_num_keypoints=kp, detection_threshold=DETECT_THR).eval().to(dev)
    # adaptive early-exit: prune confident points / stop at confident depth
    lg = LightGlue(features="aliked", depth_confidence=0.9,
                   width_confidence=0.95).eval().to(dev)
    from contextlib import nullcontext
    amp = (lambda: torch.autocast("cuda", dtype=torch.float16)) if fp16 else nullcontext

    feats = {}
    def feat(k):
        if k not in feats:
            buf = rot_cache[nodes[k]]
            im = buf if isinstance(buf, np.ndarray) else \
                cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
            t = torch.from_numpy(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().div(255).to(dev)[None]
            with torch.no_grad(), amp():
                feats[k] = ext.extract(t)
        return feats[k]

    ef = []
    for st in steps:
        for i in range(len(nodes) - st):
            f0, f1 = feat(i), feat(i + st)
            with torch.no_grad(), amp():
                m = lg({"image0": f0, "image1": f1})
            f0r, f1r, mr = [rbd(x) for x in (f0, f1, m)]
            idx = mr["matches"].cpu().numpy()
            if len(idx) < min_inl:
                continue
            p0 = f0r["keypoints"].float().cpu().numpy()[idx[:, 0]].astype(np.float64)
            p1 = f1r["keypoints"].float().cpu().numpy()[idx[:, 1]].astype(np.float64)
            E, inl = cv2.findEssentialMat(p0, p1, K_half, method=cv2.USAC_MAGSAC, prob=0.999, threshold=1.0)
            if E is None:
                continue
            ninl, Rf, _, _ = cv2.recoverPose(E, p0, p1, K_half, mask=inl)
            if ninl < min_inl:
                continue
            ef.append((i, i + st, Rf, float(min(ninl, 300))))

    return _fuse(extr_g, nodes, ef, w_da3, log, _t0)


def _fuse(extr_g, nodes, ef, w_da3, log, _t0, extra=None):
    """Robust SO(3) averaging over feature + DA3 edges; replace rotations only."""
    R_init = [extr_g[f][:3, :3].T.copy() for f in nodes]      # R_wc at nodes
    ed = []
    for st in (1, 2):
        for i in range(len(nodes) - st):
            ed.append((i, i + st, R_init[i + st].T @ R_init[i], w_da3))
    if extra:
        # Second-backbone relative rotations (node-position indexed), e.g. a
        # sparse Pi3X pass: Pi3X's rotations are markedly better than DA3's
        # (v065 CV rot 2.20 vs 4.2) while its centres are worse, and RPE_rot
        # only reads rotations, so they enter here as extra prior edges.
        ed.extend(extra)

    if PROF:
        log(f"  rotavg-prof: extract {_T['extract']:.1f}s  match {_T['match']:.1f}s  "
            f"match1 {_T['match1']:.1f}s  "
            f"(batched nodes {_T['n_batched']}, single {_T['n_single']})  "
            f"ransac+recoverPose {_T['ransac']:.1f}s over {_T['n_edge']} edges "
            f"({1000*_T['ransac']/max(_T['n_edge'],1):.1f} ms/edge)")
    _t_solve = time.perf_counter()
    R1 = _solve(R_init, ef + ed)
    kept = [e for e in ef if _geo(R1[e[1]], R1[e[0]] @ e[2].T) < 6.0]
    R2 = _solve(R1, kept + ed)
    if PROF:
        log(f"  rotavg-prof: SO(3) solve (CPU python) {time.perf_counter() - _t_solve:.1f}s")
    moved = float(np.mean([_geo(a, b) for a, b in zip(R_init, R2)]))
    log(f"  rotavg: {len(ef)} feat ({len(kept)} kept) + {len(ed)} da3 edges over "
        f"{len(nodes)} nodes, rotations moved {moved:.2f} deg, "
        f"tail {time.perf_counter() - _t0:.1f}s")

    from scipy.spatial.transform import Rotation, Slerp
    sl = Slerp(np.array(nodes, float), Rotation.from_matrix(np.stack(R2)))
    n = len(extr_g)
    q = np.clip(np.arange(n, dtype=float), nodes[0], nodes[-1])
    R_all = sl(q).as_matrix()
    out = extr_g.copy()
    for f in range(n):
        C = -(extr_g[f][:3, :3].T @ extr_g[f][:3, 3])
        out[f][:3, :3] = R_all[f].T
        out[f][:3, 3] = -R_all[f].T @ C
    return out


class RotWorker(threading.Thread):
    """Extracts features and epipolar rotation edges IN PARALLEL with the DA3
    chain. Why: run serially after the chain, matching costs ~150 ms/node of
    pure wall-clock (per-call GPU overhead that a faster eval GPU does NOT
    shrink -> v006 paid W_t=1.156 on the LB). The edges only need pixels, not
    poses, so they are computed while DA3 owns the GPU; the visible cost
    collapses to the IRLS solve tail (~1-2 s per clip)."""

    def __init__(self, K, kp=None, steps=None, min_inl=15, log=print, photo=None):
        super().__init__(daemon=True)
        self.K = K
        self.kp = kp or int(os.environ.get("DA3_ROT_KP", 512))
        self.steps = steps or tuple(
            int(x) for x in os.environ.get("DA3_ROT_STEPS", "1,4,7").split(","))
        self.min_inl = min_inl
        self.log = log
        self.photo = photo          # (mu, D) of the endoscope's light model
        self._gain = None           # built on the first frame, when hw is known
        self.q = queue.Queue()
        self.nodes, self.edges = [], []
        self.failed = None

    def add(self, idx, img):
        self.q.put((idx, img))

    def close(self):
        self.q.put(None)

    def _edge(self, a, k, p0, p1):
        _t0e = time.perf_counter()
        if SOLVER == "rot2pt":
            f0, f1 = _bearings(p0, self.K), _bearings(p1, self.K)
            Rf, ninl = _rot_only(f0, f1, min_inl=self.min_inl)
            if Rf is None:
                return
        else:
            E, inl = cv2.findEssentialMat(p0, p1, self.K, method=cv2.USAC_MAGSAC,
                                          prob=0.999, threshold=1.0)
            if E is None:
                return
            ninl, Rf, _, _ = cv2.recoverPose(E, p0, p1, self.K, mask=inl)
            if ninl < self.min_inl:
                return
        w = float(min(ninl, 300))
        if GATE:
            # Down-weight (or drop) edges whose flow field is incoherent: those
            # are deformation-dominated, where a rigid relative rotation is not
            # a meaningful quantity.
            sig = _motion_gate(p0, p1)
            if sig > GATE_HARD:
                return
            w *= float(np.clip((GATE_HARD - sig) / max(GATE_HARD - GATE_SOFT, 1e-6), 0.0, 1.0)) \
                if sig > GATE_SOFT else 1.0
            if w < 1.0:
                return
        self.edges.append((a, k, Rf, w))
        if PROF:
            _T["ransac"] += time.perf_counter() - _t0e; _T["n_edge"] += 1

    def run(self):
        try:
            import torch
            from lightglue import ALIKED, LightGlue
            from lightglue.utils import rbd
        except Exception as exc:
            self.failed = f"{type(exc).__name__}: {exc}"
            while self.q.get() is not None:
                pass
            return
        try:
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            fp16 = dev == "cuda" and int(os.environ.get("DA3_ROT_FP16", "0"))
            adaptive = int(os.environ.get("DA3_ROT_ADAPTIVE", "0"))
            despec = int(os.environ.get("DA3_ROT_DESPEC", "0"))
            kw = dict(depth_confidence=0.9, width_confidence=0.95) if adaptive \
                else dict(depth_confidence=-1, width_confidence=-1)
            # ALIKED's default detection_threshold=0.2 gates the top-k, so the
            # returned count varies frame to frame -- which is why the batched
            # LightGlue path (one call per node instead of one per pair) never
            # fired: it requires equal keypoint counts. Dropping the threshold
            # makes every frame return exactly max_num_keypoints, so the batch
            # path always applies. Profiled: 117 of 129 nodes took the per-pair
            # fallback, at 0.90 s/node against 0.44 s/node batched.
            ext = ALIKED(max_num_keypoints=self.kp,
                         detection_threshold=DETECT_THR).eval().to(dev)
            lg = LightGlue(features="aliked", **kw).eval().to(dev)
            from contextlib import nullcontext
            amp = (lambda: torch.autocast("cuda", dtype=torch.float16)) if fp16 \
                else nullcontext
        except Exception as exc:
            self.failed = f"{type(exc).__name__}: {exc}"
            while self.q.get() is not None:
                pass
            return
        feats = {}
        maxst = max(self.steps)
        pool = ThreadPoolExecutor(max_workers=ROT_POOL) if ROT_POOL > 0 else None
        submit = (lambda *a: pool.submit(self._edge, *a)) if pool else self._edge
        while True:
            item = self.q.get()
            if item is None:
                if pool:
                    pool.shutdown(wait=True)      # every edge lands before fuse
                return
            idx, im = item
            try:
                if PHOTO_ON and self.photo is not None:
                    if self._gain is None:
                        self._gain = photo_gain(self.K, im.shape[:2], *self.photo)
                        self.log(f"  rotavg: photometric correction on "
                            f"(mu={self.photo[0]:.2f}, gain "
                            f"{self._gain.min():.2f}-{self._gain.max():.2f})")
                    im = photo_correct(im, self._gain)
                if despec:
                    im = _despec(im)
                t = torch.from_numpy(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)) \
                    .permute(2, 0, 1).float().div(255).to(dev)[None]
                _t = time.perf_counter()
                with torch.no_grad(), amp():
                    fk = ext.extract(t)
                if PROF:
                    torch.cuda.synchronize(); _T["extract"] += time.perf_counter() - _t
                self.nodes.append(idx)
                k = len(self.nodes) - 1
                feats[k] = fk
                pairs = [k - st for st in self.steps
                         if k - st >= 0 and (k - st) in feats]
                # ALIKED returns a VARIABLE number of keypoints per frame, so
                # the equal-shape test below never once passed in a profiled
                # 975-frame clip -- every pair fell through to the one-call-per-
                # pair branch the batching was written to avoid. Keypoints come
                # back score-ordered, so truncating the group to its shortest
                # member costs only the weakest detections, and only when the
                # counts are already close.
                cnt = [feats[a]["keypoints"].shape[1] for a in pairs] + \
                      [feats[k]["keypoints"].shape[1]] if pairs else []
                batched = bool(pairs) and min(cnt) >= TRUNC_RATIO * max(cnt)
                if PROF:
                    _T["n_batched" if batched else "n_single"] += 1
                if batched:
                    kk = min(cnt)
                    cut = (lambda d: {key: (v[:, :kk] if v.ndim >= 2 and
                                            v.shape[1] == d["keypoints"].shape[1]
                                            else v) for key, v in d.items()})
                    # ONE LightGlue call for all steps of this node (B=|steps|):
                    # per-call GPU overhead, not compute, dominated the cost.
                    fc = {a: cut(feats[a]) for a in pairs}
                    kc = cut(fk)
                    d0 = {key: torch.cat([fc[a][key] for a in pairs]) for key in fk}
                    d1 = {key: torch.cat([kc[key]] * len(pairs)) for key in fk}
                    _t = time.perf_counter()
                    with torch.no_grad(), amp():
                        m = lg({"image0": d0, "image1": d1})
                    if PROF:
                        torch.cuda.synchronize(); _T["match"] += time.perf_counter() - _t
                    m0 = m["matches0"].cpu().numpy()          # (B, K)
                    kp1 = kc["keypoints"][0].float().cpu().numpy()
                    for bi, a in enumerate(pairs):
                        sel = np.nonzero(m0[bi] >= 0)[0]
                        if len(sel) < self.min_inl:
                            continue
                        p0 = fc[a]["keypoints"][0].float().cpu().numpy()[sel].astype(np.float64)
                        p1 = kp1[m0[bi][sel]].astype(np.float64)
                        submit(a, k, p0, p1)
                else:
                    for a in pairs:
                        _t = time.perf_counter()
                        with torch.no_grad(), amp():
                            m = lg({"image0": feats[a], "image1": feats[k]})
                        if PROF:
                            torch.cuda.synchronize()
                            _T["match1"] += time.perf_counter() - _t
                        f0r, f1r, mr = [rbd(x) for x in (feats[a], feats[k], m)]
                        mi = mr["matches"].cpu().numpy()
                        if len(mi) < self.min_inl:
                            continue
                        p0 = f0r["keypoints"].float().cpu().numpy()[mi[:, 0]].astype(np.float64)
                        p1 = f1r["keypoints"].float().cpu().numpy()[mi[:, 1]].astype(np.float64)
                        submit(a, k, p0, p1)
                if k - maxst in feats:
                    del feats[k - maxst]
            except Exception:
                continue


def finalize(worker, extr_g, log=print, w_da3=None, extra=None):
    """Join the worker and fuse its edges into extr_g (rotations only)."""
    _t0 = time.perf_counter()
    if w_da3 is None:
        w_da3 = float(os.environ.get("DA3_ROT_WDA3", "10"))
    worker.close()
    worker.join(timeout=900)
    if worker.is_alive():
        log("  rotavg: worker did not finish; keeping chain rotations")
        return extr_g
    if worker.failed:
        log(f"  rotavg: worker unavailable ({worker.failed}); skipping")
        return extr_g
    if len(worker.nodes) < 3 or not worker.edges:
        log("  rotavg: too few nodes/edges; skipping")
        return extr_g
    # The pool appends out of order; _solve's Huber reweighting is order
    # sensitive, so restore a canonical order to keep runs reproducible.
    edges = sorted(worker.edges, key=lambda e: (e[0], e[1]))
    return _fuse(extr_g, worker.nodes, edges, w_da3, log, _t0, extra=extra)
