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
import time
import numpy as np
import cv2


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
    # Same Gauss-Seidel/Huber iteration as before, but with a precomputed
    # adjacency list: the old per-node scan over ALL edges was O(n*m*iters)
    # pure Python (10-26 s/clip of wall-clock that the eval GPU cannot
    # shrink); per-node incident edges make it <1 s with identical results.
    R = [r.copy() for r in R_init]
    adj = [[] for _ in R]                     # i -> [(j, Rab_oriented, w)]
    for (a, b, Rab, w) in edges:
        adj[a].append((b, Rab, w))            # pred_a = R[b] @ Rab
        adj[b].append((a, Rab.T.copy(), w))   # pred_b = R[a] @ Rab.T
    for _ in range(iters):
        delta = 0.0
        for i in range(len(R)):
            if not adj[i]:
                continue
            preds = [R[j] @ Rab for (j, Rab, _) in adj[i]]
            ws = [w for (_, _, w) in adj[i]]
            res = np.array([_geo(R[i], Pm) for Pm in preds])
            hw = np.where(res <= huber_deg, 1.0, huber_deg / np.maximum(res, 1e-9))
            W = np.array(ws) * hw
            Rn = _proj_so3(sum(w * Pm for w, Pm in zip(W, preds)))
            delta = max(delta, _geo(R[i], Rn))
            R[i] = Rn
        if delta < 0.01:
            break
    return R


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
    ext = ALIKED(max_num_keypoints=kp).eval().to(dev)
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


def _fuse(extr_g, nodes, ef, w_da3, log, _t0):
    """Robust SO(3) averaging over feature + DA3 edges; replace rotations only."""
    R_init = [extr_g[f][:3, :3].T.copy() for f in nodes]      # R_wc at nodes
    ed = []
    for st in (1, 2):
        for i in range(len(nodes) - st):
            ed.append((i, i + st, R_init[i + st].T @ R_init[i], w_da3))

    R1 = _solve(R_init, ef + ed)
    kept = [e for e in ef if _geo(R1[e[1]], R1[e[0]] @ e[2].T) < 6.0]
    R2 = _solve(R1, kept + ed)
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


def _ba_refine(extr_g, nodes, corr, depth_cache, K, log, _t0,
               max_pairs=int(os.environ.get("DA3_ROT_BA_PAIRS", 60)),
               huber_px=float(os.environ.get("DA3_ROT_BA_HUBER", 2.0)),
               max_nfev=int(os.environ.get("DA3_ROT_BA_NFEV", 25))):
    """Depth-anchored, pose-only bundle adjustment over the keyframe graph.

    What the feed-forward heads and the 2-view rotation averaging both lack is
    a reprojection-consistent refinement: classical SLAM gets its rotation
    quality from exactly that (ORB-SLAM3 reached 1.24 deg on our real clips
    where it tracked). Full BA is off the table for runtime, and triangulating
    points is ill-conditioned under the forward motion that dominates
    colonoscopy -- the very failure mode we measured. So anchor the 3D points on
    the model's own depth instead (Track2Map's lift-with-depth), and optimise
    ONLY the keyframe poses to minimise symmetric reprojection of the LightGlue
    inliers. Each residual touches two poses, so scipy's sparse finite
    differences need only a few dozen evaluations per Jacobian. Rotations are
    then substituted back with the chain's camera centres kept, which leaves
    the ATE untouched (measured on v006/v007).
    """
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix
    from scipy.spatial.transform import Rotation

    idx = {f: i for i, f in enumerate(nodes)}
    Kinv = np.linalg.inv(K)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    # Lift each inlier keypoint to a 3D point in its OWN camera frame using the
    # cached depth of that keyframe (depth is z-depth of the ray through p).
    def lift(f, p):
        d, intr = depth_cache[f]
        H, W = d.shape
        xy1 = (Kinv @ np.c_[p, np.ones(len(p))].T).T          # rays (x, y, 1)
        u = intr[0, 0] * xy1[:, 0] + intr[0, 2]
        v = intr[1, 1] * xy1[:, 1] + intr[1, 2]
        ui = np.clip(np.round(u).astype(int), 0, W - 1)
        vi = np.clip(np.round(v).astype(int), 0, H - 1)
        z = d[vi, ui]
        ok = np.isfinite(z) & (z > 1e-4) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        return xy1 * z[:, None], ok

    I, J, XA, PB, XB, PA = [], [], [], [], [], []
    rng = np.random.default_rng(0)
    for (a, b), (p0, p1) in corr.items():
        fa, fb = nodes[a], nodes[b]
        if fa not in depth_cache or fb not in depth_cache or len(p0) < 8:
            continue
        if len(p0) > max_pairs:
            sel = rng.choice(len(p0), max_pairs, replace=False); p0, p1 = p0[sel], p1[sel]
        Xa, oka = lift(fa, p0.astype(np.float64))
        Xb, okb = lift(fb, p1.astype(np.float64))
        ok = oka & okb
        if ok.sum() < 8:
            continue
        I.append(np.full(ok.sum(), a)); J.append(np.full(ok.sum(), b))
        XA.append(Xa[ok]); PB.append(p1[ok].astype(np.float64))
        XB.append(Xb[ok]); PA.append(p0[ok].astype(np.float64))
    if not I:
        log("  rotavg-BA: no usable correspondences; skipping")
        return None
    I = np.concatenate(I); J = np.concatenate(J)
    XA = np.concatenate(XA); PB = np.concatenate(PB); XB = np.concatenate(XB); PA = np.concatenate(PA)
    n_pairs = len(I)

    # Parameters: per node a ROTATION (rotvec, w2c) and a log depth-SCALE.
    # Translations stay frozen at the chain's values. Freeing them let the
    # optimiser trade rotation for translation to fit per-window depth-scale
    # errors (reprojection fell 13.3 -> 2.8 px while RPE_rot went 4.6 -> 10.1
    # deg); the scale parameter gives those errors a legitimate outlet, and the
    # rotations are the only thing we substitute back anyway.
    E0 = np.stack([extr_g[f] for f in nodes])
    rv0 = Rotation.from_matrix(E0[:, :3, :3]).as_rotvec()
    t0 = E0[:, :3, 3]
    N = len(nodes)
    x0 = np.c_[rv0, np.zeros(N)].ravel()               # (rotvec[3], log_s) per node

    def unpack(x):
        x = x.reshape(N, 4)
        return Rotation.from_rotvec(x[:, :3]).as_matrix(), np.exp(x[:, 3])

    def proj(X):
        z = np.maximum(X[:, 2], 1e-6)
        return np.c_[fx * X[:, 0] / z + cx, fy * X[:, 1] / z + cy]

    def resid_full(x, Ii, Jj, XAm, PBm, XBm, PAm):
        R, sc = unpack(x)
        Xa = XAm * sc[Ii, None]; Xb = XBm * sc[Jj, None]
        Xw = np.einsum('nij,nj->ni', np.transpose(R[Ii], (0, 2, 1)), Xa - t0[Ii])
        r1 = proj(np.einsum('nij,nj->ni', R[Jj], Xw) + t0[Jj]) - PBm
        Xw2 = np.einsum('nij,nj->ni', np.transpose(R[Jj], (0, 2, 1)), Xb - t0[Jj])
        r2 = proj(np.einsum('nij,nj->ni', R[Ii], Xw2) + t0[Ii]) - PAm
        return np.concatenate([r1.ravel(), r2.ravel()])

    def resid(x):
        return resid_full(x, I, J, XA, PB, XB, PA)

    r0 = resid(x0)
    gate_px = float(os.environ.get("DA3_ROT_BA_GATE", 60.0))

    def gate(r, thr):
        rr = np.linalg.norm(r.reshape(2, n_pairs, 2), axis=2)
        return (rr[0] < thr) & (rr[1] < thr)

    keep = gate(r0, gate_px)
    if keep.sum() < 50:
        log(f"  rotavg-BA: only {keep.sum()} pairs pass the {gate_px:.0f}px gate; skipping")
        return None

    def solve(x_init, mask):
        Ii, Jj = I[mask], J[mask]
        args = (Ii, Jj, XA[mask], PB[mask], XB[mask], PA[mask])
        n = int(mask.sum())
        Sm = lil_matrix((4 * n, 4 * N), dtype=np.uint8)
        rows = np.arange(n)
        for blk in (0, 1):
            for c in (0, 1):
                r = 2 * n * blk + 2 * rows + c
                for node in (Ii, Jj):
                    for d in range(4):
                        Sm[r, 4 * node + d] = 1
        dense = int(os.environ.get("DA3_ROT_BA_DENSE", "0"))
        r = least_squares(lambda x: resid_full(x, *args), x_init, jac='2-point',
                          jac_sparsity=None if dense else Sm, method='trf',
                          loss='huber', f_scale=huber_px, max_nfev=max_nfev,
                          x_scale=1.0, verbose=0)
        if int(os.environ.get("DA3_ROT_BA_DEBUG", "0")):
            log(f"  BA-dbg: status {r.status} '{r.message}' nfev {r.nfev} njev {r.njev} "
                f"cost {r.cost:.1f} optimality {r.optimality:.2e} params {len(x_init)} resid {4*n}")
        return r

    res = solve(x0, keep)
    keep2 = gate(resid(res.x), max(3 * huber_px, gate_px / 4))
    if keep2.sum() >= 50:
        res = solve(res.x, keep2)
        keep = keep2
    R, sc = unpack(res.x)
    rk = r0.reshape(2, n_pairs, 2)[:, keep].ravel()
    rms0 = float(np.sqrt(np.mean(rk ** 2))); rms1 = float(np.sqrt(np.mean(res.fun ** 2)))
    log(f"  rotavg-BA: gate kept {keep.sum()}/{n_pairs} pairs")
    moved = float(np.mean([_geo(a.T, b.T) for a, b in zip(E0[:, :3, :3], R)]))
    log(f"  rotavg-BA: {n_pairs} pairs over {N} nodes, reproj rms {rms0:.2f}->{rms1:.2f} px, "
        f"nfev {res.nfev}, rotations moved {moved:.2f} deg, depth-scale spread "
        f"{np.exp(np.std(np.log(sc))):.3f}x, tail {time.perf_counter() - _t0:.1f}s")
    return [r.T for r in R]          # R_wc per node


class RotWorker(threading.Thread):
    """Extracts features and epipolar rotation edges IN PARALLEL with the DA3
    chain. Why: run serially after the chain, matching costs ~150 ms/node of
    pure wall-clock (per-call GPU overhead that a faster eval GPU does NOT
    shrink -> v006 paid W_t=1.156 on the LB). The edges only need pixels, not
    poses, so they are computed while DA3 owns the GPU; the visible cost
    collapses to the IRLS solve tail (~1-2 s per clip)."""

    def __init__(self, K, kp=None, steps=None, min_inl=15, log=print):
        super().__init__(daemon=True)
        self.K = K
        self.corr = {}                       # (node_a, node_b) -> inlier pixel pairs
        self.kp = kp or int(os.environ.get("DA3_ROT_KP", 512))
        self.steps = steps or tuple(
            int(x) for x in os.environ.get("DA3_ROT_STEPS", "1,4,7").split(","))
        self.min_inl = min_inl
        self.log = log
        self.q = queue.Queue()
        self.nodes, self.edges = [], []
        self.failed = None

    def add(self, idx, img):
        self.q.put((idx, img))

    def close(self):
        self.q.put(None)

    def _edge(self, a, k, p0, p1):
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
            ninl, Rf, _, pose_mask = cv2.recoverPose(E, p0, p1, self.K, mask=inl)
            if ninl < self.min_inl:
                return
            # Keep the cheirality inliers: the bundle adjustment in finalize()
            # needs actual pixel correspondences, not just the 2-view rotation
            # they produced. The averaging path ignores this field.
            m = pose_mask.ravel() > 0
            self.corr[(a, k)] = (p0[m].astype(np.float32), p1[m].astype(np.float32))
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
            ext = ALIKED(max_num_keypoints=self.kp).eval().to(dev)
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
        while True:
            item = self.q.get()
            if item is None:
                return
            idx, im = item
            try:
                if despec:
                    im = _despec(im)
                t = torch.from_numpy(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)) \
                    .permute(2, 0, 1).float().div(255).to(dev)[None]
                with torch.no_grad(), amp():
                    fk = ext.extract(t)
                self.nodes.append(idx)
                k = len(self.nodes) - 1
                feats[k] = fk
                pairs = [k - st for st in self.steps
                         if k - st >= 0 and (k - st) in feats]
                kk = feats[k]["keypoints"].shape[1]
                batched = pairs and all(
                    feats[a]["keypoints"].shape[1] == kk for a in pairs)
                if batched:
                    # ONE LightGlue call for all steps of this node (B=|steps|):
                    # per-call GPU overhead, not compute, dominated the cost.
                    d0 = {key: torch.cat([feats[a][key] for a in pairs])
                          for key in fk}
                    d1 = {key: torch.cat([fk[key]] * len(pairs)) for key in fk}
                    with torch.no_grad(), amp():
                        m = lg({"image0": d0, "image1": d1})
                    m0 = m["matches0"].cpu().numpy()          # (B, K)
                    kp1 = fk["keypoints"][0].float().cpu().numpy()
                    for bi, a in enumerate(pairs):
                        sel = np.nonzero(m0[bi] >= 0)[0]
                        if len(sel) < self.min_inl:
                            continue
                        p0 = feats[a]["keypoints"][0].float().cpu().numpy()[sel].astype(np.float64)
                        p1 = kp1[m0[bi][sel]].astype(np.float64)
                        self._edge(a, k, p0, p1)
                else:
                    for a in pairs:
                        with torch.no_grad(), amp():
                            m = lg({"image0": feats[a], "image1": feats[k]})
                        f0r, f1r, mr = [rbd(x) for x in (feats[a], feats[k], m)]
                        mi = mr["matches"].cpu().numpy()
                        if len(mi) < self.min_inl:
                            continue
                        p0 = f0r["keypoints"].float().cpu().numpy()[mi[:, 0]].astype(np.float64)
                        p1 = f1r["keypoints"].float().cpu().numpy()[mi[:, 1]].astype(np.float64)
                        self._edge(a, k, p0, p1)
                if k - maxst in feats:
                    del feats[k - maxst]
            except Exception:
                continue


def finalize(worker, extr_g, log=print, w_da3=None, depth_cache=None):
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
    out = _fuse(extr_g, worker.nodes, worker.edges, w_da3, log, _t0)
    if int(os.environ.get("DA3_ROT_BA", "0")) and depth_cache:
        try:
            R_nodes = _ba_refine(out, worker.nodes, worker.corr, depth_cache, worker.K, log, _t0)
        except Exception as exc:
            log(f"  rotavg-BA failed ({type(exc).__name__}: {exc}); keeping averaged rotations")
            R_nodes = None
        if R_nodes is not None:
            out = _substitute_rotations(out, worker.nodes, R_nodes)
    return out


def _substitute_rotations(extr_g, nodes, R_wc_nodes):
    """SLERP node rotations onto every frame; camera centres are kept."""
    from scipy.spatial.transform import Rotation, Slerp
    sl = Slerp(np.array(nodes, float), Rotation.from_matrix(np.stack(R_wc_nodes)))
    n = len(extr_g)
    q = np.clip(np.arange(n, dtype=float), nodes[0], nodes[-1])
    R_all = sl(q).as_matrix()
    out = extr_g.copy()
    for f in range(n):
        C = -(extr_g[f][:3, :3].T @ extr_g[f][:3, 3])
        out[f][:3, :3] = R_all[f].T
        out[f][:3, 3] = -R_all[f].T @ C
    return out
