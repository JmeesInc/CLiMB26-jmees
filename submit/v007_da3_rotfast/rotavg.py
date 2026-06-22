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
        E, inl = cv2.findEssentialMat(p0, p1, self.K, method=cv2.USAC_MAGSAC,
                                      prob=0.999, threshold=1.0)
        if E is None:
            return
        ninl, Rf, _, _ = cv2.recoverPose(E, p0, p1, self.K, mask=inl)
        if ninl < self.min_inl:
            return
        self.edges.append((a, k, Rf, float(min(ninl, 300))))

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


def finalize(worker, extr_g, log=print, w_da3=None):
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
    return _fuse(extr_g, worker.nodes, worker.edges, w_da3, log, _t0)
