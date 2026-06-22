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
import numpy as np
import cv2


def _proj_so3(M):
    U, _, Vt = np.linalg.svd(M)
    return U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt


def _geo(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def _solve(R_init, edges, iters=30, huber_deg=2.0):
    R = [r.copy() for r in R_init]
    for _ in range(iters):
        delta = 0.0
        for i in range(len(R)):
            preds, ws = [], []
            for (a, b, Rab, w) in edges:
                if a == i:
                    preds.append(R[b] @ Rab); ws.append(w)
                elif b == i:
                    preds.append(R[a] @ Rab.T); ws.append(w)
            if not preds:
                continue
            res = np.array([_geo(R[i], Pm) for Pm in preds])
            hw = np.where(res <= huber_deg, 1.0, huber_deg / np.maximum(res, 1e-9))
            W = np.array(ws) * hw
            Rn = _proj_so3(sum(w * Pm for w, Pm in zip(W, preds)))
            delta = max(delta, _geo(R[i], Rn))
            R[i] = Rn
        if delta < 0.01:
            break
    return R


def refine(extr_g, rot_cache, K_half, log=print,
           kp=None, steps=None, min_inl=15, w_da3=None):
    """extr_g: (n,4,4) w2c full trajectory. rot_cache: {frame_idx: encoded jpg}.
    K_half: pinhole K of the cached (rectified+cropped, half-res) frames.
    Returns extr_g with rotations replaced (centres untouched)."""
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
    ext = ALIKED(max_num_keypoints=kp).eval().to(dev)
    lg = LightGlue(features="aliked").eval().to(dev)

    feats = {}
    def feat(k):
        if k not in feats:
            im = cv2.imdecode(np.frombuffer(rot_cache[nodes[k]], np.uint8), cv2.IMREAD_COLOR)
            t = torch.from_numpy(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().div(255).to(dev)[None]
            with torch.no_grad():
                feats[k] = ext.extract(t)
        return feats[k]

    ef = []
    for st in steps:
        for i in range(len(nodes) - st):
            f0, f1 = feat(i), feat(i + st)
            with torch.no_grad():
                m = lg({"image0": f0, "image1": f1})
            f0r, f1r, mr = [rbd(x) for x in (f0, f1, m)]
            idx = mr["matches"].cpu().numpy()
            if len(idx) < min_inl:
                continue
            p0 = f0r["keypoints"].cpu().numpy()[idx[:, 0]]
            p1 = f1r["keypoints"].cpu().numpy()[idx[:, 1]]
            E, inl = cv2.findEssentialMat(p0, p1, K_half, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            if E is None:
                continue
            ninl, Rf, _, _ = cv2.recoverPose(E, p0, p1, K_half, mask=inl)
            if ninl < min_inl:
                continue
            ef.append((i, i + st, Rf, float(min(ninl, 300))))

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
        f"{len(nodes)} nodes, rotations moved {moved:.2f} deg")

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
