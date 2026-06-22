#!/usr/bin/env python3
"""Pack the full GT-vs-prediction error picture for the diagnostic viewer.

Three things, all on the same Sim(3)/Horn alignment the evaluator uses (fitted
on the camera centres of shared frame IDs), so what is drawn is what is scored:

  trajectory : both paths, plus per-frame position error in mm
  rotation   : the evaluator's RPE at d=40, decomposed onto the camera's own
               axes. A wrong focal length trades rotation against translation
               about the axes PERPENDICULAR to the optical axis, so if that is
               the mechanism the error should sit in pitch/yaw and not in roll.
  points     : both clouds, quantised for transport

Note the LoRA training clips cannot be used here: they come from trainval
sequences that have no COLMAP reference, which is exactly why that training is
self-supervised. These four clips are the only real data with ground truth.
"""
import argparse
import base64
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

N_PTS = 12000
FRUSTA = 60


def load_gt(images_txt):
    """COLMAP images.txt (world-to-camera) -> {frame_id: (R_wc, centre)}."""
    out = {}
    for ln in Path(images_txt).read_text().splitlines():
        if ln.startswith("#") or not ln.strip():
            continue
        f = ln.split()
        if len(f) < 10 or not f[9].endswith(".png"):
            continue
        R_cw = Rotation.from_quat([float(f[2]), float(f[3]), float(f[4]), float(f[1])]).as_matrix()
        t = np.array([float(f[5]), float(f[6]), float(f[7])])
        out[int(f[9].replace(".png", ""))] = (R_cw.T, -R_cw.T @ t)
    return out


def load_pred(path):
    """CLiMB trajectory (camera-to-world) -> {frame_id: (R_wc, centre)}."""
    out = {}
    for ln in Path(path).read_text().splitlines():
        if ln.startswith("#") or not ln.strip():
            continue
        v = ln.split(",")
        C = np.array([float(v[2]), float(v[3]), float(v[4])])
        R = Rotation.from_quat([float(v[6]), float(v[7]), float(v[8]), float(v[5])]).as_matrix()
        out[int(v[1].replace(".png", ""))] = (R, C)
    return out


def load_points(p):
    xyz, rgb = [], []
    for ln in Path(p).read_text().splitlines():
        if ln.startswith("#") or not ln.strip():
            continue
        f = ln.split()
        xyz.append((float(f[1]), float(f[2]), float(f[3])))
        rgb.append((int(f[4]), int(f[5]), int(f[6])))
    return np.array(xyz), np.array(rgb, np.uint8)


def horn(src, dst):
    ms, md = src.mean(0), dst.mean(0)
    s0, d0 = src - ms, dst - md
    U, D, Vt = np.linalg.svd(d0.T @ s0)
    S = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / (s0 ** 2).sum()
    return s, R, md - s * R @ ms


def rpe_axis(gt, pr, ids, delta=40):
    """Evaluator RPE(d=40) per anchor frame, split onto the camera's own axes.

    Relative rotation is invariant to the global alignment, so this needs no
    Sim(3): it is the same number the official evaluator reports.
    """
    have = set(ids)
    fid, tot, comp = [], [], []

    def rel(d, a, b):
        Ra, Ca = d[a]
        Rb, Cb = d[b]
        T = np.eye(4)
        T[:3, :3] = Ra.T @ Rb
        T[:3, 3] = Ra.T @ (Cb - Ca)
        return T

    for i in ids:
        j = i + delta
        if j not in have:
            continue
        E = np.linalg.inv(rel(gt, i, j)) @ rel(pr, i, j)
        rv = Rotation.from_matrix(E[:3, :3]).as_rotvec(degrees=True)
        fid.append(i)
        tot.append(float(np.linalg.norm(rv)))
        comp.append(np.abs(rv))
    return np.array(fid), np.array(tot), np.array(comp)


def sub(x, c, n):
    if len(x) <= n:
        return x, c
    i = np.random.default_rng(0).choice(len(x), n, replace=False)
    return x[i], c[i]


def q16(a, lo, span):
    return np.clip(np.round((a - lo) / span * 65534 - 32767), -32767, 32767).astype("<i2")


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_root", required=True)
    ap.add_argument("--pred_root", required=True)
    ap.add_argument("--scales", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    scales = {l.split(",")[0]: float(l.split(",")[1])
              for l in Path(a.scales).read_text().splitlines()[1:] if l.strip()}
    pay = {}
    for d in sorted(p for p in Path(a.gt_root).iterdir() if p.is_dir()):
        seq = d.name
        tf = Path(a.pred_root) / seq / "1" / "camera_trajectory" / "cam_traj_map_000.txt"
        pf = Path(a.pred_root) / seq / "1" / "3D_maps" / "000" / "points3D.txt"
        if not tf.exists():
            continue
        sc = scales[seq]
        gt, pr = load_gt(d / "results_txt" / "images.txt"), load_pred(tf)
        ids = sorted(set(gt) & set(pr))
        gC = np.array([gt[i][1] for i in ids])
        pC = np.array([pr[i][1] for i in ids])
        S, R, T = horn(pC, gC)

        def xf(X):
            return (S * (R @ X.T).T + T) * sc

        gP, gRGB = load_points(d / "results_txt" / "points3D.txt")
        pP, pRGB = load_points(pf)
        gP, gRGB = sub(gP, gRGB, N_PTS)
        pP, pRGB = sub(pP, pRGB, N_PTS)
        gP = gP * sc
        pP = xf(pP)
        gT, pT = gC * sc, xf(pC)
        perr = np.linalg.norm(pT - gT, axis=1)

        fid, rtot, rcomp = rpe_axis(gt, pr, ids)

        ctr = gT.mean(0)
        m = np.linalg.norm(gP - ctr, axis=1) < np.percentile(np.linalg.norm(gP - ctr, axis=1), 98)
        gP, gRGB = gP[m], gRGB[m]
        m = np.linalg.norm(pP - ctr, axis=1) < np.percentile(np.linalg.norm(pP - ctr, axis=1), 98)
        pP, pRGB = pP[m], pRGB[m]

        allp = np.vstack([gP, pP, gT, pT])
        lo = allp.min(0)
        span = np.maximum(allp.max(0) - allp.min(0), 1e-6)

        # Sparse camera frusta so the two trajectories can be compared as
        # oriented objects, not just as paths.
        step = max(1, len(ids) // FRUSTA)
        k = list(range(0, len(ids), step))
        gR = np.stack([gt[ids[i]][0] for i in k])
        pR = np.stack([R @ pr[ids[i]][0] for i in k])

        pay[seq] = dict(
            lo=lo.tolist(), span=span.tolist(),
            ate=float(perr.mean()), rot=float(rtot.mean()),
            axes=[float(rcomp[:, i].mean()) for i in range(3)],
            n_gt=int(len(gP)), n_pred=int(len(pP)), matched=len(ids),
            frames=int(max(gt)),
            ids=[int(i) for i in ids], fid=[int(i) for i in fid],
            gt=b64(q16(gP, lo, span)), gtc=b64(gRGB),
            pr=b64(q16(pP, lo, span)), prc=b64(pRGB),
            gtr=b64(q16(gT, lo, span)), prt=b64(q16(pT, lo, span)),
            err=b64(perr.astype("<f4")),
            rerr=b64(rtot.astype("<f4")),
            rxyz=b64(rcomp.astype("<f4")),
            fk=[int(ids[i]) for i in k],
            gR=b64(gR.astype("<f4")), pR=b64(pR.astype("<f4")),
            gTk=b64(q16(gT[k], lo, span)), pTk=b64(q16(pT[k], lo, span)),
        )
        print(f"{seq}: ATE {perr.mean():5.2f} mm  rot {rtot.mean():5.2f} deg  "
              f"axes |x|{rcomp[:, 0].mean():5.2f} |y|{rcomp[:, 1].mean():5.2f} "
              f"|z|{rcomp[:, 2].mean():5.2f}  {len(ids)} cams")
    Path(a.out).write_text(json.dumps(pay))
    print(f"-> {a.out} ({Path(a.out).stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
