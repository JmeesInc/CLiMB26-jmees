#!/usr/bin/env python3
"""Pack GT-vs-prediction geometry into a compact payload for a 3D viewer.

Alignment is the evaluator's: Sim(3)/Horn fitted on the camera centres of the
frame IDs the two trajectories share, then applied to the predicted points as
well, so what you see is exactly the geometry the ATE is computed on. Everything
is expressed in mm via scales.csv.

Positions are quantised to int16 over a per-sequence bounding box (sub-0.1 mm
resolution at these scales) and colours to uint8, so four sequences of both
clouds fit in ~1 MB of base64 instead of ~40 MB of text.
"""
import argparse, base64, json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

N_PTS = 12000


def load_gt_images(p):
    """COLMAP images.txt (w2c) -> {frame_id: camera centre}."""
    out = {}
    for ln in Path(p).read_text().splitlines():
        if ln.startswith("#") or not ln.strip():
            continue
        f = ln.split()
        if len(f) < 10 or not f[9].endswith(".png"):
            continue
        R = Rotation.from_quat([float(f[2]), float(f[3]), float(f[4]), float(f[1])]).as_matrix()
        t = np.array([float(f[5]), float(f[6]), float(f[7])])
        out[int(f[9].replace(".png", ""))] = -R.T @ t
    return out


def load_colmap_points(p):
    xyz, rgb = [], []
    for ln in Path(p).read_text().splitlines():
        if ln.startswith("#") or not ln.strip():
            continue
        f = ln.split()
        xyz.append((float(f[1]), float(f[2]), float(f[3])))
        rgb.append((int(f[4]), int(f[5]), int(f[6])))
    return np.array(xyz), np.array(rgb, np.uint8)


def load_pred(traj, pts):
    C, ids = {}, []
    for ln in Path(traj).read_text().splitlines():
        if ln.startswith("#") or not ln.strip():
            continue
        v = ln.split(",")
        C[int(v[1].replace(".png", ""))] = np.array([float(v[2]), float(v[3]), float(v[4])])
    xyz, rgb = [], []
    for ln in Path(pts).read_text().splitlines():
        if ln.startswith("#") or not ln.strip():
            continue
        f = ln.split()
        xyz.append((float(f[1]), float(f[2]), float(f[3])))
        rgb.append((int(f[4]), int(f[5]), int(f[6])))
    return C, np.array(xyz), np.array(rgb, np.uint8)


def horn(src, dst):
    """Similarity src->dst; returns (s, R, t)."""
    ms, md = src.mean(0), dst.mean(0)
    s0, d0 = src - ms, dst - md
    U, D, Vt = np.linalg.svd(d0.T @ s0)
    S = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / (s0 ** 2).sum()
    return s, R, md - s * R @ ms


def sub(xyz, rgb, n):
    if len(xyz) <= n:
        return xyz, rgb
    i = np.random.default_rng(0).choice(len(xyz), n, replace=False)
    return xyz[i], rgb[i]


def quant(a, lo, span):
    return np.clip(np.round((a - lo) / span * 65534 - 32767), -32767, 32767).astype("<i2")


def b64(a):
    return base64.b64encode(a.tobytes()).decode("ascii")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_root", required=True)
    ap.add_argument("--pred_root", required=True)
    ap.add_argument("--scales", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    scales = {l.split(",")[0]: float(l.split(",")[1])
              for l in Path(a.scales).read_text().splitlines()[1:] if l.strip()}
    payload = {}
    for d in sorted(p for p in Path(a.gt_root).iterdir() if p.is_dir()):
        seq = d.name
        tf = Path(a.pred_root) / seq / "1" / "camera_trajectory" / "cam_traj_map_000.txt"
        pf = Path(a.pred_root) / seq / "1" / "3D_maps" / "000" / "points3D.txt"
        if not tf.exists():
            continue
        sc = scales[seq]
        gtC = load_gt_images(d / "results_txt" / "images.txt")
        gP, gRGB = load_colmap_points(d / "results_txt" / "points3D.txt")
        pC, pP, pRGB = load_pred(tf, pf)

        ids = sorted(set(gtC) & set(pC))
        S, R, T = horn(np.array([pC[i] for i in ids]), np.array([gtC[i] for i in ids]))
        xf = lambda X: (S * (R @ X.T).T + T) * sc

        gP, gRGB = sub(gP, gRGB, N_PTS)
        pP, pRGB = sub(pP, pRGB, N_PTS)
        gP = gP * sc
        pP = xf(pP)
        gTraj = np.array([gtC[i] for i in ids]) * sc
        pTraj = xf(np.array([pC[i] for i in ids]))
        err = np.linalg.norm(pTraj - gTraj, axis=1)

        # Trim the long tail of stray points so the view frames the anatomy
        # rather than a handful of outliers 100x further out.
        ctr = gTraj.mean(0)
        keep = lambda X: X[np.linalg.norm(X - ctr, axis=1) <
                           np.percentile(np.linalg.norm(gP - ctr, axis=1), 98) * 1.5]
        m = np.linalg.norm(gP - ctr, axis=1) < np.percentile(np.linalg.norm(gP - ctr, axis=1), 98)
        gP, gRGB = gP[m], gRGB[m]
        m = np.linalg.norm(pP - ctr, axis=1) < np.percentile(np.linalg.norm(pP - ctr, axis=1), 98)
        pP, pRGB = pP[m], pRGB[m]

        allpts = np.vstack([gP, pP, gTraj, pTraj])
        lo, hi = allpts.min(0), allpts.max(0)
        span = np.maximum(hi - lo, 1e-6)

        payload[seq] = dict(
            lo=lo.tolist(), span=span.tolist(),
            ate=float(err.mean()), n_gt=int(len(gP)), n_pred=int(len(pP)),
            frames=int(max(gtC)), matched=len(ids),
            gt=b64(quant(gP, lo, span)), gtc=b64(gRGB.astype(np.uint8)),
            pr=b64(quant(pP, lo, span)), prc=b64(pRGB.astype(np.uint8)),
            gtr=b64(quant(gTraj, lo, span)), prt=b64(quant(pTraj, lo, span)),
            err=b64(np.clip(err, 0, 60).astype("<f4")),
        )
        print(f"{seq}: GT {len(gP)} pts, pred {len(pP)} pts, {len(ids)} matched cams, "
              f"ATE {err.mean():.2f} mm, extent {span.max():.0f} mm")

    Path(a.out).write_text(json.dumps(payload))
    print(f"-> {a.out}  ({Path(a.out).stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
