#!/usr/bin/env python3
"""Why is the rotational RPE ~23 deg when ATE is ~3 mm?

pred_eda.py established that this is NOT a transposed-quaternion bug (transposing
makes it worse) and that RPE_rot grows like a random walk: 3.45 deg at delta=1 vs
23.3 at delta=40, and sqrt(40) x 3.45 = 21.8.

This script asks the two questions that decide whether the 3.45 deg/frame is a
real modelling failure or an artifact:

  A. How much does the camera ACTUALLY rotate per frame in GT? A 3.45 deg error is
     catastrophic if GT moves 0.3 deg/frame and unremarkable if it moves 10.
  B. Is there a constant camera-frame basis mismatch B (R_gt ~ A R_pred B)? That is
     a right-multiplication, invisible to the world-side fit in pred_eda.py, and it
     WOULD inflate RPE_rot (A cancels in relative poses, B does not).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pred_eda import read_gt, read_pred, REPO  # noqa: E402


def ang(R):
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def proj_so3(M):
    U, _, Vt = np.linalg.svd(M)
    return U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt


def fit_AB(Rg, Rp, iters=50):
    """Alternating fit of R_gt ~ A R_pred B over stacked rotations."""
    B = np.eye(3)
    A = np.eye(3)
    for _ in range(iters):
        A = proj_so3(sum(g @ (p @ B).T for g, p in zip(Rg, Rp)))
        B = proj_so3(sum((A @ p).T @ g for g, p in zip(Rg, Rp)))
    res = [ang(g.T @ (A @ p @ B)) for g, p in zip(Rg, Rp)]
    return A, B, np.array(res)


def main():
    gt_root = REPO / "workspace/expA00_baseline_eval/colmap_gt"
    pred_root = REPO / "submit/v001_da3_submap/output"

    print(f"{'seq':7s} {'GT rot/frame':>13s} {'pred rot/frame':>15s} {'rel-rot err':>12s} "
          f"{'absrot A-only':>14s} {'absrot A+B':>11s} {'B angle':>9s}")
    for d in sorted(pred_root.iterdir()):
        if not d.is_dir():
            continue
        gt = read_gt(gt_root / d.name / "results_txt" / "images.txt")
        pred = read_pred(d / "1" / "camera_trajectory" / "cam_traj_map_000.txt")
        common = sorted(set(gt) & set(pred))
        if len(common) < 10:
            continue

        # A: how much does the camera really turn between consecutive frames?
        gsteps, psteps, errs = [], [], []
        for i, j in zip(common[:-1], common[1:]):
            if j != i + 1:
                continue
            Rg_rel = gt[i][0].T @ gt[j][0]
            Rp_rel = pred[i][0].T @ pred[j][0]
            gsteps.append(ang(Rg_rel))
            psteps.append(ang(Rp_rel))
            errs.append(ang(Rg_rel.T @ Rp_rel))

        # B: constant camera-frame basis mismatch?
        Rg = [gt[f][0] for f in common]
        Rp = [pred[f][0] for f in common]
        M = sum(g @ p.T for g, p in zip(Rg, Rp))
        A_only = proj_so3(M)
        res_A = np.array([ang(g.T @ (A_only @ p)) for g, p in zip(Rg, Rp)])
        _, B, res_AB = fit_AB(Rg, Rp)

        print(f"{d.name:7s} {np.mean(gsteps):12.3f}° {np.mean(psteps):14.3f}° "
              f"{np.mean(errs):11.3f}° {res_A.mean():13.2f}° {res_AB.mean():10.2f}° "
              f"{ang(B):8.2f}°")

    print("\nReading:")
    print("  * 'GT rot/frame' vs 'pred rot/frame': if pred >> GT we are inventing rotation.")
    print("  * 'absrot A+B' << 'absrot A-only' would mean a fixed camera-basis mismatch")
    print("    that we could simply divide out; similar values mean the error is genuine.")


if __name__ == "__main__":
    main()
