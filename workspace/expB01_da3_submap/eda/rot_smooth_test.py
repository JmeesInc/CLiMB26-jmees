#!/usr/bin/env python3
"""Can temporal smoothing of the ROTATIONS alone cut the rotational RPE?

rot_diag.py showed the per-frame rotation error (3.45 deg) is LARGER than the true
per-frame rotation (2.04 deg): DA3's orientation output is noise-dominated at this
frame rate, and that noise random-walks to ~23 deg at delta=40.

Because the true motion is smooth, averaging rotations over a short window should
remove noise without removing signal. Crucially the CLiMB metrics are separable:
ATE is computed from camera CENTRES only, so smoothing rotations cannot hurt ATE.
It only moves RPE_rot, which is the new climb_score tie-break
(Score_Rot = RotErr(delta=40) x W_rtf x W_t).

This measures RPE_rot vs smoothing window, so the win (if any) is quantified before
touching the submission.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pred_eda import read_gt, read_pred, rpe, REPO  # noqa: E402


def R_to_q(R):
    """Rotation matrix -> quaternion (w,x,y,z)."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s,
                      (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                      (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                      0.25 * s, (R[1, 2] + R[2, 1]) / s])
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                      (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    return q / np.linalg.norm(q)


def q_to_R(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def smooth_rotations(pred, win):
    """Chordal-L2 mean of quaternions in a centred window of `win` frames."""
    if win <= 1:
        return pred
    ids = sorted(pred)
    qs = np.array([R_to_q(pred[f][0]) for f in ids])
    # resolve sign ambiguity against the running reference so averaging is valid
    for i in range(1, len(qs)):
        if qs[i] @ qs[i - 1] < 0:
            qs[i] = -qs[i]
    half = win // 2
    out = {}
    for k, f in enumerate(ids):
        lo, hi = max(0, k - half), min(len(ids), k + half + 1)
        M = qs[lo:hi].T @ qs[lo:hi]
        w, V = np.linalg.eigh(M)              # dominant eigenvector = chordal mean
        out[f] = (q_to_R(V[:, -1]), pred[f][1])
    return out


def main():
    gt_root = REPO / "workspace/expA00_baseline_eval/colmap_gt"
    pred_root = REPO / "submit/v001_da3_submap/output"
    wins = [1, 3, 5, 9, 15, 25]

    print("Rotational RPE at delta=40 (deg), by smoothing window")
    print(f"{'seq':7s}" + "".join(f"{('w=%d' % w):>9s}" for w in wins))
    tot = {w: [] for w in wins}
    for d in sorted(pred_root.iterdir()):
        if not d.is_dir():
            continue
        gt = read_gt(gt_root / d.name / "results_txt" / "images.txt")
        pred = read_pred(d / "1" / "camera_trajectory" / "cam_traj_map_000.txt")
        if len(set(gt) & set(pred)) < 50:
            continue
        row = f"{d.name:7s}"
        for w in wins:
            _, ro = rpe(gt, smooth_rotations(pred, w), 40)
            tot[w].append(ro.mean())
            row += f"{ro.mean():9.2f}"
        print(row, flush=True)
    print(f"{'MEAN':7s}" + "".join(f"{np.mean(tot[w]):9.2f}" for w in wins))

    base, best_w = np.mean(tot[1]), min(wins, key=lambda w: np.mean(tot[w]))
    print(f"\nbest window = {best_w}: {np.mean(tot[best_w]):.2f} deg "
          f"vs {base:.2f} baseline ({100*(1-np.mean(tot[best_w])/base):.1f}% better)")
    print("ATE is unaffected by construction: it is computed from camera centres only.")


if __name__ == "__main__":
    main()
