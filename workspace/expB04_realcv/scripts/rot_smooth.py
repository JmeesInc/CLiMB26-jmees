#!/usr/bin/env python3
"""Does temporal smoothing of the rotations lower RPE(d=40) without hurting ATE?

RPE at d=40 is a RELATIVE rotation over ~0.8 s. Our poses come from keyframes
6 frames apart, SLERP'd in between, so any per-keyframe jitter in the estimated
orientation shows up directly in that relative measure. If the error is jitter
rather than bias, a low-pass on the rotation sequence should cut RPE while
leaving the camera CENTRES -- and therefore the ATE -- untouched. If RPE barely
moves, the error is low-frequency and smoothing is a dead end.

This is a pure post-process on an existing submission tree: zero inference cost,
which is the binding constraint (T <= 0.0198 s/frame, 0.0028 spare).
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rot_budget import load_gt, load_pred, rpe_rot  # noqa: E402


def smooth_rot(ids, R, win):
    """Geodesic moving average over a centred window of `win` frames."""
    if win <= 1:
        return R
    q = Rotation.from_matrix(R).as_quat()
    # keep the quaternion track continuous before averaging
    for i in range(1, len(q)):
        if q[i] @ q[i - 1] < 0:
            q[i] = -q[i]
    out = np.empty_like(q)
    half = win // 2
    for i in range(len(q)):
        a, b = max(0, i - half), min(len(q), i + half + 1)
        m = q[a:b].mean(0)
        out[i] = m / np.linalg.norm(m)
    return Rotation.from_quat(out).as_matrix()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_root", required=True)
    ap.add_argument("--pred_root", required=True)
    ap.add_argument("--scales", required=True)
    ap.add_argument("--wins", default="1,5,9,15,21,31,41,61")
    a = ap.parse_args()

    scales = {l.split(",")[0]: float(l.split(",")[1])
              for l in Path(a.scales).read_text().splitlines()[1:] if l.strip()}
    wins = [int(x) for x in a.wins.split(",")]
    rows = {w: [] for w in wins}
    for d in sorted(p for p in Path(a.gt_root).iterdir() if p.is_dir()):
        f = Path(a.pred_root) / d.name / "1" / "camera_trajectory" / "cam_traj_map_000.txt"
        if not f.exists():
            continue
        gt = load_gt(d / "results_txt" / "images.txt")
        pr = load_pred(f)
        ids = sorted(pr)
        R = np.stack([pr[i][0] for i in ids])
        C = {i: pr[i][1] for i in ids}
        for w in wins:
            Rs = smooth_rot(ids, R, w)
            sm = {i: (Rs[n], C[i]) for n, i in enumerate(ids)}
            rows[w].append(rpe_rot(gt, sm))
    print(f"{'window':>8}{'rot d40 (deg)':>16}   per-sequence")
    for w in wins:
        v = rows[w]
        print(f"{w:8d}{np.mean(v):16.3f}   " + "  ".join(f"{x:.2f}" for x in v))
    print("\nATE is unaffected: only rotations are smoothed, camera centres are untouched.")


if __name__ == "__main__":
    main()
