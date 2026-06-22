#!/usr/bin/env python3
"""Where does our rotational RPE(d=40) come from?

The leaderboard tie-break is Score_Rot, and our 7.487 deg is the worst among the
top four (3.269-4.27). This splits the error into the two things we control:

  estimate : the rotation DA3 + the chain produce at KEYFRAMES
  interp   : SLERP filling the 5 of every 6 frames we never look at

d=40 is measured in FRAME IDs and 40 % 6 != 0, so EVERY RPE pair at stride 6 has
at least one interpolated endpoint -- interpolation is not a second-order effect.

Oracle 'gt_kf': GT rotations at our keyframes, SLERP'd exactly as predict.py
does, GT centres kept. That is the floor the stride imposes; whatever our real
output sits above it is estimation error.
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def load_gt(images_txt):
    """COLMAP images.txt (world-to-camera) -> {frame_id: (R_wc, C)}."""
    out = {}
    for ln in Path(images_txt).read_text().splitlines():
        if ln.startswith("#") or not ln.strip():
            continue
        f = ln.split()
        if len(f) < 10 or not f[9].endswith(".png"):
            continue
        qw, qx, qy, qz = (float(f[1]), float(f[2]), float(f[3]), float(f[4]))
        t = np.array([float(f[5]), float(f[6]), float(f[7])])
        R_cw = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        out[int(f[9].replace(".png", ""))] = (R_cw.T, -R_cw.T @ t)
    return out


def load_pred(path):
    """CLiMB trajectory (camera-to-world) -> {frame_id: (R_wc, C)}."""
    out = {}
    for ln in Path(path).read_text().splitlines():
        if ln.startswith("#") or not ln.strip():
            continue
        v = ln.split(",")
        C = np.array([float(v[2]), float(v[3]), float(v[4])])
        R = Rotation.from_quat([float(v[6]), float(v[7]), float(v[8]), float(v[5])]).as_matrix()
        out[int(v[1].replace(".png", ""))] = (R, C)
    return out


def rpe_rot(ref, est, delta=40):
    """Exactly the evaluator's rotational RPE (utils.compute_rpe), in degrees."""
    common = set(ref) & set(est)
    errs = []
    for i in sorted(common):
        j = i + delta
        if j not in common:
            continue
        def T(d, k):
            R, C = d[k]
            M = np.eye(4); M[:3, :3] = R; M[:3, 3] = C
            return M
        rel_r = np.linalg.inv(T(ref, i)) @ T(ref, j)
        rel_e = np.linalg.inv(T(est, i)) @ T(est, j)
        E = np.linalg.inv(rel_r) @ rel_e
        errs.append(np.degrees(np.arccos(np.clip((np.trace(E[:3, :3]) - 1) / 2, -1, 1))))
    return float(np.mean(errs)) if errs else float("nan")


def slerp_from_kf(poses, kf_ids, all_ids):
    """Rebuild a dense trajectory from keyframe poses the way predict.py does."""
    kf = [k for k in kf_ids if k in poses]
    if len(kf) < 2:
        return None
    R = Rotation.from_matrix(np.stack([poses[k][0] for k in kf]))
    C = np.stack([poses[k][1] for k in kf])
    q = np.clip(np.asarray(all_ids, float), kf[0], kf[-1])
    Rd = Slerp(np.asarray(kf, float), R)(q).as_matrix()
    Cd = np.stack([np.interp(q, kf, C[:, d]) for d in range(3)], 1)
    return {i: (Rd[n], Cd[n]) for n, i in enumerate(all_ids)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_root", required=True)
    ap.add_argument("--pred_root", required=True)
    ap.add_argument("--stride", type=int, default=6)
    a = ap.parse_args()

    print(f"{'seq':12}{'ours':>8}{'gt_kf(floor)':>14}{'gt_dense':>10}   interp share")
    tot = {"ours": [], "floor": []}
    for d in sorted(p for p in Path(a.gt_root).iterdir() if p.is_dir()):
        gt = load_gt(d / "results_txt" / "images.txt")
        pf = Path(a.pred_root) / d.name / "1" / "camera_trajectory" / "cam_traj_map_000.txt"
        if not pf.exists():
            continue
        pred = load_pred(pf)
        ids = sorted(pred)
        # our keyframes: 1-based ids 1, 1+stride, ...
        kf_ids = list(range(1, max(ids) + 1, a.stride))

        ours = rpe_rot(gt, pred)
        # floor: GT poses, subsampled to our keyframes, SLERP'd back
        gt_kf = slerp_from_kf(gt, kf_ids, ids)
        floor = rpe_rot(gt, gt_kf) if gt_kf else float("nan")
        dense = rpe_rot(gt, gt)          # sanity: must be 0
        share = floor / ours * 100 if ours else float("nan")
        print(f"{d.name:12}{ours:8.2f}{floor:14.2f}{dense:10.2f}{share:14.0f}%")
        tot["ours"].append(ours); tot["floor"].append(floor)
    print(f"{'MEAN':12}{np.mean(tot['ours']):8.2f}{np.mean(tot['floor']):14.2f}"
          f"{0.0:10.2f}{np.mean(tot['floor'])/np.mean(tot['ours'])*100:14.0f}%")


if __name__ == "__main__":
    main()
