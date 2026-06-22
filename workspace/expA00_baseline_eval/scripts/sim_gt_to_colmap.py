#!/usr/bin/env python3
"""Convert EndoMapper simulated-sequence GT (trajectory.csv) into the COLMAP
reference tree expected by reference/evaluation/slam_evaluation.py.

Sim GT format (trajectory.csv, ';'-separated):
    tX;tY;tZ;rX;rY;rZ;rW;time(s)
This is TUM-style camera-to-world: (tX,tY,tZ) = camera center in world (C_w),
(rX,rY,rZ,rW) = camera-to-world rotation quaternion (x,y,z,w order).
Units: decimeters (dm) per info.txt -> scale_to_target = 100 (dm -> mm).

COLMAP images.txt stores world-to-camera (T_cw): per image two lines,
  IMAGE_ID  QW QX QY QZ  TX TY TZ  CAMERA_ID  NAME
  <points line (ignored by evaluator, may be blank)>

Frame-ID convention: trajectory row k (0-based) <-> rgb/image_{k:04d}.png
<-> frame ID k+1 (1-based, matches ORB-SLAM3 output and the submission spec).
GT therefore covers frames 1..N_poses; any extra rgb frames are left unlabeled
(the evaluator only scores frame IDs common to GT and SLAM).

Output tree:
    <out>/<seq>/results_txt/images.txt
    <out>/<seq>/results_txt/points3D.txt
    <out>/scales.csv
    <out>/traj_lengths_mm.csv
    <out>/endomapper_short_seq_frames.csv
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

DM_TO_MM = 100.0  # 1 decimeter = 100 mm


def rotmat_to_qvec_wxyz(R_cw):
    """COLMAP quaternion order is (w, x, y, z)."""
    x, y, z, w = Rotation.from_matrix(R_cw).as_quat()  # scipy returns xyzw
    return np.array([w, x, y, z])


def parse_trajectory(traj_csv):
    """Return list of (C_w[3], R_wc[3x3]); drops malformed/truncated rows."""
    poses = []
    with open(traj_csv) as f:
        lines = f.read().splitlines()
    for ln in lines[1:]:  # skip header
        toks = ln.strip().split(";")
        if len(toks) < 7:
            continue
        try:
            vals = [float(t) for t in toks[:7]]
        except ValueError:
            continue  # truncated / partial line
        tX, tY, tZ, rX, rY, rZ, rW = vals
        q = np.array([rX, rY, rZ, rW])
        n = np.linalg.norm(q)
        if n < 1e-8:
            continue
        R_wc = Rotation.from_quat(q / n).as_matrix()
        poses.append((np.array([tX, tY, tZ]), R_wc))
    return poses


def write_images_txt(path, poses):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for k, (C_w, R_wc) in enumerate(poses):
            frame_id = k + 1  # 1-based
            R_cw = R_wc.T
            t_cw = -R_cw @ C_w
            qw, qx, qy, qz = rotmat_to_qvec_wxyz(R_cw)
            name = f"{frame_id:06d}.png"
            f.write(
                f"{frame_id} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} "
                f"{t_cw[0]:.9f} {t_cw[1]:.9f} {t_cw[2]:.9f} 1 {name}\n"
            )
            f.write("\n")  # points2D line (skipped by evaluator)


def write_points3d_stub(path, poses):
    """Minimal points file (not used for metrics, only visualization).
    Place a few points near the trajectory so readers don't choke."""
    path.parent.mkdir(parents=True, exist_ok=True)
    centers = np.array([c for c, _ in poses])
    mid = centers.mean(axis=0)
    with open(path, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        for i in range(8):
            off = np.array([(i % 2), ((i // 2) % 2), ((i // 4) % 2)], dtype=float) * 0.1
            p = mid + off
            f.write(f"{i+1} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} 128 128 128 0.0\n")


def traj_lengths_mm(poses):
    centers = np.array([c for c, _ in poses]) * DM_TO_MM
    total = float(np.sum(np.linalg.norm(np.diff(centers, axis=0), axis=1)))
    disp = float(np.linalg.norm(centers[-1] - centers[0]))
    return total, disp


def count_rgb(seq_dir):
    rgb = seq_dir / "rgb"
    if not rgb.is_dir():
        return 0
    return sum(1 for p in rgb.glob("*.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim_root", default="data/Simulated_Sequences")
    ap.add_argument("--out", default="workspace/expA00_baseline_eval/colmap_gt")
    args = ap.parse_args()

    sim_root = Path(args.sim_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    seq_dirs = sorted([d for d in sim_root.iterdir() if d.is_dir() and (d / "trajectory.csv").is_file()])
    if not seq_dirs:
        raise SystemExit(f"No sim sequences with trajectory.csv under {sim_root}")

    scales_rows = []
    traj_rows = []
    frames_rows = []

    for d in seq_dirs:
        seq = d.name  # e.g. Seq_0
        poses = parse_trajectory(d / "trajectory.csv")
        if len(poses) < 2:
            print(f"[WARN] {seq}: only {len(poses)} valid poses, skipping")
            continue
        write_images_txt(out / seq / "results_txt" / "images.txt", poses)
        write_points3d_stub(out / seq / "results_txt" / "points3D.txt", poses)

        total_mm, disp_mm = traj_lengths_mm(poses)
        n_rgb = count_rgb(d)
        scales_rows.append({"sequence": seq, "scale_to_target": f"{DM_TO_MM:.6f}"})
        traj_rows.append({"seq_name": seq,
                          "total_trajectory_mm": f"{total_mm:.6f}",
                          "displacement_mm": f"{disp_mm:.6f}"})
        frames_rows.append({"video_name": seq, "num_frames": n_rgb})
        print(f"{seq}: poses={len(poses)} rgb={n_rgb} traj_len={total_mm:.1f}mm disp={disp_mm:.1f}mm")

    with open(out / "scales.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sequence", "scale_to_target"])
        w.writeheader(); w.writerows(scales_rows)
    with open(out / "traj_lengths_mm.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seq_name", "total_trajectory_mm", "displacement_mm"])
        w.writeheader(); w.writerows(traj_rows)
    with open(out / "endomapper_short_seq_frames.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video_name", "num_frames"])
        w.writeheader(); w.writerows(frames_rows)

    print(f"\nWrote COLMAP GT tree + CSVs to {out}")


if __name__ == "__main__":
    main()
