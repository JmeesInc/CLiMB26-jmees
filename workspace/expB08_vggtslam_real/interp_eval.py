#!/usr/bin/env python3
"""Interpolate VGGT-SLAM keyframe poses to every frame (same SLERP/linear rule as
the DA3 chain) so the 4 clips are scored on the same footing as v003."""
import sys, numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, "/data4/src/shunsuke/MICCAI2026/CLiMB/workspace/expB06_posecond/scripts")
from predict_cond import interpolate_poses, write_trajectory, write_points3d
import cv2
HERE = Path(__file__).resolve().parent
for seq in ["Seq_001_a", "Seq_001_c", "Seq_003_a", "Seq_003_b"]:
    rows = [l.split() for l in (HERE/"raw"/seq/"poses.txt").read_text().splitlines() if l.strip() and not l.startswith("#")]
    by_id = {}
    for r in rows:                                              # submaps overlap -> keep the last pose per frame
        fid = int(float(r[0])) - 1                               # 0-based frame index
        t = np.array(list(map(float, r[1:4]))); q = list(map(float, r[4:8]))  # xyzw, c2w
        R_wc = Rot.from_quat(q).as_matrix(); E = np.eye(4); E[:3, :3] = R_wc.T; E[:3, 3] = -R_wc.T @ t
        by_id[fid] = E
    kf = sorted(by_id); w2c = [by_id[k] for k in kf]
    cap = cv2.VideoCapture(str(HERE.parent/"expB04_realcv"/"input"/f"{seq}.mp4")); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS); cap.release()
    full = interpolate_poses(kf, np.stack(w2c), n)
    out = HERE/"output_interp"/seq/"1"
    write_trajectory(out/"camera_trajectory"/"cam_traj_map_000.txt", list(range(1, n+1)), full, fps)
    src = HERE/"output"/seq/"1"
    (out/"3D_maps"/"000").mkdir(parents=True, exist_ok=True)
    (out/"3D_maps"/"000"/"points3D.txt").write_text((src/"3D_maps"/"000"/"points3D.txt").read_text())
    (out/"runtime.txt").write_text((src/"runtime.txt").read_text())
    print(f"{seq}: {len(kf)} kf -> {n} frames", flush=True)
