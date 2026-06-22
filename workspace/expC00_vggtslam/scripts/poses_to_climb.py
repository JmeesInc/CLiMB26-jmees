#!/usr/bin/env python3
"""Convert VGGT-SLAM output (poses.txt + optional _points.pcd) into a CLiMB
submission subtree:  <out>/<run>/camera_trajectory/cam_traj_map_000.txt
                     <out>/<run>/3D_maps/000/points3D.txt
                     <out>/<run>/runtime.txt

VGGT-SLAM poses.txt line:  frame_id  x y z  qx qy qz qw   (quat xyzw)
Filenames were prepared as {frame_id:06d}.png so frame_id == CLiMB 1-based ID.

Pose convention is decided empirically (--pose_conv). The (x,y,z, R) from
VGGT-SLAM's projective decompose_camera may be world-to-camera or camera-to-world;
we test both and keep whichever yields lower ATE.
"""
import argparse
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def read_poses(path):
    ids, ts, qs = [], [], []
    for ln in Path(path).read_text().splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        v = [float(x) for x in ln.split()]
        if len(v) < 8:
            continue
        ids.append(int(round(v[0])))
        ts.append(np.array(v[1:4]))
        qs.append(np.array(v[4:8]))  # xyzw
    return ids, ts, qs


def write_traj(path, ids, ts, qs, conv):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("# timestamp, name_image, tx, ty, tz, qw, qx, qy, qz\n")
        for i, (fid, t, q) in enumerate(zip(ids, ts, qs)):
            R_mat = Rotation.from_quat(q).as_matrix()  # q is xyzw
            if conv == "w2c":            # decomposed pose is world-to-camera [R_cw|t_cw]
                R_wc = R_mat.T
                C_w = -R_wc @ t
            else:                        # conv == "c2w": pose already camera-to-world
                R_wc = R_mat
                C_w = t
            qx, qy, qz, qw = Rotation.from_matrix(R_wc).as_quat()
            f.write(f"{i/30.0:.6f},{fid:06d}.png,{C_w[0]:.9f},{C_w[1]:.9f},{C_w[2]:.9f},"
                    f"{qw:.9f},{qx:.9f},{qy:.9f},{qz:.9f}\n")


def write_points(path, pcd_file, max_points=200000):
    path.parent.mkdir(parents=True, exist_ok=True)
    pts = None
    if pcd_file and Path(pcd_file).is_file():
        try:
            import open3d as o3d
            pc = o3d.io.read_point_cloud(str(pcd_file))
            pts = np.asarray(pc.points)
            cols = (np.asarray(pc.colors) * 255).astype(int) if pc.has_colors() else None
        except Exception:
            pts = None
    with open(path, "w") as f:
        f.write("# POINT3D_ID X Y Z R G B ERROR\n")
        if pts is None or len(pts) == 0:
            f.write("1 0.0 0.0 0.0 128 128 128 0.0\n")
            return 0
        if len(pts) > max_points:
            idx = np.random.RandomState(0).choice(len(pts), max_points, replace=False)
            pts = pts[idx]; cols = cols[idx] if cols is not None else None
        for j, p in enumerate(pts):
            c = cols[j] if cols is not None else (128, 128, 128)
            f.write(f"{j+1} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                    f"{int(c[0])} {int(c[1])} {int(c[2])} 0.0\n")
    return len(pts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", required=True)
    ap.add_argument("--pcd", default=None)
    ap.add_argument("--out_run_dir", required=True, help="<seq>/<run> dir")
    ap.add_argument("--pose_conv", choices=["w2c", "c2w"], default="c2w")
    ap.add_argument("--proc_seconds", type=float, default=0.0)
    args = ap.parse_args()

    ids, ts, qs = read_poses(args.poses)
    run = Path(args.out_run_dir)
    write_traj(run / "camera_trajectory" / "cam_traj_map_000.txt", ids, ts, qs, args.pose_conv)
    npts = write_points(run / "3D_maps" / "000" / "points3D.txt", args.pcd)
    (run / "runtime.txt").write_text(
        f"init_seconds=0.000000\nprocessing_seconds={args.proc_seconds:.6f}\n")
    print(f"poses={len(ids)} (IDs {ids[0]}..{ids[-1]}) points={npts} conv={args.pose_conv} -> {run}")


if __name__ == "__main__":
    main()
