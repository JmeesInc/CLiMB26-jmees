#!/usr/bin/env python3
"""Feed-forward pose+map with Depth-Anything-3 on EndoMapper sim sequences,
written into the CLiMB submission tree so reference/evaluation can score it.

DA3 outputs (prediction):
  extrinsics [N,4,4]  opencv/colmap world-to-camera  E = [R_cw | t_cw]
  intrinsics [N,3,3]  (at processed resolution, matches depth H,W)
  depth      [N,H,W]
  conf       [N,H,W]

CLiMB wants camera-to-world (T_wc): C_w = -R_cw^T t_cw, R_wc = R_cw^T, quat w-first.
Points3D: unproject depth to camera, R_wc @ X_cam + C_w -> world.

Frame-ID: rgb/image_{k:04d}.png -> frame ID k+1 (1-based; matches sim GT converter).
A submap needs >=100 frame IDs common with the COLMAP reference, so we must emit
>=100 poses -> stride chosen to keep >= MIN_POSES frames.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch


def rotmat_to_qvec_wxyz(R):
    # robust matrix->quaternion, returns (w,x,y,z)
    m = R
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s; x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s; z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s; x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s; z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s; x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s; z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def select_frames(rgb_dir, min_poses, max_poses):
    pngs = sorted(rgb_dir.glob("image_*.png"))
    items = []
    for p in pngs:
        try:
            num = int(p.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        items.append((num + 1, p))  # (frame_id, path)
    items.sort()
    n = len(items)
    if n <= max_poses:
        sel = items
    else:
        # stride to keep <= max_poses but >= min_poses
        stride = int(np.ceil(n / max_poses))
        sel = items[::stride]
        if len(sel) < min_poses:
            stride = max(1, int(np.floor(n / min_poses)))
            sel = items[::stride]
    return sel


def write_trajectory(path, frame_ids, extr, conv="w2c"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("# timestamp, name_image, tx, ty, tz, qw, qx, qy, qz\n")
        for i, fid in enumerate(frame_ids):
            E = extr[i]
            if conv == "w2c":  # E = [R_cw | t_cw] (opencv/colmap world-to-camera)
                R_cw = E[:3, :3]; t_cw = E[:3, 3]
                R_wc = R_cw.T
                C_w = -R_wc @ t_cw
            else:              # conv == "c2w": E already camera-to-world [R_wc | C_w]
                R_wc = E[:3, :3]; C_w = E[:3, 3]
            qw, qx, qy, qz = rotmat_to_qvec_wxyz(R_wc)
            ts = i / 30.0
            f.write(f"{ts:.6f},{fid:06d}.png,{C_w[0]:.9f},{C_w[1]:.9f},{C_w[2]:.9f},"
                    f"{qw:.9f},{qx:.9f},{qy:.9f},{qz:.9f}\n")


def write_points3d(path, depth, conf, intr, extr, images, conf_pct, max_points, stride_px=4):
    path.parent.mkdir(parents=True, exist_ok=True)
    N, H, W = depth.shape
    thr = np.percentile(conf, conf_pct) if conf is not None else -np.inf
    pts, cols = [], []
    ys = np.arange(0, H, stride_px)
    xs = np.arange(0, W, stride_px)
    gx, gy = np.meshgrid(xs, ys)
    gx = gx.ravel(); gy = gy.ravel()
    for i in range(N):
        K = intr[i]; Kinv = np.linalg.inv(K)
        E = extr[i]; R_cw = E[:3, :3]; t_cw = E[:3, 3]
        R_wc = R_cw.T; C_w = -R_wc @ t_cw
        d = depth[i][gy, gx]
        c = conf[i][gy, gx] if conf is not None else np.ones_like(d)
        m = (c >= thr) & np.isfinite(d) & (d > 0)
        if not m.any():
            continue
        uu = gx[m]; vv = gy[m]; dd = d[m]
        rays = (Kinv @ np.stack([uu, vv, np.ones_like(uu)], 0))  # 3xM
        cam = rays * dd[None, :]                                 # 3xM
        world = (R_wc @ cam) + C_w[:, None]                      # 3xM
        col = images[i][vv, uu] if images is not None else np.full((m.sum(), 3), 128)
        pts.append(world.T); cols.append(col)
    if pts:
        pts = np.concatenate(pts, 0); cols = np.concatenate(cols, 0)
        if len(pts) > max_points:
            idx = np.random.RandomState(0).choice(len(pts), max_points, replace=False)
            pts = pts[idx]; cols = cols[idx]
    else:
        pts = np.zeros((0, 3)); cols = np.zeros((0, 3))
    with open(path, "w") as f:
        f.write("# POINT3D_ID X Y Z R G B ERROR\n")
        for j, (p, c) in enumerate(zip(pts, cols)):
            f.write(f"{j+1} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                    f"{int(c[0])} {int(c[1])} {int(c[2])} 0.0\n")
    return len(pts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim_root", default="data/Simulated_Sequences")
    ap.add_argument("--out", default="workspace/expB00_da3/output")
    ap.add_argument("--model", default="depth-anything/DA3NESTED-GIANT-LARGE")
    ap.add_argument("--seqs", nargs="*", default=None, help="subset e.g. Seq_0")
    ap.add_argument("--min_poses", type=int, default=110)
    ap.add_argument("--max_poses", type=int, default=160)
    ap.add_argument("--process_res", type=int, default=504)
    ap.add_argument("--conf_pct", type=float, default=40.0)
    ap.add_argument("--max_points", type=int, default=200000)
    ap.add_argument("--runs", type=int, default=1, help="copy run1 result to runs 1..R")
    ap.add_argument("--pose_conv", choices=["w2c", "c2w"], default="w2c")
    args = ap.parse_args()

    from depth_anything_3.api import DepthAnything3
    device = torch.device("cuda")
    print(f"Loading {args.model} ...", flush=True)
    model = DepthAnything3.from_pretrained(args.model).to(device).eval()

    sim_root = Path(args.sim_root); out = Path(args.out)
    seq_dirs = sorted([d for d in sim_root.glob("Seq_*") if (d / "rgb").is_dir()])
    if args.seqs:
        seq_dirs = [d for d in seq_dirs if d.name in args.seqs]

    for d in seq_dirs:
        sel = select_frames(d / "rgb", args.min_poses, args.max_poses)
        frame_ids = [fid for fid, _ in sel]
        paths = [str(p) for _, p in sel]
        print(f"\n{d.name}: feeding {len(paths)} frames "
              f"(IDs {frame_ids[0]}..{frame_ids[-1]})", flush=True)

        t_proc0 = time.perf_counter()
        with torch.no_grad():
            pred = model.inference(paths, process_res=args.process_res,
                                   export_format="mini_npz")
        t_proc = time.perf_counter() - t_proc0

        extr = np.asarray(pred.extrinsics, dtype=np.float64)   # N,4,4 w2c
        intr = np.asarray(pred.intrinsics, dtype=np.float64)   # N,3,3
        depth = np.asarray(pred.depth, dtype=np.float32)       # N,H,W
        conf = np.asarray(pred.conf, dtype=np.float32) if pred.conf is not None else None
        imgs = np.asarray(pred.processed_images) if pred.processed_images is not None else None
        print(f"  extr{extr.shape} intr{intr.shape} depth{depth.shape} "
              f"proc={t_proc:.1f}s", flush=True)

        for run in range(1, args.runs + 1):
            run_dir = out / d.name / str(run)
            write_trajectory(run_dir / "camera_trajectory" / "cam_traj_map_000.txt",
                             frame_ids, extr, conv=args.pose_conv)
            npts = write_points3d(run_dir / "3D_maps" / "000" / "points3D.txt",
                                  depth, conf, intr, extr, imgs,
                                  args.conf_pct, args.max_points)
            rt = run_dir / "runtime.txt"
            rt.parent.mkdir(parents=True, exist_ok=True)
            with open(rt, "w") as f:
                f.write("init_seconds=0.000000\n")
                f.write(f"processing_seconds={t_proc:.6f}\n")
        print(f"  wrote {len(frame_ids)} poses, {npts} points x {args.runs} run(s)", flush=True)

    print("\nDone.")


if __name__ == "__main__":
    main()
