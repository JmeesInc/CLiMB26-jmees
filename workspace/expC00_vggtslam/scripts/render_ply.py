#!/usr/bin/env python3
"""Render the evaluator's PLY outputs (Sim3-aligned) into clean composite images.

The evaluator writes trajectories as dense POINT clouds (not polylines), so we
render GT(red) / SLAM(green) as bold points framed on the trajectory, with the
reconstructed map as a clipped, lightened point cloud behind them. 3 viewpoints.
"""
import argparse
from pathlib import Path
import numpy as np
import open3d as o3d
from open3d.visualization import rendering


def umat(point_size=2.0):
    m = rendering.MaterialRecord(); m.shader = "defaultUnlit"; m.point_size = point_size
    return m


def load_pts(path):
    pc = o3d.io.read_point_cloud(str(path))
    return pc if len(pc.points) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--W", type=int, default=800)
    ap.add_argument("--H", type=int, default=720)
    ap.add_argument("--max_points", type=int, default=80000)
    ap.add_argument("--clip", type=float, default=1.6)
    args = ap.parse_args()
    md = Path(args.map_dir)

    r = rendering.OffscreenRenderer(args.W, args.H)
    r.scene.set_background([1, 1, 1, 1])

    # trajectories (point clouds) -> framing + bold colored points
    traj_pts = []
    traj_geo = []
    for fname, color in [("colmap_trajectory.ply", [0.88, 0.1, 0.1]),
                         ("slam_trajectory.ply", [0.1, 0.65, 0.15])]:
        pc = load_pts(md / fname)
        if pc is None:
            continue
        pc.paint_uniform_color(color)
        traj_pts.append(np.asarray(pc.points))
        traj_geo.append((fname, pc, color))
    if not traj_pts:
        raise SystemExit(f"no trajectory PLY in {md}")
    tp = np.concatenate(traj_pts, 0)
    center = tp.mean(0)
    extent = float(np.linalg.norm(tp.max(0) - tp.min(0))) or 1.0

    # reconstructed map points: clip to trajectory neighbourhood, lighten
    pc = load_pts(md / "slam_points.ply")
    if pc is not None:
        P = np.asarray(pc.points)
        keep = np.where(np.linalg.norm(P - center, axis=1) < args.clip * extent)[0]
        pc = pc.select_by_index(keep)
        if len(pc.points) > args.max_points:
            idx = np.random.RandomState(0).choice(len(pc.points), args.max_points, replace=False)
            pc = pc.select_by_index(idx)
        if len(pc.points):
            if pc.has_colors():
                pc.colors = o3d.utility.Vector3dVector(np.asarray(pc.colors) * 0.45 + 0.5)
            else:
                pc.paint_uniform_color([0.72, 0.72, 0.74])
            r.scene.add_geometry("map", pc, umat(point_size=2.0))

    for fname, pc, color in traj_geo:
        r.scene.add_geometry(fname, pc, umat(point_size=7.0))

    views = [center + extent * np.array([0.0, -0.25, -1.1]),
             center + extent * np.array([1.1, -0.5, 0.15]),
             center + extent * np.array([0.05, -1.2, 0.05])]
    imgs = []
    for eye in views:
        r.scene.camera.look_at(center, eye, [0, -1, 0])
        imgs.append(np.asarray(r.render_to_image()))
    combo = np.concatenate(imgs, axis=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(combo.shape[1] / 100, combo.shape[0] / 100 + 0.4))
    ax.imshow(combo); ax.axis("off")
    ax.set_title(f"{args.title}    red=GT   green=SLAM   (3 views; map pts clipped+lightened)", fontsize=10)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=115, bbox_inches="tight")
    print(f"saved {args.out} (traj_extent={extent:.1f}, traj_pts={len(tp)})")


if __name__ == "__main__":
    main()
