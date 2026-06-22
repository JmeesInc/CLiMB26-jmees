#!/usr/bin/env python3
"""3D visualisation of the v001_da3_submap predictions: trajectory vs GT, the
predicted point cloud, and orbit / fly-through videos.

Everything is drawn in the GT frame: the prediction is mapped through the same
Sim(3)/Horn alignment the official evaluator applies before measuring ATE, so the
residual you see IS the ATE (in mm).

Outputs (figs/):
  traj3d_<seq>.png      GT vs prediction with per-frame error lines
  points3d_<seq>.png    predicted sparse map, GT trajectory overlaid
  orbit_<seq>.mp4       camera orbiting the aligned trajectories + map
  fly_<seq>.mp4         trajectory drawn frame by frame with a live error readout
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pred_eda import read_gt, read_pred, horn_sim3, REPO  # noqa: E402

GT_C, PR_C, ERR_C = "#2E7D32", "#C62828", "#FF9800"


def load_points(path, max_pts, sim3):
    """points3D.txt -> (N,3) in the GT frame + (N,3) colours in [0,1]."""
    xyz, rgb = [], []
    for ln in Path(path).read_text().splitlines():
        if not ln or ln.startswith("#"):
            continue
        p = ln.split()
        if len(p) < 7:
            continue
        xyz.append([float(p[1]), float(p[2]), float(p[3])])
        rgb.append([int(p[4]), int(p[5]), int(p[6])])
    if not xyz:
        return np.zeros((0, 3)), np.zeros((0, 3))
    xyz = np.array(xyz)
    rgb = np.array(rgb) / 255.0
    if len(xyz) > max_pts:
        i = np.random.RandomState(0).choice(len(xyz), max_pts, replace=False)
        xyz, rgb = xyz[i], rgb[i]
    s, R, t = sim3
    return (s * (R @ xyz.T).T + t), np.clip(rgb, 0, 1)


def equal_box(ax, P):
    c = P.mean(0)
    r = np.abs(P - c).max() * 1.05
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)


def trim_outliers(pts, cols, keep=0.99):
    """Drop the farthest points before drawing.

    The unprojected map has a thin tail of depth outliers; left in, they blow up
    the axis box and squash the actual lumen into a sliver.
    """
    if len(pts) == 0:
        return pts, cols
    d = np.linalg.norm(pts - np.median(pts, axis=0), axis=1)
    m = d <= np.quantile(d, keep)
    return pts[m], cols[m]


def topdown(ax):
    """Readable top-down: orthographic, no z clutter."""
    ax.view_init(elev=90, azim=-90)
    try:
        ax.set_proj_type("ortho")
    except Exception:
        pass
    ax.set_zticks([])
    ax.zaxis.line.set_lw(0.0)


def prepare(seq, gt_root, pred_root, scale_mm, max_pts):
    gt = read_gt(gt_root / seq / "results_txt" / "images.txt")
    run = pred_root / seq / "1"
    pred = read_pred(run / "camera_trajectory" / "cam_traj_map_000.txt")
    common = sorted(set(gt) & set(pred))
    src = np.array([pred[f][1] for f in common])
    dst = np.array([gt[f][1] for f in common])
    s, R, t = horn_sim3(src, dst)
    # work in mm so the numbers on the figure are the reported metric
    P = (s * (R @ src.T).T + t) * scale_mm
    G = dst * scale_mm
    err = np.linalg.norm(P - G, axis=1)
    pts, cols = load_points(run / "3D_maps" / "000" / "points3D.txt", max_pts, (s, R, t))
    return np.array(common), G, P, err, pts * scale_mm, cols


def fig_traj(seq, G, P, err, out):
    fig = plt.figure(figsize=(13, 5.6))
    for k, (elev, azim, title) in enumerate(
            [(22, -60, "3/4 view"), (89, -90, "top-down")]):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        # error lines first so the trajectories draw on top
        step = max(1, len(G) // 160)
        for i in range(0, len(G), step):
            ax.plot(*zip(G[i], P[i]), color=ERR_C, lw=0.7, alpha=0.8)
        ax.plot(*G.T, color=GT_C, lw=2.0, label="ground truth")
        ax.plot(*P.T, color=PR_C, lw=1.6, label="prediction (Sim(3)-aligned)")
        ax.scatter(*G[0], color="k", s=28, marker="o", label="start")
        if k == 1:
            topdown(ax)
        else:
            ax.view_init(elev=elev, azim=azim)
        equal_box(ax, np.vstack([G, P]))
        ax.set_title(f"{title}", fontsize=10)
        ax.tick_params(labelsize=6)
        if k == 0:
            ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(f"{seq} — trajectory vs GT   (ATE mean {err.mean():.2f} mm, "
                 f"max {err.max():.2f} mm; orange = per-frame error)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / f"traj3d_{seq}.png", dpi=130)
    plt.close(fig)


def fig_points(seq, G, P, pts, cols, out):
    pts, cols = trim_outliers(pts, cols)
    fig = plt.figure(figsize=(12, 5.6))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    if len(pts):
        ax.scatter(*pts.T, c=cols, s=0.25, alpha=0.35, linewidths=0)
    ax.plot(*G.T, color=GT_C, lw=2.0, label="GT trajectory")
    ax.plot(*P.T, color=PR_C, lw=1.4, label="prediction")
    ax.view_init(elev=22, azim=-60)
    equal_box(ax, np.vstack([G, P]))
    ax.set_title(f"predicted sparse map ({len(pts):,} pts shown)", fontsize=10)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=6)

    # depth-coloured view: the lumen structure is easier to read than RGB
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    if len(pts):
        d = np.linalg.norm(pts - pts.mean(0), axis=1)
        ax2.scatter(*pts.T, c=d, cmap="viridis", s=0.25, alpha=0.35, linewidths=0)
    ax2.plot(*G.T, color=GT_C, lw=2.0)
    topdown(ax2)
    equal_box(ax2, np.vstack([G, P]))
    ax2.set_title("top-down, coloured by distance from centroid", fontsize=10)
    ax2.tick_params(labelsize=6)
    fig.suptitle(f"{seq} — predicted point cloud", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / f"points3d_{seq}.png", dpi=130)
    plt.close(fig)


def video_orbit(seq, G, P, pts, cols, out, n=120, fps=24, vid_pts=6000):
    # matplotlib redraws the whole 3D scatter every frame, so the point budget --
    # not the frame count -- dominates render time. 40k points made this
    # effectively never finish; 6k renders in a couple of minutes and reads the same.
    pts, cols = trim_outliers(pts, cols)
    if len(pts) > vid_pts:
        i = np.random.RandomState(0).choice(len(pts), vid_pts, replace=False)
        pts, cols = pts[i], cols[i]
    fig = plt.figure(figsize=(7.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    if len(pts):
        ax.scatter(*pts.T, c=cols, s=0.6, alpha=0.3, linewidths=0)
    ax.plot(*G.T, color=GT_C, lw=2.2, label="ground truth")
    ax.plot(*P.T, color=PR_C, lw=1.6, label="prediction")
    equal_box(ax, np.vstack([G, P]))
    ax.legend(fontsize=9, loc="upper left")
    ax.set_title(f"{seq} — prediction vs GT + sparse map", fontsize=11)
    ax.tick_params(labelsize=6)

    def upd(i):
        ax.view_init(elev=18 + 12 * np.sin(2 * np.pi * i / n), azim=-180 + 360 * i / n)
        return ()

    anim = animation.FuncAnimation(fig, upd, frames=n, interval=1000 / fps, blit=False)
    anim.save(out / f"orbit_{seq}.mp4", writer=animation.FFMpegWriter(fps=fps, bitrate=3200))
    plt.close(fig)


def video_fly(seq, ids, G, P, err, out, fps=24, stride=2):
    fig = plt.figure(figsize=(11, 5.6))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    axe = fig.add_subplot(1, 2, 2)

    ax.plot(*G.T, color=GT_C, lw=1.0, alpha=0.25)
    lg, = ax.plot([], [], [], color=GT_C, lw=2.2, label="ground truth")
    lp, = ax.plot([], [], [], color=PR_C, lw=1.8, label="prediction")
    hg = ax.scatter([], [], [], color=GT_C, s=40)
    hp = ax.scatter([], [], [], color=PR_C, s=40)
    lerr, = ax.plot([], [], [], color=ERR_C, lw=2.0)
    equal_box(ax, np.vstack([G, P]))
    ax.view_init(elev=22, azim=-60)
    ax.legend(fontsize=8, loc="upper left")
    ax.tick_params(labelsize=6)

    axe.plot(ids, err, color="#455A64", lw=0.9)
    vline = axe.axvline(ids[0], color=PR_C, lw=1.4)
    axe.set_xlabel("frame id", fontsize=8)
    axe.set_ylabel("per-frame ATE (mm)", fontsize=8)
    axe.tick_params(labelsize=7)
    axe.grid(alpha=0.3)
    txt = axe.set_title("", fontsize=10)

    frames = range(1, len(G), stride)

    def upd(i):
        lg.set_data(G[:i, 0], G[:i, 1]); lg.set_3d_properties(G[:i, 2])
        lp.set_data(P[:i, 0], P[:i, 1]); lp.set_3d_properties(P[:i, 2])
        hg._offsets3d = ([G[i - 1, 0]], [G[i - 1, 1]], [G[i - 1, 2]])
        hp._offsets3d = ([P[i - 1, 0]], [P[i - 1, 1]], [P[i - 1, 2]])
        lerr.set_data([G[i - 1, 0], P[i - 1, 0]], [G[i - 1, 1], P[i - 1, 1]])
        lerr.set_3d_properties([G[i - 1, 2], P[i - 1, 2]])
        vline.set_xdata([ids[i - 1], ids[i - 1]])
        txt.set_text(f"{seq}  frame {ids[i-1]}   error {err[i-1]:.2f} mm")
        return ()

    anim = animation.FuncAnimation(fig, upd, frames=frames, interval=1000 / fps, blit=False)
    anim.save(out / f"fly_{seq}.mp4", writer=animation.FFMpegWriter(fps=fps, bitrate=3200))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(REPO / "submit/v001_da3_submap/output"))
    ap.add_argument("--gt", default=str(REPO / "workspace/expA00_baseline_eval/colmap_gt"))
    ap.add_argument("--out", default=str(HERE / "figs"))
    ap.add_argument("--seqs", nargs="*", default=None)
    ap.add_argument("--video_seqs", nargs="*", default=["Seq_0", "Seq_5"])
    ap.add_argument("--max_pts", type=int, default=40000)
    a = ap.parse_args()

    gt_root, pred_root, out = Path(a.gt), Path(a.pred), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    scales = {}
    sc = gt_root / "scales.csv"
    if sc.is_file():
        for ln in sc.read_text().splitlines()[1:]:
            p = ln.split(",")
            if len(p) >= 2:
                scales[p[0].strip()] = float(p[1])

    seqs = a.seqs or [d.name for d in sorted(pred_root.iterdir()) if d.is_dir()]
    for seq in seqs:
        ids, G, P, err, pts, cols = prepare(seq, gt_root, pred_root,
                                            scales.get(seq, 1.0), a.max_pts)
        fig_traj(seq, G, P, err, out)
        fig_points(seq, G, P, pts, cols, out)
        print(f"{seq}: figures done (ATE {err.mean():.2f} mm, {len(pts):,} pts)", flush=True)
        if seq in a.video_seqs:
            video_orbit(seq, G, P, pts, cols, out)
            video_fly(seq, ids, G, P, err, out)
            print(f"{seq}: videos done", flush=True)
    print(f"\nwrote to {out}")


if __name__ == "__main__":
    main()
