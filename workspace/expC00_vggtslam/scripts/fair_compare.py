#!/usr/bin/env python3
"""Honest ORB vs VGGT comparison for one sim seq.

The evaluator aligns EACH submap independently (own Sim3), then ponders. So a
fragmented method (ORB) gets one alignment per fragment, hiding inter-fragment
drift. This plots GT + every ORB fragment (each independently aligned, distinct
colors) + VGGT (single aligned map), plus per-frame coverage/error, so the
trade-off (ORB = locally-tight fragments / VGGT = one globally-consistent map)
is visible.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "reference" / "evaluation"))
from evaluate_ate_scale import align, build_correspondence_matrices  # noqa
from colmap_utils import read_colmap_images  # noqa
SCALE = 100.0


def read_traj(p):
    ids, c = [], []
    for ln in Path(p).read_text().splitlines():
        if not ln.strip() or ln.startswith("#"):
            continue
        t = ln.split(",")
        ids.append(int("".join(ch for ch in t[1] if ch.isdigit())))
        c.append([float(t[2]), float(t[3]), float(t[4])])
    return np.array(ids), np.array(c)


def align_to_gt(ids, c, gt_c, gt_ids):
    model, data, common = build_correspondence_matrices(c, ids, gt_c, gt_ids)
    rot, tG, eG, tr, e, s = align(model, data)
    al = (s * rot @ c.T + tG).T
    return ids, al, common, eG


def main():
    seq = sys.argv[1] if len(sys.argv) > 1 else "Seq_5"
    gtf = REPO / f"workspace/expA00_baseline_eval/colmap_gt/{seq}/results_txt/images.txt"
    _, gt_c, gt_ids, _ = read_colmap_images(str(gtf)); gt_c = gt_c * SCALE

    fig = plt.figure(figsize=(16, 5.5))
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    ax.plot(*gt_c.T, "k-", lw=2.5, label="GT", zorder=1)

    # ORB: every map across all runs/maps with >=40 poses, each independently aligned
    orb_dirs = sorted((REPO / f"workspace/expA00_baseline_eval/output/{seq}").glob("*/camera_trajectory"))
    # pick the run with most total matched coverage (use run with max sum of poses)
    best_run, best_tot = None, -1
    for d in orb_dirs:
        tot = sum(sum(1 for ln in f.read_text().splitlines() if ln.strip() and not ln.startswith("#"))
                  for f in d.glob("cam_traj_map_*.txt"))
        if tot > best_tot:
            best_run, best_tot = d, tot
    oranges = ["#d73027", "#fc8d59", "#fee090", "#e08214", "#b35806"]
    ax2 = fig.add_subplot(1, 3, 2)
    orb_cov, orb_errs = [], []
    if best_run:
        for i, f in enumerate(sorted(best_run.glob("cam_traj_map_*.txt"))):
            ids, c = read_traj(f)
            if len(ids) < 40:
                continue
            ids, al, common, eG = align_to_gt(ids, c, gt_c, gt_ids)
            col = oranges[i % len(oranges)]
            ax.plot(*al.T, color=col, lw=1.3, alpha=0.9,
                    label=f"ORB frag{i} (n={len(common)}, ATE{eG.mean():.0f})")
            ax2.scatter(common, eG, s=6, color=col, label=f"ORB frag{i}")
            orb_cov += list(common); orb_errs += list(eG)

    # VGGT: single map
    vg = REPO / f"workspace/expC00_vggtslam/output/{seq}/1/camera_trajectory/cam_traj_map_000.txt"
    if vg.exists():
        ids, c = read_traj(vg)
        ids, al, common, eG = align_to_gt(ids, c, gt_c, gt_ids)
        ax.plot(*al.T, color="tab:green", lw=1.6, alpha=0.9,
                label=f"VGGT single (n={len(common)}, ATE{eG.mean():.0f})")
        ax2.plot(common, eG, color="tab:green", lw=1.2, label="VGGT")
        vg_cov = len(common)
    else:
        vg_cov = 0

    ax.set_title(f"{seq}: GT(black) / ORB fragments(reds, each own align) / VGGT(green, 1 align)", fontsize=9)
    ax.legend(fontsize=7)
    ax2.set_xlabel("frame ID"); ax2.set_ylabel("ATE (mm)"); ax2.set_title("per-frame error & coverage")
    ax2.legend(fontsize=7); ax2.grid(alpha=0.3); ax2.set_xlim(0, gt_ids.max())

    # coverage summary
    ax3 = fig.add_subplot(1, 3, 3)
    orb_unique = len(set(orb_cov))
    ax3.axis("off")
    txt = (f"{seq} coverage (of {len(gt_ids)} GT frames)\n\n"
           f"ORB-SLAM3: {orb_unique} frames via {len([1 for f in sorted(best_run.glob('cam_traj_map_*.txt')) if sum(1 for ln in f.read_text().splitlines() if ln.strip() and not ln.startswith('#'))>=40])} fragments\n"
           f"   each fragment independently Sim3-aligned\n"
           f"   pondered ATE (scored) ~ {np.mean(orb_errs):.1f}mm\n\n"
           f"VGGT-SLAM: {vg_cov} frames in 1 consistent map\n"
           f"   single Sim3 alignment (drift exposed)\n\n"
           f"=> ORB's lower scored ATE benefits from\n   per-fragment alignment + measuring only\n   on reconstructed frames.")
    ax3.text(0.02, 0.98, txt, va="top", ha="left", fontsize=10, family="monospace")

    out = REPO / f"survey/competition/figs/fair_{seq}.png"
    fig.tight_layout(); fig.savefig(out, dpi=115, bbox_inches="tight")
    print(f"saved {out}  ORB_cov={orb_unique} VGGT_cov={vg_cov}")


if __name__ == "__main__":
    main()
