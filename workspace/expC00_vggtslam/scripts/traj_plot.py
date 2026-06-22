#!/usr/bin/env python3
"""Visualize GT vs SLAM trajectories (Sim(3)-aligned, like the evaluator) to see
WHERE error comes from. Plots ORB-SLAM3 and VGGT-SLAM against GT for one sim seq,
saves PNG + prints per-segment error.
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


def read_traj(path):
    ids, c = [], []
    for ln in Path(path).read_text().splitlines():
        if not ln.strip() or ln.startswith("#"):
            continue
        t = ln.split(",")
        ids.append(int("".join(ch for ch in t[1] if ch.isdigit())))
        c.append([float(t[2]), float(t[3]), float(t[4])])
    return np.array(ids), np.array(c)


def aligned(traj_path, gt_centers, gt_ids):
    sl_ids, sl_c = read_traj(traj_path)
    model, data, common = build_correspondence_matrices(sl_c, sl_ids, gt_centers, gt_ids)
    rot, transGT, errGT, trans, err, s = align(model, data)
    sl_aligned = (s * rot @ sl_c.T + transGT).T
    return sl_ids, sl_aligned, common, errGT


def main():
    seq = sys.argv[1] if len(sys.argv) > 1 else "Seq_5"
    gt_file = REPO / f"workspace/expA00_baseline_eval/colmap_gt/{seq}/results_txt/images.txt"
    _, gt_c, gt_ids, _ = read_colmap_images(str(gt_file))
    gt_c = gt_c * SCALE

    def largest_traj(base):
        cands = list(Path(base).glob(f"{seq}/*/camera_trajectory/cam_traj_map_*.txt"))
        best, bestn = None, -1
        for c in cands:
            n = sum(1 for ln in c.read_text().splitlines() if ln.strip() and not ln.startswith("#"))
            if n > bestn:
                best, bestn = c, n
        return best

    trajs = {
        "ORB-SLAM3": largest_traj(REPO / "workspace/expA00_baseline_eval/output"),
        "VGGT-SLAM": largest_traj(REPO / "workspace/expC00_vggtslam/output"),
    }
    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    ax.plot(gt_c[:, 0], gt_c[:, 1], gt_c[:, 2], "k-", lw=2, label="GT")
    colors = {"ORB-SLAM3": "tab:red", "VGGT-SLAM": "tab:green"}
    err_curves = {}
    for name, p in trajs.items():
        if not p.exists():
            print(f"[skip] {name}: {p} not found"); continue
        ids, al, common, errGT = aligned(p, gt_c, gt_ids)
        ax.plot(al[:, 0], al[:, 1], al[:, 2], color=colors[name], lw=1, alpha=0.8,
                label=f"{name} (ATE {errGT.mean():.1f}mm, n={len(common)})")
        err_curves[name] = (common, errGT)
        print(f"{seq} {name}: common={len(common)} ATE mean={errGT.mean():.2f} "
              f"median={np.median(errGT):.2f} max={errGT.max():.2f} mm")
    ax.set_title(f"{seq}: trajectories (Sim3-aligned)"); ax.legend(fontsize=8)

    # per-frame error vs frame id
    ax2 = fig.add_subplot(1, 3, 2)
    for name, (common, errGT) in err_curves.items():
        ax2.plot(common, errGT, color=colors[name], label=name, lw=1)
    ax2.set_xlabel("frame ID"); ax2.set_ylabel("ATE (mm)")
    ax2.set_title("per-frame error"); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    # error histogram
    ax3 = fig.add_subplot(1, 3, 3)
    for name, (common, errGT) in err_curves.items():
        ax3.hist(errGT, bins=30, alpha=0.5, color=colors[name], label=name)
    ax3.set_xlabel("ATE (mm)"); ax3.set_title("error distribution"); ax3.legend(fontsize=8)

    out = REPO / f"survey/competition/figs/traj_{seq}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=110)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
