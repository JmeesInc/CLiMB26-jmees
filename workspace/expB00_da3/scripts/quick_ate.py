#!/usr/bin/env python3
"""Standalone Sim(3) ATE of a CLiMB trajectory file vs sim GT, bypassing the
100-frame submap threshold so we can probe pose quality at any frame count."""
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "reference" / "evaluation"))
from evaluate_ate_scale import align, build_correspondence_matrices  # noqa
from colmap_utils import read_colmap_images  # noqa

SCALE = 100.0  # dm -> mm


def read_traj(path):
    ids, centers = [], []
    for ln in Path(path).read_text().splitlines():
        if not ln.strip() or ln.startswith("#"):
            continue
        t = ln.split(",")
        fid = int("".join(ch for ch in t[1] if ch.isdigit()))
        ids.append(fid)
        centers.append([float(t[2]), float(t[3]), float(t[4])])
    return np.array(ids), np.array(centers)


def main():
    traj = sys.argv[1]
    gt_images = sys.argv[2]
    _, gt_centers, gt_ids, _ = read_colmap_images(gt_images)
    gt_centers = gt_centers * SCALE
    sl_ids, sl_centers = read_traj(traj)
    model, data, common = build_correspondence_matrices(sl_centers, sl_ids, gt_centers, gt_ids)
    rot, transGT, trans_errorGT, trans, trans_error, s = align(model, data)
    print(f"common={len(common)}  scale={s:.4f}  "
          f"ATE mean={trans_errorGT.mean():.3f}mm  median={np.median(trans_errorGT):.3f}mm  "
          f"rmse={np.sqrt((trans_errorGT**2).mean()):.3f}mm")


if __name__ == "__main__":
    main()
