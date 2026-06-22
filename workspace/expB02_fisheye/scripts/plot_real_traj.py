#!/usr/bin/env python3
"""Qualitative comparison of v002 (rectified) vs v001-style (passthrough)
trajectories on a real clip (no GT): 3D path + per-axis camera centre curves.
Smooth, bounded curves = healthy; jagged / exploding = broken geometry."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_traj(run_dir):
    p = Path(run_dir) / "camera_trajectory" / "cam_traj_map_000.txt"
    C = []
    for ln in p.read_text().splitlines():
        if ln.startswith("#") or not ln.strip():
            continue
        v = ln.split(",")
        C.append([float(v[2]), float(v[3]), float(v[4])])
    return np.array(C)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rect", required=True, help="run dir of rectified output")
    ap.add_argument("--raw", default=None, help="run dir of passthrough output")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    trajs = {"v002 rectified": load_traj(a.rect)}
    if a.raw:
        trajs["v001 passthrough"] = load_traj(a.raw)

    fig = plt.figure(figsize=(14, 5))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    for name, C in trajs.items():
        ax.plot(*C.T, lw=0.8, label=f"{name} (n={len(C)})")
    ax.set_title("camera centres (arbitrary metric scale)")
    ax.legend()

    ax2 = fig.add_subplot(1, 2, 2)
    for name, C in trajs.items():
        step = np.linalg.norm(np.diff(C, axis=0), axis=1)
        ax2.plot(step, lw=0.6, label=f"{name} |ΔC| median={np.median(step):.4f}")
    ax2.set_title("per-frame step length (jumps = breaks)")
    ax2.set_yscale("log")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(a.out, dpi=110)
    print(f"saved {a.out}")
    for name, C in trajs.items():
        step = np.linalg.norm(np.diff(C, axis=0), axis=1)
        print(f"{name}: n={len(C)} extent={np.ptp(C,0).round(3)} "
              f"step median={np.median(step):.4f} p99={np.percentile(step,99):.4f} "
              f"max={step.max():.4f}")


if __name__ == "__main__":
    main()
