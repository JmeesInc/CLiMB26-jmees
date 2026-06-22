#!/usr/bin/env python3
"""Decompose the post-Sim(3) error on the real clips.

§11 established the error is not drift and not high-frequency noise. Normalising
by trajectory length then showed ATE/displacement is a near-constant 6-12% across
all four clips, i.e. the error tracks how far the camera actually travels. This
script asks *which component* of the motion is wrong:

  along-track  : error projected on the local GT motion direction -> a LOCAL
                 SCALE error (we cover ground too fast / too slow). Sim(3)
                 removes the global scale, so anything left here is local.
  cross-track  : error perpendicular to it -> a HEADING error (rotation).

and tracks the running path-length ratio pred/GT, which is the direct readout of
local scale consistency.
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diagnose import load_gt, load_traj, horn_sim3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_root", required=True)
    ap.add_argument("--pred_root", required=True)
    ap.add_argument("--scales", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smooth", type=int, default=15, help="frames, for the motion direction")
    a = ap.parse_args()

    scales = {ln.split(",")[0]: float(ln.split(",")[1])
              for ln in Path(a.scales).read_text().splitlines()[1:] if ln.strip()}
    seqs = sorted(d.name for d in Path(a.gt_root).iterdir() if d.is_dir())

    fig, axes = plt.subplots(3, len(seqs), figsize=(4.6 * len(seqs), 10))
    print(f"{'seq':12}{'ATE':>7}{'along':>8}{'cross':>8}{'along%':>8}"
          f"{'len_pred/len_gt':>16}{'ratio_std':>11}")
    for c, s in enumerate(seqs):
        gt = load_gt(Path(a.gt_root) / s / "results_txt" / "images.txt")
        pr = load_traj(Path(a.pred_root) / s / "1" / "camera_trajectory" / "cam_traj_map_000.txt")
        ids = sorted(set(gt) & set(pr))
        G = np.array([gt[i] for i in ids]) * scales[s]
        P = horn_sim3(np.array([pr[i] for i in ids]), np.array([gt[i] for i in ids])) * scales[s]
        E = P - G

        # local GT motion direction, smoothed so per-frame jitter does not
        # dominate the projection axis
        k = a.smooth
        Gs = np.stack([np.convolve(G[:, d], np.ones(k) / k, "same") for d in range(3)], 1)
        T = np.gradient(Gs, axis=0)
        n = np.linalg.norm(T, axis=1, keepdims=True)
        T = np.divide(T, n, out=np.zeros_like(T), where=n > 1e-12)

        along = np.einsum("ij,ij->i", E, T)
        cross = np.linalg.norm(E - along[:, None] * T, axis=1)
        ate = np.linalg.norm(E, axis=1)

        # running path length
        Lg = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(G, axis=0), axis=1))])
        Lp = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
        # per-window ratio over 30-frame blocks (local scale, not the global one)
        w = 30
        rat = [np.linalg.norm(np.diff(P[i:i + w], axis=0), axis=1).sum() /
               max(np.linalg.norm(np.diff(G[i:i + w], axis=0), axis=1).sum(), 1e-9)
               for i in range(0, len(G) - w, w)]
        rat = np.array(rat)

        frac = (np.abs(along).mean() /
                (np.abs(along).mean() + cross.mean()) * 100)
        print(f"{s:12}{ate.mean():7.2f}{np.abs(along).mean():8.2f}{cross.mean():8.2f}"
              f"{frac:7.1f}%{Lp[-1] / Lg[-1]:16.3f}{rat.std():11.3f}")

        axes[0, c].plot(ids, ate, lw=.7, label="ATE")
        axes[0, c].plot(ids, np.abs(along), lw=.7, label="|along|")
        axes[0, c].plot(ids, cross, lw=.7, label="cross")
        axes[0, c].set_title(f"{s}  ATE {ate.mean():.2f}mm"); axes[0, c].legend(fontsize=7)
        axes[0, c].set_xlabel("frame"); axes[0, c].set_ylabel("mm")

        axes[1, c].plot(ids, Lg, lw=.9, label="GT")
        axes[1, c].plot(ids, Lp, lw=.9, label="pred")
        axes[1, c].set_title(f"path length  ratio {Lp[-1]/Lg[-1]:.3f}"); axes[1, c].legend(fontsize=7)
        axes[1, c].set_xlabel("frame"); axes[1, c].set_ylabel("mm")

        axes[2, c].axhline(1.0, color="k", lw=.6)
        axes[2, c].plot(np.arange(len(rat)) * w, rat, marker="o", ms=2.5, lw=.8)
        axes[2, c].set_title(f"local scale (30f blocks) std {rat.std():.3f}")
        axes[2, c].set_xlabel("frame"); axes[2, c].set_ylabel("len_pred/len_gt")

    fig.tight_layout(); fig.savefig(a.out, dpi=110)
    print("saved", a.out)


if __name__ == "__main__":
    main()
