#!/usr/bin/env python3
"""Per-frame error profile on the REAL clips: is our error DRIFT or NOISE?

Drift (error grows with time) => the open-loop chaining is the problem =>
invest in pose-graph optimisation / loop closure.
Noise (flat, oscillating) => local geometry is the problem => invest in the
front-end (FOV, focal normalisation, fine-tune).
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def qwxyz_to_R(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)]])


def load_gt(images_txt):
    """COLMAP images.txt (world-to-camera) -> {frame_id: camera centre}."""
    out = {}
    for ln in Path(images_txt).read_text().splitlines():
        if ln.startswith("#") or not ln.strip():
            continue
        f = ln.split()
        if len(f) < 10 or not f[9].endswith(".png"):
            continue
        q = np.array([float(f[1]), float(f[2]), float(f[3]), float(f[4])])
        t = np.array([float(f[5]), float(f[6]), float(f[7])])
        R = qwxyz_to_R(q)
        out[int(f[9].replace(".png", ""))] = -R.T @ t
    return out


def load_traj(path):
    out = {}
    for ln in Path(path).read_text().splitlines():
        if ln.startswith("#") or not ln.strip():
            continue
        v = ln.split(",")
        out[int(v[1].replace(".png", ""))] = np.array([float(v[2]), float(v[3]), float(v[4])])
    return out


def horn_sim3(src, dst):
    """Least-squares similarity src->dst (Horn). Returns aligned src."""
    ms, md = src.mean(0), dst.mean(0)
    s0, d0 = src - ms, dst - md
    U, D, Vt = np.linalg.svd(d0.T @ s0)
    S = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / (s0 ** 2).sum()
    return s * (R @ src.T).T + (md - s * R @ ms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_root", required=True)
    ap.add_argument("--pred_root", required=True)
    ap.add_argument("--scales", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    scales = {}
    for ln in Path(a.scales).read_text().splitlines()[1:]:
        if ln.strip():
            k, v = ln.split(",")
            scales[k] = float(v)

    seqs = sorted(d.name for d in Path(a.gt_root).iterdir() if d.is_dir())
    fig, axes = plt.subplots(1, len(seqs), figsize=(5 * len(seqs), 4))
    axes = np.atleast_1d(axes)
    print(f"{'seq':12} {'ATE':>7} {'first⅓':>8} {'last⅓':>8} {'ratio':>6} {'verdict'}")
    for ax, s in zip(axes, seqs):
        gt = load_gt(Path(a.gt_root) / s / "results_txt" / "images.txt")
        pr = load_traj(Path(a.pred_root) / s / "1" / "camera_trajectory" / "cam_traj_map_000.txt")
        ids = sorted(set(gt) & set(pr))
        G = np.array([gt[i] for i in ids])
        P = np.array([pr[i] for i in ids])
        A = horn_sim3(P, G)
        err = np.linalg.norm(A - G, axis=1) * scales[s]
        k = len(err) // 3
        f3, l3 = err[:k].mean(), err[-k:].mean()
        verdict = "DRIFT" if l3 / f3 > 1.6 else ("noise" if l3 / f3 < 1.25 else "mixed")
        print(f"{s:12} {err.mean():7.2f} {f3:8.2f} {l3:8.2f} {l3/f3:6.2f}  {verdict}")
        ax.plot(ids, err, lw=0.7)
        ax.set_title(f"{s}  ATE {err.mean():.2f}mm  last/first {l3/f3:.2f}")
        ax.set_xlabel("frame id"); ax.set_ylabel("error [mm]")
    fig.tight_layout(); fig.savefig(a.out, dpi=110)
    print("saved", a.out)


if __name__ == "__main__":
    main()
