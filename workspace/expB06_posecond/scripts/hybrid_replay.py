#!/usr/bin/env python3
"""Hybrid test: chain windows, with each window's scale borrowed from a GLOBAL
one-shot DA3 reconstruction of the whole clip -- gated by how well that global
reconstruction agrees with the window.

Why this might work where §14.2 (local signals) and §14.3 (long-span windows)
did not: the global one-shot (glob_s15_r392) is a second, INDEPENDENT estimate
of the clip's geometry with a single consistent scale. Where it agrees with the
chain it is trustworthy (003_a: 2.65 mm vs chain 3.45); where it disagrees it
is broken (001_a: 33 mm). The chain-vs-global residual is measurable without
GT, so it is a legitimate inference-time gate.

Per window w:
  s_G   = Horn scale, window-local geometry (all views)  -> global recon
  s_C   = Horn scale, chain-frame positions (shared views) -> global recon
  s_w   = s_G / s_C          # window scale expressed in chain units
  gate  = Horn residual of the window-local -> global fit, / window extent
Replayed with the §14 replay_scales machinery (no DA3 re-run) and scored.
"""
import argparse, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "workspace/expB04_realcv/scripts"))
sys.path.insert(0, str(REPO / "workspace/expB01_da3_submap/eda"))
from oracle_scale import replay_scales, ate, load_gt   # noqa: E402
from pred_eda import read_pred                         # noqa: E402


def horn(src, dst):
    ms, md = src.mean(0), dst.mean(0)
    s0, d0 = src - ms, dst - md
    U, D, Vt = np.linalg.svd(d0.T @ s0)
    S = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ S @ Vt
    den = (s0 ** 2).sum()
    s = float(np.trace(np.diag(D) @ S) / den) if den > 1e-12 else 1.0
    t = md - s * R @ ms
    res = np.linalg.norm((s * (R @ src.T).T + t) - dst, axis=1)
    return s, R, t, res


def window_scales(win, kf, glob, thr):
    """Return per-window scale vector (chain units) + gate stats."""
    C_g = [None] * len(kf)
    s_vec, gates, used = [], [], 0
    for w in win:
        s0, s1 = int(w["s0"]), int(w["s1"])
        C_loc = np.asarray(w["C_loc"], float)
        shared = np.asarray(w["shared"], int)
        views = list(range(s0, s1))
        G = np.array([glob[kf[j] + 1][1] if (kf[j] + 1) in glob else [np.nan] * 3 for j in views])
        ok = ~np.isnan(G[:, 0])
        s = 1.0
        if len(shared):
            gate = np.inf
            if ok.sum() >= 4:
                sG, _, _, resG = horn(C_loc[ok], G[ok])
                ext = np.linalg.norm(G[ok] - G[ok].mean(0), axis=1).mean() + 1e-12
                gate = resG.mean() / ext
                Cc = np.stack([C_g[k] for k in shared])
                Gs = np.array([glob[kf[k] + 1][1] for k in shared if (kf[k] + 1) in glob])
                if len(Gs) == len(shared) and len(Gs) >= 3:
                    sC, _, _, _ = horn(Cc, Gs)
                    if gate < thr and np.isfinite(sG) and np.isfinite(sC) and sC > 1e-9:
                        s = float(np.clip(sG / sC, 0.25, 4.0)); used += 1
            gates.append(gate)
            s_vec.append(s)
        # replay this window into C_g (same rule as replay_scales, s applied)
        R_loc = np.asarray(w["R_loc"], float)
        if len(shared) == 0:
            sA, RA, tA = 1.0, np.eye(3), np.zeros(3)
        else:
            loc = [k - s0 for k in shared]
            Cl = C_loc[loc]; Cg = np.stack([C_g[k] for k in shared])
            # rotation from orientations is inside replay_scales; here only centres are needed
            sA, RA, tA, _ = horn(Cl, Cg); sA = s
            tA = Cg.mean(0) - sA * RA @ Cl.mean(0)
        for k in range(s0, s1):
            if C_g[k] is None:
                C_g[k] = sA * RA @ C_loc[k - s0] + tA
    return np.array(s_vec), np.array(gates), used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--glob", required=True, help="global one-shot output tree")
    ap.add_argument("--gt_root", required=True)
    ap.add_argument("--scales", required=True)
    ap.add_argument("--thr", type=float, nargs="*", default=[0.0, 0.05, 0.1, 0.2, 0.3, 1e9])
    a = ap.parse_args()
    scales = {ln.split(",")[0]: float(ln.split(",")[1])
              for ln in Path(a.scales).read_text().splitlines()[1:] if ln.strip()}
    print(f"{'seq':10s}" + "".join(f"{('thr=%g' % t):>11s}" for t in a.thr) + "   (used windows @ each thr)")
    tot = {t: [] for t in a.thr}
    for f in sorted(Path(a.dump).glob("*.npz")):
        seq = f.stem
        d = np.load(f, allow_pickle=True)
        win, kf = d["windows"], d["kf"]
        gt = load_gt(Path(a.gt_root) / seq / "results_txt" / "images.txt")
        glob = read_pred(Path(a.glob) / seq / "1" / "camera_trajectory" / "cam_traj_map_000.txt")
        row, useds = [], []
        for t in a.thr:
            s_vec, gates, used = window_scales(win, kf, glob, t)
            P = replay_scales(win, kf, s_vec)
            v = ate(P, kf, gt, scales[seq]); row.append(v); tot[t].append(v); useds.append(used)
        print(f"{seq:10s}" + "".join(f"{v:11.2f}" for v in row) + "   " + "/".join(map(str, useds))
              + f"   gate median {np.median(gates):.3f}")
    print(f"{'MEAN':10s}" + "".join(f"{np.mean(tot[t]):11.2f}" for t in a.thr))
    print("thr=0 は常に s=1（= 現行）、thr=1e9 は常にグローバルのスケールを採用")


if __name__ == "__main__":
    main()
