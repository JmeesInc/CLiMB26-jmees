#!/usr/bin/env python3
"""Upper bound for per-window scale correction, measured against the real GT.

The chain applies, per window, a similarity (s, R, t) fitted on the cameras it
shares with the map so far. The submission fixes s=1. §11.6 showed that
*estimating* s from the overlap does not help. This script asks the prior
question: how much is there to win at all?

For every window we replay the chain three ways -- s=1, s=overlap-estimate, and
s=ORACLE (the scale that best matches GT for that window) -- and score the
resulting trajectory. If the oracle barely beats s=1, per-window scale is a dead
end regardless of how cleverly it is gated. If the oracle wins big, the gap tells
us what a confidence gate is competing for, and we can check which signal
(overlap baseline, ratio spread) predicts when the estimate is near the oracle.
"""
import argparse
from pathlib import Path

import numpy as np

from diagnose import load_gt, horn_sim3


def replay(win, kf, gt_by_kf, mode, gate=None):
    """Rebuild the keyframe poses with a given per-window scale rule.

    Faithful to predict.py: the rotation of each window fit comes from the
    camera ORIENTATIONS (M = sum Rg Rl^T, projected to SO(3)), not from centre
    correspondences -- with only 3 nearly-collinear shared cameras the two are
    very different, and a centre-based fit is much worse.
    """
    C_g = [None] * len(kf)
    R_g = [None] * len(kf)
    s_log = []
    for w in win:
        s0, s1 = int(w["s0"]), int(w["s1"])
        C_loc = np.asarray(w["C_loc"], float)
        R_loc = np.asarray(w["R_loc"], float)
        shared = np.asarray(w["shared"], int)

        if len(shared) == 0:
            s, R, t = 1.0, np.eye(3), np.zeros(3)
        else:
            loc = [k - s0 for k in shared]
            Cl, Rl = C_loc[loc], R_loc[loc]
            Cg = np.stack([C_g[k] for k in shared])
            Rg = np.stack([R_g[k] for k in shared])

            s_est = float(w["s_ovl"])
            if mode == "fixed1":
                s = 1.0
            elif mode == "overlap":
                s = s_est
            elif mode == "gated":
                s = s_est if gate(w) else 1.0
            elif mode == "oracle":
                s = oracle_scale(w, C_loc, Cl, Cg, gt_by_kf, s0, s1)
            else:
                raise ValueError(mode)
            if not (np.isfinite(s) and 0.25 < s < 4.0):
                s = 1.0

            M = sum(b @ a.T for b, a in zip(Rg, Rl))
            U, _, Vt = np.linalg.svd(M)
            R = U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt
            t = Cg.mean(0) - s * R @ Cl.mean(0)
            s_log.append(s)

        for k in range(s0, s1):
            if C_g[k] is None:
                C_g[k] = s * R @ C_loc[k - s0] + t
                R_g[k] = R @ R_loc[k - s0]
    return (np.array([c if c is not None else np.zeros(3) for c in C_g]),
            np.array(s_log))


def oracle_scale(w, C_loc, Cl, Cg, gt_by_kf, s0, s1):
    """The scale this window SHOULD have had, read off the GT.

    The map lives in an arbitrary global scale fixed by the seed window, so the
    GT-implied window extent has to be expressed in map units first: sigma is
    measured on the shared cameras, which are in both frames already.
    """
    Gs = [gt_by_kf.get(k) for k in range(s0, s1) if gt_by_kf.get(k) is not None]
    idx = [k - s0 for k in range(s0, s1) if gt_by_kf.get(k) is not None]
    if len(Gs) < 2:
        return 1.0
    G = np.stack(Gs)
    dl = np.linalg.norm(C_loc[idx] - C_loc[idx].mean(0), axis=1).mean()
    dg = np.linalg.norm(G - G.mean(0), axis=1).mean()
    if dl < 1e-12 or dg < 1e-12:
        return 1.0
    # sigma: map units per GT unit, from the shared cameras
    n_sh = len(Cg)
    Gsh = [gt_by_kf.get(k) for k in range(s0, s0 + n_sh)]
    if any(v is None for v in Gsh) or n_sh < 2:
        return 1.0
    Gsh = np.stack(Gsh)
    a = np.linalg.norm(Gsh - Gsh.mean(0), axis=1).mean()
    b = np.linalg.norm(Cg - Cg.mean(0), axis=1).mean()
    if a < 1e-12 or b < 1e-12:
        return 1.0
    return (b / a) * (dg / dl)


def ate(P, kf, gt, scale):
    ids = [i for i, k in enumerate(kf) if (k + 1) in gt]
    if len(ids) < 3:
        return float("nan")
    G = np.array([gt[kf[i] + 1] for i in ids])
    A = horn_sim3(P[ids], G)
    return float(np.linalg.norm(A - G, axis=1).mean() * scale)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--gt_root", required=True)
    ap.add_argument("--scales", required=True)
    a = ap.parse_args()

    scales = {ln.split(",")[0]: float(ln.split(",")[1])
              for ln in Path(a.scales).read_text().splitlines()[1:] if ln.strip()}

    gates = {
        "gate_base": lambda w: float(w["base_loc"]) > 0.02,
        "gate_spread": lambda w: float(w["spread"]) < 0.25,
        "gate_both": lambda w: float(w["base_loc"]) > 0.02 and float(w["spread"]) < 0.25,
    }
    modes = ["fixed1", "overlap", "oracle"] + list(gates)
    tot = {m: [] for m in modes}

    print(f"{'seq':12}" + "".join(f"{m:>10}" for m in modes))
    for f in sorted(Path(a.dump).glob("*.npz")):
        seq = f.stem
        d = np.load(f, allow_pickle=True)
        win, kf, n = d["windows"], d["kf"], int(d["n"])
        gt = load_gt(Path(a.gt_root) / seq / "results_txt" / "images.txt")
        gt_by_kf = {j: gt[k + 1] for j, k in enumerate(kf) if (k + 1) in gt}
        row = {}
        for m in modes:
            P, _ = replay(win, kf, gt_by_kf, m if m in ("fixed1", "overlap", "oracle") else "gated",
                          gates.get(m))
            row[m] = ate(P, kf, gt, scales[seq])
            tot[m].append(row[m])
        print(f"{seq:12}" + "".join(f"{row[m]:10.2f}" for m in modes))
    print(f"{'MEAN':12}" + "".join(f"{np.mean(tot[m]):10.2f}" for m in modes))


if __name__ == "__main__":
    main()


def true_oracle(win, kf, gt, scale):
    """Directly optimise the per-window scales against the final ATE.

    The extent-ratio 'oracle' above is only a heuristic and lost to s=1, which
    proves nothing on its own. This is the real upper bound: if a set of scales
    chosen with full knowledge of the GT -- and optimised for the metric itself
    -- cannot beat s=1, then no estimator or gate can, and per-window scale is
    a dead end. Scale errors compound down the chain, so s=1 is a strong prior.
    """
    from scipy.optimize import minimize

    ids = [i for i, k in enumerate(kf) if (k + 1) in gt]
    G = np.array([gt[kf[i] + 1] for i in ids])

    def obj(z):
        P = replay_scales(win, kf, np.exp(z))
        A = horn_sim3(P[ids], G)
        return float(np.linalg.norm(A - G, axis=1).mean() * scale)

    n_w = sum(1 for w in win if len(np.asarray(w["shared"], int)) > 0)
    z0 = np.zeros(n_w)
    r = minimize(obj, z0, method="Powell",
                 options=dict(maxiter=20000, maxfev=40000, xtol=1e-4, ftol=1e-4))
    return obj(z0), float(r.fun), np.exp(r.x)


def replay_scales(win, kf, s_vec):
    """replay() with an explicit per-window scale vector."""
    C_g = [None] * len(kf)
    R_g = [None] * len(kf)
    qi = 0
    for w in win:
        s0, s1 = int(w["s0"]), int(w["s1"])
        C_loc = np.asarray(w["C_loc"], float)
        R_loc = np.asarray(w["R_loc"], float)
        shared = np.asarray(w["shared"], int)
        if len(shared) == 0:
            s, R, t = 1.0, np.eye(3), np.zeros(3)
        else:
            loc = [k - s0 for k in shared]
            Cl, Rl = C_loc[loc], R_loc[loc]
            Cg = np.stack([C_g[k] for k in shared])
            Rg = np.stack([R_g[k] for k in shared])
            s = float(np.clip(s_vec[qi], 0.25, 4.0)); qi += 1
            M = sum(b @ a.T for b, a in zip(Rg, Rl))
            U, _, Vt = np.linalg.svd(M)
            R = U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt
            t = Cg.mean(0) - s * R @ Cl.mean(0)
        for k in range(s0, s1):
            if C_g[k] is None:
                C_g[k] = s * R @ C_loc[k - s0] + t
                R_g[k] = R @ R_loc[k - s0]
    return np.array([c if c is not None else np.zeros(3) for c in C_g])
