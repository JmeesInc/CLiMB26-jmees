#!/usr/bin/env python3
"""Prediction EDA for the v001_da3_submap submission tree.

Follows CLAUDE.md "エラー分析の原則": look at the output before chasing the score.

The headline question: the submission scores an excellent ATE (2.8-8.1 mm) but a
terrible rotational RPE (23-25 deg at delta=40) on every sequence. RPE rotation is
computed from RELATIVE transforms, so it is invariant to the Sim(3) alignment that
rescues ATE -- and ATE only ever touches camera centres. A rotation-convention
mistake would therefore produce exactly this signature: great ATE, awful RPE_rot.
That now matters directly, because the new climb_score breaks ties on
Score_Rot = RotErr(delta=40) x W_rtf x W_t.

So this script:
  1. re-implements the evaluator's RPE (same formula as reference/evaluation/utils.py)
     and re-runs it under alternative rotation conventions, to test whether ours is
     simply wrong;
  2. profiles per-frame ATE after Sim(3)/Horn alignment (where does error live?);
  3. checks whether error concentrates at DA3 chunk boundaries (a chaining artifact
     would show up as a sawtooth with period step = CHUNK - OVERLAP = 5);
  4. writes figures to figs/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
CHUNK, OVERLAP = 8, 3
STEP = CHUNK - OVERLAP


# ----------------------------------------------------------------- loading --
def qvec_wxyz_to_R(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def read_gt(images_txt):
    """COLMAP images.txt -> {frame_id: (R_wc, C_w)}. COLMAP stores world-to-camera."""
    out = {}
    lines = [l for l in Path(images_txt).read_text().splitlines()]
    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        if not ln or ln.startswith("#"):
            i += 1
            continue
        p = ln.split()
        if len(p) < 10:
            i += 1
            continue
        qw, qx, qy, qz = map(float, p[1:5])
        tx, ty, tz = map(float, p[5:8])
        name = p[9]
        fid = int("".join(c for c in Path(name).stem if c.isdigit()))
        R_cw = qvec_wxyz_to_R(np.array([qw, qx, qy, qz]))
        R_wc = R_cw.T
        C_w = -R_wc @ np.array([tx, ty, tz])
        out[fid] = (R_wc, C_w)
        i += 2                      # COLMAP writes a points2D line per image
    return out


def read_pred(traj_txt):
    """Submission trajectory -> {frame_id: (R_wc, C_w)} (camera-to-world)."""
    out = {}
    for ln in Path(traj_txt).read_text().splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        p = ln.split(",")
        if len(p) < 9:
            continue
        fid = int("".join(c for c in Path(p[1]).stem if c.isdigit()))
        C = np.array([float(p[2]), float(p[3]), float(p[4])])
        R = qvec_wxyz_to_R(np.array([float(p[5]), float(p[6]), float(p[7]), float(p[8])]))
        out[fid] = (R, C)
    return out


# ---------------------------------------------------------------- geometry --
def horn_sim3(src, dst):
    """Sim(3) src->dst (Horn), as the evaluator does before measuring ATE."""
    ms, md = src.mean(0), dst.mean(0)
    s0, d0 = src - ms, dst - md
    U, D, Vt = np.linalg.svd(d0.T @ s0 / len(src))
    S = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ S @ Vt
    var = (s0 ** 2).sum() / len(src)
    s = np.trace(np.diag(D) @ S) / var if var > 1e-12 else 1.0
    t = md - s * R @ ms
    return s, R, t


def rpe(gt, pred, delta, rot_mode="asis"):
    """Evaluator's RPE, with the estimate's rotation optionally reinterpreted.

    rot_mode: asis | transpose (i.e. treat our stored quaternion as w2c instead)
    """
    common = sorted(set(gt) & set(pred))
    cs = set(common)
    trans, rot = [], []
    for i in common:
        j = i + delta
        if j not in cs:
            continue

        def T(d, fid, mode="asis"):
            R, C = d[fid]
            if mode == "transpose":
                R = R.T
            M = np.eye(4)
            M[:3, :3] = R
            M[:3, 3] = C
            return M

        Tr = np.linalg.inv(T(gt, i)) @ T(gt, j)
        Te = np.linalg.inv(T(pred, i, rot_mode)) @ T(pred, j, rot_mode)
        E = np.linalg.inv(Tr) @ Te
        trans.append(np.linalg.norm(E[:3, 3]))
        c = np.clip((np.trace(E[:3, :3]) - 1) / 2, -1, 1)
        rot.append(np.degrees(np.arccos(c)))
    return np.array(trans), np.array(rot)


def abs_rot_error(gt, pred, rot_mode="asis"):
    """Absolute orientation error after removing the single best global rotation.

    A convention error shows up as a large residual here too; honest drift shows a
    residual that grows along the sequence.
    """
    common = sorted(set(gt) & set(pred))
    M = np.zeros((3, 3))
    for f in common:
        Rg = gt[f][0]
        Rp = pred[f][0].T if rot_mode == "transpose" else pred[f][0]
        M += Rg @ Rp.T
    U, _, Vt = np.linalg.svd(M)
    A = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    errs = []
    for f in common:
        Rg = gt[f][0]
        Rp = pred[f][0].T if rot_mode == "transpose" else pred[f][0]
        E = Rg.T @ (A @ Rp)
        c = np.clip((np.trace(E) - 1) / 2, -1, 1)
        errs.append(np.degrees(np.arccos(c)))
    return np.array(common), np.array(errs)


# -------------------------------------------------------------------- main --
def analyse(seq, gt_root, pred_root, scale_mm):
    gt = read_gt(gt_root / seq / "results_txt" / "images.txt")
    pred = read_pred(pred_root / seq / "1" / "camera_trajectory" / "cam_traj_map_000.txt")
    common = sorted(set(gt) & set(pred))
    if len(common) < 10:
        return None

    src = np.array([pred[f][1] for f in common])
    dst = np.array([gt[f][1] for f in common])
    s, R, t = horn_sim3(src, dst)
    aligned = (s * (R @ src.T).T + t)
    ate = np.linalg.norm(aligned - dst, axis=1) * scale_mm

    res = {
        "seq": seq,
        "n_common": len(common),
        "n_pred": len(pred),
        "n_gt": len(gt),
        "sim3_scale": float(s),
        "ate_mean": float(ate.mean()),
        "ate_median": float(np.median(ate)),
        "ate_p95": float(np.percentile(ate, 95)),
        "ate_max": float(ate.max()),
    }
    for mode in ("asis", "transpose"):
        for d in (1, 40):
            tr, ro = rpe(gt, pred, d, mode)
            res[f"rpe{d}_rot_{mode}"] = float(ro.mean())
        _, ae = abs_rot_error(gt, pred, mode)
        res[f"abs_rot_{mode}"] = float(ae.mean())

    ids = np.array(common)
    return res, ids, ate, gt, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(REPO / "submit/v001_da3_submap/output"))
    ap.add_argument("--gt", default=str(REPO / "workspace/expA00_baseline_eval/colmap_gt"))
    ap.add_argument("--out", default=str(Path(__file__).parent))
    a = ap.parse_args()

    gt_root, pred_root, out = Path(a.gt), Path(a.pred), Path(a.out)
    (out / "figs").mkdir(parents=True, exist_ok=True)

    scales = {}
    sc = gt_root / "scales.csv"
    if sc.is_file():
        for ln in sc.read_text().splitlines()[1:]:
            p = ln.split(",")
            if len(p) >= 2:
                scales[p[0].strip()] = float(p[1])

    rows, per_seq = [], {}
    for d in sorted(pred_root.iterdir()):
        if not d.is_dir():
            continue
        r = analyse(d.name, gt_root, pred_root, scales.get(d.name, 1.0))
        if r is None:
            continue
        res, ids, ate, gt, pred = r
        rows.append(res)
        per_seq[d.name] = (ids, ate, gt, pred)
        print(f"{res['seq']}: ATE mean {res['ate_mean']:.2f} med {res['ate_median']:.2f} "
              f"p95 {res['ate_p95']:.2f} max {res['ate_max']:.2f} mm | "
              f"RPE40rot asis {res['rpe40_rot_asis']:.2f} / transpose "
              f"{res['rpe40_rot_transpose']:.2f} deg | absrot asis "
              f"{res['abs_rot_asis']:.2f} / transpose {res['abs_rot_transpose']:.2f}",
              flush=True)

    (out / "pred_eda.json").write_text(json.dumps(rows, indent=2))

    # ---- rotation convention verdict -------------------------------------
    a_r = np.mean([r["rpe40_rot_asis"] for r in rows])
    t_r = np.mean([r["rpe40_rot_transpose"] for r in rows])
    print(f"\nMEAN RPE40 rot: as-is {a_r:.2f} deg  vs  transposed {t_r:.2f} deg")
    print("VERDICT:", "convention looks WRONG (transpose is better)" if t_r < a_r * 0.9
          else "current convention is the better of the two")

    # ---- figures ---------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(per_seq)
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 6.4), squeeze=False)
    for k, (seq, (ids, ate, gt, pred)) in enumerate(per_seq.items()):
        ax = axes[0][k]
        ax.plot(ids, ate, lw=0.8)
        ax.set_title(f"{seq}\nper-frame ATE (mm)", fontsize=9)
        ax.set_xlabel("frame id", fontsize=7)
        ax.tick_params(labelsize=6)

        # error vs position within the chunk stride: a chaining artifact would
        # make this non-flat.
        ax2 = axes[1][k]
        phase = (ids - ids.min()) % STEP
        ax2.boxplot([ate[phase == p] for p in range(STEP)], widths=0.6,
                    flierprops=dict(markersize=1))
        ax2.set_title(f"ATE by chunk phase (step={STEP})", fontsize=8)
        ax2.set_xlabel("phase", fontsize=7)
        ax2.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(out / "figs" / "per_frame_ate.png", dpi=120)
    print(f"\nwrote {out/'figs'/'per_frame_ate.png'}")


if __name__ == "__main__":
    main()
