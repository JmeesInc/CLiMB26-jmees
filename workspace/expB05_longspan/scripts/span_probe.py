#!/usr/bin/env python3
"""Go/no-go for §14.3: can DA3 reconstruct LONG-SPAN windows at all?

§14.3 diagnosed the real weakness of the submission: with CHUNK=12/OVERLAP=3 the
window i only ever shares cameras with window i+1, so the greedy chain fixes each
window's scale forever and NOTHING constrains the trajectory over long ranges.
§14.1 showed 37% of ATE is recoverable by fixing per-window scale, and §14.2
showed no LOCAL signal predicts it. A long-span window is the obvious source of
non-local information.

But there is a hard prerequisite that must be measured before building any
solver: DA3 needs co-visibility. Keyframes far apart in a colonoscopy see
different parts of the lumen, so a window sampling every d-th keyframe may
reconstruct nothing usable. This script measures exactly that.

For each sub-sampling factor d, we build windows of CHUNK keyframes spaced d
apart (d=1 is the current, contiguous case), run DA3, Sim(3)-align the window's
own reconstruction to GT, and report the residual. If the residual stays flat as
d grows, long-span constraints are real and worth solving for. If it explodes,
§14.3 is dead and we stop here instead of building a solver on sand.

Reported per d:
  span_kf     keyframes covered by one window (CHUNK-1)*d + 1
  rel_err     Sim(3)-aligned residual / window extent  -- scale-free quality
  scale_err   |log(fitted Sim(3) scale / median)|      -- consistency of scale
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "submit/v003_da3_stride"))
sys.path.insert(0, str(REPO / "workspace/expB04_realcv/scripts"))

os.environ.setdefault("DA3_PRECISION", "fp16")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import predict as P                                    # noqa: E402
from diagnose import load_gt                            # noqa: E402


def sim3(src, dst):
    """Horn similarity src->dst returning (s, R, t) -- diagnose.horn_sim3
    returns only the aligned points and we need the scale itself."""
    ms, md = src.mean(0), dst.mean(0)
    s0, d0 = src - ms, dst - md
    U, D, Vt = np.linalg.svd(d0.T @ s0)
    S = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ S @ Vt
    sc = np.trace(np.diag(D) @ S) / max((s0 ** 2).sum(), 1e-12)
    return sc, R, md - sc * R @ ms


def gt_centers_by_kf(gt, kf):
    """GT camera centre per keyframe index (1-based frame ids), None if absent."""
    out = []
    for k in kf:
        fid = k + 1
        out.append(gt[fid] if fid in gt else None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--gt", required=True, help="<seq>/results_txt/images.txt")
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--chunk", type=int, default=12)
    ap.add_argument("--dilations", type=int, nargs="*", default=[1, 2, 4, 8, 16])
    ap.add_argument("--max_windows", type=int, default=20)
    ap.add_argument("--res", type=int, default=504)
    ap.add_argument("--min_gt", type=int, default=6)
    a = ap.parse_args()

    # predict.py applies the fp16 autocast patch inside main(), which we never
    # call -- without this the probe runs DA3 through Turing's EMULATED bf16 at
    # ~11 s/call instead of ~2.6 s (see daily_reports/20260828.md).
    import torch
    if P.PRECISION in ("fp16", "bf16"):
        torch.cuda.is_bf16_supported = (lambda *a, **k: P.PRECISION == "bf16")
        print(f"forcing {P.PRECISION} autocast")
    from depth_anything_3.api import DepthAnything3
    model = DepthAnything3.from_pretrained(P.MODEL_ID).to("cuda").eval()

    # rectify exactly as the submission does, so this measures the deployed path
    import cv2
    probe = cv2.VideoCapture(a.video)
    ok, first = probe.read()
    probe.release()
    h, w = first.shape[:2]
    calib, which = P.calib_for_sequence(Path(a.video).stem)
    rect_maps = P.build_rectify_maps(calib, w, h, P.RECT_BALANCE)
    rect_first = cv2.remap(first, rect_maps[0], rect_maps[1], cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    crop_box = P.content_crop_box(rect_first)

    vp = P.VideoWindow(a.video, [], rect_maps, crop_box)
    n = vp.count_frames(a.video)
    vp.release()
    kf = list(range(0, n, a.stride))
    if kf[-1] != n - 1:
        kf.append(n - 1)
    n_kf = len(kf)
    print(f"{Path(a.video).stem}: {n} frames, {n_kf} keyframes (stride {a.stride}), calib {which}")

    gt = load_gt(a.gt)
    gtc = gt_centers_by_kf(gt, kf)

    print(f"\n{'d':>3s} {'span_kf':>8s} {'span_fr':>8s} {'n_win':>6s} "
          f"{'rel_err':>9s} {'scale_cv':>9s} {'conf_ok':>8s}")
    for d in a.dilations:
        span = (a.chunk - 1) * d + 1
        if span > n_kf:
            print(f"{d:3d} {span:8d} {'-':>8s} {'0':>6s}  (span exceeds clip)")
            continue
        # evenly spaced window starts over the clip
        # overlapping starts are fine here: we are measuring per-window
        # reconstruction quality, not building a chain.
        starts = np.unique(np.linspace(0, n_kf - span, a.max_windows, dtype=int))
        rel_errs, scales = [], []
        for s0 in starts:
            idx = [s0 + j * d for j in range(a.chunk)]
            # COLMAP GT covers only part of some clips (Seq_001_a: 66% of
            # keyframes), so score on whatever subset of the window has GT.
            have = [j for j, i in enumerate(idx) if gtc[i] is not None]
            if len(have) < a.min_gt:
                continue
            frames = [kf[i] for i in idx]
            vid = P.VideoWindow(a.video, frames, rect_maps, crop_box)
            imgs = vid.window(0, len(frames))
            vid.release()
            if len(imgs) < a.chunk:
                continue
            import torch
            with torch.no_grad():
                pred = model.inference(imgs, process_res=a.res, export_format="mini_npz")
            extr = P.as_4x4(pred.extrinsics)
            C_loc_all, _ = P.centers_rots(extr)
            C_loc = C_loc_all[have]
            G = np.stack([gtc[idx[j]] for j in have])
            s, R, t = sim3(C_loc, G)
            resid = np.linalg.norm((s * (R @ C_loc.T).T + t) - G, axis=1)
            extent = np.linalg.norm(G - G.mean(0), axis=1).mean()
            rel_errs.append(resid.mean() / max(extent, 1e-9))
            scales.append(s)
        if not rel_errs:
            print(f"{d:3d} {span:8d} {'-':>8s} {'0':>6s}  (no GT coverage)")
            continue
        sc = np.array(scales)
        print(f"{d:3d} {span:8d} {span*a.stride:8d} {len(rel_errs):6d} "
              f"{np.mean(rel_errs):9.4f} {np.std(np.log(sc)):9.4f} "
              f"{np.mean(np.array(rel_errs) < 0.25)*100:7.0f}%")

    print("\n読み方:")
    print("  rel_err が d とともに横ばい → 長スパン窓は成立、§14.3 の求解に進む")
    print("  rel_err が急増          → 共視野が無く DA3 が破綻。§14.3 は中止")
    print("  参考: d=1 が現行の連続窓ベースライン")


if __name__ == "__main__":
    main()
