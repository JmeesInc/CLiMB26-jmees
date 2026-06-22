#!/usr/bin/env python3
"""Disparity / keyframe-density EDA.

Replicates VGGT-SLAM's FrameTracker.compute_disparity (goodFeaturesToTrack +
LK optical flow, mean displacement from the last KEYFRAME) to answer:
  (A) sim: how many keyframes are kept at each min_disparity -> predicted TFR
      (TFR ceiling = keyframes_with_GT / 322), to tune the 堅実案.
  (B) real: per-consecutive-frame disparity at native fps -> how far we can
      downsample (stride) before motion-per-frame gets too large.
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import cv2


def good_features(gray):
    return cv2.goodFeaturesToTrack(gray, maxCorners=1000, qualityLevel=0.01,
                                   minDistance=8, blockSize=7)


def disp_between(gray0, pts0, gray1):
    if pts0 is None or len(pts0) < 10:
        return None
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(gray0, gray1, pts0, None,
                                          winSize=(21, 21), maxLevel=3,
                                          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    st = st.flatten()
    g0, g1 = pts0[st == 1], nxt[st == 1]
    if len(g0) < 10:
        return None
    return float(np.mean(np.linalg.norm(g1 - g0, axis=1)))


def keyframe_count(paths, min_disp, downsample=1):
    """Simulate VGGT-SLAM keyframe selection; return list of kept frame indices (into paths)."""
    paths = paths[::downsample]
    kept = []
    kf_gray, kf_pts = None, None
    for i, p in enumerate(paths):
        img = cv2.imread(str(p))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if kf_gray is None:
            kept.append(i); kf_gray, kf_pts = gray, good_features(gray); continue
        d = disp_between(kf_gray, kf_pts, gray)
        if d is None or d > min_disp:
            kept.append(i); kf_gray, kf_pts = gray, good_features(gray)
    return kept, len(paths)


def consec_disparity(paths, downsample=1):
    """Per-consecutive-frame (after downsample) mean disparity distribution."""
    paths = paths[::downsample]
    ds = []
    prev_gray, prev_pts = None, None
    for p in paths:
        img = cv2.imread(str(p)); gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            d = disp_between(prev_gray, prev_pts, gray)
            if d is not None:
                ds.append(d)
        prev_gray, prev_pts = gray, good_features(gray)
    return np.array(ds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sim", "real"], required=True)
    ap.add_argument("--folder", required=True)
    ap.add_argument("--pattern", default="*.png")
    ap.add_argument("--gt_frames", type=int, default=322, help="sim: # GT frames for TFR ceiling")
    args = ap.parse_args()

    paths = sorted(Path(args.folder).glob(args.pattern))
    print(f"folder={args.folder}  frames={len(paths)}")

    if args.mode == "sim":
        print("min_disp |  #KF | predicted TFR ceiling (%)")
        for md in [0, 2, 5, 8, 10, 15, 20, 30, 50]:
            kept, n = keyframe_count(paths, md)
            tfr = 100.0 * min(len(kept), args.gt_frames) / args.gt_frames
            print(f"  {md:5.0f}  | {len(kept):4d} | {tfr:6.1f}")
        dc = consec_disparity(paths)
        print(f"\nconsec-frame disparity (px): n={len(dc)} "
              f"min={dc.min():.2f} median={np.median(dc):.2f} mean={dc.mean():.2f} max={dc.max():.2f}")

    else:  # real
        for ds in [1, 2, 4, 8, 16]:
            dc = consec_disparity(paths, downsample=ds)
            if len(dc) == 0:
                continue
            print(f"downsample x{ds:<2d} (eff fps~{50/ds:5.1f}): consec-disp px "
                  f"median={np.median(dc):6.2f} mean={dc.mean():6.2f} p90={np.percentile(dc,90):6.2f} max={dc.max():6.2f}")
        print("\n(VGGT-SLAM default min_disparity=50; submap overlap uses 1 frame. "
              "Pick downsample so per-frame disparity stays well below ~50px but frames remain dense.)")


if __name__ == "__main__":
    main()
