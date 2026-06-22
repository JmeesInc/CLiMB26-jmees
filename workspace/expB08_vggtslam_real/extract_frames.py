#!/usr/bin/env python3
"""Rectified stride-6 keyframes as {frame_id:06d}.png (1-based) for VGGT-SLAM."""
import sys; from pathlib import Path; import cv2
sys.path.insert(0, "/data4/src/shunsuke/MICCAI2026/CLiMB/workspace/expB06_posecond/scripts")
import predict_cond as P
SRC = Path("/data4/src/shunsuke/MICCAI2026/CLiMB/workspace/expB04_realcv/input"); HERE = Path(__file__).resolve().parent
STRIDE = 6
for mp4 in sorted(SRC.glob("*.mp4")):
    cap = cv2.VideoCapture(str(mp4)); ok, f = cap.read(); h, w = f.shape[:2]
    calib, which = P.calib_for_sequence(mp4.stem); mapx, mapy, K = P.build_rectify_maps(calib, w, h, P.RECT_BALANCE)
    out = HERE / "frames" / mp4.stem; out.mkdir(parents=True, exist_ok=True)
    i = n = 0
    while ok:
        if i % STRIDE == 0:
            cv2.imwrite(str(out / f"{i+1:06d}.png"), cv2.remap(f, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)); n += 1
        i += 1; ok, f = cap.read()
    print(f"{mp4.stem}: {n} keyframes of {i} ({which})", flush=True)
