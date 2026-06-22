#!/usr/bin/env python3
"""Rectify the 4 real CV clips (kb4 -> pinhole) into mp4 + write the ORB-SLAM3
pinhole yaml, so the classical baseline can be scored on real data with the
same undistortion the DA3 submissions use."""
import sys, subprocess
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, "/data4/src/shunsuke/MICCAI2026/CLiMB/workspace/expB06_posecond/scripts")
import predict_cond as P

HERE = Path(__file__).resolve().parent
SRC = Path("/data4/src/shunsuke/MICCAI2026/CLiMB/workspace/expB04_realcv/input")
Ks = {}
for mp4 in sorted(SRC.glob("*.mp4")):
    cap = cv2.VideoCapture(str(mp4)); fps = cap.get(cv2.CAP_PROP_FPS)
    ok, first = cap.read(); h, w = first.shape[:2]
    calib, which = P.calib_for_sequence(mp4.stem)
    mapx, mapy, K = P.build_rectify_maps(calib, w, h, P.RECT_BALANCE)
    out = HERE / "input" / mp4.name
    ff = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                           "-s", f"{w}x{h}", "-r", f"{fps:.3f}", "-i", "-",
                           "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p", str(out)], stdin=subprocess.PIPE)
    n = 0
    while ok:
        ff.stdin.write(cv2.remap(first, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT).tobytes())
        n += 1; ok, first = cap.read()
    ff.stdin.close(); ff.wait(); cap.release()
    Ks[mp4.stem] = (K, w, h, fps, which)
    print(f"{mp4.stem}: {n} frames @ {fps:.1f} fps, {which}, fx={K[0,0]:.1f} cx={K[0,2]:.1f}", flush=True)

# one yaml per calibration (both seqs of a family share it); ORB entrypoint takes one SETTINGS,
# so write per-seq yamls and let run.sh pick per clip.
tmpl = (HERE.parent / "expA00_baseline_eval" / "scripts" / "Sim_Pinhole.yaml").read_text()
for seq, (K, w, h, fps, which) in Ks.items():
    y = tmpl
    import re
    y = re.sub(r"Camera\.fx: .*", f"Camera.fx: {K[0,0]:.6f}", y)
    y = re.sub(r"Camera\.fy: .*", f"Camera.fy: {K[1,1]:.6f}", y)
    y = re.sub(r"Camera\.cx: .*", f"Camera.cx: {K[0,2]:.6f}", y)
    y = re.sub(r"Camera\.cy: .*", f"Camera.cy: {K[1,2]:.6f}", y)
    y = re.sub(r"Camera\.width: .*", f"Camera.width: {w}", y)
    y = re.sub(r"Camera\.height: .*", f"Camera.height: {h}", y)
    y = re.sub(r"Camera\.fps: .*", f"Camera.fps: {fps:.1f}", y)
    (HERE / f"{seq}.yaml").write_text(y)
print("yamls written")
