#!/usr/bin/env python3
"""Training clips for self-supervised LoRA: random 12-s segments of the TRAINVAL
real sequences, decoded at the deployed keyframe stride (6), rectified kb4->pinhole
exactly as predict.py does, resized to long side 504, saved as JPEG.

Excluded: every sequence in the official test split (forbidden) and Seq_001 /
Seq_003 (they are the real-CV clips; keeping them out keeps the CV honest).
"""
import os, sys, json, glob, random
from pathlib import Path
from multiprocessing import Pool
import cv2, numpy as np
sys.path.insert(0, "/data4/src/shunsuke/MICCAI2026/CLiMB/workspace/expB06_posecond/scripts")
import predict_cond as P

ROOT = Path("/data4/src/shunsuke/MICCAI2026/CLiMB")
OUT = ROOT / "workspace/expB09_lora/clips"
STRIDE, CLIP_S, N_CLIPS, LONG = 6, 12.0, 20, 504
test = {l.strip() for l in open(ROOT / "official_data/EndoMapper Splits/test.txt") if l.strip() and not l.startswith("#")}
tv = {l.strip() for l in open(ROOT / "official_data/EndoMapper Splits/trainval.txt") if l.strip() and not l.startswith("#")}
EXCL = {"Seq_001", "Seq_003"}

def work(seq_dir):
    seq = os.path.basename(seq_dir)
    mov = os.path.join(seq_dir, f"{seq}.mov")
    if seq in test or seq not in tv or seq in EXCL or not os.path.exists(mov):
        return seq, 0
    info = json.load(open(os.path.join(seq_dir, f"{seq}_info.json"))) if os.path.exists(os.path.join(seq_dir, f"{seq}_info.json")) else {}
    if str(info.get("procedure", info.get("type", "colonoscopy"))).lower().startswith("gastro"):
        return seq, 0
    cap = cv2.VideoCapture(mov); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS) or 40
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    calib, _ = P.calib_for_sequence(seq); mapx, mapy, K = P.build_rectify_maps(calib, w, h, P.RECT_BALANCE)
    L = int(CLIP_S * fps); rng = random.Random(hash(seq) & 0xffff); made = 0
    for c in range(N_CLIPS):
        if n <= L + 1: break
        s0 = rng.randrange(0, n - L)
        d = OUT / f"{seq}_c{c:02d}"; d.mkdir(parents=True, exist_ok=True)
        cap.set(cv2.CAP_PROP_POS_FRAMES, s0); k = 0; ok = True
        for i in range(L):
            ok = cap.grab()
            if not ok: break
            if i % STRIDE == 0:
                ok, f = cap.retrieve()
                if not ok: break
                f = cv2.remap(f, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                sc = LONG / max(f.shape[:2]); f = cv2.resize(f, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(d / f"{k:04d}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 95]); k += 1
        if k >= 18: made += 1
        (d / "meta.json").write_text(json.dumps(dict(seq=seq, start=s0, fps=fps, stride=STRIDE, K=(K * sc).tolist())))
    cap.release(); return seq, made

if __name__ == "__main__":
    dirs = sorted(glob.glob(str(ROOT / "data/Sequences/Seq_*")))
    with Pool(12) as p:
        tot = 0
        for seq, made in p.imap_unordered(work, dirs):
            if made: tot += made; print(f"{seq}: {made} clips", flush=True)
    print(f"TOTAL clips: {tot}")
