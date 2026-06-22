#!/usr/bin/env python3
"""Re-cut the training clips at a new virtual-pinhole FOV, frame-for-frame.

Deployment is moving to DA3_RECT_FSCALE=1.8 (real CV: ATE 4.374->4.15,
rot 5.18->4.99, zero inference cost), and an adapter trained on the old 111-deg
geometry would be learning a different camera. So the clips have to be re-cut.

The point of reading meta.json instead of re-running prep_clips.py is frame
correspondence: prep_clips picks its start with random.Random(hash(seq)), and
Python salts string hashes per process, so a re-run would silently choose
DIFFERENT segments. meta.json recorded the exact start, so re-cutting from it
keeps clip frame k the same physical frame -- which is what lets the 85k-edge
LightGlue rotation teacher carry over unchanged (a relative rotation is a
property of the camera motion, not of the rectification used to measure it).
"""
import json, os, shutil, sys
from multiprocessing import Pool
from pathlib import Path

import cv2

ROOT = Path("/data4/src/shunsuke/MICCAI2026/CLiMB")
sys.path.insert(0, str(ROOT / "submit/v003_da3_stride"))
import predict as P  # noqa: E402

SRC = ROOT / "workspace/expB09_lora/clips"
DST = ROOT / "workspace/expB09_lora/clips_f18"
FSCALE = float(os.environ.get("FSCALE", 1.8))
LONG = 504


def work(d):
    d = Path(d)
    meta = json.loads((d / "meta.json").read_text())
    seq, start, stride = meta["seq"], meta["start"], meta["stride"]
    n_want = len(list(d.glob("*.jpg")))
    out = DST / d.name
    if (out / f"{n_want-1:04d}.jpg").exists():
        return d.name, 0
    mov = ROOT / "data/Sequences" / seq / f"{seq}.mov"
    if not mov.exists():
        return d.name, 0
    cap = cv2.VideoCapture(str(mov))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    calib, _ = P.calib_for_sequence(seq)
    mapx, mapy = P.build_rectify_maps(calib, w, h, P.RECT_BALANCE, fscale=FSCALE)
    # the K that goes with these maps, at the resized resolution
    K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        __import__("numpy").array([[calib["fu"] * w / 1440.0, 0, calib["u0"] * w / 1440.0],
                                   [0, calib["fv"] * h / 1080.0, calib["v0"] * h / 1080.0],
                                   [0, 0, 1]], float),
        __import__("numpy").array(calib["kb4"], float).reshape(4, 1),
        (w, h), __import__("numpy").eye(3), balance=P.RECT_BALANCE).copy()
    K[0, 0] *= FSCALE; K[1, 1] *= FSCALE

    out.mkdir(parents=True, exist_ok=True)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    k = 0
    for i in range(n_want * stride):
        if not cap.grab():
            break
        if i % stride:
            continue
        ok, f = cap.retrieve()
        if not ok:
            break
        f = cv2.remap(f, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        sc = LONG / max(f.shape[:2])
        f = cv2.resize(f, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out / f"{k:04d}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 95])
        k += 1
        if k >= n_want:
            break
    cap.release()
    Ks = K * sc; Ks[2, 2] = 1.0
    meta["K"] = Ks.tolist(); meta["fscale"] = FSCALE
    (out / "meta.json").write_text(json.dumps(meta))
    tf = d / "lg_rot.npz"
    if tf.exists():
        shutil.copy2(tf, out / "lg_rot.npz")     # same frames -> same teacher
    return d.name, k


if __name__ == "__main__":
    dirs = sorted(str(p) for p in SRC.iterdir() if p.is_dir())
    DST.mkdir(parents=True, exist_ok=True)
    done = 0
    with Pool(10) as pool:
        for name, k in pool.imap_unordered(work, dirs):
            if k:
                done += 1
                if done % 100 == 0:
                    print(f"{done}/{len(dirs)} clips", flush=True)
    print(f"DONE {done} clips -> {DST}")
