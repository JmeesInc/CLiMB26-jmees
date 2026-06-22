#!/usr/bin/env python3
"""Fisheye domain tools for the v001 LB failure analysis (ATE 28.3 vs CV 4.13).

Hypothesis: the pipeline breaks on real Kannala-Brandt kb4 fisheye video because
the local CV (sim) is pinhole. Two subcommands build the 2x2 evidence:

  synth    sim pinhole PNGs -> synthetic kb4 fisheye PNGs.
           Real Endoscope_01 calibration (1440x1080) scaled 2/3 to the sim
           960x720 canvas; kb4 coefficients are functions of theta only, so
           they transfer unchanged. Pixels beyond the sim 45.5deg half-FOV
           have no source -> black, reproducing the circular-border look.
  rectify  fisheye PNGs -> pinhole PNGs via cv2.fisheye undistortion
           (balance=0 -> all-valid crop, no black borders). This is the
           module that would ship in submit v002.

Both precompute one remap grid and apply it to every frame.
"""
import argparse
import re
from pathlib import Path

import cv2
import numpy as np

# calibu_fu_fv_u0_v0_kb4 params from data/Calibrations/Endoscope_01_geometrical.xml
CALIB_1440x1080 = dict(fu=717.6911, fv=718.0214, u0=734.7278, v0=552.0724,
                       kb4=[-0.1396589, -0.0003139987, 0.001505504, -0.0001316715])
SIM_K = dict(fx=472.64955100886374, fy=472.64955100886374, cx=479.5, cy=359.5,
             w=960, h=720)


def parse_calib_xml(path):
    """Read calibu kb4 params from a *_geometrical.xml."""
    txt = Path(path).read_text()
    m = re.search(r"<params>\s*\[(.*?)\]\s*</params>", txt, re.S)
    vals = [float(x) for x in m.group(1).replace(";", " ").split()]
    return dict(fu=vals[0], fv=vals[1], u0=vals[2], v0=vals[3], kb4=vals[4:8])


def fisheye_KD(calib, sx, sy):
    K = np.array([[calib["fu"] * sx, 0, calib["u0"] * sx],
                  [0, calib["fv"] * sy, calib["v0"] * sy],
                  [0, 0, 1]], np.float64)
    D = np.array(calib["kb4"], np.float64).reshape(4, 1)
    return K, D


def synth_maps(calib, out_w, out_h):
    """For each output fisheye pixel, the source pixel in the sim pinhole image."""
    sx, sy = out_w / 1440, out_h / 1080
    K_fe, D = fisheye_KD(calib, sx, sy)
    uu, vv = np.meshgrid(np.arange(out_w, dtype=np.float64),
                         np.arange(out_h, dtype=np.float64))
    pts = np.stack([uu.ravel(), vv.ravel()], 1)[None]        # 1xN x2
    norm = cv2.fisheye.undistortPoints(pts, K_fe, D)[0]      # normalized pinhole coords
    mapx = (SIM_K["fx"] * norm[:, 0] + SIM_K["cx"]).reshape(out_h, out_w).astype(np.float32)
    mapy = (SIM_K["fy"] * norm[:, 1] + SIM_K["cy"]).reshape(out_h, out_w).astype(np.float32)
    return mapx, mapy


def rectify_maps(calib, in_w, in_h, balance=0.0, p_mode="auto"):
    """p_mode='sim': rectify onto the known sim pinhole K (exact inverse of
    synth, the right config-C for the 2x2). 'auto': estimate P from the fisheye
    FOV (the deployable path for real videos, where the true pinhole is unknown)."""
    sx, sy = in_w / 1440, in_h / 1080
    K_fe, D = fisheye_KD(calib, sx, sy)
    if p_mode == "sim":
        P = np.array([[SIM_K["fx"], 0, SIM_K["cx"]],
                      [0, SIM_K["fy"], SIM_K["cy"]],
                      [0, 0, 1]], np.float64)
    else:
        P = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K_fe, D, (in_w, in_h), np.eye(3), balance=balance)
    mapx, mapy = cv2.fisheye.initUndistortRectifyMap(
        K_fe, D, np.eye(3), P, (in_w, in_h), cv2.CV_32FC1)
    return mapx, mapy, P


def process_dir(src, dst, mapx, mapy):
    dst.mkdir(parents=True, exist_ok=True)
    pngs = sorted(src.glob("*.png"))
    for p in pngs:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        out = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        cv2.imwrite(str(dst / p.name), out)
    return len(pngs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["synth", "rectify"])
    ap.add_argument("--src_root", required=True, help="dir of <seq>/*.png")
    ap.add_argument("--dst_root", required=True)
    ap.add_argument("--seqs", nargs="*", default=None)
    ap.add_argument("--calib_xml", default=None,
                    help="override Endoscope_01 with another *_geometrical.xml")
    ap.add_argument("--balance", type=float, default=0.0)
    ap.add_argument("--p_mode", choices=["auto", "sim"], default="sim")
    args = ap.parse_args()

    calib = parse_calib_xml(args.calib_xml) if args.calib_xml else CALIB_1440x1080
    src_root, dst_root = Path(args.src_root), Path(args.dst_root)
    seq_dirs = sorted(d for d in src_root.iterdir() if d.is_dir())
    if args.seqs:
        seq_dirs = [d for d in seq_dirs if d.name in args.seqs]

    maps = None
    for d in seq_dirs:
        sample = next(d.glob("*.png"))
        h, w = cv2.imread(str(sample)).shape[:2]
        if maps is None:
            if args.cmd == "synth":
                maps = synth_maps(calib, w, h)
                valid = ((maps[0] >= 0) & (maps[0] < w)
                         & (maps[1] >= 0) & (maps[1] < h)).mean()
                print(f"synth maps {w}x{h}: valid source coverage {valid*100:.1f}%")
            else:
                mx, my, P = rectify_maps(calib, w, h, args.balance, args.p_mode)
                maps = (mx, my)
                print(f"rectify maps {w}x{h}: P fx={P[0,0]:.1f} cx={P[0,2]:.1f}")
        n = process_dir(d, dst_root / d.name, *maps)
        print(f"  {d.name}: {n} frames -> {dst_root / d.name}", flush=True)


if __name__ == "__main__":
    main()
