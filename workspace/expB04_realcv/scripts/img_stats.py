#!/usr/bin/env python3
"""Why does LightGlue matching collapse on some clips but not others?

rotavg quality tracks the surviving feature-edge density almost exactly
(003_a 5.07 edges/node -> rot 0.88; 001_c 1.11 -> 4.65), so the rotation problem
is really a matching problem. This measures the endoscopy-specific image
properties that are known to break matching, per clip:

  specular : saturated pixels -- highlights move WITH the co-located light, so
             they are strong but non-rigid "features"
  texture  : Laplacian energy on the non-specular, non-dark region
  dark     : underexposed lumen (no signal at all)
  blur     : ratio of high- to low-frequency gradient energy
  dI       : frame-to-frame brightness change -- the co-located light source
             makes intensity a function of distance, breaking photometric
             constancy that descriptors implicitly rely on
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def stats(gray, bgr):
    sat = (bgr.max(2) >= 250)
    dark = (gray < 25)
    valid = ~(sat | dark)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    tex = float(np.abs(lap[valid]).mean()) if valid.sum() > 1000 else 0.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    hi = float(np.abs(lap).mean())
    lo = float(np.hypot(gx, gy).mean())
    return dict(spec=sat.mean() * 100, dark=dark.mean() * 100, tex=tex,
                blur=hi / max(lo, 1e-6), mean=float(gray.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--every", type=int, default=20)
    a = ap.parse_args()

    print(f"{'seq':12}{'spec%':>7}{'dark%':>7}{'texture':>9}{'blur':>7}"
          f"{'bright':>8}{'dI/I %':>8}")
    for d in sorted(p for p in Path(a.root).iterdir() if p.is_dir()):
        mp4 = d / f"{d.name}.mp4"
        if not mp4.exists():
            continue
        cap = cv2.VideoCapture(str(mp4))
        acc, prev, dI, i = [], None, [], 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            if i % a.every == 0:
                g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                acc.append(stats(g, f))
                m = float(g.mean())
                if prev is not None:
                    dI.append(abs(m - prev) / max(prev, 1e-6) * 100)
                prev = m
            i += 1
        cap.release()
        k = lambda n: np.mean([x[n] for x in acc])
        print(f"{d.name:12}{k('spec'):7.2f}{k('dark'):7.1f}{k('tex'):9.2f}"
              f"{k('blur'):7.3f}{k('mean'):8.1f}{np.mean(dI):8.2f}")


if __name__ == "__main__":
    main()
