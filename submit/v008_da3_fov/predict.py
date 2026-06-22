#!/usr/bin/env python3
"""CLiMB submission (v006 = v004 [LoRA] + rotation averaging, workspace/expB12_rotation) (v004 = v003 + LoRA adapter, see workspace/expB09_lora/SESSION_NOTES.md) v002 — DA3 submap SLAM + Kannala-Brandt rectification.

v001 scored LB climb 42.419 (rank 7/7): ATE 28.28 mm / RPE_rot 34.0 deg against
a local sim CV of 4.13 mm. Root cause isolated in workspace/expB02_fisheye by
synthesising the real kb4 fisheye onto the sim sequences (official evaluator,
chunk8/ov3/s=1, Seq_0+Seq_5):

  A  pinhole passthrough (v001's CV condition)   ATE  5.75 mm / rot ~24 deg
  B  kb4 fisheye passthrough (v001 on real data) ATE 19.83    / rot 36.0   <- matches LB 28.3/34.0
  C  fisheye -> cv2.fisheye rectify              ATE  5.57    / rot 24.0   <- full recovery

v002 therefore adds ONE change over v001: every decoded frame is undistorted to
a pinhole view (cv2.fisheye, balance=0 -> no black borders) before DA3. The
calibration comes from the official Calibrations/ release baked into the image
(calib_table.py): the sequence number parsed from the video name selects the
exact per-endoscope kb4 when the public seq->endoscope map covers it, otherwise
the mean of all 18 calibrations (they agree within ~2% in-cluster and share
k1~-0.12; DA3 self-estimates focal length, so removing the non-pinhole
distortion shape is what matters). DA3_RECTIFY=0 disables it (local pinhole-sim
regression tests).

Source: workspace/expB01_da3_submap (chunk_slam.py) + workspace/expB02_fisheye,
sim 6-seq local CV with the official evaluator:

  ORB-SLAM3 baseline (reference/ORBSLAM)          6.63 mm ATE / 72.6 TFR / 5-6 success
  VGGT-SLAM 2.0 (min_disparity=2)                11.73      / 78.4     / 6-6
  DA3 submap, per-chunk Sim(3) scale estimate     7.09      / 96.3     / 6-6
  DA3 submap, scale FIXED to 1 (v001/v002 core)   5.25      / 96.3     / 6-6

METHOD
  Depth-Anything-3 is run on short overlapping windows of CHUNK frames (feed-
  forward, no training). Each call returns window-local world-to-camera poses;
  consecutive windows share OVERLAP frames, and the local window is mapped onto
  the global map by a rigid transform fitted on those shared cameras:

    - rotation from the camera ORIENTATIONS (sum of R_glob @ R_loc^T, projected
      to SO(3)). Forward colonoscopy motion puts the camera centres on a
      near-straight line, where a centres-only fit is ill-conditioned.
    - translation from the centroids.
    - scale FIXED AT 1. DA3 depth is metric and measurably consistent across
      windows; estimating a per-window scale only injects noise (5.25 -> 7.09 mm
      with a clamped robust estimator, -> 19.27 mm with a plain least-squares
      one, which degenerates on near-static overlaps).

  The sparse map is each window's confident depth unprojected into the global
  frame. No bundle adjustment and no loop closure.

DETERMINISM / THE 5 RUNS
  The pipeline is deterministic: DA3 runs under torch.no_grad in eval mode, the
  chaining is closed-form, and the only sampling (point-cloud subsampling) uses
  a fixed seed. Each sequence is therefore computed ONCE and written identically
  to runs 1..5, rather than recomputing the same numbers five times. The runtime
  reported in every run is the true single-run cost of that computation.

CONTRACT (reference/submission_instructions/README.md)
  /input  read-only, flat *.mp4        ->  /output/<seq>/<1..5>/
                                             3D_maps/000/points3D.txt
                                             camera_trajectory/cam_traj_map_000.txt
                                             runtime.txt
  Frame IDs are 1-based: the first video frame is ID 1 (000001.png).
  Trajectory poses are camera-to-world (T_wc): (tx,ty,tz) is the camera centre
  in world coordinates and (qw,qx,qy,qz) is w-first.
  Runs with --network=none: DA3 weights are baked into the image at build time
  (HF_HOME=/opt/weights, HF_HUB_OFFLINE=1).
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
import traceback
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import torch

from calib_table import CALIBS, SEQ2ENDO, MEAN_CALIB

# Inference knobs. Values are the ones validated in expB01 (5.25 mm ATE).
MODEL_ID = os.environ.get("DA3_MODEL", "depth-anything/DA3NESTED-GIANT-LARGE")
# Backend: "da3" (default, everything measured so far) or "pi3" (Pi3/Pi3X, see
# pi3_backend.py). Pi3X additionally accepts the true intrinsics, which DA3
# cannot use -- the reason DA3_RECT_FSCALE exists at all.
BACKEND = os.environ.get("DA3_BACKEND", "da3")
CHUNK = int(os.environ.get("DA3_CHUNK", 12))       # frames per DA3 call; peak 14.6GB @ res504
OVERLAP = int(os.environ.get("DA3_OVERLAP", 6))    # shared frames used to align windows
PROCESS_RES = int(os.environ.get("DA3_RES", 504))  # longest side fed to DA3
CONF_PCT = float(os.environ.get("DA3_CONF_PCT", 40.0))   # depth confidence percentile kept
STRIDE_PX = int(os.environ.get("DA3_STRIDE_PX", 8))      # pixel grid step for the map
POINT_EVERY = int(os.environ.get("DA3_POINT_EVERY", 4))  # map points from every Nth frame
MAX_POINTS = int(os.environ.get("DA3_MAX_POINTS", 200000))
NUM_RUNS = int(os.environ.get("NUM_RUNS", 5))
# auto | bf16 | fp16. DA3 picks bf16 whenever torch reports bf16 support, which
# on Turing (sm_75) means EMULATED bf16 -- measured 4.2x slower than fp16 there.
# The eval host is Blackwell, where bf16 is native, so "auto" is right for the
# submission; "fp16" exists as a safety valve if the host ever lacks fast bf16.
PRECISION = os.environ.get("DA3_PRECISION", "auto").lower()
# v006: rotation refinement (rotavg.py). RPE_rot uses only the rotations, ATE
# only the centres -- so refined rotations are spliced in with ATE untouched.
ROTAVG = int(os.environ.get("DA3_ROTAVG", 1))
ROT_SCALE = float(os.environ.get("DA3_ROT_SCALE", "0.5"))
ROT_CODEC = os.environ.get("DA3_ROT_CODEC", "raw")
ROT_STRIDE = max(1, int(os.environ.get("DA3_ROT_STRIDE", 6)))
# v002: undistort each decoded frame (kb4 fisheye -> pinhole) before DA3.
# Disable only for pinhole inputs (the local sim regression test).
RECTIFY = os.environ.get("DA3_RECTIFY", "1") != "0"
# v003: run DA3 on every STRIDE-th frame only and interpolate the rest. The
# per-frame runtime that the ranking metric charges is processing_seconds/N over
# ALL N frames, so this divides the dominant DA3 cost by STRIDE while still
# emitting a pose for every frame (TFR stays 100).
STRIDE = max(1, int(os.environ.get("DA3_STRIDE", 6)))

# Inter-window scale. The chain assumes DA3 depth is metric so consecutive
# windows share a scale (s=1). That holds on sim, but on the REAL clips the
# overlap-implied scale spreads 0.34..2.78 (|s-1| median 0.20) and its running
# product collapses to 0.014, i.e. raw per-window estimation drifts the map to
# nothing. DA3_SCALE selects how much of that estimate we trust:
#   fixed1  - ignore it (v001/v002/v003 behaviour)
#   damped  - use s**ALPHA, clamped to [1/CLAMP, CLAMP], per window
#   overlap - use it raw (ablation; expected to blow up)
SCALE_MODE = os.environ.get("DA3_SCALE", "fixed1")
SCALE_ALPHA = float(os.environ.get("DA3_SCALE_ALPHA", 0.5))
SCALE_CLAMP = float(os.environ.get("DA3_SCALE_CLAMP", 1.15))

# Debug hook: DA3_DUMP=<dir> writes one .npz per sequence with the per-window
# quantities the chaining is built from (local camera centres, which cameras
# overlapped, the estimated scale, its spread). Used offline in
# workspace/expB04_realcv to compute an oracle-scale upper bound against the
# real COLMAP GT. Unset in the submission -> zero cost.
DUMP_DIR = os.environ.get("DA3_DUMP", "")
RECT_BALANCE = float(os.environ.get("DA3_RECT_BALANCE", 0.0))
# Virtual-pinhole focal multiplier: 1.0 keeps the full ~111 deg cone (v001-v004),
# larger values crop toward DA3's ~71 deg comfort zone. See build_rectify_maps.
RECT_FSCALE = float(os.environ.get("DA3_RECT_FSCALE", 1.0))
# Separate FOV for the rotation-averaging path. Narrowing the cone helps DA3
# (its self-estimated focal finally matches reality) but HURTS LightGlue: with
# less shared field of view between frames Delta apart, surviving feature edges
# roughly halve (Seq_003_a 888 -> 404, Seq_001_c 139 -> 39), and rotavg quality
# tracks edge density almost exactly. The two consumers are independent, so give
# DA3 the narrow cone and rotavg the wide one. Costs one extra quarter-res remap.
ROT_FSCALE = float(os.environ.get("DA3_ROT_FSCALE", "0") or 0) or None



# ------------------------------------------------------------ rectification --
def calib_for_sequence(seq_name):
    """Exact per-endoscope kb4 when the seq number is recognisable, else mean.
    DA3_FORCE_ENDO overrides (local ablations)."""
    force = os.environ.get("DA3_FORCE_ENDO")
    if force and int(force) in CALIBS:
        return CALIBS[int(force)], f"Endoscope_{int(force):02d} (forced)"
    m = re.search(r"(\d+)", seq_name)
    if m:
        endo = SEQ2ENDO.get(int(m.group(1)))
        if endo in CALIBS:
            return CALIBS[endo], f"Endoscope_{endo:02d} (seq {int(m.group(1)):03d})"
    return MEAN_CALIB, "mean-of-18 fallback"


def build_rectify_maps(calib, w, h, balance=0.0, fscale=None):
    """kb4 -> pinhole remap for a w x h video (calibration is native 1440x1080;
    kb4 coefficients depend only on the ray angle, so they rescale freely).

    fscale narrows the virtual pinhole. balance=0 keeps the endoscope's full
    valid cone, which lands at ~111 deg horizontal FOV -- geometrically correct
    but far outside DA3's training distribution: it self-estimates a focal
    1.6-2.5x the true one (i.e. it believes ~71 deg). A wrong focal does not
    merely rescale the reconstruction, it shears rotation into translation, so
    it shows up in RPE_rot as well as in the map. Multiplying the focal by
    fscale zooms into a narrower cone -- fewer peripheral pixels, but geometry
    DA3 can actually read, at higher effective resolution per solid angle.
    """
    sx, sy = w / 1440.0, h / 1080.0
    K = np.array([[calib["fu"] * sx, 0, calib["u0"] * sx],
                  [0, calib["fv"] * sy, calib["v0"] * sy],
                  [0, 0, 1]], np.float64)
    D = np.array(calib["kb4"], np.float64).reshape(4, 1)
    P = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (w, h), np.eye(3), balance=balance)
    f = RECT_FSCALE if fscale is None else fscale
    if f != 1.0:
        P = P.copy()
        P[0, 0] *= f
        P[1, 1] *= f
    mapx, mapy = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), P, (w, h), cv2.CV_32FC1)
    return mapx, mapy, P


def border_black_mask(frame_bgr, thr=12):
    """Pixels that are dark AND connected to the image edge = endoscope border /
    out-of-FOV fill. Interior dark tissue (the lumen) does not touch the edge
    and is preserved."""
    dark = (cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) < thr).astype(np.uint8)
    h, w = dark.shape
    ff = dark.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    for seed in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1),
                 (w // 2, 0), (w // 2, h - 1), (0, h // 2), (w - 1, h // 2)]:
        if dark[seed[1], seed[0]]:
            cv2.floodFill(ff, mask, seed, 2)
    return ff == 2


def central_valid_crop(valid):
    """Exact maximum-area all-valid axis-aligned rectangle (largest rectangle
    in histogram, row by row). Isolated invalid speckles are closed first so a
    single bad pixel cannot pinch the rectangle. Returns (x0, x1, y0, y1)."""
    v = cv2.morphologyEx(valid.astype(np.uint8), cv2.MORPH_CLOSE,
                         np.ones((5, 5), np.uint8)).astype(bool)
    h, w = v.shape
    heights = np.zeros(w, np.int32)
    best = (0, (0, 0, 0, 0))
    for y in range(h):
        heights = np.where(v[y], heights + 1, 0)
        stack = []          # (start_index, height)
        for x in range(w + 1):
            cur = heights[x] if x < w else 0
            start = x
            while stack and stack[-1][1] >= cur:
                sx, sh = stack.pop()
                area = sh * (x - sx)
                if area > best[0]:
                    best = (area, (sx, x, y + 1 - sh, y + 1))
                start = sx
            if not stack or cur > 0:
                stack.append((start, cur))
    return best[1]


def content_crop_box(rectified_first_frame):
    """Crop box excluding the rectified border blacks. No-op-sized when the
    frame is fully valid."""
    valid = ~border_black_mask(rectified_first_frame)
    if valid.mean() > 0.995:
        return None
    x0, x1, y0, y1 = central_valid_crop(valid)
    h, w = valid.shape
    if (x1 - x0) * (y1 - y0) < 0.25 * w * h:   # degenerate detection; keep full
        return None
    return x0, x1, y0, y1


# --------------------------------------------------------------- geometry --
def rotmat_to_qvec_wxyz(m):
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s; x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s; z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s; x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s; z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s; x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s; z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def centers_rots(extr):
    """w2c [N,4,4] -> camera centres C_w [N,3] and orientations R_wc [N,3,3]."""
    R_cw = extr[:, :3, :3]
    t_cw = extr[:, :3, 3]
    R_wc = np.transpose(R_cw, (0, 2, 1))
    C = -np.einsum("nij,nj->ni", R_wc, t_cw)
    return C, R_wc


def proj_so3(M):
    U, _, Vt = np.linalg.svd(M)
    return U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt


def align_window(C_loc, R_loc, C_glob, R_glob, mode=None):
    """Rotation-first rigid local->global fit on the shared cameras.

    `mode` is DA3_SCALE (see module top): fixed1 / damped / overlap.
    """
    mode = SCALE_MODE if mode is None else mode
    M = sum(Rg @ Rl.T for Rg, Rl in zip(R_glob, R_loc))
    R_A = proj_so3(M)
    s = 1.0
    if mode != "fixed1":
        ratios = []
        for i, j in combinations(range(len(C_loc)), 2):
            a = np.linalg.norm(C_loc[i] - C_loc[j])
            b = np.linalg.norm(C_glob[i] - C_glob[j])
            if a > 1e-9 and b > 1e-9:
                ratios.append(b / a)
        s = float(np.median(ratios)) if ratios else 1.0
        if not (np.isfinite(s) and 0.25 < s < 4.0):
            s = 1.0
        if mode == "damped":
            s = s ** SCALE_ALPHA
            s = float(np.clip(s, 1.0 / SCALE_CLAMP, SCALE_CLAMP))
    t_A = C_glob.mean(0) - s * R_A @ C_loc.mean(0)
    return s, R_A, t_A


def overlap_scale_stats(C_loc, C_glob):
    """Raw pairwise-distance ratios on the shared cameras, plus the two signals
    a confidence gate could use: how much the shared cameras actually moved
    (a near-static overlap makes the ratio ill-conditioned) and how much the
    individual ratios disagree."""
    ratios, base_l, base_g = [], [], []
    for i, j in combinations(range(len(C_loc)), 2):
        a = np.linalg.norm(C_loc[i] - C_loc[j])
        b = np.linalg.norm(C_glob[i] - C_glob[j])
        base_l.append(a); base_g.append(b)
        if a > 1e-9 and b > 1e-9:
            ratios.append(b / a)
    if not ratios:
        return 1.0, 0.0, 0.0, 0.0
    r = np.asarray(ratios)
    med = float(np.median(r))
    spread = float(np.subtract(*np.percentile(r, [75, 25])) / max(med, 1e-9))
    return med, spread, float(np.mean(base_l)), float(np.mean(base_g))


def as_4x4(extr):
    """DA3 returns [N,3,4]; homogenise."""
    extr = np.asarray(extr, np.float64)
    if extr.shape[1] == 3:
        bot = np.tile([[[0.0, 0.0, 0.0, 1.0]]], (extr.shape[0], 1, 1))
        extr = np.concatenate([extr, bot], axis=1)
    return extr


# ---------------------------------------------------------------- writers --
def write_trajectory(path, frame_ids, extr, fps):
    """extr: global world-to-camera [N,4,4] -> CLiMB camera-to-world trajectory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    dt = 1.0 / fps if fps and fps > 0 else 1.0 / 30.0
    with open(path, "w") as f:
        f.write("# timestamp, name_image, tx, ty, tz, qw, qx, qy, qz\n")
        for i, fid in enumerate(frame_ids):
            R_cw = extr[i][:3, :3]
            t_cw = extr[i][:3, 3]
            R_wc = R_cw.T
            C_w = -R_wc @ t_cw
            qw, qx, qy, qz = rotmat_to_qvec_wxyz(R_wc)
            f.write(f"{i*dt:.6f},{fid:06d}.png,{C_w[0]:.9f},{C_w[1]:.9f},{C_w[2]:.9f},"
                    f"{qw:.9f},{qx:.9f},{qy:.9f},{qz:.9f}\n")


def write_points3d(path, pts, cols):
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(pts) > MAX_POINTS:
        idx = np.random.RandomState(0).choice(len(pts), MAX_POINTS, replace=False)
        pts, cols = pts[idx], cols[idx]
    with open(path, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR\n")
        for j, (p, c) in enumerate(zip(pts, cols)):
            f.write(f"{j+1} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                    f"{int(c[0])} {int(c[1])} {int(c[2])} 0.0\n")
    return len(pts)


def write_runs(out_root, seq, frame_ids, extr, pts, cols, fps, init_s, proc_s):
    """Write the identical deterministic result to runs 1..NUM_RUNS."""
    npts = 0
    for run in range(1, NUM_RUNS + 1):
        rd = out_root / seq / str(run)
        write_trajectory(rd / "camera_trajectory" / "cam_traj_map_000.txt",
                         frame_ids, extr, fps)
        npts = write_points3d(rd / "3D_maps" / "000" / "points3D.txt", pts, cols)
        rt = rd / "runtime.txt"
        rt.parent.mkdir(parents=True, exist_ok=True)
        rt.write_text(f"init_seconds={init_s:.6f}\nprocessing_seconds={proc_s:.6f}\n")
    return npts


# ------------------------------------------------------------ video input --
class VideoWindow:
    """Sequential mp4 reader that decodes only the KEYFRAMES, in order.

    Non-keyframes are advanced with grab() (no decode/colour-convert/rectify),
    which is what makes striding actually cheap on the CPU side: measured on a
    real 1440x1080 clip, read-all+rectify is 4.2 ms/frame while grab-skip at
    stride 4 is 2.0 ms/frame. Only the frames of the current DA3 window are
    held in memory.
    """

    def __init__(self, path, kf_indices, rect_maps=None, crop_box=None,
                 rot_maps=None, rot_crop_box=None):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open video: {path}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self.kf = list(kf_indices)          # absolute frame indices to decode
        self.rect_maps = rect_maps
        self.crop_box = crop_box
        # v007: low-res rectify maps for the rot cache (remap cost scales with
        # OUTPUT size -> ~16x cheaper than full-res remap + resize; CPU-bound
        # cost does not shrink on the eval GPU, so it must be cheap here).
        self.rot_maps, self.rot_crop = None, None
        rm = rot_maps if rot_maps is not None else rect_maps
        rb = rot_crop_box if rot_maps is not None else crop_box
        if rm is not None:
            self.rot_maps = (
                cv2.resize(rm[0], None, fx=ROT_SCALE, fy=ROT_SCALE,
                           interpolation=cv2.INTER_LINEAR),
                cv2.resize(rm[1], None, fx=ROT_SCALE, fy=ROT_SCALE,
                           interpolation=cv2.INTER_LINEAR))
        if rb is not None:
            x0, x1, y0, y1 = rb
            self.rot_crop = (int(x0 * ROT_SCALE), int(x1 * ROT_SCALE),
                             int(y0 * ROT_SCALE), int(y1 * ROT_SCALE))
        self.buf = {}                       # keyframe-list index -> RGB frame
        self.rot_cache = {}                 # frame idx -> half-res rectified jpg (rotavg)
        self.pos = 0                        # next absolute frame index in the stream
        self.kf_ptr = 0                     # next keyframe-list index to decode
        self.exhausted = False

    def count_frames(self, path):
        """Frame count from metadata, verified by decoding when it looks wrong."""
        n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n > 0:
            return n
        cap = cv2.VideoCapture(str(path))
        n = 0
        while cap.grab():
            n += 1
        cap.release()
        return n

    def ensure(self, upto):
        """Decode forward so that keyframe-list entries [.., upto) are buffered."""
        while self.kf_ptr < upto and not self.exhausted:
            target = self.kf[self.kf_ptr]
            while self.pos < target:
                if not self.cap.grab():
                    self.exhausted = True
                    break
                if ROTAVG and self.pos % ROT_STRIDE == 0:
                    okr, fr = self.cap.retrieve()
                    if okr:
                        self._rot_store(self.pos, fr, already_rect=False)
                self.pos += 1
            if self.exhausted:
                break
            ok, frame = self.cap.read()
            if not ok:
                self.exhausted = True
                break
            self.pos += 1
            if self.rect_maps is not None:
                frame = cv2.remap(frame, self.rect_maps[0], self.rect_maps[1],
                                  cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(0, 0, 0))
            if self.crop_box is not None:
                x0, x1, y0, y1 = self.crop_box
                frame = frame[y0:y1, x0:x1]
            self.buf[self.kf_ptr] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if ROTAVG and (self.pos - 1) % ROT_STRIDE == 0:
                self._rot_store(self.pos - 1, frame, already_rect=True)
            self.kf_ptr += 1

    def _rot_store(self, idx, frame_bgr, already_rect):
        """Low-res rectified BGR frame for rotavg (raw ndarray by default)."""
        if already_rect:
            # frame is already full-res rectified+cropped; resize to the exact
            # dims of the low-res-remap path so K matches across all nodes.
            if self.rot_crop is not None:
                x0, x1, y0, y1 = self.rot_crop
                small = cv2.resize(frame_bgr, (x1 - x0, y1 - y0),
                                   interpolation=cv2.INTER_AREA)
            else:
                small = cv2.resize(frame_bgr, None, fx=ROT_SCALE, fy=ROT_SCALE,
                                   interpolation=cv2.INTER_AREA)
        else:
            f = frame_bgr
            if self.rot_maps is not None:
                f = cv2.remap(f, self.rot_maps[0], self.rot_maps[1],
                              cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(0, 0, 0))
                if self.rot_crop is not None:
                    x0, x1, y0, y1 = self.rot_crop
                    f = f[y0:y1, x0:x1]
            else:
                f = cv2.resize(f, None, fx=ROT_SCALE, fy=ROT_SCALE,
                               interpolation=cv2.INTER_AREA)
            small = f
        w = getattr(self, "rot_worker", None)
        if w is not None:
            w.add(idx, small)
        elif ROT_CODEC == "raw":
            self.rot_cache[idx] = small
        elif ROT_CODEC == "png":
            ok, enc = cv2.imencode(".png", small, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            if ok:
                self.rot_cache[idx] = enc.tobytes()
        else:
            ok, enc = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if ok:
                self.rot_cache[idx] = enc.tobytes()

    def window(self, a, b):
        """Frames for keyframe-list indices [a, b)."""
        self.ensure(b)
        for k in list(self.buf):
            if k < a:
                del self.buf[k]
        return [self.buf[k] for k in range(a, min(b, self.kf_ptr)) if k in self.buf]

    def release(self):
        self.cap.release()
        self.buf.clear()


def interpolate_poses(kf_idx, kf_w2c, n):
    """Keyframe world-to-camera poses -> a pose for every frame in [0, n).

    Interpolation is done on the camera CENTRE (linear) and the camera
    ORIENTATION (SLERP), i.e. on the quantities the metrics are computed from,
    never on the raw w2c translation (which mixes rotation into the position).
    Frames outside the keyframe span are clamped to the end keyframes.
    """
    from scipy.spatial.transform import Rotation, Slerp

    kf_idx = np.asarray(kf_idx, float)
    R_wc = np.stack([E[:3, :3].T for E in kf_w2c])
    C = np.stack([-(E[:3, :3].T @ E[:3, 3]) for E in kf_w2c])

    q = np.clip(np.arange(n, dtype=float), kf_idx[0], kf_idx[-1])
    if len(kf_idx) >= 2:
        R_all = Slerp(kf_idx, Rotation.from_matrix(R_wc))(q).as_matrix()
        C_all = np.stack([np.interp(q, kf_idx, C[:, d]) for d in range(3)], 1)
    else:
        R_all = np.tile(R_wc[0], (n, 1, 1))
        C_all = np.tile(C[0], (n, 1))

    out = np.tile(np.eye(4), (n, 1, 1))
    out[:, :3, :3] = np.transpose(R_all, (0, 2, 1))          # R_cw = R_wc^T
    out[:, :3, 3] = -np.einsum("nij,nj->ni", out[:, :3, :3], C_all)
    return out


# --------------------------------------------------------------- pipeline --
def process_sequence(model, video_path, log):
    """Return (frame_ids, global w2c [N,4,4], points, colours, fps, proc_seconds)."""
    rect_maps, crop_box, rect_K = None, None, None
    if RECTIFY:
        probe = cv2.VideoCapture(str(video_path))
        ok, first = probe.read()
        probe.release()
        if not ok:
            raise RuntimeError(f"cannot decode first frame: {video_path}")
        h, w = first.shape[:2]
        calib, which = calib_for_sequence(Path(video_path).stem)
        rect_maps = build_rectify_maps(calib, w, h, RECT_BALANCE)
        rect_first = cv2.remap(first, rect_maps[0], rect_maps[1], cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        crop_box = content_crop_box(rect_first)
        rect_K = rect_maps[2].copy()
        if crop_box is not None:
            rect_K[0, 2] -= crop_box[0]
            rect_K[1, 2] -= crop_box[2]
        log(f"  rectify kb4->pinhole @ {w}x{h}, calib: {which}, "
            f"crop: {crop_box if crop_box else 'none (fully valid)'}")
    rot_maps, rot_crop_box = None, None
    if ROT_FSCALE is not None and rect_maps is not None and ROT_FSCALE != RECT_FSCALE:
        rot_maps = build_rectify_maps(calib, w, h, RECT_BALANCE, fscale=ROT_FSCALE)
        rot_crop_box = content_crop_box(
            cv2.remap(first, rot_maps[0], rot_maps[1], cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)))
        log(f"  rotavg FOV: fscale {ROT_FSCALE} (DA3 uses {RECT_FSCALE}), "
            f"crop: {rot_crop_box if rot_crop_box else 'none'}")
    rot_worker = None
    if ROTAVG and rect_maps is not None:
        try:
            import rotavg as _rotavg
            src = rot_maps if rot_maps is not None else rect_maps
            cb = rot_crop_box if rot_maps is not None else crop_box
            K_small = src[2].copy(); K_small[:2] *= ROT_SCALE
            if cb is not None:
                K_small[0, 2] -= int(cb[0] * ROT_SCALE)
                K_small[1, 2] -= int(cb[2] * ROT_SCALE)
            rot_worker = _rotavg.RotWorker(K_small, log=log)
            rot_worker.start()
        except Exception as exc:
            log(f"  rotavg worker init failed ({type(exc).__name__}: {exc}); disabled")
            rot_worker = None
    probe = VideoWindow(video_path, [], rect_maps, crop_box)
    n = probe.count_frames(video_path)
    fps = probe.fps
    probe.release()
    if n < 2:
        raise RuntimeError(f"too few frames ({n})")

    # Keyframes: every STRIDE-th frame, always including the last one so the
    # interpolation spans the whole clip instead of clamping a ragged tail.
    kf = list(range(0, n, STRIDE))
    if kf[-1] != n - 1:
        kf.append(n - 1)
    n_kf = len(kf)
    log(f"  {n} frames @ {fps:.2f} fps -> {n_kf} keyframes (stride {STRIDE})")

    vid = VideoWindow(video_path, kf, rect_maps, crop_box, rot_maps, rot_crop_box)
    vid.rot_worker = rot_worker

    step = max(CHUNK - OVERLAP, 1)
    starts = list(range(0, max(n_kf - OVERLAP, 1), step))
    if starts and starts[-1] + CHUNK < n_kf:
        starts.append(n_kf - CHUNK)

    glob_w2c = [None] * n_kf
    pts_all, cols_all = [], []
    n_failed = 0
    dump = []
    t0 = time.perf_counter()

    for ci, s0 in enumerate(starts):
        s1 = min(s0 + CHUNK, n_kf)
        frames = vid.window(s0, s1)
        if len(frames) < 2:
            break
        s1 = s0 + len(frames)          # video may be shorter than metadata claimed

        try:
            with torch.no_grad():
                if BACKEND == "pi3":
                    pred = model.inference(frames, K=rect_K if rect_maps is not None else None)
                else:
                    pred = model.inference(frames, process_res=PROCESS_RES,
                                           export_format="mini_npz")
            extr = as_4x4(pred.extrinsics)
            depth = np.asarray(pred.depth, np.float32)
            conf = np.asarray(pred.conf, np.float32) if pred.conf is not None else None
            intr = np.asarray(pred.intrinsics, np.float64)
            imgs = (np.asarray(pred.processed_images)
                    if pred.processed_images is not None else None)
        except Exception as exc:                     # keep the sequence alive
            n_failed += 1
            log(f"    chunk {ci+1}/{len(starts)} [{s0}:{s1}] FAILED ({type(exc).__name__}); "
                "holding last pose")
            for k in range(s0, s1):
                if glob_w2c[k] is None:
                    prev = next((glob_w2c[j] for j in range(k - 1, -1, -1)
                                 if glob_w2c[j] is not None), np.eye(4))
                    glob_w2c[k] = prev.copy()
            torch.cuda.empty_cache()
            continue

        C_l, R_l = centers_rots(extr)
        shared = [k for k in range(s0, s1) if glob_w2c[k] is not None]
        if not shared:
            sA, RA, tA = 1.0, np.eye(3), np.zeros(3)
            if DUMP_DIR:
                dump.append(dict(s0=s0, s1=s1, shared=np.zeros(0, int),
                                 C_loc=C_l.copy(), R_loc=R_l.copy(), s_used=1.0, s_ovl=1.0,
                                 spread=0.0, base_loc=0.0, base_glob=0.0,
                                 focal=float(np.mean(intr[:, 0, 0]))))
        else:
            Cg, Rg = centers_rots(np.stack([glob_w2c[k] for k in shared]))
            loc = [k - s0 for k in shared]
            sA, RA, tA = align_window(C_l[loc], R_l[loc], Cg, Rg)
            if DUMP_DIR:
                med, spread, bl, bg = overlap_scale_stats(C_l[loc], Cg)
                dump.append(dict(s0=s0, s1=s1, shared=np.asarray(shared),
                                 C_loc=C_l.copy(), R_loc=R_l.copy(), s_used=sA,
                                 s_ovl=med, spread=spread, base_loc=bl, base_glob=bg,
                                 focal=float(np.mean(intr[:, 0, 0]))))

        for k in range(s0, s1):
            if glob_w2c[k] is not None:
                continue
            j = k - s0
            C_g = sA * RA @ C_l[j] + tA
            R_wc_g = RA @ R_l[j]
            E = np.eye(4)
            E[:3, :3] = R_wc_g.T
            E[:3, 3] = -R_wc_g.T @ C_g
            glob_w2c[k] = E

        H, W = depth.shape[1:]
        thr = np.percentile(conf, CONF_PCT) if conf is not None else -np.inf
        if DUMP_DIR and dump and dump[-1]["s0"] == s0:
            # Confidence-weighted median depth per frame, in this window's local
            # units. Two windows that share a frame observe the same physical
            # scene, so the ratio of their medians is a scale estimate drawn
            # from ~1e5 pixels instead of 3 camera centres.
            md = []
            for j in range(depth.shape[0]):
                dj, cj = depth[j], (conf[j] if conf is not None else None)
                mm = np.isfinite(dj) & (dj > 0)
                if cj is not None:
                    mm &= cj >= thr
                md.append(float(np.median(dj[mm])) if mm.sum() > 100 else np.nan)
            dump[-1]["med_depth"] = np.asarray(md)
        ys, xs = np.arange(0, H, STRIDE_PX), np.arange(0, W, STRIDE_PX)
        gx, gy = np.meshgrid(xs, ys)
        gx, gy = gx.ravel(), gy.ravel()
        for j in range(0 if ci == 0 else OVERLAP, s1 - s0, POINT_EVERY):
            d = depth[j][gy, gx]
            c = conf[j][gy, gx] if conf is not None else np.ones_like(d)
            m = (c >= thr) & np.isfinite(d) & (d > 0)
            if not m.any():
                continue
            uu, vv, dd = gx[m], gy[m], d[m]
            rays = np.linalg.inv(intr[j]) @ np.stack([uu, vv, np.ones_like(uu)], 0)
            world_l = (R_l[j] @ (rays * dd[None, :])).T + C_l[j]
            pts_all.append(sA * (RA @ world_l.T).T + tA)
            cols_all.append(imgs[j][vv, uu] if imgs is not None
                            else np.full((int(m.sum()), 3), 128))
        if (ci + 1) % 20 == 0 or ci + 1 == len(starts):
            log(f"    chunk {ci+1}/{len(starts)} [{s0}:{s1}]")

    solved = [j for j in range(n_kf) if glob_w2c[j] is not None]
    if len(solved) < 2:
        raise RuntimeError("no poses produced")
    # Interpolate the keyframe poses back onto every frame (TFR stays 100).
    extr_g = interpolate_poses([kf[j] for j in solved],
                               [glob_w2c[j] for j in solved], n)
    if rot_worker is not None:
        try:
            import rotavg as _rotavg
            extr_g = _rotavg.finalize(rot_worker, extr_g, log=log)
        except Exception as exc:
            log(f"  rotavg failed ({type(exc).__name__}: {exc}); keeping chain rotations")
    elif ROTAVG:
        log("  rotavg: skipped (no rectified K)")
    proc = time.perf_counter() - t0
    vid.release()

    frame_ids = [k + 1 for k in range(n)]          # 1-based CLiMB frame IDs
    pts = np.concatenate(pts_all, 0) if pts_all else np.zeros((0, 3))
    cols = np.concatenate(cols_all, 0) if cols_all else np.zeros((0, 3))
    if n_failed:
        log(f"  WARNING: {n_failed}/{len(starts)} chunks failed")
    if DUMP_DIR:
        d = Path(DUMP_DIR); d.mkdir(parents=True, exist_ok=True)
        np.savez(d / f"{video_path.stem}.npz", kf=np.asarray(kf), n=n,
                 windows=np.asarray(dump, dtype=object), allow_pickle=True)
        log(f"  dumped {len(dump)} windows -> {d / (video_path.stem + '.npz')}")
    return frame_ids, extr_g, pts, cols, fps, proc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/input")
    ap.add_argument("--output", default="/output")
    args = ap.parse_args()

    def log(msg):
        print(msg, flush=True)

    in_dir, out_dir = Path(args.input), Path(args.output)
    videos = sorted(p for p in in_dir.iterdir()
                    if p.is_file() and p.suffix.lower() == ".mp4")
    if not videos:
        log(f"ERROR: no .mp4 files under {in_dir}")
        return 1
    log(f"CLiMB DA3-submap submission: {len(videos)} sequence(s), "
        f"chunk={CHUNK} overlap={OVERLAP} res={PROCESS_RES} model={MODEL_ID}")

    t_init0 = time.perf_counter()
    if PRECISION in ("fp16", "bf16"):
        # DA3 selects its autocast dtype via torch.cuda.is_bf16_supported().
        torch.cuda.is_bf16_supported = (lambda *a, **k: PRECISION == "bf16")
        log(f"forcing {PRECISION} autocast")
    from depth_anything_3.api import DepthAnything3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        log("WARNING: no CUDA device visible; falling back to CPU (very slow)")
    if BACKEND == "pi3":
        sys.path.insert(0, os.environ.get("PI3_SRC", "/data4/src/shunsuke/Pi3"))
        from pi3_backend import Pi3Backend
        model = Pi3Backend(log=log)
    else:
        model = DepthAnything3.from_pretrained(MODEL_ID).to(device).eval()
    # v004: LoRA adapter (workspace/expB09_lora) -- self-supervised cross-window
    # consistency fine-tune of the any-view backbone on TRAINVAL colonoscopy.
    # Baked into the image as /opt/submission/lora.pt; DA3_LORA="" disables it.
    lora_path = os.environ.get("DA3_LORA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "lora.pt"))
    if lora_path and os.path.isfile(lora_path):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from lora import inject_lora
        inject_lora(model.model.da3.backbone.pretrained,
                    r=int(os.environ.get("DA3_LORA_R", 8)), alpha=int(os.environ.get("DA3_LORA_ALPHA", 16)))
        sd = torch.load(lora_path, map_location=device)
        _, unexpected = model.model.load_state_dict(sd, strict=False)
        log(f"LoRA loaded: {lora_path} ({len(sd)} tensors, unexpected={len(unexpected)})")
    else:
        log("LoRA: none")
    init_s = time.perf_counter() - t_init0
    log(f"model loaded in {init_s:.1f}s on {device}")

    n_ok, t_all = 0, time.perf_counter()
    for vp in videos:
        seq = vp.stem
        log(f"[{seq}]")
        try:
            ids, extr, pts, cols, fps, proc = process_sequence(model, vp, log)
            npts = write_runs(out_dir, seq, ids, extr, pts, cols, fps, init_s, proc)
            log(f"  {len(ids)} poses, {npts} points, {proc:.1f}s "
                f"({proc/max(len(ids),1):.3f} s/frame) -> {NUM_RUNS} runs")
            n_ok += 1
        except Exception:
            log(f"  ERROR processing {seq}:\n{traceback.format_exc()}")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    log(f"Done: {n_ok}/{len(videos)} sequences in {time.perf_counter()-t_all:.1f}s")
    return 0 if n_ok else 1


if __name__ == "__main__":
    sys.exit(main())
