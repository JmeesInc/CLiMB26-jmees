#!/usr/bin/env python3
"""CLiMB submission — DA3 submap-chunked monocular SLAM.

Source: workspace/expB01_da3_submap (chunk_slam.py), sim 6-seq local CV with the
official evaluator against workspace/expA00_baseline_eval/colmap_gt:

  ORB-SLAM3 baseline (reference/ORBSLAM)          6.63 mm ATE / 72.6 TFR / 5-6 success
  VGGT-SLAM 2.0 (min_disparity=2)                11.73      / 78.4     / 6-6
  DA3 submap, per-chunk Sim(3) scale estimate     7.09      / 96.3     / 6-6
  DA3 submap, scale FIXED to 1 (this submission)  5.25      / 96.3     / 6-6

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
import sys
import time
import traceback
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import torch

# Inference knobs. Values are the ones validated in expB01 (5.25 mm ATE).
MODEL_ID = os.environ.get("DA3_MODEL", "depth-anything/DA3NESTED-GIANT-LARGE")
CHUNK = int(os.environ.get("DA3_CHUNK", 8))        # frames per DA3 call; 16 OOMs at res504
OVERLAP = int(os.environ.get("DA3_OVERLAP", 3))    # shared frames used to align windows
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


def align_window(C_loc, R_loc, C_glob, R_glob, estimate_scale=False):
    """Rigid (scale-1) local->global fit on the shared cameras.

    estimate_scale is kept only so the ablation stays runnable; the submission
    uses scale 1 because DA3 depth is metric (see module docstring).
    """
    M = sum(Rg @ Rl.T for Rg, Rl in zip(R_glob, R_loc))
    R_A = proj_so3(M)
    s = 1.0
    if estimate_scale:
        ratios = []
        for i, j in combinations(range(len(C_loc)), 2):
            a = np.linalg.norm(C_loc[i] - C_loc[j])
            b = np.linalg.norm(C_glob[i] - C_glob[j])
            if a > 1e-9 and b > 1e-9:
                ratios.append(b / a)
        s = float(np.median(ratios)) if ratios else 1.0
        if not (np.isfinite(s) and 0.25 < s < 4.0):
            s = 1.0
    t_A = C_glob.mean(0) - s * R_A @ C_loc.mean(0)
    return s, R_A, t_A


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
    """Sequential mp4 reader exposing a sliding window of decoded RGB frames.

    Frames are decoded once, in order, and only the frames of the current window
    are held in memory (a full 1440x1080 sequence would otherwise need tens of
    GB). DA3 wants HxWxC uint8 RGB; OpenCV decodes BGR.
    """

    def __init__(self, path):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open video: {path}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self.buf = {}
        self.next_read = 0
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
        """Decode forward so that frames [.., upto) are buffered."""
        while self.next_read < upto and not self.exhausted:
            ok, frame = self.cap.read()
            if not ok:
                self.exhausted = True
                break
            self.buf[self.next_read] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.next_read += 1

    def window(self, s0, s1):
        self.ensure(s1)
        for k in list(self.buf):
            if k < s0:
                del self.buf[k]
        return [self.buf[k] for k in range(s0, min(s1, self.next_read)) if k in self.buf]

    def release(self):
        self.cap.release()
        self.buf.clear()


# --------------------------------------------------------------- pipeline --
def process_sequence(model, video_path, log):
    """Return (frame_ids, global w2c [N,4,4], points, colours, fps, proc_seconds)."""
    vid = VideoWindow(video_path)
    n = vid.count_frames(video_path)
    log(f"  {n} frames @ {vid.fps:.2f} fps")
    if n < 2:
        raise RuntimeError(f"too few frames ({n})")

    step = max(CHUNK - OVERLAP, 1)
    starts = list(range(0, max(n - OVERLAP, 1), step))
    if starts and starts[-1] + CHUNK < n:
        starts.append(n - CHUNK)

    glob_w2c = [None] * n
    pts_all, cols_all = [], []
    n_failed = 0
    t0 = time.perf_counter()

    for ci, s0 in enumerate(starts):
        s1 = min(s0 + CHUNK, n)
        frames = vid.window(s0, s1)
        if len(frames) < 2:
            break
        s1 = s0 + len(frames)          # video may be shorter than metadata claimed

        try:
            with torch.no_grad():
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
        else:
            Cg, Rg = centers_rots(np.stack([glob_w2c[k] for k in shared]))
            loc = [k - s0 for k in shared]
            sA, RA, tA = align_window(C_l[loc], R_l[loc], Cg, Rg)

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

    proc = time.perf_counter() - t0
    vid.release()

    keep = [k for k in range(n) if glob_w2c[k] is not None]
    if len(keep) < 2:
        raise RuntimeError("no poses produced")
    frame_ids = [k + 1 for k in keep]              # 1-based CLiMB frame IDs
    extr_g = np.stack([glob_w2c[k] for k in keep])
    pts = np.concatenate(pts_all, 0) if pts_all else np.zeros((0, 3))
    cols = np.concatenate(cols_all, 0) if cols_all else np.zeros((0, 3))
    if n_failed:
        log(f"  WARNING: {n_failed}/{len(starts)} chunks failed")
    return frame_ids, extr_g, pts, cols, vid.fps, proc


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
    model = DepthAnything3.from_pretrained(MODEL_ID).to(device).eval()
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
