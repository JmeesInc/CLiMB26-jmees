#!/usr/bin/env python3
"""DA3 submap-chunked feedforward SLAM for CLiMB sim sequences.

expB00 showed full-batch DA3 is memory-bound (all-to-all cross-view attention,
math-kernel O((S*N)^2) in this env) and only runs at res182 -> ATE 35mm.
This experiment keeps DA3 but feeds CHUNK frames per call (with OVERLAP shared
frames), then chains the chunk-local w2c extrinsics into one global map via
rotation-first Sim(3) alignment on the overlapping cameras (VGGT-SLAM-style
submapping with a DA3 backbone, no loop closure).

Env: iMED venv (/data4/src/shunsuke/MICCAI2026/iMED/.venv) - has DA3 + torch cu126.
Input frames: expC00 sim_input/<seq>/{frame_id:06d}.png (1-based CLiMB ids).
Output: CLiMB run tree  <out>/<seq>/<run>/camera_trajectory/cam_traj_map_000.txt
        + 3D_maps/000/points3D.txt + runtime.txt   (scored vs expA00 colmap_gt).
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch

# DA3 picks bf16 whenever torch reports support; on Turing (sm_75) that is
# EMULATED bf16, measured 4.2x slower than fp16 with no ATE change (see
# submit/v001_da3_submap/predict.py). DA3_PRECISION=fp16 forces the fast path.
_PREC = os.environ.get("DA3_PRECISION", "auto").lower()
if _PREC in ("fp16", "bf16"):
    torch.cuda.is_bf16_supported = (lambda *a, **k: _PREC == "bf16")


# ----------------------------------------------------------------- helpers --
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
    """w2c [N,4,4] -> camera centers C_w [N,3] and R_wc [N,3,3]."""
    R_cw = extr[:, :3, :3]
    t_cw = extr[:, :3, 3]
    R_wc = np.transpose(R_cw, (0, 2, 1))
    C = -np.einsum("nij,nj->ni", R_wc, t_cw)
    return C, R_wc


def proj_so3(M):
    U, _, Vt = np.linalg.svd(M)
    R = U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    return R


def sim3_rotation_first(C_loc, R_loc, C_glob, R_glob, s_lo=0.25, s_hi=4.0):
    """Sim(3) local->global from overlapping cameras.

    Rotation from the camera orientations (well-conditioned even when the
    centers are near-collinear, which forward colonoscopy motion often is).
    Scale from the median ratio of pairwise center distances, clamped to
    [0.25, 4] with fallback 1.0: DA3 depth is metric so the true inter-chunk
    scale is ~1, and the tiny near-static overlap baselines otherwise make a
    least-squares scale blow up (observed s=9 on Seq_0 before this guard).
    """
    from itertools import combinations
    M = sum(Rg @ Rl.T for Rg, Rl in zip(R_glob, R_loc))
    R_A = proj_so3(M)
    ratios = []
    for i, j in combinations(range(len(C_loc)), 2):
        a = np.linalg.norm(C_loc[i] - C_loc[j])
        b = np.linalg.norm(C_glob[i] - C_glob[j])
        if a > 1e-9 and b > 1e-9:
            ratios.append(b / a)
    s = float(np.median(ratios)) if ratios else 1.0
    if not (np.isfinite(s) and s_lo < s < s_hi):
        s = 1.0
    ml, mg = C_loc.mean(0), C_glob.mean(0)
    t_A = mg - s * R_A @ ml
    return s, R_A, t_A


def apply_sim3(s, R_A, t_A, pts):
    return s * (R_A @ pts.T).T + t_A


# ------------------------------------------------------------------ writers --
def write_trajectory(path, frame_ids, extr):
    """extr: global w2c [N,4,4] -> CLiMB c2w trajectory file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("# timestamp, name_image, tx, ty, tz, qw, qx, qy, qz\n")
        for i, fid in enumerate(frame_ids):
            R_cw = extr[i][:3, :3]; t_cw = extr[i][:3, 3]
            R_wc = R_cw.T
            C_w = -R_wc @ t_cw
            qw, qx, qy, qz = rotmat_to_qvec_wxyz(R_wc)
            f.write(f"{i/30.0:.6f},{fid:06d}.png,{C_w[0]:.9f},{C_w[1]:.9f},{C_w[2]:.9f},"
                    f"{qw:.9f},{qx:.9f},{qy:.9f},{qz:.9f}\n")


def write_points3d(path, pts, cols, max_points):
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(pts) > max_points:
        idx = np.random.RandomState(0).choice(len(pts), max_points, replace=False)
        pts, cols = pts[idx], cols[idx]
    with open(path, "w") as f:
        f.write("# POINT3D_ID X Y Z R G B ERROR\n")
        for j, (p, c) in enumerate(zip(pts, cols)):
            f.write(f"{j+1} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                    f"{int(c[0])} {int(c[1])} {int(c[2])} 0.0\n")
    return len(pts)


# --------------------------------------------------------------------- main --
def run_seq(model, frames, args):
    """frames: list of (frame_id, path). Returns global w2c [N,4,4], pts, cols, sec."""
    n = len(frames)
    step = args.chunk - args.overlap
    starts = list(range(0, max(n - args.overlap, 1), step))
    # make sure the tail is covered
    if starts[-1] + args.chunk < n:
        starts.append(n - args.chunk)

    glob_w2c = [None] * n
    pts_all, col_all = [], []
    t0 = time.perf_counter()
    for ci, s0 in enumerate(starts):
        s1 = min(s0 + args.chunk, n)
        paths = [str(p) for _, p in frames[s0:s1]]
        with torch.no_grad():
            pred = model.inference(paths, process_res=args.process_res,
                                   export_format="mini_npz")
        extr = np.asarray(pred.extrinsics, np.float64)     # [K,4,4] w2c, chunk-local
        depth = np.asarray(pred.depth, np.float32)
        conf = np.asarray(pred.conf, np.float32) if pred.conf is not None else None
        intr = np.asarray(pred.intrinsics, np.float64)
        imgs = np.asarray(pred.processed_images) if pred.processed_images is not None else None

        C_l, R_l = centers_rots(extr)
        if ci == 0:
            sA, RA, tA = 1.0, np.eye(3), np.zeros(3)
        else:
            ov = [k for k in range(s0, s1) if glob_w2c[k] is not None]
            Cg, Rg = centers_rots(np.stack([glob_w2c[k] for k in ov]))
            loc_idx = [k - s0 for k in ov]
            sA, RA, tA = sim3_rotation_first(C_l[loc_idx], R_l[loc_idx], Cg, Rg,
                                             s_lo=args.s_lo, s_hi=args.s_hi)

        # globalize chunk poses (keep already-set overlap poses from earlier chunks)
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

        # sparse map: subsample new frames of this chunk, unproject, lift to global
        thr = np.percentile(conf, args.conf_pct) if conf is not None else -np.inf
        H, W = depth.shape[1:]
        ys, xs = np.arange(0, H, args.stride_px), np.arange(0, W, args.stride_px)
        gx, gy = np.meshgrid(xs, ys)
        gx, gy = gx.ravel(), gy.ravel()
        for j in range(0 if ci == 0 else args.overlap, s1 - s0, args.point_every):
            d = depth[j][gy, gx]
            c = conf[j][gy, gx] if conf is not None else np.ones_like(d)
            m = (c >= thr) & np.isfinite(d) & (d > 0)
            if not m.any():
                continue
            uu, vv, dd = gx[m], gy[m], d[m]
            rays = np.linalg.inv(intr[j]) @ np.stack([uu, vv, np.ones_like(uu)], 0)
            cam = rays * dd[None, :]
            world_l = (R_l[j] @ cam).T + C_l[j]
            pts_all.append(apply_sim3(sA, RA, tA, world_l))
            col_all.append(imgs[j][vv, uu] if imgs is not None
                           else np.full((int(m.sum()), 3), 128))
        print(f"    chunk {ci+1}/{len(starts)} [{s0}:{s1}] scale={sA:.4f}", flush=True)

    sec = time.perf_counter() - t0
    pts = np.concatenate(pts_all, 0) if pts_all else np.zeros((0, 3))
    cols = np.concatenate(col_all, 0) if col_all else np.zeros((0, 3))
    return np.stack(glob_w2c), pts, cols, sec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_root", required=True,
                    help="dir with <seq>/{frame_id:06d}.png (expC00 sim_input)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seqs", nargs="*", default=None)
    ap.add_argument("--model", default="depth-anything/DA3NESTED-GIANT-LARGE")
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--overlap", type=int, default=6)
    ap.add_argument("--process_res", type=int, default=504)
    ap.add_argument("--s_lo", type=float, default=0.25,
                    help="scale clamp; s_lo>=s_hi forces s=1 (pure SE3, trust metric depth)")
    ap.add_argument("--s_hi", type=float, default=4.0)
    ap.add_argument("--conf_pct", type=float, default=40.0)
    ap.add_argument("--stride_px", type=int, default=8)
    ap.add_argument("--point_every", type=int, default=4)
    ap.add_argument("--max_points", type=int, default=200000)
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args()

    from depth_anything_3.api import DepthAnything3
    t_init0 = time.perf_counter()
    model = DepthAnything3.from_pretrained(args.model).to("cuda").eval()
    t_init = time.perf_counter() - t_init0
    print(f"model {args.model} loaded in {t_init:.1f}s", flush=True)

    root, out = Path(args.frames_root), Path(args.out)
    seq_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if args.seqs:
        seq_dirs = [d for d in seq_dirs if d.name in args.seqs]

    for d in seq_dirs:
        frames = sorted((int(p.stem), p) for p in d.glob("*.png"))
        print(f"{d.name}: {len(frames)} frames, chunk={args.chunk} ov={args.overlap}",
              flush=True)
        extr, pts, cols, sec = run_seq(model, frames, args)
        ids = [fid for fid, _ in frames]
        for run in range(1, args.runs + 1):
            rd = out / d.name / str(run)
            write_trajectory(rd / "camera_trajectory" / "cam_traj_map_000.txt", ids, extr)
            npts = write_points3d(rd / "3D_maps" / "000" / "points3D.txt",
                                  pts, cols, args.max_points)
            rt = rd / "runtime.txt"
            rt.parent.mkdir(parents=True, exist_ok=True)
            rt.write_text(f"init_seconds={t_init:.6f}\n"
                          f"processing_seconds={sec:.6f}\n")
        print(f"  {len(ids)} poses, {npts} pts, {sec:.1f}s ({sec/len(ids):.2f} s/frame)",
              flush=True)
    print("Done.")


if __name__ == "__main__":
    main()
