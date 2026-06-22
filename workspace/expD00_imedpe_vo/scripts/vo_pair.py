#!/usr/bin/env python3
"""iMED-PE tri3d method ported to monocular CLiMB VO.

Source method: ../iMED submit/v004_pe_tri3d (mean ATE 0.953mm on iMED PE):
stereo triangulation -> cross-camera ALIKED+LightGlue matches -> 3D-3D
IRLS(Tukey) Umeyama -> grid BA. CLiMB is monocular, so the stereo triangulation
is replaced by DA3 two-view metric depth: one DA3 call per consecutive frame
pair (t,t+1) yields pair-consistent depth + intrinsics; ALIKED+LightGlue
matches (Matcher imported from the iMED repo) are unprojected through both
depths into 3D-3D correspondences, and the tri3d IRLS rigid fit gives the
relative pose, chained into the trajectory.

The same DA3 call also outputs its own two-view extrinsics; chaining those is
kept as a free ablation baseline ("da3pair" tree) - it is expB01 with chunk=2.

Env: iMED venv (/data4/src/shunsuke/MICCAI2026/iMED/.venv) - DA3 + lightglue.
Input frames: expC00 sim_input/<seq>/{frame_id:06d}.png (1-based CLiMB ids).
Output: CLiMB run trees under <out_lg> (matcher VO) and <out_da3> (ablation).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

IMED = "/data4/src/shunsuke/MICCAI2026/iMED/submit/v004_pe_tri3d"
sys.path.insert(0, IMED)
from predict import Matcher  # noqa: E402  (ALIKED+LightGlue front-end, as in tri3d)


# ------------------------------------------------------- rigid fit (ported) --
def rigid_fit(src, dst, fit="se3", n_iter=5):
    """IRLS(Tukey) rigid/similarity fit src->dst, ported from tri3d umeyama_3d.

    fit="se3" fixes scale=1 (DA3 pair depth is jointly predicted, so the two
    frames of one call share a metric scale); "sim3" also estimates scale.
    Returns (s, R, t) or None.
    """
    src = np.asarray(src, float); dst = np.asarray(dst, float)
    n = len(src)
    if n < 4:
        return None
    w = np.ones(n)

    def solve(w):
        wsum = w.sum()
        ms, md = (w[:, None] * src).sum(0) / wsum, (w[:, None] * dst).sum(0) / wsum
        sc, dc = src - ms, dst - md
        cov = (w[:, None] * dc).T @ sc / wsum
        U, D, Vt = np.linalg.svd(cov)
        S = np.diag([1.0, 1.0, np.sign(np.linalg.det(U) * np.linalg.det(Vt))])
        R = U @ S @ Vt
        if fit == "sim3":
            var = (w * (sc ** 2).sum(1)).sum() / wsum
            s = np.trace(np.diag(D) @ S) / var if var > 1e-12 else 1.0
        else:
            s = 1.0
        t = md - s * R @ ms
        return s, R, t

    try:
        s, R, t = solve(w)
        for _ in range(n_iter):
            res = np.linalg.norm((s * (R @ src.T).T + t) - dst, axis=1)
            sig = np.median(res) + 1e-9
            u = res / (4.685 * sig)                       # Tukey biweight
            tw = np.where(u < 1, (1 - u ** 2) ** 2, 0.0)
            if tw.sum() < 1e-9 or (tw > 0).sum() < 4:
                break
            s, R, t = solve(tw)
    except np.linalg.LinAlgError:
        return None
    return s, R, t


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


def unproject(kpts_orig, depth, K, orig_wh, conf=None, conf_thr=-np.inf):
    """Original-res keypoints -> 3D camera points via processed-res depth/K.

    Returns (pts [M,3], valid mask over input kpts).
    """
    H, W = depth.shape
    sx, sy = W / orig_wh[0], H / orig_wh[1]
    u = np.clip(np.round(kpts_orig[:, 0] * sx).astype(int), 0, W - 1)
    v = np.clip(np.round(kpts_orig[:, 1] * sy).astype(int), 0, H - 1)
    d = depth[v, u]
    ok = np.isfinite(d) & (d > 0)
    if conf is not None:
        ok &= conf[v, u] >= conf_thr
    rays = np.linalg.inv(K) @ np.stack([u, v, np.ones_like(u)], 0).astype(float)
    pts = (rays * d[None, :]).T
    return pts, ok


def write_trajectory(path, frame_ids, extr):
    """extr may be Sim(3) 4x4 (sim3 chaining): normalize out the scale."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("# timestamp, name_image, tx, ty, tz, qw, qx, qy, qz\n")
        for i, fid in enumerate(frame_ids):
            M = extr[i]
            s = np.cbrt(np.linalg.det(M[:3, :3]))
            R_cw = M[:3, :3] / s
            R_wc = R_cw.T
            C_w = -R_wc @ (M[:3, 3] / s)
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


def write_tree(out, seq, runs, ids, extr, pts, cols, max_points, t_init, sec):
    for run in range(1, runs + 1):
        rd = out / seq / str(run)
        write_trajectory(rd / "camera_trajectory" / "cam_traj_map_000.txt", ids, extr)
        write_points3d(rd / "3D_maps" / "000" / "points3D.txt", pts, cols, max_points)
        rt = rd / "runtime.txt"
        rt.parent.mkdir(parents=True, exist_ok=True)
        rt.write_text(f"init_seconds={t_init:.6f}\nprocessing_seconds={sec:.6f}\n")


# --------------------------------------------------------------------- main --
def run_seq(model, matcher, frames, args):
    from PIL import Image
    n = len(frames)
    orig_wh = Image.open(frames[0][1]).size

    w2c_lg = [np.eye(4)]
    w2c_da3 = [np.eye(4)]
    pts_lg, col_lg = [], []
    n_fallback = 0
    feats_prev = matcher.feats(str(frames[0][1]))

    t0 = time.perf_counter()
    for i in range(n - 1):
        p0, p1 = str(frames[i][1]), str(frames[i + 1][1])
        with torch.no_grad():
            pred = model.inference([p0, p1], process_res=args.process_res,
                                   export_format="mini_npz")
        extr = np.asarray(pred.extrinsics, np.float64)      # [2,3,4] or [2,4,4] w2c
        if extr.shape[1] == 3:
            extr = np.concatenate(
                [extr, np.tile([[[0.0, 0.0, 0.0, 1.0]]], (extr.shape[0], 1, 1))], axis=1)
        depth = np.asarray(pred.depth, np.float32)          # [2,H,W]
        conf = np.asarray(pred.conf, np.float32) if pred.conf is not None else None
        intr = np.asarray(pred.intrinsics, np.float64)      # [2,3,3]
        imgs = np.asarray(pred.processed_images) if pred.processed_images is not None else None

        T_da3 = extr[1] @ np.linalg.inv(extr[0])            # cam_i -> cam_{i+1}
        w2c_da3.append(T_da3 @ w2c_da3[-1])

        feats_next = matcher.feats(p1)
        idx, k0, k1 = matcher.match(feats_prev, feats_next)
        feats_prev = feats_next

        T_lg = None
        if len(idx) >= args.min_matches:
            thr0 = np.percentile(conf[0], args.conf_pct) if conf is not None else -np.inf
            thr1 = np.percentile(conf[1], args.conf_pct) if conf is not None else -np.inf
            X0, ok0 = unproject(k0[idx[:, 0]], depth[0], intr[0], orig_wh,
                                None if conf is None else conf[0], thr0)
            X1, ok1 = unproject(k1[idx[:, 1]], depth[1], intr[1], orig_wh,
                                None if conf is None else conf[1], thr1)
            ok = ok0 & ok1
            if ok.sum() >= args.min_matches:
                out = rigid_fit(X0[ok], X1[ok], fit=args.fit)
                if out is not None:
                    s, R, t = out
                    T_lg = np.eye(4)
                    T_lg[:3, :3] = s * R      # Sim(3) as 4x4; s=1 in se3 mode
                    T_lg[:3, 3] = t
        if T_lg is None:
            T_lg = T_da3
            n_fallback += 1
        w2c_lg.append(T_lg @ w2c_lg[-1])

        # sparse map from frame i's depth in the matcher-VO global frame
        if i % args.point_every == 0:
            H, W = depth.shape[1:]
            ys, xs = np.arange(0, H, args.stride_px), np.arange(0, W, args.stride_px)
            gx, gy = np.meshgrid(xs, ys)
            gx, gy = gx.ravel(), gy.ravel()
            d = depth[0][gy, gx]
            c = conf[0][gy, gx] if conf is not None else np.ones_like(d)
            thr = np.percentile(conf[0], args.conf_pct) if conf is not None else -np.inf
            m = (c >= thr) & np.isfinite(d) & (d > 0)
            if m.any():
                uu, vv, dd = gx[m], gy[m], d[m]
                rays = np.linalg.inv(intr[0]) @ np.stack([uu, vv, np.ones_like(uu)], 0)
                cam = (rays * dd[None, :])
                c2w = np.linalg.inv(w2c_lg[i])
                world = (c2w[:3, :3] @ cam).T + c2w[:3, 3]
                pts_lg.append(world)
                col_lg.append(imgs[0][vv, uu] if imgs is not None
                              else np.full((int(m.sum()), 3), 128))
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{n-1} pairs, fallbacks={n_fallback}", flush=True)

    sec = time.perf_counter() - t0
    pts = np.concatenate(pts_lg, 0) if pts_lg else np.zeros((0, 3))
    cols = np.concatenate(col_lg, 0) if col_lg else np.zeros((0, 3))
    return np.stack(w2c_lg), np.stack(w2c_da3), pts, cols, sec, n_fallback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_root", required=True)
    ap.add_argument("--out_lg", required=True, help="matcher+IRLS VO tree")
    ap.add_argument("--out_da3", default=None, help="DA3 pair-extrinsic ablation tree")
    ap.add_argument("--seqs", nargs="*", default=None)
    ap.add_argument("--model", default="depth-anything/DA3NESTED-GIANT-LARGE")
    ap.add_argument("--fit", choices=["se3", "sim3"], default="se3")
    ap.add_argument("--process_res", type=int, default=504)
    ap.add_argument("--conf_pct", type=float, default=30.0)
    ap.add_argument("--min_matches", type=int, default=8)
    ap.add_argument("--stride_px", type=int, default=8)
    ap.add_argument("--point_every", type=int, default=8)
    ap.add_argument("--max_points", type=int, default=200000)
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args()

    from depth_anything_3.api import DepthAnything3
    t_init0 = time.perf_counter()
    model = DepthAnything3.from_pretrained(args.model).to("cuda").eval()
    matcher = Matcher(device="cuda", fp16=False)   # fp16 hurt ATE in tri3d
    t_init = time.perf_counter() - t_init0
    print(f"model {args.model} + ALIKED/LightGlue loaded in {t_init:.1f}s", flush=True)

    root = Path(args.frames_root)
    out_lg = Path(args.out_lg)
    out_da3 = Path(args.out_da3) if args.out_da3 else None
    seq_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if args.seqs:
        seq_dirs = [d for d in seq_dirs if d.name in args.seqs]

    for d in seq_dirs:
        frames = sorted((int(p.stem), p) for p in d.glob("*.png"))
        print(f"{d.name}: {len(frames)} frames, fit={args.fit}", flush=True)
        w2c_lg, w2c_da3, pts, cols, sec, nfb = run_seq(model, matcher, frames, args)
        ids = [fid for fid, _ in frames]
        write_tree(out_lg, d.name, args.runs, ids, w2c_lg, pts, cols,
                   args.max_points, t_init, sec)
        if out_da3 is not None:
            write_tree(out_da3, d.name, args.runs, ids, w2c_da3, pts, cols,
                       args.max_points, t_init, sec)
        print(f"  {len(ids)} poses, {sec:.1f}s ({sec/len(ids):.2f} s/frame), "
              f"fallbacks={nfb}", flush=True)
    print("Done.")


if __name__ == "__main__":
    main()
