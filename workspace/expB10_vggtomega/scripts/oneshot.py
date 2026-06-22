#!/usr/bin/env python3
"""VGGT-Omega single-shot whole-clip reconstruction on the real-CV clips.

Why this architecture bet: our DA3 chain's dominant error is inter-window scale
inconsistency; every remedy that keeps windows (estimation, optimisation,
conditioning, LoRA) hit a ceiling. VGGT-Omega's register-based inter-frame
attention gives LINEAR memory in the number of views (README: 100 frames
13.4 GB, 500 frames 43 GB), so an entire clip fits in ONE forward pass -- a
single consistent scale that the evaluator's per-clip Sim(3) alignment absorbs
entirely. DA3's global one-shot failed for two reasons VGGT-Omega may not share:
O(N^2) attention forced res<=252 locally, and no long-range mechanism.

Pipeline: rectified stride-6 keyframes (expB08 frames/) -> one forward ->
extrinsics (camera-from-world, OpenCV) -> SLERP/linear interpolation to every
frame (predict_cond machinery) -> CLiMB tree -> official evaluator.
"""
import argparse, sys, time
from pathlib import Path
import numpy as np
import torch, cv2

sys.path.insert(0, "/data4/src/shunsuke/MICCAI2026/CLiMB/workspace/expB10_vggtomega/repo")
sys.path.insert(0, "/data4/src/shunsuke/MICCAI2026/CLiMB/workspace/expB06_posecond/scripts")
from predict_cond import interpolate_poses, write_trajectory, rotmat_to_qvec_wxyz  # noqa

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera


def write_points(path, depth, conf, K, extr, imgs, max_points=200000, stride_px=8, every=4):
    path.parent.mkdir(parents=True, exist_ok=True)
    N, H, W = depth.shape
    thr = np.percentile(conf, 40.0)
    pts, cols = [], []
    ys, xs = np.arange(0, H, stride_px), np.arange(0, W, stride_px)
    gx, gy = np.meshgrid(xs, ys); gx, gy = gx.ravel(), gy.ravel()
    for i in range(0, N, every):
        d = depth[i][gy, gx]; c = conf[i][gy, gx]
        m = (c >= thr) & np.isfinite(d) & (d > 0)
        if not m.any(): continue
        uu, vv, dd = gx[m], gy[m], d[m]
        rays = np.linalg.inv(K[i]) @ np.stack([uu, vv, np.ones_like(uu)], 0).astype(float)
        cam = rays * dd[None]
        R_cw, t_cw = extr[i][:3, :3], extr[i][:3, 3]
        world = (R_cw.T @ (cam - t_cw[:, None]))
        pts.append(world.T); cols.append(imgs[i][vv, uu])
    P = np.concatenate(pts, 0) if pts else np.zeros((0, 3))
    C = np.concatenate(cols, 0) if cols else np.zeros((0, 3))
    if len(P) > max_points:
        i = np.random.RandomState(0).choice(len(P), max_points, replace=False); P, C = P[i], C[i]
    with open(path, "w") as f:
        f.write("# POINT3D_ID X Y Z R G B ERROR\n")
        for j, (p, c) in enumerate(zip(P, C)):
            f.write(f"{j+1} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])} 0.0\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_root", default="/data4/src/shunsuke/MICCAI2026/CLiMB/workspace/expB08_vggtslam_real/frames")
    ap.add_argument("--videos_root", default="/data4/src/shunsuke/MICCAI2026/CLiMB/workspace/expB04_realcv/input")
    ap.add_argument("--ckpt", default="/data4/src/shunsuke/MICCAI2026/CLiMB/workspace/expB10_vggtomega/weights/vggt_omega_1b_512.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--seqs", nargs="*", default=None)
    ap.add_argument("--fp16", type=int, default=0, help="1 = force fp16 autocast (Turing)")
    ap.add_argument("--seg", type=int, default=0, help=">0: segment size in views (0 = whole clip)")
    ap.add_argument("--seg_ov", type=int, default=24, help="views shared between consecutive segments")
    a = ap.parse_args()

    if a.fp16:
        torch.cuda.is_bf16_supported = lambda *x, **k: False
    t0 = time.time()
    model = VGGTOmega().to("cuda").eval()
    sd = torch.load(a.ckpt, map_location="cpu")
    model.load_state_dict(sd)
    init_s = time.time() - t0
    print(f"model loaded in {init_s:.1f}s", flush=True)

    for d in sorted(Path(a.frames_root).iterdir()):
        if not d.is_dir() or (a.seqs and d.name not in a.seqs): continue
        files = sorted(d.glob("*.png"))
        kf = [int(p.stem) - 1 for p in files]                     # 0-based frame idx
        cap = cv2.VideoCapture(str(Path(a.videos_root) / f"{d.name}.mp4"))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS) or 40
        cap.release()
        t1 = time.time()

        def run_views(paths):
            images = load_and_preprocess_images(paths, image_resolution=a.res).to("cuda")
            with torch.inference_mode():
                pred = model(images)
            Hp, Wp = images.shape[-2:]
            ex, K = encoding_to_camera(pred["pose_enc"], (Hp, Wp))
            E = np.tile(np.eye(4), (len(paths), 1, 1))
            E[:, :3, :] = ex[0].float().cpu().numpy()
            dep = pred["depth"][0].float().cpu().numpy()
            if dep.ndim == 4: dep = dep.squeeze(-1)
            return (E, K[0].float().cpu().numpy(), dep,
                    pred["depth_conf"][0].float().cpu().numpy(), images)

        paths = [str(p) for p in files]
        if a.seg <= 0 or len(paths) <= a.seg:
            w2c, intr, depth, conf, images = run_views(paths)
        else:
            # Overlapping segments merged by Sim(3) fitted on the seg_ov shared
            # cameras: with ~24 well-spread cameras the scale is well-conditioned,
            # unlike the DA3 chain's 3-6 near-collinear overlap views.
            step = a.seg - a.seg_ov
            starts = list(range(0, len(paths) - a.seg_ov, step))
            if starts[-1] + a.seg < len(paths):
                starts.append(len(paths) - a.seg)
            glob = [None] * len(paths)
            depth = intr = conf = images = None
            for s0 in starts:
                s1 = min(s0 + a.seg, len(paths))
                E, K, dep, cf, ims = run_views(paths[s0:s1])
                C = np.stack([-E[i, :3, :3].T @ E[i, :3, 3] for i in range(len(E))])
                Rwc = np.stack([E[i, :3, :3].T for i in range(len(E))])
                shared = [k for k in range(s0, s1) if glob[k] is not None]
                if not shared:
                    sA, RA, tA = 1.0, np.eye(3), np.zeros(3)
                else:
                    Cg = np.stack([-glob[k][:3, :3].T @ glob[k][:3, 3] for k in shared])
                    Rg = np.stack([glob[k][:3, :3].T for k in shared])
                    loc = [k - s0 for k in shared]
                    M = sum(Rg[i] @ Rwc[loc[i]].T for i in range(len(shared)))
                    U, _, Vt = np.linalg.svd(M)
                    RA = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
                    dl = C[loc] - C[loc].mean(0); dg = Cg - Cg.mean(0)
                    sA = float((dg * (RA @ dl.T).T).sum() / max((dl ** 2).sum(), 1e-12))
                    tA = Cg.mean(0) - sA * RA @ C[loc].mean(0)
                for k in range(s0, s1):
                    if glob[k] is not None: continue
                    j = k - s0
                    Cn = sA * RA @ C[j] + tA; Rn = RA @ Rwc[j]
                    E4 = np.eye(4); E4[:3, :3] = Rn.T; E4[:3, 3] = -Rn.T @ Cn
                    glob[k] = E4
                if depth is None:
                    depth, intr, conf, images = dep, K, cf, ims
            w2c = np.stack(glob)
        proc = time.time() - t1
        full = interpolate_poses(kf, w2c, n)
        out = Path(a.out) / d.name / "1"
        write_trajectory(out / "camera_trajectory" / "cam_traj_map_000.txt", list(range(1, n + 1)), full, fps)
        imgs_np = (images.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        write_points(out / "3D_maps" / "000" / "points3D.txt", depth, conf, intr, w2c, imgs_np)
        (out / "runtime.txt").write_text(f"init_seconds={init_s:.6f}\nprocessing_seconds={proc:.6f}\n")
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(f"{d.name}: {len(kf)} views -> {n} frames, {proc:.1f}s ({proc/n:.4f} s/frame), peak {peak:.1f} GB", flush=True)
        torch.cuda.reset_peak_memory_stats()
    print("done")


if __name__ == "__main__":
    main()
