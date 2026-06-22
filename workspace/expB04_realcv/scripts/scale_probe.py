#!/usr/bin/env python3
"""Is DA3's metric scale consistent across windows on REAL clips?

v001-v003 chain windows with scale FIXED to 1, justified on sim ("DA3 depth is
metric and measurably consistent"). The real-clip error profile is a smooth
low-frequency excursion (not drift, not noise), which is what a slowly varying
scale would produce. This logs, per window:
  f_pred  - DA3's self-estimated focal (it scales metric depth by this)
  s_ovl   - the scale the overlap WOULD imply (we currently force 1.0)
"""
import sys, os, argparse
import numpy as np, torch, cv2

sys.path.insert(0, "/data4/src/shunsuke/MICCAI2026/CLiMB/submit/v003_da3_stride")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import predict as P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--endo", type=int, default=7)
    a = ap.parse_args()

    from depth_anything_3.api import DepthAnything3
    torch.cuda.is_bf16_supported = lambda *x, **k: False      # fp16 fast path
    model = DepthAnything3.from_pretrained(P.MODEL_ID).to("cuda").eval()

    cap = cv2.VideoCapture(a.video); ok, first = cap.read()
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
    h, w = first.shape[:2]
    from calib_table import CALIBS
    maps = P.build_rectify_maps(CALIBS[a.endo], w, h)
    crop = P.content_crop_box(cv2.remap(first, maps[0], maps[1], cv2.INTER_LINEAR))

    kf = list(range(0, n, a.stride))
    vid = P.VideoWindow(a.video, kf, maps, crop)
    step = P.CHUNK - P.OVERLAP
    starts = list(range(0, max(len(kf) - P.OVERLAP, 1), step))

    glob, rows = [None] * len(kf), []
    for ci, s0 in enumerate(starts):
        s1 = min(s0 + P.CHUNK, len(kf))
        frames = vid.window(s0, s1)
        if len(frames) < 2:
            break
        with torch.no_grad():
            pred = model.inference(frames, process_res=P.PROCESS_RES, export_format="mini_npz")
        extr = P.as_4x4(pred.extrinsics)
        f_pred = float(np.asarray(pred.intrinsics, np.float64)[0][0, 0])
        C_l, R_l = P.centers_rots(extr)
        shared = [k for k in range(s0, s1) if glob[k] is not None]
        if shared:
            Cg, Rg = P.centers_rots(np.stack([glob[k] for k in shared]))
            loc = [k - s0 for k in shared]
            # what the overlap says the scale is (we deploy s=1)
            _, RA, tA = P.align_window(C_l[loc], R_l[loc], Cg, Rg, estimate_scale=False)
            s_ovl, _, _ = P.align_window(C_l[loc], R_l[loc], Cg, Rg, estimate_scale=True)
        else:
            RA, tA, s_ovl = np.eye(3), np.zeros(3), 1.0
        for k in range(s0, s1):
            if glob[k] is not None:
                continue
            j = k - s0
            C_g = RA @ C_l[j] + tA
            R_g = RA @ R_l[j]
            E = np.eye(4); E[:3, :3] = R_g.T; E[:3, 3] = -R_g.T @ C_g
            glob[k] = E
        rows.append((ci, f_pred, s_ovl))

    f = np.array([r[1] for r in rows]); s = np.array([r[2] for r in rows[1:]])
    # Does the window-to-window focal ratio explain the scale disagreement?
    ratio_prev_cur = f[:-1] / f[1:]
    ratio_cur_prev = f[1:] / f[:-1]
    for nm, r in (("f_prev/f_cur", ratio_prev_cur), ("f_cur/f_prev", ratio_cur_prev)):
        c = np.corrcoef(np.log(s), np.log(r))[0, 1]
        print(f"  corr(log s_ovl, log {nm}) = {c:+.3f}")
    print(f"  residual if we used {nm}: |log s - log r| mean "
          f"{np.abs(np.log(s) - np.log(ratio_prev_cur)).mean():.3f} vs raw |log s| "
          f"{np.abs(np.log(s)).mean():.3f}")
    print(f"windows: {len(rows)}")
    print(f"DA3 predicted focal : mean {f.mean():.2f}  std {f.std():.2f} "
          f"({f.std()/f.mean()*100:.2f}%)  min {f.min():.2f} max {f.max():.2f}")
    print(f"overlap-implied scale: mean {s.mean():.4f}  std {s.std():.4f}  "
          f"min {s.min():.4f} max {s.max():.4f}")
    print(f"  |s-1| mean {np.abs(s-1).mean():.4f}  median {np.median(np.abs(s-1)):.4f}  "
          f"p90 {np.percentile(np.abs(s-1),90):.4f}")
    print(f"  cumulative product of s: {np.prod(s):.4f}  (1.0 = no net scale drift)")


if __name__ == "__main__":
    main()
