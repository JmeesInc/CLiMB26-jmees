#!/usr/bin/env python3
"""Which rotation source can reach rot@40 < 3.27 deg (the CCLAB tie-break)?

score_rot uses ONLY the rotations in the trajectory file; ATE uses only the
centres. So rotations can be replaced independently. Our chain's rotation error
random-walks: per-keyframe-step error eps accumulates ~ eps*sqrt(span/stride);
hitting 3.27@40 needs eps <~ 1.2 deg. This probe measures eps per source:

  da3   : relative rotations read from the deployed v005 chain output
  feat  : ALIKED+LightGlue matches on rectified keyframes -> essential -> R
          (rotation from epipolar geometry is well-conditioned even at small
          baseline -- the mechanism behind ORB-SLAM3's 1.24deg@40 on 003_a)
  ident : R = I per step (the toy baseline's implied per-step error)

GT: COLMAP images.txt. Edges: consecutive stride-6 keyframes (6 frames apart)
and skip edges (12 frames) for the averaging-graph design.
"""
import sys, time, argparse
from pathlib import Path
import numpy as np, cv2, torch

R_ = "/data4/src/shunsuke/MICCAI2026/CLiMB"
sys.path.insert(0, f"{R_}/workspace/expB01_da3_submap/eda")
from pred_eda import read_gt, read_pred  # noqa

sys.path.insert(0, f"{R_}/submit/v004_pe_tri3d")  # not used; lightglue is a package
from lightglue import ALIKED, LightGlue
from lightglue.utils import rbd


def geo(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def rel(Rw_i, Rw_j):
    return Rw_j.T @ Rw_i           # cam_i -> cam_j (camera coords)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default=f"{R_}/workspace/expB08_vggtslam_real/frames")
    ap.add_argument("--gt", default=f"{R_}/workspace/expB04_realcv/colmap_gt")
    ap.add_argument("--chain", default=f"{R_}/workspace/expB04_realcv/out_r392_c16o8_s4")
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--steps", type=int, nargs="*", default=[1, 2])
    a = ap.parse_args()

    dev = "cuda"
    ext = ALIKED(max_num_keypoints=1024).eval().to(dev)
    lg = LightGlue(features="aliked").eval().to(dev)

    # rectified pinhole K per family (from predict_cond build_rectify_maps)
    sys.path.insert(0, f"{R_}/workspace/expB06_posecond/scripts")
    import predict_cond as P

    for seq in ["Seq_001_a", "Seq_001_c", "Seq_003_a", "Seq_003_b"]:
        gt = read_gt(Path(a.gt) / seq / "results_txt" / "images.txt")
        ch = read_pred(Path(a.chain) / seq / "1" / "camera_trajectory" / "cam_traj_map_000.txt")
        files = sorted((Path(a.frames) / seq).glob("*.png"))
        ids = [int(f.stem) for f in files]
        calib, _ = P.calib_for_sequence(seq)
        _, _, K = P.build_rectify_maps(calib, 1440, 1080, P.RECT_BALANCE)
        Ks = K.copy(); Ks[:2] *= a.scale

        feats, t_feat = {}, 0.0
        def feat(idx):
            nonlocal t_feat
            if idx not in feats:
                im = cv2.imread(str(files[idx]))
                im = cv2.resize(im, None, fx=a.scale, fy=a.scale)
                t0 = time.time()
                ten = torch.from_numpy(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)).permute(2,0,1).float().div(255).to(dev)[None]
                with torch.no_grad():
                    feats[idx] = ext.extract(ten)
                t_feat += time.time() - t0
            return feats[idx]

        for st in a.steps:
            errs_f, errs_d, errs_i, n_fail, t_match = [], [], [], 0, 0.0
            for i in range(0, len(files) - st, 1):
                j = i + st
                fi, fj = ids[i], ids[j]
                if fi not in gt or fj not in gt or fi not in ch or fj not in ch:
                    continue
                Rgt = rel(gt[fi][0], gt[fj][0])
                errs_d.append(geo(Rgt, rel(ch[fi][0], ch[fj][0])))
                errs_i.append(geo(Rgt, np.eye(3)))
                f0, f1 = feat(i), feat(j)
                t0 = time.time()
                with torch.no_grad():
                    m = lg({"image0": f0, "image1": f1})
                t_match += time.time() - t0
                f0r, f1r, mr = [rbd(x) for x in (f0, f1, m)]
                idx = mr["matches"].cpu().numpy()
                if len(idx) < 15:
                    n_fail += 1; errs_f.append(None); continue
                p0 = f0r["keypoints"].cpu().numpy()[idx[:, 0]]
                p1 = f1r["keypoints"].cpu().numpy()[idx[:, 1]]
                E, inl = cv2.findEssentialMat(p0, p1, Ks, method=cv2.RANSAC, prob=0.999, threshold=1.0)
                if E is None:
                    n_fail += 1; errs_f.append(None); continue
                _, Rf, tf, _ = cv2.recoverPose(E, p0, p1, Ks, mask=inl)
                errs_f.append(geo(Rgt, Rf))
            ef = np.array([e for e in errs_f if e is not None])
            ed, ei = np.array(errs_d), np.array(errs_i)
            span = 6 * st
            proj = lambda med: med * np.sqrt(40 / span)
            print(f"{seq} step{st} (Δ{span}f, n={len(ed)}): "
                  f"da3 med {np.median(ed):.2f}° | feat med {np.median(ef):.2f}° "
                  f"(mean {ef.mean():.2f}, fail {n_fail}) | ident med {np.median(ei):.2f}° "
                  f"|| @40 proj: da3 {proj(np.median(ed)):.1f} feat {proj(np.median(ef)):.1f}", flush=True)
        print(f"{seq}: feat extract {t_feat/len(feats)*1000:.0f} ms/kf, match {t_match/max(len(errs_d),1)*1000:.0f} ms/pair (RTX8000 half-res)", flush=True)


if __name__ == "__main__":
    main()
