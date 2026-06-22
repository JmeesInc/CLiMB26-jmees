#!/usr/bin/env python3
"""Robust rotation averaging over feature + DA3 relative-rotation edges, then
splice the solved rotations into the v005 trajectory (centres untouched).

Edges (stride-6 keyframe nodes):
  feat Δ1/Δ2 : ALIKED+LightGlue -> essential -> R, weight = inlier count
               (probe: median 0.5-1.4°/edge, but heavy-tailed -> Huber)
  da3  Δ1/Δ2 : relative rotations from the deployed chain, constant weight
               (median 0.45-1.7°/edge, no tail -> the safety net)
Solver: Govindu-style iterative local averaging with Huber IRLS on the
geodesic residual, initialised from the chain (so a no-edge node keeps its
chain rotation). Convention: edge R_ij = R_wc_j^T R_wc_i  =>  prediction of
R_wc_j from i is R_wc_i @ R_ij^T.
"""
import sys, time, argparse
from pathlib import Path
import numpy as np, cv2, torch

R_ = "/data4/src/shunsuke/MICCAI2026/CLiMB"
sys.path.insert(0, f"{R_}/workspace/expB01_da3_submap/eda")
sys.path.insert(0, f"{R_}/workspace/expB06_posecond/scripts")
from pred_eda import read_pred                      # noqa
import predict_cond as P                            # noqa
from predict_cond import rotmat_to_qvec_wxyz        # noqa
from lightglue import ALIKED, LightGlue             # noqa
from lightglue.utils import rbd                     # noqa
from scipy.spatial.transform import Rotation, Slerp


def proj_so3(M):
    U, _, Vt = np.linalg.svd(M)
    return U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt


def geo(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def feat_edges(files, Ks, ext, lg, steps=(1, 4, 7), min_inl=15, scale=0.5):
    feats = {}
    def get(i):
        if i not in feats:
            im = cv2.imread(str(files[i])); im = cv2.resize(im, None, fx=scale, fy=scale)
            t = torch.from_numpy(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().div(255).cuda()[None]
            with torch.no_grad():
                feats[i] = ext.extract(t)
        return feats[i]
    edges = []
    for st in steps:
        for i in range(0, len(files) - st):
            f0, f1 = get(i), get(i + st)
            with torch.no_grad():
                m = lg({"image0": f0, "image1": f1})
            f0r, f1r, mr = [rbd(x) for x in (f0, f1, m)]
            idx = mr["matches"].cpu().numpy()
            if len(idx) < min_inl:
                continue
            p0 = f0r["keypoints"].cpu().numpy()[idx[:, 0]]
            p1 = f1r["keypoints"].cpu().numpy()[idx[:, 1]]
            meth = cv2.USAC_MAGSAC if int(__import__("os").environ.get("MAGSAC", "0")) else cv2.RANSAC
            E, inl = cv2.findEssentialMat(p0, p1, Ks, method=meth, prob=0.999, threshold=1.0)
            if E is None:
                continue
            ninl, Rf, _, _ = cv2.recoverPose(E, p0, p1, Ks, mask=inl)
            if ninl < min_inl:
                continue
            edges.append((i, i + st, Rf, float(min(ninl, 300))))   # cap so one edge cannot dominate
    return edges


def solve(R_init, edges, iters=30, huber_deg=2.0):
    R = [r.copy() for r in R_init]
    for it in range(iters):
        delta = 0.0
        for i in range(len(R)):
            preds, ws = [], []
            for (a, b, Rab, w) in edges:
                if a == i:      # predict R_i from R_b: R_wc_i = R_wc_b @ R_ab... R_wc_b = R_wc_a @ Rab^T => R_wc_a = R_wc_b @ Rab
                    preds.append(R[b] @ Rab); ws.append(w)
                elif b == i:
                    preds.append(R[a] @ Rab.T); ws.append(w)
            if not preds:
                continue
            # Huber IRLS weights on geodesic residual to current estimate
            res = np.array([geo(R[i], Pm) for Pm in preds])
            hw = np.where(res <= huber_deg, 1.0, huber_deg / np.maximum(res, 1e-9))
            W = np.array(ws) * hw
            M = sum(w * Pm for w, Pm in zip(W, preds))
            Rn = proj_so3(M)
            delta = max(delta, geo(R[i], Rn))
            R[i] = Rn
        if delta < 0.01:
            break
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj_root", default=f"{R_}/workspace/expB04_realcv/out_r392_c16o8_s4")
    ap.add_argument("--frames", default=f"{R_}/workspace/expB08_vggtslam_real/frames")
    ap.add_argument("--out", default=f"{R_}/workspace/expB12_rotation/out_rotfused3")
    ap.add_argument("--w_da3", type=float, default=30.0, help="constant weight for DA3 edges (feat weight = min(inliers,300))")
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--kp", type=int, default=768)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--node_every", type=int, default=1, help="subsample the stride-6 frame list")
    ap.add_argument("--steps", type=str, default="1,4,7")
    a = ap.parse_args()

    ext = ALIKED(max_num_keypoints=a.kp).eval().cuda()
    lg = LightGlue(features="aliked").eval().cuda()

    for seq in ["Seq_001_a", "Seq_001_c", "Seq_003_a", "Seq_003_b"]:
        traj_f = Path(a.traj_root) / seq / "1" / "camera_trajectory" / "cam_traj_map_000.txt"
        ch = read_pred(traj_f)
        fids = sorted(ch)
        files = sorted((Path(a.frames) / seq).glob("*.png"))[::a.node_every]
        kf_ids = [int(f.stem) for f in files]
        kf_set = {fid: k for k, fid in enumerate(kf_ids)}
        calib, _ = P.calib_for_sequence(seq)
        _, _, K = P.build_rectify_maps(calib, 1440, 1080, P.RECT_BALANCE)
        Ks = K.copy(); Ks[:2] *= a.scale

        t0 = time.time()
        ef = feat_edges(files, Ks, ext, lg, steps=tuple(int(x) for x in a.steps.split(",")), scale=a.scale)
        R_init = [ch[fid][0] for fid in kf_ids]                # R_wc at keyframes (chain)
        ed = []
        for st in (1, 2):
            for i in range(len(kf_ids) - st):
                Rab = R_init[i + st].T @ R_init[i]
                ed.append((i, i + st, Rab, a.w_da3))
        R_solved = solve(R_init, ef + ed)
        # round 2: drop edges inconsistent with the round-1 solution (>3x Huber)
        def ok(e):
            a_, b_, Rab, _ = e
            return geo(R_solved[b_], R_solved[a_] @ Rab.T) < 6.0
        kept = [e for e in ef if ok(e)]
        R_solved = solve(R_solved, kept + ed)
        print(f"  round2 kept {len(kept)}/{len(ef)} feat edges", flush=True)
        moved = np.mean([geo(x, y) for x, y in zip(R_init, R_solved)])

        # splice: keyframe rotations from the solve, SLERP between, centres as-is
        key_R = Rotation.from_matrix(np.stack(R_solved))
        sl = Slerp(np.array(kf_ids, float), key_R)
        out = Path(a.out) / seq / "1"
        (out / "camera_trajectory").mkdir(parents=True, exist_ok=True)
        import shutil
        for extra in ["3D_maps", "runtime.txt"]:
            src = Path(a.traj_root) / seq / "1" / extra
            dst = out / extra
            if src.is_dir() and not dst.exists():
                shutil.copytree(src, dst)
            elif src.is_file():
                shutil.copy(src, dst)
        with open(out / "camera_trajectory" / "cam_traj_map_000.txt", "w") as f:
            f.write("# timestamp, name_image, tx, ty, tz, qw, qx, qy, qz\n")
            for k, fid in enumerate(fids):
                C = ch[fid][1]
                t = float(np.clip(fid, kf_ids[0], kf_ids[-1]))
                Rw = sl([t]).as_matrix()[0]
                qw, qx, qy, qz = rotmat_to_qvec_wxyz(Rw)
                f.write(f"{k/30.0:.6f},{fid:06d}.png,{C[0]:.9f},{C[1]:.9f},{C[2]:.9f},"
                        f"{qw:.9f},{qx:.9f},{qy:.9f},{qz:.9f}\n")
        print(f"{seq}: {len(ef)} feat + {len(ed)} da3 edges, moved {moved:.2f}°, {time.time()-t0:.0f}s", flush=True)
    print("done")


if __name__ == "__main__":
    main()
