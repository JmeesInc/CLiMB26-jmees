#!/usr/bin/env python3
"""Precompute LightGlue relative rotations as a TEACHER for the LoRA.

Why: the existing self-supervised loss rewards two overlapping windows for
AGREEING on the poses they share. That is satisfiable by a degenerate solution —
emit a smooth, input-insensitive rotation and consistency is perfect — which is
exactly what run C did (ATE 4.374->4.075 while rot 5.18->6.00->7.01->7.91).
Anchoring the rotation to the frozen model (run E) cannot fix it: the anchor
pulls back toward the very thing we are trying to improve, so it killed the ATE
gain too.

So supervise rotation with an INDEPENDENT measurement instead. ALIKED+LightGlue
+ essential matrix gives a relative rotation computed from image correspondences
alone; v007 showed those rotations reach CV rot 3.98 versus DA3's 7.01. Using
them at inference cost +0.006 s/f and busted the runtime budget (W_t 1.063 ->
climb +0.713). Distilling them into the adapter moves that signal into the
weights, where it costs nothing at inference.

Output per clip: lg_rot.npz with i, j, R (P,3,3) and ninl (P,).
"""
import argparse, glob, json, os, sys, time
from pathlib import Path

import cv2
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips_dir", default=str(Path(__file__).resolve().parent.parent / "clips"))
    ap.add_argument("--max_off", type=int, default=11, help="rel_poses uses all pairs in a V=12 window")
    ap.add_argument("--kp", type=int, default=1024)
    ap.add_argument("--min_inl", type=int, default=25)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--anchor_step", type=int, default=2,
                    help="supervise every Nth frame; training samples random windows "
                         "so sparse teacher coverage is enough and this halves the cost")
    ap.add_argument("--overwrite", type=int, default=0)
    a = ap.parse_args()

    from lightglue import ALIKED, LightGlue
    dev = "cuda"
    ext = ALIKED(max_num_keypoints=a.kp).eval().to(dev)
    lg = LightGlue(features="aliked", depth_confidence=-1, width_confidence=-1).eval().to(dev)

    clips = sorted(d for d in glob.glob(os.path.join(a.clips_dir, "*")) if os.path.isdir(d))
    clips = clips[a.start:(a.start + a.limit) if a.limit else None]
    t0, done, tot_edges = time.time(), 0, 0
    for ci, d in enumerate(clips):
        outp = Path(d) / "lg_rot.npz"
        if outp.exists() and not a.overwrite:
            continue
        meta = json.loads((Path(d) / "meta.json").read_text())
        K = np.asarray(meta["K"], np.float64)
        files = sorted(glob.glob(os.path.join(d, "*.jpg")))
        if len(files) < 4:
            continue
        feats = []
        for f in files:
            im = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(im).permute(2, 0, 1).float().div(255).to(dev)[None]
            with torch.no_grad():
                feats.append(ext.extract(t))
        I, J, RR, NI = [], [], [], []
        n = len(files)
        for k in range(0, n, a.anchor_step):
            offs = [o for o in range(1, a.max_off + 1) if k + o < n]
            if not offs:
                continue
            # ALIKED returns a VARIABLE keypoint count (max_num_keypoints is a cap,
            # not a target), so batching only works within a group that happens to
            # agree. Group by count and run one call per group -- dropping the
            # mismatched partners instead (as a first cut did) threw away 99.8% of
            # the pairs.
            kk = feats[k]["keypoints"].shape[1]
            groups = {}
            for o in offs:
                groups.setdefault(feats[k + o]["keypoints"].shape[1], []).append(k + o)
            kp0 = feats[k]["keypoints"][0].float().cpu().numpy()
            pairs_out = []
            for _, partners in groups.items():
                d0 = {key: torch.cat([feats[k][key]] * len(partners)) for key in feats[k]}
                d1 = {key: torch.cat([feats[p][key] for p in partners]) for key in feats[k]}
                with torch.no_grad():
                    m = lg({"image0": d0, "image1": d1})
                m0 = m["matches0"].cpu().numpy()
                for bi, p in enumerate(partners):
                    pairs_out.append((p, m0[bi]))
            for p, row in pairs_out:
                sel = np.nonzero(row >= 0)[0]
                if len(sel) < a.min_inl:
                    continue
                kp1 = feats[p]["keypoints"][0].float().cpu().numpy()
                p0 = kp0[sel].astype(np.float64)
                p1 = kp1[row[sel]].astype(np.float64)
                E, inl = cv2.findEssentialMat(p0, p1, K, method=cv2.USAC_MAGSAC,
                                              prob=0.999, threshold=1.0)
                if E is None or E.shape != (3, 3):
                    continue
                ninl, R, _, _ = cv2.recoverPose(E, p0, p1, K, mask=inl)
                if ninl < a.min_inl:
                    continue
                I.append(k); J.append(p); RR.append(R); NI.append(int(ninl))
        np.savez(outp, i=np.array(I, np.int32), j=np.array(J, np.int32),
                 R=np.array(RR, np.float32).reshape(-1, 3, 3), ninl=np.array(NI, np.int32))
        done += 1; tot_edges += len(I)
        if done % 20 == 0:
            el = time.time() - t0
            print(f"{done}/{len(clips)} clips  {tot_edges} edges  "
                  f"{el/done:.2f}s/clip  eta {(len(clips)-ci-1)*el/done/60:.0f}min", flush=True)
    print(f"DONE {done} clips, {tot_edges} edges, {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
