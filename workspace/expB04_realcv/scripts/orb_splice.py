#!/usr/bin/env python3
"""Offline test of the classical-rotation hybrid: splice ORB-SLAM3 rotations
into our trajectory wherever ORB tracked, keep our camera centres everywhere.

Why bother: ORB-SLAM3 reached RPE_rot 1.24 deg on the segments it tracked on
these clips, against 3.0 for our best (Pi3X + rotation averaging). It fails to
track 2 of 4 clips, so it cannot be the primary system -- but rotation
substitution is ATE-invariant (measured on v006/v007), so its rotations can be
borrowed where they exist. Each ORB map lives in its own world frame, so one
global rotation per map is fitted on the shared frames before substituting.
"""
import sys, glob, shutil
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

def load(path):
    out = {}
    for ln in Path(path).read_text().splitlines():
        if ln.startswith("#") or not ln.strip(): continue
        v = ln.split(",")
        q = [float(v[6]), float(v[7]), float(v[8]), float(v[5])]     # xyzw
        out[int(v[1].replace(".png",""))] = (Rotation.from_quat(q).as_matrix(),
                                             np.array([float(v[2]),float(v[3]),float(v[4])]), v[0])
    return out

def proj_so3(M):
    U,_,Vt = np.linalg.svd(M); return U@np.diag([1,1,np.sign(np.linalg.det(U@Vt))])@Vt

ours_root, orb_root, out_root = map(Path, sys.argv[1:4])
min_map = 100
if out_root.exists(): shutil.rmtree(out_root)
shutil.copytree(ours_root, out_root)
for seq_dir in sorted(p for p in ours_root.iterdir() if p.is_dir()):
    seq = seq_dir.name
    ours = load(seq_dir/"1"/"camera_trajectory"/"cam_traj_map_000.txt")
    new = {k:(R.copy(),C,ts) for k,(R,C,ts) in ours.items()}
    n_sub = 0; report = []
    for mf in sorted(glob.glob(str(orb_root/seq/"1"/"camera_trajectory"/"cam_traj_map_*.txt"))):
        orb = load(mf)
        common = sorted(set(orb) & set(ours))
        if len(common) < min_map: report.append(f"{Path(mf).stem}:{len(common)}f skip"); continue
        M = sum(ours[f][0] @ orb[f][0].T for f in common)
        Ra = proj_so3(M)
        resid = np.degrees(np.mean([np.arccos(np.clip((np.trace((Ra@orb[f][0]).T@ours[f][0])-1)/2,-1,1)) for f in common]))
        for f in common:
            new[f] = (Ra @ orb[f][0], ours[f][1], ours[f][2]); n_sub += 1
        report.append(f"{Path(mf).stem}:{len(common)}f align-resid {resid:.1f}deg")
    for rd in sorted(p for p in (out_root/seq).iterdir() if p.is_dir()):   # whatever runs exist
        p = rd/"camera_trajectory"/"cam_traj_map_000.txt"
        if not p.exists(): continue
        with open(p,"w") as fh:
            fh.write("# timestamp, name_image, tx, ty, tz, qw, qx, qy, qz\n")
            for f in sorted(new):
                R,C,ts = new[f]; q = Rotation.from_matrix(R).as_quat()  # xyzw
                # exact evaluator format: no spaces after commas (a leading space
                # in the image name breaks its frame-id parse)
                fh.write(f"{ts},{f:06d}.png,{C[0]:.9f},{C[1]:.9f},{C[2]:.9f},{q[3]:.9f},{q[0]:.9f},{q[1]:.9f},{q[2]:.9f}\n")
    print(f"{seq}: substituted {n_sub}/{len(ours)} frames  [{'; '.join(report) or 'no ORB maps'}]")
