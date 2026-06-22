#!/usr/bin/env python3
"""B3: average two chains that differ only in window phase.

Per-window scale error is the dominant ATE term (daily 8/29 §14.1). Two chains
whose window boundaries are offset by half a step make different scale
mistakes, so averaging the two Sim(3)-aligned trajectories should cut that
variance -- IF the mistakes are independent. This script measures that: merge
tree A (phase 0) and tree B (phase step/2) into a new submission tree and let the
official evaluator score it next to A and B.

Usage: merge_phases.py --a out_A --b out_B --out out_AB
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "expB01_da3_submap" / "eda"))
from pred_eda import read_pred                       # {fid: (R_wc, C_w)}
from predict_cond import rotmat_to_qvec_wxyz, proj_so3


def sim3_orient(Cs, Rs, Cd, Rd):
    RA = proj_so3(sum(b @ a.T for a, b in zip(Rs, Rd)))
    ds, dd = Cs - Cs.mean(0), Cd - Cd.mean(0)
    s = float((dd * (RA @ ds.T).T).sum() / max((ds ** 2).sum(), 1e-12))
    return s, RA, Cd.mean(0) - s * RA @ Cs.mean(0)


def chordal_mean(Ra, Rb):
    return proj_so3(Ra + Rb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    A, B, O = Path(a.a), Path(a.b), Path(a.out)
    for seq in sorted(d.name for d in A.iterdir() if d.is_dir()):
        ta = read_pred(A / seq / "1" / "camera_trajectory" / "cam_traj_map_000.txt")
        tb = read_pred(B / seq / "1" / "camera_trajectory" / "cam_traj_map_000.txt")
        ids = sorted(set(ta) & set(tb))
        Ca = np.array([ta[i][1] for i in ids]); Ra = np.array([ta[i][0] for i in ids])
        Cb = np.array([tb[i][1] for i in ids]); Rb = np.array([tb[i][0] for i in ids])
        s, RA, tA = sim3_orient(Cb, Rb, Ca, Ra)
        Cb2 = s * (RA @ Cb.T).T + tA
        Rb2 = np.einsum("ij,njk->nik", RA, Rb)
        resid = np.linalg.norm(Cb2 - Ca, axis=1)
        C = 0.5 * (Ca + Cb2)
        R = [chordal_mean(x, y) for x, y in zip(Ra, Rb2)]
        for src in sorted(p for p in (A / seq).iterdir() if p.is_dir() and p.name.isdigit()):
            run = src.name
            dst = O / seq / str(run)
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            with open(dst / "camera_trajectory" / "cam_traj_map_000.txt", "w") as f:
                f.write("# timestamp, name_image, tx, ty, tz, qw, qx, qy, qz\n")
                for k, fid in enumerate(ids):
                    qw, qx, qy, qz = rotmat_to_qvec_wxyz(R[k])
                    f.write(f"{k/30.0:.6f},{fid:06d}.png,{C[k,0]:.9f},{C[k,1]:.9f},{C[k,2]:.9f},"
                            f"{qw:.9f},{qx:.9f},{qy:.9f},{qz:.9f}\n")
        print(f"{seq}: {len(ids)} frames, A-vs-B disagreement mean {resid.mean():.4f} "
              f"(x{s:.3f} scale)", flush=True)


if __name__ == "__main__":
    main()
