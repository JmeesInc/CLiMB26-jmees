#!/usr/bin/env python3
"""Supervised LoRA on the simulated sequences -- the only ground-truth poses we have.

`data/Simulated_Sequences/Seq_0..5` ship `trajectory.csv` (camera-to-world
quaternion + centre, decimetres) alongside the renders. Everything trained so
far has been self-supervised (cross-window consistency), because the real
sequences have no reference; this is the one place an absolute rotation target
exists, and rotation is where the leaderboard says we are weakest (score_rot
5.674 against CCLAB's 3.206).

Two properties of this data bound what it can teach, and the objective is
shaped around them:

  * All six sequences share a byte-identical trajectory -- only the deformation
    differs. So there is effectively ONE trajectory, and a pose-regression
    objective would memorise it. Training therefore targets only RELATIVE
    rotations inside a window, which are a local motion property rather than a
    property of the path, and keeps a frozen-model anchor so the adapter cannot
    drift far on inputs unlike these.
  * The rendered path alternates long static stretches with fast sweeps: 172 of
    321 consecutive frames do not move the camera at all, and a 12-view window
    at stride 5 spans 48 deg of rotation at the median and 127 deg at the worst.
    Real colonoscopy windows are nothing like that (our RPE at d=40, ~0.8 s, is
    ~5 deg), so windows are drawn at stride 2 and rejected unless the camera
    actually moves and the largest pairwise rotation stays under `motion_cap`.
    That keeps 1297 of 1728 windows with an 11.7 deg median span -- the regime
    the submission actually runs in.
  * The renders are pinhole at 45.4 deg half-angle, while deployment feeds DA3 a
    kb4-rectified virtual pinhole at 28.8 deg (DA3_RECT_FSCALE=1.8). Feeding the
    wider cone would train the adapter on a focal length the submission never
    sees -- the exact error that cost 0.7 mm ATE before fscale was corrected --
    so frames are centre-cropped to match the deployed cone.

Translation is supervised only as a DIRECTION: the sim scale (decimetres of a
synthetic colon) has no relationship to what DA3 outputs on real clips.
"""
import argparse, glob, json, logging, math, os, random, sys, time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation

os.environ.setdefault("HF_HUB_OFFLINE", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lora import (inject_lora, enable_block_checkpointing, lora_state_dict,  # noqa: E402
                  LoRALinear)
from train_lora import rel_poses, geodesic, set_lora_enabled        # noqa: E402

SIM = Path("/data4/src/shunsuke/MICCAI2026/CLiMB/data/Simulated_Sequences")
SIM_FX, SIM_CX, SIM_CY = 472.64955100886374, 479.5, 359.5
DEPLOY_HALF_ANGLE = math.degrees(math.atan(720.0 / (727.1851 * 1.8)))   # 28.8 deg

CFG = dict(
    experiment_name="sim_rot", seed=0, views=12, stride=2, res=448, motion_cap=25.0,
    rank=8, alpha=16, lr=2e-4, wd=0.01, steps=400,
    w_rot=1.0, w_tdir=0.2, w_anchor=0.5, grad_clip=1.0,
    log_every=10, save_every=50,
    out_root=str(HERE.parent / "results"),
)


def _geo(A, B):
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(A.T @ B) - 1) / 2))))


def load_sequence(d):
    """-> (sorted image paths, list of (C, R_wc)) truncated to a common length."""
    ims = sorted(p for p in glob.glob(str(Path(d) / "rgb" / "*.png")))
    poses = []
    for ln in Path(d, "trajectory.csv").read_text().splitlines()[1:]:
        t = ln.strip().split(";")
        if len(t) < 7:
            continue
        try:
            v = [float(x) for x in t[:7]]
        except ValueError:
            continue
        q = np.array(v[3:7])
        n = np.linalg.norm(q)
        if n < 1e-8:
            continue
        poses.append((np.array(v[:3]), Rotation.from_quat(q / n).as_matrix()))
    k = min(len(ims), len(poses))
    return ims[:k], poses[:k]


def crop_to_deploy_fov(im):
    """Centre-crop so the remaining cone matches the deployed virtual pinhole."""
    half = int(round(SIM_FX * math.tan(math.radians(DEPLOY_HALF_ANGLE))))
    x0, y0 = int(SIM_CX) - half, int(SIM_CY) - half * im.shape[0] // im.shape[1]
    hh = half * im.shape[0] // im.shape[1]
    return im[max(y0, 0):int(SIM_CY) + hh, max(x0, 0):int(SIM_CX) + half]


def gt_w2c(poses, idx, device):
    E = []
    for i in idx:
        C, R_wc = poses[i]
        R = R_wc.T
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = -R @ C
        E.append(T)
    return torch.from_numpy(np.stack(E)).float().to(device)


def main():
    ap = argparse.ArgumentParser()
    for k, v in CFG.items():
        ap.add_argument(f"--{k}", type=type(v), default=v)
    a = ap.parse_args()
    cfg = vars(a)
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])

    out = Path(cfg["out_root"]) / cfg["experiment_name"]
    i = 1
    while out.exists():
        out = Path(cfg["out_root"]) / f"{cfg['experiment_name']}_{i:03d}"; i += 1
    out.mkdir(parents=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.FileHandler(out / "train.log"),
                                  logging.StreamHandler(sys.stdout)])
    log = logging.getLogger("simlora")
    log.info("config: %s", json.dumps(cfg))

    V, S = cfg["views"], cfg["stride"]
    seqs, n_all = [], 0
    for d in sorted(glob.glob(str(SIM / "Seq_*"))):
        ims, poses = load_sequence(d)
        if len(ims) <= V * S:
            continue
        C = np.stack([p[0] for p in poses])
        Rw = np.stack([p[1] for p in poses])
        starts = []
        for s0 in range(len(ims) - V * S):
            idx = [s0 + k * S for k in range(V)]
            if np.linalg.norm(C[idx[-1]] - C[idx[0]]) < 1e-6:
                continue                                  # camera never moved
            m = max(_geo(Rw[i], Rw[j]) for a, i in enumerate(idx) for j in idx[a + 1:])
            if m <= cfg["motion_cap"]:
                starts.append(s0)
        n_all += len(ims) - V * S
        if starts:
            seqs.append((d, ims, poses, starts))
            log.info("%s: %d frames, %d usable windows", Path(d).name, len(ims), len(starts))
    log.info("total usable windows: %d / %d",
             sum(len(x[3]) for x in seqs), n_all)
    if not seqs:
        log.error("no simulated sequences found under %s", SIM); return 1

    torch.cuda.is_bf16_supported = lambda *x, **k: False    # Turing: real fp16
    from depth_anything_3.api import DepthAnything3
    api = DepthAnything3.from_pretrained(
        os.environ.get("DA3_MODEL", "depth-anything/DA3NESTED-GIANT-LARGE")).to("cuda").eval()
    for p in api.model.parameters():
        p.requires_grad_(False)
    params = inject_lora(api.model.da3.backbone.pretrained,
                         r=cfg["rank"], alpha=cfg["alpha"])
    # 12 views at res448 does not fit without activation recompute (measured:
    # OOM at 30 GB), and this GPU is shared.
    enable_block_checkpointing(api.model.da3.backbone.pretrained)
    log.info("LoRA %.2fM params on %d linears (block checkpointing on)",
             sum(p.numel() for p in params) / 1e6, len(params) // 2)

    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["steps"])
    scaler = torch.amp.GradScaler("cuda")

    def fwd(ims):
        imgs_cpu, _, _ = api._preprocess_inputs(ims, None, None, cfg["res"],
                                                "upper_bound_resize")
        imgs = imgs_cpu.to("cuda", non_blocking=True)[None].float()
        with torch.autocast("cuda", dtype=torch.float16):
            return api.model(imgs, None, None, export_feat_layers=[], infer_gs=False)

    hist, t0 = [], time.time()
    for step in range(1, cfg["steps"] + 1):
        d, ims, poses, starts = random.choice(seqs)
        # ONE start per window. Drawing inside the comprehension re-rolled it
        # for every k, so the model saw 12 unrelated frames and the rotation
        # error read 72 deg against a 25 deg cap.
        s0 = random.choice(starts)
        idx = [s0 + k * S for k in range(V)]
        batch = [crop_to_deploy_fov(cv2.cvtColor(cv2.imread(ims[i]), cv2.COLOR_BGR2RGB))
                 for i in idx]

        set_lora_enabled(api.model, True)
        outP = fwd(batch)
        Egt = gt_w2c(poses, idx, "cuda")
        Rp, tp = rel_poses(outP["extrinsics"][0].float())
        Rg, tg = rel_poses(Egt)
        l_rot = geodesic(Rp, Rg).mean()
        # Direction only: the sim's decimetre scale is unrelated to DA3's output.
        tpn = tp / (tp.norm(dim=-1, keepdim=True) + 1e-8)
        tgn = tg / (tg.norm(dim=-1, keepdim=True) + 1e-8)
        l_tdir = (1.0 - (tpn * tgn).sum(-1)).mean()

        with torch.no_grad():
            set_lora_enabled(api.model, False)
            ref = fwd(batch)
            set_lora_enabled(api.model, True)
        Rr, tr = rel_poses(ref["extrinsics"][0].float())
        l_anc = geodesic(Rp, Rr).mean()

        loss = cfg["w_rot"] * l_rot + cfg["w_tdir"] * l_tdir + cfg["w_anchor"] * l_anc
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
        scaler.step(opt); scaler.update(); sched.step()

        hist.append(dict(step=step, loss=float(loss), rot_deg=math.degrees(float(l_rot)),
                         tdir=float(l_tdir), anchor_deg=math.degrees(float(l_anc)),
                         seq=Path(d).name))
        if step % cfg["log_every"] == 0:
            r = hist[-1]
            log.info("step %d loss %.4f  rot %.3f deg  tdir %.4f  anchor %.3f deg  %.1fs/step",
                     step, r["loss"], r["rot_deg"], r["tdir"], r["anchor_deg"],
                     (time.time() - t0) / step)
        if step % cfg["save_every"] == 0 or step == cfg["steps"]:
            torch.save(lora_state_dict(api.model), out / f"lora_{step:04d}.pt")
            (out / "training_log.json").write_text(json.dumps(hist))
    log.info("done -> %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
