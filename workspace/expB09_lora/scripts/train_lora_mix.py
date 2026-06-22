#!/usr/bin/env python3
"""Self-supervised real clips + supervised sim rotations, alternating.

The shipped adapter (`r8_anc02_lr5e4/lora_step0200.pt`) is trained purely on
cross-window consistency over real clips. The trainer's own comment names that
objective's weakness: consistency "is happy with any smooth, input-insensitive
rotation" -- it constrains two predictions to agree, not to be right. The fix
tried before was distilling LightGlue's relative rotations, which failed (real
CV 3.35 -> 4.42) because the teacher is itself a noisy measurement on
low-texture mucosa.

The simulated sequences supply the same kind of term with an EXACT target, and
it works: sim-only supervision cut real-clip rotation error 12% (4.27 -> 3.76)
on top of rotation averaging. It also cost 19% ATE, because sim translations
drag the model toward a synthetic domain -- so here sim teaches ROTATION ONLY
and the real self-supervised objective (which includes cross-window translation
and depth consistency) keeps the rest anchored.

Alternating rather than summing: the real objective needs three forward passes
per step and the sim one needs two, so a combined step would cost five.
"""
import argparse, glob, json, logging, math, os, random, sys, time
from pathlib import Path

import numpy as np
import torch
import yaml

os.environ.setdefault("HF_HUB_OFFLINE", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lora import (inject_lora, enable_block_checkpointing, lora_state_dict,   # noqa: E402
                  LoRALinear)
from train_lora import (load_clip, consistency, anchor, rel_poses,            # noqa: E402
                        geodesic, set_lora_enabled)
from train_lora_sim import (SIM, load_sequence, crop_to_deploy_fov, gt_w2c,   # noqa: E402
                            _geo)

CFG = dict(
    # real half: exactly what produced the shipped adapter
    experiment_name="mix", seed=0, model="depth-anything/DA3NESTED-GIANT-LARGE",
    views=12, overlap=6, res=504, rank=8, alpha=16, lr=5e-4, wd=0.01, steps=400,
    w_rot=1.0, w_trans=1.0, w_depth=1.0, w_anchor=0.2, grad_clip=1.0,
    clips_dir=str(HERE.parent / "clips"),
    # sim half
    p_sim=0.5, sim_stride=2, sim_motion_cap=25.0, w_sim_rot=1.0, w_sim_anchor=0.5,
    log_every=10, save_every=50, out_root=str(HERE.parent / "results"),
)


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
    log = logging.getLogger("mix")
    log.info("config: %s", json.dumps(cfg))

    clips = sorted(d for d in glob.glob(os.path.join(cfg["clips_dir"], "*")) if os.path.isdir(d))
    V, O, S = cfg["views"], cfg["overlap"], cfg["sim_stride"]
    N = V + (V - O)

    sims = []
    for d in sorted(glob.glob(str(SIM / "Seq_*"))):
        ims, poses = load_sequence(d)
        if len(ims) <= V * S:
            continue
        C = np.stack([p[0] for p in poses]); Rw = np.stack([p[1] for p in poses])
        starts = [s0 for s0 in range(len(ims) - V * S)
                  if np.linalg.norm(C[[s0 + (V - 1) * S]] - C[[s0]]) >= 1e-6
                  and max(_geo(Rw[s0 + x * S], Rw[s0 + y * S])
                          for x in range(V) for y in range(x + 1, V)) <= cfg["sim_motion_cap"]]
        if starts:
            sims.append((ims, poses, starts))
    log.info("%d real clips, %d sim sequences (%d usable windows)",
             len(clips), len(sims), sum(len(s[2]) for s in sims))

    torch.cuda.is_bf16_supported = lambda *x, **k: False
    from depth_anything_3.api import DepthAnything3
    api = DepthAnything3.from_pretrained(cfg["model"]).to("cuda").eval()
    for p in api.model.parameters():
        p.requires_grad_(False)
    vit = api.model.da3.backbone.pretrained
    params = inject_lora(vit, r=cfg["rank"], alpha=cfg["alpha"])
    enable_block_checkpointing(vit)
    log.info("LoRA %.2fM params on %d linears", sum(p.numel() for p in params) / 1e6,
             len(params) // 2)

    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["steps"])
    scaler = torch.amp.GradScaler("cuda")

    def fwd(ims, res):
        c, _, _ = api._preprocess_inputs(ims, None, None, res, "upper_bound_resize")
        with torch.autocast("cuda", dtype=torch.float16):
            return api.model(c.to("cuda", non_blocking=True)[None].float(), None, None,
                             export_feat_layers=[], infer_gs=False)

    hist, t0 = [], time.time()
    for step in range(1, cfg["steps"] + 1):
        is_sim = sims and random.random() < cfg["p_sim"]
        set_lora_enabled(api.model, True)
        if is_sim:
            ims, poses, starts = random.choice(sims)
            s0 = random.choice(starts)
            idx = [s0 + k * S for k in range(V)]
            import cv2
            batch = [crop_to_deploy_fov(cv2.cvtColor(cv2.imread(ims[i]), cv2.COLOR_BGR2RGB))
                     for i in idx]
            outP = fwd(batch, cfg["res"])
            Rp, _ = rel_poses(outP["extrinsics"][0].float())
            Rg, _ = rel_poses(gt_w2c(poses, idx, "cuda"))
            l_main = geodesic(Rp, Rg).mean()          # rotation ONLY
            with torch.no_grad():
                set_lora_enabled(api.model, False); ref = fwd(batch, cfg["res"])
                set_lora_enabled(api.model, True)
            Rr, _ = rel_poses(ref["extrinsics"][0].float())
            l_anc = geodesic(Rp, Rr).mean()
            loss = cfg["w_sim_rot"] * l_main + cfg["w_sim_anchor"] * l_anc
            rec = dict(step=step, kind="sim", loss=float(loss),
                       rot_deg=math.degrees(float(l_main)), anchor=float(l_anc))
        else:
            cims = None
            while cims is None:
                cims, _ = load_clip(random.choice(clips), N, cfg["res"])
            A = cims[:V]
            outA, outB = fwd(A, cfg["res"]), fwd(cims[V - O:], cfg["res"])
            l_rot, l_trans, l_depth = consistency(outA, outB, V, O)
            with torch.no_grad():
                set_lora_enabled(api.model, False); ref = fwd(A, cfg["res"])
                set_lora_enabled(api.model, True)
            l_anc, _ = anchor(outA, ref, V)
            loss = (cfg["w_rot"] * l_rot + cfg["w_trans"] * l_trans
                    + cfg["w_depth"] * l_depth + cfg["w_anchor"] * l_anc)
            rec = dict(step=step, kind="real", loss=float(loss), rot=float(l_rot),
                       trans=float(l_trans), depth=float(l_depth), anchor=float(l_anc))

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
        scaler.step(opt); scaler.update(); sched.step()
        hist.append(rec)
        if step % cfg["log_every"] == 0:
            ns = sum(1 for h in hist if h["kind"] == "sim")
            log.info("step %d [%s] loss %.4f  (sim %d / real %d)  %.1fs/step",
                     step, rec["kind"], rec["loss"], ns, len(hist) - ns,
                     (time.time() - t0) / step)
        if step % cfg["save_every"] == 0 or step == cfg["steps"]:
            torch.save(lora_state_dict(api.model), out / f"lora_{step:04d}.pt")
            (out / "training_log.json").write_text(json.dumps(hist))
    log.info("done -> %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
