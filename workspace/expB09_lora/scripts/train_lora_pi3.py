#!/usr/bin/env python3
"""Self-supervised LoRA for the Pi3X backbone.

Same objective that produced v004 on DA3 (cross-window consistency on the frames
two overlapping windows share, held near the frozen model by an anchor term):
CV -6.8% translated into LB -16.2%, the single largest gain we have had. Pi3X
uses `dinov2_vitl14_reg` as its encoder, i.e. the same blocks[i].attn.{qkv,proj}
layout the existing injector targets, so the adapter transfers unchanged even
though the trained weights do not.

Two differences from the DA3 version:
  * clips come from `clips/` (the fscale-1.0 cut), not `clips_f18/` -- Pi3X takes
    the true intrinsics directly, so there is no reason to feed it a narrowed
    virtual pinhole. The FOV workaround exists only because DA3 ignores the
    intrinsics argument.
  * the run-C lesson stands: this objective peaks and then degrades
    (4.249 @100, 4.075 @200, 4.486 @300), so every checkpoint is kept.
"""
import argparse
import glob
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

os.environ.setdefault("HF_HUB_OFFLINE", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, os.environ.get("PI3_SRC", "/data4/src/shunsuke/Pi3"))
from lora import inject_lora, enable_block_checkpointing, lora_state_dict, LoRALinear  # noqa: E402
from train_lora import consistency, anchor, load_clip  # noqa: E402

CFG = dict(
    experiment_name="pi3x_lora", seed=0, model="yyfz233/Pi3X",
    views=12, overlap=6, rank=8, alpha=16, lr=5e-4, wd=0.01, steps=400,
    w_rot=1.0, w_trans=1.0, w_depth=1.0, w_anchor=0.2, grad_clip=1.0,
    log_every=10, save_every=50, pixel_limit=255000,
    clips_dir=str(HERE.parent / "clips"), out_root=str(HERE.parent / "results"),
)
PATCH = 14


def set_lora_enabled(model, on):
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.enabled = on


def target_hw(h, w, limit):
    import math
    s = math.sqrt(limit / max(h * w, 1))
    k, m = max(1, round(w * s / PATCH)), max(1, round(h * s / PATCH))
    while (k * PATCH) * (m * PATCH) > limit and (k > 1 or m > 1):
        if (k / max(m, 1)) > (w / max(h, 1)):
            k -= 1
        else:
            m -= 1
    return m * PATCH, k * PATCH


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
    log = logging.getLogger("pi3lora")
    log.info("config: %s", json.dumps(cfg))

    torch.cuda.is_bf16_supported = lambda *x, **k: False      # Turing: real fp16 tensor cores
    from pi3.models.pi3x import Pi3X
    model = Pi3X.from_pretrained(cfg["model"]).to("cuda").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    params = inject_lora(model.encoder, r=cfg["rank"], alpha=cfg["alpha"])
    # Pi3X carries a second ViT (depth_encoder = deepcopy(encoder)) plus decoder
    # blocks, so checkpointing only the main encoder leaves most of the
    # activation memory in place -- it OOMs at 47 GB otherwise.
    for name in ("encoder", "depth_encoder", "decoder"):
        sub = getattr(model, name, None)
        if sub is not None and hasattr(sub, "blocks"):
            enable_block_checkpointing(sub)
    log.info("LoRA %.2fM params on %d linears", sum(p.numel() for p in params) / 1e6, len(params) // 2)

    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["steps"])
    scaler = torch.amp.GradScaler("cuda")

    clips = sorted(d for d in glob.glob(os.path.join(cfg["clips_dir"], "*")) if os.path.isdir(d))
    log.info("%d clips", len(clips))
    V, O = cfg["views"], cfg["overlap"]
    N = V + (V - O)

    def fwd(ims, K):
        h0, w0 = ims[0].shape[:2]
        th, tw = target_hw(h0, w0, cfg["pixel_limit"])
        batch = np.stack([cv2.resize(f, (tw, th), interpolation=cv2.INTER_AREA) for f in ims])
        t = torch.from_numpy(batch).permute(0, 3, 1, 2).float().div(255).to("cuda")
        Ks = np.tile(np.asarray(K, np.float32)[None], (len(ims), 1, 1)).copy()
        Ks[:, 0, :] *= tw / float(w0)
        Ks[:, 1, :] *= th / float(h0)
        kt = torch.from_numpy(Ks).to("cuda")[None]
        with torch.autocast("cuda", dtype=torch.float16):
            res = model(t[None], intrinsics=kt)
        c2w = res["camera_poses"][0].float()
        lp = res["local_points"][0].float()
        return {"extrinsics": torch.linalg.inv(c2w)[None],
                "depth": lp[..., 2][None],
                "depth_conf": torch.sigmoid(res["conf"][0, ..., 0]).float()[None]}

    hist, t0 = [], time.time()
    for step in range(1, cfg["steps"] + 1):
        ims, cd = None, None
        while ims is None:
            cd = random.choice(clips)
            ims, _ = load_clip(cd, N, None)
        K = np.asarray(json.loads(Path(cd, "meta.json").read_text())["K"], np.float64)
        K = K / K[2, 2]                        # prep_clips scaled the whole matrix
        A = ims[:V]

        set_lora_enabled(model, True)
        outA, outB = fwd(A, K), fwd(ims[V - O:], K)
        l_rot, l_trans, l_depth = consistency(outA, outB, V, O)
        with torch.no_grad():
            set_lora_enabled(model, False)
            ref = fwd(A, K)
            set_lora_enabled(model, True)
        l_anc, _ = anchor(outA, ref, V)
        loss = (cfg["w_rot"] * l_rot + cfg["w_trans"] * l_trans
                + cfg["w_depth"] * l_depth + cfg["w_anchor"] * l_anc)

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
        scaler.step(opt); scaler.update(); sched.step()

        hist.append(dict(step=step, loss=float(loss), rot=float(l_rot),
                         trans=float(l_trans), depth=float(l_depth), anchor=float(l_anc)))
        if step % cfg["log_every"] == 0:
            r = hist[-1]
            log.info("step %d loss %.4f rot %.4f trans %.4f depth %.4f anchor %.4f  %.1fs/step",
                     step, r["loss"], r["rot"], r["trans"], r["depth"], r["anchor"],
                     (time.time() - t0) / step)
        if step % cfg["save_every"] == 0 or step == cfg["steps"]:
            torch.save(lora_state_dict(model), out / f"lora_{step:04d}.pt")
            (out / "training_log.json").write_text(json.dumps(hist))
    log.info("done -> %s", out)


if __name__ == "__main__":
    main()
