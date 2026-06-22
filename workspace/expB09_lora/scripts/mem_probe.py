#!/usr/bin/env python3
"""Can we backprop through DA3-giant with LoRA on one 48 GB card?
Two overlapping windows (V views each, sharing V/2) at RES, fp16 autocast,
optional block checkpointing. Prints peak memory and step time."""
import os, sys, time, argparse, numpy as np, torch
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(__file__))
from lora import inject_lora, enable_block_checkpointing
ap = argparse.ArgumentParser(); ap.add_argument("--views", type=int, default=8); ap.add_argument("--res", type=int, default=280)
ap.add_argument("--ckpt", type=int, default=1); ap.add_argument("--r", type=int, default=8); a = ap.parse_args()
torch.cuda.is_bf16_supported = lambda *x, **k: False        # Turing: fp16 tensor cores
from depth_anything_3.api import DepthAnything3
api = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE").to("cuda")
net = api.model.da3; vit = net.backbone.pretrained
for p in api.model.parameters(): p.requires_grad_(False)
params = inject_lora(vit, r=a.r)
if a.ckpt: enable_block_checkpointing(vit)
print(f"LoRA params: {sum(p.numel() for p in params)/1e6:.2f}M over {len(params)//2} linears; blocks={len(vit.blocks)}")
opt = torch.optim.AdamW(params, lr=1e-4)
H = int(round(a.res * 1080 / 1440 / 14)) * 14; W = a.res
imgs = torch.rand(1, a.views, 3, H, W, device="cuda")
api.model.train(False)
torch.cuda.reset_peak_memory_stats(); t0 = time.time()
with torch.autocast("cuda", dtype=torch.float16):
    outA = net(imgs); outB = net(imgs.flip(1))
    loss = (outA["depth"][:, a.views//2:] - outB["depth"][:, :a.views//2].flip(1)).abs().mean() \
         + (outA["extrinsics"] - outB["extrinsics"].flip(1)).abs().mean()
loss.float().backward(); opt.step(); torch.cuda.synchronize()
print(f"views={a.views} res={W}x{H} ckpt={a.ckpt}: loss {loss.item():.4f}  peak {torch.cuda.max_memory_allocated()/2**30:.1f} GB  step {time.time()-t0:.1f}s")
