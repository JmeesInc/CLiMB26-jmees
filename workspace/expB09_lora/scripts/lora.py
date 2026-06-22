"""Minimal LoRA for DA3's any-view backbone (no peft dependency).

Targets the attention qkv/proj Linear layers of the DINOv2-giant backbone
(model.model.da3.backbone.pretrained.blocks[*].attn.{qkv,proj}). Everything
else stays frozen, including the monocular metric branch and the heads.
"""
import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r, self.scale = r, alpha / r
        dev, dt = base.weight.device, base.weight.dtype
        self.A = nn.Parameter(torch.zeros(r, base.in_features, device=dev, dtype=dt))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=dev, dtype=dt))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.base(x) + (self.drop(x) @ self.A.t() @ self.B.t()) * self.scale


def inject_lora(vit, r=8, alpha=16, targets=("qkv", "proj"), block_range=None):
    """Wrap attn.{qkv,proj} in each transformer block. Returns list of LoRA params."""
    params = []
    blocks = list(vit.blocks)
    idx = range(len(blocks)) if block_range is None else block_range
    for i in idx:
        attn = blocks[i].attn
        for name in targets:
            base = getattr(attn, name)
            if isinstance(base, nn.Linear):
                lora = LoRALinear(base, r=r, alpha=alpha)
                setattr(attn, name, lora)
                params += [lora.A, lora.B]
    return params


def enable_block_checkpointing(vit):
    """Wrap each block's forward in torch.utils.checkpoint (activation recompute)."""
    for blk in vit.blocks:
        if getattr(blk, "_ckpt_wrapped", False):
            continue
        orig = blk.forward

        def fwd(*args, _orig=orig, **kw):
            if torch.is_grad_enabled():
                return ckpt.checkpoint(lambda *a: _orig(*a, **kw), *args, use_reentrant=False)
            return _orig(*args, **kw)
        blk.forward = fwd
        blk._ckpt_wrapped = True


def lora_state_dict(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if ".A" in k or ".B" in k}
