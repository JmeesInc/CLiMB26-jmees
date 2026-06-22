#!/usr/bin/env python3
"""Self-supervised LoRA adaptation of DA3 (nested giant) to colonoscopy.

Why: the deployed chain assumes s=1 between consecutive windows, but on real
data DA3's per-window metric geometry is context-dependent -- the SAME frames
get different depth/relative poses depending on which other frames share the
window (daily 8/30 §8: a 3-keyframe phase shift swings Seq_001_c 2.49 -> 6.08).
No local signal predicts it (§14.2); optimisation/one-shot/conditioning do not
fix it (§9). So we train the backbone to be context-stable, without labels:

  two overlapping windows A=[k,k+12), B=[k+6,k+18) share 6 keyframes ->
    L_pose  : relative (R,t) between shared views must agree (t in METRIC units,
              i.e. exactly the s=1 assumption the chain makes)
    L_depth : metric depth of a shared frame must agree between the windows
    L_anchor: stay close to the frozen model on window A (no collapse)

LoRA on attn.qkv/proj of the any-view DINOv2-giant backbone only; the monocular
metric branch, heads and camera decoder stay frozen.  Data: random 12-s clips
of TRAINVAL real sequences (test split and Seq_001/003 excluded; see prep_clips.py).
"""
import os, sys, glob, json, time, math, random, argparse, logging
from pathlib import Path
import numpy as np, torch, yaml, cv2

os.environ.setdefault("HF_HUB_OFFLINE", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lora import inject_lora, enable_block_checkpointing, lora_state_dict, LoRALinear

CFG = dict(
    experiment_name="lora_r8_consist", seed=0, model="depth-anything/DA3NESTED-GIANT-LARGE",
    views=12, overlap=6, res=504, rank=8, alpha=16, lr=1e-4, wd=0.01, steps=1500,
    w_rot=1.0, w_trans=1.0, w_depth=1.0, w_anchor=0.2, grad_clip=1.0, log_every=10, save_every=100,
    w_anc_rot=0.0,   # extra anchor on relative ROTATIONS vs frozen model (rotation degraded in run C)
    w_lg=0.0,        # distillation onto LightGlue+essential relative rotations (see below)
    lg_min_inl=40,   # only trust teacher edges with this many inliers
    lg_huber_deg=15.0,   # cap each edge's pull so a wrong teacher edge cannot dominate
    clips_dir=str(HERE.parent / "clips"), out_root=str(HERE.parent / "results"),
    overfit=0,   # 1 = always the same clip+offset (gradient-path sanity check)
)


def set_lora_enabled(model, on: bool):
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m._on = on
    LoRALinear.forward = _fwd_on if on else _fwd_off


def _fwd_on(self, x):
    return self.base(x) + (self.drop(x) @ self.A.t() @ self.B.t()) * self.scale


def _fwd_off(self, x):
    return self.base(x)


def load_clip(d, n_frames, res, fixed=False):
    files = sorted(glob.glob(os.path.join(d, "*.jpg")))
    if len(files) < n_frames:
        return None, 0
    k = 0 if fixed else random.randrange(0, len(files) - n_frames + 1)
    ims = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in files[k:k + n_frames]]
    return ims, k


_LG_CACHE = {}


def lg_edges(clip_dir):
    """Teacher: {(frame_i, frame_j): (R_ij, n_inliers)} from prep_lg_rot.py."""
    if clip_dir not in _LG_CACHE:
        f = os.path.join(clip_dir, "lg_rot.npz")
        if not os.path.exists(f):
            # Do NOT memoize the miss: prep_lg_rot.py may still be filling these
            # in while training runs, and a cached None would be permanent.
            return None
        else:
            z = np.load(f)
            _LG_CACHE[clip_dir] = {(int(i), int(j)): (R, int(n))
                                   for i, j, R, n in zip(z["i"], z["j"], z["R"], z["ninl"])}
    return _LG_CACHE[clip_dir]


def lg_target(edges, k0, V, min_inl, device):
    """Align the teacher with rel_poses' pair ordering (i<j, i-major).

    Returns (target R, per-pair weight, index) or None when the window has no
    trustworthy edge -- sparse supervision is expected and fine.
    """
    if not edges:
        return None
    idx, Rs, w = [], [], []
    p = 0
    for i in range(V):
        for j in range(i + 1, V):
            e = edges.get((k0 + i, k0 + j))
            if e is not None and e[1] >= min_inl:
                idx.append(p); Rs.append(e[0]); w.append(float(e[1]))
            p += 1
    if not idx:
        return None
    R = torch.from_numpy(np.stack(Rs)).float().to(device)
    ww = torch.tensor(w, device=device)
    return R, ww / ww.sum(), torch.tensor(idx, device=device, dtype=torch.long)


def rel_poses(E):
    """E: (N,3,4) or (N,4,4) w2c -> relative (R_ij, t_ij) for i<j as tensors (P,3,3),(P,3)."""
    N = E.shape[0]
    if E.shape[-2] == 3:
        bottom = torch.tensor([0., 0., 0., 1.], device=E.device, dtype=E.dtype).expand(N, 1, 4)
        E = torch.cat([E, bottom], dim=1)
    Ei = torch.linalg.inv(E)
    Rs, ts = [], []
    for i in range(N):
        for j in range(i + 1, N):
            T = E[j] @ Ei[i]
            Rs.append(T[:3, :3]); ts.append(T[:3, 3])
    return torch.stack(Rs), torch.stack(ts)


def geodesic(Ra, Rb):
    cos = ((Ra.transpose(-1, -2) @ Rb).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    return torch.arccos(cos.clamp(-1 + 1e-6, 1 - 1e-6))


def consistency(outA, outB, V, O):
    """Shared views are A[V-O:] and B[:O] (window B starts O frames later)."""
    EA = outA["extrinsics"][0, V - O:].float()
    EB = outB["extrinsics"][0, :O].float()
    RA, tA = rel_poses(EA); RB, tB = rel_poses(EB)
    l_rot = geodesic(RA, RB).mean()
    l_trans = ((tA - tB).norm(dim=-1) / (tA.norm(dim=-1) + tB.norm(dim=-1) + 1e-6)).mean()
    DA = outA["depth"][0, V - O:].float(); DB = outB["depth"][0, :O].float()
    cA = outA["depth_conf"][0, V - O:].float(); cB = outB["depth_conf"][0, :O].float()
    m = (cA > cA.median()) & (cB > cB.median()) & (DA > 0) & (DB > 0)
    l_depth = ((DA - DB).abs() / (DA + DB + 1e-6))[m].mean() if m.any() else DA.sum() * 0
    return l_rot, l_trans, l_depth


def anchor(out, ref, V):
    """Returns (anchor_total, anchor_rot): total is the original mixed term;
    anchor_rot (relative rotations vs frozen) can be weighted separately."""
    E, Er = out["extrinsics"][0].float(), ref["extrinsics"][0].float()
    R, t = rel_poses(E); Rr, tr = rel_poses(Er)
    a_rot = geodesic(R, Rr).mean()
    l_pose = a_rot + ((t - tr).norm(dim=-1) / (tr.norm(dim=-1) + 1e-6)).mean()
    D, Dr = out["depth"][0].float(), ref["depth"][0].float()
    l_depth = ((D - Dr).abs() / (D + Dr + 1e-6)).mean()
    return l_pose + l_depth, a_rot


def main():
    ap = argparse.ArgumentParser()
    for k, v in CFG.items():
        ap.add_argument(f"--{k}", type=type(v), default=v)
    a = ap.parse_args(); cfg = vars(a)
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])

    out = Path(cfg["out_root"]) / cfg["experiment_name"]
    i = 1
    while out.exists():
        out = Path(cfg["out_root"]) / f"{cfg['experiment_name']}_{i:03d}"; i += 1
    out.mkdir(parents=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg))
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.FileHandler(out / f"train_{time.strftime('%Y%m%d_%H%M%S')}.log"),
                                  logging.StreamHandler(sys.stdout)])
    log = logging.getLogger("lora"); logging.getLogger().handlers[1].setLevel(logging.INFO)
    log.info("config: %s", json.dumps(cfg))

    torch.cuda.is_bf16_supported = lambda *x, **k: False       # Turing: real fp16 tensor cores
    from depth_anything_3.api import DepthAnything3
    api = DepthAnything3.from_pretrained(cfg["model"]).to("cuda")
    api.model.eval()
    for p in api.model.parameters():
        p.requires_grad_(False)
    vit = api.model.da3.backbone.pretrained
    params = inject_lora(vit, r=cfg["rank"], alpha=cfg["alpha"])
    enable_block_checkpointing(vit)
    log.info("LoRA params %.2fM on %d linears", sum(p.numel() for p in params) / 1e6, len(params) // 2)
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["steps"])
    scaler = torch.amp.GradScaler("cuda")

    clips = sorted(d for d in glob.glob(os.path.join(cfg["clips_dir"], "*")) if os.path.isdir(d))
    log.info("%d clips", len(clips))
    V, O = cfg["views"], cfg["overlap"]
    N = V + (V - O)

    def fwd(ims):
        imgs_cpu, _, _ = api._preprocess_inputs(ims, None, None, cfg["res"], "upper_bound_resize")
        imgs = imgs_cpu.to("cuda", non_blocking=True)[None].float()
        with torch.autocast("cuda", dtype=torch.float16):
            return api.model(imgs, None, None, export_feat_layers=[], infer_gs=False)

    hist, t0 = [], time.time()
    lg_seen, lg_gap = 0, []
    for step in range(1, cfg["steps"] + 1):
        ims, cd, k0 = None, None, 0
        while ims is None:
            cd = clips[0] if cfg["overfit"] else random.choice(clips)
            ims, k0 = load_clip(cd, N, cfg["res"], fixed=bool(cfg["overfit"]))
        A, B = ims[:V], ims[V - O:]
        set_lora_enabled(api.model, True)
        outA = fwd(A); outB = fwd(B)
        l_rot, l_trans, l_depth = consistency(outA, outB, V, O)
        with torch.no_grad():
            set_lora_enabled(api.model, False)
            ref = fwd(A)
            set_lora_enabled(api.model, True)
        l_anc, a_rot = anchor(outA, ref, V)

        # Distillation onto the LightGlue teacher. This is the term that breaks
        # the degeneracy of the consistency loss: consistency is happy with any
        # smooth, input-insensitive rotation, whereas the teacher is an actual
        # measurement of THIS pair's relative rotation. Huber-capped so a wrong
        # teacher edge (matching does fail on low-texture mucosa) cannot dominate.
        l_lg = torch.zeros((), device="cuda")
        if cfg["w_lg"] > 0:
            tgt = lg_target(lg_edges(cd), k0, V, cfg["lg_min_inl"], "cuda")
            if tgt is not None:
                Rt, wt, ix = tgt
                R_pred, _ = rel_poses(outA["extrinsics"][0].float())
                g = geodesic(R_pred[ix], Rt)
                cap = math.radians(cfg["lg_huber_deg"])
                l_lg = (wt * torch.where(g <= cap, g, cap * (2 * g / cap - 1).sqrt())).sum()
                lg_seen += 1
                lg_gap.append(float(g.mean()) * 180 / math.pi)

        loss = (cfg["w_rot"] * l_rot + cfg["w_trans"] * l_trans + cfg["w_depth"] * l_depth
                + cfg["w_anchor"] * l_anc + cfg["w_anc_rot"] * a_rot + cfg["w_lg"] * l_lg)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
        scaler.step(opt); scaler.update(); sched.step()
        rec = dict(step=step, loss=loss.item(), rot=l_rot.item(), trans=l_trans.item(),
                   depth=float(l_depth), anchor=l_anc.item(), lg=float(l_lg),
                   lr=sched.get_last_lr()[0])
        hist.append(rec)
        if step % cfg["log_every"] == 0:
            log.info("step %d loss %.4f rot %.4f trans %.4f depth %.4f anchor %.4f "
                     "lg %.4f (%d/%d steps, mean gap %.1f deg)  %.1fs/step",
                     step, rec["loss"], rec["rot"], rec["trans"], rec["depth"], rec["anchor"],
                     rec["lg"], lg_seen, step,
                     float(np.mean(lg_gap[-100:])) if lg_gap else float("nan"),
                     (time.time() - t0) / step)
        if step % cfg["save_every"] == 0 or step == cfg["steps"]:
            # Keep every checkpoint, not just the last: this objective peaks and
            # then degrades (run C: 4.249 @100, 4.075 @200, 4.486 @300), so the
            # step that wins is not known until they are all scored. Overwriting
            # one file cost us the whole first f18 run.
            torch.save(lora_state_dict(api.model), out / f"lora_{step:04d}.pt")
            torch.save(lora_state_dict(api.model), out / "lora_last.pt")
            (out / "training_log.json").write_text(json.dumps(hist))
    log.info("done -> %s", out)


if __name__ == "__main__":
    main()
