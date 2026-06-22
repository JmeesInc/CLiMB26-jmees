# CLiMB 2026 — team Jmees26

Solution for the **CLiMB Challenge 2026** (Colonoscopy Localization and Mapping Benchmark,
EndoVis @ MICCAI 2026, Strasbourg): monocular SLAM / VO from colonoscopy `mp4` video,
outputting camera trajectories (`T_wc`) and sparse COLMAP point clouds.

| | |
|---|---|
| Final leaderboard | **rank 2**, `climb_score` **4.078** (1st place 3.559) |
| Synapse submission ID | **9780121** (image alias `v099`) |
| Breakdown | ATE 3.669 mm · RPE_rot@40 5.45° · TFR 81.3 % · 32/32 clips successful · 0.0197 s/frame |
| Write-up | [`writeup/climb2026_jmees26.pdf`](writeup/climb2026_jmees26.pdf) (source in `writeup/`) |

## Method in one paragraph

A feed-forward multi-view foundation model, **Depth Anything 3** (`DA3NESTED-GIANT-LARGE`),
is run on short overlapping windows of keyframes and the windows are chained in closed form:
no bundle adjustment, no loop closure, no tracking state. Everything is deterministic and runs
at ~51 fps on the evaluation host (RTX 5090).

```
mp4 ─► decode every 6th frame ─► kb4 fisheye → virtual pinhole (focal ×1.8)
    ─► DA3 windows (14 frames / 8 overlap, 448 px, fp16, + LoRA)
    ─► rotation-first SE(3) chaining with scale ≡ 1
    ─► SLERP / linear pose interpolation for the skipped frames (TFR stays 100 %)
    ─► ALIKED + LightGlue + MAGSAC++ essential ─► robust SO(3) averaging (orientations only)
    ─► sub-map partition (≥ 275 frames per map) + worst-map rejection (clips with ≥ 4 maps)
    ─► cam_traj_map_*.txt / points3D.txt / runtime.txt × 5 runs
```

The three findings that carried most of the leaderboard gain:

1. **The error is inter-window scale inconsistency, not drift.** BA, pose graphs and loop
   closure do nothing; the *map granularity* (sub-map span) is the right lever.
2. **The virtual field of view matters, not just undistortion.** DA3 self-estimates its focal
   length and, on a 111° rectified image, gets it 1.6–2.5× wrong, shearing rotation into
   translation. Scaling the virtual focal by 1.8 (~72° FOV) cut real-clip ATE by 18 % for free.
3. **Rotations can be replaced wholesale.** The evaluator's ATE uses camera centres only and
   RPE_rot uses orientations only, so robust epipolar rotation averaging is zero-risk for ATE.

The only training we did is a self-supervised **LoRA adapter** (rank 8, 2.95 M parameters,
11.8 MB) on DA3's backbone: two overlapping windows must agree on their shared-view relative
poses and metric depths. See [`workspace/expB09_lora/`](workspace/expB09_lora/).

## Repository layout

```
submit/v010_submaps/   ← the submission tree (v010 … v104 are all built from here, ENV-only diffs)
  Dockerfile, build.sh, make_variant.sh, test.sh, export.sh
  predict.py            entry point (container: python predict.py --input /input --output /output)
  rotavg.py             ALIKED + LightGlue + SO(3) averaging
  calib_table.py        18 official kb4 endoscope calibrations + seq→endoscope map
  lora.py               LoRA injection (no peft dependency)
  lora.pt               shipped adapter (md5 909a4b31…, used by every leaderboard entry incl. v099)
  lora_mix.pt           real + sim-rotation adapter (md5 08a72be0…, used by v017 only)
  results_*.json        local CV results of each variant on the 4 real example clips
submit/v001 … v009/    earlier submission trees, kept for the ablation history
submit/SUBMISSIONS.md  every submission: build args, local CV, leaderboard result
workspace/             experiments (expB01 DA3 windows, expB02 fisheye, expB04 real-clip CV,
                       expB09 LoRA, expB12 rotation averaging, …). Each has SESSION_NOTES.md
workspace/tools/       Synapse leaderboard watcher
writeup/               3-page challenge write-up (LaTeX + PDF)
reference/             official challenge kit (NOT in git — clone from the organisers)
```

Not tracked in git: challenge data (`data/`, `official_data/`), the official `reference/`
kit, base-model weights, vendored upstream sources, and container test outputs.

## Weights: what is baked into the image and where it comes from

The evaluation container runs with `--network=none`, so every weight is copied into the image
at build time by `build.sh`.

| Weight | Size | In this repo? | Source |
|---|---|---|---|
| `submit/v010_submaps/lora.pt` | 11.8 MB | **yes** | trained by us (`workspace/expB09_lora/scripts/train_lora.py`, step 200) |
| `submit/v010_submaps/lora_mix.pt` | 11.8 MB | **yes** | trained by us (`train_lora_mix.py`, step 100); v017 only |
| `depth-anything/DA3NESTED-GIANT-LARGE` | ~6.3 GB | no | `huggingface-cli download depth-anything/DA3NESTED-GIANT-LARGE` |
| ALIKED `aliked-n16.pth`, LightGlue `aliked_lightglue_v0-1_arxiv.pth` | 50 MB | no | downloaded by the `lightglue` package on first use into `~/.cache/torch/hub/checkpoints/` |
| Depth-Anything-3 source | — | no | `git clone https://github.com/ByteDance-Seed/Depth-Anything-3` @ `f64bffe` |
| `lightglue` package source | — | no | `pip install git+https://github.com/cvg/LightGlue` |
| `yyfz233/Pi3X` + Pi3 source | ~2 GB | no | only for the rejected `BACKEND=pi3` variants (v065); not needed for v099 |

DA3 GIANT weights are released under **CC BY-NC 4.0** (research use only); the LoRA adapters
are derived from them and inherit that restriction.

## Reproducing the submitted image (v099)

Prerequisites on the build host: Docker, the DA3 checkout, the HF weights and the `lightglue`
package installed in some Python environment (its two checkpoints appear in the torch hub
cache after one forward pass, or download them from the LightGlue GitHub release).

```bash
cd submit/v010_submaps
export DA3_SRC=/path/to/Depth-Anything-3          # commit f64bffe
export HF_CACHE=$HOME/.cache/huggingface/hub      # holds models--depth-anything--DA3NESTED-GIANT-LARGE
export LIGHTGLUE_SRC=$(python -c 'import lightglue,os;print(os.path.dirname(lightglue.__file__))')
export TORCH_CKPT=$HOME/.cache/torch/hub/checkpoints

# exact build arguments of the submitted image (baked ENV dumped from climb-submaps:v099)
IMAGE=climb-submaps:v099 \
CHUNK=14 OVERLAP=8 STRIDE=6 RES=448 FSCALE=1.8 PRECISION=fp16 LORA=1 LORA_FILE=lora.pt \
MAP_FRAMES=260 MAP_MIN_SPAN=275 MAP_FRAMES_LONG=380 MAP_MIN_SPAN_LONG=285 \
MAP_LONG_BANDS=901-950:380/285 MAP_DROP=inlier MAP_DROP_MIN=4 \
ROTAVG=1 ROT_STEPS=1,2,6,7,8 ROT_WDA3=10 \
bash build.sh
```

`make_variant.sh` wraps the same build plus the real-clip regression and the Synapse tag;
every leaderboard row in `submit/SUBMISSIONS.md` lists its build arguments.

Run it the way the organisers do:

```bash
docker run --rm --gpus all --network=none --user $(id -u):$(id -g) \
  -v /path/to/input:/input:ro -v /path/to/output:/output climb-submaps:v099
```

Output tree: `/output/<seq>/<run 1..5>/{3D_maps/<map>/points3D.txt, camera_trajectory/cam_traj_map_<map>.txt, runtime.txt}`.
Poses are camera-to-world, `frame_id tx ty tz qw qx qy qz`, frame IDs 1-based.

## Local evaluation

- **Simulated sequences (regression test):** `bash submit/v010_submaps/test.sh` runs the
  image on the 6 simulated sequences and scores it with the official evaluator against the
  COLMAP-format ground truth produced by `workspace/expA00_baseline_eval/` (`STAGE=gt`).
- **Real example clips (the CV we made decisions on):** the 4 released clips with COLMAP
  reference (`official_data/Examples`, Seq_001_a/c, Seq_003_a/b), harness in
  `workspace/expB04_realcv/`. When pointing `test.sh` at real clips set `DA3_RECTIFY=1`
  (the default 0 is for the pinhole sim and silently disables rectification, rotation
  averaging and map rejection).
- Decisions were taken from the container itself (`test.sh` with baked ENV), not from the
  exploration scripts, after two false positives caused by configuration drift.

## Data disclosure

LoRA training used 1100 twelve-second clips from 56 train/val EndoMapper sequences of the
official release. The 18 official **test sequences were excluded** from all training, tuning,
model selection and calibration; Seq_001 / Seq_003 were additionally excluded because their
released example clips are our validation set. The only other inputs are the official
calibration release (18 endoscopes) and the sequence→endoscope metadata.

## Acknowledgements / licenses

Depth Anything 3 (ByteDance-Seed, CC BY-NC 4.0), ALIKED, LightGlue (Apache-2.0), the CLiMB
organisers' evaluation kit (`reference/`, ORB-SLAM3 baseline is GPLv3). Our own code in
`submit/` and `workspace/` is provided for research reproduction of the challenge entry.

Authors: Shunsuke Kikuchi, Atsushi Kouno, Ryosuke Goto, Hiroki Matsuzaki (Jmees Inc.).
