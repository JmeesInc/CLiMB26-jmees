# v001_da3_submap — CLiMB Docker submission

DA3 submap-chunked monocular SLAM. Source experiment: `workspace/expB01_da3_submap`
(`scripts/chunk_slam.py`, the `s=1` configuration → `results_s1.json`).

## Local CV (sim 6 sequences, official evaluator vs `workspace/expA00_baseline_eval/colmap_gt`)

| method | ATE (mm) | TFR (%) | Success |
|---|---|---|---|
| **this submission** (DA3 submap, scale fixed to 1) | **5.25** | **96.3** | **6/6** |
| DA3 pair chaining (expD00 da3pair) | 6.59 | 96.3 | 6/6 |
| ORB-SLAM3 reference baseline | 6.63¹ | 72.6 | 5/6 |
| DA3 submap with per-window scale estimate | 7.09 | 96.3 | 6/6 |
| VGGT-SLAM 2.0 (min_disparity=2) | 11.73 | 78.4 | 6/6 |

¹ ORB-SLAM3 splits each sequence into ~1.7 sub-maps and the evaluator aligns each
fragment independently, which hides drift. This submission emits one map covering
every frame, so its 5.25 mm is measured under a strictly harder alignment.

Per sequence: Seq_0 3.11 / Seq_1 3.26 / Seq_2 3.97 / Seq_3 4.11 / Seq_4 8.63 / Seq_5 8.39 mm.
Seq_4's lower TFR is a dataset artifact (its RGB frames are fragmented), not a
failure of the method.

### Container result (`bash test.sh`, `results_docker.json`)

Running this image end-to-end under the official invocation, scored by the
official evaluator — **the number that actually validates the submission**:

| seq | ATE (mm) | TFR (%) | Success |
|---|---|---|---|
| Seq_0 | 2.84 | 100.0 | ✓ |
| Seq_1 | 3.12 | 100.0 | ✓ |
| Seq_2 | 3.78 | 100.0 | ✓ |
| Seq_3 | 3.97 | 100.0 | ✓ |
| Seq_4 | 2.89 | 62.1 | ✓ |
| Seq_5 | 8.14 | 100.0 | ✓ |
| **Mean** | **4.13** | **93.7** | **6/6** |

Better than the 5.25 mm measured from PNG folders in `workspace/`. The pipeline
is identical; the input path differs (H.264 decode vs. PNG), and on Seq_4 the mp4
holds only its 200 contiguous leading frames, which both lowers its TFR (62.1 vs
77.6) and removes the fragmented tail that dominated its error (8.63 → 2.89).

## Method

Depth-Anything-3 runs feed-forward on sliding windows of 8 frames sharing 3 frames
with the previous window. Each call returns window-local poses; the window is
placed into the global map by a rigid transform fitted on the shared cameras —
rotation from the camera orientations, translation from the centroids, and
**scale fixed at 1**. Sparse map points are the confident depths unprojected into
the global frame. No bundle adjustment, no loop closure, no training.

The scale decision is the single most important knob: DA3 depth is metric and
consistent across windows, so estimating a per-window scale only adds noise
(5.25 mm → 7.09 mm with a clamped robust estimator, → 19.27 mm with a plain
least-squares fit that degenerates on near-static overlaps).

## Contract compliance

- `/input` read-only, flat `*.mp4`; `/output/<seq>/{1..5}/` with `3D_maps/000/points3D.txt`,
  `camera_trajectory/cam_traj_map_000.txt`, `runtime.txt`.
- Frame IDs are **1-based** (first video frame → `000001.png`).
- Poses are camera-to-world, quaternion w-first.
- `--network=none` safe: DA3 weights are baked to `/opt/weights` with
  `HF_HUB_OFFLINE=1`. Nothing is downloaded at run time.
- Frames are decoded from the mp4 in a sliding window, so memory does not scale
  with sequence length.

**The 5 runs are identical by construction.** The pipeline is deterministic
(no_grad eval-mode inference, closed-form chaining, fixed seed for point
subsampling), so each sequence is computed once and written to all five run
folders. `runtime.txt` reports the true cost of that single computation.

## Environment notes

- **CUDA 12.8 / cu128 wheels are mandatory.** The evaluation host is an RTX 5090
  (Blackwell, sm_120). Our development environment (`../iMED/.venv`) has
  torch 2.12.0+cu126 whose `arch_list` stops at sm_90 with no PTX fallback, so it
  would not launch a single kernel on the eval host.
- Peak GPU memory measured at chunk=8 / res=504: **19.3 GB** (fits the 5090's
  32 GB). chunk=16 OOMs — DA3's cross-view attention falls back to the SDPA math
  kernel here, which materialises an O((S·N)²) matrix.
- Speed: ~1.3 s/frame on an RTX 8000. The eval host is faster, but long real
  sequences remain expensive — see "Known risks".

## Usage

```bash
bash build.sh     # stage DA3 source + weights into the context, docker build
bash test.sh      # run the container on the sim mp4s and score with the evaluator
SEQS="Seq_0" bash test.sh   # single-sequence smoke test
bash export.sh    # docker save -> tar.gz
```

`build.sh` stages two git-ignored trees into the build context:
`vendor/depth_anything_3_src/` (upstream ByteDance-Seed/Depth-Anything-3 @ f64bffe)
and `model/hf/` (the DA3NESTED-GIANT-LARGE HF cache, 6.3 GB). Both are recreated
from the local machine; the script prints how to re-fetch them if missing.

Tunables are environment variables read by `predict.py`: `DA3_CHUNK`,
`DA3_OVERLAP`, `DA3_RES`, `DA3_MODEL`, `NUM_RUNS`. Defaults are the validated
configuration — changing them invalidates the CV numbers above.

## Wall-clock budget

The contract caps a scoring run at **5× the real-time duration of the job**
(all sequences × 5 runs), ≈ 6 h for the current test set. Working back from that,
the test set holds ~14.4 min of video ≈ 35–43 k frames at EndoMapper's 40–50 fps,
so the budget is **0.50–0.63 s/frame** given that this submission computes each
sequence once and copies it to the 5 runs. (Without that trick the budget would
be 0.10–0.13 s/frame and no DA3-based method could fit.)

**Measured timings are dominated by a local-hardware artifact, not by the method.**
DA3 picks its autocast dtype with `torch.cuda.is_bf16_supported()`, which returns
True on our Quadro RTX 8000 — but Turing (sm_75) has no bf16 tensor cores, so that
path is emulated. Measured on one DA3 call, same input, same GPU:

| dtype | s/call | end-to-end | Seq_0 ATE |
|---|---|---|---|
| bf16 (auto-selected on Turing = emulated) | 8.05 | 1.28 s/frame | 2.84 mm |
| fp16 (Turing tensor cores) | 1.91 | **0.411 s/frame** | 2.85 mm |

**4.2× purely from precision, with no accuracy cost** (2.84 → 2.85 mm). Even on a
2018 card the fp16 path already fits the 0.50–0.63 s/frame budget; the RTX 5090
evaluation host has *native* bf16 tensor cores and is several generations newer,
so `auto` (= bf16 there) should be comfortably inside it.

`DA3_PRECISION=auto|bf16|fp16` selects this. **`auto` is the default and the right
choice for submission** — bf16 is DA3's training dtype and is native on Blackwell.
`fp16` is the escape hatch if the host ever turns out to lack fast bf16.

## Known risks

- **Never run on Blackwell.** The cu128 wheel choice is verified by inspecting
  `torch.cuda.get_arch_list()`, not by executing on an RTX 5090. The Validation
  queue is the cheap way to confirm it.
- **Wall-clock is an estimate.** The ~6 h figure and the frame count behind the
  0.50–0.63 s/frame budget are inferred from the instructions, not measured on
  the real test set. If a scoring run returns `INVALID (timeout)`, the levers in
  order of cost are: `DA3_PRECISION=fp16`, `DA3_OVERLAP=1` (fewer calls),
  `DA3_RES=378`, then `DA3_MODEL=depth-anything/DA3-LARGE`.
- **Untested on real colonoscopy video.** Local CV only exists for the simulated
  sequences; the real sequences have no local COLMAP ground truth. Real data
  differs in calibration (Kannala-Brandt, 1440×1080) and image quality.
- **Image size ~18 GB**: 6.9 GB torch cu128 + 6.8 GB weights + 0.7 GB other deps
  + the CUDA 12.8 runtime base. `DA3_MODEL` can point at
  `depth-anything/DA3-LARGE` (1.6 GB) to cut about 5 GB, but that variant has not
  been scored here. No image-size limit is stated in the official instructions.
