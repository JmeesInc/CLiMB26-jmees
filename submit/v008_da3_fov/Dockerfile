# CLiMB submission: DA3 submap-chunked monocular SLAM (workspace/expB01_da3_submap).
#
# CUDA 12.8 base and cu128 wheels are REQUIRED, not cosmetic: the evaluation host
# is an RTX 5090 (Blackwell, sm_120) and the cu126 build we develop against ships
# arch_list up to sm_90 with no PTX, so it cannot launch a single kernel there.
# Ubuntu 24.04 for python 3.12: it matches the environment the CV was measured
# in, and the pinned numpy/scipy wheels require >=3.11 (22.04 ships 3.10).
FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    NVIDIA_DRIVER_CAPABILITIES=all \
    NVIDIA_VISIBLE_DEVICES=all \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        python3 \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

# A venv, because 24.04's system python is PEP 668 externally-managed.
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# torch first, from the cu128 index (the default PyPI wheel is cu126-based).
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.8.0 torchvision==0.23.0

COPY requirements.txt /opt/submission/requirements.txt
RUN pip install --no-cache-dir -r /opt/submission/requirements.txt

# Depth-Anything-3 source (ByteDance-Seed/Depth-Anything-3 @ f64bffe), installed
# without deps so it cannot pull a different torch over the cu128 one above.
COPY vendor/depth_anything_3_src /opt/da3
RUN pip install --no-cache-dir --no-deps /opt/da3

# Weights baked in: evaluation runs with --network=none, so nothing may download.
# --chmod (not a separate RUN chmod) because the container runs as
# `--user $(id -u):$(id -g)`, an arbitrary uid that still has to take
# huggingface_hub's cache lock -- and a chmod in its own layer would duplicate
# the whole 6.8 GB tree.
COPY --chmod=777 model/hf /opt/weights
# v004: the LoRA adapter was trained under fp16 autocast; under bf16 (DA3's
# default whenever the GPU reports bf16 support, i.e. on the RTX 5090) the
# adapted model degrades badly (sim 6-seq: 5.25 mm fp16 -> 8.87 mm bf16). Pin
# the inference dtype to what the adapter was trained with.
ENV DA3_PRECISION=fp16
# v006: rotation refinement (ALIKED+LightGlue epipolar edges + SO(3) averaging)
# v008: virtual-pinhole FOV correction. balance=0 rectification keeps the
# endoscope's full ~111 deg cone; DA3 self-estimates a focal 1.6-2.5x the true
# one (it believes ~71 deg), and a wrong focal shears rotation into translation.
# fscale 1.8 puts the true focal where DA3 expects it. Real CV 4.08 -> 3.35 mm.
ENV DA3_RECT_FSCALE=1.8
# Rotation averaging stays OFF: it costs +0.006 s/f, and at this ATE the score is
# already clear of the 5% tie-break band where score_rot would matter (v006/v007
# both lost more to W_t than they gained in rot).
ENV DA3_ROTAVG=0 DA3_ROT_STRIDE=6 DA3_ROT_KP=512 DA3_ROT_STEPS=1,4,7,10,13,16 DA3_ROT_SCALE=0.25 DA3_ROT_WDA3=10 DA3_ROT_CODEC=raw DA3_ROT_FP16=0 DA3_ROT_ADAPTIVE=0
ENV TORCH_HOME=/opt/torchhub
ENV HF_HUB_CACHE=/opt/weights \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# That arbitrary uid has no home directory, so point every library that wants a
# cache at /tmp (1777). Without this, matplotlib -- imported transitively by
# depth_anything_3.api -- fails at import time trying to build its font cache.
ENV HOME=/tmp \
    XDG_CACHE_HOME=/tmp/.cache \
    MPLCONFIGDIR=/tmp/mpl

COPY predict.py calib_table.py lora.py lora.pt rotavg.py /opt/submission/
COPY vendor/lightglue /opt/venv/lib/python3.12/site-packages/lightglue
COPY torchhub /opt/torchhub
WORKDIR /opt/submission

ENTRYPOINT ["/opt/venv/bin/python3", "/opt/submission/predict.py"]
CMD ["--input", "/input", "--output", "/output"]
