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
ENV HF_HUB_CACHE=/opt/weights \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# That arbitrary uid has no home directory, so point every library that wants a
# cache at /tmp (1777). Without this, matplotlib -- imported transitively by
# depth_anything_3.api -- fails at import time trying to build its font cache.
ENV HOME=/tmp \
    XDG_CACHE_HOME=/tmp/.cache \
    MPLCONFIGDIR=/tmp/mpl

COPY predict.py /opt/submission/predict.py
WORKDIR /opt/submission

ENTRYPOINT ["/opt/venv/bin/python3", "/opt/submission/predict.py"]
CMD ["--input", "/input", "--output", "/output"]
