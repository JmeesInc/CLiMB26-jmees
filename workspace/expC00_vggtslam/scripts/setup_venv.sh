#!/usr/bin/env bash
# Dedicated venv for VGGT-SLAM (torch 2.3.1), isolated from the DA3/eval env.
# Skips SAM3 + Perception Encoder (open-set only). Keeps Salad (imported
# unconditionally by solver.py->loop_closure.py) and gtsam (pose graph).
set -e
EXP=/data4/src/shunsuke/MICCAI2026/CLiMB/workspace/expC00_vggtslam
REPO=$EXP/repo
VENV=$EXP/.venv

cd "$REPO"
echo "== create venv (py3.11) =="
uv venv --python python3.11 "$VENV"
source "$VENV/bin/activate"

echo "== torch 2.3.1 (cu121) =="
uv pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121

echo "== base requirements =="
uv pip install -r requirements.txt

mkdir -p third_party && cd third_party
echo "== Salad (loop-closure place recognition; imported unconditionally) =="
[ -d salad ] || git clone --depth 1 https://github.com/Dominic101/salad.git
uv pip install -e ./salad
echo "== VGGT (MIT-SPARK fork) =="
[ -d vggt ] || git clone --depth 1 https://github.com/MIT-SPARK/VGGT_SPARK.git vggt
uv pip install -e ./vggt
cd ..

echo "== install VGGT-SLAM repo =="
uv pip install -e .
echo "== DONE =="
