#!/usr/bin/env bash
# Render PLY (map+trajectory) for all 6 sim sequences, ORB and VGGT, into figs/.
# Picks, per seq+method, the map dir whose slam_trajectory.ply has the most points.
set -u
REPO=/data4/src/shunsuke/MICCAI2026/CLiMB
RENDER="$REPO/workspace/expC00_vggtslam/scripts/render_ply.py"
FIGS="$REPO/survey/competition/figs"

pick_map() {  # $1 = output root, $2 = seq ; echoes best map dir
  local root="$1" seq="$2" best="" bestn=-1
  for f in $(find "$root/$seq" -name slam_trajectory.ply 2>/dev/null); do
    local n
    n=$(python3 -c "import open3d as o3d;print(len(o3d.io.read_point_cloud('$f').points))" 2>/dev/null)
    [ -z "$n" ] && n=0
    if [ "$n" -gt "$bestn" ]; then bestn=$n; best=$(dirname "$f"); fi
  done
  echo "$best"
}

for seq in Seq_0 Seq_1 Seq_2 Seq_3 Seq_4 Seq_5; do
  for m in "ORB workspace/expA00_baseline_eval/output" "VGGT workspace/expC00_vggtslam/output"; do
    name=${m%% *}; root="$REPO/${m#* }"
    md=$(pick_map "$root" "$seq")
    if [ -n "$md" ]; then
      python3 "$RENDER" --map_dir "$md" --out "$FIGS/ply_${seq}_${name}.png" \
        --title "$seq $name" 2>/dev/null | grep saved
    else
      echo "[skip] $seq $name: no trajectory PLY"
    fi
  done
done
echo "ALL DONE -> $FIGS"
