# expA00_baseline_eval — ORB-SLAM3 ベースライン評価 (シミュ系列)

## 目的
公式 ORB-SLAM3 ベースラインをローカルで定量評価し、評価パイプライン
(GT変換→SLAM実行→公式評価器) を確立する。実系列にはローカル COLMAP GT が
無いため、GT を持つ Simulated_Sequences 6本を評価対象とする。

## データの前提（重要）
- **実系列 (93本)** にはローカル COLMAP GT 無し → 手元で定量評価不可
- **シミュ系列 (Seq_0..5)** は `trajectory.csv` に GT あり → 評価可能
- シミュ6本は **同一カメラ軌跡・同一内視鏡(pinhole 960×720)**、**変形(deformation)
  パラメータのみ違う**レンダリング違い:
  | seq | Amplitude[mm] | omega | 性質 |
  |---|---|---|---|
  | Seq_0 | 0 | 0 | 剛体 |
  | Seq_1 | 2.5 | 2.5 | |
  | Seq_2 | 2.5 | 5 | |
  | Seq_3 | 5 | 2.5 | |
  | Seq_4 | 5 | 5 | (rgb断片化, mp4=200フレーム) |
  | Seq_5 | 10 | 5 | 最大変形 |
  → **固定軌跡下での組織変形に対する SLAM ロバスト性**を測るablationになる

## パイプライン (`run.sh`, STAGE=mp4|gt|slam|eval|all)
1. **mp4**: `rgb/image_%04d.png` → `sim_input/<seq>.mp4` (ffmpeg, libx264 -qp0 lossless, 30fps)
2. **gt**: `scripts/sim_gt_to_colmap.py` が `trajectory.csv` → COLMAP参照ツリー
   - `colmap_gt/<seq>/results_txt/{images.txt, points3D.txt}` + `scales.csv` + `traj_lengths_mm.csv` + `endomapper_short_seq_frames.csv`
3. **slam**: ORB-SLAM3 docker を sim用 PinHole yaml で 5ラン×6系列実行 → `output/<seq>/<run>/`
4. **eval**: `reference/evaluation/slam_evaluation.py` → `results.json`

## 確定した技術仕様（ハマりどころ）
- **座標規約**: sim `trajectory.csv` = TUM形式 camera-to-world。
  `(tX,tY,tZ)`=世界座標カメラ中心 C_w、quat `(rX,rY,rZ,rW)`=xyzw=R_wc。
  COLMAP images.txt は world-to-camera (R_cw,t_cw) なので変換: R_cw=R_wc.T,
  t_cw=-R_cw·C_w, quat は w-first で出力。
  → **検証OK**: ORB-SLAM3 Seq_0 で ATE≈1.08mm（規約が誤りなら桁違いになる）
- **単位**: sim GT は dm。`scales.csv` の `scale_to_target=100` で dm→mm。ATEはmm単位で出る
- **フレームID**: 1-based。trajectory.csv row k(0始) ↔ image_{k:04d}.png ↔ frame_id k+1。
  ORB-SLAM3 も最初のフレームを ID=1 で出力（OpenCV CAP_PROP_POS_FRAMES が read後に+1される）→ 整合
- **COLMAP images.txt は2行/画像**。評価器は `i+=2` で2行目(points2D)をスキップ → 空行を必ず書く
- **docker GPU**: このホストは docker default-runtime=nvidia だが nvidia-container-runtime
  バイナリ欠如で全コンテナ起動失敗。**ORB-SLAM3 monocular はCPUのみ**なので
  `--runtime=runc` を明示してバイパス（run.sh に実装済み）
- **評価器の依存**: open3d 必須（PLY出力）。`pip install open3d` 済 (0.19.0)
- **100フレーム閾値**: COLMAP参照と100 frame ID以上共通のサブマップのみ評価対象。
  ORB-SLAM3はトラッキング途切れで複数サブマップを作る（map0=55<100は除外, map1=124は採用）

## 結果 (5ラン×6系列, `results.json`)
| seq | A/ω | ATE(mm) | TFR(%) | RPE_r(δ40°) | Success |
|---|---|---|---|---|---|
| Seq_0 | 0/0 | 1.11 | 71.7 | 20.8 | ✓ |
| Seq_1 | 2.5/2.5 | 5.47 | 87.9 | 18.6 | ✓ |
| Seq_2 | 2.5/5 | 3.93 | 79.5 | 20.9 | ✓ |
| Seq_3 | 5/2.5 | 8.06 | 74.5 | 20.9 | ✓ |
| Seq_4 | 5/5 | 5.00 | 38.5 | 18.8 | ✗ (rgb断片化) |
| Seq_5 | 10/5 | 16.21 | 83.2 | 22.6 | ✓ |
| **Mean** | | **6.63** | **72.6** | 20.5 | **5/6** |

→ 変形量↑でATE↑（剛体1.11mm→最大変形16.21mm）。変形ロバスト性が本質的課題と定量確認。
→ smoke(Seq_0 1run)は ATE1.08mm/TFR35% だったが、5ラン集約・複数サブマップ統合で
  TFRは71.7%に上昇（評価器がサブマップを統合採点するため）。

## 既知の注意点 / TODO
- Seq_4 は rgb 番号が断片的(0-199,262-299,310-333)。mp4は連番先頭200フレームのみ → GT 1..200 と整合（>100なので評価可）
- quat規約は ATE で検証済だが RPE回転 の絶対値は規約依存。c2w/w2c の取り違えがあっても
  ATEは中心座標のみ使うので頑健（要・公式評価器との突合）
- sim と本番(実系列)は別キャリブ・別画質。sim の良スコアが実系列に直結しない点に注意
