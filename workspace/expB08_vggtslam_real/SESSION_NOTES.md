# expB08_vggtslam_real — VGGT-SLAM（最適化ベースのサブマップ SLAM）の実データ CV

## 目的
DA3 連鎖の弱点は「greedy 連結による窓間スケール不整合」。VGGT-SLAM は重複フレーム上で
Sim(3)/SL(4) 姿勢グラフ最適化を行う = 連結を最適化で置き換えた別アーキ。6 月 sim では 11.7mm（md2）。

## 手順
- `extract_frames.py`: rectify 済み stride-6 キーフレームを `{id:06d}.png`（1-based）で書き出し
- `expC00` の venv/repo で `main.py --submap_size 16 --max_loops 0 --min_disparity 0`（GPU3、0.97 s/keyframe）
- `interp_eval.py`: キーフレーム姿勢を連鎖と同じ SLERP/線形で全フレーム化（keyframe のみだと共通 ID<100 で 3/4 が評価外）
- 落とし穴: `main.py` はリポジトリ内で実行（cd 必須）／poses.txt はサブマップ重複で同一 ID が重複 → 除去してソート

## 結果（`results_vggtslam_interp.json`）
| seq | VGGT-SLAM | 連鎖 v003 | ORB-SLAM3 |
|---|---|---|---|
| Seq_001_a | 11.70 | **6.85** | 10.22 |
| Seq_001_c | 5.47 | **2.49** | ✗ |
| Seq_003_a | 8.17 | **3.45** | 8.85 |
| Seq_003_b | 8.81 | **4.70** | ✗ |
| **Mean** | 8.54 / rot 7.22 | **4.374 / 5.18** | 9.53 (2/4) |
速度: 0.97 s/keyframe = 0.16 s/frame @RTX8000 → 評価機 ~0.04 → W_t 1.5

## 判断
- **不採用**。4/4 クリップで連鎖に 2 倍前後劣り、速度も W_t 崖越え
- 最適化ベースの連結でも DA3 連鎖に届かない = 問題は「連結方法」より「各窓の推定品質」側にある
