# expC00_vggtslam — VGGT-SLAM ベースライン評価 (sim系列)

## 動機
expB00 で DA3 を全フレーム一括投入したところ、全対全 cross-view attention が
O((S·N)²) でメモリ爆発し、本env(xformers効率attention不可/SDPA mathカーネル)では
低解像度(res182)に縛られ ATE 35mm 止まり。**submap分割**が必須と判明 →
VGGT-SLAM (MIT-SPARK) はまさに submap 分割 + 射影(SL(4))整列 + loop closure で
長尺・省メモリを両立する学習系SLAM。

## セットアップ (`scripts/setup_venv.sh`, 専用venv)
- torch 2.3.1+cu121 の専用 venv (`.venv`, py3.11) でDA3/評価envと分離
- 公式 repo (VGGT-SLAM 2.0, default branch) を `repo/` に clone
- third_party: Salad (loop closure用; solver初期化で**無条件ロード**), VGGT(MIT-SPARK fork)
- SAM3 / Perception Encoder は open-set検出(--run_os)専用なので**省略**
- VGGT-1B 重み: HFキャッシュ済(facebook/VGGT-1B, 4.7G)
- Salad checkpoint `dino_salad.ckpt`(352MB, GDrive) → `~/.cache/torch/hub/checkpoints/`
  ※ run_predictions が submap embedding計算で常時使うため max_loops=0 でも必須

## フレームID対応 (ハマりどころ)
- VGGT-SLAM の frame_id = **画像ファイル名の数字部分** (`set_frame_ids` の regex)
- sim rgb `image_{k:04d}.png` を `sim_input/<seq>/{k+1:06d}.png` に **symlink** して命名
  → VGGT-SLAM frame_id = CLiMB の 1-based frame ID に直結 (image_0000→ID1)
- min_disparity でキーフレーム選択され出力は部分集合 → 100共通ID閾値を満たすため
  min_disparity は小さめ(10)から。足りなければ下げる

## パイプライン (`run.sh`, STAGE=slam|adapt|eval|all)
1. slam: `repo/main.py --image_folder sim_input/<seq> --submap_size 16 --max_loops 0
   --log_results` → `raw/<seq>/poses.txt` (+ _points.pcd)
2. adapt: `scripts/poses_to_climb.py` で poses.txt → `output/<seq>/<run>/...`
   - poses.txt 形式: `frame_id x y z qx qy qz qw` (quat xyzw)
   - pose規約は c2w/w2c 両対応フラグ。射影分解なので**経験的に低ATE側を採用**(要確認)
3. eval: expA00 と同じ公式評価器 + sim→COLMAP GT で採点 → ORB-SLAM3/DA3と同一土俵

## 結果 (VGGT-SLAM, c2w規約, submap16/min_disp10/max_loops0, 5ラン=決定的で同値)
| seq | 変形A/ω | TFR% | ATE(mm) | RPE_r(δ40°) | keyframes | Success |
|---|---|---|---|---|---|---|
| Seq_0 | 0/0 | 45.3 | 9.87 | 23.8 | 155 | ✗ |
| Seq_1 | 2.5/2.5 | 47.5 | 9.25 | 26.7 | 163 | ✗ |
| Seq_2 | 2.5/5 | 50.3 | 10.54 | 27.2 | 175 | ✓ |
| Seq_3 | 5/2.5 | 49.4 | 11.63 | 27.7 | 169 | ✗ |
| Seq_4 | 5/5 | 44.4 | 12.19 | 24.0 | 157 | ✗ |
| Seq_5 | 10/5 | 76.7 | 10.68 | 24.5 | 268 | ✓ |
| **Mean** | | **52.3** | **10.69** | 25.6 | 181 | **2/6** |

## 三手法比較 (sim Mean)
| 手法 | ATE(mm) | TFR% | Success |
|---|---|---|---|
| ORB-SLAM3 (expA00) | **6.63** | **72.6** | 5/6 |
| VGGT-SLAM (expC00) | 10.69 | 52.3 | 2/6 |
| DA3一括 (expB00, res182) | 35.6(Seq_0) | - | - |

## 知見（重要）
- **VGGT-SLAMは変形ロバスト**: ATEが変形量に依らずほぼ平坦(9.25〜12.19mm)。
  対してORB-SLAM3は剛体1.11mm→最大変形16.21mmと劣化。
  **最難の Seq_5(A=10) では VGGT-SLAM 10.68mm < ORB-SLAM3 16.21mm で逆転勝利**。
  生体は常時変形するので、実系列ではVGGT-SLAM優位の可能性
- **単一マップ (#Maps=1.0)**: ORB-SLAMのような分割が起きず一貫した1本の軌跡。途切れに強い
- **TFRがキーフレーム選択で頭打ち**: 出力は keyframe のみ(155〜268/322)。
  min_disparity=10で間引かれ、全GT(322)に対しTFR上限が~48-83%に制限される。
  → **min_disparityを下げる/非KFを補間すれば TFR は大きく改善余地あり**（ORB-SLAMは全フレーム出力なので公平でない）
- 速度: VGGT 0.76s/frame, 全体 ~1 FPS (333frame=142s)。実系列3万frameだと長時間→提出のRuntime指標に注意
- 規約確定: poses.txt の (x,y,z,R) は **camera-to-world (c2w)** (ATE 9.87 vs w2c 20.2 で確定)

## min_disparity=2 完走結果 (2026-08-28, `results_md2.json`)
中断していた Seq_3〜5 を再実行し全 6 系列で確定（5ラン複製, 1マップ）:
| seq | ATE(mm) | TFR% | Success |
|---|---|---|---|
| Seq_0 | 12.81 | 63.0 | ✓ |
| Seq_1 | 15.88 | 64.9 | ✓ |
| Seq_2 | 5.89 | 85.1 | ✓ |
| Seq_3 | 9.82 | 85.1 | ✓ |
| Seq_4 | 15.01 | 73.3 | ✓ |
| Seq_5 | 10.98 | 99.1 | ✓ |
| **Mean** | **11.73** | **78.4** | **6/6** |

- md10 比: TFR 52.3→78.4、**Success 2/6→6/6（全手法初）**、ATE 10.69→11.73 の微増のみ
- **Seq_4（ORB-SLAM3 唯一の✗系列）も 73.3% で ✓** → 被覆の頑健性は VGGT が明確に上
- EDA 予測（min_disp≈2 で TFR 大幅改善）を実測で裏付け。ただし 100% には届かず
  （min_disp=2 でも低視差フレームは間引かれる。TFR>90% を狙うなら非KF補間が次の一手）
