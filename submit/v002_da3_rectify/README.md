# v002_da3_rectify — DA3 submap SLAM + kb4 魚眼 rectification

**v001 との差分は 1 点のみ**: デコード直後の各フレームを Kannala-Brandt kb4 → pinhole に
undistort（+ 有効域クロップ）してから DA3 に渡す。SLAM 本体・連結・出力は v001 と同一。

## 背景（なぜこの 1 点か）
- v001 LB: climb 42.419 (rank 7/7) = ATE 28.28mm / rot 34.0°。ローカル CV 4.13mm の ×6.8
- `workspace/expB02_fisheye` の対照実験で原因を単離:
  sim に実キャリブの魚眼を合成 → **素通し ATE 19.83 / rot 36.0°（LB 実測と一致）**、
  **rectify で 5.57 / 24.0° に完全回復**
- コンテナ監査はシロ（cv2 デコード = ffmpeg 抽出とビット一致、S1 棄却）

## 実装
1. **キャリブ選択** (`calib_table.py`, 公式 Calibrations/ 由来):
   動画名から `Seq_(\d+)` を読み、公開 seq→endoscope 対応（75/93 系列）で正確な kb4 を選択。
   不明時は 18 本平均（クラスタ内 ±2% で均質、DA3 は焦点自己推定のため歪み形状の除去が本質）
2. **rectify**: `cv2.fisheye.initUndistortRectifyMap`（balance=0）。解像度は w/1440 スケールで追従
3. **有効域クロップ**: rectify 後の初フレームで「画像端に接続する暗領域 = 黒枠」を flood-fill 検出
   （内腔の暗部は端に接続しないので保護）→ ほぼ全有効（≥99.5%）の最大中央矩形に切り出し。
   フル有効なら no-op。auto-P が知り得ない画像内黒枠（実機の八角形ボーダー等）への保険
4. `DA3_RECTIFY=0` で無効化（pinhole sim のローカル回帰テスト用）

## ローカル検証
**1. 合成魚眼 mp4 の end-to-end（公式評価器, Seq_0/Seq_5）**

| 条件 | Mean ATE | rot(δ40) |
|---|---|---|
| 素通し（= v001 が実系列でやっていたこと） | 19.83 | 36.0° |
| **v002** | **7.27** | **24.4°** |
| v002（Seq_0 も正確キャリブ強制） | 3.35 (Seq_0) | 23.3° |
| 参考: pinhole 基準（歪みなし） | 5.75 / Seq_0 3.11 | ~24° |

→ **正確な per-seq キャリブなら魚眼劣化をほぼ完全に打ち消す**（3.35 vs 基準 3.11）。
平均 fallback でも素通し比 3〜4× 改善。

**2. 実クリップ（Seq_001 30s@40fps, GT なし・定性）**
`Seq_001_a` → Endoscope_01 を正しく選択、crop no-op、1200/1200 ポーズ。
ステップ長 median 0.0199（素通し 0.0708 = **3.6×**）、素通しは軌跡が発散。
図: `workspace/expB02_fisheye/figs/real_traj_compare.png`

**3. コンテナ回帰**: `GPU_INDEX=3 DA3_RECTIFY=0 DA3_PRECISION=fp16 bash test.sh`
（pinhole sim なので rectify OFF、v001 の 4.13mm 再現を確認）

## ビルド・提出
```bash
bash build.sh   # vendor/DA3 + model/hf を stage して docker build
bash test.sh    # ローカル回帰
bash export.sh  # docker save
```
