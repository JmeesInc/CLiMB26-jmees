# expB00_da3 — DA3 (Depth-Anything-3) フィードフォワード姿勢の試行

## 目的
VGGT/DA3系フィードフォワード3D再構成を単眼大腸内視鏡SLAMに適用できるか検証。
DA3は `model.inference(images)` で extrinsics(w2c)/intrinsics/depth/conf を一括出力。

## セットアップ
- `/data4/src/shunsuke/Depth-Anything-3` を editable install（既存env, torch2.9.1）
- 補完依存: einops>=0.7, moviepy<2, trimesh, plyfile, e3nn 等
- 重み: DA3NESTED-GIANT-LARGE(1.4B, cached) と DA3-LARGE(0.3B, DL) を使用

## 結果（Seq_0, Sim3整列ATE, `quick_ate.py`で測定）
| 設定 | ATE | 備考 |
|---|---|---|
| 111frame@res182 (DA3-LARGE) | **35.6mm** | 実行可能枠の上限 |
| 40frame@res252 | 47.2mm | |
| res378/504 | OOM | 少フレームでも不可 |

→ ORB-SLAM3(1.11mm)・VGGT-SLAM(9.87mm)に対し大幅に劣る。

## 重大な知見（なぜ筋悪か）
- DA3はDINOv2偶数層以降で**全ビュー連結の全対全cross-view attention** → O((S·N)²)メモリ
- `attn_mask=None`でもPyTorch SDPAがflash/efficientカーネルを使えず(RoPE/qk_norm依存で
  「No available kernel」)、**mathカーネルのN²行列を実体化** → 高解像度/多フレームでOOM
- xformers効率attentionも本envでは効かず。bf16 autocastでも mask実体化は回避不可
- 実行可能枠(res182×111frame)では解像度不足でATE35mm止まり
- 座標規約: w2c(35.6) vs c2w(39.4) で w2c がわずかに良い（docstring通り opencv/colmap w2c が正）

## 結論
**全フレーム一括投入は本env(効率attention不可)では低解像度に縛られ筋が悪い。**
→ submap分割（短チャンク毎に推論して連結）が必須 → expC00 VGGT-SLAM へ移行。

## 成果物（流用可）
- `scripts/quick_ate.py`: CLiMB軌跡 vs sim GT を Sim3整列ATE測定（100frame閾値非依存）。
  expC00でも規約確認に使用
- `scripts/da3_infer.py`: DA3推論→CLiMB形式変換（pose_conv c2w/w2c, 点群unproject）
