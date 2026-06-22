# expD00_imedpe_vo — iMED-PE (tri3d) 手法の単眼移植

## 目的
../iMED PE で採用した tri3d（ALIKED+LightGlue → 3D-3D IRLS Umeyama → 格子BA,
iMED mean ATE 0.953mm）を CLiMB に移植する。CLiMB は単眼なので、tri3d の
ステレオ三角測量を **DA3 の 2-view metric depth** で置換する。

## 手法 (`scripts/vo_pair.py`)
- 連続ペア (t, t+1) ごとに DA3 を 1 コール → ペア一貫の depth + intrinsics
  （+ DA3 自身の 2-view extrinsics も出るので、その連鎖を無料の ablation として保存）
- ALIKED+LightGlue マッチ（**iMED v004_pe_tri3d から Matcher を import**, fp16=False —
  tri3d で fp16 は精度劣化と確認済みのため）
- マッチ点を両フレームの depth で unproject → 3D-3D 対応 → **IRLS(Tukey) rigid fit**
  （tri3d umeyama_3d の移植。fit=se3 がデフォルト: ペア内 depth は同一コールで
  スケール共有のため。sim3 は ablation）→ 相対姿勢を連鎖
- マッチ不足/fit 失敗時は DA3 extrinsics にフォールバック（回数をログ）
- 出力 2 系統: `output_lg/`（本命 matcher VO）と `output_da3pair/`（DA3 連鎖 = expB01 の chunk=2 相当）
- 環境: iMED venv 流用。フレームは expC00 sim_input 流用

## 既知の制約
- 格子BA はまだ未移植（ベースの VO が立ってから検討）
- points3D はコール毎スケールの不整合を無視して蓄積（ATE/TFR には影響しない）

## 結果
### Seq_0 smoke (NESTED-GIANT res504, fit=se3, 1.48 s/frame)
| 系統 | ATE(mm) | TFR% | 備考 |
|---|---|---|---|
| da3pair (DA3 2-view extrinsic 連鎖) | **6.42** | 100 | VGGT-SLAM Seq_0 (9.87) 超え |
| lg (matcher+depth IRLS Umeyama) | 9.74 | 100 | フォールバック 0 件 |

- **DA3 自身の 2-view extrinsics 連鎖が matcher+Umeyama を上回った**。DA3 の
  pose head はペア内で depth と整合した推定をしており、疎マッチ+depth
  unprojection より情報を使えている模様
- どちらも単一マップ・全フレーム被覆（TFR 100%）
- フル 6 系列で変形ロバスト性（Seq_5）を確認中

## フル 6 系列結果 (2026-08-28, NESTED-GIANT res504, fit=se3)
| seq | lg ATE / TFR | da3pair ATE / TFR |
|---|---|---|
| Seq_0 | 9.74 / 100 | 6.42 / 100 |
| Seq_1 | 10.16 / 100 | 6.29 / 100 |
| Seq_2 | 9.99 / 100 | 5.74 / 100 |
| Seq_3 | 11.97 / 100 | 6.34 / 100 |
| Seq_4 | 8.21 / 77.6 | 7.51 / 77.6 |
| Seq_5 | 8.50 / 100 | 7.23 / 100 |
| **Mean** | **9.76 / 96.3 (6/6)** | **6.59 / 96.3 (6/6)** |

### 知見
- **da3pair が現ベスト (6.59mm)**: BA なしのペア連鎖で ORB 6.63(断片化恩恵込み)を
  単一マップ・全被覆のまま上回る。**変形量に対し ATE 完全フラット (5.7〜7.5, std±0.60)**
- lg (matcher+depth 3D-3D Umeyama) は全系列で da3pair に劣後 (9.76)。DA3 pose head の
  dense な情報 > 疎マッチ+depth unprojection。iMED では tri3d(=lg相当) > DA3 だったが、
  それは真のステレオ三角測量があったから。単眼+推定depthでは DA3 pose が上
- フォールバック 0 件、1.48 s/frame（ペア毎 DA3 コールが支配的）
- expB01 chunk8 (7.09) より da3pair (6.59) が良い → チャンク間 Sim(3) 連結誤差 >
  ペア連鎖のドリフト、が現状の力関係。ペア連鎖+スキップ接続 pose graph が次の一手

### 次の一手候補
- [ ] 格子BA 移植: da3pair 連鎖に (t,t+k) スキップペアの拘束を足して translation を
  線形最小二乗で refine（iMED 格子BA と同型、-4.5% の実績）
- [ ] lg と da3pair の融合（iMED の tri3d×DA3 blend が有効だった前例）
- [ ] Runtime: 1.48 s/frame は実系列3万フレームで12時間超 → stride/keyframe 予算必須
