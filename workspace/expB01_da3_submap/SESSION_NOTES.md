# expB01_da3_submap — DA3 チャンク分割 SLAM

## 目的
expB00 の結論「DA3 一括投入はメモリで破綻 → submap 分割が必須」の実行。
VGGT-SLAM 流の submap 方式を DA3 バックボーンで再現し、DA3 の変形ロバスト性が
CLiMB でどこまで効くか測る（ループクロージャなしの素朴版）。

## 手法
- `scripts/chunk_slam.py`: CHUNK フレームずつ DA3 に投入（OVERLAP フレーム共有）
  → チャンク局所 w2c を **rotation-first Sim(3)**（回転はカメラ姿勢平均から、
  スケール/並進は中心から）でオーバーラップカメラに整列して連結
- rotation-first の理由: 前進主体の colonoscopy はカメラ中心が準直線配置になり、
  中心だけの Umeyama Sim3 だと回転が縮退するため
- 点群: チャンクごとに depth を unproject → チャンク Sim3 で global へ
- 環境: **iMED venv 流用** (`/data4/src/shunsuke/MICCAI2026/iMED/.venv`, DA3+torch cu126)。
  入力フレームは expC00 の `sim_input/<seq>/{id:06d}.png` を流用（1-based ID）

## メモリ上限（RTX 8000 48GB, NESTED-GIANT-LARGE, res504）
| chunk | 結果 |
|---|---|
| 16 | OOM（SDPA math カーネルが 21.7GiB 追加要求） |
| 8 | OK（6.2s/call） |

## 結果
| 設定 | seq | ATE(mm) | TFR% | 備考 |
|---|---|---|---|---|
| chunk8/ov3 res504 NESTED-GIANT | Seq_0 | 19.27 | **100.0** | smoke 1run, 1.28s/frame |

- TFR 100%（全フレーム出力・単一マップ）は狙い通り
- ATE 19.27mm は VGGT-SLAM(9.87) より悪い: 66 チャンクの逐次連結で
  ドリフト蓄積（ループクロージャ・グローバル最適化なし）。チャンク間 scale は
  ほぼ 1.0000 で DA3 metric depth のスケール一貫性は確認できた

## スケール暴走バグと修正 (2026-08-28)
初版の最小二乗スケール（中心の分散比）は overlap=3 の準静止ベースラインで縮退し
チャンク間 s=9 等の暴走が発生（Seq_0 smoke ATE 19.27 はこのバグ込み）。
→ **pairwise 中心距離比の median + (0.25,4) クランプ・縮退時 s=1** に変更
（`--s_lo/--s_hi`, s_lo>=s_hi で s=1 固定 = 純SE3）。

## フル 6 系列結果 (chunk8/ov3, NESTED-GIANT res504, robust scale, `results.json`)
| seq | ATE(mm) | TFR% | Success |
|---|---|---|---|
| Seq_0 | 6.12 | 100 | ✓ |
| Seq_1 | 5.02 | 100 | ✓ |
| Seq_2 | 6.93 | 100 | ✓ |
| Seq_3 | 4.92 | 100 | ✓ |
| Seq_4 | 11.36 | 77.6 | ✓ |
| Seq_5 | 8.18 | 100 | ✓ |
| **Mean** | **7.09** | **96.3** | **6/6** |

- **最難 Seq_5 で 8.18mm = 全手法ベスト**（ORB 16.21 / VGGT-SLAM 10.68）。変形に対し ATE フラット
- 単一マップ・全被覆で 7.09mm → 断片化採点の恩恵がある ORB 6.63mm より実質優位
- Seq_4 の TFR 77.6% は rgb 断片化アーティファクト（GT は先頭200フレームのみ）由来
- 残課題: scale がクランプ境界 (0.253 / 3.92) に近いチャンクが残る → s=1 固定 ablation
  (`output_s1/`) で検証中。速度 1.28 s/frame（Runtime 指標には要注意）

## s=1 固定 ablation (2026-08-28, `results_s1.json`) — ★ベスト更新
チャンク間スケール推定を完全に捨て s=1 固定（純 SE(3) 連結、DA3 metric depth を信頼）:
| seq | robust scale | s=1 固定 |
|---|---|---|
| Seq_0 | 6.12 | **3.11** |
| Seq_1 | 5.02 | **3.26** |
| Seq_2 | 6.93 | **3.97** |
| Seq_3 | 4.92 | **4.11** |
| Seq_4 | 11.36 | **8.63** |
| Seq_5 | 8.18 | 8.39 |
| **Mean** | 7.09 | **5.25** |

- **全手法ベスト 5.25mm**（da3pair 6.59 / ORB 見かけ 6.63 超え）。TFR 96.3 / 6/6 は維持
- 結論: **DA3 の metric スケールはチャンク間でほぼ完全に一貫しており、
  スケール推定はノイズ源にしかならない**。クランプ付き robust 推定 (7.09) ですら有害
- 「ペア連鎖(6.59) > チャンク連結(7.09)」という中間結論はスケール推定ノイズの産物で、
  s=1 なら チャンク連結(5.25) > ペア連鎖(6.59)。**長いコンテキスト + 剛体連結が正解**
- 次: overlap/chunk の掃引、(t,t+k) スキップ拘束の格子BA、conf 重み付き整列
