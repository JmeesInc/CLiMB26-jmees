#!/usr/bin/env python3
"""CLiMB EDA -> survey/competition/eda_report.md
Covers: real-sequence video stats (ffprobe headers), info.json aggregation,
per-endoscope calibration spread, SamePatient grouping, sim-sequence summary.
Stdlib only (json/xml/subprocess/statistics) so it runs without extra deps."""
import json
import re
import statistics as st
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "data"
OUT = REPO / "survey" / "competition" / "eda_report.md"
OUT.parent.mkdir(parents=True, exist_ok=True)


def ffprobe(path):
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration",
             "-of", "json", str(path)],
            capture_output=True, text=True, timeout=60)
        s = json.loads(r.stdout)["streams"][0]
        fr = s.get("r_frame_rate", "0/1")
        num, den = fr.split("/")
        fps = float(num) / float(den) if float(den) else 0.0
        dur = float(s.get("duration", 0) or 0)
        nbf = s.get("nb_frames")
        nframes = int(nbf) if nbf and nbf.isdigit() else int(round(dur * fps))
        return {"w": s.get("width"), "h": s.get("height"), "fps": round(fps, 2),
                "dur": dur, "nframes": nframes}
    except Exception as e:
        return {"error": str(e)}


def fmt_stats(vals):
    vals = [v for v in vals if v]
    if not vals:
        return "n/a"
    return (f"min={min(vals):.1f}  median={st.median(vals):.1f}  "
            f"max={max(vals):.1f}  sum={sum(vals):.1f}")


def main():
    lines = ["# CLiMB EDA レポート", "",
             "自動生成 (workspace/expA00_baseline_eval/scripts/eda.py)。", ""]

    # --- Real sequences ---
    seq_dirs = sorted([d for d in (DATA / "Sequences").iterdir()
                       if d.is_dir() and d.name.startswith("Seq_")])
    lines += [f"## 1. 実系列 (Sequences/) — {len(seq_dirs)} 本", ""]
    stats, endo_counter, type_counter, endo_of_seq = [], Counter(), Counter(), {}
    rows = []
    for d in seq_dirs:
        mov = next((p for p in d.glob("*.mov")), None)
        info = d / f"{d.name}_info.json"
        endo = typ = None
        if info.is_file():
            try:
                j = json.loads(info.read_text())
                endo = str(j.get("endoscope_number", "")).zfill(2)
                typ = j.get("type")
            except Exception:
                pass
        if endo:
            endo_counter[endo] += 1
            endo_of_seq[d.name] = endo
        if typ:
            type_counter[typ] += 1
        meta = ffprobe(mov) if mov else {"error": "no mov"}
        if "error" not in meta:
            stats.append(meta)
            rows.append((d.name, endo, typ, meta["w"], meta["h"], meta["fps"],
                         meta["nframes"], meta["dur"]))
    lines += [f"- 動画統計が取れた本数: {len(stats)}",
              f"- 総フレーム数: {fmt_stats([s['nframes'] for s in stats])}",
              f"- 尺(秒): {fmt_stats([s['dur'] for s in stats])}",
              f"- 解像度の分布: {Counter((s['w'], s['h']) for s in stats)}",
              f"- fpsの分布: {Counter(s['fps'] for s in stats)}", ""]
    lines += ["### type 分布", ""]
    for k, v in type_counter.most_common():
        lines.append(f"- {k}: {v}")
    lines += ["", "### endoscope 使用本数", ""]
    for k, v in sorted(endo_counter.items()):
        lines.append(f"- Endoscope_{k}: {v} 本")
    lines += ["", "<details><summary>全系列テーブル</summary>", "",
              "| seq | endo | type | W | H | fps | frames | dur(s) |",
              "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    lines += ["", "</details>", ""]

    # --- Calibration ---
    lines += ["## 2. キャリブレーション (Calibrations/)", "",
              "| endoscope | fx | fy | cx | cy | k1 | k2 | k3 | k4 | W | H |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    calib_dir = DATA / "Calibrations"
    if calib_dir.is_dir():
        for d in sorted(calib_dir.iterdir()):
            geo = next((p for p in d.glob("*_geometrical.xml")), None)
            if not geo:
                continue
            try:
                root = ET.fromstring(geo.read_text())
                params = root.find(".//params").text
                nums = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", params)]
                cm = root.find(".//camera_model")
                w = cm.get("width") if cm is not None else (root.findtext(".//width") or "")
                h = cm.get("height") if cm is not None else (root.findtext(".//height") or "")
                w = (root.findtext(".//width") or w or "").strip()
                h = (root.findtext(".//height") or h or "").strip()
                vals = nums[:8] + [None] * (8 - len(nums))
                row = [d.name.replace("Endoscope_", "")] + \
                      [f"{v:.4g}" if isinstance(v, float) else "" for v in vals] + [w, h]
                lines.append("| " + " | ".join(str(x) for x in row) + " |")
            except Exception as e:
                lines.append(f"| {d.name} | parse error: {e} |")
    lines.append("")

    # --- SamePatient ---
    lines += ["## 3. 同一患者グループ (SamePatient.json)", ""]
    sp = DATA / "SamePatient.json"
    if sp.is_file():
        j = json.loads(sp.read_text())
        lines.append(f"- 記載ペア数: {len(j)}")
        for k, v in j.items():
            seqs = [vv for kk, vv in v.items() if kk.lower().startswith("sequence")]
            extra = {kk: vv for kk, vv in v.items() if not kk.lower().startswith("sequence")}
            endos = [endo_of_seq.get(s, "?") for s in seqs]
            lines.append(f"  - group {k}: {seqs} (endoscope={endos}) {extra}")
        lines += ["", "> 注: 明示ペアは少数。endoscope_number が同一患者の代理になり得るか要検討（"
                  "同一内視鏡=同一施設/時期の可能性。fold ではリーク回避のため endoscope 単位 GroupKFold を検討）。"]
    lines.append("")

    # --- Simulated ---
    lines += ["## 4. シミュ系列 (Simulated_Sequences/)", "",
              "全系列で trajectory.csv / calibration.txt は**同一**（同一カメラ軌跡・同一内視鏡）。"
              "変形(deformation)パラメータのみ異なるレンダリング違い。", "",
              "| seq | Amplitude[mm] | omega[rad/s] | rgb枚数 |",
              "|---|---|---|---|"]
    sim_dir = DATA / "Simulated_Sequences"
    for d in sorted(sim_dir.glob("Seq_*")):
        if not d.is_dir():
            continue
        info = (d / "info.txt").read_text() if (d / "info.txt").is_file() else ""
        A = re.search(r"Amplitude.*?:\s*([\d.]+)", info)
        om = re.search(r"Deformation speed.*?:\s*([\d.]+)", info)
        nrgb = sum(1 for _ in (d / "rgb").glob("*.png")) if (d / "rgb").is_dir() else 0
        lines.append(f"| {d.name} | {A.group(1) if A else '?'} | "
                     f"{om.group(1) if om else '?'} | {nrgb} |")
    lines += ["", "- GT軌跡: 322 poses, 30fps, 全長≈210.6mm / 直線変位≈149.6mm (dm→mm換算)。",
              "- 座標系: TUM形式 camera-to-world と解釈 (tX,tY,tZ=世界座標カメラ中心, quat=xyzw)。", ""]

    OUT.write_text("\n".join(lines))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
