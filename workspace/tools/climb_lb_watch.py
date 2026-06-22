#!/usr/bin/env python3
"""CLiMB leaderboard / own-submission watcher.

Polls the Synapse SubmissionViews behind the CLiMB challenge, snapshots them,
and appends a diff report (new submissions, team-best changes, rank changes,
own submission status) to lb_watch.md.

Views:
  syn76297417  CLiMB Leaderboard   (scope = evaluation 9619644 "CLiMB Scoring")
  syn76974964  CLiMB My Submissions (scope = 9620015 Validation + 9619644 Scoring)

NOTE: despite its name, syn76974964 currently returns EVERY participant's rows
(names, submission_status, error_message, container_log_tail), not just ours --
an ACL misconfiguration on the organizers' side. We therefore split the report
into our own team's submissions and a coarse per-team activity summary.

Usage: python3 climb_lb_watch.py [--quiet]
Deadline guard: exits without polling after DEADLINE_JST.
"""
import csv
import datetime as dt
import json
import logging
import os
import sys

OWN_TEAM = "Jmees26"

LB_VIEW = "syn76297417"
MY_VIEW = "syn76974964"
# Docker submission deadline 2026-09-10 23:59 AoE == 2026-09-11 20:59 JST.
DOCKER_DEADLINE_JST = dt.datetime(2026, 9, 11, 20, 59)
# Write-up deadline 2026-09-12 23:59 AoE == 2026-09-13 20:59 JST.
WRITEUP_DEADLINE_JST = dt.datetime(2026, 9, 13, 20, 59)
# The watcher keeps polling past the Docker deadline: the queue is still being
# scored afterwards and other teams' bests can still move. It retires an hour
# after the write-up deadline.
DEADLINE_JST = WRITEUP_DEADLINE_JST + dt.timedelta(hours=1)

HERE = os.path.dirname(os.path.abspath(__file__))
SNAP_DIR = os.path.join(HERE, "lb_snapshots")
REPORT = os.path.join(HERE, "lb_watch.md")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(HERE, "lb_watch.log")),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("lbwatch")

NUM_COLS = ("climb_score", "score_rot", "mean_ate_mm", "mean_rpe_rot_deg_d40",
            "mean_tfr_pct", "mean_runtime_s_per_frame")


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def fetch(syn, view, cols):
    q = syn.tableQuery(f"SELECT {','.join(cols)} FROM {view}")
    with open(q.filepath) as f:
        rows = list(csv.DictReader(f))
    # SubmissionView queries prepend ROW_ID/ROW_VERSION/ROW_ETAG
    return [{k: v for k, v in r.items() if k in cols} for r in rows]


def team_best(rows):
    """Best (lowest climb_score) SCORED submission per team."""
    best = {}
    for r in rows:
        if r.get("status") != "SCORED":
            continue
        s = fnum(r.get("climb_score"))
        if s is None:
            continue
        t = r.get("team") or "(individual)"
        if t not in best or s < fnum(best[t]["climb_score"]):
            best[t] = r
    return best


def fetch_own_bundles(syn):
    """Fallback for when syn76974964 is unqueryable.

    The organizers' view periodically 400s ("The size of the column
    'submission_status' is too small") because a status string outgrew their
    column, which breaks every query against it. This endpoint returns only the
    submissions made by THIS account (teammates' entries are not included), so
    the numbers are a subset of the team's, but the statuses are authoritative.
    """
    rows = []
    for ev in ("9619644", "9620015"):
        offset = 0
        while True:
            r = syn.restGET(f"/evaluation/{ev}/submission/bundle"
                            f"?limit=100&offset={offset}")
            for b in r.get("results", []):
                sub, st = b.get("submission", {}), b.get("submissionStatus", {})
                anno = {a["key"]: a["value"]
                        for a in st.get("annotations", {}).get("stringAnnos", [])}
                rows.append({
                    "id": sub.get("id"), "evaluationid": ev,
                    "name": sub.get("name"), "createdOn": sub.get("createdOn"),
                    "status": st.get("status"),
                    "error_message": anno.get("error_message", ""),
                    "container_log_tail": anno.get("container_log_tail", ""),
                })
            offset += 100
            if offset >= r.get("totalNumberOfResults", 0):
                break
    return rows


def latest_snapshot():
    snaps = sorted(f for f in os.listdir(SNAP_DIR) if f.endswith(".json"))
    if not snaps:
        return None, None
    p = os.path.join(SNAP_DIR, snaps[-1])
    with open(p) as f:
        return snaps[-1], json.load(f)


def main():
    now = dt.datetime.now()
    if now > DEADLINE_JST:
        log.info("past deadline (%s) - watcher retired, nothing to do", DEADLINE_JST)
        return 0

    import synapseclient
    syn = synapseclient.Synapse(silent=True)
    syn.login(silent=True)

    mine_source = "view"
    lb = fetch(syn, LB_VIEW, ["id", "team", "submitterAlias", "status", "climb_score",
                              "score_rot", "mean_ate_mm", "mean_rpe_rot_deg_d40",
                              "mean_tfr_pct", "success_count", "num_sequences",
                              "mean_runtime_s_per_frame", "rank"])
    try:
        mine = fetch(syn, MY_VIEW, ["id", "evaluationid", "name", "submitterAlias",
                                    "createdOn", "status",
                                    "error_message", "container_log_tail"])
        # NOTE: `submission_status` is deliberately not selected -- the organizers'
        # view declares it too narrow for newer status strings and the query fails
        # with "The size of the column 'submission_status' is too small".
    except Exception as e:  # own view may be empty / not yet provisioned
        log.warning("could not read %s: %s -- falling back to /submission/bundle",
                    MY_VIEW, str(e)[:200])
        try:
            mine = fetch_own_bundles(syn)
            mine_source = "own-bundle"
        except Exception as e2:
            log.warning("fallback failed too: %s", str(e2)[:200])
            mine = []

    prev_name, prev = latest_snapshot()
    stamp = now.strftime("%Y%m%d_%H%M")
    with open(os.path.join(SNAP_DIR, f"{stamp}.json"), "w") as f:
        json.dump({"fetched_at": now.isoformat(), "leaderboard": lb, "mine": mine}, f, indent=1)

    prev_lb = prev.get("leaderboard", []) if prev else []
    prev_ids = {r["id"] for r in prev_lb}
    new_rows = [r for r in lb if r["id"] not in prev_ids]
    cur_best, old_best = team_best(lb), team_best(prev_lb)

    out = [f"\n## {now:%Y-%m-%d %H:%M} JST  (prev: {prev_name or 'none'})",
           f"- scored submissions: {len([r for r in lb if r.get('status') == 'SCORED'])}"
           f" (was {len([r for r in prev_lb if r.get('status') == 'SCORED'])})"]

    if new_rows:
        out.append(f"\n### 新規提出 {len(new_rows)} 件")
        out.append("| id | team | status | climb_score | ATE mm | TFR % | s/frame | n_seq |")
        out.append("|---|---|---|---|---|---|---|---|")
        for r in sorted(new_rows, key=lambda x: fnum(x.get("climb_score")) or 9e9):
            out.append("| {id} | {team} | {status} | {climb_score} | {mean_ate_mm} | "
                       "{mean_tfr_pct} | {mean_runtime_s_per_frame} | {num_sequences} |"
                       .format(**{k: (r.get(k) or "-") for k in
                                  ("id", "team", "status", "climb_score", "mean_ate_mm",
                                   "mean_tfr_pct", "mean_runtime_s_per_frame", "num_sequences")}))
    else:
        out.append("- 新規提出: なし")

    moved = []
    for t, r in cur_best.items():
        o = old_best.get(t)
        if o is None:
            moved.append(f"  - **{t}**: NEW best {fnum(r['climb_score']):.3f} (sub {r['id']})")
        elif r["id"] != o["id"]:
            moved.append(f"  - **{t}**: {fnum(o['climb_score']):.3f} -> "
                         f"**{fnum(r['climb_score']):.3f}** (sub {r['id']})")
    if moved:
        out.append("\n### チームベスト更新")
        out.extend(moved)

    out.append("\n### 現在の順位 (team best)")
    out.append("| # | team | climb_score | ATE mm | RotErr d40 | TFR % | s/frame |")
    out.append("|---|---|---|---|---|---|---|")
    for i, (t, r) in enumerate(sorted(cur_best.items(),
                                      key=lambda kv: fnum(kv[1]["climb_score"])), 1):
        out.append(f"| {i} | {t} | {fnum(r['climb_score']):.3f} | "
                   f"{fnum(r['mean_ate_mm']):.3f} | {fnum(r['mean_rpe_rot_deg_d40']):.3f} | "
                   f"{fnum(r['mean_tfr_pct']):.1f} | {fnum(r['mean_runtime_s_per_frame']):.4f} |")

    if mine:
        qname = {"9619644": "Scoring", "9620015": "Validation"}
        # Ownership: the leaked view carries no team/user column, and
        # /evaluation/submission/{id} is 403 for participants. Attribute via the
        # leaderboard (scored Scoring rows carry `team`), then propagate that
        # team to same-named rows (a Validation row and its Scoring twin share
        # the image name). Names mapping to >1 team stay unattributed.
        team_of = {r["id"]: (r.get("team") or "") for r in lb if r.get("team")}
        name_team = {}
        for r in lb:
            t = r.get("team")
            if not t:
                continue
            nm = next((m.get("name") for m in mine if m["id"] == r["id"]), None)
            if nm:
                name_team.setdefault(nm, set()).add(t)
        name_team = {k: v.pop() for k, v in name_team.items() if len(v) == 1}
        extra = os.path.join(HERE, "own_names.txt")
        own_names = set()
        if os.path.exists(extra):
            own_names = {l.strip() for l in open(extra) if l.strip()}

        def owner(r):
            if r["id"] in team_of:
                return team_of[r["id"]]
            if (r.get("name") or "") in own_names:
                return OWN_TEAM
            return name_team.get(r.get("name") or "", "?")

        prev_my = {r["id"] for r in (prev.get("mine", []) if prev else [])}
        own = [r for r in mine if owner(r) == OWN_TEAM]
        others = [r for r in mine if owner(r) not in (OWN_TEAM,)]

        note = ("" if mine_source == "view"
                else "  ※ 組織側ビューが 400 のため自分のアカウント分のみ（チームメイトの提出は含まれない）")
        out.append(f"\n### {OWN_TEAM} の提出 (直近 15 / 全 {len(own)} 件){note}")
        out.append("| id | queue | name | status | error |")
        out.append("|---|---|---|---|---|")
        for r in sorted(own, key=lambda x: x.get("createdOn") or "")[-15:]:
            flag = " **NEW**" if r["id"] not in prev_my else ""
            out.append("| {}{} | {} | {} | {} | {} |".format(
                r.get("id"), flag, qname.get(str(r.get("evaluationid")), r.get("evaluationid")),
                (r.get("name") or "")[:40],
                r.get("submission_status") or r.get("status"),
                (r.get("error_message") or "")[:80]))

        fresh = [r for r in others if r["id"] not in prev_my]
        out.append(f"\n### 他チーム / 帰属不明 (全 {len(others)} 件 / 前回以降 {len(fresh)} 件)")
        if fresh:
            out.append("| id | team | queue | name | status |")
            out.append("|---|---|---|---|---|")
            for r in sorted(fresh, key=lambda x: x.get("createdOn") or ""):
                out.append("| {} | {} | {} | {} | {} |".format(
                    r.get("id"), owner(r),
                    qname.get(str(r.get("evaluationid")), r.get("evaluationid")),
                    (r.get("name") or "")[:40],
                    r.get("submission_status") or r.get("status")))
    else:
        out.append("\n### 提出ビュー: 0 件 (読めていない可能性)")

    if now < DOCKER_DEADLINE_JST:
        remain, label = DOCKER_DEADLINE_JST - now, "Docker 提出 2026-09-10 23:59 AoE = 09-11 20:59 JST"
    else:
        remain, label = WRITEUP_DEADLINE_JST - now, "write-up 2026-09-12 23:59 AoE = 09-13 20:59 JST（Docker は締切済）"
    out.append(f"\n- 締切まで **{remain.days}d {remain.seconds // 3600}h {(remain.seconds % 3600) // 60}m**"
               f" ({label})")

    text = "\n".join(out)
    if not os.path.exists(REPORT):
        with open(REPORT, "w") as f:
            f.write("# CLiMB leaderboard watch\n\n"
                    "`climb_lb_watch.py` の 12h ごとの自動レポート（新しいものが下）。\n")
    with open(REPORT, "a") as f:
        f.write(text + "\n")
    if "--quiet" not in sys.argv:
        print(text)
    log.info("report appended (new=%d, teams=%d)", len(new_rows), len(cur_best))
    return 0


if __name__ == "__main__":
    sys.exit(main())
