#!/usr/bin/env python3
"""
Strike Zone Results Tracker
--------------------------------------------------------------
Same architecture as the other three trackers, adapted for MLB Stats
API (fully open, no key needed) and Strike Zone's mixed structure:
pitcher props (Strikeouts, Outs Recorded) need a specific pitcher's
actual line in a specific game (via the boxscore), while team props
(Team Runs, Team Hits, Run Form, K Form and their Real Streak versions)
need the team's actual final total (via the linescore).

Designed to be imported and called from build_slate.py's main().

Output:
    docs/strike-zone/results/log.json    -- the full log
    docs/strike-zone/results/index.html  -- dashboard: overall +
                                             per-category win rate,
                                             recent history
"""

import os
import json
import hashlib
import requests
from datetime import datetime, timezone

BASE = "https://statsapi.mlb.com/api/v1"
LOG_PATH = "docs/strike-zone/results/log.json"
DASHBOARD_PATH = "docs/strike-zone/results/index.html"


def _get(path, params=None):
    try:
        r = requests.get(f"{BASE}{path}", params=params or {}, timeout=15)
    except Exception as e:
        print(f"    [!] verification request failed: {path} ({e})")
        return None
    if r.status_code != 200:
        print(f"    [!] {r.status_code} on {path}: {r.text[:150]}")
        return None
    return r.json()


def _entry_id(scanner, subject, date_key, market):
    raw = f"{scanner}|{subject}|{date_key}|{market}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def load_log():
    if not os.path.exists(LOG_PATH):
        return []
    try:
        with open(LOG_PATH) as f:
            return json.load(f)
    except Exception as e:
        print(f"  [!] Couldn't read existing results log ({e}) -- starting fresh.")
        return []


def save_log(entries):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "w") as f:
        json.dump(entries, f, indent=2, default=str)


LEG_CATEGORY_SCANNER = {
    "Strikeouts": "strikeouts", "Outs Recorded": "outs",
    "Team Runs": "team_runs", "Team Hits": "team_hits",
}


def log_todays_signals(legs, run_entries, real_run_entries, k_entries, real_k_entries, log):
    """Logs every Safest-Bet-Builder leg plus every Run/K Form and Real
    Run/K Streak entry. Each already carries game_pk (+ pitcher_id where
    relevant) added specifically for this tracker when they were built."""
    existing_ids = {e["id"] for e in log}
    added = 0

    def add(scanner, subject, market, value, game_pk, game_date, pitcher_id=None,
             is_home=None, line=None, threshold=None, detail=None):
        nonlocal added
        eid = _entry_id(scanner, subject, game_date, market)
        if eid in existing_ids:
            return
        log.append({
            "id": eid, "scanner": scanner, "subject": subject, "market": market,
            "value": value, "detail": detail,
            "game_pk": game_pk, "pitcher_id": pitcher_id, "is_home": is_home,
            "line": line, "threshold": threshold,
            "game_date": game_date, "date_key": game_date,
            "logged_at": datetime.now(timezone.utc).isoformat(),
            "status": "pending", "result": None, "actual": None,
        })
        existing_ids.add(eid)
        added += 1

    for leg in legs:
        scanner = LEG_CATEGORY_SCANNER.get(leg["category"])
        if not scanner or not leg.get("game_pk"):
            continue
        add(scanner, leg["subject"], leg["market"], leg["prob"],
            leg["game_pk"], leg["game_date"], pitcher_id=leg.get("pitcher_id"),
            is_home=leg.get("is_home"), line=leg["line"], detail=leg.get("detail"))

    for e in run_entries:
        if not e.get("game_pk"):
            continue
        add("run_form", e["team"], f"Run Form vs {e['opponent']}", e["last5_avg"],
            e["game_pk"], e["game_date"], is_home=e["is_home"])

    for e in real_run_entries:
        if not e.get("game_pk"):
            continue
        add("real_run_streak", e["team"], f"Real Run Streak vs {e['opponent']}", e["streak_len"],
            e["game_pk"], e["game_date"], is_home=e["is_home"], threshold=REAL_RUN_STREAK_THRESHOLD)

    for e in k_entries:
        if not e.get("game_pk"):
            continue
        add("k_form", e["name"], f"K Form ({e['team']} vs {e['opp']})", e["last5_avg"],
            e["game_pk"], e["game_date"], pitcher_id=e.get("pitcher_id"))

    for e in real_k_entries:
        if not e.get("game_pk"):
            continue
        add("real_k_streak", e["name"], f"Real K Streak ({e['team']} vs {e['opp']})", e["streak_len"],
            e["game_pk"], e["game_date"], pitcher_id=e.get("pitcher_id"), threshold=REAL_K_STREAK_THRESHOLD)

    print(f"  Results log: {added} new pick(s) logged, {len(log)} total in log")
    return log


def _get_final_linescore(game_pk):
    # BUG FIX: this was fetching /schedule WITHOUT hydrate=linescore,
    # which means MLB Stats API returns an empty linescore every time --
    # game.get("linescore", {}) was always {}, so home_runs/away_runs
    # were always None, and this function returned None on EVERY call.
    # That's what was silently keeping Team Runs, Team Hits, Run Form
    # and Real Run Streak permanently stuck at 0/0 verified -- not a
    # timing issue, a genuine missing parameter.
    sched = _get("/schedule", {"gamePk": game_pk, "hydrate": "linescore"})
    if not sched or not sched.get("dates"):
        return None
    games = sched["dates"][0].get("games", [])
    if not games:
        return None
    game = games[0]
    status = game.get("status", {}).get("abstractGameState")
    if status != "Final":
        return None
    ls = game.get("linescore", {})
    teams = ls.get("teams", {})
    home_runs = teams.get("home", {}).get("runs")
    away_runs = teams.get("away", {}).get("runs")
    home_hits = teams.get("home", {}).get("hits")
    away_hits = teams.get("away", {}).get("hits")
    if home_runs is None or away_runs is None:
        return None
    return {"home_runs": home_runs, "away_runs": away_runs, "home_hits": home_hits, "away_hits": away_hits}


def _get_pitcher_boxscore_line(game_pk, pitcher_id):
    data = _get(f"/game/{game_pk}/boxscore")
    if not data:
        return None
    for side in ("home", "away"):
        players = data.get("teams", {}).get(side, {}).get("players", {})
        p = players.get(f"ID{pitcher_id}")
        if p:
            stat = p.get("stats", {}).get("pitching", {})
            if stat.get("strikeOuts") is not None:
                return {"strikeouts": stat.get("strikeOuts"), "outs": stat.get("outs")}
    return None


def _verify_pitcher_leg(entry):
    line = _get_pitcher_boxscore_line(entry["game_pk"], entry["pitcher_id"])
    if not line:
        return None
    key = "strikeouts" if entry["scanner"] == "strikeouts" else "outs"
    actual = line.get(key)
    if actual is None:
        return None
    return {"actual": actual, "result": "hit" if actual > entry["line"] else "miss"}


def _verify_pitcher_streak_entry(entry, threshold):
    line = _get_pitcher_boxscore_line(entry["game_pk"], entry["pitcher_id"])
    if not line:
        return None
    actual = line.get("strikeouts")
    if actual is None:
        return None
    return {"actual": actual, "result": "hit" if actual >= threshold else "miss"}


def _verify_team_leg(entry):
    ls = _get_final_linescore(entry["game_pk"])
    if not ls:
        return None
    if entry["scanner"] == "team_runs":
        actual = ls["home_runs"] if entry["is_home"] else ls["away_runs"]
    else:
        actual = ls["home_hits"] if entry["is_home"] else ls["away_hits"]
    if actual is None:
        return None
    return {"actual": actual, "result": "hit" if actual > entry["line"] else "miss"}


def _verify_team_streak_entry(entry, threshold):
    ls = _get_final_linescore(entry["game_pk"])
    if not ls:
        return None
    actual = ls["home_runs"] if entry["is_home"] else ls["away_runs"]
    if actual is None:
        return None
    return {"actual": actual, "result": "hit" if actual >= threshold else "miss"}


def verify_pending_results(log, real_run_streak_threshold, real_k_streak_threshold, max_checks=150):
    # Raised from 60 -- MLB runs far more games/day than the other sports
    # this pattern was built for, so this tracker genuinely logs more
    # picks per run than a 60-cap could keep pace with, independent of
    # the linescore bug fixed above.
    today = datetime.now(timezone.utc).date().isoformat()
    checked = 0
    updated = 0

    for entry in log:
        if entry["status"] != "pending":
            continue
        if entry["date_key"] >= today:
            continue
        if checked >= max_checks:
            break
        checked += 1

        result = None
        try:
            if entry["scanner"] in ("strikeouts", "outs"):
                result = _verify_pitcher_leg(entry)
            elif entry["scanner"] == "real_k_streak":
                result = _verify_pitcher_streak_entry(entry, real_k_streak_threshold)
            elif entry["scanner"] in ("team_runs", "team_hits"):
                result = _verify_team_leg(entry)
            elif entry["scanner"] in ("run_form", "k_form"):
                # Form entries use the SAME "did they hit their own
                # numbers threshold in this game" check as the streak
                # entries, since Form isn't tied to a specific betting
                # line -- reuses the streak verifiers with the entry's
                # own historical avg as a rough continuity check instead.
                if entry["scanner"] == "run_form":
                    result = _verify_team_streak_entry(entry, entry["value"])
                else:
                    result = _verify_pitcher_streak_entry(entry, entry["value"])
            elif entry["scanner"] == "real_run_streak":
                result = _verify_team_streak_entry(entry, real_run_streak_threshold)
        except Exception as e:
            print(f"    [!] verification error for entry {entry['id']} ({entry['scanner']}): {e}")
            result = None

        if result:
            entry["status"] = "verified"
            entry["result"] = result["result"]
            entry["actual"] = result["actual"]
            entry["verified_at"] = datetime.now(timezone.utc).isoformat()
            updated += 1

    print(f"  Results verification: checked {checked} pending entries, {updated} newly verified")
    return log


def build_results_dashboard(log):
    verified = [e for e in log if e["status"] == "verified"]
    pending = [e for e in log if e["status"] == "pending"]

    by_scanner = {}
    for e in verified:
        d = by_scanner.setdefault(e["scanner"], {"hit": 0, "miss": 0})
        d[e["result"]] += 1

    SCANNER_LABELS = {
        "strikeouts": "Strikeouts", "outs": "Outs Recorded",
        "team_runs": "Team Runs", "team_hits": "Team Hits",
        "run_form": "Run Form", "real_run_streak": "Real Run Streak",
        "k_form": "K Form", "real_k_streak": "Real K Streak",
    }

    total_hit = sum(d["hit"] for d in by_scanner.values())
    total_miss = sum(d["miss"] for d in by_scanner.values())
    total = total_hit + total_miss
    overall_pct = round(100 * total_hit / total) if total else None

    rows = ""
    for scanner, label in SCANNER_LABELS.items():
        d = by_scanner.get(scanner, {"hit": 0, "miss": 0})
        n = d["hit"] + d["miss"]
        pct = round(100 * d["hit"] / n) if n else None
        pct_str = f"{pct}%" if pct is not None else "—"
        rows += f"""<div class="legRow"><span>{label}</span><span style="color:var(--yellow);font-weight:bold">{pct_str}</span><span style="color:var(--sub);font-size:12px">{d['hit']}/{n}</span></div>"""

    recent = sorted(verified, key=lambda e: e.get("verified_at", ""), reverse=True)[:30]
    recent_rows = ""
    for e in recent:
        color = "#22c55e" if e["result"] == "hit" else "#ef4444"
        recent_rows += f"""<div class="legRow" style="font-size:12px"><span>{e['subject']} — {e['market']}</span><span style="color:{color};font-weight:bold">{e['result'].upper()}</span></div>"""

    html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0"><title>Results — Strike Zone</title>
<style>
  :root{{--bg:#0b0f14; --panel:#121820; --panel2:#161d27; --border:#233040; --text:#e8edf2; --sub:#8b98a8; --yellow:#facc15; --green:#22c55e;}}
  body{{margin:0;background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;padding:16px;max-width:640px;margin:0 auto;}}
  .legRow{{display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid var(--border);}}
</style></head>
<body>
<p style="text-align:center;margin-bottom:6px"><a href="../index.html" style="color:#7ec8ff;text-decoration:none;font-size:12px">← Strike Zone</a></p>
<h1 style="text-align:center;font-size:20px;margin-bottom:2px">📊 Results Tracker</h1>
<p style="text-align:center;color:var(--sub);font-size:11px;margin-top:0">{datetime.now().strftime("%Y-%m-%d %H:%M")} · every pick, auto-verified against real results</p>

<div style="background:var(--panel);border-radius:12px;padding:16px;margin:14px 0;border:1px solid var(--border);text-align:center">
  <div style="font-size:11px;color:var(--sub)">OVERALL</div>
  <div style="font-size:32px;font-weight:bold;color:var(--yellow)">{overall_pct if overall_pct is not None else "—"}{"%" if overall_pct is not None else ""}</div>
  <div style="font-size:12px;color:var(--sub)">{total_hit}/{total} verified picks · {len(pending)} pending (game not final yet)</div>
</div>

<div style="background:var(--panel);border-radius:12px;padding:16px;margin:14px 0;border:1px solid var(--border)">
  <div style="font-weight:bold;margin-bottom:8px">By Category</div>
  {rows}
</div>

<div style="background:var(--panel);border-radius:12px;padding:16px;margin:14px 0;border:1px solid var(--border)">
  <div style="font-weight:bold;margin-bottom:8px">Recent Results</div>
  {recent_rows or '<p style="color:var(--sub);font-size:12px">Nothing verified yet — check back after a few days of picks have had time to play out.</p>'}
</div>

<div style="font-size:11px;color:var(--sub);text-align:center;margin-top:20px;line-height:1.6">
  Run Form / K Form are verified against the pick's OWN historical average as a rough
  continuity check (did the team/pitcher keep producing at roughly that level), since Form
  isn't tied to a specific betting line the way the Safest Bet Builder legs are. Real Run/K
  Streak use the fixed streak threshold. Sample sizes are still small early on — treat
  percentages with real caution until there's a few weeks of data.
</div>
</body></html>"""

    os.makedirs(os.path.dirname(DASHBOARD_PATH), exist_ok=True)
    with open(DASHBOARD_PATH, "w") as f:
        f.write(html)
    print(f"  Results dashboard: {total} verified, {overall_pct}% overall" if total else "  Results dashboard: no verified picks yet")


def run_results_tracker(legs, run_entries, real_run_entries, k_entries, real_k_entries,
                          real_run_streak_threshold, real_k_streak_threshold):
    """Single entry point called from build_slate.py's main()."""
    global REAL_RUN_STREAK_THRESHOLD, REAL_K_STREAK_THRESHOLD
    REAL_RUN_STREAK_THRESHOLD = real_run_streak_threshold
    REAL_K_STREAK_THRESHOLD = real_k_streak_threshold

    print("\nRunning results tracker...")
    log = load_log()
    log = log_todays_signals(legs, run_entries, real_run_entries, k_entries, real_k_entries, log)
    log = verify_pending_results(log, real_run_streak_threshold, real_k_streak_threshold)
    save_log(log)
    build_results_dashboard(log)
