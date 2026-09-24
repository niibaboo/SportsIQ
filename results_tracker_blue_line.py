#!/usr/bin/env python3
"""
Blue Line Results Tracker
--------------------------------------------------------------
Same architecture as the other four trackers, adapted for the NHL's
official public API (api-web.nhle.com, no key needed) and Blue Line's
mixed structure: team markets (Team Total, Game Total) need a final
score, player props (Anytime Goalscorer, To Record a Point, Shots on
Goal) need a specific player's actual boxscore line in a specific game.

FIELD-NAME CAVEAT (read before trusting this blindly): unlike the MLB/
TheStatsAPI trackers, this file's exact field names (gameState, score,
playerByGameStats.{homeTeam,awayTeam}.{forwards,defense,goalies},
goals/points/shots) are NOT confirmed against a live response in this
session -- they're based on multiple independent, mutually-consistent
public sources (a Medium walkthrough using this exact endpoint, an
unofficial API reference, and a third-party API docs page), which is
solid but not the same as having actually seen the real JSON. If the
dashboard sits at 0 verified for more than a few days after games have
finished, that's the first thing to suspect -- check the [!] warnings
this prints to stderr for the actual field it choked on.

Designed to be imported and called from blue_line.py's main() -- save
this file as results_tracker.py alongside blue_line.py.

Output:
    docs/blue-line/results/log.json    -- the full log
    docs/blue-line/results/index.html  -- dashboard: overall + per-category
                                           win rate, recent history
"""

import os
import json
import hashlib
import requests
from datetime import datetime, timezone

BASE = "https://api-web.nhle.com/v1"
H = {"User-Agent": "Mozilla/5.0"}
LOG_PATH = "docs/blue-line/results/log.json"
DASHBOARD_PATH = "docs/blue-line/results/index.html"

BOXSCORE_CACHE = {}  # one fetch per game_id per run, several legs share a game


def _get(path):
    try:
        r = requests.get(f"{BASE}{path}", headers=H, timeout=20)
    except Exception as e:
        print(f"    [!] verification request failed: {path} ({e})")
        return None
    if r.status_code != 200:
        print(f"    [!] {r.status_code} on {path}")
        return None
    return r.json()


def _get_boxscore(game_id):
    if game_id in BOXSCORE_CACHE:
        return BOXSCORE_CACHE[game_id]
    data = _get(f"/gamecenter/{game_id}/boxscore")
    BOXSCORE_CACHE[game_id] = data
    return data


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


CATEGORY_SCANNER = {
    "Team Total": "team_total", "Game Total": "game_total",
    "Anytime Goalscorer": "anytime_goal", "To Record a Point": "to_record_point",
    "Shots on Goal": "shots_on_goal",
}


def log_todays_signals(legs, log):
    """Legs already carry game_id, game_date, is_home, line, and (for
    player props) player_id -- added specifically for this tracker when
    the legs were built."""
    existing_ids = {e["id"] for e in log}
    added = 0

    for leg in legs:
        scanner = CATEGORY_SCANNER.get(leg.get("category"))
        if not scanner or not leg.get("game_id"):
            continue
        eid = _entry_id(scanner, leg["market"], leg["game_date"], leg["match"])
        if eid in existing_ids:
            continue
        log.append({
            "id": eid, "scanner": scanner, "subject": leg["market"].rsplit(" ", 0)[0],
            "match": leg["match"], "market": leg["market"], "value": leg["prob"],
            "detail": leg.get("detail"), "line": leg.get("line"),
            "game_id": leg["game_id"], "game_date": leg["game_date"],
            "is_home": leg.get("is_home"), "player_id": leg.get("player_id"),
            "date_key": leg["game_date"],
            "logged_at": datetime.now(timezone.utc).isoformat(),
            "status": "pending", "result": None, "actual": None,
        })
        existing_ids.add(eid)
        added += 1

    print(f"  Results log: {added} new pick(s) logged, {len(log)} total in log")
    return log


def _find_player_stat(boxscore, is_home, player_id):
    side = "homeTeam" if is_home else "awayTeam"
    groups = boxscore.get("playerByGameStats", {}).get(side, {})
    for group_name in ("forwards", "defense", "goalies"):
        for p in groups.get(group_name, []):
            if p.get("playerId") == player_id:
                return p
    return None


def _verify_team_leg(entry):
    box = _get_boxscore(entry["game_id"])
    if not box or box.get("gameState") != "OFF":
        return None
    home_score = box.get("homeTeam", {}).get("score")
    away_score = box.get("awayTeam", {}).get("score")
    if home_score is None or away_score is None:
        return None
    if entry["scanner"] == "game_total":
        actual = home_score + away_score
    else:
        actual = home_score if entry["is_home"] else away_score
    return {"actual": actual, "result": "hit" if actual > entry["line"] else "miss"}


def _verify_player_leg(entry):
    if not entry.get("player_id"):
        return None
    box = _get_boxscore(entry["game_id"])
    if not box or box.get("gameState") != "OFF":
        return None
    p = _find_player_stat(box, entry["is_home"], entry["player_id"])
    if not p:
        return None
    if entry["scanner"] == "anytime_goal":
        actual = p.get("goals")
        if actual is None:
            return None
        return {"actual": actual, "result": "hit" if actual >= 1 else "miss"}
    if entry["scanner"] == "to_record_point":
        actual = p.get("points")
        if actual is None:
            return None
        return {"actual": actual, "result": "hit" if actual >= 1 else "miss"}
    if entry["scanner"] == "shots_on_goal":
        actual = p.get("shots")
        if actual is None:
            return None
        return {"actual": actual, "result": "hit" if actual > entry["line"] else "miss"}
    return None


def verify_pending_results(log, max_checks=60):
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
            if entry["scanner"] in ("team_total", "game_total"):
                result = _verify_team_leg(entry)
            elif entry["scanner"] in ("anytime_goal", "to_record_point", "shots_on_goal"):
                result = _verify_player_leg(entry)
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
        "team_total": "Team Total", "game_total": "Game Total",
        "anytime_goal": "Anytime Goalscorer", "to_record_point": "To Record a Point",
        "shots_on_goal": "Shots on Goal",
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
        rows += f"""<div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #1e3a6a">
  <span>{label}</span><span style="color:#4ea1ff;font-weight:bold">{pct_str}</span>
  <span style="color:#5a6a8a;font-size:12px">{d['hit']}/{n}</span>
</div>"""

    recent = sorted(verified, key=lambda e: e.get("verified_at", ""), reverse=True)[:30]
    recent_rows = ""
    for e in recent:
        color = "#22c55e" if e["result"] == "hit" else "#ff4d5a"
        recent_rows += f"""<div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #1e3a6a;font-size:12px">
  <span>{e['match']} — {e['market']}</span><span style="color:{color};font-weight:bold">{e['result'].upper()}</span>
</div>"""

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Results — Blue Line</title></head>
<body style="background:#081229;color:#fff;font-family:-apple-system,system-ui,sans-serif;padding:16px;max-width:640px;margin:0 auto">
<p style="text-align:center;margin-bottom:6px"><a href="../index.html" style="color:#4ea1ff;text-decoration:none;font-size:12px">← Blue Line</a></p>
<h1 style="text-align:center;font-size:20px;margin-bottom:2px">📊 Results Tracker</h1>
<p style="text-align:center;color:#8aa;font-size:11px;margin-top:0">{datetime.now().strftime("%Y-%m-%d %H:%M")} · every pick, auto-verified against real results</p>

<div style="background:#0f1e3a;border-radius:12px;padding:16px;margin:14px 0;border:1px solid #1e3a6a;text-align:center">
  <div style="font-size:11px;color:#8aa">OVERALL</div>
  <div style="font-size:32px;font-weight:bold;color:#4ea1ff">{overall_pct if overall_pct is not None else "—"}{"%" if overall_pct is not None else ""}</div>
  <div style="font-size:12px;color:#8aa">{total_hit}/{total} verified picks · {len(pending)} pending (game not final yet)</div>
</div>

<div style="background:#0f1e3a;border-radius:12px;padding:16px;margin:14px 0;border:1px solid #1e3a6a">
  <div style="font-weight:bold;margin-bottom:8px">By Category</div>
  {rows}
</div>

<div style="background:#0f1e3a;border-radius:12px;padding:16px;margin:14px 0;border:1px solid #1e3a6a">
  <div style="font-weight:bold;margin-bottom:8px">Recent Results</div>
  {recent_rows or '<p style="color:#8aa;font-size:12px">Nothing verified yet — check back after a few days of picks have had time to play out.</p>'}
</div>

<div style="font-size:11px;color:#8aa;text-align:center;margin-top:20px;line-height:1.6">
  Field names for this API weren't confirmed against a live response before this was built —
  based on multiple consistent public docs instead. If this stays empty for more than a few
  days after games have finished, that's worth checking directly. Sample sizes are still
  small early on — treat percentages with real caution until there's a few weeks of data.
</div>
</body></html>"""

    os.makedirs(os.path.dirname(DASHBOARD_PATH), exist_ok=True)
    with open(DASHBOARD_PATH, "w") as f:
        f.write(html)
    print(f"  Results dashboard: {total} verified, {overall_pct}% overall" if total else "  Results dashboard: no verified picks yet")


def run_results_tracker(legs):
    """Single entry point called from blue_line.py's main()."""
    print("\nRunning results tracker...")
    log = load_log()
    log = log_todays_signals(legs, log)
    log = verify_pending_results(log)
    save_log(log)
    build_results_dashboard(log)
