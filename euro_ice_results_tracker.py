#!/usr/bin/env python3
"""
Euro Ice Results Tracker
--------------------------------------------------------------
Same architecture as Match IQ / Under IQ's trackers, adapted for
Highlightly's API and Euro Ice's per-TEAM-total leg structure (unlike
football's per-MATCH totals). Logs every qualifying leg + Hot Form/Real
Streak entry into a persistent JSON log, then on LATER runs looks the
match back up via a date+league search (no confirmed single-match-by-ID
endpoint for this API, so this reuses the same list-and-filter approach
already used elsewhere in euro_ice.py) and marks hit/miss against the
real final score.

Designed to be imported and called from euro_ice.py's main() -- save
this file as results_tracker.py in the SportsIQ repo (same folder as
euro_ice.py).

Output:
    docs/euro-ice/results/log.json    -- the full log
    docs/euro-ice/results/index.html  -- dashboard: overall + per-category
                                          win rate, recent history
"""

import os
import json
import hashlib
import requests
from datetime import datetime, timezone

BASE = "https://hockey.highlightly.net"
LOG_PATH = "docs/euro-ice/results/log.json"
DASHBOARD_PATH = "docs/euro-ice/results/index.html"
FINISHED_STATES = {"Finished", "Finished after penalties", "Finished after over time"}


def _get(path, key, params=None):
    headers = {"x-rapidapi-key": key}
    try:
        r = requests.get(f"{BASE}{path}", headers=headers, params=params or {}, timeout=15)
    except Exception as e:
        print(f"    [!] verification request failed: {path} ({e})")
        return None
    if r.status_code != 200:
        print(f"    [!] {r.status_code} on {path}: {r.text[:150]}")
        return None
    return r.json()


def parse_score(score_str):
    if not score_str:
        return None
    parts = score_str.replace(" ", "").split("-")
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def _entry_id(scanner, subject, match_date_key, market):
    raw = f"{scanner}|{subject}|{match_date_key}|{market}"
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


def log_todays_signals(legs, streak_entries, real_streak_entries, log):
    """Team Total legs already carry everything needed for later
    verification (line, is_home, league_id, home/away names, match
    date) -- added specifically for this tracker when the legs were
    built. Game Total legs are skipped here: they combine two teams'
    SEPARATE scoring histories into one number, so there's no single
    real "side" to check the way there is for a team total -- same
    reasoning that already excludes them from hit_rate elsewhere."""
    existing_ids = {e["id"] for e in log}
    added = 0

    def add(scanner, subject, market, value, league, league_id, date, date_key,
             home_name, away_name, is_home=None, detail=None):
        nonlocal added
        eid = _entry_id(scanner, subject, date_key, market)
        if eid in existing_ids:
            return
        log.append({
            "id": eid, "scanner": scanner, "subject": subject, "market": market,
            "value": value, "detail": detail, "is_home": is_home,
            "league": league, "league_id": league_id,
            "home_name": home_name, "away_name": away_name,
            "match_date": date, "date_key": date_key,
            "logged_at": datetime.now(timezone.utc).isoformat(),
            "status": "pending", "result": None, "actual": None,
        })
        existing_ids.add(eid)
        added += 1

    for leg in legs:
        if "Team Total" not in leg.get("category", ""):
            continue  # skip Game Total legs -- see docstring
        league_name = leg["category"].replace(" Team Total", "")
        date_key = (leg.get("match_date") or "")[:10]
        add("team_total", leg["subject"], leg["market"], leg["prob"],
            league_name, leg["league_id"], leg["match_date"], date_key,
            leg["home_name"], leg["away_name"], is_home=leg["is_home"],
            detail=f"line={leg['line']}")

    for e in streak_entries:
        date_key = (e["date"] or "")[:10]
        add("hot_form", e["team"], f"Hot Form vs {e['opponent']}", e["last5_avg"],
            e["league"], e["league_id"], e["date"], date_key,
            e["home_name"], e["away_name"], is_home=e["is_home"])

    for e in real_streak_entries:
        date_key = (e["date"] or "")[:10]
        add("real_streak", e["team"], f"Real Streak vs {e['opponent']}", e["streak_len"],
            e["league"], e["league_id"], e["date"], date_key,
            e["home_name"], e["away_name"], is_home=e["is_home"])

    print(f"  Results log: {added} new pick(s) logged, {len(log)} total in log")
    return log


def _find_finished_match(entry, key):
    """No confirmed single-match-by-ID lookup for this API -- same
    list-and-filter approach already used for get_upcoming_matches,
    just searching by date+league and matching on team names."""
    data = _get("/matches", key, params={"leagueId": entry["league_id"], "date": entry["date_key"]})
    if not data or not data.get("data"):
        return None
    for m in data["data"]:
        if (m.get("homeTeam", {}).get("name") == entry["home_name"] and
                m.get("awayTeam", {}).get("name") == entry["away_name"]):
            if m.get("state", {}).get("description") not in FINISHED_STATES:
                return None
            return m
    return None


def _verify_team_total_entry(entry, key):
    m = _find_finished_match(entry, key)
    if not m:
        return None
    score = parse_score(m.get("state", {}).get("score", {}).get("current"))
    if not score:
        return None
    home_goals, away_goals = score
    team_goals = home_goals if entry["is_home"] else away_goals
    line = float(entry["detail"].split("=")[1])
    return {"actual": team_goals, "result": "hit" if team_goals > line else "miss"}


def _verify_form_streak_entry(entry, key, threshold):
    m = _find_finished_match(entry, key)
    if not m:
        return None
    score = parse_score(m.get("state", {}).get("score", {}).get("current"))
    if not score:
        return None
    home_goals, away_goals = score
    team_goals = home_goals if entry["is_home"] else away_goals
    return {"actual": team_goals, "result": "hit" if team_goals >= threshold else "miss"}


def verify_pending_results(log, key, real_streak_threshold, max_checks=60):
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
            if entry["scanner"] == "team_total":
                result = _verify_team_total_entry(entry, key)
            elif entry["scanner"] in ("hot_form", "real_streak"):
                result = _verify_form_streak_entry(entry, key, real_streak_threshold)
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

    SCANNER_LABELS = {"team_total": "Team Total Goals", "hot_form": "Hot Form", "real_streak": "Real Streak"}

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
        rows += f"""<div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid var(--border)">
  <span>{label}</span>
  <span style="color:var(--green);font-weight:bold">{pct_str}</span>
  <span style="color:var(--sub);font-size:12px">{d['hit']}/{n}</span>
</div>"""

    recent = sorted(verified, key=lambda e: e.get("verified_at", ""), reverse=True)[:30]
    recent_rows = ""
    for e in recent:
        color = "#22c55e" if e["result"] == "hit" else "#ef4444"
        recent_rows += f"""<div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid var(--border);font-size:12px">
  <span>{e['subject']} — {e['market']}</span>
  <span style="color:{color};font-weight:bold">{e['result'].upper()}</span>
</div>"""

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Results — Euro Ice</title>
<style>:root{{--bg:#0b0f14;--panel:#121820;--border:#233040;--text:#e8edf2;--sub:#8b98a8;--green:#22c55e;}}</style></head>
<body style="background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;padding:16px;max-width:640px;margin:0 auto">
<p style="text-align:center;margin-bottom:6px"><a href="../index.html" style="color:#7ec8ff;text-decoration:none;font-size:12px">← Euro Ice</a></p>
<h2 style="text-align:center;margin-bottom:2px">📊 Results Tracker</h2>
<p style="text-align:center;color:var(--sub);font-size:11px;margin-top:0">{datetime.now().strftime("%d %b %H:%M")} · every pick, auto-verified against real results</p>

<div style="background:var(--panel);border-radius:12px;padding:16px;margin:14px 0;border:1px solid var(--border);text-align:center">
  <div style="font-size:11px;color:var(--sub)">OVERALL</div>
  <div style="font-size:32px;font-weight:bold;color:var(--green)">{overall_pct if overall_pct is not None else "—"}{"%" if overall_pct is not None else ""}</div>
  <div style="font-size:12px;color:var(--sub)">{total_hit}/{total} verified picks · {len(pending)} pending (match not finished yet)</div>
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
  Game Total legs aren't tracked here — they combine two teams' separate scoring histories
  into one number, so there's no single real "side" to check against. Hot Form / Real Streak
  are verified against whether the flagged team scored 2+ in the SAME match the signal was
  flagged alongside. Sample sizes are still small early on — treat percentages with real
  caution until there's a few weeks of data.
</div>
</body></html>"""

    os.makedirs(os.path.dirname(DASHBOARD_PATH), exist_ok=True)
    with open(DASHBOARD_PATH, "w") as f:
        f.write(html)
    print(f"  Results dashboard: {total} verified, {overall_pct}% overall" if total else "  Results dashboard: no verified picks yet")


def run_results_tracker(legs, streak_entries, real_streak_entries, key, real_streak_threshold):
    """Single entry point called from euro_ice.py's main()."""
    print("\nRunning results tracker...")
    log = load_log()
    log = log_todays_signals(legs, streak_entries, real_streak_entries, log)
    log = verify_pending_results(log, key, real_streak_threshold)
    save_log(log)
    build_results_dashboard(log)
