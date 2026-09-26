#!/usr/bin/env python3
"""
Cards & Corners IQ Results Tracker
--------------------------------------------------------------
Same pattern as every other tracker in this suite (Match IQ, Strike Zone,
Blitz IQ, Blue Line, Euro Ice, and Horse Racing): log every qualifying
pick as "pending" BEFORE its match kicks off, on later runs check back on
anything whose match has since finished, fetch the REAL result, and mark
it hit/miss -- an honest track record, not a screenshot.

NOTE ON FILENAME: Cards & Corners IQ lives in the shared SportsIQ repo
alongside Match IQ's own results_tracker.py (and Euro Ice's, Strike
Zone's, Blitz IQ's, Blue Line's -- each already renamed to a unique file
per project). This file must stay named cards_corners_results_tracker.py
in that repo -- do NOT rename it to plain results_tracker.py, that name
is already taken by Match IQ's tracker in the same folder.

WHAT GETS LOGGED: every leg cards_corners_iq.py's own build_legs() would
put in the Safest Bet Builder -- i.e. every team's Team Corners line and
Team Cards line for every upcoming match, at the SAME safety-margined
line the page actually shows (line = floor(projected * 0.72, nearest 0.5)
duplicated here rather than imported, per this suite's self-contained-
tool convention -- see cards_corners_iq.py's own module docstring for why
each tool duplicates rather than imports). That means this tracker
answers "does the model's own safety-margined line actually clear in
real matches" for BOTH markets, split out separately so Corners and
Cards can be judged on their own merits (Corners is opponent-adjusted;
Cards deliberately isn't -- see cards_corners_iq.py for why).

HOW RESULTS ARE VERIFIED: TheStatsAPI's /football/matches/{id} for match
status, then /football/matches/{id}/stats for the actual corner_kicks and
yellow_cards split by home/away -- same fields cards_corners_iq.py itself
reads when building the projection, just read post-match instead of pre-
match. A pick is graded once the match is marked "finished" AND the
relevant stat is actually present in that response; if a finished match's
stats never populate corner_kicks or yellow_cards, that entry stays
pending indefinitely rather than being scored on a guess (watch the
dashboard's pending count -- if it only grows, this is the first thing to
check against a real finished match's raw stats response).

Usage (called automatically from cards_corners_iq.py's main block):
    import cards_corners_results_tracker
    cards_corners_results_tracker.run_results_tracker(predictions, api_key)

Output:
    docs/cards-corners/results/log.json
    docs/cards-corners/results/index.html
"""

import os
import math
import json
import hashlib
import requests
from datetime import datetime, timezone

BASE = "https://api.thestatsapi.com/api"
LOG_PATH = "docs/cards-corners/results/log.json"
DASHBOARD_PATH = "docs/cards-corners/results/index.html"

# Caps how many pending entries get a fresh API call in one run, so a
# backlog can't blow through the metered trial tier in a single
# execution -- it just catches up gradually over a few runs instead.
# Note this is PER ENTRY (not per match, unlike Horse Racing's tracker),
# because a Corners pick and a Cards pick for the same team both need
# the same /stats call anyway -- see _verify_entry's small in-run cache.
MAX_CHECKS_PER_RUN = 80


def _headers(key):
    return {"Authorization": f"Bearer {key}"}


def _get(path, key, params=None, timeout=15):
    try:
        r = requests.get(f"{BASE}{path}", headers=_headers(key), params=params or {}, timeout=timeout)
    except Exception as e:
        print(f"    [!] verification request failed: {path} ({e})")
        return None
    if r.status_code != 200:
        print(f"    [!] {r.status_code} on {path}: {r.text[:150]}")
        return None
    return r.json()


def safe_line(lam, factor=0.72, round_to=0.5):
    """Duplicated verbatim from cards_corners_iq.py so the line this
    tracker grades against always matches the line actually shown on the
    page -- see that file's own copy for the reasoning."""
    if lam is None:
        return None
    raw = lam * factor
    line = math.floor(raw / round_to) * round_to
    return max(line, round_to)


def _entry_id(category, match_id, team):
    """Stable dedup key, hashed to a fixed length -- same convention as
    Match IQ's own tracker."""
    raw = f"{category}|{match_id}|{team}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def load_log():
    if not os.path.exists(LOG_PATH):
        return []
    try:
        with open(LOG_PATH) as f:
            return json.load(f)
    except Exception as e:
        print(f"  [!] couldn't read existing results log ({e}) -- starting fresh. "
              f"The old file is still on disk if this needs investigating.")
        return []


def save_log(entries):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "w") as f:
        json.dump(entries, f, indent=2, default=str)


def log_todays_signals(predictions, log):
    """Appends one pending entry per team per market (Corners, Cards)
    for every upcoming match in this run's predictions -- skipping
    anything already logged under the same category+match+team, so
    re-running the same day's build never duplicates rows."""
    existing_ids = {e["id"] for e in log}
    added = 0

    def add(category, match_id, team, is_home, line, lam, league, date, date_key, home_team, away_team):
        nonlocal added
        if line is None:
            return
        eid = _entry_id(category, match_id, team)
        if eid in existing_ids:
            return
        log.append({
            "id": eid, "category": category, "match_id": match_id, "team": team,
            "is_home": is_home, "line": line, "projected": lam, "league": league,
            "match_date": date, "date_key": date_key, "home_team": home_team, "away_team": away_team,
            "logged_at": datetime.now(timezone.utc).isoformat(),
            "status": "pending", "result": None, "actual": None,
        })
        existing_ids.add(eid)
        added += 1

    for p in predictions:
        base = dict(match_id=p["match_id"], league=p["league"], date=p["date"], date_key=p["date_key"],
                    home_team=p["home_team"], away_team=p["away_team"])

        home_corners_line = safe_line(p["home_corners"]["lambda"])
        add("Team Corners", team=p["home_team"], is_home=True,
            line=home_corners_line, lam=p["home_corners"]["lambda"], **base)
        away_corners_line = safe_line(p["away_corners"]["lambda"])
        add("Team Corners", team=p["away_team"], is_home=False,
            line=away_corners_line, lam=p["away_corners"]["lambda"], **base)

        home_cards_line = safe_line(p["home_cards"]["lambda"])
        add("Team Cards", team=p["home_team"], is_home=True,
            line=home_cards_line, lam=p["home_cards"]["lambda"], **base)
        away_cards_line = safe_line(p["away_cards"]["lambda"])
        add("Team Cards", team=p["away_team"], is_home=False,
            line=away_cards_line, lam=p["away_cards"]["lambda"], **base)

    print(f"  Results log: {added} new pick(s) logged, {len(log)} total in log")
    return log


# In-run cache so a match with both a Corners and a Cards pending entry
# (the normal case -- every team gets both) only costs ONE /matches call
# and ONE /stats call for that match per run, not two.
_match_cache = {}
_stats_cache = {}


def _finished_match(match_id, key):
    if match_id not in _match_cache:
        data = _get(f"/football/matches/{match_id}", key)
        m = data["data"] if data and data.get("data") else None
        _match_cache[match_id] = m if m and m.get("status") == "finished" else None
    return _match_cache[match_id]


def _match_stats(match_id, key):
    if match_id not in _stats_cache:
        data = _get(f"/football/matches/{match_id}/stats", key)
        _stats_cache[match_id] = data["data"]["overview"] if data and data.get("data") else None
    return _stats_cache[match_id]


def _verify_entry(entry, key):
    m = _finished_match(entry["match_id"], key)
    if not m:
        return None  # not marked finished yet -- leave pending, try again next run

    overview = _match_stats(entry["match_id"], key)
    if not overview:
        return None

    stat_key = "corner_kicks" if entry["category"] == "Team Corners" else "yellow_cards"
    item = overview.get(stat_key, {}).get("all")
    if not item:
        return None  # stat never populated for this match -- stays pending, see module docstring

    side_val = item["home"] if entry["is_home"] else item["away"]
    if side_val is None:
        return None

    hit = side_val > entry["line"]
    return {"actual": side_val, "result": "hit" if hit else "miss"}


def verify_pending_results(log, key, max_checks=MAX_CHECKS_PER_RUN):
    """Looks at every PENDING entry whose match date has already passed
    (so the game should be over by now), tries to fetch the real result,
    and updates it in place. Capped per run so a backlog can't blow
    through the metered API tier in one go."""
    today = datetime.now(timezone.utc).date().isoformat()
    checked = 0
    updated = 0

    for entry in log:
        if entry["status"] != "pending":
            continue
        if entry["date_key"] >= today:
            continue  # match hasn't happened yet
        if checked >= max_checks:
            break
        checked += 1

        try:
            result = _verify_entry(entry, key)
        except Exception as e:
            print(f"    [!] verification error for entry {entry['id']} ({entry['category']}): {e}")
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

    by_cat = {}
    for e in verified:
        d = by_cat.setdefault(e["category"], {"hit": 0, "miss": 0})
        d[e["result"]] += 1

    CATEGORY_ORDER = ["Team Corners", "Team Cards"]
    CATEGORY_NOTE = {
        "Team Corners": "opponent-adjusted projection · safety-margined line",
        "Team Cards": "yellow cards only, no opponent adjustment · safety-margined line",
    }

    total_hit = sum(d["hit"] for d in by_cat.values())
    total_miss = sum(d["miss"] for d in by_cat.values())
    total = total_hit + total_miss
    overall_pct = round(100 * total_hit / total) if total else None

    rows = ""
    for cat in CATEGORY_ORDER:
        d = by_cat.get(cat, {"hit": 0, "miss": 0})
        n = d["hit"] + d["miss"]
        pct = round(100 * d["hit"] / n) if n else None
        pct_str = f"{pct}%" if pct is not None else "—"
        rows += f"""<div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #233040">
  <div><b>{cat}</b><br><span style="color:#8b98a8;font-size:11px">{CATEGORY_NOTE.get(cat, "")}</span></div>
  <div style="text-align:right"><span style="color:#a0e8a0;font-weight:bold;font-size:16px">{pct_str}</span><br><span style="color:#8b98a8;font-size:11px">{d['hit']}/{n}</span></div>
</div>"""

    recent = sorted(verified, key=lambda e: e.get("verified_at", ""), reverse=True)[:30]
    recent_rows = ""
    for e in recent:
        color = "#22c55e" if e["result"] == "hit" else "#ef4444"
        recent_rows += f"""<div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #233040;font-size:12px">
  <div><b>{e['team']}</b><br><span style="color:#8b98a8">{e['category']} · line {e['line']} · actual {e['actual']}</span></div>
  <span style="color:{color};font-weight:bold">{e['result'].upper()}</span>
</div>"""

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Results — Cards &amp; Corners IQ</title></head>
<body style="background:#0b0f14;color:white;font-family:Arial;padding:12px;max-width:600px;margin:auto">
<h2 style="text-align:center;margin-bottom:2px">📋 Results Tracker</h2>
<p style="text-align:center;color:#888;font-size:11px;margin-top:0">{datetime.now().strftime("%d %b %H:%M")} · every pick, auto-verified against real results</p>

<div style="background:#1a1f26;border-radius:12px;padding:16px;margin:14px 0;border:1px solid #2a3038;text-align:center">
  <div style="font-size:11px;color:#888">OVERALL</div>
  <div style="font-size:32px;font-weight:bold;color:#a0e8a0">{overall_pct if overall_pct is not None else "—"}{"%" if overall_pct is not None else ""}</div>
  <div style="font-size:12px;color:#8b98a8">{total_hit}/{total} verified picks · {len(pending)} pending (match not finished yet)</div>
</div>

<div style="background:#1a1f26;border-radius:12px;padding:16px;margin:14px 0;border:1px solid #2a3038">
  <div style="font-weight:bold;margin-bottom:8px">By Market</div>
  {rows or '<p style="color:#888;text-align:center">No graded picks yet.</p>'}
</div>

<div style="background:#1a1f26;border-radius:12px;padding:16px;margin:14px 0;border:1px solid #2a3038">
  <div style="font-weight:bold;margin-bottom:8px">Recent Results</div>
  {recent_rows or '<p style="color:#888;font-size:12px">Nothing verified yet — check back after a few days of picks have had time to play out.</p>'}
</div>

<div style="font-size:11px;color:#8b98a8;text-align:center;margin-top:20px;line-height:1.6">
  Every pick is logged BEFORE its match kicks off, against the exact same safety-margined line
  shown on the page — nothing here is cherry-picked after the fact. Team Corners and Team Cards
  are tracked separately since Corners is opponent-adjusted and Cards isn't; if one market's win
  rate consistently lags the other, that's the signal to revisit that market's line factor first.
</div>
</body></html>"""

    os.makedirs(os.path.dirname(DASHBOARD_PATH), exist_ok=True)
    with open(DASHBOARD_PATH, "w") as f:
        f.write(html)
    print(f"  Results dashboard: {total} verified, {overall_pct}% overall" if total else "  Results dashboard: no verified picks yet")


def run_results_tracker(predictions, key):
    """Single entry point called from cards_corners_iq.py's main block."""
    print("\nRunning results tracker...")
    log = load_log()
    log = log_todays_signals(predictions, log)
    log = verify_pending_results(log, key)
    save_log(log)
    build_results_dashboard(log)
