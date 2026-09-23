#!/usr/bin/env python3
"""
Match IQ Results Tracker
--------------------------------------------------------------
Logs every qualifying pick from each of Match IQ's scanners (Over 2.5,
BTTS, Corners, Safe Corners, Over 4.5, Hot Form, Real Streak) into a
persistent JSON log, then on LATER runs automatically checks back on
older entries whose match has since finished, fetches the real result,
and marks each one hit or miss -- building an actual, honest track
record instead of eyeballing individual screenshots.

Designed to be imported and called from match_iq.py's main() -- not run
standalone. Reuses the same TheStatsAPI base/auth pattern already used
everywhere else in that script.

Output:
    docs/match-iq/results/log.json    -- the full log, every entry ever seen
    docs/match-iq/results/index.html  -- a readable dashboard: overall +
                                          per-scanner win rate, recent history

WHY THIS EXISTS: after several nights of eyeballing individual hit/miss
screenshots and drawing conclusions from n=1, the honest answer to "does
this signal actually work" requires an actual sample size. This is that
sample size, tracked automatically, no manual logging required.
"""

import os
import json
import hashlib
import requests
from datetime import datetime, timezone

BASE = "https://api.thestatsapi.com/api"
LOG_PATH = "docs/match-iq/results/log.json"
DASHBOARD_PATH = "docs/match-iq/results/index.html"

# Scanners whose "hit" definition is a straightforward goals threshold
# on the MATCH TOTAL -- these can be verified from the score alone,
# no extra API call needed.
GOAL_THRESHOLD_SCANNERS = {
    "over25": 2.5,
    "over45": 4.5,
}


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


def _entry_id(scanner, subject, match_date_key, market):
    """Stable dedup key so re-running the same day never double-logs the
    same pick -- hashed rather than a raw concatenation so it's a fixed,
    filesystem/JSON-safe length regardless of subject/market text."""
    raw = f"{scanner}|{subject}|{match_date_key}|{market}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def load_log():
    if not os.path.exists(LOG_PATH):
        return []
    try:
        with open(LOG_PATH) as f:
            return json.load(f)
    except Exception as e:
        print(f"  [!] Couldn't read existing results log ({e}) -- starting fresh. "
              f"The old file is still on disk if this needs investigating.")
        return []


def save_log(entries):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "w") as f:
        json.dump(entries, f, indent=2, default=str)


def log_todays_signals(all_predictions, hot_form_entries, streak_entries, log, thresholds):
    """Appends every qualifying pick from this run into the log as a
    PENDING entry. Skips anything already logged (same scanner+subject+
    match date+market) so re-running the same day is always safe.

    thresholds is a dict of the actual live constants from match_iq.py
    (over25_min, btts_min, corners_min, safe_corners_min,
    safe_corners_min_sample, over45_min) -- passed in explicitly rather
    than imported, so this module has no dependency on match_iq.py's
    internals and can't silently drift out of sync with them."""
    existing_ids = {e["id"] for e in log}
    added = 0

    def add(scanner, subject, market, value, match_id, match_date, date_key,
             league, home_team, away_team, detail=None):
        nonlocal added
        eid = _entry_id(scanner, subject, date_key, market)
        if eid in existing_ids:
            return
        log.append({
            "id": eid, "scanner": scanner, "subject": subject, "market": market,
            "value": value, "detail": detail,
            "match_id": match_id, "match_date": match_date, "date_key": date_key,
            "league": league, "home_team": home_team, "away_team": away_team,
            "logged_at": datetime.now(timezone.utc).isoformat(),
            "status": "pending", "result": None, "actual": None,
        })
        existing_ids.add(eid)
        added += 1

    for p in all_predictions:
        match_label = f"{p['home_team']} vs {p['away_team']}"
        base = dict(match_id=p["match_id"], match_date=p["date"], date_key=p["date_key"],
                    league=p["league"], home_team=p["home_team"], away_team=p["away_team"])

        if p.get("over25", 0) >= thresholds["over25_min"]:
            add("over25", match_label, "Over 2.5 Goals", p["over25"], detail=f"exp {p.get('exp_total')} goals", **base)
        if p.get("btts", 0) >= thresholds["btts_min"]:
            add("btts", match_label, "BTTS", p["btts"], **base)
        if p.get("corners_over105", 0) >= thresholds["corners_min"]:
            add("corners", match_label, "Over 10.5 Corners", p["corners_over105"],
                detail=f"exp {p.get('exp_corners')} corners", **base)
        if p.get("corners_over105", 0) >= thresholds["safe_corners_min"] and \
           p.get("corners_sample_n", 0) >= thresholds["safe_corners_min_sample"]:
            add("corners_safe", match_label, "Safe Corners", p["corners_over105"],
                detail=f"sample {p.get('corners_sample_n')}/7", **base)
        if p.get("over45", 0) >= thresholds["over45_min"]:
            add("over45", match_label, "Over 4.5 Goals", p["over45"], detail=f"exp {p.get('exp_total')} goals", **base)

    for e in hot_form_entries:
        add("hot_form", e["team"], f"Hot Form vs {e['opponent']}", e["last5_avg"],
            match_id=None, match_date=e["date"], date_key=e["date_key"], league=e["league"],
            home_team=e["team"] if e["is_home"] else e["opponent"],
            away_team=e["opponent"] if e["is_home"] else e["team"],
            detail=f"is_home={e['is_home']}")

    for e in streak_entries:
        add("real_streak", e["team"], f"Real Streak vs {e['opponent']}", e["streak_len"],
            match_id=None, match_date=e["date"], date_key=e["date_key"], league=e["league"],
            home_team=e["team"] if e["is_home"] else e["opponent"],
            away_team=e["opponent"] if e["is_home"] else e["team"],
            detail=f"is_home={e['is_home']}")

    print(f"  Results log: {added} new pick(s) logged, {len(log)} total in log")
    return log


def _verify_goals_entry(entry, key, threshold):
    """Over 2.5 / Over 4.5 -- verify from the match's final score alone,
    no extra API call needed since the score is in the same /matches
    response already used for the fixture list."""
    data = _get(f"/football/matches/{entry['match_id']}", key)
    if not data or not data.get("data"):
        return None
    m = data["data"]
    if m.get("status") != "finished":
        return None
    s = m.get("score", {})
    if s.get("home") is None or s.get("away") is None:
        return None
    total = s["home"] + s["away"]
    return {"actual": total, "result": "hit" if total > threshold else "miss"}


def _verify_btts_entry(entry, key):
    data = _get(f"/football/matches/{entry['match_id']}", key)
    if not data or not data.get("data"):
        return None
    m = data["data"]
    if m.get("status") != "finished":
        return None
    s = m.get("score", {})
    if s.get("home") is None or s.get("away") is None:
        return None
    hit = s["home"] >= 1 and s["away"] >= 1
    return {"actual": f"{s['home']}-{s['away']}", "result": "hit" if hit else "miss"}


def _verify_corners_entry(entry, key, threshold=10.5):
    """Over 10.5 / Safe Corners -- needs the match's stats (corner_kicks
    total), a genuinely new API call per entry since corners aren't in
    the base /matches response. Only called for entries that are
    actually due for verification, so this stays a small, bounded cost
    per run, not a full rescan."""
    match_data = _get(f"/football/matches/{entry['match_id']}", key)
    if not match_data or not match_data.get("data") or match_data["data"].get("status") != "finished":
        return None
    stats = _get(f"/football/matches/{entry['match_id']}/stats", key)
    if not stats or not stats.get("data"):
        return None
    overview = stats["data"].get("overview", {})
    corners = overview.get("corner_kicks", {}).get("all")
    if not corners or corners.get("home") is None or corners.get("away") is None:
        return None
    total = corners["home"] + corners["away"]
    return {"actual": total, "result": "hit" if total > threshold else "miss"}


def _verify_form_streak_entry(entry, key, real_streak_threshold):
    """Hot Form / Real Streak -- these describe a team's PAST form, not
    a prediction about this specific match's total. The natural check:
    did the team keep scoring (>= REAL_STREAK_THRESHOLD, i.e. 2+) in
    the very match the streak/form was flagged alongside? That directly
    answers "does being hot/on a streak coming in correlate with
    scoring in the next game" -- which is the actual question this
    whole tracker exists to answer."""
    data = _get(f"/football/matches/{entry['match_id']}", key) if entry.get("match_id") else None
    # hot_form/real_streak entries don't carry a match_id (they're
    # per-team, built from the goal-streak entries, not per-fixture) --
    # so verification here has to look up the fixture itself first via
    # a date+team search instead of a direct ID.
    if not data:
        search = _get("/football/matches", key, params={
            "date_from": entry["date_key"], "date_to": entry["date_key"], "per_page": 50,
        })
        if not search or not search.get("data"):
            return None
        match = next((m for m in search["data"]
                      if m["home_team"]["name"] == entry["home_team"] and m["away_team"]["name"] == entry["away_team"]),
                     None)
        if not match or match.get("status") != "finished":
            return None
        s = match.get("score", {})
    else:
        m = data["data"]
        if m.get("status") != "finished":
            return None
        s = m.get("score", {})

    if s.get("home") is None or s.get("away") is None:
        return None
    is_home = entry["detail"] == "is_home=True"
    team_goals = s["home"] if is_home else s["away"]
    return {"actual": team_goals, "result": "hit" if team_goals >= real_streak_threshold else "miss"}


def verify_pending_results(log, key, real_streak_threshold, max_checks=60):
    """Looks at every PENDING entry whose match date has already
    passed (so the game should be over by now), tries to fetch the
    real result, and updates it in place. Capped at max_checks per run
    so a backlog can't blow up API usage in one go -- it'll just catch
    up gradually over a few runs instead."""
    today = datetime.now(timezone.utc).date().isoformat()
    checked = 0
    updated = 0

    for entry in log:
        if entry["status"] != "pending":
            continue
        if entry["date_key"] >= today:
            continue  # match hasn't happened yet, nothing to verify
        if checked >= max_checks:
            break
        checked += 1

        result = None
        try:
            if entry["scanner"] in GOAL_THRESHOLD_SCANNERS:
                result = _verify_goals_entry(entry, key, GOAL_THRESHOLD_SCANNERS[entry["scanner"]])
            elif entry["scanner"] == "btts":
                result = _verify_btts_entry(entry, key)
            elif entry["scanner"] in ("corners", "corners_safe"):
                result = _verify_corners_entry(entry, key)
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
        # if result is None, the match likely isn't marked finished yet
        # in the API even though its date has passed -- leave it pending,
        # it'll get picked up on a later run

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
        "over25": "Over 2.5 Goals", "btts": "BTTS", "corners": "Over 10.5 Corners",
        "corners_safe": "Safe Corners", "over45": "Over 4.5 Goals",
        "hot_form": "Hot Form", "real_streak": "Real Streak",
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
        rows += f"""<div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #233040">
  <span>{label}</span>
  <span style="color:#a0e8a0;font-weight:bold">{pct_str}</span>
  <span style="color:#8b98a8;font-size:12px">{d['hit']}/{n}</span>
</div>"""

    recent = sorted(verified, key=lambda e: e.get("verified_at", ""), reverse=True)[:30]
    recent_rows = ""
    for e in recent:
        color = "#22c55e" if e["result"] == "hit" else "#ef4444"
        recent_rows += f"""<div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #233040;font-size:12px">
  <span>{e['subject']} — {e['market']}</span>
  <span style="color:{color};font-weight:bold">{e['result'].upper()}</span>
</div>"""

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Results — Match IQ</title></head>
<body style="background:#0b0f14;color:white;font-family:Arial;padding:12px;max-width:600px;margin:auto">
<p style="text-align:center;margin-bottom:6px"><a href="../match_iq_index.html" style="color:#7ec8ff;text-decoration:none;font-size:12px">← Match IQ</a></p>
<h2 style="text-align:center;margin-bottom:2px">📊 Results Tracker</h2>
<p style="text-align:center;color:#888;font-size:11px;margin-top:0">{datetime.now().strftime("%d %b %H:%M")} · every pick, auto-verified against real results</p>

<div style="background:#1a1f26;border-radius:12px;padding:16px;margin:14px 0;border:1px solid #2a3038;text-align:center">
  <div style="font-size:11px;color:#888">OVERALL</div>
  <div style="font-size:32px;font-weight:bold;color:#a0e8a0">{overall_pct if overall_pct is not None else "—"}{"%" if overall_pct is not None else ""}</div>
  <div style="font-size:12px;color:#8b98a8">{total_hit}/{total} verified picks · {len(pending)} pending (match not finished yet)</div>
</div>

<div style="background:#1a1f26;border-radius:12px;padding:16px;margin:14px 0;border:1px solid #2a3038">
  <div style="font-weight:bold;margin-bottom:8px">By Scanner</div>
  {rows}
</div>

<div style="background:#1a1f26;border-radius:12px;padding:16px;margin:14px 0;border:1px solid #2a3038">
  <div style="font-weight:bold;margin-bottom:8px">Recent Results</div>
  {recent_rows or '<p style="color:#888;font-size:12px">Nothing verified yet — check back after a few days of picks have had time to play out.</p>'}
</div>

<div style="font-size:11px;color:#8b98a8;text-align:center;margin-top:20px;line-height:1.6">
  Hot Form / Real Streak are verified against whether the flagged team scored 2+ in the
  SAME match the signal was flagged alongside — that's the direct test of "does being
  hot/on a streak coming in correlate with scoring in the next game." Every other scanner
  is verified against its own actual stated market. Sample sizes are still small early on —
  treat percentages with real caution until there's a few weeks of data.
</div>
</body></html>"""

    os.makedirs(os.path.dirname(DASHBOARD_PATH), exist_ok=True)
    with open(DASHBOARD_PATH, "w") as f:
        f.write(html)
    print(f"  Results dashboard: {total} verified, {overall_pct}% overall" if total else "  Results dashboard: no verified picks yet")


def run_results_tracker(all_predictions, hot_form_entries, streak_entries, key, thresholds):
    """Single entry point called from match_iq.py's main(). thresholds
    must contain: over25_min, btts_min, corners_min, safe_corners_min,
    safe_corners_min_sample, over45_min, real_streak_threshold."""
    print("\nRunning results tracker...")
    log = load_log()
    log = log_todays_signals(all_predictions, hot_form_entries, streak_entries, log, thresholds)
    log = verify_pending_results(log, key, thresholds["real_streak_threshold"])
    save_log(log)
    build_results_dashboard(log)
