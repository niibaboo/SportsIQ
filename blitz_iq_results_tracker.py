#!/usr/bin/env python3
"""
Blitz IQ Results Tracker
--------------------------------------------------------------
Same architecture as the other five trackers, adapted for ESPN's public
NFL API. Team Total / Game Total legs reuse the EXACT SAME score.value +
status.type.completed pattern nfl_model.py already relies on for team
form -- not a guess, genuinely proven logic from this same codebase.

Player props (Passing/Rushing Yards, Receptions) are less certain: ESPN's
per-game boxscore player-stat shape isn't confirmed in this session, only
that /summary?event={id} returns a .boxscore key containing it somewhere.
Rather than assume one shape, this tries a few plausible ones (mirroring
nfl_model.py's OWN existing defensive style for the same API's gamelog
endpoint) and prints a diagnostic if none match, so a wrong guess fails
safely and visibly instead of silently producing garbage.

Designed to be imported and called from nfl_model.py's main() -- save
this file as blitz_iq_results_tracker.py alongside nfl_model.py (this
repo has several projects sharing one root folder, so this name is
deliberately distinct -- do NOT rename it to results_tracker.py).

Output:
    docs/blitz-iq/results/log.json    -- the full log
    docs/blitz-iq/results/index.html  -- dashboard: overall + per-category
                                          win rate, recent history
"""

import os
import json
import hashlib
import requests
from datetime import datetime, timezone

SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary"
LOG_PATH = "docs/blitz-iq/results/log.json"
DASHBOARD_PATH = "docs/blitz-iq/results/index.html"

SUMMARY_CACHE = {}  # one fetch per game_id per run, several legs share a game

CATEGORY_SCANNER = {
    "Team Total": "team_total", "Game Total": "game_total",
    "Passing Yards": "passing_yards", "Rushing Yards": "rushing_yards",
    "Receptions": "receptions", "Anytime TD": "anytime_touchdown",
}
STAT_LABEL_TO_SCANNER = {"YDS": None, "REC": "receptions"}  # YDS is ambiguous
                                                                # (passing vs rushing) --
                                                                # resolved via category instead

# Hot Form / Real Streak scanners (added 2026-09-27) -- these don't come
# from build_legs()'s betting-line legs at all, they come from
# blitz_iq.py's build_team_form_entries()/build_player_form_entries().
# Verified differently too: not "did the actual value clear a betting
# line", but "did the team/player clear the SAME threshold that got them
# flagged, in the very game the flag was made for" -- the direct test of
# whether being hot/streaking coming in says anything about the next
# game, same question Match IQ's own tracker asks of its hot_form/
# real_streak categories.
FORM_SCANNERS = {"team_hot_form", "team_real_streak", "player_hot_form", "player_real_streak"}


def _get(params):
    try:
        r = requests.get(SUMMARY_URL, params=params, timeout=20)
    except Exception as e:
        print(f"    [!] verification request failed: event={params.get('event')} ({e})")
        return None
    if r.status_code != 200:
        print(f"    [!] {r.status_code} on summary for event={params.get('event')}")
        return None
    return r.json()


def _get_summary(game_id):
    if game_id in SUMMARY_CACHE:
        return SUMMARY_CACHE[game_id]
    data = _get({"event": game_id})
    SUMMARY_CACHE[game_id] = data
    return data


def _entry_id(scanner, market, date_key, match):
    raw = f"{scanner}|{market}|{date_key}|{match}"
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


def log_todays_signals(legs, team_hot_form, team_real_streak, player_hot_form, player_real_streak, log):
    """Legs already carry game_id, game_date, is_home, line, and (for
    player props) athlete_id/stat_key -- added specifically for this
    tracker when the legs were built.

    team_hot_form/team_real_streak/player_hot_form/player_real_streak are
    the entries from blitz_iq.py's build_team_form_entries()/
    build_player_form_entries() -- a different shape from legs (no
    betting line/prob, but a 'threshold' each entry was flagged against),
    so they're logged through their own path below rather than being
    forced into the leg-shaped block above."""
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
            "id": eid, "scanner": scanner, "match": leg["match"], "market": leg["market"],
            "value": leg["prob"], "detail": leg.get("detail"), "line": leg.get("line"),
            "game_id": leg["game_id"], "game_date": leg["game_date"], "date_key": leg["game_date"],
            "is_home": leg.get("is_home"), "athlete_id": leg.get("athlete_id"),
            "stat_key": leg.get("stat_key"),
            "logged_at": datetime.now(timezone.utc).isoformat(),
            "status": "pending", "result": None, "actual": None,
        })
        existing_ids.add(eid)
        added += 1

    def _add_form(scanner, subject, match, date_key, game_id, threshold, market, extra=None):
        nonlocal added
        if game_id is None:
            return  # can't verify a pick with no game to look up later
        eid = _entry_id(scanner, subject, date_key, match)
        if eid in existing_ids:
            return
        entry = {
            "id": eid, "scanner": scanner, "match": match, "market": market,
            "value": None, "detail": None, "line": None, "threshold": threshold,
            "game_id": game_id, "game_date": date_key, "date_key": date_key,
            "logged_at": datetime.now(timezone.utc).isoformat(),
            "status": "pending", "result": None, "actual": None,
        }
        if extra:
            entry.update(extra)
        log.append(entry)
        existing_ids.add(eid)
        added += 1

    for e in (team_hot_form or []):
        date_key = (e.get("date") or "")[:10]
        _add_form("team_hot_form", e["team"], e["match"], date_key, e.get("game_id"), e["threshold"],
                  f"{e['team']} Hot Form (avg {e['avg']} pts, {e['n_games']}gm)",
                  extra={"is_home": e.get("is_home")})

    for e in (team_real_streak or []):
        date_key = (e.get("date") or "")[:10]
        _add_form("team_real_streak", e["team"], e["match"], date_key, e.get("game_id"), e["threshold"],
                  f"{e['team']} Real Streak ({e['streak']}+ straight {e['threshold']}+ pt games)",
                  extra={"is_home": e.get("is_home")})

    for e in (player_hot_form or []):
        date_key = (e.get("date") or "")[:10]
        _add_form("player_hot_form", e["name"], e["match"], date_key, e.get("game_id"), e["threshold"],
                  f"{e['name']} Hot Form (avg {e['avg']} {e['label'].lower()}, {e['n_games']}gm)",
                  extra={"athlete_id": e.get("athlete_id"), "stat_key": e.get("stat_key"), "label": e.get("label")})

    for e in (player_real_streak or []):
        date_key = (e.get("date") or "")[:10]
        _add_form("player_real_streak", e["name"], e["match"], date_key, e.get("game_id"), e["threshold"],
                  f"{e['name']} Real Streak ({e['streak']}+ straight games ≥{e['threshold']} {e['label'].lower()})",
                  extra={"athlete_id": e.get("athlete_id"), "stat_key": e.get("stat_key"), "label": e.get("label")})

    print(f"  Results log: {added} new pick(s) logged, {len(log)} total in log")
    return log


def _is_final(summary):
    status = (summary.get("header", {}).get("competitions", [{}])[0]
              .get("status", {}).get("type", {}))
    return bool(status.get("completed"))


def _team_scores(summary):
    comps = summary.get("header", {}).get("competitions", [{}])[0].get("competitors", [])
    home = next((c for c in comps if c.get("homeAway") == "home"), None)
    away = next((c for c in comps if c.get("homeAway") == "away"), None)
    if not home or not away:
        return None, None
    try:
        return float(home.get("score")), float(away.get("score"))
    except (TypeError, ValueError):
        return None, None


def _verify_team_leg(entry):
    summary = _get_summary(entry["game_id"])
    if not summary or not _is_final(summary):
        return None
    home_score, away_score = _team_scores(summary)
    if home_score is None or away_score is None:
        return None
    if entry["scanner"] == "game_total":
        actual = home_score + away_score
    else:
        actual = home_score if entry["is_home"] else away_score
    return {"actual": actual, "result": "hit" if actual > entry["line"] else "miss"}


def _find_player_boxscore_stat(summary, athlete_id, stat_key, prefer_keyword=None):
    """DEFENSIVE, multi-shape attempt -- same philosophy as
    nfl_model.py's own _fetch_gamelog_values for this API. ESPN's
    boxscore.players[] typically holds one entry per TEAM, each with a
    'statistics' list of categories (passing/rushing/receiving), each
    category having parallel 'labels' and per-athlete 'stats' arrays --
    but this hasn't been confirmed against a live response in this
    session, so every step here is guarded and prints a diagnostic
    rather than assuming silently.

    prefer_keyword disambiguates the SAME issue nfl_model.py's own
    comments already flag for 'YDS' (a QB's passing yards and an RB's
    rushing yards can both carry that label) -- when a category name
    is available, this prefers the category whose name matches the
    expected stat type (e.g. "passing" for a Passing Yards leg) before
    falling back to whichever category matches first."""
    players = summary.get("boxscore", {}).get("players", [])
    if not players:
        print(f"    [DIAG] boxscore has no 'players' key or it's empty. "
              f"Top-level boxscore keys: {list(summary.get('boxscore', {}).keys())}")
        return None

    fallback_match = None
    for team_block in players:
        for stat_category in team_block.get("statistics", []):
            labels = stat_category.get("labels") or stat_category.get("names")
            athletes = stat_category.get("athletes", [])
            if not labels or stat_key not in labels:
                continue
            idx = labels.index(stat_key)
            cat_name = (stat_category.get("name") or stat_category.get("displayName") or "").lower()
            for a in athletes:
                athlete_info = a.get("athlete", {})
                if str(athlete_info.get("id")) != str(athlete_id):
                    continue
                stats = a.get("stats", [])
                try:
                    value = float(stats[idx])
                except (IndexError, ValueError, TypeError):
                    continue
                if prefer_keyword and prefer_keyword in cat_name:
                    return value  # exact category match, no ambiguity
                if fallback_match is None:
                    fallback_match = value  # keep the first match as a fallback only

    if fallback_match is not None:
        return fallback_match

    print(f"    [DIAG] couldn't find athlete {athlete_id}'s '{stat_key}' stat in boxscore -- "
          f"either the shape assumed here doesn't match ESPN's real response, or this player "
          f"didn't record that stat category this game. Team blocks found: {len(players)}")
    return None


SCANNER_CATEGORY_KEYWORD = {
    "passing_yards": "passing", "rushing_yards": "rushing", "receptions": "receiving",
}


def _find_player_any_td(summary, athlete_id):
    """Anytime TD needs this player's TOTAL touchdowns across every
    non-passing category in the game (rushing + receiving -- a TD is a
    TD whichever way it came in, but a thrown TD doesn't count), unlike
    _find_player_boxscore_stat above which returns a single preferred
    category's value. Same defensive labels/athletes walk, just summed
    across every matching category instead of returning the first hit."""
    players = summary.get("boxscore", {}).get("players", [])
    if not players:
        print(f"    [DIAG] boxscore has no 'players' key or it's empty. "
              f"Top-level boxscore keys: {list(summary.get('boxscore', {}).keys())}")
        return None
    total = 0.0
    found = False
    for team_block in players:
        for stat_category in team_block.get("statistics", []):
            cat_name = (stat_category.get("name") or stat_category.get("displayName") or "").lower()
            if "pass" in cat_name:
                continue  # a thrown TD doesn't pay out an Anytime TD bet
            labels = stat_category.get("labels") or stat_category.get("names")
            athletes = stat_category.get("athletes", [])
            if not labels or "TD" not in labels:
                continue
            idx = labels.index("TD")
            for a in athletes:
                athlete_info = a.get("athlete", {})
                if str(athlete_info.get("id")) != str(athlete_id):
                    continue
                stats = a.get("stats", [])
                try:
                    total += float(stats[idx])
                    found = True
                except (IndexError, ValueError, TypeError):
                    continue
    if not found:
        print(f"    [DIAG] couldn't find athlete {athlete_id}'s TD stat in any non-passing "
              f"boxscore category -- either this player didn't appear in one this game, or "
              f"the shape assumed here doesn't match ESPN's real response.")
        return None
    return total


def _verify_player_leg(entry):
    if not entry.get("athlete_id") or not entry.get("stat_key"):
        return None
    summary = _get_summary(entry["game_id"])
    if not summary or not _is_final(summary):
        return None
    if entry["scanner"] == "anytime_touchdown":
        actual = _find_player_any_td(summary, entry["athlete_id"])
    else:
        actual = _find_player_boxscore_stat(
            summary, entry["athlete_id"], entry["stat_key"],
            prefer_keyword=SCANNER_CATEGORY_KEYWORD.get(entry["scanner"]),
        )
    if actual is None:
        return None
    return {"actual": actual, "result": "hit" if actual > entry["line"] else "miss"}


def _verify_team_form_entry(entry):
    """Hot Form / Real Streak for a TEAM -- checks whether that team's
    actual score in the very game the pick was flagged for cleared the
    same threshold (24+ pts by default) that got them flagged in the
    first place. Reuses _get_summary/_is_final/_team_scores exactly as
    the team_total/game_total scanners already do."""
    summary = _get_summary(entry["game_id"])
    if not summary or not _is_final(summary):
        return None
    home_score, away_score = _team_scores(summary)
    if home_score is None or away_score is None:
        return None
    actual = home_score if entry.get("is_home") else away_score
    return {"actual": actual, "result": "hit" if actual >= entry["threshold"] else "miss"}


PREFER_KEYWORD_BY_LABEL = {"Passing Yards": "passing", "Rushing Yards": "rushing", "Receptions": "receiving"}


def _verify_player_form_entry(entry):
    """Hot Form / Real Streak for a PLAYER -- same boxscore lookup as
    _verify_player_leg, just checked against the flagged threshold
    instead of a betting line."""
    if not entry.get("athlete_id") or not entry.get("stat_key"):
        return None
    summary = _get_summary(entry["game_id"])
    if not summary or not _is_final(summary):
        return None
    if entry.get("label") == "Anytime TD":
        actual = _find_player_any_td(summary, entry["athlete_id"])
    else:
        actual = _find_player_boxscore_stat(
            summary, entry["athlete_id"], entry["stat_key"],
            prefer_keyword=PREFER_KEYWORD_BY_LABEL.get(entry.get("label")),
        )
    if actual is None:
        return None
    return {"actual": actual, "result": "hit" if actual >= entry["threshold"] else "miss"}


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
            elif entry["scanner"] in ("passing_yards", "rushing_yards", "receptions", "anytime_touchdown"):
                result = _verify_player_leg(entry)
            elif entry["scanner"] in ("team_hot_form", "team_real_streak"):
                result = _verify_team_form_entry(entry)
            elif entry["scanner"] in ("player_hot_form", "player_real_streak"):
                result = _verify_player_form_entry(entry)
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
        "passing_yards": "Passing Yards", "rushing_yards": "Rushing Yards",
        "receptions": "Receptions", "anytime_touchdown": "Anytime TD",
        "team_hot_form": "Team Hot Form", "team_real_streak": "Team Real Streak",
        "player_hot_form": "Player Hot Form", "player_real_streak": "Player Real Streak",
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
        rows += f"""<div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #2a3038">
  <span>{label}</span><span style="color:#ffeb3b;font-weight:bold">{pct_str}</span>
  <span style="color:#888;font-size:12px">{d['hit']}/{n}</span>
</div>"""

    recent = sorted(verified, key=lambda e: e.get("verified_at", ""), reverse=True)[:30]
    recent_rows = ""
    for e in recent:
        color = "#22c55e" if e["result"] == "hit" else "#ef4444"
        recent_rows += f"""<div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #2a3038;font-size:12px">
  <span>{e['match']} — {e['market']}</span><span style="color:{color};font-weight:bold">{e['result'].upper()}</span>
</div>"""

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Results — Blitz IQ</title></head>
<body style="background:#0b0f14;color:white;font-family:Arial;padding:12px;max-width:600px;margin:auto">
<p style="text-align:center;margin-bottom:6px"><a href="../index.html" style="color:#7ec8ff;text-decoration:none;font-size:12px">← Blitz IQ</a></p>
<h2 style="text-align:center;margin-bottom:2px">📊 Results Tracker</h2>
<p style="text-align:center;color:#888;font-size:11px;margin-top:0">{datetime.now().strftime("%d %b %H:%M")} · every pick, auto-verified against real results</p>

<div style="background:#1a1f26;border-radius:12px;padding:16px;margin:14px 0;border:1px solid #2a3038;text-align:center">
  <div style="font-size:11px;color:#888">OVERALL</div>
  <div style="font-size:32px;font-weight:bold;color:#ffeb3b">{overall_pct if overall_pct is not None else "—"}{"%" if overall_pct is not None else ""}</div>
  <div style="font-size:12px;color:#888">{total_hit}/{total} verified picks · {len(pending)} pending (game not final yet)</div>
</div>

<div style="background:#1a1f26;border-radius:12px;padding:16px;margin:14px 0;border:1px solid #2a3038">
  <div style="font-weight:bold;margin-bottom:8px">By Category</div>
  {rows}
</div>

<div style="background:#1a1f26;border-radius:12px;padding:16px;margin:14px 0;border:1px solid #2a3038">
  <div style="font-weight:bold;margin-bottom:8px">Recent Results</div>
  {recent_rows or '<p style="color:#888;font-size:12px">Nothing verified yet — check back after a few days of picks have had time to play out.</p>'}
</div>

<div style="font-size:11px;color:#888;text-align:center;margin-top:20px;line-height:1.6">
  Team Total / Game Total use the same proven score field this model already relies on.
  Passing Yards / Rushing Yards / Receptions / Anytime TD use a best-effort read of ESPN's
  boxscore that wasn't confirmed against a live response — if those categories stay empty
  for more than a few days after games finish, check the Actions log for [DIAG] lines, which
  show exactly what shape the boxscore actually came back in. Sample sizes are still small
  early on — treat percentages with real caution until there's a few weeks of data.
</div>
</body></html>"""

    os.makedirs(os.path.dirname(DASHBOARD_PATH), exist_ok=True)
    with open(DASHBOARD_PATH, "w") as f:
        f.write(html)
    print(f"  Results dashboard: {total} verified, {overall_pct}% overall" if total else "  Results dashboard: no verified picks yet")


def run_results_tracker(legs, team_hot_form=None, team_real_streak=None,
                          player_hot_form=None, player_real_streak=None):
    """Single entry point called from blitz_iq.py's main(). The four
    form/streak args are optional (default to nothing logged) so this
    stays callable the old way too, but blitz_iq.py's __main__ block
    always passes all four now."""
    print("\nRunning results tracker...")
    log = load_log()
    log = log_todays_signals(legs, team_hot_form, team_real_streak, player_hot_form, player_real_streak, log)
    log = verify_pending_results(log)
    save_log(log)
    build_results_dashboard(log)
