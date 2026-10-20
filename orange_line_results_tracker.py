#!/usr/bin/env python3
"""
Orange Line Results Tracker
--------------------------------------------------------------
Same architecture as Blitz IQ's own tracker (blitz_iq_results_tracker.py),
adapted for ESPN's basketball API and Orange Line's player-props-only
shape:

- Team Total / Game Total legs reuse the same score.value +
  status.type.completed pattern orange_line.py's own get_team_form()
  already relies on.
- Rebounds / Assists / Rebounds+Assists (RA) are less certain: ESPN's
  per-game basketball boxscore player-stat shape isn't confirmed in this
  session, only that /summary?event={id} returns a .boxscore key
  containing it somewhere, same honesty caveat orange_line.py's own
  module docstring already carries for the gamelog endpoint. This tries
  the same "REB"/"AST" labels orange_line.py already assumes and prints
  a [DIAG] line with whatever labels ESPN actually returned if they
  don't match, rather than guessing silently. RA is just REB + AST
  summed from the same athlete's boxscore row (not a separate category
  to search for, unlike Blitz IQ's cross-category Anytime TD).
- Orange Line spans TWO leagues (NBA/WNBA) sharing one predictor run, so
  unlike Blitz IQ's single-sport summary URL, this needs the league key
  (nba/wnba) per entry -- carried through as the full league NAME
  ("NBA"/"WNBA", same string orange_line.py's own legs/entries already
  use) and mapped back to the URL's league key here.
- Hot Form / Real Streak here are PLAYER-only (Orange Line has no
  team-level Hot Form/Real Streak panel, unlike Blitz IQ) -- verified by
  checking whether that player's actual stat in the SAME predicted game
  cleared the threshold that got them flagged (now carried directly on
  each entry as "threshold" -- added to orange_line.py's own
  build_hot_form_entries()/build_real_streak_entries() at the same time
  this tracker was added), same "did the form continue" question
  Blitz IQ's own player_hot_form/player_real_streak scanners ask.

Designed to be imported and called from orange_line.py's main().

Output:
    docs/orange-line/results/log.json    -- the full log
    docs/orange-line/results/index.html  -- dashboard: overall + per-category
                                             win rate, recent history
"""

import os
import json
import hashlib
import requests
from datetime import datetime, timezone

BASE_FMT = "https://site.api.espn.com/apis/site/v2/sports/basketball/{league}/summary"
LOG_PATH = "docs/orange-line/results/log.json"
DASHBOARD_PATH = "docs/orange-line/results/index.html"

LEAGUE_NAME_TO_KEY = {"NBA": "nba", "WNBA": "wnba"}

SUMMARY_CACHE = {}  # (league_key, game_id) -> summary, several legs share a game

CATEGORY_SCANNER = {
    "Team Total": "team_total", "Game Total": "game_total",
    "Rebounds": "rebounds", "Assists": "assists", "Rebounds+Assists": "ra",
}
STAT_LABEL_BY_KEY = {"REB": "REB", "AST": "AST"}  # RA is REB+AST combined below, not a single label
SCANNER_TO_STAT_KEY = {"rebounds": "REB", "assists": "AST"}  # ra handled separately (both summed)


def _get(league_key, params):
    url = BASE_FMT.format(league=league_key)
    try:
        r = requests.get(url, params=params, timeout=20)
    except Exception as e:
        print(f"    [!] verification request failed: league={league_key} event={params.get('event')} ({e})")
        return None
    if r.status_code != 200:
        print(f"    [!] {r.status_code} on summary for league={league_key} event={params.get('event')}")
        return None
    return r.json()


def _get_summary(league_name, game_id):
    league_key = LEAGUE_NAME_TO_KEY.get(league_name)
    if not league_key or not game_id:
        return None
    key = (league_key, game_id)
    if key in SUMMARY_CACHE:
        return SUMMARY_CACHE[key]
    data = _get(league_key, {"event": game_id})
    SUMMARY_CACHE[key] = data
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


def log_todays_signals(legs, hot_form_entries, real_streak_entries, log):
    """Legs already carry game_id, game_date, league, is_home, line, and
    (for player props) player_id/stat_key -- added specifically for this
    tracker when the legs were built. Hot Form/Real Streak entries carry
    player_id, game_id, date and threshold -- added to orange_line.py's
    own build_hot_form_entries()/build_real_streak_entries() at the same
    time this tracker was added (they didn't carry enough to look the
    game back up before that)."""
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
            "value": leg["prob"], "detail": leg.get("detail"), "line": leg.get("line"), "threshold": None,
            "league": leg.get("league"), "game_id": leg["game_id"], "game_date": leg["game_date"],
            "date_key": leg["game_date"], "is_home": leg.get("is_home"),
            "player_id": leg.get("player_id"), "stat_key": leg.get("stat_key"),
            "logged_at": datetime.now(timezone.utc).isoformat(),
            "status": "pending", "result": None, "actual": None,
        })
        existing_ids.add(eid)
        added += 1

    def _add_form(scanner, e, stat_key):
        nonlocal added
        game_id, player_id = e.get("game_id"), e.get("player_id")
        if game_id is None or player_id is None:
            return  # can't verify a pick with no game/player to look up later
        date_key = (e.get("date") or "")[:10]
        if scanner == "player_hot_form":
            market = f"{e['name']} Hot Form (avg {e['avg']} {stat_key}, last {len(e['values'])}gm)"
        else:
            market = f"{e['name']} Real Streak ({e['streak_len']}+ straight games ≥{e['threshold']} {stat_key})"
        # subject includes the stat -- a player can qualify for Hot Form
        # (or Real Streak) in REB, AST and RA simultaneously in the same
        # game, and without the stat in here all three would collide onto
        # one entry id (same name/match/date), silently dropping two of
        # the three picks as "already logged".
        eid = _entry_id(scanner, f"{e['name']}|{e['stat']}", date_key, e["match"])
        if eid in existing_ids:
            return
        log.append({
            "id": eid, "scanner": scanner, "match": e["match"], "market": market,
            "value": None, "detail": None, "line": None, "threshold": e["threshold"],
            "league": e.get("league"), "game_id": game_id, "game_date": date_key, "date_key": date_key,
            "is_home": None, "player_id": player_id, "stat_key": e["stat"],
            "logged_at": datetime.now(timezone.utc).isoformat(),
            "status": "pending", "result": None, "actual": None,
        })
        existing_ids.add(eid)
        added += 1

    for e in (hot_form_entries or []):
        _add_form("player_hot_form", e, e["stat"])
    for e in (real_streak_entries or []):
        _add_form("player_real_streak", e, e["stat"])

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
    summary = _get_summary(entry["league"], entry["game_id"])
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


def _find_player_boxscore_stat(summary, player_id, stat_key):
    """DEFENSIVE, label-based lookup -- same philosophy as Blitz IQ's own
    _find_player_boxscore_stat, simplified for basketball's single
    stat-category-per-team boxscore shape (unlike football's separate
    passing/rushing/receiving categories, so there's no category-keyword
    disambiguation needed here -- just find the category whose labels
    include stat_key, same REB/AST label orange_line.py's own gamelog
    fetcher already assumes)."""
    players = summary.get("boxscore", {}).get("players", [])
    if not players:
        print(f"    [DIAG] boxscore has no 'players' key or it's empty. "
              f"Top-level boxscore keys: {list(summary.get('boxscore', {}).keys())}")
        return None
    for team_block in players:
        for stat_category in team_block.get("statistics", []):
            labels = stat_category.get("labels") or stat_category.get("names")
            athletes = stat_category.get("athletes", [])
            if not labels or stat_key not in labels:
                continue
            idx = labels.index(stat_key)
            for a in athletes:
                athlete_info = a.get("athlete", {})
                if str(athlete_info.get("id")) != str(player_id):
                    continue
                stats = a.get("stats", [])
                try:
                    return float(stats[idx])
                except (IndexError, ValueError, TypeError):
                    continue
    print(f"    [DIAG] couldn't find player {player_id}'s '{stat_key}' stat in boxscore -- "
          f"either the shape assumed here doesn't match ESPN's real response, or this player "
          f"didn't feature this game (DNP). Team blocks found: {len(players)}")
    return None


def _player_stat_value(summary, player_id, scanner):
    """RA is REB + AST summed from the SAME athlete row, not a separate
    boxscore label to search for -- both lookups share one cached
    summary, so this is still just one extra label scan, not an extra
    request."""
    if scanner == "ra":
        reb = _find_player_boxscore_stat(summary, player_id, "REB")
        ast = _find_player_boxscore_stat(summary, player_id, "AST")
        if reb is None or ast is None:
            return None
        return reb + ast
    stat_key = SCANNER_TO_STAT_KEY.get(scanner)
    if not stat_key:
        return None
    return _find_player_boxscore_stat(summary, player_id, stat_key)


def _verify_player_leg(entry):
    if not entry.get("player_id"):
        return None
    summary = _get_summary(entry["league"], entry["game_id"])
    if not summary or not _is_final(summary):
        return None
    actual = _player_stat_value(summary, entry["player_id"], entry["scanner"])
    if actual is None:
        return None
    return {"actual": actual, "result": "hit" if actual > entry["line"] else "miss"}


def _verify_player_form_entry(entry):
    """Hot Form / Real Streak for a PLAYER -- same boxscore lookup as
    _verify_player_leg, just checked against the flagged threshold
    instead of a betting line. stat_key here is REB/AST/RA (the short
    form orange_line.py's own entries use), not a scanner name."""
    if not entry.get("player_id") or not entry.get("stat_key"):
        return None
    summary = _get_summary(entry["league"], entry["game_id"])
    if not summary or not _is_final(summary):
        return None
    scanner_key = {"REB": "rebounds", "AST": "assists", "RA": "ra"}.get(entry["stat_key"])
    actual = _player_stat_value(summary, entry["player_id"], scanner_key)
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
            elif entry["scanner"] in ("rebounds", "assists", "ra"):
                result = _verify_player_leg(entry)
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

    print(f"  Results verification: checked {checked} pending entries, {updated} newly verified "
          f"({len(SUMMARY_CACHE)} distinct game(s) looked up)")
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
        "rebounds": "Rebounds", "assists": "Assists", "ra": "Rebounds+Assists",
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
        rows += f"""<div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #3a2a20">
  <span>{label}</span><span style="color:#ffeb3b;font-weight:bold">{pct_str}</span>
  <span style="color:#998;font-size:12px">{d['hit']}/{n}</span>
</div>"""

    recent = sorted(verified, key=lambda e: e.get("verified_at", ""), reverse=True)[:30]
    recent_rows = ""
    for e in recent:
        color = "#22c55e" if e["result"] == "hit" else "#ef4444"
        recent_rows += f"""<div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #3a2a20;font-size:12px">
  <span>{e['match']} — {e['market']}</span><span style="color:{color};font-weight:bold">{e['result'].upper()}</span>
</div>"""

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Results — Orange Line</title></head>
<body style="background:#0f0a08;color:#e8dcd0;font-family:Arial;padding:12px;max-width:600px;margin:auto">
<p style="text-align:center;margin-bottom:6px"><a href="../index.html" style="color:#ff9a2e;text-decoration:none;font-size:12px">← Orange Line</a></p>
<h2 style="text-align:center;margin-bottom:2px">📊 Results Tracker</h2>
<p style="text-align:center;color:#998;font-size:11px;margin-top:0">{datetime.now().strftime("%d %b %H:%M")} · every pick, auto-verified against real results</p>

<div style="background:#1a1310;border-radius:12px;padding:16px;margin:14px 0;border:1px solid #3a2a20;text-align:center">
  <div style="font-size:11px;color:#998">OVERALL</div>
  <div style="font-size:32px;font-weight:bold;color:#ffeb3b">{overall_pct if overall_pct is not None else "—"}{"%" if overall_pct is not None else ""}</div>
  <div style="font-size:12px;color:#998">{total_hit}/{total} verified picks · {len(pending)} pending (game not final yet)</div>
</div>

<div style="background:#1a1310;border-radius:12px;padding:16px;margin:14px 0;border:1px solid #3a2a20">
  <div style="font-weight:bold;margin-bottom:8px">By Category</div>
  {rows}
</div>

<div style="background:#1a1310;border-radius:12px;padding:16px;margin:14px 0;border:1px solid #3a2a20">
  <div style="font-weight:bold;margin-bottom:8px">Recent Results</div>
  {recent_rows or '<p style="color:#998;font-size:12px">Nothing verified yet — check back after a few days of picks have had time to play out.</p>'}
</div>

<div style="font-size:11px;color:#998;text-align:center;margin-top:20px;line-height:1.6">
  Team Total / Game Total use the same proven score field this model already relies on.
  Rebounds / Assists / Rebounds+Assists / Hot Form / Real Streak use a best-effort read of
  ESPN's basketball boxscore that wasn't confirmed against a live response — if those
  categories stay empty for more than a few days after games finish, check the Actions log
  for [DIAG] lines, which show exactly what shape the boxscore actually came back in. Sample
  sizes are still small early on — treat percentages with real caution until there's a few
  weeks of data.
</div>
</body></html>"""

    os.makedirs(os.path.dirname(DASHBOARD_PATH), exist_ok=True)
    with open(DASHBOARD_PATH, "w") as f:
        f.write(html)
    print(f"  Results dashboard: {total} verified, {overall_pct}% overall" if total else "  Results dashboard: no verified picks yet")


def run_results_tracker(legs, hot_form_entries=None, real_streak_entries=None):
    """Single entry point called from orange_line.py's main()."""
    print("\nRunning results tracker...")
    log = load_log()
    log = log_todays_signals(legs, hot_form_entries, real_streak_entries, log)
    log = verify_pending_results(log)
    save_log(log)
    build_results_dashboard(log)
