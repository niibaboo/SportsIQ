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
from datetime import datetime, timezone, timedelta

BASE_FMT = "https://site.api.espn.com/apis/site/v2/sports/basketball/{league}/summary"
LOG_PATH = "docs/orange-line/results/log.json"
DASHBOARD_PATH = "docs/orange-line/results/index.html"
ARCHIVE_PATH = "docs/orange-line/results/archive.json"

# Why these exist (user report, 2026-11-16: "the pending result is too much" --
# 39,691 pending vs 122/211 verified). The tracker logs ~2,000 candidate legs
# a day, but used to grade at most 60 ENTRIES per run, oldest first, so the
# backlog could only grow (Points and 3PT never even got reached), and a leg
# whose player didn't play stayed "pending" forever. Now:
#   - the per-run cap counts distinct GAME lookups (one request each, shared
#     by every leg of that game), not entries;
#   - a player who isn't in a finished game's boxscore is "void" (a DNP is a
#     void bet at a bookmaker), and anything still unverifiable after
#     VOID_AFTER_DAYS is voided too;
#   - verified entries older than ARCHIVE_AFTER_DAYS are folded into running
#     hit/miss totals in archive.json and dropped from log.json, so the log
#     stops growing without bound (it was heading for GitHub's file size limit).
MAX_GAME_LOOKUPS = 150
CALIBRATION_SCANNERS = ("team_stat_ou", "team_stat_1x2", "first_half_margin")
VOID_AFTER_DAYS = 10
ARCHIVE_AFTER_DAYS = 3

LEAGUE_NAME_TO_KEY = {"NBA": "nba", "WNBA": "wnba",
                      "NCAAB": "mens-college-basketball", "WNCAAB": "womens-college-basketball"}

SUMMARY_CACHE = {}  # (league_key, game_id) -> summary, several legs share a game

CATEGORY_SCANNER = {
    "Team Total": "team_total", "Game Total": "game_total",
    "Rebounds": "rebounds", "Assists": "assists", "Rebounds+Assists": "ra",
    # Points added 2026-10-22 -- orange_line.py's build_legs() has logged
    # "Points" category legs (STAT_LABELS["PTS"]) since the Points prop
    # was added, but this tracker was never updated to recognize that
    # category, so every Points pick has been silently skipped by
    # log_todays_signals() ("if not scanner... continue") and never shown
    # on the results dashboard at all -- same class of gap as Cards &
    # Corners IQ's Real Streak picks missing their own tracker earlier.
    "Points": "points",
    # 3-Pointers Made added 2026-10-22 -- same gap as Points above: Orange
    # Line's build_legs() now logs "3-Pointers Made" category legs
    # (STAT_LABELS["3PM"]) but without this entry they'd be silently
    # skipped by log_todays_signals() the exact same way Points was.
    "3-Pointers Made": "tpm",
}
STAT_LABEL_BY_KEY = {"REB": "REB", "AST": "AST", "PTS": "PTS", "3PM": "3PT"}  # RA is REB+AST combined below, not a single label
# "tpm" maps to the real ESPN boxscore label "3PT" (not "3PM" -- orange_line.py's
# own PROP_CONFIG uses the same '3PT' root label for the live gamelog pull, and
# this tracker's final-boxscore endpoint is assumed to share it, unconfirmed
# either way). See _player_stat_value()'s value_parser note below for why this
# one stat can't just be float()'d like the others.
SCANNER_TO_STAT_KEY = {"rebounds": "REB", "assists": "AST", "points": "PTS", "tpm": "3PT"}  # ra handled separately (both summed)


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
        json.dump(entries, f, separators=(",", ":"), default=str)


def load_archive():
    if not os.path.exists(ARCHIVE_PATH):
        return {"by_scanner": {}, "void": 0}
    try:
        with open(ARCHIVE_PATH) as f:
            a = json.load(f)
        a.setdefault("by_scanner", {})
        a.setdefault("void", 0)
        return a
    except Exception as e:
        print(f"  [!] Couldn't read archive ({e}) -- starting a fresh one.")
        return {"by_scanner": {}, "void": 0}


def save_archive(archive):
    os.makedirs(os.path.dirname(ARCHIVE_PATH), exist_ok=True)
    with open(ARCHIVE_PATH, "w") as f:
        json.dump(archive, f, indent=2)


def fold_old_entries(log, archive, now=None):
    """Move settled entries out of the log into the archive's running totals:
    verified ones older than ARCHIVE_AFTER_DAYS (counted by hit/miss per
    scanner), and every void one (counted). Pending entries always stay."""
    today = (now or datetime.now(timezone.utc)).date()
    cutoff = (today - timedelta(days=ARCHIVE_AFTER_DAYS)).isoformat()
    kept, folded, voided = [], 0, 0
    for e in log:
        if e["status"] == "void":
            archive["void"] = archive.get("void", 0) + 1
            voided += 1
        elif e["status"] == "verified" and (e.get("verified_at") or "9999")[:10] < cutoff:
            d = archive["by_scanner"].setdefault(e["scanner"], {"hit": 0, "miss": 0})
            d[e["result"]] += 1
            if e["scanner"] in CALIBRATION_SCANNERS and isinstance(e.get("value"), (int, float)):
                d["psum"] = d.get("psum", 0.0) + e["value"]
                d["pn"] = d.get("pn", 0) + 1
            if e["scanner"] == "first_half_margin" and isinstance(e.get("actual"), (int, float)) and e.get("pred_mean") is not None:
                err = e["actual"] - e["pred_mean"]
                d["sq"] = d.get("sq", 0.0) + err * err
                d["err"] = d.get("err", 0.0) + err
                d["sdsum"] = d.get("sdsum", 0.0) + (e.get("sd") or 0)
                d["sqn"] = d.get("sqn", 0) + 1
            folded += 1
        else:
            kept.append(e)
    print(f"  Archive: folded {folded} verified + {voided} void entries out of the log; {len(kept)} remain")
    return kept, archive


def log_todays_signals(legs, hot_form_entries, real_streak_entries, log, team_stat_entries=None):
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

    for e in (team_stat_entries or []):
        eid = _entry_id(e["scanner"] + "|" + e["stat_key"], e["market"], e["game_date"], e["match"])
        if eid in existing_ids:
            continue
        log.append({
            "id": eid, "scanner": e["scanner"], "match": e["match"], "market": e["market"],
            "value": e["prob"], "detail": None, "line": e.get("line"), "threshold": None,
            "league": e.get("league"), "game_id": e["game_id"], "game_date": e["game_date"],
            "date_key": e["game_date"], "is_home": None, "player_id": None,
            "stat_key": e["stat_key"], "pick": e.get("pick"),
            "pred_mean": e.get("pred_mean"), "sd": e.get("sd"),
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
            "value": e.get("streak_len") if scanner == "player_real_streak" else None,
            "detail": None, "streak_games": e.get("streak_games"), "line": None, "threshold": e["threshold"],
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


def _first_half_margin(summary, league_key):
    """Real first-half margin (home - away) from the per-period linescores, or
    None. NBA/WNBA/women's college play quarters (first half = 2 periods);
    men's college plays two halves (1 period). Prints [DIAG] when the shape
    isn't what's expected -- the linescores field is not confirmed live."""
    need = 1 if league_key == "mens-college-basketball" else 2
    comps = summary.get("header", {}).get("competitions", [{}])[0].get("competitors", [])
    home = next((c for c in comps if c.get("homeAway") == "home"), None)
    away = next((c for c in comps if c.get("homeAway") == "away"), None)
    if not home or not away:
        return None
    def half(c):
        ls = c.get("linescores") or []
        if len(ls) < need + 1:  # a finished game has more periods than the first half
            return None
        total = 0.0
        for item in ls[:need]:
            v = item.get("value", item.get("displayValue"))
            try:
                total += float(v)
            except (TypeError, ValueError):
                return None
        return total
    h, a = half(home), half(away)
    if h is None or a is None:
        print(f"    [DIAG] first-half margin: couldn't read linescores for {league_key} "
              f"(home has {len(home.get('linescores') or [])} period(s), away {len(away.get('linescores') or [])})")
        return None
    return h - a


def _verify_first_half_entry(entry):
    summary = _get_summary(entry["league"], entry["game_id"])
    if not summary or not _is_final(summary):
        return None
    m = _first_half_margin(summary, LEAGUE_NAME_TO_KEY.get(entry["league"]))
    if m is None or entry.get("pred_mean") is None or not entry.get("sd"):
        return None
    err = m - entry["pred_mean"]
    return {"actual": m, "result": "hit" if abs(err) <= entry["sd"] else "miss"}


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


def _player_in_boxscore(summary, player_id):
    """(boxscore_has_players, player_found). Used to tell a DNP (the game is
    final, ESPN returned a real boxscore, and this player is simply not in
    it) apart from "couldn't read the boxscore" (leave pending)."""
    players = summary.get("boxscore", {}).get("players", [])
    if not players:
        return False, False
    for team_block in players:
        for cat in team_block.get("statistics", []):
            for a in cat.get("athletes", []):
                if str(a.get("athlete", {}).get("id")) == str(player_id):
                    return True, True
    return True, False


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


def _parse_makes_attempts(raw):
    """Value parser for a combined 'makes-attempts' boxscore stat, e.g.
    3PT commonly rendering as "3-7" -- same handling, and same
    unconfirmed-against-a-live-response caveat, as orange_line.py's own
    _parse_makes_attempts (duplicated here, not imported, per this
    suite's one-script-per-tool convention). Returns None (not 0) on
    total failure, so the caller treats it as "couldn't read this
    stat", not a real zero."""
    s = str(raw).strip()
    if '-' in s:
        made_str = s.split('-', 1)[0].strip()
        try:
            return float(made_str)
        except (ValueError, TypeError):
            return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _find_player_boxscore_stat(summary, player_id, stat_key, value_parser=None):
    """DEFENSIVE, label-based lookup -- same philosophy as Blitz IQ's own
    _find_player_boxscore_stat, simplified for basketball's single
    stat-category-per-team boxscore shape (unlike football's separate
    passing/rushing/receiving categories, so there's no category-keyword
    disambiguation needed here -- just find the category whose labels
    include stat_key, same REB/AST label orange_line.py's own gamelog
    fetcher already assumes).

    value_parser (added for 3-Pointers Made, user request, 2026-10-25):
    every OTHER stat this tracker reads is a confirmed-plain number and
    uses the default plain float(); 3PT is passed _parse_makes_attempts
    instead, since ESPN's shooting splits are commonly rendered as a
    combined "makes-attempts" string rather than a made-only column."""
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
                    raw = stats[idx]
                except IndexError:
                    continue
                if value_parser:
                    val = value_parser(raw)
                    if val is not None:
                        return val
                    continue
                try:
                    return float(raw)
                except (ValueError, TypeError):
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
    # "tpm" (3-Pointers Made) needs the makes-attempts parser -- see
    # _find_player_boxscore_stat's own value_parser note. Every other
    # scanner here reads a confirmed-plain number via the default.
    value_parser = _parse_makes_attempts if scanner == "tpm" else None
    return _find_player_boxscore_stat(summary, player_id, stat_key, value_parser=value_parser)


def _verify_player_leg(entry):
    if not entry.get("player_id"):
        return None
    summary = _get_summary(entry["league"], entry["game_id"])
    if not summary or not _is_final(summary):
        return None
    has_players, found = _player_in_boxscore(summary, entry["player_id"])
    if has_players and not found:
        return {"void": "dnp"}
    actual = _player_stat_value(summary, entry["player_id"], entry["scanner"])
    if actual is None:
        return None
    return {"actual": actual, "result": "hit" if actual > entry["line"] else "miss"}


def _team_stat_totals(summary, stat_key):
    """(home_total, away_total): each team's boxscore player stats summed.
    None when any player row can't be read, so a partial sum never grades a
    pick. Uses the home/away order of header.competitors to name the teams."""
    players = summary.get("boxscore", {}).get("players", [])
    comps = summary.get("header", {}).get("competitions", [{}])[0].get("competitors", [])
    if len(players) != 2 or len(comps) != 2:
        print(f"    [DIAG] team stat: expected 2 team blocks and 2 competitors, got {len(players)}/{len(comps)}")
        return None, None
    parser = _parse_makes_attempts if stat_key == "3PT" else None
    totals = {}
    for block in players:
        tid = str(block.get("team", {}).get("id"))
        total, seen = 0.0, 0
        for cat in block.get("statistics", []):
            labels = cat.get("labels") or cat.get("names")
            if not labels or stat_key not in labels:
                continue
            idx = labels.index(stat_key)
            for a in cat.get("athletes", []):
                stats = a.get("stats", [])
                if idx >= len(stats):
                    continue  # DNP rows come back empty
                raw = stats[idx]
                val = parser(raw) if parser else None
                if parser is None:
                    try:
                        val = float(raw)
                    except (TypeError, ValueError):
                        val = None
                if val is None:
                    print(f"    [DIAG] team stat: unreadable {stat_key} value {raw!r} for team {tid}")
                    return None, None
                total += val
                seen += 1
        if not seen:
            print(f"    [DIAG] team stat: no '{stat_key}' column found for team {tid}")
            return None, None
        totals[tid] = total
    home = next((c for c in comps if c.get("homeAway") == "home"), None)
    away = next((c for c in comps if c.get("homeAway") == "away"), None)
    try:
        return totals[str(home["team"]["id"])], totals[str(away["team"]["id"])]
    except (KeyError, TypeError):
        print(f"    [DIAG] team stat: competitor team ids {[(c.get('team') or {}).get('id') for c in comps]} "
              f"don't match boxscore team ids {list(totals)}")
        return None, None


def _verify_team_stat_entry(entry):
    summary = _get_summary(entry["league"], entry["game_id"])
    if not summary or not _is_final(summary):
        return None
    h, a = _team_stat_totals(summary, entry["stat_key"])
    if h is None:
        return None
    if entry["scanner"] == "team_stat_ou":
        actual = h + a
        return {"actual": actual, "result": "hit" if actual > entry["line"] else "miss"}
    d = h - a
    outcome = "home" if d > 0 else "away" if d < 0 else "tie"
    return {"actual": f"{h:g}-{a:g}", "result": "hit" if outcome == entry.get("pick") else "miss"}


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
    # PTS added alongside the Points category above -- without it, any
    # Points-based Hot Form/Real Streak entry (orange_line.py has flagged
    # these since HOT_FORM_MIN['PTS']/REAL_STREAK_THRESHOLD['PTS'] were
    # added) was already being logged fine, but could never verify: this
    # map returned None for stat_key "PTS", so _player_stat_value got
    # scanner=None and always returned None, leaving those entries stuck
    # "pending" forever instead of counted hit/miss.
    scanner_key = {"REB": "rebounds", "AST": "assists", "RA": "ra", "PTS": "points", "3PM": "tpm"}.get(entry["stat_key"])
    has_players, found = _player_in_boxscore(summary, entry["player_id"])
    if has_players and not found:
        return {"void": "dnp"}
    actual = _player_stat_value(summary, entry["player_id"], scanner_key)
    if actual is None:
        return None
    return {"actual": actual, "result": "hit" if actual >= entry["threshold"] else "miss"}


def verify_pending_results(log, max_lookups=MAX_GAME_LOOKUPS):
    now = datetime.now(timezone.utc)
    today = now.date().isoformat()
    void_cutoff = (now.date() - timedelta(days=VOID_AFTER_DAYS)).isoformat()
    lookups = checked = updated = voided = deferred = 0

    for entry in log:
        if entry["status"] != "pending":
            continue
        if entry["date_key"] >= today:
            continue
        league_key = LEAGUE_NAME_TO_KEY.get(entry.get("league"))
        cache_key = (league_key, entry.get("game_id"))
        if cache_key not in SUMMARY_CACHE:
            if lookups >= max_lookups:
                deferred += 1       # picked up on a later run; cached games above are still free
                continue
            lookups += 1
        checked += 1

        result = None
        try:
            if entry["scanner"] in ("team_total", "game_total"):
                result = _verify_team_leg(entry)
            elif entry["scanner"] == "first_half_margin":
                result = _verify_first_half_entry(entry)
            elif entry["scanner"] in ("team_stat_ou", "team_stat_1x2"):
                result = _verify_team_stat_entry(entry)
            elif entry["scanner"] in ("rebounds", "assists", "ra", "points", "tpm"):
                result = _verify_player_leg(entry)
            elif entry["scanner"] in ("player_hot_form", "player_real_streak"):
                result = _verify_player_form_entry(entry)
        except Exception as e:
            print(f"    [!] verification error for entry {entry['id']} ({entry['scanner']}): {e}")
            result = None

        if result and result.get("void"):
            entry["status"] = "void"
            entry["void_reason"] = result["void"]
            voided += 1
        elif result:
            entry["status"] = "verified"
            entry["result"] = result["result"]
            entry["actual"] = result["actual"]
            entry["verified_at"] = datetime.now(timezone.utc).isoformat()
            updated += 1
        elif entry["date_key"] < void_cutoff:
            entry["status"] = "void"
            entry["void_reason"] = f"unverifiable after {VOID_AFTER_DAYS} days"
            voided += 1

    print(f"  Results verification: checked {checked} pending entries across {lookups} new game lookup(s), "
          f"{updated} newly verified, {voided} voided, {deferred} deferred to the next run "
          f"({len(SUMMARY_CACHE)} distinct game(s) cached)")
    return log


def build_results_dashboard(log, archive=None):
    archive = archive or {"by_scanner": {}, "void": 0}
    verified = [e for e in log if e["status"] == "verified"]
    pending = [e for e in log if e["status"] == "pending"]
    void_total = archive.get("void", 0) + sum(1 for e in log if e["status"] == "void")

    # Older settled picks live in the archive's running totals, not the log.
    by_scanner = {k: dict({"hit": v.get("hit", 0), "miss": v.get("miss", 0)},
                          psum=v.get("psum", 0.0), pn=v.get("pn", 0))
                  for k, v in archive.get("by_scanner", {}).items()}
    for e in verified:
        d = by_scanner.setdefault(e["scanner"], {"hit": 0, "miss": 0})
        d[e["result"]] += 1
    # Calibration needs the stored probability, so split team-stat picks by stat.
    cal = {}
    for e in verified:
        if e["scanner"] in CALIBRATION_SCANNERS and isinstance(e.get("value"), (int, float)):
            c = cal.setdefault((e["scanner"], e.get("stat_key")), {"n": 0, "hit": 0, "psum": 0.0})
            c["n"] += 1
            c["hit"] += e["result"] == "hit"
            c["psum"] += e["value"]
    for sc in CALIBRATION_SCANNERS:  # archived picks (no per-stat split kept)
        a = archive.get("by_scanner", {}).get(sc)
        if a and a.get("pn"):
            c = cal.setdefault((sc, "older, all stats"), {"n": 0, "hit": 0, "psum": 0.0})
            c["n"] += a["pn"]
            c["hit"] += a.get("hit", 0)
            c["psum"] += a["psum"]
    # 1st-half margin spread check: realised spread vs the one the model assumed.
    fh_n, fh_sq, fh_err, fh_sd = 0, 0.0, 0.0, 0.0
    for e in verified:
        if e["scanner"] == "first_half_margin" and isinstance(e.get("actual"), (int, float)) and e.get("pred_mean") is not None:
            err = e["actual"] - e["pred_mean"]
            fh_n, fh_sq, fh_err, fh_sd = fh_n + 1, fh_sq + err * err, fh_err + err, fh_sd + (e.get("sd") or 0)
    a_fh = archive.get("by_scanner", {}).get("first_half_margin") or {}
    fh_n += a_fh.get("sqn", 0); fh_sq += a_fh.get("sq", 0.0); fh_err += a_fh.get("err", 0.0); fh_sd += a_fh.get("sdsum", 0.0)
    fh_note = ""
    if fh_n:
        real_sd = (fh_sq / fh_n) ** 0.5
        fh_note = (f'<div style="padding:8px 0;font-size:12px;color:#e8dcd0">1st-half margin spread: real &plusmn;{real_sd:.1f} pts vs '
                   f'&plusmn;{fh_sd / fh_n:.1f} assumed · average miss {fh_err / fh_n:+.1f} (home minus model) · n={fh_n}'
                   f'{"" if fh_n >= 100 else " (too few to change the spread yet)"}</div>')
    cal_rows = ""
    for (sc, stat), c in sorted(cal.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        if sc == "first_half_margin":
            name = "1st-half margin inside the model's ±1 spread (model says ~68%)"
        else:
            name = ("Combined over at middle line" if sc == "team_stat_ou" else "1X2 top pick") + f" · {stat}"
        pred, act = 100 * c["psum"] / c["n"], 100 * c["hit"] / c["n"]
        gap = act - pred
        flag = "" if c["n"] >= 30 else " (too few to judge)"
        cal_rows += (f'<div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #3a2a20;font-size:12px">'
                     f'<span>{name}</span><span>model {pred:.0f}% · actual {act:.0f}% ({gap:+.0f}) · n={c["n"]}{flag}</span></div>')
    cal_rows += fh_note
    cal_html = ""
    if cal_rows:
        cal_html = ('<div style="background:#1a1310;border-radius:12px;padding:16px;margin:14px 0;border:1px solid #3a2a20">'
                    '<div style="font-weight:bold;margin-bottom:4px">Team rebounds / assists / 3PT and 1st-half margin — calibration</div>'
                    '<div style="color:#998;font-size:11px;margin-bottom:8px">Does what the model calls 50% land near 50%? '
                    'Model = average probability it gave; actual = how often it happened. Gaps within a few points are noise below ~100 picks.</div>'
                    f'{cal_rows}</div>')

    SCANNER_LABELS = {
        "team_total": "Team Total", "game_total": "Game Total",
        "rebounds": "Rebounds", "assists": "Assists", "ra": "Rebounds+Assists",
        "points": "Points", "tpm": "3-Pointers Made",
        "player_hot_form": "Player Hot Form", "player_real_streak": "Player Real Streak",
        "team_stat_ou": "Team REB/AST/3PT Over (mid line)", "team_stat_1x2": "Team REB/AST/3PT 1X2 pick",
    }

    # Calibration picks (~50% by design) are judged on their own panel, not the overall figure.
    total_hit = sum(d["hit"] for k, d in by_scanner.items() if k not in CALIBRATION_SCANNERS)
    total_miss = sum(d["miss"] for k, d in by_scanner.items() if k not in CALIBRATION_SCANNERS)
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
  <div style="font-size:12px;color:#998">{total_hit}/{total} verified picks · {len(pending)} waiting for the game to finish{f" · {void_total} void (player did not play / unverifiable)" if void_total else ""}</div>
</div>

<div style="background:#1a1310;border-radius:12px;padding:16px;margin:14px 0;border:1px solid #3a2a20">
  <div style="font-weight:bold;margin-bottom:8px">By Category</div>
  {rows}
</div>

{cal_html}
<div style="background:#1a1310;border-radius:12px;padding:16px;margin:14px 0;border:1px solid #3a2a20">
  <div style="font-weight:bold;margin-bottom:8px">Recent Results</div>
  {recent_rows or '<p style="color:#998;font-size:12px">Nothing verified yet — check back after a few days of picks have had time to play out.</p>'}
</div>

<div style="font-size:11px;color:#998;text-align:center;margin-top:20px;line-height:1.6">
  Team Total / Game Total use the same proven score field this model already relies on.
  Rebounds / Assists / Rebounds+Assists / Points / 3-Pointers Made / Hot Form / Real Streak use a best-effort read of
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


def run_results_tracker(legs, hot_form_entries=None, real_streak_entries=None, team_stat_entries=None):
    """Single entry point called from orange_line.py's main()."""
    print("\nRunning results tracker...")
    log = load_log()
    log = log_todays_signals(legs, hot_form_entries, real_streak_entries, log, team_stat_entries)
    log = verify_pending_results(log)
    archive = load_archive()
    log, archive = fold_old_entries(log, archive)
    save_log(log)
    save_archive(archive)
    build_results_dashboard(log, archive)
