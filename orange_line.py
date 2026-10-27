#!/usr/bin/env python3
"""
Orange Line — NBA Player Props Predictor (Points, Rebounds, Assists, RA)
Same architecture as Blitz IQ (NFL) and the rest of this suite, adapted for
NBA player props:

- Uses ESPN's public (undocumented, no-key-needed) API, same host family as
  Blitz IQ -- site.api.espn.com/apis/site/v2/sports/basketball/nba/...
- Rebounds and Assists are modeled with POISSON, not Normal -- same
  reasoning as Blitz IQ's WR/TE receptions: these are discrete, low-count
  stats, not a bell-curve shape. RA (rebounds+assists combined) is also
  Poisson -- same idea, just a bigger count.
- Recency-weighted last-7-games average, shrunk toward a league-wide prior
  for small samples -- same small-sample protection as every other tool
  in this suite.

DATA-SOURCE HONESTY NOTE: ESPN's exact gamelog label names for NBA player
stats (whether rebounds is really labeled "REB", assists "AST") are NOT
confirmed against a live response in this build -- only that the gamelog
ENDPOINT itself exists (same host/pattern already proven working for
Blitz IQ's NFL players). This script uses the exact same defensive,
diagnostic-first parsing style Blitz IQ's own gamelog fetcher already
uses for this identical uncertainty: try the assumed label, and if it's
wrong, print a [DIAG] line showing the real labels ESPN actually
returned, rather than silently guessing wrong or crashing.

Setup:
    pip3 install requests --break-system-packages
    python3 orange_line.py

Output:
    docs/orange-line/index.html, docs/orange-line/orange_line_predictions.csv
"""

import os
import sys
import time
import math
import csv
import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import requests

# NBA/WNBA schedules are published on a US Eastern "game day" boundary (a
# 7pm ET tip-off is still "today" in ESPN's own scheduling, even though
# it's already past midnight UTC/past midnight in the UK) -- using UTC
# calendar dates for "today"/"tomorrow" drifts by a day depending on what
# time this script happens to run and what timezone is reading the result
# (user report, 2026-10-21: Orange Line showing the 23rd's games and
# missing the 22nd's, while bet365 -- showing UK local time -- had the
# 22nd's slate live). Both scoreboard pulls below are pinned to this same
# Eastern "today"/"tomorrow", computed fresh each run, rather than mixing
# ESPN's own undocumented "today" default (unknown timezone) with an
# explicitly UTC-computed "tomorrow" -- that mismatch between the two
# calls' timezone bases was the actual bug.
EASTERN = ZoneInfo("America/New_York")

LEAGUES = [
    {"key": "nba", "name": "NBA"},
    {"key": "wnba", "name": "WNBA"},
]
BASE_FMT = "https://site.api.espn.com/apis/site/v2/sports/basketball/{league}"
GAMELOG_BASE_FMT = "https://site.web.api.espn.com/apis/common/v3/sports/basketball/{league}/athletes/{athlete_id}/gamelog"
REQUEST_DELAY = 1.2  # same conservative pacing as Blitz IQ -- no official rate limit published


def _current_season_year(league_key):
    """ESPN's basketball 'season' param is keyed by the year a season
    ENDS. True for NBA, which crosses the calendar year -- the 2026-27
    season (tips off October 2026, Finals June 2027) is season=2027, not
    2026 -- but NOT for WNBA, which runs entirely within one calendar
    year (May-Oct), so season == datetime.now().year with no October
    rollover needed there.

    Every "current season" lookup in this file used to just take
    datetime.now().year regardless of league. That's only right for
    NBA's January-September tail (and for WNBA year-round) -- wrong for
    NBA's own October-December season START, where it silently pointed
    at the season that JUST ENDED (a full ~82-game past season) instead
    of the handful of games played so far this season.

    CONFIRMED as a real bug, not a theory (2026-10-26): the "100% This
    Season" panel showed players with 55-63/55-63 streaks only two
    weeks into the 2026-27 NBA season -- those counts only exist in a
    season that's already finished. This is also the most likely real
    explanation for an earlier, separate user report that DeAndre
    Ayton's Rebounds and LaMelo Ball's 3-Pointers Made history didn't
    match their actual recent games -- real numbers, just pulled from
    the wrong (just-finished) season rather than the current one."""
    now = datetime.now()
    if league_key == "nba" and now.month >= 10:
        return now.year + 1
    return now.year
RECENT_GAMES = 7  # matches this project's existing "last 7 games" framing
PRIOR_STRENGTH = 4
DEFAULT_TEAM_STD = 11.0  # rough per-team points stdev, same role as Blitz IQ's
ROSTER_CAP = 15  # players checked per team -- covers the full rotation without
                   # hammering the gamelog endpoint for deep-bench players who'd
                   # get filtered out by the games-played floor anyway


def _get(url, params=None, timeout=15):
    try:
        r = requests.get(url, params=params or {}, timeout=timeout)
        time.sleep(REQUEST_DELAY)
        if r.status_code != 200:
            print(f"  [!] {r.status_code} on {url}")
            return None
        return r
    except Exception as e:
        time.sleep(REQUEST_DELAY)
        print(f"  [!] request failed: {url} ({e})")
        return None


def poisson_pmf(k, lam):
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def poisson_cdf(k, lam):
    return sum(poisson_pmf(i, lam) for i in range(int(k) + 1))


def get_teams(league_key):
    r = _get(f"{BASE_FMT.format(league=league_key)}/teams", params={"limit": 40})
    if r is None:
        return []
    try:
        return r.json()['sports'][0]['leagues'][0]['teams']
    except Exception as e:
        print(f"  [!] couldn't parse teams response: {e}")
        return []


def get_scoreboard(league_key, date_str=None):
    """date_str: optional ESPN-style YYYYMMDD -- omit for "today" (ESPN's own
    default), pass tomorrow's date to pull that day's slate too (see
    build_predictions(), which calls this twice and merges the two)."""
    params = {"dates": date_str} if date_str else None
    r = _get(f"{BASE_FMT.format(league=league_key)}/scoreboard", params=params)
    if r is None:
        return []
    try:
        return r.json().get('events', [])
    except Exception as e:
        print(f"  [!] couldn't parse scoreboard: {e}")
        return []


roster_cache = {}
team_form_cache = {}


def _extract_completed_team_games(events, team_id):
    """Same shared parsing logic as Blitz IQ's _extract_completed_games --
    identical ESPN schedule response shape (competitions[0].competitors[]
    with a score.value per side), just ported to the basketball host."""
    completed = [e for e in events if e.get('competitions', [{}])[0].get('status', {})
                 .get('type', {}).get('completed')]
    completed.sort(key=lambda e: e.get('date', ''))
    recent = completed[-RECENT_GAMES:]
    scored, allowed = [], []
    for e in recent:
        comp = e['competitions'][0]
        competitors = comp.get('competitors', [])
        me = next((c for c in competitors if str(c['team']['id']) == str(team_id)), None)
        opp = next((c for c in competitors if str(c['team']['id']) != str(team_id)), None)
        if not me or not opp:
            continue
        try:
            scored.append(float(me['score']['value']))
            allowed.append(float(opp['score']['value']))
        except (KeyError, TypeError, ValueError):
            continue
    return scored, allowed


def get_team_form(league_key, team_id):
    """Last N completed games for a team -- points scored and allowed.
    Same Week-1-style fallback as Blitz IQ: early in a new season a team
    can have zero completed games yet, so this falls back to the tail of
    last season rather than going dark for the first week or two."""
    cache_key = (league_key, team_id)
    if cache_key in team_form_cache:
        return team_form_cache[cache_key]

    r = _get(f"{BASE_FMT.format(league=league_key)}/teams/{team_id}/schedule")
    scored, allowed, source = [], [], 'current'
    if r is not None:
        try:
            events = r.json().get('events', [])
            scored, allowed = _extract_completed_team_games(events, team_id)
        except Exception as e:
            print(f"  [!] couldn't parse schedule for team {team_id}: {e}")

    if not scored:
        current_year = _current_season_year(league_key)
        r2 = _get(f"{BASE_FMT.format(league=league_key)}/teams/{team_id}/schedule",
                   params={"season": current_year - 1})
        if r2 is not None:
            try:
                events2 = r2.json().get('events', [])
                scored, allowed = _extract_completed_team_games(events2, team_id)
                source = 'prior_season'
            except Exception as e:
                print(f"  [!] couldn't parse prior-season schedule for team {team_id}: {e}")

    if not scored:
        return None

    n = len(scored)
    form = {
        'avg_scored': round(sum(scored) / n, 1), 'avg_allowed': round(sum(allowed) / n, 1),
        'n_games': n, 'source': source, 'scored_list': scored, 'allowed_list': allowed,
    }
    team_form_cache[cache_key] = form
    return form


def league_averages(all_forms, fallback_scored):
    """IMPORTANT: must be called SEPARATELY per league (NBA vs WNBA),
    never on a blended set -- NBA teams average ~110-115 pts/game, WNBA
    meaningfully less, so a shared average would make WNBA teams look
    artificially cold and NBA teams artificially hot against the wrong
    baseline."""
    scored = [f['avg_scored'] for f in all_forms if f]
    allowed = [f['avg_allowed'] for f in all_forms if f]
    lg_scored = sum(scored) / len(scored) if scored else fallback_scored
    lg_allowed = sum(allowed) / len(allowed) if allowed else fallback_scored
    return lg_scored, lg_allowed


def recency_weighted_team(values):
    n = len(values)
    if n == 0:
        return None
    wts = [1.3 ** i for i in range(n)]
    return sum(w * v for w, v in zip(wts, values)) / sum(wts)


def shrink_team(value, n, league_avg, prior_strength=PRIOR_STRENGTH):
    return (n * value + prior_strength * league_avg) / (n + prior_strength)


def predict_team_total(h_form, a_form, lg_scored, lg_allowed):
    """Same Normal-distribution approach as Blitz IQ -- team scores are
    large, non-discrete-feeling totals, not a Poisson-shaped count."""
    h_recent_scored = recency_weighted_team(h_form['scored_list'])
    h_recent_allowed = recency_weighted_team(h_form['allowed_list'])
    a_recent_scored = recency_weighted_team(a_form['scored_list'])
    a_recent_allowed = recency_weighted_team(a_form['allowed_list'])

    h_scored = shrink_team(h_recent_scored, h_form['n_games'], lg_scored)
    h_allowed = shrink_team(h_recent_allowed, h_form['n_games'], lg_allowed)
    a_scored = shrink_team(a_recent_scored, a_form['n_games'], lg_scored)
    a_allowed = shrink_team(a_recent_allowed, a_form['n_games'], lg_allowed)

    exp_home = h_scored * (a_allowed / lg_allowed)
    exp_away = a_scored * (h_allowed / lg_allowed)
    exp_total = round(exp_home + exp_away, 1)
    total_std = math.sqrt(DEFAULT_TEAM_STD ** 2 + DEFAULT_TEAM_STD ** 2)

    return {
        'exp_home': round(exp_home, 1), 'exp_away': round(exp_away, 1),
        'exp_total': exp_total, 'total_std': round(total_std, 1),
    }


PACE_FAST_THRESHOLD = 1.04   # combined scoring env >= 4% above league -> "Fast"
PACE_SLOW_THRESHOLD = 0.96   # combined scoring env <= 4% below league -> "Slow"


def _pace_label(team_form, lg_scored, lg_allowed):
    """Pace/style proxy (user request, 2026-10-22). ESPN's scoreboard/
    team endpoints used elsewhere in this script don't carry a real
    possessions-per-game figure, so this uses the same combined
    scored+allowed environment a team's games actually play out in as a
    stand-in -- a team involved in high-combined-scoring games on both
    ends tends to be a fast, possession-heavy team even without a true
    pace count, the same practical proxy bettors reach for when real
    pace data isn't on hand. Informational only, same as Euro Ice's H2H
    line -- NOT blended into predict_team_total()'s projection, which
    already properly accounts for scored/allowed on each side."""
    team_env = team_form['avg_scored'] + team_form['avg_allowed']
    lg_env = lg_scored + lg_allowed
    if lg_env <= 0:
        return 'Average', 0.0
    ratio = team_env / lg_env
    if ratio >= PACE_FAST_THRESHOLD:
        label = 'Fast'
    elif ratio <= PACE_SLOW_THRESHOLD:
        label = 'Slow'
    else:
        label = 'Average'
    return label, round((ratio - 1) * 100, 1)  # label, % vs league


def pace_matchup(h_form, a_form, lg_scored, lg_allowed):
    h_label, h_pct = _pace_label(h_form, lg_scored, lg_allowed)
    a_label, a_pct = _pace_label(a_form, lg_scored, lg_allowed)
    if h_label == 'Fast' and a_label == 'Fast':
        verdict, icon = 'Fast-paced matchup — leans Over', '🔥'
    elif h_label == 'Slow' and a_label == 'Slow':
        verdict, icon = 'Slow-paced matchup — leans Under', '🐢'
    elif 'Average' in (h_label, a_label) and h_label == a_label:
        verdict, icon = 'Average pace both sides', '⚖️'
    else:
        verdict, icon = 'Mixed pace — no clear lean', '↔️'
    return {
        'home_label': h_label, 'home_pct': h_pct,
        'away_label': a_label, 'away_pct': a_pct,
        'verdict': verdict, 'icon': icon,
    }


def norm_cdf(x, mean, std):
    if std <= 0:
        return 1.0 if x >= mean else 0.0
    z = (x - mean) / (std * math.sqrt(2))
    return 0.5 * (1 + math.erf(z))


def normal_prop(mean, std, factor=0.72, round_to=0.5):
    if mean is None or std is None:
        return None
    raw_line = mean * factor
    line = math.floor(raw_line / round_to) * round_to
    if line < round_to:
        line = round_to
    prob_over = 1 - norm_cdf(line, mean, std)
    return {"line": line, "prob": round(prob_over * 100), "avg": mean}


def both_teams_over_ladder(exp_home, exp_away, std=DEFAULT_TEAM_STD):
    """'Both Teams to Score X Points' -- a bet365 market, NOT a combined/sum
    total (a real NBA game total is always 200+, so odds like "90 Points
    Yes 1.005 / 100 Points Yes 1.10 / 110 Points Yes 1.66" only make sense
    read as each team INDIVIDUALLY clearing X -- an AND of two Team
    Totals). Modeled as P(home >= X) * P(away >= X), same independence
    assumption Game Total already makes when it adds the two teams' std
    together. User request, 2026-10-26.

    Three rungs around the LOWER-projected team (the binding constraint
    for an AND condition, same logic as a parlay's weakest leg), stepped
    by 10 points and rounded to the nearest 10 -- a rough, easy-to-retune
    ladder shape, not calibrated against a real book's specific rungs."""
    if exp_home is None or exp_away is None:
        return []
    base = round(min(exp_home, exp_away) / 10) * 10
    rungs = []
    for x in (base - 10, base, base + 10):
        if x <= 0:
            continue
        prob = (1 - norm_cdf(x, exp_home, std)) * (1 - norm_cdf(x, exp_away, std))
        rungs.append({"threshold": x, "prob": round(prob * 100)})
    return rungs


def get_roster(league_key, team_id):
    """Full team roster -- NBA/WNBA don't have as clean a "starters by
    position" concept as NFL's depth chart, so instead of guessing at a
    depth-chart endpoint, this just pulls the roster and lets the
    games-played floor in get_player_gamelog naturally filter out
    deep-bench players who haven't logged enough games to project."""
    cache_key = (league_key, team_id)
    if cache_key in roster_cache:
        return roster_cache[cache_key]
    r = _get(f"{BASE_FMT.format(league=league_key)}/teams/{team_id}/roster")
    players = []
    if r is None:
        roster_cache[cache_key] = players
        return players
    try:
        data = r.json()
        athletes = data.get('athletes', [])
        # ESPN sometimes groups roster by position group (a list of
        # groups each with their own 'items'), sometimes returns a flat
        # list directly -- handle both rather than assume one shape.
        # Position (user request, 2026-10-22 -- wants scorers and real
        # ball-handlers told apart rather than picking an assist prop on
        # name recognition; a scoring wing/forward going 3+ assists is a
        # much shakier bet than a team's actual point guard at the same
        # line). ESPN's athlete objects elsewhere in this API family
        # carry a "position" object with an "abbreviation" (PG/SG/SF/PF/C)
        # -- not confirmed live for THIS specific roster endpoint, so
        # pulled defensively: missing/unexpected shape degrades to "" (no
        # tag shown) rather than crashing, same diagnostic-first style as
        # _fetch_gamelog_values above.
        def _position(a):
            pos = a.get('position')
            if isinstance(pos, dict):
                return pos.get('abbreviation', '') or ''
            return ''
        if athletes and isinstance(athletes[0], dict) and 'items' in athletes[0]:
            for group in athletes:
                for a in group.get('items', []):
                    players.append({'id': a.get('id'), 'name': a.get('displayName', a.get('fullName', '?')),
                                     'position': _position(a)})
        else:
            for a in athletes:
                players.append({'id': a.get('id'), 'name': a.get('displayName', a.get('fullName', '?')),
                                 'position': _position(a)})
    except Exception as e:
        print(f"  [!] couldn't parse roster for team {team_id}: {e}")
    players = players[:ROSTER_CAP]
    roster_cache[cache_key] = players
    return players


def recency_weighted(values):
    n = len(values)
    if n == 0:
        return None
    wts = [1.3 ** i for i in range(n)]
    return sum(w * v for w, v in zip(wts, values)) / sum(wts)


def shrink(value, n, prior, prior_strength=PRIOR_STRENGTH):
    return (n * value + prior_strength * prior) / (n + prior_strength)


player_gamelog_cache = {}

# Side-channel cache, populated as a byproduct of the SAME fetch
# get_player_gamelog already triggers for REB/AST/PTS/MIN -- holds each
# game's opponent tag alongside its date/value (the main cache above
# only keeps the plain value list, since every existing caller --
# recency_weighted, shrink, hit_rate, Hot Form/Real Streak, card
# rendering -- expects that exact shape and changing it would risk all
# of those). Keyed the same way: (league_key, athlete_id, stat_key,
# season) -> list of (date_str, value, opponent_dict_or_None), sorted
# the same as the main cache. Used only by get_player_vs_opponent()
# below, added for the "matchup history vs this exact opponent" feature
# (user request, 2026-10-25, reverse-engineering props.cash) -- reading
# this cache triggers ZERO extra API calls beyond what Orange Line was
# already making for its existing props.
player_gamelog_meta_cache = {}

_OPPONENT_SHAPE_LOGGED = False


def _extract_game_opponent(game, root_events):
    """Best-effort opponent tag for one gamelog event.

    UNCONFIRMED against a live response, same honesty-first caveat as
    every other gamelog field in this file (see this function's own
    module docstring) -- no [DIAG] sample gathered so far happened to
    include an opponent field, so this tries several plausible shapes
    rather than assuming one:
      1. An inline 'opponent' dict on the per-stat event itself
         (mirrors how 'gameDate' is read inline just above).
      2. The SAME event cross-referenced by id against this gamelog
         response's top-level 'events' dict, if there is one -- some of
         ESPN's other gamelog-shaped endpoints put per-game metadata
         there instead of inline on the stat-category event.
      3. A short 'atVs' text field (e.g. "vs LAL" / "@ BOS"), stripped
         down to a best-effort abbreviation.
    Returns {'abbrev', 'name'} (either half may be '') if ANY of these
    matched, or None if none did -- callers must treat None as "can't
    tell", never as "no opponent", and must degrade to showing nothing
    rather than a wrong matchup."""
    def _from_dict(d):
        if isinstance(d, dict) and (d.get('abbreviation') or d.get('displayName')):
            return {"abbrev": d.get('abbreviation', '') or '', "name": d.get('displayName', '') or ''}
        return None

    found = _from_dict(game.get('opponent'))
    if found:
        return found

    event_id = game.get('eventId') or game.get('id')
    if event_id and isinstance(root_events, dict):
        ev = root_events.get(str(event_id)) or root_events.get(event_id)
        if isinstance(ev, dict):
            found = _from_dict(ev.get('opponent'))
            if found:
                return found

    at_vs = game.get('atVs')
    if isinstance(at_vs, str) and at_vs.strip():
        token = at_vs.replace('vs', '').replace('@', '').strip()
        if token:
            return {"abbrev": token, "name": ""}

    return None


def _extract_game_date(game, root_events):
    """Best-effort date for one gamelog event -- same honesty-first,
    try-several-shapes approach as _extract_game_opponent just above
    (added 2026-10-26, after a real [DIAG] log confirmed the inline
    'gameDate'/'date'/'eventDate' fields this used to rely on ARE
    missing on ESPN's actual response, silently tripping the "falling
    back to API order as-is" branch below -- the leading suspect for
    Orange Line showing the wrong games/wrong order for real players
    (user report, 2026-10-26: DeAndre Ayton's Rebounds and LaMelo
    Ball's 3-Pointers Made history both failed to match real box
    scores/odds, on two different stats for two different players,
    which points at something upstream of any single stat's column
    lookup -- this ordering bug is exactly that kind of upstream
    issue).

    Tries, in order:
      1. The same inline fields previously tried directly.
      2. The SAME event cross-referenced by id against this response's
         top-level 'events' -- exactly like _extract_game_opponent
         already does (root_events' real shape, dict-keyed-by-id vs a
         plain list, is handled either way, same as there).
    Returns a date string, or None if nothing matched -- the caller
    already has a visible, documented fallback for None (sorts by API
    order and prints a [DIAG] line), so this never has to guess."""
    inline = game.get('gameDate') or game.get('date') or game.get('eventDate')
    if inline:
        return inline

    event_id = game.get('eventId') or game.get('id')
    if not event_id:
        return None

    ev = None
    if isinstance(root_events, dict):
        ev = root_events.get(str(event_id)) or root_events.get(event_id)
    elif isinstance(root_events, list):
        ev = next((e for e in root_events
                   if isinstance(e, dict) and str(e.get('id')) == str(event_id)), None)

    if isinstance(ev, dict):
        return ev.get('date') or ev.get('gameDate') or ev.get('eventDate')
    return None


def get_player_vs_opponent(league_key, athlete_id, stat_key, opponent_abbr, opponent_name):
    """This player's raw per-game values (current + prior season,
    whichever have already been fetched via get_player_gamelog) from
    games specifically against `opponent_abbr`/`opponent_name` -- the
    "matchup history vs this exact opponent" props.cash has and Orange
    Line's own recent-form screens don't (those are opponent-agnostic).

    Reads ONLY from player_gamelog_meta_cache -- never fetches on its
    own -- so this costs zero extra API calls on top of whatever Orange
    Line was already pulling for this player's props, but also means it
    can only see opponent data for seasons a normal prop call already
    touched (current season, and prior season only if current season's
    fetch came back empty). Returns [] whenever nothing matched --
    which, given _extract_game_opponent's own unconfirmed-shape caveat
    above, may mean "no data available to tell" rather than a genuine
    "never played them" -- never treat an empty list here as proof of
    anything."""
    if not opponent_abbr and not opponent_name:
        return []
    out = []
    current_year = _current_season_year(league_key)
    for season in (current_year, current_year - 1):
        meta = player_gamelog_meta_cache.get((league_key, athlete_id, stat_key, season), [])
        for date_str, value, opp in meta:
            if not opp:
                continue
            abbr_match = bool(opponent_abbr) and opp.get('abbrev', '').lower() == opponent_abbr.lower()
            name_match = bool(opponent_name) and opp.get('name') and opponent_name.lower() in opp.get('name', '').lower()
            if abbr_match or name_match:
                out.append(value)
    return out


def _parse_plain_stat(raw):
    """Default value parser -- REB/AST/PTS/MIN are confirmed-plain
    numbers on this endpoint, so this is just float()."""
    return float(raw)


def _parse_makes_attempts(raw):
    """Value parser for a combined 'makes-attempts' stat, e.g. 3PT
    commonly rendering as "3-7" (3 made, 7 attempted) rather than a
    separate made-only column -- UNCONFIRMED for this specific gamelog
    endpoint (no [DIAG] sample gathered so far happened to include a
    3PT value), same honesty-first caveat as every other field in this
    file not yet proven against a live response. Tries the
    "makes-attempts" shape first (splits on '-', keeps the made count);
    falls back to treating the whole field as a plain number, in case
    this one turns out to already be made-only like every OTHER
    confirmed stat here. Returns None (not 0) on total failure, so the
    caller can skip the game rather than silently recording a zero."""
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


def _fetch_gamelog_values(league_key, athlete_id, stat_key, season, value_parser=_parse_plain_stat):
    """Same defensive parsing as Blitz IQ's _fetch_gamelog_values --
    identical API family, identical uncertainty about exact label names,
    so identical treatment: try the plausible label, print a diagnostic
    with the REAL labels found if it doesn't match, never guess silently.

    ORDER FIX (Orange Line only, not carried in Blitz IQ): Blitz IQ's own
    comments flag that this event list's chronological order was never
    confirmed -- fine there, since nothing in that script depends on
    "most recent" specifically. Orange Line's Real Streak DOES depend on
    it (walking backward from the most recent game), so this version
    explicitly captures each game's date where available and SORTS by
    it, rather than trusting whatever order the API happens to return.
    If no date field is found at all, falls back to API order as-is and
    prints a diagnostic -- so a missing date field degrades visibly
    (streak feature may be unreliable) instead of silently.

    value_parser (added for 3-Pointers Made, user request, 2026-10-25):
    REB/AST/PTS/MIN are confirmed-plain numbers and use the default
    _parse_plain_stat (a thin float() wrapper, same behavior as before
    this parameter existed). 3PT is passed _parse_makes_attempts
    instead, since ESPN's shooting splits are commonly rendered as a
    combined "makes-attempts" string rather than a made-only column."""
    global _OPPONENT_SHAPE_LOGGED
    r = _get(GAMELOG_BASE_FMT.format(league=league_key, athlete_id=athlete_id),
             params={"season": season})
    dated_values = []  # (date_str_or_None, value, opponent_dict_or_None)
    if r is None:
        print(f"    [!] gamelog fetch failed for athlete {athlete_id}, season {season} (no response)")
        return dated_values
    try:
        data = r.json()
        root_events = data.get('events')  # see _extract_game_opponent's cross-reference path
        top_keys = list(data.keys())
        if top_keys == ['filters']:
            print(f"    [!] gamelog for athlete {athlete_id}, season {season}: still only got 'filters' even with season param.")
        season_data = data.get('seasonTypes', [])
        if not season_data and top_keys != ['filters']:
            print(f"    [!] gamelog for athlete {athlete_id}, season {season}: no 'seasonTypes' key. Top-level keys: {top_keys}")

        root_labels = data.get('labels') or data.get('names') or data.get('displayNames')
        if season_data and not root_labels:
            print(f"    [DIAG] athlete {athlete_id}, season {season}: no root-level labels/names/displayNames. Top-level keys: {top_keys}")

        # CONFIRMED bug, 2026-10-27: season_data was being looped over
        # in full with no filter for season TYPE -- ESPN's 'seasonTypes'
        # list holds preseason, regular season, and (eventually)
        # postseason as SEPARATE entries, and this was silently
        # concatenating all of them into one "this season" game list.
        # Caught because SGA's "100% This Season" card showed 6/6 games
        # by Oct 27, but the Thunder's real schedule (confirmed via web
        # search of the official 2026-27 schedule) shows only 4 games
        # played by that date -- the extra 2 match their October
        # preseason slate. Filtering to the regular-season entry only
        # below. ESPN's exact field name for season type isn't confirmed
        # from a live response in this environment, so this tries a few
        # plausible shapes (displayName/name/shortDisplayName, plus a
        # 'type' field that may itself be a dict) and only filters when
        # at least one entry is positively identified as regular season
        # -- if none can be identified, it logs a loud diagnostic and
        # falls back to the old include-everything behavior rather than
        # risk silently dropping all games on an unexpected shape.
        def _season_type_name(st):
            name = st.get('displayName') or st.get('name') or st.get('shortDisplayName') or ''
            type_field = st.get('type')
            if isinstance(type_field, dict):
                type_name = type_field.get('name') or type_field.get('type') or type_field.get('abbreviation') or ''
            else:
                type_name = type_field or ''
            return f"{name} {type_name}".strip().lower()

        regular_season_data = [st for st in season_data if 'regular' in _season_type_name(st)]
        if regular_season_data:
            season_data = regular_season_data
        elif season_data:
            print(f"    [DIAG] athlete {athlete_id}, season {season}: couldn't identify a 'regular season' "
                  f"entry among {len(season_data)} seasonTypes entries -- shapes seen: "
                  f"{[{k: st.get(k) for k in ('displayName', 'name', 'shortDisplayName', 'type')} for st in season_data]}. "
                  f"Including ALL season types (may mix preseason/postseason games into 'this season' counts) "
                  f"until this is confirmed against a real response.")

        printed_sample = False
        for st in season_data:
            # Logs EVERY category now, not just the first -- if ESPN's
            # response actually carries more than one category here
            # (e.g. a totals/split row alongside the real per-game
            # list), the old "only log the first" version would have
            # hidden that completely, silently concatenating whatever
            # categories exist into one game list. Added 2026-10-26
            # alongside _extract_game_date, chasing the same user
            # report (see that function's own comment).
            cats = st.get('categories', [])
            for cat in cats:
                cat_events = cat.get('events', [])
                print(f"    [DIAG] athlete {athlete_id}, season {season}: category "
                      f"'{cat.get('type') or cat.get('displayName') or '?'}' has {len(cat_events)} events, "
                      f"keys: {list(cat.keys())}")
                if not printed_sample:
                    print(f"    [DIAG] athlete {athlete_id}, season {season}: {len(cats)} total categories, "
                          f"root_labels sample: {str(root_labels)[:200]}")
                    printed_sample = True
                if not root_labels:
                    continue
                for game in cat_events:
                    stats = game.get('stats', [])
                    if stat_key in root_labels:
                        idx = root_labels.index(stat_key)
                        try:
                            value = value_parser(stats[idx])
                        except (IndexError, ValueError, TypeError):
                            continue
                        if value is None:
                            continue
                        date_str = _extract_game_date(game, root_events)
                        opp = _extract_game_opponent(game, root_events)
                        dated_values.append((date_str, value, opp))
        if season_data and not dated_values:
            print(f"    [!] gamelog for athlete {athlete_id}, season {season}, stat '{stat_key}': no matching values found "
                  f"-- stat_key likely doesn't match ESPN's actual label name. Check the [DIAG] line above for real labels.")
    except Exception as e:
        print(f"  [!] couldn't parse gamelog for athlete {athlete_id}, season {season}: {e}")

    if dated_values and all(d is not None for d, _, _ in dated_values):
        dated_values.sort(key=lambda t: t[0])  # guarantees oldest -> newest
    elif dated_values:
        print(f"    [DIAG] athlete {athlete_id}, season {season}, stat '{stat_key}': some/all games missing a "
              f"date field -- falling back to API order as-is. Real Streak for this player may be unreliable "
              f"until this is confirmed against a real response.")

    if dated_values and not _OPPONENT_SHAPE_LOGGED and all(opp is None for _, _, opp in dated_values):
        print(f"    [DIAG] athlete {athlete_id}, season {season}: no opponent field found on any gamelog event "
              f"(checked inline 'opponent', a top-level 'events' cross-reference, and 'atVs') -- matchup-history "
              f"vs a specific opponent will show nothing until this is confirmed against a real response.")
        _OPPONENT_SHAPE_LOGGED = True

    player_gamelog_meta_cache[(league_key, athlete_id, stat_key, season)] = \
        [(d, v, opp) for d, v, opp in dated_values]
    return [v for _, v, _ in dated_values]


def get_player_gamelog(league_key, athlete_id, stat_key, value_parser=_parse_plain_stat):
    # value_parser isn't part of the cache key -- a given (league, athlete,
    # stat) is only ever fetched with one parser in practice (PROP_CONFIG
    # fixes the pairing per stat_key), so this mirrors the existing
    # cache-key shape rather than adding a parser identity to it.
    cache_key = (league_key, athlete_id, stat_key)
    if cache_key in player_gamelog_cache:
        return player_gamelog_cache[cache_key]
    current_year = _current_season_year(league_key)
    values = _fetch_gamelog_values(league_key, athlete_id, stat_key, current_year, value_parser=value_parser)
    if not values:
        values = _fetch_gamelog_values(league_key, athlete_id, stat_key, current_year - 1, value_parser=value_parser)
    values = values[-RECENT_GAMES:]  # oldest-first list -- last N entries ARE the most recent N games
    player_gamelog_cache[cache_key] = values
    return values


PROP_CONFIG = {
    'REB': {'stat': 'REB', 'label': 'Rebounds', 'prior': 5.5},
    'AST': {'stat': 'AST', 'label': 'Assists', 'prior': 3.0},
    'PTS': {'stat': 'PTS', 'label': 'Points', 'prior': 10.0},
    # 3-Pointers Made (user request, 2026-10-25). 'stat': '3PT' is the
    # confirmed-live root label (seen in project_player_props' own 'MIN'
    # comment: [...'3PT','3P%'...]) -- but that confirmation only covers
    # the LABEL existing, not its value shape, so this pairs it with
    # _parse_makes_attempts rather than the plain-float default every
    # other stat here uses. prior=1.2 is a rough role-player-average
    # starting point (lower than REB/AST's priors, matching that most
    # rotation players make 0-2 threes/game) -- easy to retune once real
    # data is flowing.
    '3PM': {'stat': '3PT', 'label': '3-Pointers Made', 'prior': 1.2, 'value_parser': _parse_makes_attempts},
}


PACE_SCALE_MIN = 0.85
PACE_SCALE_MAX = 1.15
MIN_REST_RISK_RATIO = 0.70  # last game's minutes vs recent avg, below this -> flagged
MIN_REST_RISK_FLOOR = 15.0  # don't flag bench noise -- only players averaging real minutes


def project_player_props(league_key, team_id, pts_scale=1.0):
    """pts_scale (user request, 2026-10-22 -- "will top scorers hit their
    average point zone" given the pace matchup): scales pts_proj by how
    much MORE or LESS this team is projected to score in TONIGHT's
    specific matchup vs their own season average, i.e.
    exp_team / team_form['avg_scored'] from predict_team_total() --
    already accounts for both pace AND the opponent's own defense
    (exp_home/exp_away are themselves scored * opponent_allowed/lg_allowed),
    so this is a better-grounded adjustment than reapplying the blunt
    pace label a second time. Clamped to [0.85, 1.15] -- same defensive
    clamping philosophy as predict_team_total()'s own lh/la bounds --
    so one early-season outlier game can't send a player's whole
    projection off a cliff. Caller passes 1.0 (the default) when no team
    form was available, leaving pts_proj unadjusted. Scoped to points
    only, per the specific ask -- rebounds/assists/RA are untouched."""
    roster = get_roster(league_key, team_id)
    if not roster:
        print(f"    no roster found for team {team_id} — 0 player props possible")
    props = []
    for player in roster:
        if not player.get('id'):
            continue
        rebs = get_player_gamelog(league_key, player['id'], PROP_CONFIG['REB']['stat'])
        asts = get_player_gamelog(league_key, player['id'], PROP_CONFIG['AST']['stat'])
        pts = get_player_gamelog(league_key, player['id'], PROP_CONFIG['PTS']['stat'])
        if len(rebs) < 3 or len(asts) < 3 or len(pts) < 3 or len(rebs) != len(asts) or len(rebs) != len(pts):
            continue  # not enough data, or mismatched game counts between
                        # the separate stat pulls -- skip rather than pair
                        # up games that may not actually correspond
        n = len(rebs)
        reb_proj = shrink(recency_weighted(rebs), n, PROP_CONFIG['REB']['prior'])
        ast_proj = shrink(recency_weighted(asts), n, PROP_CONFIG['AST']['prior'])
        pts_proj_raw = shrink(recency_weighted(pts), n, PROP_CONFIG['PTS']['prior'])
        pts_proj = pts_proj_raw * pts_scale
        ra_values = [r + a for r, a in zip(rebs, asts)]
        ra_proj = reb_proj + ast_proj

        # 3-Pointers Made (user request, 2026-10-25) -- fetched and
        # gated SEPARATELY from the REB/AST/PTS trio above rather than
        # folded into that same mismatch check: _parse_makes_attempts is
        # unconfirmed against a live response (see its own docstring)
        # and may legitimately drop more games than the plain-float
        # stats do, so a thin/failed 3PT pull degrades to "no 3PM prop
        # for this player" rather than killing their REB/AST/PTS/RA
        # props too. tpm_n_games tracked separately from `n` for the
        # same reason -- build_legs uses it instead of `n` for this
        # stat's own game-count display.
        tpm = get_player_gamelog(league_key, player['id'], PROP_CONFIG['3PM']['stat'],
                                  value_parser=PROP_CONFIG['3PM']['value_parser'])
        tpm_proj, tpm_n_games = None, 0
        if len(tpm) >= 3:
            tpm_n_games = len(tpm)
            tpm_proj = round(shrink(recency_weighted(tpm), tpm_n_games, PROP_CONFIG['3PM']['prior']), 1)

        # Minutes/rest risk (user request, same turn) -- same gamelog
        # endpoint already hit for REB/AST/PTS, 'MIN' is one of the same
        # root-level labels (confirmed live via the [DIAG] sample the
        # user shared earlier: ['MIN','FG','FG%','3PT','3P%','FT','FT%',
        # 'REB','AST','BLK','STL','PF','TO','PTS']). Best-effort: missing
        # or short minutes data just leaves rest_risk False rather than
        # blocking the rest of this player's prop.
        mins = get_player_gamelog(league_key, player['id'], 'MIN')
        min_avg, min_last, rest_risk = None, None, False
        if len(mins) >= 3:
            min_avg = round(recency_weighted(mins), 1)
            min_last = mins[-1]
            rest_risk = min_avg >= MIN_REST_RISK_FLOOR and min_last < MIN_REST_RISK_RATIO * min_avg

        props.append({
            'name': player['name'], 'player_id': player['id'], 'position': player.get('position', ''),
            'reb_proj': round(reb_proj, 1), 'ast_proj': round(ast_proj, 1), 'ra_proj': round(ra_proj, 1),
            'pts_proj': round(pts_proj, 1), 'pts_proj_raw': round(pts_proj_raw, 1),
            'tpm_proj': tpm_proj,
            'min_avg': min_avg, 'min_last': min_last, 'rest_risk': rest_risk,
            'rebs': rebs, 'asts': asts, 'ra_values': ra_values, 'pts': pts, 'tpm': tpm,
            'n_games': n, 'tpm_n_games': tpm_n_games,
        })
    return props


def safe_line(mean, factor=0.72, round_to=0.5):
    if mean is None:
        return None
    raw_line = mean * factor
    line = math.floor(raw_line * 2) / 2
    if line < 0.5:
        line = 0.5
    threshold = int(math.floor(line)) + 1
    prob = 1 - poisson_cdf(threshold - 1, mean)
    return {"line": line, "prob": round(prob * 100), "avg": mean}


def hit_rate(values, line):
    if not values:
        return None
    hits = sum(1 for v in values if v > line)
    return {"hits": hits, "total": len(values)}


LEAGUE_FALLBACK_AVG = {"nba": 113.0, "wnba": 82.0}  # only used if literally no team
                                                        # in a league has any form data yet


def build_predictions():
    predictions = []
    for league in LEAGUES:
        league_key, league_name = league["key"], league["name"]
        print(f"\nFetching {league_name} teams…")
        teams = get_teams(league_key)
        if not teams:
            print(f"No {league_name} teams returned — skipping this league for now, "
                  f"still trying the others.")
            continue

        print(f"Fetching {league_name} team form for {len(teams)} teams…")
        all_forms = {}
        for t in teams:
            tid = t['team']['id']
            all_forms[tid] = get_team_form(league_key, tid)
        lg_scored, lg_allowed = league_averages(all_forms.values(), LEAGUE_FALLBACK_AVG.get(league_key, 100.0))
        print(f"  {league_name} averages: {lg_scored:.1f} scored/game, {lg_allowed:.1f} allowed/game")

        print(f"Fetching {league_name}'s scoreboard for today and tomorrow…")
        now_et = datetime.now(EASTERN)
        today_et = now_et.strftime('%Y%m%d')
        tomorrow_et = (now_et + timedelta(days=1)).strftime('%Y%m%d')
        events_today = get_scoreboard(league_key, today_et)
        events_tomorrow = get_scoreboard(league_key, tomorrow_et)
        seen_ids = set()
        events = []
        for ev in events_today + events_tomorrow:
            eid = ev.get('id')
            if eid and eid in seen_ids:
                continue  # same game showed up in both pulls -- keep the first copy only
            if eid:
                seen_ids.add(eid)
            events.append(ev)
        print(f"{len(events)} {league_name} games today & tomorrow "
              f"({len(events_today)} today, {len(events_tomorrow)} tomorrow)")

        for e in events:
            comp = e.get('competitions', [{}])[0]
            competitors = comp.get('competitors', [])
            home = next((c for c in competitors if c.get('homeAway') == 'home'), None)
            away = next((c for c in competitors if c.get('homeAway') == 'away'), None)
            if not home or not away:
                continue
            if comp.get('status', {}).get('type', {}).get('completed'):
                continue

            h_id, a_id = home['team']['id'], away['team']['id']
            h_form = all_forms.get(h_id) or get_team_form(league_key, h_id)
            a_form = all_forms.get(a_id) or get_team_form(league_key, a_id)
            team_total_proj = {}
            pace = None
            home_pts_scale, away_pts_scale = 1.0, 1.0
            if h_form and a_form:
                team_total_proj = predict_team_total(h_form, a_form, lg_scored, lg_allowed)
                pace = pace_matchup(h_form, a_form, lg_scored, lg_allowed)
                # Points pace-adjustment scale -- see project_player_props()'s
                # own docstring for why this ratio (not the pace label) is
                # used. avg_scored is already a confirmed-nonzero value by
                # the time a form dict exists (get_team_form only returns
                # one once it has at least one completed game).
                if team_total_proj.get('exp_home') is not None:
                    home_pts_scale = min(max(team_total_proj['exp_home'] / h_form['avg_scored'],
                                              PACE_SCALE_MIN), PACE_SCALE_MAX)
                if team_total_proj.get('exp_away') is not None:
                    away_pts_scale = min(max(team_total_proj['exp_away'] / a_form['avg_scored'],
                                              PACE_SCALE_MIN), PACE_SCALE_MAX)
            else:
                print(f"    no team form for one/both sides of {away['team']['displayName']} @ "
                      f"{home['team']['displayName']} — Team/Game Total skipped for this game, "
                      f"player props unaffected")

            print(f"  Player props: {away['team']['displayName']} @ {home['team']['displayName']} ({league_name})")
            home_props = project_player_props(league_key, h_id, pts_scale=home_pts_scale)
            away_props = project_player_props(league_key, a_id, pts_scale=away_pts_scale)
            predictions.append({
                'date': e.get('date', ''), 'game_id': e.get('id'),
                'league': league_name,
                'match': f"{away['team']['displayName']} @ {home['team']['displayName']}",
                'home_team': home['team']['displayName'], 'away_team': away['team']['displayName'],
                # Abbreviations, for matching against get_player_vs_opponent's
                # opponent tags (matchup-history feature, see that function) --
                # UNCONFIRMED field on this scoreboard competitor shape, same
                # defensive '' fallback as get_roster's own position lookup.
                'home_abbr': home['team'].get('abbreviation', ''), 'away_abbr': away['team'].get('abbreviation', ''),
                'home_props': home_props, 'away_props': away_props,
                'home_form': h_form, 'away_form': a_form, 'pace': pace,
                **team_total_proj,
            })
    return predictions


def build_legs(predictions):
    legs = []
    for p in predictions:
        match_label = p["match"]
        game_id = p.get("game_id")
        game_date = (p.get("date") or "")[:10]
        league_name = p.get("league", "")
        hf, af = p.get("home_form"), p.get("away_form")

        if hf and af and p.get("exp_home") is not None:
            # factor=1.0 -- unlike player props, a Team/Game Total line
            # needs to sit at the model's actual projected number, not
            # shaded down by the same 0.72 margin player props use. The
            # shared default produced lines ~28% below the projection
            # (e.g. a 230.6pt projected game total printing "Over 166.0"),
            # a number no real book offers -- user report, 2026-10-21.
            home_total = normal_prop(p["exp_home"], DEFAULT_TEAM_STD, factor=1.0)
            if home_total:
                legs.append({
                    "match": f"{match_label} ({league_name})",
                    "market": f"{p['home_team']} Over {home_total['line']} Points",
                    "prob": home_total["prob"], "category": "Team Total",
                    "hit_rate": hit_rate(hf.get("scored_list"), home_total["line"]),
                    "detail": f"{league_name} · proj {home_total['avg']} pts ({hf['n_games']}gm"
                              f"{' · last season' if hf.get('source') == 'prior_season' else ''})",
                    "history": "/".join(str(v) for v in hf.get("scored_list", [])),
                    "game_id": game_id, "game_date": game_date, "line": home_total["line"],
                    "is_home": True, "league": league_name,
                })
            away_total = normal_prop(p["exp_away"], DEFAULT_TEAM_STD, factor=1.0)
            if away_total:
                legs.append({
                    "match": f"{match_label} ({league_name})",
                    "market": f"{p['away_team']} Over {away_total['line']} Points",
                    "prob": away_total["prob"], "category": "Team Total",
                    "hit_rate": hit_rate(af.get("scored_list"), away_total["line"]),
                    "detail": f"{league_name} · proj {away_total['avg']} pts ({af['n_games']}gm"
                              f"{' · last season' if af.get('source') == 'prior_season' else ''})",
                    "history": "/".join(str(v) for v in af.get("scored_list", [])),
                    "game_id": game_id, "game_date": game_date, "line": away_total["line"],
                    "is_home": False, "league": league_name,
                })
            game_total = normal_prop(p["exp_total"], p["total_std"], factor=1.0)
            if game_total:
                legs.append({
                    "match": f"{match_label} ({league_name})",
                    "market": f"Game Over {game_total['line']} Total Points",
                    "prob": game_total["prob"], "category": "Game Total",
                    # No hit_rate -- same reasoning as Blitz IQ: no real paired
                    # "these two teams' actual combined score" history exists.
                    "hit_rate": None,
                    "detail": f"{league_name} · proj {game_total['avg']} pts ({hf['n_games']}v{af['n_games']}gm)",
                    "history": None,
                    "game_id": game_id, "game_date": game_date, "line": game_total["line"],
                    "league": league_name,
                })

            for rung in both_teams_over_ladder(p["exp_home"], p["exp_away"]):
                if rung["prob"] <= 0 or rung["prob"] >= 100:
                    continue  # a 0% or 100% rung tells the user nothing a real ladder doesn't already imply
                legs.append({
                    "match": f"{match_label} ({league_name})",
                    "market": f"Both Teams Over {rung['threshold']} Points",
                    "prob": rung["prob"], "category": "Both Teams Total",
                    # Same reasoning as Game Total -- no real paired "both
                    # teams' actual scores together" history exists.
                    "hit_rate": None,
                    "detail": f"{league_name} · needs {p['home_team']} AND {p['away_team']} each "
                              f"{rung['threshold']}+ · proj {p['exp_home']} / {p['exp_away']}",
                    "history": None,
                    "game_id": game_id, "game_date": game_date, "line": rung["threshold"],
                    "league": league_name,
                })

        for team_name, props in [(p["home_team"], p.get("home_props") or []),
                                   (p["away_team"], p.get("away_props") or [])]:
            for prop in props:
                for stat_key, values, proj, label, n_games in [
                    ("REB", prop["rebs"], prop["reb_proj"], "Rebounds", prop["n_games"]),
                    ("AST", prop["asts"], prop["ast_proj"], "Assists", prop["n_games"]),
                    ("RA", prop["ra_values"], prop["ra_proj"], "Rebounds+Assists", prop["n_games"]),
                    ("PTS", prop["pts"], prop["pts_proj"], "Points", prop["n_games"]),
                    # 3-Pointers Made (user request, 2026-10-25) -- its own
                    # n_games (tpm_n_games), separate from the REB/AST/PTS
                    # trio's shared `n_games`, since _parse_makes_attempts
                    # can legitimately drop a different number of games
                    # (see project_player_props). tpm_proj is None whenever
                    # fewer than 3 games parsed, and safe_line(None) already
                    # returns None -- same "if not result: continue" skip
                    # below handles that with no extra check needed here.
                    ("3PM", prop["tpm"], prop["tpm_proj"], "3-Pointers Made", prop["tpm_n_games"]),
                ]:
                    result = safe_line(proj)
                    if not result:
                        continue

                    # Matchup history vs the EXACT upcoming opponent (user
                    # request, 2026-10-25, reverse-engineering props.cash) --
                    # scoped to PTS only to keep this to one stat while the
                    # underlying opponent-tag extraction is still unconfirmed
                    # against a live response (see get_player_vs_opponent's
                    # own docstring). Costs zero extra API calls: reads from
                    # whatever get_player_gamelog('PTS') already populated
                    # for this player a few lines up in project_player_props.
                    vs_opponent = None
                    if stat_key == "PTS":
                        is_home_leg = team_name == p["home_team"]
                        opp_abbr = p.get("away_abbr") if is_home_leg else p.get("home_abbr")
                        opp_name = p["away_team"] if is_home_leg else p["home_team"]
                        # league_name.lower() ("NBA"->"nba"/"WNBA"->"wnba")
                        # recovers the lowercase league_key get_player_gamelog
                        # was actually cached under -- true by construction of
                        # the LEAGUES config above, not a guess.
                        vs_values = get_player_vs_opponent(
                            league_name.lower(), prop["player_id"], "PTS", opp_abbr, opp_name)
                        if vs_values:
                            vs_opponent = {
                                "opponent": opp_name,
                                "values": vs_values,
                                "hit_rate": hit_rate(vs_values, result["line"]),
                            }

                    legs.append({
                        "match": f"{match_label} ({league_name})",
                        "market": f"{prop['name']} Over {result['line']} {label}",
                        "prob": result["prob"], "category": label,
                        "hit_rate": hit_rate(values, result["line"]),
                        "detail": f"{team_name} · {league_name} · proj {result['avg']} ({n_games}gm)",
                        "history": "/".join(str(v) for v in values),
                        "vs_opponent": vs_opponent,
                        "game_id": game_id, "game_date": game_date, "line": result["line"],
                        "player_id": prop["player_id"], "stat_key": stat_key,
                        "is_home": team_name == p["home_team"], "league": league_name,
                    })
    return legs


# --- Hot Form / Real Streak ---------------------------------------------
# Same distinction as every other tool in this suite -- "Hot Form" is an
# AVERAGE over the last HOT_FORM_MIN_GAMES games, which can flag a player
# even if their most recent game was quiet, as long as earlier games
# pulled the average up. "Real Streak" is a stricter, separate check:
# walking backward from the most recent game (now that _fetch_gamelog_values
# sorts by date rather than trusting API order) and counting how many in a
# ROW cleared a per-game threshold, stopping at the first one that didn't.
# PTS thresholds follow the same ratio every other stat here already uses
# between its Hot Form bar (an AVERAGE, so set higher) and its Real Streak
# bar (every one of several CONSECUTIVE games must individually clear it,
# so set lower) -- roughly 1.3x. Real Streak's 15.0 is per user request
# (2026-10-21: "include 15 point streak players"); Hot Form's 20.0 follows
# from that same ratio, not a separate request -- easy to retune later.
HOT_FORM_MIN = {"REB": 8.0, "AST": 5.0, "RA": 13.0, "PTS": 20.0,
                 # 3-Pointers Made (user request, 2026-10-25) -- a looser
                 # ~1.5x ratio to its own Real Streak bar below rather than
                 # the ~1.3x every other stat here uses, since 3PM is a much
                 # lower-count stat where that smaller gap would barely
                 # separate the two screens at all. Both numbers are a
                 # starting judgment call, not derived from real data yet --
                 # easy to retune once results are flowing.
                 "3PM": 3.0}
HOT_FORM_MIN_GAMES = 5
REAL_STREAK_THRESHOLD = {"REB": 6.0, "AST": 4.0, "RA": 10.0, "PTS": 15.0, "3PM": 2.0}
REAL_STREAK_MIN_LENGTH = 3
STAT_LABELS = {"REB": "Rebounds", "AST": "Assists", "RA": "Rebounds+Assists", "PTS": "Points", "3PM": "3-Pointers Made"}


def _last_n_avg(values, n):
    if len(values) < n:
        return None
    lastn = values[-n:]
    return round(sum(lastn) / len(lastn), 2), lastn


def _current_stat_streak(values, threshold):
    """values must be oldest-first (see _fetch_gamelog_values) -- walking
    in REVERSE goes from the most recent game backward."""
    streak = 0
    for v in reversed(values):
        if v >= threshold:
            streak += 1
        else:
            break
    return streak


def build_hot_form_entries(predictions):
    entries = []
    for p in predictions:
        for team_name, props in [(p["home_team"], p.get("home_props") or []),
                                   (p["away_team"], p.get("away_props") or [])]:
            for prop in props:
                for stat_key, values in [("REB", prop["rebs"]), ("AST", prop["asts"]), ("RA", prop["ra_values"]),
                                          ("PTS", prop["pts"]), ("3PM", prop["tpm"])]:
                    result = _last_n_avg(values, HOT_FORM_MIN_GAMES)
                    if not result:
                        continue
                    avg, lastn = result
                    if avg >= HOT_FORM_MIN[stat_key]:
                        entries.append({
                            "name": prop["name"], "team": team_name, "league": p.get("league", ""),
                            "match": p["match"], "stat": stat_key, "avg": avg, "values": lastn,
                            "player_id": prop["player_id"], "game_id": p.get("game_id"),
                            "date": p.get("date", ""), "threshold": HOT_FORM_MIN[stat_key],
                        })
    entries.sort(key=lambda e: -e["avg"])
    return entries


def build_real_streak_entries(predictions):
    entries = []
    for p in predictions:
        for team_name, props in [(p["home_team"], p.get("home_props") or []),
                                   (p["away_team"], p.get("away_props") or [])]:
            for prop in props:
                for stat_key, values in [("REB", prop["rebs"]), ("AST", prop["asts"]), ("RA", prop["ra_values"]),
                                          ("PTS", prop["pts"]), ("3PM", prop["tpm"])]:
                    streak_len = _current_stat_streak(values, REAL_STREAK_THRESHOLD[stat_key])
                    if streak_len >= REAL_STREAK_MIN_LENGTH:
                        entries.append({
                            "name": prop["name"], "team": team_name, "league": p.get("league", ""),
                            "match": p["match"], "stat": stat_key, "streak_len": streak_len,
                            "streak_games": values[-streak_len:], "full_sample": streak_len >= len(values),
                            "player_id": prop["player_id"], "game_id": p.get("game_id"),
                            "date": p.get("date", ""), "threshold": REAL_STREAK_THRESHOLD[stat_key],
                        })
    entries.sort(key=lambda e: -e["streak_len"])
    return entries


# --- 100% This Season -----------------------------------------------------
# A stricter, SEASON-LONG cousin of Real Streak (user request, 2026-10-26,
# reverse-engineered from a props.cash-style "Props" screen showing e.g.
# "100% 3/3" / "100% 10/10" per player+line): every game played so far
# THIS SEASON clears the line, not just the last RECENT_GAMES=7 window Real
# Streak is capped to. Both an Over and an Under version, since the
# screenshot's own list mixes both directions.
#
# Reads straight from player_gamelog_meta_cache -- the FULL, un-trimmed
# per-game list _fetch_gamelog_values already builds for the CURRENT season
# before get_player_gamelog slices it down to RECENT_GAMES -- so this costs
# zero extra API calls, the same reuse trick get_player_vs_opponent already
# relies on. Only the current season's cache entry is read (never the
# prior-season fallback get_player_gamelog falls back to), since "this
# season only" is the whole point -- a player whose current season hasn't
# started yet just won't appear.
#
# Being a perfect-record check (every value must clear the line) makes this
# ORDER-INDEPENDENT, unlike Real Streak -- it doesn't matter whether
# _extract_game_date's cross-reference fix actually recovered every game's
# real date, since "all N values clear the line" gives the same answer
# regardless of what order those N values are listed in. It can still be
# thrown off by a DIFFERENT, not-yet-ruled-out failure mode though (extra or
# wrong games getting concatenated into the per-game list in the first
# place, from _fetch_gamelog_values' per-category loop) -- the expanded
# [DIAG] category logging added alongside _extract_game_date will show that
# if it's happening.
#
# Lines reuse REAL_STREAK_THRESHOLD (Over) and a parallel, separately tuned
# SEASON_STREAK_UNDER_THRESHOLD (Under) -- both rough judgment calls like
# every other threshold in this file, not fitted to real data yet.
SEASON_STREAK_MIN_GAMES = 3  # "100%, 1/1" after one game isn't a signal worth flagging
SEASON_STREAK_UNDER_THRESHOLD = {"REB": 4.0, "AST": 2.0, "RA": 6.0, "PTS": 10.0, "3PM": 1.0}


def build_season_perfect_entries(predictions):
    entries = []
    for p in predictions:
        league_key = p.get("league", "").lower()
        current_year = _current_season_year(league_key)  # NBA/WNBA differ -- see _current_season_year
        for team_name, props in [(p["home_team"], p.get("home_props") or []),
                                   (p["away_team"], p.get("away_props") or [])]:
            for prop in props:
                for stat_key in ("REB", "AST", "RA", "PTS", "3PM"):
                    dated = player_gamelog_meta_cache.get((league_key, prop["player_id"], stat_key, current_year))
                    if not dated:
                        continue
                    values = [v for _, v, _ in dated]
                    if len(values) < SEASON_STREAK_MIN_GAMES:
                        continue
                    base = {
                        "name": prop["name"], "team": team_name, "league": p.get("league", ""),
                        "match": p["match"], "stat": stat_key, "total": len(values), "values": values,
                        "player_id": prop["player_id"], "game_id": p.get("game_id"), "date": p.get("date", ""),
                    }
                    over_line = REAL_STREAK_THRESHOLD[stat_key]
                    if all(v >= over_line for v in values):
                        entries.append({**base, "direction": "over", "line": over_line})
                    under_line = SEASON_STREAK_UNDER_THRESHOLD[stat_key]
                    if all(v <= under_line for v in values):
                        entries.append({**base, "direction": "under", "line": under_line})
    entries.sort(key=lambda e: -e["total"])
    return entries


SEASON_PERFECT_ENTRY_TEMPLATE = """<div class="season-row" data-match="{match}" data-direction="{direction}" data-stat="{stat}" style="background:#241a14;border-radius:8px;padding:10px 12px;margin:8px 0;display:flex;gap:10px;align-items:flex-start">
  <div style="min-width:56px;text-align:center;background:#0f0a08;border:1px solid #10b981;border-radius:8px;padding:6px 4px;flex-shrink:0">
    <div style="font-size:9px;color:#998">100%</div>
    <div style="font-size:15px;font-weight:bold;color:#10b981">{hits}/{hits}</div>
  </div>
  <div style="flex:1;min-width:0">
    <div style="font-size:10px;color:#998">{league} · {match}</div>
    <div style="font-size:14px;font-weight:bold;margin:1px 0 4px">{name} ({team}) <span style="color:#998;font-weight:normal;font-size:11px">{direction_label} {line} {stat_label}</span></div>
    <div style="font-size:10px;color:#cbd5e1">every game this season: {values_str}</div>
  </div>
</div>"""

SEASON_FILTER_BAR = """<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px;font-size:11px">
  <select id="seasonFixtureFilter" onchange="filterSeasonPerfect()" style="background:#0f0a08;border:1px solid #3a2a20;color:#f59e0b;padding:5px 8px;border-radius:6px">
    <option value="all">All Fixtures</option>
    {fixture_options}
  </select>
  <select id="seasonDirectionFilter" onchange="filterSeasonPerfect()" style="background:#0f0a08;border:1px solid #3a2a20;color:#f59e0b;padding:5px 8px;border-radius:6px">
    <option value="all">Over &amp; Under</option>
    <option value="over">Over only</option>
    <option value="under">Under only</option>
  </select>
  <select id="seasonStatFilter" onchange="filterSeasonPerfect()" style="background:#0f0a08;border:1px solid #3a2a20;color:#f59e0b;padding:5px 8px;border-radius:6px">
    <option value="all">All Bet Types</option>
    {stat_options}
  </select>
</div>
<script>
function filterSeasonPerfect() {{
  var fixture = document.getElementById('seasonFixtureFilter').value;
  var direction = document.getElementById('seasonDirectionFilter').value;
  var stat = document.getElementById('seasonStatFilter').value;
  document.querySelectorAll('.season-row').forEach(function(row) {{
    var show = (fixture === 'all' || row.getAttribute('data-match') === fixture)
      && (direction === 'all' || row.getAttribute('data-direction') === direction)
      && (stat === 'all' || row.getAttribute('data-stat') === stat);
    row.style.display = show ? 'flex' : 'none';
  }});
}}
</script>"""

SEASON_PERFECT_PANEL_TEMPLATE = """<div style="background:#1a1310;border-radius:12px;padding:16px;margin:12px 0;border:1px solid #3a2a20">
  <div style="font-size:14px;font-weight:bold;margin-bottom:10px">💯 100% This Season</div>
  <div style="font-size:11px;color:#998;margin-bottom:10px">
    Every game played so far THIS SEASON has cleared the line shown -- a stricter, longer
    window than Real Streak above (which caps at the last {recent}). Resets each new
    season. A perfect record over a short season is still a short sample -- "3/3" means
    three games, not three months.
  </div>
  {filters}
  <div id="seasonPerfectRows">{cards}</div>
</div>"""


def render_season_perfect_panel(entries):
    if not entries:
        return ""
    seen_fixtures, seen_stats = [], []
    for e in entries:
        if e["match"] not in seen_fixtures:
            seen_fixtures.append(e["match"])
        if e["stat"] not in seen_stats:
            seen_stats.append(e["stat"])
    fixture_options = "".join(f'<option value="{m}">{m}</option>' for m in seen_fixtures)
    stat_options = "".join(f'<option value="{s}">{STAT_LABELS[s]}</option>' for s in seen_stats)
    cards = "".join(
        SEASON_PERFECT_ENTRY_TEMPLATE.format(
            match=e["match"], direction=e["direction"], stat=e["stat"], hits=e["total"],
            league=e["league"], name=e["name"], team=e["team"],
            direction_label="Over" if e["direction"] == "over" else "Under", line=e["line"],
            stat_label=STAT_LABELS[e["stat"]], values_str="/".join(str(v) for v in e["values"]),
        )
        for e in entries
    )
    filters = SEASON_FILTER_BAR.format(fixture_options=fixture_options, stat_options=stat_options)
    return SEASON_PERFECT_PANEL_TEMPLATE.format(recent=RECENT_GAMES, filters=filters, cards=cards)


STREAK_ENTRY_TEMPLATE = """<div class="streak-row" data-match="{match}" data-date="{date_key}" style="background:#241a14;border-radius:8px;padding:10px 12px;margin:8px 0;display:flex;gap:10px;align-items:flex-start">
  <div style="min-width:56px;text-align:center;background:#0f0a08;border:1px solid #3a2a20;border-radius:8px;padding:6px 4px;flex-shrink:0">
    <div style="font-size:9px;color:#998">AVG</div>
    <div style="font-size:17px;font-weight:bold;color:#7dd3a8">{avg}</div>
  </div>
  <div style="flex:1;min-width:0">
    <div style="font-size:10px;color:#998">{league} · {match}</div>
    <div style="font-size:14px;font-weight:bold;margin:1px 0 4px">{name} ({team}) <span style="color:#998;font-weight:normal;font-size:11px">{stat_label}</span></div>
    <div style="font-size:10px;color:#cbd5e1">last {n} (old→new): {values_str}</div>
  </div>
</div>"""

REAL_STREAK_ENTRY_TEMPLATE = """<div class="streak-row" data-match="{match}" data-date="{date_key}" style="background:#241a14;border-radius:8px;padding:10px 12px;margin:8px 0;display:flex;gap:10px;align-items:flex-start">
  <div style="min-width:56px;text-align:center;background:#0f0a08;border:1px solid #f59e0b;border-radius:8px;padding:6px 4px;flex-shrink:0">
    <div style="font-size:9px;color:#998">STREAK</div>
    <div style="font-size:17px;font-weight:bold;color:#f59e0b">{streak_len}{plus}</div>
  </div>
  <div style="flex:1;min-width:0">
    <div style="font-size:10px;color:#998">{league} · {match}</div>
    <div style="font-size:14px;font-weight:bold;margin:1px 0 4px">{name} ({team}) <span style="color:#998;font-weight:normal;font-size:11px">{stat_label}</span></div>
    <div style="font-size:10px;color:#cbd5e1">{streak_len} straight ≥{threshold} (old→new): {values_str}</div>
  </div>
</div>"""

STREAK_PANEL_TEMPLATE = """<div style="background:#1a1310;border-radius:12px;padding:16px;margin:12px 0;border:1px solid #3a2a20">
  <div style="font-size:14px;font-weight:bold;margin-bottom:10px">🔥 Hot Form &amp; Streaks</div>
  <div style="font-size:11px;color:#998;margin-bottom:10px">
    Raw recent-FORM screens, not probabilistic predictions like the props above. Hot Form
    (average) and Real Streak (consecutive, no break) measure genuinely different things --
    a player can appear in one, both, or neither. Cross-check against that player's own prop
    line above before treating either alone as a signal.
  </div>
  {fixture_filter_bar}
  {hot_form_section}
  {real_streak_section}
</div>"""

# Fixture filter bar (user request, 2026-10-22: "the form and streak
# look clamzy at a glance" -- this panel is one flat list across every
# game on the slate, today AND tomorrow, both leagues, sorted by
# avg/streak length rather than grouped by game, so a busy slate turns
# into a wall of cards with no way to jump to one fixture). Same
# client-side button-filter pattern already used on the Streaks Scanner
# page (streaks_scanner.py's tool/date filter) -- no server round trip,
# just toggling data-match on each row.
FIXTURE_FILTER_BAR = """<div style="display:flex;gap:6px;margin-bottom:10px;flex-wrap:wrap">
  <button class="fixture-filter-btn active-filter" data-match="all" onclick="filterStreaksByFixture('all',this)" style="background:#f59e0b;border:1px solid #f59e0b;color:#1a1310;padding:5px 10px;border-radius:999px;font-size:11px;font-weight:700;cursor:pointer">All Fixtures</button>
  {buttons}
</div>
<script>
function filterStreaksByFixture(match, btn) {{
  var rows = document.querySelectorAll('.streak-row');
  rows.forEach(function(row) {{
    row.style.display = (match === 'all' || row.getAttribute('data-match') === match) ? 'flex' : 'none';
  }});
  document.querySelectorAll('.fixture-filter-btn').forEach(function(b) {{
    b.style.background = '#241a14'; b.style.color = '#f59e0b'; b.style.fontWeight = 'normal';
  }});
  btn.style.background = '#f59e0b'; btn.style.color = '#1a1310'; btn.style.fontWeight = '700';
}}
</script>"""


def _fixture_filter_bar(entries):
    # One button per unique fixture appearing in either section, in the
    # order each match first appears (entries are already sorted by
    # avg/streak_len, not match, so this naturally groups same-game
    # buttons together without a separate sort pass).
    seen = []
    for e in entries:
        if e["match"] not in seen:
            seen.append(e["match"])
    if len(seen) <= 1:
        return ""  # nothing to filter with only one (or zero) fixtures on the slate
    buttons = "".join(
        f'''<button class="fixture-filter-btn" data-match="{m}" onclick="filterStreaksByFixture('{m}',this)" '''
        f'''style="background:#241a14;border:1px solid #3a2a20;color:#f59e0b;padding:5px 10px;border-radius:999px;font-size:11px;cursor:pointer">{m}</button>'''
        for m in seen
    )
    return FIXTURE_FILTER_BAR.format(buttons=buttons)


def _render_stat_section(entries, template, icon, label, extra_fields_fn):
    if not entries:
        return ""
    cards = "".join(
        template.format(**extra_fields_fn(e), name=e["name"], team=e["team"], league=e["league"],
                         match=e["match"], date_key=(e.get("date") or "")[:10], stat_label=STAT_LABELS[e["stat"]])
        for e in entries
    )
    return f'<div style="font-size:12px;font-weight:700;color:#e8dcd0;margin:10px 0 4px">{icon} {label}</div>{cards}'


def render_streak_panel(hot_form_entries, real_streak_entries):
    if not hot_form_entries and not real_streak_entries:
        return ""
    hot_form_html = _render_stat_section(
        hot_form_entries, STREAK_ENTRY_TEMPLATE, "📊",
        f"Hot Form (avg over last {HOT_FORM_MIN_GAMES})",
        lambda e: {"avg": e["avg"], "n": HOT_FORM_MIN_GAMES, "values_str": "/".join(str(v) for v in e["values"])},
    )
    real_streak_html = _render_stat_section(
        real_streak_entries, REAL_STREAK_ENTRY_TEMPLATE, "🔥",
        f"Real Streak (≥{REAL_STREAK_MIN_LENGTH}+ CONSECUTIVE games)",
        lambda e: {"streak_len": e["streak_len"], "plus": "+" if e["full_sample"] else "",
                    "threshold": REAL_STREAK_THRESHOLD[e["stat"]],
                    "values_str": "/".join(str(v) for v in e["streak_games"])},
    )
    fixture_filter_bar = _fixture_filter_bar(hot_form_entries + real_streak_entries)
    return STREAK_PANEL_TEMPLATE.format(
        fixture_filter_bar=fixture_filter_bar, hot_form_section=hot_form_html, real_streak_section=real_streak_html,
    )



BUILDER_TEMPLATE = """
<div style="background:#1a1310;border-radius:12px;padding:16px;margin:12px 0;border:1px solid #3a2a20">
  <div style="font-size:14px;font-weight:bold;margin-bottom:10px">🎯 Safest Bet Builder</div>
  <div id="categoryToggles" style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:10px;font-size:12px"></div>
  <div style="display:flex;gap:8px;align-items:center;margin-bottom:6px;flex-wrap:wrap">
    <label style="font-size:12px;color:#c9a">Target odds:</label>
    <input id="targetOdds" type="number" step="0.1" min="1.1" value="5.0"
      style="width:70px;background:#0f0a08;border:1px solid #443;color:white;border-radius:6px;padding:6px 8px;font-size:13px">
    <label style="font-size:12px;color:#c9a">Max legs:</label>
    <input id="maxLegs" type="number" step="1" min="2" value="8"
      style="width:55px;background:#0f0a08;border:1px solid #443;color:white;border-radius:6px;padding:6px 8px;font-size:13px">
    <button onclick="buildSafest()"
      style="background:#c9642a;border:none;color:white;padding:7px 14px;border-radius:6px;font-size:13px;cursor:pointer">
      Build
    </button>
    <button onclick="buildSafest()"
      style="background:#2a201a;border:1px solid #443;color:white;padding:7px 14px;border-radius:6px;font-size:13px;cursor:pointer">
      🔀 Shuffle
    </button>
  </div>
  <div id="builderResult" style="font-size:12px;color:#998">
    Untick any market type you don't want considered, set a target odds and
    leg cap, then tap Build. It rotates through whichever categories are
    ticked, groups near-tied legs and shuffles within each group so it draws
    from more of today's games rather than always the exact same few, and
    caps at 2 legs per game to avoid stacking correlated legs from one
    matchup. Tap Shuffle for a fresh pick among equally-safe options without
    changing your settings.
  </div>
</div>
<script>
const LEGS = {legs_json};

function initCategoryToggles() {{
  const container = document.getElementById('categoryToggles');
  const cats = [...new Set(LEGS.map(l => l.category))];
  container.innerHTML = cats.map(c => `
    <label style="display:flex;align-items:center;gap:4px;color:#ccc;cursor:pointer">
      <input type="checkbox" class="catToggle" value="${{c}}" checked>
      ${{c}}
    </label>
  `).join('');
}}
initCategoryToggles();

function shuffle(arr) {{
  for (let i = arr.length - 1; i > 0; i--) {{
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }}
  return arr;
}}

function tieredShuffle(legs, bandSize) {{
  const bands = {{}};
  legs.forEach(l => {{
    const band = Math.floor(l.prob / bandSize);
    (bands[band] = bands[band] || []).push(l);
  }});
  const bandKeys = Object.keys(bands).map(Number).sort((a, b) => b - a);
  let result = [];
  bandKeys.forEach(b => {{ result = result.concat(shuffle(bands[b])); }});
  return result;
}}

function buildSafest() {{
  const target = parseFloat(document.getElementById('targetOdds').value) || 5.0;
  const maxLegs = parseInt(document.getElementById('maxLegs').value) || 8;
  const activeCats = [...document.querySelectorAll('.catToggle:checked')].map(el => el.value);

  const byCategory = {{}};
  LEGS.filter(l => l.prob > 0 && activeCats.includes(l.category)).forEach(l => {{
    (byCategory[l.category] = byCategory[l.category] || []).push(l);
  }});
  const categories = Object.keys(byCategory);
  categories.forEach(c => {{ byCategory[c] = tieredShuffle(byCategory[c], 5); }});
  const cursor = {{}};
  categories.forEach(c => cursor[c] = 0);

  const chosen = [];
  const matchCount = {{}};
  let combinedOdds = 1;
  let addedThisPass = true;

  while (addedThisPass && combinedOdds < target && chosen.length < maxLegs) {{
    addedThisPass = false;
    for (const cat of categories) {{
      if (combinedOdds >= target || chosen.length >= maxLegs) break;
      const arr = byCategory[cat];
      while (cursor[cat] < arr.length) {{
        const leg = arr[cursor[cat]];
        cursor[cat]++;
        const count = matchCount[leg.match] || 0;
        if (count >= 2) continue;
        chosen.push(leg);
        combinedOdds *= 100 / leg.prob;
        matchCount[leg.match] = count + 1;
        addedThisPass = true;
        break;
      }}
    }}
  }}

  const el = document.getElementById('builderResult');
  if (!chosen.length) {{
    el.innerHTML = 'No legs available to build from.';
    return;
  }}

  // Reverse-engineered from props.cash (user request, 2026-10-25):
  // "Hit Rate -> Fair Odds" (prob%, as a comparable price) and an
  // adjustable custom line with a live-recomputed hit-rate, both now
  // added per leg below -- see recomputeCustomLine()/fairOddsStr().
  window.__CHOSEN_LEGS = chosen;

  const rows = chosen.map((l, idx) => {{
    const fo = fairOddsStr(l.prob);
    const vsOppLine = l.vs_opponent
      ? `<br><span style="color:#7dd3a8;font-size:10px">vs ${{l.vs_opponent.opponent}} (all-time): `
        + `${{l.vs_opponent.hit_rate ? l.vs_opponent.hit_rate.hits + '/' + l.vs_opponent.hit_rate.total : '—'}} `
        + `(${{l.vs_opponent.values.join('/')}})</span>`
      : '';
    return `<div style="padding:5px 0;border-bottom:1px solid #3a2a20">
       <div style="display:flex;justify-content:space-between">
         <span>${{l.match}}<br><span style="color:#ff9a2e">${{l.market}}</span> <span style="color:#665">· ${{l.category}}</span>
         ${{l.detail ? `<br><span style="color:#776;font-size:10px">${{l.detail}}</span>` : ''}}
         ${{l.history ? `<br><span style="color:#665;font-size:10px">last games: ${{l.history}}</span>` : ''}}${{vsOppLine}}</span>
         <span style="text-align:right">
           <span style="color:#ffeb3b;font-weight:bold">${{l.prob}}%</span>
           ${{fo ? `<br><span style="color:#998;font-size:11px">fair ${{fo.decimal}} / ${{fo.american}}</span>` : ''}}
           ${{l.hit_rate ? `<br><span style="color:#998;font-size:11px">${{l.hit_rate.hits}}/${{l.hit_rate.total}}</span>` : ''}}
         </span>
       </div>
       ${{l.history ? `<div style="display:flex;align-items:center;gap:6px;margin-top:4px;font-size:11px;color:#998">
         <label>Try line:</label>
         <input type="number" step="0.5" value="${{l.line}}" id="customLine${{idx}}"
           oninput="recomputeCustomLine(${{idx}})"
           style="width:55px;background:#0f0a08;border:1px solid #443;color:white;border-radius:4px;padding:3px 5px;font-size:11px">
         <span id="customLineResult${{idx}}"></span>
       </div>` : ''}}
     </div>`;
  }}).join('');

  const capNote = chosen.length >= maxLegs && combinedOdds < target
    ? ' (hit the leg cap before reaching target — raise Max legs or lower Target odds)'
    : (combinedOdds < target ? ' (ran out of legs before reaching target)' : '');

  el.innerHTML = `
    <div style="color:white;font-size:13px;margin-bottom:6px">
      ${{chosen.length}} legs · est. combined odds ~<b>${{combinedOdds.toFixed(2)}}</b>${{capNote}}
    </div>
    ${{rows}}
    <div style="color:#776;font-size:10px;margin-top:8px;line-height:1.4">
      Estimate multiplies each leg's fair odds (100/probability) — real
      sportsbook odds include their margin and legs within the same game
      aren't fully independent, so treat this as a ranking tool, not a firm
      price. All three markets use Poisson projections with the line set
      below the model's expectation for a safety margin. "Fair" odds above
      are the zero-margin break-even price for that probability (1/prob) —
      a real sportsbook price will always be a little worse than this.
      "Try line" recomputes an EMPIRICAL hit-rate straight from the actual
      recent games shown, not a re-run of the model.
    </div>
  `;
}}

// Converts a probability percentage (0-100) into a zero-vig "fair"
// price -- same hit-rate-to-odds conversion props.cash's own tool does
// (user request, 2026-10-25). This is a ceiling to compare a real book
// price against, not a quote you'd actually get -- any real sportsbook
// price is worse than this by their margin.
function fairOddsStr(probPct) {{
  if (!probPct || probPct <= 0 || probPct >= 100) return null;
  const decimal = 100 / probPct;
  const american = decimal >= 2
    ? '+' + Math.round((decimal - 1) * 100)
    : String(Math.round(-100 / (decimal - 1)));
  return {{ decimal: decimal.toFixed(2), american }};
}}

// Adjustable custom line (user request, 2026-10-25, reverse-engineering
// props.cash's own "change the line, see the value" filter) -- recomputes
// an empirical hit-rate (and its own fair-odds-from-hit-rate) straight
// from the SAME raw per-game values already shown in "last games", with
// no server round trip and no re-running the Poisson model.
function recomputeCustomLine(idx) {{
  const leg = window.__CHOSEN_LEGS[idx];
  const input = document.getElementById('customLine' + idx);
  const out = document.getElementById('customLineResult' + idx);
  const line = parseFloat(input.value);
  if (isNaN(line) || !leg || !leg.history) {{ out.textContent = ''; return; }}
  const values = leg.history.split('/').map(Number).filter(v => !isNaN(v));
  if (!values.length) {{ out.textContent = 'no game data'; return; }}
  const hits = values.filter(v => v > line).length;
  const pct = Math.round((hits / values.length) * 100);
  const fo = fairOddsStr(pct);
  out.textContent = `${{hits}}/${{values.length}} (${{pct}}%)` + (fo ? ` · fair ${{fo.decimal}} / ${{fo.american}}` : '');
}}
</script>
"""

HTML_TEMPLATE = """<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Orange Line — NBA</title></head>
<body style="background:#0f0a08;color:#e8dcd0;font-family:Arial;padding:12px;max-width:600px;margin:auto">
<h2 style="text-align:center;color:#ff9a2e">🟠 ORANGE LINE — NBA Player Props</h2>
<p style="text-align:center;color:#998;font-size:11px">Recency-weighted, Poisson-projected · Last {recent}g · {generated}</p>
<p style="text-align:center;margin-bottom:16px"><a href="orange_line_predictions.csv" download style="background:#2a201a;border:1px solid #443;color:white;padding:8px 14px;border-radius:8px;text-decoration:none;font-size:13px">⬇ Download CSV</a></p>
<p style="text-align:center;margin-bottom:16px">
  <a href="results/index.html" style="color:#ffeb3b;text-decoration:none;font-size:12px">📊 Results Tracker</a>
</p>
{builder}
{streak_panel}
{season_panel}
{cards}
</body></html>"""

CARD_TEMPLATE = """<div style="background:#1a1310;border-radius:12px;padding:16px;margin:12px 0;border:1px solid #3a2a20">
  <div style="font-size:11px;color:#998;margin-bottom:4px">{date} · <span style="color:#ff9a2e">{league}</span></div>
  <div style="font-size:17px;font-weight:bold;margin-bottom:10px">{match}</div>
  {team_total_html}
  {pace_html}
  {top_players_html}
</div>"""

PACE_ROW = """<div style="font-size:11px;color:#998;margin-bottom:8px">{icon} <span style="color:#e8dcd0">{verdict}</span> — {away_team} {away_label} ({away_pct_str}) · {home_team} {home_label} ({home_pct_str})</div>"""

TEAM_TOTAL_ROW = """<div style="display:flex;justify-content:space-between;text-align:center;background:#0f0a08;border-radius:8px;padding:8px;margin-bottom:8px">
  <div><div style="color:#998;font-size:11px">{away_team}</div><div style="color:#ffeb3b;font-size:18px;font-weight:bold">{exp_away}</div></div>
  <div><div style="color:#998;font-size:11px">TOTAL</div><div style="color:#7ec8ff;font-size:20px;font-weight:bold">{exp_total}</div></div>
  <div><div style="color:#998;font-size:11px">{home_team}</div><div style="color:#ffeb3b;font-size:18px;font-weight:bold">{exp_home}</div></div>
</div>"""

# Top 3 Point Shooters / Top 3 Rebounders / Top 3 Playmakers per team
# (user request, 2026-10-22). Different from the full-roster per-player
# listing dropped on 2026-10-21 -- that was every projected player with
# all 4 stats; this is just the ranked top 3 by pts_proj/reb_proj/ast_proj
# each, a leaner "who to look at" view rather than a full stat dump.
#
# Playmakers added same day, separately from Shooters/Rebounders, after
# the user flagged that their own bet-builder slips kept missing on
# assist legs specifically for scoring wings (Booker, Siakam, Curry --
# all 3+ AST picks chosen "on instinct") while assist picks for actual
# primary ball-handlers (Haliburton, Lillard, Doncic) hit. Ranking by
# ast_proj surfaces who's genuinely producing assists for THIS team
# right now, rather than picking a scorer's name on recognition. The
# position tag (PG/SG/SF/PF/C) is shown alongside each name too, pulled
# from get_roster's defensive position lookup above, as a second,
# independent signal -- not confirmed live for this roster endpoint, so
# it degrades to no tag rather than a wrong one if ESPN's shape differs.
# Reuses the same props already computed for the Safest Bet
# Builder/Hot Form panel, no new API calls.
TOP_PLAYERS_TEAM_BLOCK = """<div style="flex:1;min-width:0">
    <div style="font-size:11px;color:#998;margin-bottom:3px">{team}</div>
    <div style="font-size:10px;color:#ff9a2e;text-transform:uppercase;margin-bottom:1px">Top Shooters</div>
    {pts_rows}
    <div style="font-size:10px;color:#7ec8ff;text-transform:uppercase;margin:4px 0 1px">Top Rebounders</div>
    {reb_rows}
    <div style="font-size:10px;color:#9a7ec8;text-transform:uppercase;margin:4px 0 1px">Top Playmakers</div>
    {ast_rows}
  </div>"""
TOP_PLAYER_ROW = """<div style="display:flex;justify-content:space-between;font-size:12px;padding:1px 0"><span>{name}{pos_tag}{rest_tag}</span><span style="color:#e8dcd0;font-weight:bold">{val}{raw_note}</span></div>"""
TOP_PLAYERS_SECTION = """<div style="display:flex;gap:14px;background:#0f0a08;border-radius:8px;padding:8px 10px;margin-top:4px">{away_block}{home_block}</div>"""


def top_n(props, key, n=3):
    return sorted((p for p in props if p.get(key) is not None), key=lambda p: p[key], reverse=True)[:n]


def _pos_tag(p):
    pos = p.get('position')
    return f""" <span style="color:#776;font-weight:normal;font-size:10px">({pos})</span>""" if pos else ""


def _rest_tag(p):
    # Minutes/rest risk (user request, 2026-10-22): flags a player whose
    # most recent game's minutes dropped well below their own recent
    # average -- blowout garbage time, load management, etc. -- a signal
    # that their point/rebound/assist average may not be a reliable
    # floor tonight even before accounting for pace. Shown wherever the
    # player appears (Shooters/Rebounders/Playmakers alike), since a
    # minutes cut affects all three the same way.
    if not p.get('rest_risk'):
        return ""
    return (f""" <span title="Last game: {p['min_last']} min vs recent avg {p['min_avg']} min" """
            f"""style="color:#f59e0b;font-weight:normal;font-size:10px">⚠️ rest risk</span>""")


def _top_rows(top, val_key):
    rows = []
    for p in top:
        raw_note = ""
        if val_key == 'pts_proj' and p.get('pts_proj_raw') is not None and p['pts_proj_raw'] != p[val_key]:
            # Pace-adjusted projection now feeds the actual number/line
            # (see project_player_props()'s pts_scale docstring) -- the
            # raw season average is kept visible alongside it rather than
            # silently replaced, same honesty standard as every other
            # adjustment in this suite.
            raw_note = f""" <span style="color:#776;font-weight:normal;font-size:10px">(season {p['pts_proj_raw']})</span>"""
        rows.append(TOP_PLAYER_ROW.format(
            name=p['name'], pos_tag=_pos_tag(p), rest_tag=_rest_tag(p), val=p[val_key], raw_note=raw_note,
        ))
    return "".join(rows) or '<div style="font-size:11px;color:#776">No data</div>'


def _top_players_team_block(team, props):
    return TOP_PLAYERS_TEAM_BLOCK.format(
        team=team,
        pts_rows=_top_rows(top_n(props, 'pts_proj'), 'pts_proj'),
        reb_rows=_top_rows(top_n(props, 'reb_proj'), 'reb_proj'),
        ast_rows=_top_rows(top_n(props, 'ast_proj'), 'ast_proj'),
    )


def make_html(predictions):
    # Game cards show Team Total, then Top Shooters/Rebounders -- the
    # full per-player PTS/REB/AST/RA list that used to sit under each
    # card was dropped per user request (2026-10-21: "take off the
    # player stats from the card and keep only the form and streak").
    # The ranked top-3 view below is a separate, later request
    # (2026-10-22) for a leaner "who to look at" summary, not a return
    # to the full listing. Props are still computed once and feed the
    # Safest Bet Builder legs, the Hot Form & Real Streak panel, and now
    # this block too -- no extra API calls.
    cards = "".join(CARD_TEMPLATE.format(
        date=p['date'][:16].replace('T', ' '), match=p['match'], league=p.get('league', ''),
        team_total_html=(
            TEAM_TOTAL_ROW.format(
                away_team=p['away_team'], home_team=p['home_team'],
                exp_away=p['exp_away'], exp_home=p['exp_home'], exp_total=p['exp_total'],
            ) if p.get('exp_total') is not None else
            '<div style="color:#776;font-size:11px;margin-bottom:8px">Team Total unavailable for this game (missing team form)</div>'
        ),
        pace_html=(
            PACE_ROW.format(
                icon=p['pace']['icon'], verdict=p['pace']['verdict'],
                away_team=p['away_team'], away_label=p['pace']['away_label'],
                away_pct_str=f"{p['pace']['away_pct']:+.1f}%",
                home_team=p['home_team'], home_label=p['pace']['home_label'],
                home_pct_str=f"{p['pace']['home_pct']:+.1f}%",
            ) if p.get('pace') else ""
        ),
        top_players_html=(
            TOP_PLAYERS_SECTION.format(
                away_block=_top_players_team_block(p['away_team'], p.get('away_props') or []),
                home_block=_top_players_team_block(p['home_team'], p.get('home_props') or []),
            ) if (p.get('away_props') or p.get('home_props')) else ""
        ),
    ) for p in predictions)
    if not cards:
        cards = '<p style="text-align:center;color:#998">No games today or tomorrow with usable player data.</p>'

    legs = build_legs(predictions)
    builder = BUILDER_TEMPLATE.format(legs_json=json.dumps(legs)) if legs else ""

    hot_form_entries = build_hot_form_entries(predictions)
    real_streak_entries = build_real_streak_entries(predictions)
    streak_panel = render_streak_panel(hot_form_entries, real_streak_entries)
    season_panel = render_season_perfect_panel(build_season_perfect_entries(predictions))

    return HTML_TEMPLATE.format(
        recent=RECENT_GAMES, generated=datetime.now().strftime('%d %b %H:%M'),
        builder=builder, streak_panel=streak_panel, season_panel=season_panel, cards=cards,
    )


def write_csv(predictions, path):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Date', 'League', 'Match', 'HomeTeam', 'AwayTeam', 'ExpHome', 'ExpAway', 'ExpTotal',
                          'Team', 'Player', 'PtsProj', 'RebProj', 'AstProj', 'RaProj', 'SampleSize',
                          'PtsValues', 'RebValues', 'AstValues'])
        for p in predictions:
            for team_label, props in [(p['away_team'], p.get('away_props', [])),
                                       (p['home_team'], p.get('home_props', []))]:
                for prop in props:
                    writer.writerow([
                        p['date'], p.get('league', ''), p['match'], p['home_team'], p['away_team'],
                        p.get('exp_home', ''), p.get('exp_away', ''), p.get('exp_total', ''),
                        team_label, prop['name'],
                        prop['pts_proj'], prop['reb_proj'], prop['ast_proj'], prop['ra_proj'], prop['n_games'],
                        '; '.join(str(v) for v in prop['pts']), '; '.join(str(v) for v in prop['rebs']), '; '.join(str(v) for v in prop['asts']),
                    ])


if __name__ == "__main__":
    predictions = build_predictions()
    os.makedirs('docs/orange-line', exist_ok=True)
    with open('docs/orange-line/index.html', 'w') as f:
        f.write(make_html(predictions))
    write_csv(predictions, 'docs/orange-line/orange_line_predictions.csv')
    with open('docs/orange-line/orange_line.json', 'w') as f:
        json.dump(predictions, f, indent=2, default=str)
    print(f"\nDone — {len(predictions)} games projected.")

    try:
        import orange_line_results_tracker
        legs = build_legs(predictions)
        hot_form = build_hot_form_entries(predictions)
        real_streak = build_real_streak_entries(predictions)
        orange_line_results_tracker.run_results_tracker(legs, hot_form, real_streak)
    except Exception as e:
        print(f"[!] Results tracker failed, but the rest of this run succeeded: {e}")
