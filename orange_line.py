#!/usr/bin/env python3
"""
Orange Line — NBA Player Props Predictor (Rebounds, Assists, RA)
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
from datetime import datetime, timezone
import requests

LEAGUES = [
    {"key": "nba", "name": "NBA"},
    {"key": "wnba", "name": "WNBA"},
]
BASE_FMT = "https://site.api.espn.com/apis/site/v2/sports/basketball/{league}"
GAMELOG_BASE_FMT = "https://site.web.api.espn.com/apis/common/v3/sports/basketball/{league}/athletes/{athlete_id}/gamelog"
REQUEST_DELAY = 1.2  # same conservative pacing as Blitz IQ -- no official rate limit published
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


def get_scoreboard(league_key):
    r = _get(f"{BASE_FMT.format(league=league_key)}/scoreboard")
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
        current_year = datetime.now().year
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
        if athletes and isinstance(athletes[0], dict) and 'items' in athletes[0]:
            for group in athletes:
                for a in group.get('items', []):
                    players.append({'id': a.get('id'), 'name': a.get('displayName', a.get('fullName', '?'))})
        else:
            for a in athletes:
                players.append({'id': a.get('id'), 'name': a.get('displayName', a.get('fullName', '?'))})
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


def _fetch_gamelog_values(league_key, athlete_id, stat_key, season):
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
    (streak feature may be unreliable) instead of silently."""
    r = _get(GAMELOG_BASE_FMT.format(league=league_key, athlete_id=athlete_id),
             params={"season": season})
    dated_values = []  # (date_str_or_None, value)
    if r is None:
        print(f"    [!] gamelog fetch failed for athlete {athlete_id}, season {season} (no response)")
        return dated_values
    try:
        data = r.json()
        top_keys = list(data.keys())
        if top_keys == ['filters']:
            print(f"    [!] gamelog for athlete {athlete_id}, season {season}: still only got 'filters' even with season param.")
        season_data = data.get('seasonTypes', [])
        if not season_data and top_keys != ['filters']:
            print(f"    [!] gamelog for athlete {athlete_id}, season {season}: no 'seasonTypes' key. Top-level keys: {top_keys}")

        root_labels = data.get('labels') or data.get('names') or data.get('displayNames')
        if season_data and not root_labels:
            print(f"    [DIAG] athlete {athlete_id}, season {season}: no root-level labels/names/displayNames. Top-level keys: {top_keys}")

        printed_sample = False
        for st in season_data:
            for cat in st.get('categories', []):
                if not printed_sample:
                    print(f"    [DIAG] athlete {athlete_id}, season {season}: category keys: {list(cat.keys())}, "
                          f"root_labels sample: {str(root_labels)[:200]}")
                    printed_sample = True
                if not root_labels:
                    continue
                for game in cat.get('events', []):
                    stats = game.get('stats', [])
                    if stat_key in root_labels:
                        idx = root_labels.index(stat_key)
                        try:
                            value = float(stats[idx])
                        except (IndexError, ValueError, TypeError):
                            continue
                        date_str = (game.get('gameDate') or game.get('date')
                                    or game.get('eventDate') or None)
                        dated_values.append((date_str, value))
        if season_data and not dated_values:
            print(f"    [!] gamelog for athlete {athlete_id}, season {season}, stat '{stat_key}': no matching values found "
                  f"-- stat_key likely doesn't match ESPN's actual label name. Check the [DIAG] line above for real labels.")
    except Exception as e:
        print(f"  [!] couldn't parse gamelog for athlete {athlete_id}, season {season}: {e}")

    if dated_values and all(d is not None for d, _ in dated_values):
        dated_values.sort(key=lambda pair: pair[0])  # guarantees oldest -> newest
    elif dated_values:
        print(f"    [DIAG] athlete {athlete_id}, season {season}, stat '{stat_key}': some/all games missing a "
              f"date field -- falling back to API order as-is. Real Streak for this player may be unreliable "
              f"until this is confirmed against a real response.")
    return [v for _, v in dated_values]


def get_player_gamelog(league_key, athlete_id, stat_key):
    cache_key = (league_key, athlete_id, stat_key)
    if cache_key in player_gamelog_cache:
        return player_gamelog_cache[cache_key]
    current_year = datetime.now().year
    values = _fetch_gamelog_values(league_key, athlete_id, stat_key, current_year)
    if not values:
        values = _fetch_gamelog_values(league_key, athlete_id, stat_key, current_year - 1)
    values = values[-RECENT_GAMES:]  # oldest-first list -- last N entries ARE the most recent N games
    player_gamelog_cache[cache_key] = values
    return values


PROP_CONFIG = {
    'REB': {'stat': 'REB', 'label': 'Rebounds', 'prior': 5.5},
    'AST': {'stat': 'AST', 'label': 'Assists', 'prior': 3.0},
}


def project_player_props(league_key, team_id):
    roster = get_roster(league_key, team_id)
    if not roster:
        print(f"    no roster found for team {team_id} — 0 player props possible")
    props = []
    for player in roster:
        if not player.get('id'):
            continue
        rebs = get_player_gamelog(league_key, player['id'], PROP_CONFIG['REB']['stat'])
        asts = get_player_gamelog(league_key, player['id'], PROP_CONFIG['AST']['stat'])
        if len(rebs) < 3 or len(asts) < 3 or len(rebs) != len(asts):
            continue  # not enough data, or mismatched game counts between
                        # the two separate stat pulls -- skip rather than
                        # pair up games that may not actually correspond
        n = len(rebs)
        reb_proj = shrink(recency_weighted(rebs), n, PROP_CONFIG['REB']['prior'])
        ast_proj = shrink(recency_weighted(asts), n, PROP_CONFIG['AST']['prior'])
        ra_values = [r + a for r, a in zip(rebs, asts)]
        ra_proj = reb_proj + ast_proj
        props.append({
            'name': player['name'], 'player_id': player['id'],
            'reb_proj': round(reb_proj, 1), 'ast_proj': round(ast_proj, 1), 'ra_proj': round(ra_proj, 1),
            'rebs': rebs, 'asts': asts, 'ra_values': ra_values, 'n_games': n,
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

        print(f"Fetching {league_name}'s today's scoreboard…")
        events = get_scoreboard(league_key)
        print(f"{len(events)} {league_name} games today")

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
            if h_form and a_form:
                team_total_proj = predict_team_total(h_form, a_form, lg_scored, lg_allowed)
            else:
                print(f"    no team form for one/both sides of {away['team']['displayName']} @ "
                      f"{home['team']['displayName']} — Team/Game Total skipped for this game, "
                      f"player props unaffected")

            print(f"  Player props: {away['team']['displayName']} @ {home['team']['displayName']} ({league_name})")
            home_props = project_player_props(league_key, h_id)
            away_props = project_player_props(league_key, a_id)
            predictions.append({
                'date': e.get('date', ''), 'game_id': e.get('id'),
                'league': league_name,
                'match': f"{away['team']['displayName']} @ {home['team']['displayName']}",
                'home_team': home['team']['displayName'], 'away_team': away['team']['displayName'],
                'home_props': home_props, 'away_props': away_props,
                'home_form': h_form, 'away_form': a_form,
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
            home_total = normal_prop(p["exp_home"], DEFAULT_TEAM_STD)
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
            away_total = normal_prop(p["exp_away"], DEFAULT_TEAM_STD)
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
            game_total = normal_prop(p["exp_total"], p["total_std"])
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

        for team_name, props in [(p["home_team"], p.get("home_props") or []),
                                   (p["away_team"], p.get("away_props") or [])]:
            for prop in props:
                for stat_key, values, proj, label in [
                    ("REB", prop["rebs"], prop["reb_proj"], "Rebounds"),
                    ("AST", prop["asts"], prop["ast_proj"], "Assists"),
                    ("RA", prop["ra_values"], prop["ra_proj"], "Rebounds+Assists"),
                ]:
                    result = safe_line(proj)
                    if not result:
                        continue
                    legs.append({
                        "match": f"{match_label} ({league_name})",
                        "market": f"{prop['name']} Over {result['line']} {label}",
                        "prob": result["prob"], "category": label,
                        "hit_rate": hit_rate(values, result["line"]),
                        "detail": f"{team_name} · {league_name} · proj {result['avg']} ({prop['n_games']}gm)",
                        "history": "/".join(str(v) for v in values),
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
HOT_FORM_MIN = {"REB": 8.0, "AST": 5.0, "RA": 13.0}
HOT_FORM_MIN_GAMES = 5
REAL_STREAK_THRESHOLD = {"REB": 6.0, "AST": 4.0, "RA": 10.0}
REAL_STREAK_MIN_LENGTH = 3
STAT_LABELS = {"REB": "Rebounds", "AST": "Assists", "RA": "Rebounds+Assists"}


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
                for stat_key, values in [("REB", prop["rebs"]), ("AST", prop["asts"]), ("RA", prop["ra_values"])]:
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
                for stat_key, values in [("REB", prop["rebs"]), ("AST", prop["asts"]), ("RA", prop["ra_values"])]:
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


STREAK_ENTRY_TEMPLATE = """<div style="background:#241a14;border-radius:8px;padding:10px 12px;margin:8px 0;display:flex;gap:10px;align-items:flex-start">
  <div style="min-width:56px;text-align:center;background:#0f0a08;border:1px solid #3a2a20;border-radius:8px;padding:6px 4px;flex-shrink:0">
    <div style="font-size:9px;color:#998">AVG</div>
    <div style="font-size:17px;font-weight:bold;color:#7dd3a8">{avg}</div>
  </div>
  <div style="flex:1;min-width:0">
    <div style="font-size:10px;color:#998">{league} · {match}</div>
    <div style="font-size:14px;font-weight:bold;margin:1px 0 4px">{name} ({team}) <span style="color:#998;font-weight:normal;font-size:11px">{stat_label}</span></div>
    <div style="font-size:10px;color:#998">last {n} (old→new): {values_str}</div>
  </div>
</div>"""

REAL_STREAK_ENTRY_TEMPLATE = """<div style="background:#241a14;border-radius:8px;padding:10px 12px;margin:8px 0;display:flex;gap:10px;align-items:flex-start">
  <div style="min-width:56px;text-align:center;background:#0f0a08;border:1px solid #f59e0b;border-radius:8px;padding:6px 4px;flex-shrink:0">
    <div style="font-size:9px;color:#998">STREAK</div>
    <div style="font-size:17px;font-weight:bold;color:#f59e0b">{streak_len}{plus}</div>
  </div>
  <div style="flex:1;min-width:0">
    <div style="font-size:10px;color:#998">{league} · {match}</div>
    <div style="font-size:14px;font-weight:bold;margin:1px 0 4px">{name} ({team}) <span style="color:#998;font-weight:normal;font-size:11px">{stat_label}</span></div>
    <div style="font-size:10px;color:#998">{streak_len} straight ≥{threshold} (old→new): {values_str}</div>
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
  {hot_form_section}
  {real_streak_section}
</div>"""


def _render_stat_section(entries, template, icon, label, extra_fields_fn):
    if not entries:
        return ""
    cards = "".join(
        template.format(**extra_fields_fn(e), name=e["name"], team=e["team"], league=e["league"],
                         match=e["match"], stat_label=STAT_LABELS[e["stat"]])
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
    return STREAK_PANEL_TEMPLATE.format(hot_form_section=hot_form_html, real_streak_section=real_streak_html)



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

  const rows = chosen.map(l =>
    `<div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #3a2a20">
       <span>${{l.match}}<br><span style="color:#ff9a2e">${{l.market}}</span> <span style="color:#665">· ${{l.category}}</span>
       ${{l.detail ? `<br><span style="color:#776;font-size:10px">${{l.detail}}</span>` : ''}}
       ${{l.history ? `<br><span style="color:#665;font-size:10px">last games: ${{l.history}}</span>` : ''}}</span>
       <span style="text-align:right"><span style="color:#ffeb3b;font-weight:bold">${{l.prob}}%</span>${{l.hit_rate ? `<br><span style="color:#998;font-size:11px">${{l.hit_rate.hits}}/${{l.hit_rate.total}}</span>` : ''}}</span>
     </div>`
  ).join('');

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
      below the model's expectation for a safety margin.
    </div>
  `;
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
{cards}
</body></html>"""

CARD_TEMPLATE = """<div style="background:#1a1310;border-radius:12px;padding:16px;margin:12px 0;border:1px solid #3a2a20">
  <div style="font-size:11px;color:#998;margin-bottom:4px">{date} · <span style="color:#ff9a2e">{league}</span></div>
  <div style="font-size:17px;font-weight:bold;margin-bottom:10px">{match}</div>
  {team_total_html}
  {player_props_html}
</div>"""

TEAM_TOTAL_ROW = """<div style="display:flex;justify-content:space-between;text-align:center;background:#0f0a08;border-radius:8px;padding:8px;margin-bottom:8px">
  <div><div style="color:#998;font-size:11px">{away_team}</div><div style="color:#ffeb3b;font-size:18px;font-weight:bold">{exp_away}</div></div>
  <div><div style="color:#998;font-size:11px">TOTAL</div><div style="color:#7ec8ff;font-size:20px;font-weight:bold">{exp_total}</div></div>
  <div><div style="color:#998;font-size:11px">{home_team}</div><div style="color:#ffeb3b;font-size:18px;font-weight:bold">{exp_home}</div></div>
</div>"""

PLAYER_PROP_ROW = """<div style="display:flex;justify-content:space-between;font-size:11px;padding:5px 0;border-top:1px solid #3a2a20">
  <div>{name}</div>
  <div style="color:#ff9a2e;font-weight:bold">REB {reb_proj} · AST {ast_proj} · RA {ra_proj} <span style="color:#776;font-weight:normal">({n_games}gm)</span></div>
</div>"""


def player_props_section(team_label, props):
    if not props:
        return ""
    rows = "".join(PLAYER_PROP_ROW.format(**p) for p in props)
    return f'<div style="margin-top:8px"><div style="color:#998;font-size:10px;text-transform:uppercase;margin-bottom:2px">{team_label}</div>{rows}</div>'


def make_html(predictions):
    cards = "".join(CARD_TEMPLATE.format(
        date=p['date'][:16].replace('T', ' '), match=p['match'], league=p.get('league', ''),
        team_total_html=(
            TEAM_TOTAL_ROW.format(
                away_team=p['away_team'], home_team=p['home_team'],
                exp_away=p['exp_away'], exp_home=p['exp_home'], exp_total=p['exp_total'],
            ) if p.get('exp_total') is not None else
            '<div style="color:#776;font-size:11px;margin-bottom:8px">Team Total unavailable for this game (missing team form)</div>'
        ),
        player_props_html=(
            player_props_section(p['away_team'], p.get('away_props', []))
            + player_props_section(p['home_team'], p.get('home_props', []))
        ),
    ) for p in predictions)
    if not cards:
        cards = '<p style="text-align:center;color:#998">No games today with usable player data.</p>'

    legs = build_legs(predictions)
    builder = BUILDER_TEMPLATE.format(legs_json=json.dumps(legs)) if legs else ""

    hot_form_entries = build_hot_form_entries(predictions)
    real_streak_entries = build_real_streak_entries(predictions)
    streak_panel = render_streak_panel(hot_form_entries, real_streak_entries)

    return HTML_TEMPLATE.format(
        recent=RECENT_GAMES, generated=datetime.now().strftime('%d %b %H:%M'),
        builder=builder, streak_panel=streak_panel, cards=cards,
    )


def write_csv(predictions, path):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Date', 'League', 'Match', 'HomeTeam', 'AwayTeam', 'ExpHome', 'ExpAway', 'ExpTotal',
                          'Team', 'Player', 'RebProj', 'AstProj', 'RaProj', 'SampleSize',
                          'RebValues', 'AstValues'])
        for p in predictions:
            for team_label, props in [(p['away_team'], p.get('away_props', [])),
                                       (p['home_team'], p.get('home_props', []))]:
                for prop in props:
                    writer.writerow([
                        p['date'], p.get('league', ''), p['match'], p['home_team'], p['away_team'],
                        p.get('exp_home', ''), p.get('exp_away', ''), p.get('exp_total', ''),
                        team_label, prop['name'],
                        prop['reb_proj'], prop['ast_proj'], prop['ra_proj'], prop['n_games'],
                        '; '.join(str(v) for v in prop['rebs']), '; '.join(str(v) for v in prop['asts']),
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
