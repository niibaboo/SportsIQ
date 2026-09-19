"""
Player Stat Model
------------------
Player prop predictor covering shots / shots on target, cards, and
goals + assists. Sibling project to Corner Flag (football-data.org),
Blue Line (NHL) and Match IQ (TheStatsAPI match-level model).

Pulls per-player data from TheStatsAPI (api.thestatsapi.com):
  - season profile stats (appearances, goals, assists, minutes)
  - per-match player-stats rows for the player's team's last N
    finished matches (shots, shots on target, cards, minutes)

Blends a rolling "recent form" average (last 5-10 matches) with the
full-season average into a single projected rate per 90 minutes, then
uses a Poisson model to price over/under and yes/no prop lines.

On top of single-player lookup, the model can also pick its own pool
of players to check, two ways:
  - Watchlist scan: a fixed list of teams you set below (WATCHLIST).
  - Competition scan: every team with a fixture in a competition over
    the next N days (opt-in, since it burns a lot more API quota).

Either scan pulls the full squad for each team, drops fringe players
by expected minutes, builds a report for everyone left, then flags
players against your CRITERIA_THRESHOLDS and separately surfaces the
top-N highest-probability plays per market.

Usage:
    python3 player_stat_model.py

Set THESTATSAPI_KEY as an environment variable, or paste it into
API_KEY below. Results are appended to player_stats_data.json, which
the companion player_stat_model.html web app reads for live use.
"""

import os
import sys
import json
import time
import math
import shutil
import hashlib
from datetime import date, timedelta
from pathlib import Path

import requests

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

API_KEY = os.environ.get("THESTATSAPI_KEY", "PASTE_YOUR_KEY_HERE")
BASE_URL = "https://api.thestatsapi.com/api/football"

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

OUTPUT_JSON = Path(__file__).parent / "player_stats_data.json"

# How much weight to give recent form vs full-season average.
# 0.6 means: 60% recent form, 40% season baseline.
ROLLING_WEIGHT = 0.6
ROLLING_MATCHES = 10  # pull up to this many recent finished matches

# Below this many total minutes this season, a per-90 extrapolation is
# small-sample noise, not a real rate — e.g. "3 shots in 15 minutes
# played" extrapolates to an absurd "18 shots per 90", not a signal.
# Treat season stats as unusable below this floor rather than silently
# producing a confident-looking wrong number.
MIN_SEASON_MINUTES = 180  # roughly two full matches' worth

# Teams to scan by default (team names, resolved via search_team()).
# Edit this to your actual watchlist.
WATCHLIST = [
    "Arsenal",
    "Manchester City",
]

# Skip players who don't average at least this many minutes per
# appearance this season — filters out fringe/bench players so the
# scan doesn't waste quota (or your attention) on non-starters.
MIN_AVG_MINUTES = 60

# A player "meets criteria" if any of their prop probabilities clears
# the threshold below. Edit freely.
CRITERIA_THRESHOLDS = {
    "shots_over_1.5": 0.65,
    "sot_over_0.5": 0.60,
    "to_be_carded": 0.30,
    "goal_or_assist": 0.55,
}

# A second, independent way to flag players — arbitrary lines (not the
# fixed 1.5/2.5-style props above), checked with the same Poisson math
# used everywhere else. A player qualifies if their probability of
# clearing ANY ONE of these lines is >= CUSTOM_CRITERIA_MIN_PROB (an
# "any" match, same logic as CRITERIA_THRESHOLDS above — not an "all
# four at once" requirement).
CUSTOM_CRITERIA = {
    "shots": 2.0,
    "shots_on_target": 1.0,
    "fouls": 2.0,
    "tackles": 2.0,
}
CUSTOM_CRITERIA_MIN_PROB = 0.70

# Regardless of thresholds, also surface the top N plays per market
# from whatever pool was scanned.
TOP_N_PER_MARKET = 5

HEADERS = {"Authorization": f"Bearer {API_KEY}"}


# ----------------------------------------------------------------------
# Simple on-disk cache so re-runs don't burn API quota
# ----------------------------------------------------------------------

def _cache_path(key: str) -> Path:
    h = hashlib.sha256(key.encode()).hexdigest()[:20]
    return CACHE_DIR / f"{h}.json"


def cached_get(path: str, params: dict | None = None, ttl_hours: int = 12) -> dict:
    """GET from TheStatsAPI with a simple file cache to control quota use."""
    cache_key = path + json.dumps(params or {}, sort_keys=True)
    cfile = _cache_path(cache_key)

    if cfile.exists():
        age_hours = (time.time() - cfile.stat().st_mtime) / 3600
        if age_hours < ttl_hours:
            return json.loads(cfile.read_text())

    resp = requests.get(f"{BASE_URL}{path}", headers=HEADERS, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    cfile.write_text(json.dumps(data))
    time.sleep(0.3)  # be polite to the free-tier rate limit
    return data


# ----------------------------------------------------------------------
# TheStatsAPI calls
# ----------------------------------------------------------------------

def search_player(name: str) -> dict:
    """Search for a player by name, return the first match.

    CORRECTED against TheStatsAPI's own published tutorial
    (thestatsapi.com/blog/how-to-get-football-player-stats-api): the
    endpoint is GET /football/players?search=... — NOT /players/search.
    The earlier guess had both the path AND the param wrong; "search" as
    a param name was actually right, but appending /search to the path
    was not. Response player dicts: id, name, current_team (dict with
    at least a name — the docs don't show a team id field explicitly,
    only name, so this is still not fully confirmed), nationality,
    position."""
    data = cached_get("/players", {"search": name}, ttl_hours=24 * 7)
    results = data.get("data", [])
    if not results:
        raise ValueError(f"No player found for '{name}'")
    return results[0]


def get_player_stats(player_id: str, season_id: str) -> dict:
    """Season stats for a player: GET /players/{player_id}/stats.

    CONFIRMED against the official spec (api.thestatsapi.com/llms.txt):
    season_id is REQUIRED on this endpoint (not optional, not defaulted)
    — this was the exact cause of every 400 Bad Request seen against
    real player IDs. The nested scoring/shooting/passing/defending/
    duels/discipline shape, and top-level team_id/appearances/
    minutes_played/position/rating, are all confirmed correct."""
    data = cached_get(f"/players/{player_id}/stats", {"season_id": season_id}, ttl_hours=24)
    return data.get("data", {})


def get_team_details(team_id: str) -> dict:
    """Team profile including primary_competition — GET /teams/{team_id}."""
    data = cached_get(f"/teams/{team_id}", ttl_hours=24 * 7)
    return data.get("data", {})


def get_competition_current_season(competition_id: str) -> str | None:
    """Current season_id for a competition — GET /competitions/{id}."""
    data = cached_get(f"/competitions/{competition_id}", ttl_hours=24)
    return data.get("data", {}).get("current_season_id")


def get_team_standings_competitions(team_id: str) -> list[dict]:
    """Competitions where a team has standings history: GET
    /teams/{team_id}/standings. Newest-season-first per competition,
    per the confirmed spec."""
    data = cached_get(f"/teams/{team_id}/standings", ttl_hours=24 * 7)
    return data.get("data", [])


def find_primary_league_competition(team_id: str) -> dict | None:
    """Find the team's actual domestic-league competition, preferring
    type == "league" over cups/tournaments.

    OBSERVED LIVE: team.primary_competition is NOT reliably a club's
    domestic league — for Barcelona it resolved to "UEFA Champions
    League" (a tournament) instead of La Liga, which in turn resolved
    a Champions League season with far fewer matches played than the
    real league season, producing a misleadingly thin "90 season
    minutes" for a player who's actually played several league games.
    This checks the team's standings history instead (only real
    tables — mostly leagues, though group-stage tournaments can also
    have standings) and picks the first entry whose competition type
    is confirmed "league". Exactly why primary_competition picks what
    it picks isn't documented, so this is a workaround for an observed
    behavior, not a confirmed root cause."""
    for entry in get_team_standings_competitions(team_id):
        comp_id = entry.get("competition", {}).get("id")
        if not comp_id:
            continue
        comp_detail = cached_get(f"/competitions/{comp_id}", ttl_hours=24 * 7).get("data", {})
        if comp_detail.get("type") == "league":
            return comp_detail
    return None


def resolve_team_and_league(team_id: str) -> dict:
    """team_id -> {team_name, league_name, league_id, season_id}. Single
    source of truth for the team->league->season resolution chain —
    used by both build_player_report (single lookup) and scan_team
    (squad scans), so any team/league name shown in a report comes
    from the exact same resolution find_primary_league_competition()
    already does for season_id, not a second, possibly-inconsistent
    lookup. Values are None where nothing resolves (see
    find_primary_league_competition's docstring for why that can
    legitimately happen)."""
    team = get_team_details(team_id)
    league_comp = find_primary_league_competition(team_id)
    if league_comp:
        league_name = league_comp.get("name")
        league_id = league_comp.get("id")
        season_id = league_comp.get("current_season_id") or get_competition_current_season(league_id)
    else:
        fallback = team.get("primary_competition", {})
        league_id = fallback.get("id")
        league_name = fallback.get("name")
        season_id = get_competition_current_season(league_id) if league_id else None
    return {
        "team_name": team.get("name"),
        "league_name": league_name,
        "league_id": league_id,
        "season_id": season_id,
    }


def resolve_season_id(team_id: str) -> str | None:
    """team_id -> (league competition preferred) -> current_season_id.
    Thin wrapper around resolve_team_and_league() for callers that only
    need the season_id, not the names too."""
    return resolve_team_and_league(team_id)["season_id"]


def get_team_recent_matches(team_id: str, limit: int = ROLLING_MATCHES) -> list[dict]:
    """Most recent finished matches for a team, newest first.

    CORRECTED against the confirmed spec: there is no dedicated
    /teams/{team_id}/matches route — the Teams section only lists
    /players, /injuries-suspensions, /stats, /standings, and the team
    itself. Team matches come through the general matches endpoint,
    filtered by team_id (a real, confirmed query param on
    /football/matches). Also dropped the "order" param — it isn't in
    the confirmed query-parameters table for this endpoint either, so
    results aren't guaranteed chronologically ordered by the API.
    Fetches a larger buffer (per_page capped at the API's max of 100)
    and sorts by utc_date client-side instead, the same defensive
    pattern Match IQ's own get_team_form already uses."""
    data = cached_get(
        "/matches",
        {"team_id": team_id, "status": "finished", "per_page": 100},
        ttl_hours=6,
    )
    matches = data.get("data", [])
    print(f"  [debug] /matches?team_id={team_id}&status=finished returned {len(matches)} raw results")
    matches.sort(key=lambda m: m.get("utc_date", ""), reverse=True)
    return matches[:limit]


def get_match_player_stats(match_id: str, player_id: str) -> dict | None:
    """This player's row from a single match's player-stats endpoint."""
    data = cached_get(f"/matches/{match_id}/player-stats", ttl_hours=24 * 30)
    rows = data.get("data", data.get("player_stats", []))
    for row in rows:
        if str(row.get("player_id")) == str(player_id):
            return row
    return None


def find_competition(name: str) -> dict:
    """Search for a competition by name — GET /football/competitions.
    CONFIRMED against the official spec (api.thestatsapi.com/llms.txt):
    a real, documented `search` query param on this endpoint.

    CORRECTED: was blindly taking results[0], which matched "Canadian
    Premier League" for a search of "Premier League" — a substring
    match ranked first, not the intended English competition. Match IQ
    (match_iq.py) already solved this exact problem with an
    exact-name-match preference before falling back to the first
    result; mirrored here rather than re-inventing it."""
    data = cached_get("/competitions", {"search": name}, ttl_hours=24 * 30)
    results = data.get("data", [])
    if not results:
        raise ValueError(f"No competition found for '{name}'")
    for comp in results:
        if comp.get("name", "").lower() == name.lower():
            return comp
    return results[0]


def search_team(name: str) -> dict:
    """Search for a team by name, return the first match.

    UNVERIFIED — the tutorial that confirmed the player-search shape
    doesn't cover team search directly. Extrapolated from the same
    /{resource}?search=... pattern as players, since that pattern held
    for players despite the original /resource/search guess being
    wrong. Treat this one with the same caution until confirmed live."""
    data = cached_get("/teams", {"search": name}, ttl_hours=24 * 30)
    results = data.get("data", [])
    if not results:
        raise ValueError(f"No team found for '{name}'")
    return results[0]


def get_team_squad(team_id: str) -> list[dict]:
    """Full current squad for a team: GET /teams/{team_id}/players.

    CONFIRMED against the official spec (api.thestatsapi.com/llms.txt)
    as a real, dedicated endpoint — "Returns all players assigned to
    the club... up to 100, with extended profile fields." This is
    better than the /players?team_id=... workaround from the blog
    tutorial (which was written as an example for a different use
    case, not the intended route) — richer fields, and confirmed
    correct rather than inferred."""
    data = cached_get(f"/teams/{team_id}/players", ttl_hours=24 * 7)
    return data.get("data", [])


def get_upcoming_matches(competition_id: str, days_ahead: int = 7) -> list[dict]:
    """Scheduled matches for a competition over the next N days."""
    today = date.today()
    data = cached_get(
        "/matches",
        {
            "competition_id": competition_id,
            "date_from": today.isoformat(),
            "date_to": (today + timedelta(days=days_ahead)).isoformat(),
            "status": "scheduled",
            "per_page": 100,
        },
        ttl_hours=6,
    )
    return data.get("data", data.get("matches", []))


# ----------------------------------------------------------------------
# Stat extraction / blending
# ----------------------------------------------------------------------

STAT_FIELDS = {
    # CORRECTED against the confirmed /matches/{id}/player-stats shape
    # (api.thestatsapi.com/llms.txt): "shots" was reading shooting.shots,
    # but the real field is shooting.total_shots. Cards were read from a
    # "discipline" section that doesn't exist on THIS endpoint — match-
    # level player-stats puts yellow_cards/red_cards under "general",
    # not "discipline" (discipline is only used by the season-stats
    # endpoint, a different shape entirely — see season_baseline below).
    "shots": lambda row: row.get("shooting", {}).get("total_shots", 0),
    "shots_on_target": lambda row: row.get("shooting", {}).get("shots_on_target", 0),
    "cards": lambda row: (row.get("general", {}).get("yellow_cards", 0)
                           + row.get("general", {}).get("red_cards", 0)),
    "goals": lambda row: row.get("shooting", {}).get("goals", 0),
    "assists": lambda row: row.get("passing", {}).get("assists", 0),
    "tackles": lambda row: row.get("defending", {}).get("tackles", 0),
    # Match-level only — the season-stats endpoint's confirmed response
    # shape has no fouls field anywhere (defending only has tackles and
    # interceptions there). season_baseline() below marks this stat's
    # season per90 as unavailable (None) rather than fabricating a 0,
    # and blend() falls back to rolling-only for it — see both for why.
    "fouls": lambda row: row.get("general", {}).get("fouls", 0),
}

# Stats with no season-endpoint source at all (see "fouls" above) — a
# fabricated 0 baseline would silently drag the blended rate down by
# ~40% (season's blend weight) versus what recent matches actually show,
# every time, not just for thin samples. Tracked explicitly so
# season_baseline/blend can skip blending for these instead.
SEASON_UNAVAILABLE_STATS = {"fouls"}


def per90(total: float, minutes: float) -> float:
    if minutes <= 0:
        return 0.0
    return total * 90.0 / minutes


def rolling_form(player_id: str, matches: list[dict]) -> dict:
    """Average per-90 rate across the player's last N finished matches.
    
    NOTE: matches are passed in, not fetched — prevents redundant API calls
    when scanning an entire squad."""
    totals = {k: 0.0 for k in STAT_FIELDS}
    game_logs = {k: [] for k in STAT_FIELDS}  # per-match values, newest-first
    minutes_played = 0.0
    matches_used = 0

    for match in matches:
        row = get_match_player_stats(match["id"], player_id)
        if not row:
            continue
        # CORRECTED: was row.get("minutes", 0) — the confirmed field name
        # per the official spec is "minutes_played", not "minutes". This
        # was the actual cause of "0 matches used for form" showing up
        # consistently across every player tested: the wrong key always
        # returned the 0 default, so every match failed the `mins <= 0`
        # check and got skipped — even when get_match_player_stats found
        # a real row.
        mins = row.get("minutes_played", 0)
        if mins <= 0:
            continue
        minutes_played += mins
        matches_used += 1
        for stat, extractor in STAT_FIELDS.items():
            value = extractor(row)
            totals[stat] += value
            game_logs[stat].append(value)

    return {
        "matches_used": matches_used,
        "minutes_played": minutes_played,
        "per90": {stat: per90(totals[stat], minutes_played) for stat in STAT_FIELDS},
        # get_team_recent_matches returns newest-first, and this loop
        # preserves that order — reverse for display so a printed
        # sequence reads oldest-to-newest, left-to-right, matching the
        # convention Match IQ already uses for its own "last games" line.
        "game_log": {stat: list(reversed(values)) for stat, values in game_logs.items()},
    }


def season_baseline(stats: dict) -> dict:
    """Average per-90 rate from the player's season stats.

    CORRECTED to match the real nested response shape confirmed in
    TheStatsAPI's tutorial: minutes_played and appearances sit at the
    top level; goals/assists are under "scoring"; shots/shots_on_target
    under "shooting"; yellow_cards/red_cards under "discipline" — not
    a flat season_stats/stats dict as originally guessed.

    GUARDED against small-sample blowup: below MIN_SEASON_MINUTES, a
    per-90 rate is division-by-a-tiny-number, not a real signal (seen
    live: a player with a handful of season minutes projected at 99%+
    shots probabilities purely from that distortion). Below the floor,
    every per90 rate returns 0 rather than an inflated number — the
    caller's blend() already leans on rolling form when season data is
    this thin, and if rolling form is ALSO empty (0 matches — check
    that separately, it's a different problem, e.g. a team fetch
    issue), the report should read as "not enough data" rather than
    quietly presenting noise as a strong signal."""
    minutes = stats.get("minutes_played", 0)
    scoring = stats.get("scoring", {})
    shooting = stats.get("shooting", {})
    discipline = stats.get("discipline", {})
    defending = stats.get("defending", {})
    totals = {
        "shots": shooting.get("total_shots", 0),
        "shots_on_target": shooting.get("shots_on_target", 0),
        "cards": discipline.get("yellow_cards", 0) + discipline.get("red_cards", 0),
        "goals": scoring.get("goals", 0),
        "assists": scoring.get("assists", 0),
        "tackles": defending.get("tackles", 0),
    }
    if minutes < MIN_SEASON_MINUTES:
        return {"minutes": minutes, "per90": {stat: 0.0 for stat in STAT_FIELDS}}
    per90_rates = {stat: per90(totals[stat], minutes) for stat in totals}
    # "fouls" has no season-endpoint source at all — mark it unavailable
    # (None) rather than defaulting to 0, so blend() knows to skip
    # blending and use rolling form alone instead of silently diluting
    # a real recent-form rate with a fabricated season "0".
    for stat in SEASON_UNAVAILABLE_STATS:
        per90_rates[stat] = None
    return {
        "minutes": minutes,
        "per90": per90_rates,
    }


def blend(rolling: dict, season: dict, weight: float = ROLLING_WEIGHT) -> dict:
    """Weighted blend of rolling form and season baseline, per stat."""
    blended = {}
    for stat in STAT_FIELDS:
        r = rolling["per90"].get(stat, 0.0)
        s = season["per90"].get(stat)
        if s is None:
            # No season-endpoint source for this stat (e.g. "fouls") —
            # blending against a fabricated 0 would just drag the real
            # rolling-form rate down by the season weight, every time.
            # Use rolling form alone; if there's no rolling data either,
            # this correctly comes out to 0 same as everything else.
            blended[stat] = r
            continue
        # If we don't have enough recent-match data, lean on season instead.
        w = weight if rolling["matches_used"] >= 3 else 0.25
        blended[stat] = w * r + (1 - w) * s
    return blended


# ----------------------------------------------------------------------
# Poisson pricing for prop lines
# ----------------------------------------------------------------------

def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * lam ** k / math.factorial(k)


def prob_over(lam: float, line: float) -> float:
    """P(stat > line), e.g. shots over 2.5."""
    threshold = math.floor(line) + 1
    return 1 - sum(poisson_pmf(k, lam) for k in range(threshold))


def prob_at_least_one(lam: float) -> float:
    """P(stat >= 1), e.g. 'to be carded', 'to score or assist'."""
    return 1 - poisson_pmf(0, lam)


# ----------------------------------------------------------------------
# Building a player report
# ----------------------------------------------------------------------

def estimate_expected_minutes(stats: dict) -> float:
    """Average minutes per appearance this season — a starter proxy.

    CORRECTED to the confirmed top-level field names: appearances and
    minutes_played sit directly on the stats object, not nested under
    a season_stats/stats wrapper."""
    apps = stats.get("appearances", 0)
    minutes = stats.get("minutes_played", 0)
    if apps <= 0:
        return 0.0
    return minutes / apps


# prop key -> (game_log stat key, line) — the SAME line each prop
# prices with Poisson, used here instead to compute a plain empirical
# hit-rate ("X of the last Y games actually cleared this") straight
# from the raw per-game values. A model-free cross-check, not a second
# projection — if the Poisson % and the hit-rate disagree a lot, that's
# worth noticing, not something to silently reconcile.
PROP_LINE_MAP = {
    "shots_over_1.5": ("shots", 1.5),
    "shots_over_2.5": ("shots", 2.5),
    "sot_over_0.5": ("shots_on_target", 0.5),
    "sot_over_1.5": ("shots_on_target", 1.5),
    "to_be_carded": ("cards", 0.5),
    "to_score": ("goals", 0.5),
    "to_assist": ("assists", 0.5),
    "goal_or_assist": ("__goal_or_assist__", 0.5),  # combined goals+assists per game
    "tackles_over_1.5": ("tackles", 1.5),
    "tackles_over_2.5": ("tackles", 2.5),
    "fouls_over_1.5": ("fouls", 1.5),
    "fouls_over_2.5": ("fouls", 2.5),
}


def hit_rate(values: list, line: float) -> dict | None:
    """(hits, total) as a dict, e.g. {"hits": 5, "total": 7} = 5 of the
    last 7 games cleared this line. None if there's no game data to
    count — mirrors the same rolling_matches_used > 0 condition the
    Poisson probabilities themselves depend on."""
    if not values:
        return None
    hits = sum(1 for v in values if v > line)
    return {"hits": hits, "total": len(values)}


def compute_hit_rates(game_log: dict) -> dict:
    """Every prop's empirical hit-rate, computed from the exact same
    per-game values feeding the Poisson model — same PROP_LINE_MAP used
    by build_legs() too, so a leg's hit-rate and a report's hit-rate for
    the same prop always agree."""
    rates = {}
    for prop_key, (stat_key, line) in PROP_LINE_MAP.items():
        if stat_key == "__goal_or_assist__":
            goals, assists = game_log.get("goals", []), game_log.get("assists", [])
            values = [g + a for g, a in zip(goals, assists)] if (goals and assists and len(goals) == len(assists)) else []
        else:
            values = game_log.get(stat_key, [])
        rates[prop_key] = hit_rate(values, line)
    return rates


def build_report(player: dict, stats: dict, team_id: str, expected_minutes: float,
                  team_name: str | None = None, league_name: str | None = None,
                  matches: list[dict] | None = None) -> dict:
    player_id = player["id"]
    if matches is None:
        matches = get_team_recent_matches(team_id)
    rolling = rolling_form(player_id, matches)
    season = season_baseline(stats)
    blended_per90 = blend(rolling, season)

    # Scale each per-90 rate to the minutes we expect this player on the pitch
    projected = {stat: rate * expected_minutes / 90.0 for stat, rate in blended_per90.items()}

    hit_rates = compute_hit_rates(rolling["game_log"]) if rolling["matches_used"] > 0 else {k: None for k in PROP_LINE_MAP}

    return {
        "player_id": player_id,
        "player_name": player.get("name", "Unknown"),
        "team_id": team_id,
        "team_name": team_name or "Unknown team",
        "league_name": league_name or "Unknown league",
        "expected_minutes": round(expected_minutes, 1),
        "rolling_matches_used": rolling["matches_used"],
        "game_log": rolling["game_log"],
        "hit_rates": hit_rates,
        "season_minutes_played": season["minutes"],
        "low_data": rolling["matches_used"] == 0 and season["minutes"] < MIN_SEASON_MINUTES,
        "projected_per_match": projected,
        "props": {
            "shots_over_1.5": round(prob_over(projected["shots"], 1.5), 3),
            "shots_over_2.5": round(prob_over(projected["shots"], 2.5), 3),
            "sot_over_0.5": round(prob_over(projected["shots_on_target"], 0.5), 3),
            "sot_over_1.5": round(prob_over(projected["shots_on_target"], 1.5), 3),
            "to_be_carded": round(prob_at_least_one(projected["cards"]), 3),
            "to_score": round(prob_at_least_one(projected["goals"]), 3),
            "to_assist": round(prob_at_least_one(projected["assists"]), 3),
            "goal_or_assist": round(
                prob_at_least_one(projected["goals"] + projected["assists"]), 3
            ),
            "tackles_over_1.5": round(prob_over(projected["tackles"], 1.5), 3),
            "tackles_over_2.5": round(prob_over(projected["tackles"], 2.5), 3),
            "fouls_over_1.5": round(prob_over(projected["fouls"], 1.5), 3),
            "fouls_over_2.5": round(prob_over(projected["fouls"], 2.5), 3),
        },
    }


def build_player_report(name: str, expected_minutes: float = 90.0) -> dict:
    """Single-player lookup by name (the original manual flow)."""
    player = search_player(name)
    # CONFIRMED against the official spec: current_team.id is on the
    # search result itself, and season_id is REQUIRED to call stats —
    # so team_id has to come from here, BEFORE the stats call, to
    # resolve season_id through team -> competition -> current season.
    team_id = player.get("current_team", {}).get("id")
    if not team_id:
        raise ValueError(f"'{player.get('name', name)}' has no current_team — can't resolve a season_id for stats")

    # DIAGNOSTIC: print exactly what the resolution chain picks, so a
    # suspiciously-thin result (e.g. a known heavy-minutes player
    # showing ~90 season minutes) can be checked against what's
    # actually resolved rather than guessed at blind.
    resolved = resolve_team_and_league(team_id)
    print(f"  [debug] team={resolved['team_name']!r} ({team_id})  "
          f"competition={resolved['league_name']!r} ({resolved['league_id']})")
    if not resolved["league_id"]:
        raise ValueError(f"Team {team_id} has no resolvable competition — can't get a season_id")
    print(f"  [debug] resolved season_id={resolved['season_id']!r}")
    if not resolved["season_id"]:
        raise ValueError(f"Couldn't resolve a current season_id for competition {resolved['league_id']}")

    stats = get_player_stats(player["id"], resolved["season_id"])
    return build_report(player, stats, team_id, expected_minutes,
                         team_name=resolved["team_name"], league_name=resolved["league_name"])


# ----------------------------------------------------------------------
# Team / squad scanning — lets the model pick its own player pool
# ----------------------------------------------------------------------

def scan_team(team_id: str, min_avg_minutes: float = MIN_AVG_MINUTES) -> list[dict]:
    """Build reports for every regular starter in a team's squad.
    
    OPTIMIZED: Fetches team matches once at the start, passes to each player's report."""
    resolved = resolve_team_and_league(team_id)
    season_id = resolved["season_id"]
    if not season_id:
        print(f"    Couldn't resolve a current season_id for team {team_id} — skipping scan")
        return []
    squad = get_team_squad(team_id)
    team_matches = get_team_recent_matches(team_id)  # FETCH ONCE HERE
    reports = []
    for player in squad:
        try:
            stats = get_player_stats(player["id"], season_id)
        except Exception:
            continue
        avg_minutes = estimate_expected_minutes(stats)
        if avg_minutes < min_avg_minutes:
            continue  # fringe/bench player — skip to save quota + noise
        try:
            report = build_report(player, stats, team_id, expected_minutes=min(avg_minutes, 90),
                                   team_name=resolved["team_name"], league_name=resolved["league_name"],
                                   matches=team_matches)
        except Exception:
            continue
        reports.append(report)
    return reports


def run_watchlist_scan(team_names: list[str] = WATCHLIST) -> list[dict]:
    """Scan every team on the fixed watchlist."""
    all_reports = []
    for name in team_names:
        try:
            team = search_team(name)
        except Exception as exc:
            print(f"  Skipping '{name}': {exc}")
            continue
        print(f"Scanning {team.get('name', name)}...")
        all_reports.extend(scan_team(team["id"]))
    return all_reports


def run_competition_scan(competition_id: str, days_ahead: int = 7) -> list[dict]:
    """Scan every team with a fixture in a competition over the next N days."""
    matches = get_upcoming_matches(competition_id, days_ahead)
    team_ids = set()
    for m in matches:
        home = m.get("home_team_id") or m.get("home_team", {}).get("id")
        away = m.get("away_team_id") or m.get("away_team", {}).get("id")
        team_ids.update({home, away} - {None})

    print(f"{len(matches)} fixtures found, {len(team_ids)} teams to scan.")
    all_reports = []
    for team_id in team_ids:
        all_reports.extend(scan_team(team_id))
    return all_reports


# Leagues the automated "today's fixtures" run scans.
# OPTIMIZED: Reduced to Premier League only to stay within TheStatsAPI rate limits.
# Player-level scanning is expensive (each player = 1 stats call + up to ROLLING_MATCHES
# match lookups). Premier League alone = ~20 teams × ~25 players = ~500 API calls per run.
# Rotate through other leagues manually or expand the list once quota is confirmed.
# To add more leagues: Championship, League One, League Two, FA Cup, EFL Cup.
DAILY_SCAN_LEAGUES = [
    "Premier League",
]


def run_daily_fixture_scan(league_names: list[str] = DAILY_SCAN_LEAGUES) -> list[dict]:
    """Scan every team with a fixture TODAY (days_ahead=0) across the
    configured leagues. This is the automated-run entry point — see
    DAILY_SCAN_LEAGUES for the quota reasoning behind keeping it to one
    league by default."""
    all_reports = []
    for name in league_names:
        try:
            comp = find_competition(name)
        except Exception as exc:
            print(f"  Skipping '{name}': {exc}")
            continue
        print(f"Scanning today's fixtures — {comp.get('name', name)}...")
        all_reports.extend(run_competition_scan(comp["id"], days_ahead=0))
    return all_reports


# ----------------------------------------------------------------------
# Criteria matching + ranking
# ----------------------------------------------------------------------

def apply_criteria(reports: list[dict], thresholds: dict = CRITERIA_THRESHOLDS) -> list[dict]:
    """Flag which threshold(s) each report clears; return only the hits."""
    qualifying = []
    for report in reports:
        hits = [prop for prop, threshold in thresholds.items()
                if report["props"].get(prop, 0) >= threshold]
        if hits:
            report["criteria_hit"] = hits
            qualifying.append(report)
    qualifying.sort(key=lambda r: max(r["props"][p] for p in r["criteria_hit"]), reverse=True)
    return qualifying


def apply_custom_criteria(
    reports: list[dict], criteria: dict = CUSTOM_CRITERIA, min_prob: float = CUSTOM_CRITERIA_MIN_PROB
) -> list[dict]:
    """Flag players by arbitrary lines (e.g. "over 2 shots") rather than
    the fixed 1.5/2.5-style props in CRITERIA_THRESHOLDS. Computes
    probability at the exact line given, using the same Poisson math
    (prob_over) applied to each report's already-blended projected rate
    — no new model logic, just a different line than the pre-built
    props happen to use. A player qualifies if ANY one line's
    probability clears min_prob."""
    qualifying = []
    for report in reports:
        hits = []
        game_log = report.get("game_log", {})
        for stat, line in criteria.items():
            lam = report["projected_per_match"].get(stat, 0.0)
            prob = prob_over(lam, line)
            if prob >= min_prob:
                hits.append({
                    "stat": stat, "line": line, "prob": round(prob, 3),
                    "hit_rate": hit_rate(game_log.get(stat, []), line),
                })
        if hits:
            report["custom_criteria_hits"] = hits
            qualifying.append(report)
    qualifying.sort(key=lambda r: max(h["prob"] for h in r["custom_criteria_hits"]), reverse=True)
    return qualifying


def top_n_by_market(reports: list[dict], top_n: int = TOP_N_PER_MARKET) -> dict:
    """Best N reports per market, regardless of whether they hit a threshold."""
    markets = reports[0]["props"].keys() if reports else CRITERIA_THRESHOLDS.keys()
    ranked = {}
    for market in markets:
        ranked[market] = sorted(
            reports, key=lambda r: r["props"].get(market, 0), reverse=True
        )[:top_n]
    return ranked


# ----------------------------------------------------------------------
# Saving results
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Safest Bet Builder — same pattern as Match IQ/Blitz IQ/Blue Line: every
# prop the model prices gets flattened into an individual leg, tagged
# with a category, for the HTML's builder panel to round-robin through.
# Correlation risk here is PER PLAYER, not per match/team the way the
# other three tools cap it — a player's shots_over_1.5 and
# shots_over_2.5 are the same underlying event at two different bars,
# so the builder caps legs per PLAYER instead of per match.
# ----------------------------------------------------------------------

# prop key -> (category, human label). Category groups legs for the
# builder's round-robin; label becomes "{player name} {label}".
PROP_CATEGORY_MAP = {
    "shots_over_1.5": ("Shots", "Over 1.5 Shots"),
    "shots_over_2.5": ("Shots", "Over 2.5 Shots"),
    "sot_over_0.5": ("Shots on Target", "Over 0.5 Shots on Target"),
    "sot_over_1.5": ("Shots on Target", "Over 1.5 Shots on Target"),
    "to_be_carded": ("Cards", "To Be Carded"),
    "to_score": ("Goals", "To Score"),
    "to_assist": ("Assists", "To Assist"),
    "goal_or_assist": ("Goal or Assist", "Goal or Assist"),
    "tackles_over_1.5": ("Tackles", "Over 1.5 Tackles"),
    "tackles_over_2.5": ("Tackles", "Over 2.5 Tackles"),
    "fouls_over_1.5": ("Fouls", "Over 1.5 Fouls"),
    "fouls_over_2.5": ("Fouls", "Over 2.5 Fouls"),
}

# category -> which game_log key backs its "last games" history string.
# "Goal or Assist" has no single key — it's goals+assists summed
# per-game, handled separately below.
CATEGORY_STAT_KEY = {
    "Shots": "shots", "Shots on Target": "shots_on_target", "Cards": "cards",
    "Goals": "goals", "Assists": "assists", "Tackles": "tackles", "Fouls": "fouls",
}


def build_legs(reports: list[dict]) -> list[dict]:
    """Flatten every scanned player's props into individual bet-builder
    legs. Each leg carries its own "last games" history for just the
    one stat it's about (not the full multi-stat game log line), so the
    builder's leg list stays readable — a "Shots" leg shows the shots
    sequence, not shots+SoT+goals+assists+cards+tackles+fouls all at
    once."""
    legs = []
    for r in reports:
        game_log = r.get("game_log", {})
        has_rolling = r.get("rolling_matches_used", 0) > 0
        for prop_key, (category, label) in PROP_CATEGORY_MAP.items():
            prob = r["props"].get(prop_key)
            if prob is None:
                continue
            history = None
            if has_rolling:
                if category == "Goal or Assist":
                    goals, assists = game_log.get("goals", []), game_log.get("assists", [])
                    if goals and assists and len(goals) == len(assists):
                        history = "/".join(str(int(g + a)) for g, a in zip(goals, assists))
                else:
                    values = game_log.get(CATEGORY_STAT_KEY.get(category), [])
                    if values:
                        history = "/".join(str(int(v)) for v in values)
            legs.append({
                "player": r["player_name"],
                "market": f"{r['player_name']} {label}",
                "prob": round(prob * 100),
                "hit_rate": (r.get("hit_rates") or {}).get(prop_key),
                "category": category,
                "detail": f"{r.get('team_name', 'Unknown team')} · {r.get('league_name', 'Unknown league')}",
                "history": history,
            })
    return legs


def save_report(report: dict) -> None:
    data = _load_output()
    data.setdefault("players", {})[report["player_name"]] = report
    OUTPUT_JSON.write_text(json.dumps(data, indent=2))
    print(f"Saved {report['player_name']} to {OUTPUT_JSON}")


def save_scan(reports: list[dict], qualifying: list[dict], ranked: dict, custom_qualifying: list[dict] | None = None) -> None:
    data = _load_output()
    data.setdefault("players", {})
    for report in reports:
        data["players"][report["player_name"]] = report
    data["screener"] = {
        "generated_at": date.today().isoformat(),
        "qualifying": [r["player_name"] for r in qualifying],
        "top_by_market": {
            market: [r["player_name"] for r in reps] for market, reps in ranked.items()
        },
        "custom_criteria_qualifying": [
            {"player_name": r["player_name"], "hits": r["custom_criteria_hits"]}
            for r in (custom_qualifying or [])
        ],
    }
    data["legs"] = build_legs(reports)
    OUTPUT_JSON.write_text(json.dumps(data, indent=2))
    print(f"Saved {len(reports)} player reports + screener results to {OUTPUT_JSON}")


def _load_output() -> dict:
    if OUTPUT_JSON.exists():
        return json.loads(OUTPUT_JSON.read_text())
    return {}


def format_game_log(game_log: dict) -> str:
    """One compact line of last-N-games sequences per stat, oldest→newest
    left-to-right — same convention as Match IQ's team history lines.
    Skips any stat with no matches found rather than printing an empty
    'goals ' entry."""
    labels = [
        ("shots", "shots"), ("shots_on_target", "SoT"), ("goals", "goals"),
        ("assists", "assists"), ("cards", "cards"), ("tackles", "tackles"),
        ("fouls", "fouls"),
    ]
    parts = []
    for stat, label in labels:
        values = game_log.get(stat, [])
        if values:
            parts.append(f"{label} " + "/".join(str(int(v)) for v in values))
    return " · ".join(parts) if parts else "—"


def print_report(report: dict) -> None:
    print(f"\n{report['player_name']}  ({report['team_name']} · {report['league_name']})  "
          f"(last {report['rolling_matches_used']} matches used for form, "
          f"{report['season_minutes_played']} season minutes)")
    if report["low_data"]:
        print("  ⚠ Not enough data (no recent matches found, thin season minutes) —"
              " probabilities below aren't reliable.")
    if report["rolling_matches_used"] > 0:
        print(f"  last games — {format_game_log(report['game_log'])}")
    for prop, prob in report["props"].items():
        hr = (report.get("hit_rates") or {}).get(prop)
        hr_str = f" ({hr['hits']}/{hr['total']})" if hr else ""
        print(f"  {prop:<16} {prob * 100:5.1f}%{hr_str}")


def run_scan(reports: list[dict]) -> None:
    if not reports:
        print("No qualifying players found.")
        return
    qualifying = apply_criteria(reports)
    custom_qualifying = apply_custom_criteria(reports)
    ranked = top_n_by_market(reports)

    def hr_suffix(r, prop):
        hr = (r.get("hit_rates") or {}).get(prop)
        return f" ({hr['hits']}/{hr['total']})" if hr else ""

    print(f"\n{len(reports)} players scanned, {len(qualifying)} met a threshold.\n")
    print("=== Meets criteria ===")
    for r in qualifying:
        hits = ", ".join(f"{p} ({r['props'][p]*100:.0f}%{hr_suffix(r, p)})" for p in r["criteria_hit"])
        print(f"  {r['player_name']:<24} {hits}")

    print(f"\n=== Meets custom criteria ({len(custom_qualifying)}) ===")
    for r in custom_qualifying:
        hits = ", ".join(
            f"over {h['line']:g} {h['stat']} ({h['prob']*100:.0f}%"
            + (f", {h['hit_rate']['hits']}/{h['hit_rate']['total']}" if h.get("hit_rate") else "")
            + ")"
            for h in r["custom_criteria_hits"]
        )
        print(f"  {r['player_name']:<24} {hits}")

    print("\n=== Top plays by market ===")
    for market, reps in ranked.items():
        print(f"\n  {market}:")
        for r in reps:
            print(f"    {r['player_name']:<24} {r['props'][market]*100:5.1f}%{hr_suffix(r, market)}")

    save_scan(reports, qualifying, ranked, custom_qualifying)


if __name__ == "__main__":
    if API_KEY == "PASTE_YOUR_KEY_HERE":
        print("Set THESTATSAPI_KEY env var or edit API_KEY in this file first.")
        raise SystemExit(1)

    if len(sys.argv) > 1 and sys.argv[1] == "--auto":
        # Non-interactive entry point for GitHub Actions — the rest of
        # this file's __main__ block is interactive (input() prompts),
        # which hangs forever with no one at a terminal to answer them.
        # This path never calls input() at all.
        print("Player Stat Model — automated daily fixture scan")
        reports = run_daily_fixture_scan()
        if not reports:
            # OBSERVED LIVE: DAILY_SCAN_LEAGUES' league can legitimately
            # have zero fixtures on a given day (leagues don't play every
            # day). run_scan()'s own early-return means save_scan() is
            # never reached in that case, so OUTPUT_JSON never gets
            # (re)written — this is NOT an error, just nothing new to
            # publish today. Say so plainly and exit 0 rather than let a
            # downstream step (the git commit) fail confusingly on a
            # file that was never expected to exist this run.
            print("No usable player reports today — either no fixtures found for the "
                  "configured leagues, or none cleared the minutes/data-quality filters. "
                  "Nothing to scan; docs/ left as whatever the last successful run published.")
            raise SystemExit(0)
        run_scan(reports)

        # Publish for GitHub Pages, same docs/ convention as Match IQ:
        # index.html + the JSON it fetches, side by side, so the HTML's
        # existing relative fetch("player_stats_data.json") keeps working
        # unchanged.
        docs_dir = Path(__file__).parent.parent / "docs" / "player-stat-model"
        docs_dir.mkdir(parents=True, exist_ok=True)
        html_src = Path(__file__).parent / "player_stat_model.html"
        if html_src.exists():
            shutil.copy(html_src, docs_dir / "index.html")
        else:
            print(f"  [!] {html_src} not found — docs/index.html not created. "
                  f"Make sure player_stat_model.html sits next to this script in the repo.")
        if OUTPUT_JSON.exists():
            shutil.copy(OUTPUT_JSON, docs_dir / OUTPUT_JSON.name)
        print(f"\nPublished to {docs_dir}/ for GitHub Pages.")
        raise SystemExit(0)

    print("Player Stat Model")
    print("  1) Look up a single player")
    print("  2) Scan the watchlist")
    print("  3) Scan a competition's upcoming fixtures")
    choice = input("Choose a mode [1/2/3]: ").strip()

    if choice == "2":
        run_scan(run_watchlist_scan())

    elif choice == "3":
        competition_id = input("Competition ID: ").strip()
        days = input("Days ahead (default 7): ").strip()
        days_ahead = int(days) if days else 7
        run_scan(run_competition_scan(competition_id, days_ahead))

    else:
        print("Type a player name (blank to quit)")
        while True:
            name = input("\nPlayer name: ").strip()
            if not name:
                break
            try:
                report = build_player_report(name)
            except Exception as exc:  # noqa: BLE001 - surfaced to the user directly
                print(f"  Could not build report: {exc}")
                continue
            print_report(report)
            save_report(report)
