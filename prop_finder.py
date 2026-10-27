"""
Prop Finder
------------
Standalone stat-finder / leaderboard tool, modeled on
playerstats.football's "Prop Finder" feature (user request, 2026-10-25:
"Apply [reverse engineering] to this website and build same" against
https://playerstats.football/?verified=1).

IMPORTANT — this is NOT a predictor like the rest of this suite (Match
IQ, Blue Line, Player Stat Model, etc): it doesn't price a probability
or a betting line. It just ranks players by a raw stat (goals, shots,
shots on target, tackles, fouls, cards) over a form window, matching
what playerstats.football's own Prop Finder actually does — a research
tool, not a pick generator. That site's own framing is "make research
on football player statistics easier", not predictions.

Sibling project to Player Stat Model — reuses the exact same confirmed
TheStatsAPI endpoints and field shapes (squad lookup, recent matches,
match-level player-stats rows, standings-based team listing already
proven in match_iq.py), kept as its own file per this suite's
one-script-per-tool convention rather than importing from
player_stat_model.py.

Unlike Player Stat Model (which blends rolling form into a single
Poisson-priced number per player), the filters this tool needs — form
window (Last 10 / Last 20), appearance type (started vs any), minutes
floor — all have to change INSTANTLY in the browser without a re-scan,
the same way playerstats.football's own filter controls work. So the
scan here stores each player's raw PER-GAME log (date, opponent,
minutes played, a "started" flag, and each stat's raw count) rather
than a single blended rate, and prop_finder.html does the actual
filtering/ranking client-side from that log — same fetch-a-JSON-file
pattern Player Stat Model's own companion HTML already uses.

Usage:
    python3 prop_finder.py

Set THESTATSAPI_KEY as an environment variable, or paste it into
API_KEY below. Results are written to prop_finder_data.json, which the
companion prop_finder.html page fetches for the actual filterable UI.
"""

import os
import sys
import json
import time
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

OUTPUT_JSON = Path(__file__).parent / "prop_finder_data.json"

HEADERS = {"Authorization": f"Bearer {API_KEY}"}

# Leagues scanned. SWITCHED from League One/League Two to Championship
# (user request, 2026-10-27): bet365 doesn't offer player-prop markets
# for League One/Two at all, only down to Championship, so a raw-stat
# research tool for leagues with no market to apply it against wasn't
# useful in practice -- the quota-overlap tradeoff the original
# League One/Two choice was weighing (see below) is moot if the
# research itself can't be acted on.
#
# This DOES reintroduce the overlap the original choice was avoiding:
# Player Stat Model's own DAILY_SCAN_LEAGUES already runs a squad scan
# of this same shape against Championship every day, and TheStatsAPI
# quota isn't shared across workflow runs (GitHub Actions starts each
# one from a clean checkout, so the file cache both scripts use never
# actually survives between separate runs) -- so this run now re-fetches
# data Player Stat Model's run already paid for today, rather than
# covering new ground. Accepted as the right trade here since a
# bettable league beats a cheaper non-bettable one. Add more
# competition names here (anything find_competition() can resolve) to
# widen coverage.
LEAGUES = [
    "Championship",
]

# Longest form window the page offers (Last 20) — fetch this many of
# each team's most recent finished matches so both "Last 10" and "Last
# 20" can be computed client-side from the one scan, with no second
# API pass when the window toggle changes.
GAMES_TO_FETCH = 20

# Skip players who don't average at least this many minutes per
# appearance this season — same fringe-player filter Player Stat Model
# already applies before a squad scan, for the same reason (don't burn
# quota, or the page's attention, on non-starters).
MIN_AVG_MINUTES = 60

# "Started" flag on a game row — UNCONFIRMED as a real API field. No
# match-stats response examined anywhere in this suite's research (see
# player_stat_model.py's own STAT_FIELDS/_extract_match_saves comments)
# includes an explicit started/substitute boolean. This is a
# minutes-played PROXY, not a real field: a game counts as "started" if
# the player played at least this many minutes in it, reusing the same
# 60-minute starter threshold Player Stat Model already uses as a
# season-level proxy (MIN_AVG_MINUTES). Documented rather than silently
# presented as the real thing, same honesty-first approach as that
# file's saves-field handling.
STARTER_MINUTES_PROXY = 60

# The stats this tool can rank by, and how to pull each one out of a
# /matches/{id}/player-stats row. Field paths are the CONFIRMED shapes
# from player_stat_model.py's own research against the real API
# response (shots/shots_on_target under "shooting", tackles under
# "defending", fouls/cards under "general" — NOT "discipline", which
# only exists on the season-stats endpoint, a different shape
# entirely). "cards" combines yellow + red, same as that file —
# playerstats.football's own quick-link is labelled "Yellow cards"
# specifically, so this is a slightly broader stat than that one link,
# not a letter-for-letter match.
STAT_FIELDS = {
    "goals": ("Goals", lambda row: row.get("shooting", {}).get("goals", 0)),
    "shots": ("Shots", lambda row: row.get("shooting", {}).get("total_shots", 0)),
    "shots_on_target": ("Shots on Target", lambda row: row.get("shooting", {}).get("shots_on_target", 0)),
    "tackles": ("Tackles", lambda row: row.get("defending", {}).get("tackles", 0)),
    "fouls": ("Fouls Committed", lambda row: row.get("general", {}).get("fouls", 0)),
    "cards": ("Cards", lambda row: (row.get("general", {}).get("yellow_cards", 0)
                                     + row.get("general", {}).get("red_cards", 0))),
}


# ----------------------------------------------------------------------
# Simple on-disk cache so re-runs don't burn API quota
# ----------------------------------------------------------------------

def _cache_path(key: str) -> Path:
    h = hashlib.sha256(key.encode()).hexdigest()[:20]
    return CACHE_DIR / f"{h}.json"


def cached_get(path: str, params: dict | None = None, ttl_hours: int = 12) -> dict:
    """GET from TheStatsAPI with a simple file cache to control quota use.
    Same cache shared with player_stat_model.py (same CACHE_DIR), so an
    identical call either script already made recently is reused."""
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
# TheStatsAPI calls — same confirmed endpoints/shapes as player_stat_model.py
# ----------------------------------------------------------------------

def find_competition(name: str) -> dict:
    """GET /competitions?search=... — exact-name match preferred over a
    substring hit (same fix match_iq.py and player_stat_model.py both
    already needed: a "Premier League" search can rank a same-named
    foreign league first)."""
    data = cached_get("/competitions", {"search": name}, ttl_hours=24 * 30)
    results = data.get("data", [])
    if not results:
        raise ValueError(f"No competition found for '{name}'")
    for comp in results:
        if comp.get("name", "").lower() == name.lower():
            return comp
    return results[0]


def get_competition_current_season(competition_id: str) -> str | None:
    data = cached_get(f"/competitions/{competition_id}", ttl_hours=24)
    return data.get("data", {}).get("current_season_id")


def get_competition_teams(competition_id: str, season_id: str) -> list[dict]:
    """Every team in a competition/season, via the standings endpoint —
    CONFIRMED shape per match_iq.py's get_standings(): each row has a
    nested "team" dict with at least id/name. Standings is also how
    match_iq.py itself gets a full team list for a league, since there
    is no dedicated /competitions/{id}/teams route in the confirmed
    spec (same gap player_stat_model.py hit for /teams/{id}/matches)."""
    data = cached_get(f"/competitions/{competition_id}/seasons/{season_id}/standings", ttl_hours=12)
    rows = data.get("data", [])
    teams = []
    for row in rows:
        team = row.get("team", {})
        if team.get("id"):
            teams.append(team)
    return teams


def get_player_stats(player_id: str, season_id: str) -> dict:
    data = cached_get(f"/players/{player_id}/stats", {"season_id": season_id}, ttl_hours=24)
    return data.get("data", {})


def get_team_squad(team_id: str) -> list[dict]:
    data = cached_get(f"/teams/{team_id}/players", ttl_hours=24 * 7)
    return data.get("data", [])


def get_team_recent_matches(team_id: str, limit: int = GAMES_TO_FETCH) -> list[dict]:
    """Most recent finished matches for a team, newest first. Sorted
    client-side (same as player_stat_model.py) since /matches' "order"
    param isn't in the confirmed query-parameters table."""
    data = cached_get(
        "/matches",
        {"team_id": team_id, "status": "finished", "per_page": 100},
        ttl_hours=6,
    )
    matches = data.get("data", [])
    matches.sort(key=lambda m: m.get("utc_date", ""), reverse=True)
    return matches[:limit]


def get_match_player_stats(match_id: str, player_id: str) -> dict | None:
    data = cached_get(f"/matches/{match_id}/player-stats", ttl_hours=24 * 30)
    rows = data.get("data", data.get("player_stats", []))
    for row in rows:
        if str(row.get("player_id")) == str(player_id):
            return row
    return None


def estimate_expected_minutes(stats: dict) -> float:
    apps = stats.get("appearances", 0)
    minutes = stats.get("minutes_played", 0)
    if apps <= 0:
        return 0.0
    return minutes / apps


# ----------------------------------------------------------------------
# Per-player game log
# ----------------------------------------------------------------------

def opponent_for(match: dict, team_id: str) -> tuple[str, bool]:
    """(opponent_name, is_home) for `team_id` in this match."""
    home = match.get("home_team", {}) or {}
    away = match.get("away_team", {}) or {}
    if str(home.get("id")) == str(team_id):
        return away.get("name", "Unknown"), True
    return home.get("name", "Unknown"), False


def build_game_log(player_id: str, team_id: str, matches: list[dict]) -> list[dict]:
    """One row per match the player actually appeared in (0 minutes =
    did not play, dropped — matches rolling_form()'s own `mins <= 0`
    skip in player_stat_model.py). Newest-first, matching `matches`'
    own order, since the HTML slices "Last 10"/"Last 20" off the front
    of this list."""
    log = []
    for match in matches:
        row = get_match_player_stats(match["id"], player_id)
        if not row:
            continue
        mins = row.get("minutes_played", 0)
        if mins <= 0:
            continue
        opp, is_home = opponent_for(match, team_id)
        entry = {
            "date": (match.get("utc_date") or "")[:10],
            "opponent": opp,
            "is_home": is_home,
            "minutes": mins,
            "started": mins >= STARTER_MINUTES_PROXY,
        }
        for stat_key, (_, extractor) in STAT_FIELDS.items():
            entry[stat_key] = extractor(row)
        log.append(entry)
    return log


# ----------------------------------------------------------------------
# Team / league scanning
# ----------------------------------------------------------------------

def scan_team(team_id: str, team_name: str, league_name: str) -> list[dict]:
    """Every regular (non-fringe) player on a team's squad, each with a
    per-game log ready for the page's client-side filters."""
    squad = get_team_squad(team_id)
    team_matches = get_team_recent_matches(team_id)
    out = []
    for player in squad:
        pid = player.get("id")
        if not pid:
            continue
        game_log = build_game_log(pid, team_id, team_matches)
        if not game_log:
            continue
        # Fringe-player filter, same MIN_AVG_MINUTES convention as
        # Player Stat Model — approximated here from the games actually
        # fetched (avg minutes across appearances in the log) rather
        # than a separate season-stats call, since the log already
        # gives a reasonable read on regular-starter status without
        # spending extra quota per player.
        avg_minutes = sum(g["minutes"] for g in game_log) / len(game_log)
        if avg_minutes < MIN_AVG_MINUTES:
            continue
        out.append({
            "player_id": pid,
            "name": player.get("name", "Unknown"),
            "position": player.get("position", ""),
            "team": team_name,
            "league": league_name,
            "game_log": game_log,
        })
    return out


def run_league_scan(league_name: str) -> list[dict]:
    comp = find_competition(league_name)
    season_id = get_competition_current_season(comp["id"])
    if not season_id:
        print(f"  Couldn't resolve a current season for '{league_name}' — skipping")
        return []
    teams = get_competition_teams(comp["id"], season_id)
    print(f"{comp.get('name', league_name)}: {len(teams)} teams")
    all_players = []
    for team in teams:
        print(f"  Scanning {team.get('name')}...")
        try:
            all_players.extend(scan_team(team["id"], team.get("name", "Unknown"), comp.get("name", league_name)))
        except Exception as exc:
            print(f"    Skipping {team.get('name')}: {exc}")
            continue
    return all_players


def run_all_leagues(league_names: list[str] = LEAGUES) -> list[dict]:
    all_players = []
    for name in league_names:
        try:
            all_players.extend(run_league_scan(name))
        except Exception as exc:
            print(f"Skipping league '{name}': {exc}")
            continue
    return all_players


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    if API_KEY == "PASTE_YOUR_KEY_HERE":
        print("Set THESTATSAPI_KEY (env var) or paste a key into API_KEY before running.", file=sys.stderr)
        sys.exit(1)

    players = run_all_leagues()
    output = {
        "generated_at": date.today().isoformat(),
        "leagues": LEAGUES,
        "stats": {key: label for key, (label, _) in STAT_FIELDS.items()},
        "players": players,
    }
    OUTPUT_JSON.write_text(json.dumps(output, indent=2))
    print(f"\nWrote {len(players)} players across {len(LEAGUES)} league(s) -> {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
