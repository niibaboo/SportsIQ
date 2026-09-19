#!/usr/bin/env python3
"""
Match IQ Live Signals — next goal + BTTS probabilities for in-play matches
across 9 leagues (Premier League, Championship, Bundesliga, Eredivisie,
Primeira Liga, La Liga, UEFA Champions League, UEFA Europa League, MLS).

Runs every 60 seconds (via GitHub Actions or local scheduler), pulls live
match data from TheStatsAPI, calculates in-play probabilities, outputs JSON
that the frontend polls for real-time updates.

Setup:
    pip3 install requests --break-system-packages
    python3 match_iq_live.py YOUR_API_KEY

Output:
    docs/match-iq-live/live_matches.json (consumed by live.html every 60s)
"""

import os
import sys
import time
import math
import json
from datetime import datetime, timezone
import requests

BASE = "https://api.thestatsapi.com/api"

# Same 9 leagues as Match IQ — live signals scoped to only these
LEAGUE_SEARCH_NAMES = [
    "Premier League",
    "Championship",
    "Bundesliga",
    "Eredivisie",
    "Primeira Liga",
    "La Liga",
    "UEFA Champions League",
    "UEFA Europa League",
    "MLS",
]


def _headers(key):
    return {"Authorization": f"Bearer {key}"}


def _get(path, key, params=None, timeout=15):
    """Same adaptive rate-limiting as match_iq.py — reads X-RateLimit-Remaining
    and backs off only when needed, not speculatively."""
    try:
        r = requests.get(f"{BASE}{path}", headers=_headers(key), params=params or {}, timeout=timeout)
    except Exception as e:
        print(f"  [!] request failed: {path} ({e})")
        return None

    remaining = r.headers.get("X-RateLimit-Remaining")
    reset = r.headers.get("X-RateLimit-Reset")
    if remaining is not None:
        try:
            remaining = int(remaining)
            if remaining <= 2 and reset:
                wait = max(0, int(reset) - int(time.time())) + 3
                print(f"  Rate limit nearly exhausted ({remaining} left) — waiting {wait}s...")
                time.sleep(wait)
        except (ValueError, TypeError):
            pass

    if r.status_code == 429:
        retry_after = int(r.headers.get("Retry-After", 30))
        print(f"  [!] 429 rate limited — waiting {retry_after}s and retrying once...")
        time.sleep(retry_after)
        return _get(path, key, params, timeout)

    if r.status_code != 200:
        print(f"  [!] {r.status_code} on {path}: {r.text[:200]}")
        return None

    time.sleep(2.0)
    return r.json()


def poisson_pmf(k, lam):
    """Probability of exactly k goals scored given rate lam."""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def poisson_cdf(k, lam):
    """Probability of k or fewer goals."""
    return sum(poisson_pmf(i, lam) for i in range(k + 1))


def find_competition(name, key):
    """Find competition by exact name match."""
    data = _get("/football/competitions", key, params={"search": name, "per_page": 5})
    if not data:
        return None
    for c in data.get("data", []):
        if c["name"].lower() == name.lower():
            return c
    return data["data"][0] if data.get("data") else None


def get_live_matches(competition_id, key):
    """Fetch live/in-play matches for a competition — status code 3 = in-play."""
    data = _get("/football/matches", key, params={
        "competition_id": competition_id,
        "status": "live",  # only live/in-play matches
        "per_page": 20,
    })
    return data.get("data", []) if data else []


def get_team_season_stats(team_id, competition_id, season_id, key):
    """Fetch a team's season-level stats (goals for/against, shots, etc.)
    to use as the baseline scoring rate for this team in this league."""
    data = _get("/football/teams", key, params={"team_id": team_id})
    if not data or not data.get("data"):
        return None
    
    # Extract season stats from the team detail — this is a simplified
    # version that just gets the aggregate; a fuller implementation would
    # parse per-match history like match_iq.py does, but for live signals
    # we can start with season-level averages and refine later.
    team = data["data"][0] if isinstance(data.get("data"), list) else data.get("data", {})
    stats = team.get("statistics", {})
    
    # Fallback to a neutral estimate if we can't find season stats
    return {
        "goals_for": stats.get("goals_for", 1.5),  # season goals/match average
        "shots": stats.get("shots", 12),  # season shots/match
        "xg": stats.get("expected_goals", 1.2),  # season xG/match
    }


def estimate_remaining_time(match):
    """Extract elapsed minutes from the match object.
    TheStatsAPI provides 'elapsed_minutes' field directly on the match."""
    # Get elapsed minutes directly from match (not from status)
    elapsed = match.get("elapsed_minutes")
    if elapsed is None:
        return None
    return elapsed


def get_live_match_stats(match_id, key):
    """Fetch live match statistics from TheStatsAPI's live-stats endpoint.
    Response structure:
    {
      "data": {
        "stats": {
          "ball_possession": {"all": {"home": 48, "away": 52}},
          "total_shots": {"all": {"home": 7, "away": 9}},
          "corner_kicks": {"all": {"home": 4, "away": 5}}
        }
      }
    }
    """
    data = _get(f"/football/matches/{match_id}/live-stats", key)
    
    if data is None:
        return None
    
    # Navigate to the stats object
    if not data.get("data") or not data["data"].get("stats"):
        return None
    
    stats = data["data"]["stats"]
    
    # Extract possession
    possession = stats.get("ball_possession", {}).get("all", {})
    home_possession = possession.get("home")
    away_possession = possession.get("away")
    
    # Extract total shots
    shots = stats.get("total_shots", {}).get("all", {})
    home_shots = shots.get("home")
    away_shots = shots.get("away")
    
    # Extract shots on target (if available)
    shots_on_target = stats.get("shots_on_target", {}).get("all", {})
    home_sot = shots_on_target.get("home")
    away_sot = shots_on_target.get("away")
    
    # Extract corner kicks
    corners = stats.get("corner_kicks", {}).get("all", {})
    home_corners = corners.get("home")
    away_corners = corners.get("away")
    
    return {
        "home": {
            "shots": home_shots,
            "shots_on_target": home_sot,
            "xg": None,  # Not available in this endpoint
            "possession": home_possession,
            "corners": home_corners,
        },
        "away": {
            "shots": away_shots,
            "shots_on_target": away_sot,
            "xg": None,  # Not available in this endpoint
            "possession": away_possession,
            "corners": away_corners,
        }
    }


def calc_next_goal_prob(home_exp_rate, away_exp_rate, minutes_remaining, 
                        home_live_xg=None, away_live_xg=None, home_shots=None, away_shots=None):
    """Given expected goals per match and minutes left, estimate P(each team
    scores next). Uses live match stats (xG so far, shots so far) if available,
    otherwise falls back to season baselines.
    
    If live stats are provided, adjust the remaining expected goals based on
    what's happened so far: a team that's accumulated high xG is more likely
    to score next."""
    if minutes_remaining <= 0:
        return {"home": 0, "away": 0, "reasoning": "Match ended"}
    
    # If live xG data is available, use it to adjust; otherwise use season rates
    if home_live_xg is not None and away_live_xg is not None:
        # Adjust the remaining expected goals based on live performance so far.
        # If a team has high xG accumulated, they're likely the stronger team
        # in this match, so weight their remaining rate higher.
        
        # Normalize live xG to a multiplier (higher xG = higher multiplier on remaining rate)
        # E.g., if home has 1.5 xG so far and season rate is 1.5/match, they're
        # performing at season pace; if they have 0.5 xG, they're underperforming.
        elapsed_fraction = (90 - minutes_remaining) / 90.0
        
        home_expected_by_now = home_exp_rate * elapsed_fraction if elapsed_fraction > 0 else 0.1
        away_expected_by_now = away_exp_rate * elapsed_fraction if elapsed_fraction > 0 else 0.1
        
        # Adjust remaining rate based on actual vs expected so far
        # If home has more xG than expected, boost their remaining rate
        home_adjustment = home_live_xg / max(home_expected_by_now, 0.5)
        away_adjustment = away_live_xg / max(away_expected_by_now, 0.5)
        
        # Cap adjustments so we don't go too extreme
        home_adjustment = min(max(home_adjustment, 0.5), 2.0)
        away_adjustment = min(max(away_adjustment, 0.5), 2.0)
        
        home_exp_remaining = home_exp_rate * (minutes_remaining / 90.0) * home_adjustment
        away_exp_remaining = away_exp_rate * (minutes_remaining / 90.0) * away_adjustment
        
        reasoning = f"{minutes_remaining}min left, live xG: {home_live_xg:.2f}/{away_live_xg:.2f}, adj: {home_adjustment:.2f}x/{away_adjustment:.2f}x"
    else:
        # Fallback to season baseline if live stats aren't available yet
        home_exp_remaining = home_exp_rate * (minutes_remaining / 90.0)
        away_exp_remaining = away_exp_rate * (minutes_remaining / 90.0)
        reasoning = f"{minutes_remaining}min left, season rates {home_exp_rate:.2f}/{away_exp_rate:.2f} (no live stats yet)"
    
    # P(team scores next) ≈ ratio of their scoring rates
    total_exp = home_exp_remaining + away_exp_remaining
    if total_exp <= 0:
        return {"home": 50, "away": 50, "reasoning": "Very low scoring expected"}
    
    home_prob = (home_exp_remaining / total_exp) * 100
    away_prob = (away_exp_remaining / total_exp) * 100
    
    return {
        "home": round(home_prob),
        "away": round(away_prob),
        "reasoning": reasoning,
    }


def calc_btts_prob(home_score, away_score, home_exp_rate, away_exp_rate, minutes_remaining,
                   home_live_xg=None, away_live_xg=None):
    """BTTS probability for remaining time. Uses live xG if available, otherwise
    falls back to season rates. If either team has 0 goals already and the other
    has scored, BTTS is still possible (the team with 0 needs to score before final
    whistle). If both have scored, BTTS is already 100%.
    If minutes_remaining <= 0, it's locked at whatever the final was."""
    if minutes_remaining <= 0:
        if home_score > 0 and away_score > 0:
            return {"prob": 100, "status": "BTTS Already Hit", "reasoning": "Match ended with both teams scoring"}
        else:
            return {"prob": 0, "status": "BTTS Failed", "reasoning": "Match ended without both teams scoring"}
    
    if home_score > 0 and away_score > 0:
        return {"prob": 100, "status": "BTTS Hit", "reasoning": "Both teams have already scored"}
    
    # Adjust expected rates based on live xG if available
    if home_live_xg is not None and away_live_xg is not None:
        elapsed_fraction = (90 - minutes_remaining) / 90.0
        home_expected_by_now = home_exp_rate * elapsed_fraction if elapsed_fraction > 0 else 0.1
        away_expected_by_now = away_exp_rate * elapsed_fraction if elapsed_fraction > 0 else 0.1
        
        home_adjustment = home_live_xg / max(home_expected_by_now, 0.5)
        away_adjustment = away_live_xg / max(away_expected_by_now, 0.5)
        home_adjustment = min(max(home_adjustment, 0.5), 2.0)
        away_adjustment = min(max(away_adjustment, 0.5), 2.0)
        
        home_exp_remaining = home_exp_rate * (minutes_remaining / 90.0) * home_adjustment
        away_exp_remaining = away_exp_rate * (minutes_remaining / 90.0) * away_adjustment
        xg_note = f" (live xG: {home_live_xg:.2f}/{away_live_xg:.2f})"
    else:
        home_exp_remaining = home_exp_rate * (minutes_remaining / 90.0)
        away_exp_remaining = away_exp_rate * (minutes_remaining / 90.0)
        xg_note = ""
    
    if home_score > 0:
        # Away needs to score
        prob = (1 - poisson_pmf(0, away_exp_remaining)) * 100
        return {"prob": round(prob), "status": f"Away must score ({minutes_remaining}min left)", "reasoning": f"Away exp rate {away_exp_rate:.2f}{xg_note}"}
    
    if away_score > 0:
        # Home needs to score
        prob = (1 - poisson_pmf(0, home_exp_remaining)) * 100
        return {"prob": round(prob), "status": f"Home must score ({minutes_remaining}min left)", "reasoning": f"Home exp rate {home_exp_rate:.2f}{xg_note}"}
    
    # Both teams still at 0 — need both to score
    btts_prob = (1 - poisson_pmf(0, home_exp_remaining)) * (1 - poisson_pmf(0, away_exp_remaining))
    return {"prob": round(btts_prob * 100), "status": "0-0 (both must score)", "reasoning": f"Home {home_exp_rate:.2f}, Away {away_exp_rate:.2f}{xg_note}"}


def build_live_signals(key):
    """Fetch live matches across all 9 leagues, calculate next-goal and BTTS
    probabilities, return list of live match objects."""
    live_matches = []
    
    for league_name in LEAGUE_SEARCH_NAMES:
        print(f"Checking {league_name} for live matches...")
        comp = find_competition(league_name, key)
        if not comp:
            print(f"  [!] couldn't find {league_name}")
            continue
        
        comp_id = comp["id"]
        season_id = comp.get("current_season_id")
        if not season_id:
            details = _get(f"/football/competitions/{comp_id}", key)
            season_id = details["data"].get("current_season_id") if details else None
        if not season_id:
            print(f"  [!] no current season for {league_name}")
            continue
        
        matches = get_live_matches(comp_id, key)
        print(f"  {len(matches)} live match(es)")
        
        for m in matches:
            home_id = m["home_team"]["id"]
            away_id = m["away_team"]["id"]
            
            # Get team season stats (used as the baseline scoring rate)
            home_stats = get_team_season_stats(home_id, comp_id, season_id, key)
            away_stats = get_team_season_stats(away_id, comp_id, season_id, key)
            
            if not home_stats or not away_stats:
                print(f"    Skipping {m['home_team']['name']} vs {m['away_team']['name']} — missing stats")
                continue
            
            # Current score
            score = m.get("score", {})
            home_goals = score.get("home", 0)
            away_goals = score.get("away", 0)
            
            # Get actual elapsed minute from match status
            elapsed_minute = estimate_remaining_time(m)
            if elapsed_minute is None:
                # Can't determine minute, skip this match
                print(f"    Skipping {m['home_team']['name']} vs {m['away_team']['name']} — no minute data")
                continue
            
            # Calculate remaining time for probability calculations (assume 90 min match)
            minutes_left = max(0, 90 - elapsed_minute)
            
            # Fetch live match statistics (shots, xG so far)
            # Some matches may not have stats available (404) — that's OK, fall back to season baselines
            try:
                live_stats = get_live_match_stats(m["id"], key)
            except Exception as e:
                print(f"    Warning: couldn't fetch stats for {m['home_team']['name']} vs {m['away_team']['name']}: {e}")
                live_stats = None
            
            home_live_xg = live_stats["home"]["xg"] if live_stats else None
            away_live_xg = live_stats["away"]["xg"] if live_stats else None
            home_shots = live_stats["home"]["shots"] if live_stats else None
            away_shots = live_stats["away"]["shots"] if live_stats else None
            
            # Calculate probabilities using both season baselines and live stats
            next_goal = calc_next_goal_prob(
                home_stats["goals_for"], away_stats["goals_for"], minutes_left,
                home_live_xg=home_live_xg, away_live_xg=away_live_xg,
                home_shots=home_shots, away_shots=away_shots
            )
            btts = calc_btts_prob(
                home_goals, away_goals,
                home_stats["goals_for"], away_stats["goals_for"],
                minutes_left,
                home_live_xg=home_live_xg, away_live_xg=away_live_xg
            )
            
            live_matches.append({
                "league": league_name,
                "match_id": m["id"],
                "home_team": m["home_team"]["name"],
                "away_team": m["away_team"]["name"],
                "home_goals": home_goals,
                "away_goals": away_goals,
                "elapsed_minute": elapsed_minute,
                "minutes_remaining": minutes_left,
                "status": m.get("status", "Unknown") if isinstance(m.get("status"), str) else m.get("status", {}).get("description", "Unknown"),
                "kickoff": m.get("utc_date", ""),
                "next_goal": next_goal,
                "btts": btts,
                "home_season_stats": home_stats,  # season averages (baseline)
                "away_season_stats": away_stats,  # season averages (baseline)
                "home_live_stats": {  # live stats this match
                    "shots": home_shots,
                    "shots_on_target": live_stats["home"]["shots_on_target"] if live_stats else None,
                    "xg": live_stats["home"]["xg"] if live_stats else None,
                    "possession": live_stats["home"]["possession"] if live_stats else None,
                    "corners": live_stats["home"]["corners"] if live_stats else None,
                },
                "away_live_stats": {  # live stats this match
                    "shots": away_shots,
                    "shots_on_target": live_stats["away"]["shots_on_target"] if live_stats else None,
                    "xg": live_stats["away"]["xg"] if live_stats else None,
                    "possession": live_stats["away"]["possession"] if live_stats else None,
                    "corners": live_stats["away"]["corners"] if live_stats else None,
                },
            })
    
    return live_matches


if __name__ == "__main__":
    api_key = os.environ.get("THESTATSAPI_KEY")
    if not api_key and len(sys.argv) >= 2:
        api_key = sys.argv[1]
    if not api_key:
        print("Usage: python3 match_iq_live.py YOUR_API_KEY")
        print("  (or set the THESTATSAPI_KEY environment variable)")
        sys.exit(1)
    
    live = build_live_signals(api_key)
    
    os.makedirs("docs/match-iq-live", exist_ok=True)
    
    output = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "total_live": len(live),
        "matches": live,
    }
    
    with open("docs/match-iq-live/live_matches.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nDone — {len(live)} live match(es) found and written to live_matches.json")
