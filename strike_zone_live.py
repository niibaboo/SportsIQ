#!/usr/bin/env python3
"""
Strike Zone Live Signals — live in-game next-run & total-pace signals for
MLB games in progress. Same "Model Signal" pattern as Match IQ Live /
Blitz IQ Live: a green-dot signal that only qualifies once its probability
clears 70%, same bar used everywhere else in the suite.

Reuses build_slate.py's own pregame run projections (docs/strike-zone/
slate_report.json — the same lambda values shown on the main Strike Zone
page) as the baseline, rather than refetching each team's hitting/pitching
stats every 10 minutes. That file is written once a day by build_slate.py's
own run, so this script just reads it and matches games by gamePk.

Each team's remaining expected runs is adjusted for how their actual
scoring pace THIS game compares to what the pregame projection implied by
now — same idea as Match IQ Live's live-xG adjustment and Blitz IQ Live's
pace adjustment.

Two live signals per game:
  - Next Run: which team is more likely to score next, given innings
    remaining and each team's pace-adjusted remaining run rate.
  - Game Total Pace: whether the combined runs are tracking clearly
    above/below the pregame total projection (sum of both teams' lambdas).

Setup:
    python3 strike_zone_live.py
    (requires docs/strike-zone/slate_report.json to already exist —
    written by today's build_slate.py run)

Output:
    docs/strike-zone-live/live_games.json (consumed by live.html every 60s)
"""

import os
import json
import math
import time
from datetime import datetime, timezone

import requests

BASE = "https://statsapi.mlb.com/api/v1"
SLATE_PATH = "docs/strike-zone/slate_report.json"
MODEL_SIGNAL_THRESHOLD = 70  # same bar as Match IQ Live / Blitz IQ Live / Daily Signals scanners
TOTAL_INNINGS = 9
TOTAL_HALF_INNINGS = TOTAL_INNINGS * 2


def _get(url, params=None, timeout=15):
    try:
        r = requests.get(url, params=params or {}, timeout=timeout)
        time.sleep(0.5)
        if r.status_code != 200:
            print(f"  [!] {r.status_code} on {url}")
            return None
        return r.json()
    except Exception as e:
        time.sleep(0.5)
        print(f"  [!] request failed: {url} ({e})")
        return None


def poisson_pmf(k, lam):
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def poisson_cdf(k, lam):
    return sum(poisson_pmf(i, lam) for i in range(int(k) + 1))


def half_innings_elapsed(inning, state):
    """MLB reports the current inning number plus a state: 'Top', 'Middle'
    (top just ended), 'Bottom', or 'End' (bottom just ended). Top/Middle
    count as the top half not yet fully elapsed... actually once it's
    'Middle' the top half IS complete, so both 'Middle' and 'Bottom' mean
    one half-inning has elapsed this inning; 'Top' means zero have. 'End'
    means both halves of this inning are done."""
    if inning is None:
        return None
    base = (inning - 1) * 2
    if state == 'Top':
        return base
    if state in ('Middle', 'Bottom'):
        return base + 1
    if state == 'End':
        return base + 2
    return base  # unknown state — treat conservatively as top just started


def innings_remaining(half_elapsed):
    """Floors at 0 once regulation (9 innings) is fully elapsed — same
    known approximation as Blitz IQ Live's overtime handling: extra
    innings show 0 remaining rather than extending the total, since we
    have no way to know how many extra innings a tied game will need."""
    if half_elapsed is None:
        return 0
    return max(0, (TOTAL_HALF_INNINGS - half_elapsed) / 2)


def calc_pace_adjustment(actual_runs, exp_runs_pregame, elapsed_fraction):
    """How a team's ACTUAL runs so far compare to what their pregame
    projection implied by now — same idea as Match IQ Live's live-xG
    adjustment and Blitz IQ Live's pace adjustment. A 1-run floor on
    "expected by now" avoids dividing by a near-zero expectation early."""
    expected_by_now = exp_runs_pregame * elapsed_fraction if elapsed_fraction > 0 else 0.1
    adjustment = actual_runs / max(expected_by_now, 1.0)
    return min(max(adjustment, 0.5), 2.0)


def calc_next_run_prob(home_exp_remaining, away_exp_remaining):
    total = home_exp_remaining + away_exp_remaining
    if total <= 0:
        return {"home": 50, "away": 50}
    return {
        "home": round((home_exp_remaining / total) * 100),
        "away": round((away_exp_remaining / total) * 100),
    }


def calc_game_total_pace(actual_total, pregame_total, remaining_mean):
    """Poisson-based (matching every other pregame run/goal projection in
    the suite, unlike Blitz IQ Live's Normal-distribution approach, since
    baseball scoring is a low-count event like soccer goals)."""
    if remaining_mean <= 0:  # regulation innings fully elapsed
        prob_over = 100 if actual_total > pregame_total else 0
        status = "Final: Over pregame total" if prob_over else "Final: Under pregame total"
        return {"prob": prob_over, "status": status}

    needed = pregame_total - actual_total
    if needed < 0:
        prob_over = 100
    else:
        prob_over = round((1 - poisson_cdf(math.floor(needed), remaining_mean)) * 100)
    status = "On Pace: Over" if prob_over >= 50 else "On Pace: Under"
    return {"prob": prob_over, "status": status}


def calc_model_signal(next_run, total_pace):
    total_direction = total_pace["status"].split(": ")[-1] if ": " in total_pace["status"] else total_pace["status"]
    total_confidence = total_pace["prob"] if "Over" in total_direction else (100 - total_pace["prob"])

    candidates = [
        {"label": "Home Next Run", "prob": next_run["home"]},
        {"label": "Away Next Run", "prob": next_run["away"]},
        {"label": f"Game Total {total_direction}", "prob": total_confidence},
    ]
    best = max(candidates, key=lambda c: c["prob"])
    return {
        "label": best["label"],
        "prob": best["prob"],
        "qualifies": best["prob"] >= MODEL_SIGNAL_THRESHOLD,
    }


def load_slate_projections():
    """Reads today's already-written slate_report.json rather than
    refetching pitcher/team stats — single source of truth with the main
    Strike Zone page, and far fewer MLB Stats API calls per live run."""
    if not os.path.exists(SLATE_PATH):
        print(f"  [!] {SLATE_PATH} not found — has build_slate.py run today?")
        return {}
    try:
        with open(SLATE_PATH) as f:
            slate = json.load(f)
    except Exception as e:
        print(f"  [!] couldn't read {SLATE_PATH}: {e}")
        return {}

    by_pk = {}
    for entry in slate:
        pk = entry.get("game_pk")
        team_runs = entry.get("team_runs") or {}
        home_lam = (team_runs.get("home") or {}).get("lambda")
        away_lam = (team_runs.get("away") or {}).get("lambda")
        if pk is None or home_lam is None or away_lam is None:
            continue
        by_pk[pk] = {
            "home_team": entry.get("home"), "away_team": entry.get("away"),
            "home_lambda": home_lam, "away_lambda": away_lam,
            "pregame_total": round(home_lam + away_lam, 2),
        }
    return by_pk


def build_live_signals():
    projections = load_slate_projections()
    if not projections:
        print("No pregame projections available — aborting.")
        return []

    live_games = []
    print(f"Checking {len(projections)} game(s) from today's slate for live status...")

    for game_pk, proj in projections.items():
        data = _get(f"{BASE}/schedule", params={"gamePk": game_pk, "hydrate": "linescore"})
        if not data:
            continue
        try:
            game = data["dates"][0]["games"][0]
        except (KeyError, IndexError):
            continue

        state = game.get("status", {}).get("abstractGameState")
        if state != "Live":
            continue

        linescore = game.get("linescore") or {}
        inning = linescore.get("currentInning")
        inning_state = linescore.get("inningState")
        half_elapsed = half_innings_elapsed(inning, inning_state)
        remaining_innings = innings_remaining(half_elapsed)
        elapsed_fraction = min((half_elapsed or 0) / TOTAL_HALF_INNINGS, 1.0)

        teams_runs = (linescore.get("teams") or {})
        try:
            home_runs = float((teams_runs.get("home") or {}).get("runs") or 0)
            away_runs = float((teams_runs.get("away") or {}).get("runs") or 0)
        except (TypeError, ValueError):
            home_runs, away_runs = 0.0, 0.0

        home_adj = calc_pace_adjustment(home_runs, proj["home_lambda"], elapsed_fraction)
        away_adj = calc_pace_adjustment(away_runs, proj["away_lambda"], elapsed_fraction)

        home_exp_remaining = proj["home_lambda"] * (remaining_innings / TOTAL_INNINGS) * home_adj
        away_exp_remaining = proj["away_lambda"] * (remaining_innings / TOTAL_INNINGS) * away_adj

        next_run = calc_next_run_prob(home_exp_remaining, away_exp_remaining)
        total_pace = calc_game_total_pace(
            home_runs + away_runs, proj["pregame_total"], home_exp_remaining + away_exp_remaining
        )
        model_signal = calc_model_signal(next_run, total_pace)

        live_games.append({
            "game_pk": game_pk,
            "home_team": proj["home_team"],
            "away_team": proj["away_team"],
            "home_runs": home_runs,
            "away_runs": away_runs,
            "inning": inning,
            "inning_state": inning_state,
            "status": game.get("status", {}).get("detailedState", "Live"),
            "half_innings_remaining": round(remaining_innings * 2) if remaining_innings else 0,
            "next_run": next_run,
            "game_total_pace": total_pace,
            "model_signal": model_signal,
            "pregame_projection": {
                "home_lambda": proj["home_lambda"],
                "away_lambda": proj["away_lambda"],
                "pregame_total": proj["pregame_total"],
            },
        })

    return live_games


if __name__ == "__main__":
    live = build_live_signals()

    os.makedirs("docs/strike-zone-live", exist_ok=True)

    output = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "total_live": len(live),
        "games": live,
    }

    with open("docs/strike-zone-live/live_games.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    try:
        import strike_zone_live_results_tracker
        strike_zone_live_results_tracker.run_results_tracker(live)
    except Exception as e:
        print(f"[!] Results tracker failed, but the rest of this run succeeded: {e}")

    print(f"\nDone — {len(live)} live game(s) found and written to live_games.json")
