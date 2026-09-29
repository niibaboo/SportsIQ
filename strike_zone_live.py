#!/usr/bin/env python3
"""
Strike Zone Live Signals — live in-game signals for MLB games in progress.
Same "Model Signal" pattern as Match IQ Live / Blitz IQ Live: a green-dot
signal that only qualifies once its probability clears 70%, same bar used
everywhere else in the suite.

Reuses build_slate.py's own pregame run/hits projections (docs/strike-zone/
slate_report.json — the same lambda values shown on the main Strike Zone
page) as the baseline, rather than refetching each team's hitting/pitching
stats every 10 minutes. That file is written once a day by build_slate.py's
own run, so this script just reads it and matches games by gamePk.

Each team's remaining/per-inning expected runs (and hits) is adjusted for
how their actual pace THIS game compares to what the pregame projection
implied by now — same idea as Match IQ Live's live-xG adjustment and
Blitz IQ Live's pace adjustment.

Live signals per game:
  - Next Run: which team is more likely to score next. INFORMATIONAL
    ONLY -- unlike soccer's "next goal" or NFL's "next score", books
    generally don't offer a bettable "next team to score" market for
    baseball, so this is excluded from Model Signal (nothing to bet).
  - Game Total Pace: whether the combined runs are tracking clearly
    above/below the pregame total projection (sum of both teams' lambdas).
    A real, bettable live market (Live Total).
  - Inning Winner: 3-way (Home/Away/Tie) odds on which team outscores
    the other in the CURRENT inning specifically -- matches books'
    "Nth Inning Lines: Winner" market.
  - Inning Most Hits: 3-way (Home/Away/Tie) odds on which team gets more
    hits in the current inning -- matches books' "Nth Inning Lines: Most
    Hits" market. Needs each team's pregame HITS projection (from
    build_slate.py's project_team_hits); skipped for a game if that
    wasn't available in today's slate.
  Both inning markets use each team's PER-INNING share of their full-game
  pregame rate (lambda/9), pace-adjusted -- a simplification that doesn't
  account for outs already recorded in a half-inning already underway,
  same level of approximation as the rest of the live suite.

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
    """INFORMATIONAL ONLY -- see module docstring. Kept for display since
    it's still a genuinely informative number, just excluded from
    calc_model_signal's candidates because books don't offer this as a
    bettable market for baseball."""
    total = home_exp_remaining + away_exp_remaining
    if total <= 0:
        return {"home": 50, "away": 50}
    return {
        "home": round((home_exp_remaining / total) * 100),
        "away": round((away_exp_remaining / total) * 100),
    }


def _ordinal(n):
    if n is None:
        return "?"
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def inning_three_way_probs(home_exp, away_exp, max_n=10):
    """Home/Away/Tie probabilities for a SINGLE inning, from a Poisson
    grid over each team's per-inning expected runs (or hits) -- same
    grid approach as Euro Ice's win_probs_and_scores / Strike Zone's
    F5 matchup card, just scoped to one inning instead of a full game
    or 5-inning window. Per-inning counts are small, so max_n=10 leaves
    negligible tail probability."""
    ph = pa = pt = 0.0
    for i in range(max_n):
        for j in range(max_n):
            p = poisson_pmf(i, home_exp) * poisson_pmf(j, away_exp)
            if i > j:
                ph += p
            elif j > i:
                pa += p
            else:
                pt += p
    return {"home": round(ph * 100), "away": round(pa * 100), "tie": round(pt * 100)}


def best_of_three(label_prefix, home_team, away_team, probs):
    """Collapse a 3-way Home/Away/Tie market down to its single highest-
    probability outcome, as one Model-Signal-ready candidate -- same
    "pick the best side of this market" idea as calc_game_total_pace
    picking Over vs Under."""
    candidates = [
        {"label": f"{label_prefix}: {home_team}", "prob": probs["home"]},
        {"label": f"{label_prefix}: {away_team}", "prob": probs["away"]},
        {"label": f"{label_prefix}: Tie", "prob": probs["tie"]},
    ]
    return max(candidates, key=lambda c: c["prob"])


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


def calc_model_signal(total_pace, inning_winner_best, inning_hits_best):
    """Picks the single highest-confidence signal from only the markets
    that are actually bettable in-play: Game Total, Inning Winner, and
    Inning Most Hits. Next Run is deliberately excluded -- see module
    docstring -- since there's no market to place it against."""
    total_direction = total_pace["status"].split(": ")[-1] if ": " in total_pace["status"] else total_pace["status"]
    total_confidence = total_pace["prob"] if "Over" in total_direction else (100 - total_pace["prob"])

    candidates = [
        {"label": f"Game Total {total_direction}", "prob": total_confidence},
        inning_winner_best,
    ]
    if inning_hits_best is not None:
        candidates.append(inning_hits_best)

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

        # Hits projections are used for the Inning Most Hits market --
        # optional: a game without them still gets Next Run / Game Total /
        # Inning Winner, just not Inning Most Hits.
        team_hits = entry.get("team_hits") or {}
        home_hits_lam = (team_hits.get("home") or {}).get("lambda")
        away_hits_lam = (team_hits.get("away") or {}).get("lambda")

        by_pk[pk] = {
            "home_team": entry.get("home"), "away_team": entry.get("away"),
            "home_lambda": home_lam, "away_lambda": away_lam,
            "pregame_total": round(home_lam + away_lam, 2),
            "home_hits_lambda": home_hits_lam, "away_hits_lambda": away_hits_lam,
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

        teams_data = (linescore.get("teams") or {})
        try:
            home_runs = float((teams_data.get("home") or {}).get("runs") or 0)
            away_runs = float((teams_data.get("away") or {}).get("runs") or 0)
        except (TypeError, ValueError):
            home_runs, away_runs = 0.0, 0.0
        try:
            home_hits = float((teams_data.get("home") or {}).get("hits") or 0)
            away_hits = float((teams_data.get("away") or {}).get("hits") or 0)
        except (TypeError, ValueError):
            home_hits, away_hits = 0.0, 0.0

        home_adj = calc_pace_adjustment(home_runs, proj["home_lambda"], elapsed_fraction)
        away_adj = calc_pace_adjustment(away_runs, proj["away_lambda"], elapsed_fraction)

        home_exp_remaining = proj["home_lambda"] * (remaining_innings / TOTAL_INNINGS) * home_adj
        away_exp_remaining = proj["away_lambda"] * (remaining_innings / TOTAL_INNINGS) * away_adj

        next_run = calc_next_run_prob(home_exp_remaining, away_exp_remaining)
        total_pace = calc_game_total_pace(
            home_runs + away_runs, proj["pregame_total"], home_exp_remaining + away_exp_remaining
        )

        # Inning Winner: each team's per-inning share of their full-game
        # pregame run rate, pace-adjusted by the same home_adj/away_adj
        # already computed above (today's hot/cold factor applied evenly
        # per inning, not just to the remaining-game total).
        home_run_per_inning = proj["home_lambda"] / TOTAL_INNINGS * home_adj
        away_run_per_inning = proj["away_lambda"] / TOTAL_INNINGS * away_adj
        inning_winner = inning_three_way_probs(home_run_per_inning, away_run_per_inning)
        inning_winner_best = best_of_three(
            f"{_ordinal(inning)} Inning Winner", proj["home_team"], proj["away_team"], inning_winner
        )

        # Inning Most Hits: same idea, using each team's pregame HITS
        # projection instead of runs -- skipped if that game had no hits
        # projection in today's slate.
        inning_hits = None
        inning_hits_best = None
        home_hits_lam = proj.get("home_hits_lambda")
        away_hits_lam = proj.get("away_hits_lambda")
        if home_hits_lam is not None and away_hits_lam is not None:
            home_hits_adj = calc_pace_adjustment(home_hits, home_hits_lam, elapsed_fraction)
            away_hits_adj = calc_pace_adjustment(away_hits, away_hits_lam, elapsed_fraction)
            home_hits_per_inning = home_hits_lam / TOTAL_INNINGS * home_hits_adj
            away_hits_per_inning = away_hits_lam / TOTAL_INNINGS * away_hits_adj
            inning_hits = inning_three_way_probs(home_hits_per_inning, away_hits_per_inning)
            inning_hits_best = best_of_three(
                f"{_ordinal(inning)} Inning Most Hits", proj["home_team"], proj["away_team"], inning_hits
            )

        model_signal = calc_model_signal(total_pace, inning_winner_best, inning_hits_best)

        live_games.append({
            "game_pk": game_pk,
            "home_team": proj["home_team"],
            "away_team": proj["away_team"],
            "home_runs": home_runs,
            "away_runs": away_runs,
            "home_hits": home_hits,
            "away_hits": away_hits,
            "inning": inning,
            "inning_state": inning_state,
            "status": game.get("status", {}).get("detailedState", "Live"),
            "half_innings_remaining": round(remaining_innings * 2) if remaining_innings else 0,
            "next_run": next_run,
            "game_total_pace": total_pace,
            "inning_winner": inning_winner,
            "inning_hits": inning_hits,
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
