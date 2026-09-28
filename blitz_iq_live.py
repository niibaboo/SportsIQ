#!/usr/bin/env python3
"""
Blitz IQ Live Signals — live in-game score-pace signals for NFL games in
progress, same "Model Signal" pattern as Match IQ Live (a green-dot signal
that only qualifies once its probability clears 70%, same bar as the
suite's Daily Signals scanners).

Reuses blitz_iq.py's existing season-form model (last-5-games scored/allowed
rates, shrunk toward league average) as the baseline, then adjusts each
team's remaining expected points for how their actual scoring pace THIS
game compares to what that model expected by now — same idea as Match IQ
Live's live-xG adjustment. Unlike the soccer version, ESPN's scoreboard
already carries score, quarter and clock directly on the same response used
for the main model's weekly scoreboard fetch, so no extra live-stats API
call is needed per game.

Two live signals per game:
  - Next Score: which team is more likely to score next, given time
    remaining and each team's pace-adjusted remaining scoring rate.
  - Game Total Pace: whether the game's combined score is tracking clearly
    above/below the model's pregame total projection.

Setup:
    python3 blitz_iq_live.py
    (blitz_iq.py must be importable — same folder / on PYTHONPATH — this
    script reuses its team-form fetching and pregame projection model
    rather than duplicating them)

Output:
    docs/blitz-iq-live/live_games.json (consumed by live.html every 60s)
"""

import os
import json
import math
from datetime import datetime, timezone

import blitz_iq  # reuse the existing season-form model rather than duplicate it

MODEL_SIGNAL_THRESHOLD = 70  # same bar as Match IQ Live / Daily Signals scanners
GAME_MINUTES = 60  # regulation; overtime is handled as bonus time, see elapsed_minutes()


def elapsed_minutes(status):
    """ESPN's scoreboard gives 'period' (quarter: 1-4, 5+ = OT) and
    'displayClock' (mm:ss REMAINING in that period) directly on the same
    response blitz_iq.get_week_scoreboard() already fetches — no extra
    per-game API call needed, unlike Match IQ Live's separate live-stats
    endpoint."""
    period = status.get('period')
    clock = status.get('displayClock', '15:00')
    if period is None:
        return None
    try:
        mins, secs = clock.split(':')
        remaining_in_period = int(mins) + int(secs) / 60.0
    except (ValueError, AttributeError):
        remaining_in_period = 15.0

    if period <= 4:
        return (period - 1) * 15 + (15 - remaining_in_period)
    # Overtime: 10-minute periods (regular season). Treated as bonus time
    # past the 60-minute regulation mark — pace projections get rougher
    # this far out, but still directionally useful.
    ot_period_num = period - 4
    return 60 + (ot_period_num - 1) * 10 + (10 - remaining_in_period)


def calc_pace_adjustment(actual_scored, exp_scored_pregame, elapsed_fraction):
    """How a team's ACTUAL scoring so far this game compares to what their
    pregame projection implied by this point — same idea as Match IQ
    Live's live-xG adjustment. Capped so a small early-game sample (e.g. one
    early touchdown) doesn't swing the remaining-time projection to an
    extreme; a 3-point floor on "expected by now" avoids dividing by a
    near-zero expectation in the opening minutes."""
    expected_by_now = exp_scored_pregame * elapsed_fraction if elapsed_fraction > 0 else 0.1
    adjustment = actual_scored / max(expected_by_now, 3.0)
    return min(max(adjustment, 0.5), 2.0)


def calc_next_score_prob(home_exp_remaining, away_exp_remaining):
    """P(each team scores next) ≈ ratio of their pace-adjusted remaining
    expected points — same ratio approach as Match IQ Live's next-goal
    calculation, adapted from goals to points."""
    total = home_exp_remaining + away_exp_remaining
    if total <= 0:
        return {"home": 50, "away": 50}
    return {
        "home": round((home_exp_remaining / total) * 100),
        "away": round((away_exp_remaining / total) * 100),
    }


def calc_game_total_pace(actual_total, exp_total_pregame, total_std, minutes_remaining):
    """Probability the final combined score clears the model's PREGAME total
    projection, given what's already on the board plus the remaining time's
    share of that projection (variance scaled down with time remaining).
    Mirrors calc_btts_prob's remaining-time-aware structure from Match IQ
    Live, applied to a live "game total" market instead of BTTS."""
    if minutes_remaining <= 0:
        prob_over = 100 if actual_total > exp_total_pregame else 0
        status = "Final: Over pregame total" if prob_over else "Final: Under pregame total"
        return {"prob": prob_over, "status": status}

    remaining_fraction = minutes_remaining / GAME_MINUTES
    remaining_std = total_std * math.sqrt(max(remaining_fraction, 0.05))
    remaining_mean = exp_total_pregame * remaining_fraction
    needed = exp_total_pregame - actual_total
    prob_over = round((1 - blitz_iq.norm_cdf(needed, remaining_mean, remaining_std)) * 100)
    status = "On Pace: Over" if prob_over >= 50 else "On Pace: Under"
    return {"prob": prob_over, "status": status}


def calc_model_signal(next_score, total_pace):
    """Pick the single highest-confidence qualifying signal, same convention
    as Match IQ Live: only flagged (green dot) once it clears
    MODEL_SIGNAL_THRESHOLD (70%). For the Game Total candidate, confidence
    is in whichever direction (Over/Under) the pace already points, not
    raw prob_over."""
    total_direction = total_pace["status"].split(": ")[-1] if ": " in total_pace["status"] else total_pace["status"]
    total_confidence = total_pace["prob"] if "Over" in total_direction else (100 - total_pace["prob"])

    candidates = [
        {"label": "Home Next Score", "prob": next_score["home"]},
        {"label": "Away Next Score", "prob": next_score["away"]},
        {"label": f"Game Total {total_direction}", "prob": total_confidence},
    ]
    best = max(candidates, key=lambda c: c["prob"])
    return {
        "label": best["label"],
        "prob": best["prob"],
        "qualifies": best["prob"] >= MODEL_SIGNAL_THRESHOLD,
    }


def build_live_signals():
    """Fetch live NFL games, calculate next-score and game-total-pace
    signals, return list of live game objects."""
    print("Fetching teams & season form (same model as blitz_iq.py)...")
    teams = blitz_iq.get_teams()
    if not teams:
        print("No teams returned — aborting.")
        return []

    all_forms = {}
    for t in teams:
        tid = t['team']['id']
        all_forms[tid] = blitz_iq.get_team_form(tid)
    lg_scored, lg_allowed = blitz_iq.league_averages(all_forms.values())

    print("Fetching scoreboard for live games...")
    events = blitz_iq.get_week_scoreboard()
    live_games = []

    for e in events:
        comp = e.get('competitions', [{}])[0]
        status = comp.get('status', {})
        state = status.get('type', {}).get('state')
        if state != 'in':  # only in-progress games
            continue

        competitors = comp.get('competitors', [])
        home = next((c for c in competitors if c.get('homeAway') == 'home'), None)
        away = next((c for c in competitors if c.get('homeAway') == 'away'), None)
        if not home or not away:
            continue

        h_id, a_id = home['team']['id'], away['team']['id']
        h_form, a_form = all_forms.get(h_id), all_forms.get(a_id)
        if not h_form or not a_form:
            print(f"  Skipping {away['team']['displayName']} @ {home['team']['displayName']} — missing form data")
            continue

        try:
            home_score = float(home.get('score', 0) or 0)
            away_score = float(away.get('score', 0) or 0)
        except (TypeError, ValueError):
            home_score, away_score = 0.0, 0.0

        elapsed = elapsed_minutes(status)
        if elapsed is None:
            print(f"  Skipping {away['team']['displayName']} @ {home['team']['displayName']} — no clock data")
            continue
        minutes_left = max(0, GAME_MINUTES - elapsed)
        elapsed_fraction = min(elapsed / GAME_MINUTES, 1.0) if elapsed > 0 else 0.0

        proj = blitz_iq.predict(h_form, a_form, lg_scored, lg_allowed)

        home_adj = calc_pace_adjustment(home_score, proj['exp_home'], elapsed_fraction)
        away_adj = calc_pace_adjustment(away_score, proj['exp_away'], elapsed_fraction)

        home_exp_remaining = proj['exp_home'] * (minutes_left / GAME_MINUTES) * home_adj
        away_exp_remaining = proj['exp_away'] * (minutes_left / GAME_MINUTES) * away_adj

        next_score = calc_next_score_prob(home_exp_remaining, away_exp_remaining)
        total_pace = calc_game_total_pace(home_score + away_score, proj['exp_total'], proj['total_std'], minutes_left)
        model_signal = calc_model_signal(next_score, total_pace)

        live_games.append({
            "game_id": e.get('id'),
            "home_team": home['team']['displayName'],
            "away_team": away['team']['displayName'],
            "home_score": home_score,
            "away_score": away_score,
            "period": status.get('period'),
            "display_clock": status.get('displayClock'),
            "status": status.get('type', {}).get('shortDetail', 'Live'),
            "elapsed_minute": round(elapsed, 1),
            "minutes_remaining": round(minutes_left, 1),
            "next_score": next_score,
            "game_total_pace": total_pace,
            "model_signal": model_signal,
            "pregame_projection": {
                "exp_home": proj['exp_home'],
                "exp_away": proj['exp_away'],
                "exp_total": proj['exp_total'],
            },
        })

    return live_games


if __name__ == "__main__":
    live = build_live_signals()

    os.makedirs("docs/blitz-iq-live", exist_ok=True)

    output = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "total_live": len(live),
        "games": live,
    }

    with open("docs/blitz-iq-live/live_games.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nDone — {len(live)} live game(s) found and written to live_games.json")
