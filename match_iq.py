#!/usr/bin/env python3
"""
Match IQ — built on TheStatsAPI (api.thestatsapi.com)
Same architecture as Strike Zone / Goal IQ: recency-weighted form,
small-sample shrinkage toward league average, Poisson probability. This
one adds real shots / shots-on-target / corners / goalkeeper-saves data,
which football-data.org doesn't expose at any free tier.

IMPORTANT — API call budget:
Getting shots/SoT/corners/saves for a team's last N games (RECENT_GAMES
below, currently 7) requires ONE extra call per game (the /stats
endpoint isn't included in the base match object), so each team costs
~8 calls (1 to list recent matches + 7 for their stats) on top of the
per-league calls. Trial accounts are metered at 10% of the Starter
plan's limits — with 7 leagues × ~20 teams each, a full run could
easily need 1,100+ calls, which is very likely to exceed a trial's
budget. START WITH 1-2 LEAGUES (see LEAGUES below) to confirm this
works before scaling up. Raising RECENT_GAMES further scales this
cost linearly — each extra game per team is ~20 extra calls per
league (1 per team) across a full run.

Setup:
    pip3 install requests --break-system-packages
    python3 match_iq.py YOUR_API_KEY

Output:
    docs/match_iq_index.html, docs/match_iq_predictions.csv
"""

import os
import sys
import time
import math
import csv
import json
from datetime import datetime, timedelta, timezone
import requests

BASE = "https://api.thestatsapi.com/api"
RECENT_GAMES = 7
PRIOR_STRENGTH = 3
MIN_OVER25_PCT = 65  # only keep fixtures with Over 2.5 probability above this...
MIN_BTTS_PCT = 55    # ...AND BTTS probability above this (both required — Over 2.5
                      # is the primary filter, BTTS lowered to a softer bar so it
                      # doesn't knock out otherwise-good Over 2.5 picks)
                      # This pair only gates the MAIN Match IQ page — see
                      # build_all_predictions()/apply_main_filter() below.

# Daily Signals scanners — each applied independently to the FULL unfiltered
# fixture list (not the main-page-filtered one above), so e.g. a low-scoring
# but corner-heavy match can still qualify for the corners scanner even
# though it'd never pass the main page's Over 2.5 + BTTS filter.
SCANNER_OVER25_MIN = 70
SCANNER_BTTS_MIN = 70
SCANNER_CORNERS_MIN = 60
SCANNER_FH_BTTS_MIN = 60
SCANNER_OVER45_MIN = 50

# Start small — uncomment more once you've confirmed the API-call budget
# works for your trial quota. Names must match exactly what TheStatsAPI's
# /football/competitions search returns (verified at runtime, not guessed).
LEAGUE_SEARCH_NAMES = [
    "Premier League",
    "Championship",  # test addition — confirming TheStatsAPI's exact name match
                      # and full stats coverage (shots/corners/cards) before
                      # keeping it long-term; see find_competition()'s "couldn't
                      # find" warning if this name doesn't match what the API
                      # actually calls it
    "Bundesliga",
    "Eredivisie",
    "Primeira Liga",
    "La Liga",
    # "Ligue 1",  # de-listed — found unreliable
    "UEFA Champions League",
    "MLS",
]

FIXTURE_WINDOW_DAYS = 10


def _headers(key):
    return {"Authorization": f"Bearer {key}"}


def _get(path, key, params=None, timeout=15):
    """Every call goes through here. Rate limiting is ADAPTIVE — reads the
    real X-RateLimit-Remaining/Reset headers TheStatsAPI returns and backs
    off only when actually close to the limit, rather than guessing a fixed
    delay (we don't know the trial's exact per-minute number, so a fixed
    guess would either waste time being too conservative or risk 429s
    being too aggressive)."""
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

    time.sleep(2.0)  # baseline pacing — the earlier 0.3s was too fast for the trial's
                      # actual per-minute limit and caused repeated 429s despite the
                      # adaptive top-up above; this spaces every call out enough to
                      # avoid hitting the ceiling in the first place, rather than
                      # reacting to it after the fact
    return r.json()


def poisson_pmf(k, lam):
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def poisson_cdf(k, lam):
    return sum(poisson_pmf(i, lam) for i in range(k + 1))


def find_competition(name, key):
    data = _get("/football/competitions", key, params={"search": name, "per_page": 5})
    if not data:
        return None
    for c in data.get("data", []):
        if c["name"].lower() == name.lower():
            return c
    return data["data"][0] if data.get("data") else None


def get_standings(competition_id, season_id, key):
    data = _get(f"/football/competitions/{competition_id}/seasons/{season_id}/standings", key)
    if not data:
        return {}
    return {row["team"]["id"]: row for row in data.get("data", [])}


def league_averages(standings):
    scored = [r["goals_for"] / r["matches_played"] for r in standings.values() if r.get("matches_played")]
    conceded = [r["goals_against"] / r["matches_played"] for r in standings.values() if r.get("matches_played")]
    lg_scored = sum(scored) / len(scored) if scored else 1.4
    lg_conceded = sum(conceded) / len(conceded) if conceded else 1.4
    return lg_scored, lg_conceded


def get_upcoming_matches(competition_id, season_id, key):
    date_from = datetime.now(timezone.utc).date().isoformat()
    date_to = (datetime.now(timezone.utc).date() + timedelta(days=FIXTURE_WINDOW_DAYS)).isoformat()
    data = _get("/football/matches", key, params={
        "competition_id": competition_id, "season_id": season_id,
        "status": "scheduled", "date_from": date_from, "date_to": date_to,
        "per_page": 20,
    })
    return data.get("data", []) if data else []


team_form_cache = {}


def get_team_form(team_id, competition_id, season_id, key):
    """Last N finished matches for a team, WITH per-match shots/SoT/
    corners/saves — this is the part that costs extra calls (one /stats
    call per game) beyond what goal_iq_fixed.py needs."""
    cache_key = (team_id, competition_id, season_id)
    if cache_key in team_form_cache:
        return team_form_cache[cache_key]

    data = _get("/football/matches", key, params={
        "team_id": team_id, "competition_id": competition_id, "season_id": season_id,
        "status": "finished", "per_page": 10,
    })
    if not data or not data.get("data"):
        team_form_cache[cache_key] = None
        return None

    matches = sorted(data["data"], key=lambda m: m["utc_date"], reverse=True)[:RECENT_GAMES]
    if not matches:
        team_form_cache[cache_key] = None
        return None

    scored, conceded = [], []
    shots, shots_on_target, corners, saves = [], [], [], []
    fh_corners, tackles = [], []
    fh_goals = []  # first-half goals scored, per match — same "field name
                    # unverified against a real API response" caveat as the
                    # other period-split stats below (corner_kicks/first_half
                    # is confirmed working; goals/first_half is assumed to
                    # follow the same overview-stats shape but hasn't been
                    # checked against a live response)
    cards = []  # yellow + red combined, per match — field names unverified against
                # a real API response, same caveat as the other stats below

    for m in matches:
        is_home = m["home_team"]["id"] == team_id
        s = m["score"]
        if s.get("home") is None:
            continue
        scored.append(s["home"] if is_home else s["away"])
        conceded.append(s["away"] if is_home else s["home"])

        stats = _get(f"/football/matches/{m['id']}/stats", key)
        if not stats or not stats.get("data"):
            continue
        ov = stats["data"].get("overview", {})

        def side_val(stat_key, period="all"):
            item = ov.get(stat_key, {}).get(period)
            if not item:
                return None
            return item["home"] if is_home else item["away"]

        for lst, key_name in [(shots, "total_shots"), (shots_on_target, "shots_on_target"),
                               (corners, "corner_kicks"), (saves, "goalkeeper_saves"),
                               (tackles, "tackles")]:
            v = side_val(key_name)
            if v is not None:
                lst.append(v)

        fh_v = side_val("corner_kicks", "first_half")
        if fh_v is not None:
            fh_corners.append(fh_v)

        fh_g = side_val("goals", "first_half")
        if fh_g is not None:
            fh_goals.append(fh_g)

        yellow = side_val("yellow_cards")
        red = side_val("red_cards")
        if yellow is not None or red is not None:
            cards.append((yellow or 0) + (red or 0))

    if not scored:
        team_form_cache[cache_key] = None
        return None

    n = len(scored)
    form = {
        "n_games": n,
        "avg_scored": round(sum(scored) / n, 2),
        "avg_conceded": round(sum(conceded) / n, 2),
        "goals_list": scored,
        "avg_shots": round(sum(shots) / len(shots), 1) if shots else None,
        "shots_list": shots,
        "avg_shots_on_target": round(sum(shots_on_target) / len(shots_on_target), 1) if shots_on_target else None,
        "sot_list": shots_on_target,
        "avg_corners": round(sum(corners) / len(corners), 1) if corners else None,
        "corners_list": corners,
        "avg_saves": round(sum(saves) / len(saves), 1) if saves else None,
        "saves_list": saves,
        "avg_fh_corners": round(sum(fh_corners) / len(fh_corners), 1) if fh_corners else None,
        "fh_corners_list": fh_corners,
        "avg_fh_goals": round(sum(fh_goals) / len(fh_goals), 2) if fh_goals else None,
        "fh_goals_list": fh_goals,
        "avg_tackles": round(sum(tackles) / len(tackles), 1) if tackles else None,
        "avg_cards": round(sum(cards) / len(cards), 1) if cards else None,
        "cards_list": cards,
    }
    team_form_cache[cache_key] = form
    return form


def shrink(value, n, league_avg, prior=PRIOR_STRENGTH):
    """Same small-sample protection as every other tool in this set — a
    1-2 game sample leans mostly on the league average; by RECENT_GAMES games the
    team's own form dominates."""
    if value is None:
        return league_avg
    return round((n * value + prior * league_avg) / (n + prior), 2)


def predict_goals(h_form, a_form, lg_scored, lg_conceded):
    h_scored = shrink(h_form["avg_scored"], h_form["n_games"], lg_scored)
    h_conceded = shrink(h_form["avg_conceded"], h_form["n_games"], lg_conceded)
    a_scored = shrink(a_form["avg_scored"], a_form["n_games"], lg_scored)
    a_conceded = shrink(a_form["avg_conceded"], a_form["n_games"], lg_conceded)

    exp_home = h_scored * (a_conceded / lg_conceded)
    exp_away = a_scored * (h_conceded / lg_conceded)
    exp_total = round(exp_home + exp_away, 2)

    p_over25 = 1 - poisson_cdf(2, exp_total)
    p_over45 = 1 - poisson_cdf(4, exp_total)
    p_btts = (1 - poisson_pmf(0, exp_home)) * (1 - poisson_pmf(0, exp_away))

    return {
        "exp_home": round(exp_home, 2), "exp_away": round(exp_away, 2), "exp_total": exp_total,
        "over25": round(p_over25 * 100), "over45": round(p_over45 * 100), "btts": round(p_btts * 100),
    }


# Running average of every team's FH-corners rate seen so far in this run —
# used as the shrinkage prior below, since (unlike goals) there's no
# standings-based league-average source for corners. This converges as more
# teams get processed; early predictions in the run lean on a smaller,
# slightly less stable sample than later ones. A documented approximation,
# not a precise league average.
_fh_corners_samples = []
DEFAULT_FH_CORNERS_AVG = 2.5  # sane starting point before any real samples exist


def _lg_fh_corners_avg():
    if not _fh_corners_samples:
        return DEFAULT_FH_CORNERS_AVG
    return sum(_fh_corners_samples) / len(_fh_corners_samples)


def _record_fh_corners_sample(form):
    if form.get("avg_fh_corners") is not None:
        _fh_corners_samples.append(form["avg_fh_corners"])


def predict_1x2(exp_home, exp_away, max_goals=8):
    """Home Win / Draw / Away Win — built from the same exp_home/exp_away
    the goals model already computes, just summed differently: instead of
    combining both sides into one total, this splits the same independent-
    Poisson scoreline grid by which side has more goals. Same independence
    assumption as BTTS/correct-score elsewhere (doesn't model the slight
    real-world correlation between the two teams' scoring, e.g. a team
    already up 2-0 easing off)."""
    p_home_win = p_draw = p_away_win = 0.0
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p = poisson_pmf(h, exp_home) * poisson_pmf(a, exp_away)
            if h > a:
                p_home_win += p
            elif h == a:
                p_draw += p
            else:
                p_away_win += p
    return {
        "home_win_pct": round(p_home_win * 100),
        "draw_pct": round(p_draw * 100),
        "away_win_pct": round(p_away_win * 100),
    }


def predict_fh_corners(h_form, a_form):
    """FH corners Over/Under — no opponent adjustment (we don't track
    corners CONCEDED, only corners WON, so there's no equivalent of the
    goals model's 'weak defense' factor here). Each team's own recent FH
    corner-winning rate, shrunk toward a running league average, summed
    into a Poisson-based total."""
    _record_fh_corners_sample(h_form)
    _record_fh_corners_sample(a_form)
    lg_avg = _lg_fh_corners_avg()

    h_fh = shrink(h_form.get("avg_fh_corners"), h_form["n_games"], lg_avg)
    a_fh = shrink(a_form.get("avg_fh_corners"), a_form["n_games"], lg_avg)
    exp_fh_total = round(h_fh + a_fh, 2)

    p_over35 = 1 - poisson_cdf(3, exp_fh_total)
    p_over45 = 1 - poisson_cdf(4, exp_fh_total)

    return {
        "exp_fh_corners": exp_fh_total,
        "fh_corners_over35": round(p_over35 * 100),
        "fh_corners_over45": round(p_over45 * 100),
    }


_fh_goals_samples = []
DEFAULT_FH_GOALS_AVG = 0.65  # sane per-team starting point (~1.3 total FH goals/match) before any real samples exist


def _lg_fh_goals_avg():
    if not _fh_goals_samples:
        return DEFAULT_FH_GOALS_AVG
    return sum(_fh_goals_samples) / len(_fh_goals_samples)


def _record_fh_goals_sample(form):
    if form.get("avg_fh_goals") is not None:
        _fh_goals_samples.append(form["avg_fh_goals"])


def predict_fh_btts(h_form, a_form):
    """First-half BTTS — same running-average shrinkage as FH corners (no
    opponent adjustment; each team's own FH scoring rate, not split by who
    they faced). Relies on the goals/first_half stat existing in the API
    response — see the fh_goals extraction note in fetch_form for the
    verification caveat."""
    _record_fh_goals_sample(h_form)
    _record_fh_goals_sample(a_form)
    lg_avg = _lg_fh_goals_avg()

    h_fh = shrink(h_form.get("avg_fh_goals"), h_form["n_games"], lg_avg)
    a_fh = shrink(a_form.get("avg_fh_goals"), a_form["n_games"], lg_avg)

    p_fh_btts = (1 - poisson_pmf(0, h_fh)) * (1 - poisson_pmf(0, a_fh))

    return {
        "exp_fh_home": h_fh, "exp_fh_away": a_fh,
        "fh_btts": round(p_fh_btts * 100),
    }


_full_corners_samples = []
DEFAULT_FULL_CORNERS_AVG = 5.0  # sane per-team starting point (~10 total/match) before any real samples exist


def _lg_full_corners_avg():
    if not _full_corners_samples:
        return DEFAULT_FULL_CORNERS_AVG
    return sum(_full_corners_samples) / len(_full_corners_samples)


def _record_full_corners_sample(form):
    if form.get("avg_corners") is not None:
        _full_corners_samples.append(form["avg_corners"])


def predict_full_corners(h_form, a_form):
    """Full-match total corners Over 10.5 — same running-average shrinkage
    approach as predict_fh_corners (no corners-conceded data to build a
    proper opponent adjustment from), just using each team's full-match
    corners_list instead of the first-half-only one."""
    _record_full_corners_sample(h_form)
    _record_full_corners_sample(a_form)
    lg_avg = _lg_full_corners_avg()

    h_c = shrink(h_form.get("avg_corners"), h_form["n_games"], lg_avg)
    a_c = shrink(a_form.get("avg_corners"), a_form["n_games"], lg_avg)
    exp_corners_total = round(h_c + a_c, 2)

    p_over105 = 1 - poisson_cdf(10, exp_corners_total)

    return {
        "exp_corners": exp_corners_total,
        "corners_over105": round(p_over105 * 100),
    }


def predict_team_props(form):
    """Poisson-priced shots / shots-on-target / corners / cards
    probabilities for one team.

    Shots/SoT/corners use a line set to 72% of the recent-form average,
    rounded down to the nearest 0.5 — a real safety margin below a
    double-digit average (the same "wide cushion" reasoning used
    manually when picking bet-builder legs), turned into an actual
    probability instead of a by-eye judgment.

    Cards get a DIFFERENT, fixed line — "Over 0.5" (i.e. at least one
    booking) — rather than the same 72%-of-average approach. Cards
    averages are low (typically 1-3 per team), so scaling down by 72%
    and rounding to the nearest 0.5 was landing on lines like "Over 1.0"
    (needing 2+ cards), which is a meaningfully harder — and much less
    safe — bet than the natural "team gets booked at least once" market
    most bet builders actually offer. At low counts the generic
    safety-margin logic doesn't transfer; a fixed low bar suits this
    market better."""
    def prop(avg, factor=0.72):
        if avg is None:
            return None
        raw_line = avg * factor
        line = math.floor(raw_line * 2) / 2  # round down to nearest 0.5
        if line < 0.5:
            line = 0.5
        threshold = int(math.floor(line)) + 1
        prob = 1 - poisson_cdf(threshold - 1, avg)
        return {"line": line, "prob": round(prob * 100), "avg": avg}

    def at_least_one(avg):
        if avg is None:
            return None
        prob = 1 - poisson_pmf(0, avg)
        return {"line": 0.5, "prob": round(prob * 100), "avg": avg}

    return {
        "shots_prop": prop(form.get("avg_shots")),
        "sot_prop": prop(form.get("avg_shots_on_target")),
        "corners_prop": prop(form.get("avg_corners")),
        "cards_prop": at_least_one(form.get("avg_cards")),
    }


def build_legs(predictions):
    """Flatten every match's probability-priced markets into one list of
    individual bet-builder legs, for the safest-combo builder in the HTML.
    Only includes markets the model actually assigns a probability to —
    shots/SoT/corners props use the line from predict_team_props, not a
    fixed number, since that line is already chosen for a safety margin.
    Each leg is tagged with a "category" so the builder can mix market
    types instead of picking whichever single market happens to be
    safest across the board — shots props tend to dominate on raw
    probability, which crowds out goals/corners legs unless the builder
    deliberately rotates through categories."""
    legs = []
    for p in predictions:
        match_label = f"{p['home_team']} v {p['away_team']}"
        hf, af = p["home_form"], p["away_form"]
        h_n, a_n = hf["n_games"], af["n_games"]

        h_goals_hist = format_history(hf.get("goals_list"))
        a_goals_hist = format_history(af.get("goals_list"))
        legs.append({
            "match": match_label, "market": "Over 2.5 Goals", "prob": p["over25"], "category": "Goals",
            "detail": f"{p['exp_total']} exp goals ({h_n}v{a_n}gm)",
            "history": f"H {h_goals_hist} · A {a_goals_hist}" if h_goals_hist and a_goals_hist else None,
        })
        legs.append({
            "match": match_label, "market": "BTTS", "prob": p["btts"], "category": "BTTS",
            "detail": f"{p['exp_total']} exp goals ({h_n}v{a_n}gm)",
            "history": f"H {h_goals_hist} · A {a_goals_hist}" if h_goals_hist and a_goals_hist else None,
        })

        h_fh_hist = format_history(hf.get("fh_corners_list"))
        a_fh_hist = format_history(af.get("fh_corners_list"))
        legs.append({
            "match": match_label, "market": "FH Corners Over 3.5", "prob": p["fh_corners_over35"], "category": "FH Corners",
            "detail": f"{p['exp_fh_corners']} exp FH corners ({h_n}v{a_n}gm)",
            "history": f"H {h_fh_hist} · A {a_fh_hist}" if h_fh_hist and a_fh_hist else None,
        })

        for team_name, form, props, n_games in [
            (p["home_team"], hf, p.get("home_props") or {}, h_n),
            (p["away_team"], af, p.get("away_props") or {}, a_n),
        ]:
            for market_key, list_key, label in [
                ("shots_prop", "shots_list", "Shots"),
                ("sot_prop", "sot_list", "Shots on Target"),
                ("corners_prop", "corners_list", "Corners"),
                ("cards_prop", "cards_list", "Cards"),
            ]:
                prop = props.get(market_key)
                if prop:
                    legs.append({
                        "match": match_label,
                        "market": f"{team_name} Over {prop['line']} {label}",
                        "prob": prop["prob"],
                        "category": label,
                        "detail": f"avg {prop['avg']} ({n_games}gm)",
                        "history": format_history(form.get(list_key)),
                    })
    return legs


def build_all_predictions(key):
    all_predictions = []
    for name in LEAGUE_SEARCH_NAMES:
        print(f"Looking up competition: {name}")
        comp = find_competition(name, key)
        if not comp:
            print(f"  [!] couldn't find competition '{name}' — skipping")
            continue
        comp_id = comp["id"]
        season_id = comp.get("current_season_id")
        if not season_id:
            details = _get(f"/football/competitions/{comp_id}", key)
            season_id = details["data"].get("current_season_id") if details else None
        if not season_id:
            print(f"  [!] no current season for {name} — skipping")
            continue

        standings = get_standings(comp_id, season_id, key)
        lg_scored, lg_conceded = league_averages(standings)
        print(f"  League averages: {lg_scored:.2f} scored/gm, {lg_conceded:.2f} conceded/gm")

        matches = get_upcoming_matches(comp_id, season_id, key)
        print(f"  {len(matches)} upcoming matches in window")

        for m in matches:
            h_id, a_id = m["home_team"]["id"], m["away_team"]["id"]
            print(f"    {m['home_team']['name']} vs {m['away_team']['name']}")
            h_form = get_team_form(h_id, comp_id, season_id, key)
            a_form = get_team_form(a_id, comp_id, season_id, key)
            if not h_form or not a_form:
                print(f"      skipping — missing form data (home={bool(h_form)}, away={bool(a_form)})")
                continue

            proj = predict_goals(h_form, a_form, lg_scored, lg_conceded)
            fh_corners_proj = predict_fh_corners(h_form, a_form)
            full_corners_proj = predict_full_corners(h_form, a_form)
            fh_btts_proj = predict_fh_btts(h_form, a_form)
            result_1x2 = predict_1x2(proj["exp_home"], proj["exp_away"])
            merged = {
                "league": comp["name"], "date": m["utc_date"], "date_key": m["utc_date"][:10],
                "home_team": m["home_team"]["name"], "away_team": m["away_team"]["name"],
                "home_form": h_form, "away_form": a_form,
                "home_props": predict_team_props(h_form), "away_props": predict_team_props(a_form),
                **proj, **fh_corners_proj, **full_corners_proj, **fh_btts_proj, **result_1x2,
            }
            all_predictions.append(merged)

    # Sort by date first (so grouping into date-pages is clean), then by
    # expected goals within each date (so the most interesting fixtures
    # still show first within a given day's page).
    all_predictions.sort(key=lambda p: (p["date_key"], -p["exp_total"]))
    return all_predictions


def apply_main_filter(all_predictions):
    """The MAIN Match IQ page's filter — unchanged from before. Kept as a
    separate step (rather than baked into build_all_predictions) so the
    Daily Signals scanners below can each apply their own independent
    threshold to the full, unfiltered fixture list instead of inheriting
    this one."""
    before = len(all_predictions)
    filtered = [
        p for p in all_predictions
        if p["over25"] > MIN_OVER25_PCT and p["btts"] > MIN_BTTS_PCT
    ]
    print(f"\nMain page filter — Over 2.5 > {MIN_OVER25_PCT}% AND BTTS > {MIN_BTTS_PCT}%: "
          f"{len(filtered)} of {before} fixtures kept")
    return filtered


def group_by_date(predictions):
    by_date = {}
    for p in predictions:
        by_date.setdefault(p["date_key"], []).append(p)
    return dict(sorted(by_date.items()))


def date_page_filename(date_key):
    return f"match_iq_day-{date_key}.html"


def format_date_label(date_key):
    try:
        return datetime.strptime(date_key, "%Y-%m-%d").strftime("%a %d %b")
    except Exception:
        return date_key


BUILDER_TEMPLATE = """
<div style="background:#1a1f26;border-radius:12px;padding:16px;margin:12px 0;border:1px solid #2a3038">
  <div style="font-size:14px;font-weight:bold;margin-bottom:10px">🎯 Safest Bet Builder</div>
  <div id="categoryToggles" style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:10px;font-size:12px"></div>
  <div style="display:flex;gap:8px;align-items:center;margin-bottom:6px;flex-wrap:wrap">
    <label style="font-size:12px;color:#aaa">Target odds:</label>
    <input id="targetOdds" type="number" step="0.1" min="1.1" value="5.0"
      style="width:70px;background:#0f1318;border:1px solid #333;color:white;border-radius:6px;padding:6px 8px;font-size:13px">
    <label style="font-size:12px;color:#aaa">Max legs:</label>
    <input id="maxLegs" type="number" step="1" min="2" value="8"
      style="width:55px;background:#0f1318;border:1px solid #333;color:white;border-radius:6px;padding:6px 8px;font-size:13px">
    <button onclick="buildSafest()"
      style="background:#3a7d7a;border:none;color:white;padding:7px 14px;border-radius:6px;font-size:13px;cursor:pointer">
      Build
    </button>
    <button onclick="buildSafest()"
      style="background:#2a3038;border:1px solid #444;color:white;padding:7px 14px;border-radius:6px;font-size:13px;cursor:pointer">
      🔀 Shuffle
    </button>
  </div>
  <div id="builderResult" style="font-size:12px;color:#888">
    Untick any market type you don't want considered (Corners starts unticked —
    lower counts mean more relative variance than shots/SoT), set a target odds
    and leg cap, then tap Build. It rotates through whichever categories are
    ticked, groups near-tied legs and shuffles within each group so it draws
    from more of the day's matches rather than always the exact same few, and
    caps at 2 legs per match to avoid stacking correlated legs from one game.
    Tap Shuffle for a fresh pick among equally-safe options without changing
    your settings.
  </div>
</div>
<script>
const LEGS = {legs_json};

function initCategoryToggles() {{
  const container = document.getElementById('categoryToggles');
  const cats = [...new Set(LEGS.map(l => l.category))];
  container.innerHTML = cats.map(c => `
    <label style="display:flex;align-items:center;gap:4px;color:#ccc;cursor:pointer">
      <input type="checkbox" class="catToggle" value="${{c}}" ${{c === 'Corners' ? '' : 'checked'}}>
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

// Sorts safest-first at a coarse level (5-point probability bands) but
// shuffles legs WITHIN each band, so e.g. five different legs all sitting
// at 84-88% get picked in a different order each time instead of always
// the same one — this is what actually lets the builder draw from the
// full pool of matches instead of fixating on whichever leg happens to
// be a fraction of a percent ahead.
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

  // Group legs by category, tiered-shuffled within each, so the builder can
  // round-robin across market types instead of exhausting one category
  // (usually Shots, since it tends to have the highest raw probabilities)
  // before touching any other — and so it doesn't fixate on the same
  // handful of matches every time when many legs are near-equally safe.
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
        if (count >= 2) continue;  // cap legs per match to limit correlation risk
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
    `<div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #2a3038">
       <span>${{l.match}}<br><span style="color:#7ec8ff">${{l.market}}</span> <span style="color:#555">· ${{l.category}}</span>
       ${{l.detail ? `<br><span style="color:#666;font-size:10px">${{l.detail}}</span>` : ''}}
       ${{l.history ? `<br><span style="color:#555;font-size:10px">last games: ${{l.history}}</span>` : ''}}</span>
       <span style="color:#ffeb3b;font-weight:bold">${{l.prob}}%</span>
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
    <div style="color:#666;font-size:10px;margin-top:8px;line-height:1.4">
      Estimate multiplies each leg's fair odds (100/probability) — real bookmaker
      odds include their margin and legs within the same match aren't fully
      independent, so treat this as a ranking tool, not a firm price. Shots/SoT/
      corners lines are set automatically below each team's recent-form average
      for a safety margin; goals/BTTS/FH-corners use the model's own probabilities.
    </div>
  `;
}}
</script>
"""


HTML_TEMPLATE = """<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Match IQ</title></head>
<body style="background:#0b0f14;color:white;font-family:Arial;padding:12px;max-width:600px;margin:auto">
<h2 style="text-align:center">⚽ MATCH IQ — Full Stats</h2>
<p style="text-align:center;color:#888;font-size:11px">Powered by TheStatsAPI · {generated}</p>
<p style="text-align:center;margin:6px 0 0;font-size:12px">Daily Signals: <a href="scanners/over25/" style="color:#7ec8ff;text-decoration:none;margin:0 4px">Over 2.5</a>·<a href="scanners/btts/" style="color:#7ec8ff;text-decoration:none;margin:0 4px">BTTS</a>·<a href="scanners/corners/" style="color:#7ec8ff;text-decoration:none;margin:0 4px">Corners 10.5+</a>·<a href="scanners/fh-btts-over45/" style="color:#7ec8ff;text-decoration:none;margin:0 4px">FH BTTS/O4.5</a></p>
{date_bar}
<p style="text-align:center;margin-bottom:16px"><a href="match_iq_predictions.csv" download style="background:#222;border:1px solid #444;color:white;padding:8px 14px;border-radius:8px;text-decoration:none;font-size:13px">⬇ Download CSV</a></p>
{builder}
{cards}
</body></html>"""

CARD_TEMPLATE = """<div style="background:#1a1f26;border-radius:12px;padding:16px;margin:12px 0;border:1px solid #2a3038">
  <div style="font-size:11px;color:#999">{league} · {date}</div>
  <div style="font-size:17px;font-weight:bold;margin:6px 0 10px">{home_team} vs {away_team}</div>
  <div style="display:flex;justify-content:space-between;text-align:center;margin-bottom:10px">
    <div><div style="color:#aaa;font-size:11px">EXP TOTAL</div><div style="color:#ffeb3b;font-size:20px;font-weight:bold">{exp_total}</div></div>
    <div><div style="color:#aaa;font-size:11px">OVER 2.5</div><div style="color:#ffeb3b;font-size:20px;font-weight:bold">{over25}%</div></div>
    <div><div style="color:#aaa;font-size:11px">BTTS</div><div style="color:#ffeb3b;font-size:20px;font-weight:bold">{btts}%</div></div>
  </div>
  <div style="display:flex;justify-content:space-between;text-align:center;margin-bottom:10px">
    <div><div style="color:#aaa;font-size:11px">HOME WIN</div><div style="color:#a0e8a0;font-size:18px;font-weight:bold">{home_win_pct}%</div></div>
    <div><div style="color:#aaa;font-size:11px">DRAW</div><div style="color:#a0e8a0;font-size:18px;font-weight:bold">{draw_pct}%</div></div>
    <div><div style="color:#aaa;font-size:11px">AWAY WIN</div><div style="color:#a0e8a0;font-size:18px;font-weight:bold">{away_win_pct}%</div></div>
  </div>
  <div style="display:flex;justify-content:space-between;text-align:center;margin-bottom:10px">
    <div><div style="color:#aaa;font-size:11px">FH CORNERS EXP</div><div style="color:#7ec8ff;font-size:18px;font-weight:bold">{exp_fh_corners}</div></div>
    <div><div style="color:#aaa;font-size:11px">FH OVER 3.5</div><div style="color:#7ec8ff;font-size:18px;font-weight:bold">{fh_corners_over35}%</div></div>
    <div><div style="color:#aaa;font-size:11px">FH OVER 4.5</div><div style="color:#7ec8ff;font-size:18px;font-weight:bold">{fh_corners_over45}%</div></div>
  </div>
  <div style="background:#0f1318;border-radius:8px;padding:8px;font-size:11px">
    <div>{home_team}: {h_goals} goals/gm · shots {h_shots} · SoT {h_sot} · corners {h_corners} (FH {h_fh_corners}) · tackles {h_tackles} · saves {h_saves} · cards {h_cards} ({h_n}gm)</div>
    <div style="margin-top:4px">{away_team}: {a_goals} goals/gm · shots {a_shots} · SoT {a_sot} · corners {a_corners} (FH {a_fh_corners}) · tackles {a_tackles} · saves {a_saves} · cards {a_cards} ({a_n}gm)</div>
  </div>
  <div style="background:#0f1318;border-radius:8px;padding:8px;font-size:10px;color:#888;margin-top:6px;line-height:1.6">
    <div><span style="color:#aaa">{home_team} last games</span> — {h_history}</div>
    <div style="margin-top:3px"><span style="color:#aaa">{away_team} last games</span> — {a_history}</div>
  </div>
</div>"""


def fmt(v):
    return v if v is not None else "—"


def format_history(lst):
    """Render a team's last-N-games list as an oldest→newest string, e.g.
    [4, 5, 3] (stored newest-first) becomes "3/5/4" read left-to-right as
    a trend. Used to show the actual match-by-match numbers behind an
    average, not just the average itself."""
    if not lst:
        return None
    return "/".join(str(v) for v in reversed(lst))


def team_history_line(form):
    """One compact line of last-N-games sequences per stat for a team,
    used under the main prediction card. Skips any stat with no data
    rather than showing an empty "goals —" entry."""
    parts = []
    for label, key in [("goals", "goals_list"), ("shots", "shots_list"),
                        ("SoT", "sot_list"), ("corners", "corners_list"),
                        ("cards", "cards_list")]:
        hist = format_history(form.get(key))
        if hist:
            parts.append(f"{label} {hist}")
    return " · ".join(parts) if parts else "—"


def make_html(predictions, date_label=None, prev_href=None, next_href=None):
    cards = "".join(CARD_TEMPLATE.format(
        league=p["league"], date=p["date"][:16].replace("T", " "),
        home_team=p["home_team"], away_team=p["away_team"],
        exp_total=p["exp_total"], over25=p["over25"], btts=p["btts"],
        home_win_pct=p["home_win_pct"], draw_pct=p["draw_pct"], away_win_pct=p["away_win_pct"],
        exp_fh_corners=p["exp_fh_corners"], fh_corners_over35=p["fh_corners_over35"],
        fh_corners_over45=p["fh_corners_over45"],
        h_goals=p["home_form"]["avg_scored"], h_shots=fmt(p["home_form"]["avg_shots"]),
        h_sot=fmt(p["home_form"]["avg_shots_on_target"]), h_corners=fmt(p["home_form"]["avg_corners"]),
        h_fh_corners=fmt(p["home_form"]["avg_fh_corners"]), h_tackles=fmt(p["home_form"]["avg_tackles"]),
        h_saves=fmt(p["home_form"]["avg_saves"]), h_cards=fmt(p["home_form"].get("avg_cards")), h_n=p["home_form"]["n_games"],
        h_history=team_history_line(p["home_form"]),
        a_goals=p["away_form"]["avg_scored"], a_shots=fmt(p["away_form"]["avg_shots"]),
        a_sot=fmt(p["away_form"]["avg_shots_on_target"]), a_corners=fmt(p["away_form"]["avg_corners"]),
        a_fh_corners=fmt(p["away_form"]["avg_fh_corners"]), a_tackles=fmt(p["away_form"]["avg_tackles"]),
        a_saves=fmt(p["away_form"]["avg_saves"]), a_cards=fmt(p["away_form"].get("avg_cards")), a_n=p["away_form"]["n_games"],
        a_history=team_history_line(p["away_form"]),
    ) for p in predictions)
    if not cards:
        cards = '<p style="text-align:center;color:#888">No fixtures on this date passed the filter.</p>'

    prev_link = f'<a href="{prev_href}" style="color:#7ec8ff;text-decoration:none;font-size:20px">◀</a>' if prev_href else '<span style="color:#444;font-size:20px">◀</span>'
    next_link = f'<a href="{next_href}" style="color:#7ec8ff;text-decoration:none;font-size:20px">▶</a>' if next_href else '<span style="color:#444;font-size:20px">▶</span>'
    date_bar = f"""
<div style="display:flex;align-items:center;justify-content:center;gap:20px;margin:10px 0 4px">
  {prev_link}
  <span style="font-size:15px;font-weight:bold">{date_label or ''}</span>
  {next_link}
</div>""" if date_label else ""

    legs = build_legs(predictions)
    builder = BUILDER_TEMPLATE.format(legs_json=json.dumps(legs)) if legs else ""

    return HTML_TEMPLATE.format(
        generated=datetime.now().strftime("%d %b %H:%M"),
        date_bar=date_bar, builder=builder, cards=cards,
    )


def write_csv(predictions, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Date", "League", "HomeTeam", "AwayTeam", "ExpHome", "ExpAway", "ExpTotal", "Over25", "BTTS",
            "HomeWinPct", "DrawPct", "AwayWinPct",
            "ExpFHCorners", "FHCornersOver35", "FHCornersOver45",
            "HomeGoalsAvg", "HomeShotsAvg", "HomeSoTAvg", "HomeCornersAvg", "HomeFHCornersAvg",
            "HomeTacklesAvg", "HomeSavesAvg", "HomeCardsAvg", "HomeSample",
            "AwayGoalsAvg", "AwayShotsAvg", "AwaySoTAvg", "AwayCornersAvg", "AwayFHCornersAvg",
            "AwayTacklesAvg", "AwaySavesAvg", "AwayCardsAvg", "AwaySample",
            "ActualHomeGoals", "ActualAwayGoals", "ActualFHCorners", "Actual1X2", "HitOrMiss",
        ])
        for p in predictions:
            hf, af = p["home_form"], p["away_form"]
            writer.writerow([
                p["date"], p["league"], p["home_team"], p["away_team"],
                p["exp_home"], p["exp_away"], p["exp_total"], p["over25"], p["btts"],
                p["home_win_pct"], p["draw_pct"], p["away_win_pct"],
                p["exp_fh_corners"], p["fh_corners_over35"], p["fh_corners_over45"],
                hf["avg_scored"], fmt(hf["avg_shots"]), fmt(hf["avg_shots_on_target"]),
                fmt(hf["avg_corners"]), fmt(hf["avg_fh_corners"]), fmt(hf["avg_tackles"]),
                fmt(hf["avg_saves"]), fmt(hf.get("avg_cards")), hf["n_games"],
                af["avg_scored"], fmt(af["avg_shots"]), fmt(af["avg_shots_on_target"]),
                fmt(af["avg_corners"]), fmt(af["avg_fh_corners"]), fmt(af["avg_tackles"]),
                fmt(af["avg_saves"]), fmt(af.get("avg_cards")), af["n_games"],
                "", "", "", "", "",
            ])


SCANNER_CARD_TEMPLATE = """<div style="background:#1a1f26;border-radius:12px;padding:14px;margin:10px 0;border:1px solid #2a3038;display:flex;gap:12px;align-items:flex-start">
  <div style="min-width:72px;text-align:center;background:#0f1318;border:1px solid #2a3038;border-radius:10px;padding:8px 6px;flex-shrink:0">
    <div style="font-size:10px;color:#888">{badge_label}</div>
    <div style="font-size:20px;font-weight:bold;color:#a0e8a0">{badge_value}%</div>
  </div>
  <div style="flex:1;min-width:0">
    <div style="font-size:11px;color:#999">{league} · {time}</div>
    <div style="font-size:15px;font-weight:bold;margin:2px 0 6px">{home_team} vs {away_team}</div>
    <div style="font-size:11px;color:#aaa">O2.5: <span style="color:#a0e8a0">{over25}%</span> &nbsp;|&nbsp; BTTS: <span style="color:#a0e8a0">{btts}%</span> &nbsp;|&nbsp; Corners 10.5+: <span style="color:#a0e8a0">{corners_disp}%</span></div>
  </div>
</div>"""

SCANNER_HTML_TEMPLATE = """<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{page_title} — Match IQ</title></head>
<body style="background:#0b0f14;color:white;font-family:Arial;padding:12px;max-width:600px;margin:auto">
<p style="text-align:center;margin-bottom:6px"><a href="../../match_iq_index.html" style="color:#7ec8ff;text-decoration:none;font-size:12px">← Match IQ</a></p>
<h2 style="text-align:center;margin-bottom:2px">{icon} {page_title}</h2>
<p style="text-align:center;color:#888;font-size:11px;margin-top:0">{subtitle} · {generated}</p>
<p style="text-align:center;margin:8px 0 4px;font-size:12px"><a href="../over25/" style="color:#7ec8ff;text-decoration:none;margin:0 6px">Over 2.5</a>·<a href="../btts/" style="color:#7ec8ff;text-decoration:none;margin:0 6px">BTTS</a>·<a href="../corners/" style="color:#7ec8ff;text-decoration:none;margin:0 6px">Corners 10.5+</a>·<a href="../fh-btts-over45/" style="color:#7ec8ff;text-decoration:none;margin:0 6px">FH BTTS/O4.5</a></p>
{date_bar}
<p style="text-align:center;margin-bottom:12px"><a href="{csv_name}" download style="background:#222;border:1px solid #444;color:white;padding:8px 14px;border-radius:8px;text-decoration:none;font-size:13px">⬇ Export CSV</a></p>
<p style="text-align:center;color:#888;font-size:12px;margin-bottom:14px">{qualified_count} matches qualified</p>
{cards}
</body></html>"""


def _scanner_badge_value(p, market_key):
    return {"over25": p["over25"], "btts": p["btts"], "corners_over105": p["corners_over105"]}[market_key]


def render_scanner_cards(predictions, market_key, badge_label):
    if not predictions:
        return '<p style="text-align:center;color:#888">No fixtures on this date qualified.</p>'
    cards = ""
    for p in predictions:
        cards += SCANNER_CARD_TEMPLATE.format(
            badge_label=badge_label, badge_value=_scanner_badge_value(p, market_key),
            league=p["league"], time=p["date"][:16].replace("T", " "),
            home_team=p["home_team"], away_team=p["away_team"],
            over25=p["over25"], btts=p["btts"], corners_disp=p["corners_over105"],
        )
    return cards


def make_scanner_html(predictions, page_title, icon, subtitle, market_key, badge_label,
                       csv_name, date_label=None, prev_href=None, next_href=None):
    prev_link = f'<a href="{prev_href}" style="color:#7ec8ff;text-decoration:none;font-size:20px">◀</a>' if prev_href else '<span style="color:#444;font-size:20px">◀</span>'
    next_link = f'<a href="{next_href}" style="color:#7ec8ff;text-decoration:none;font-size:20px">▶</a>' if next_href else '<span style="color:#444;font-size:20px">▶</span>'
    date_bar = f"""
<div style="display:flex;align-items:center;justify-content:center;gap:20px;margin:10px 0 4px">
  {prev_link}
  <span style="font-size:15px;font-weight:bold">{date_label or ''}</span>
  {next_link}
</div>""" if date_label else ""

    return SCANNER_HTML_TEMPLATE.format(
        page_title=page_title, icon=icon, subtitle=subtitle,
        generated=datetime.now().strftime("%d %b %H:%M"),
        date_bar=date_bar, csv_name=csv_name,
        qualified_count=len(predictions),
        cards=render_scanner_cards(predictions, market_key, badge_label),
    )


def write_scanner_csv(predictions, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "League", "HomeTeam", "AwayTeam", "Over25", "BTTS",
                          "ExpCorners", "CornersOver105"])
        for p in predictions:
            writer.writerow([p["date"], p["league"], p["home_team"], p["away_team"],
                              p["over25"], p["btts"], p["exp_corners"], p["corners_over105"]])


# --- Combined FH BTTS + Over 4.5 Goals scanner ---------------------------
# Unlike the three single-market scanners above, a fixture here qualifies
# if it clears EITHER threshold (not both) — same "flagged" idea as the
# original Daily Signals mockup, where a match could be flagged for one
# market, the other, or both.

COMBINED_CARD_TEMPLATE = """<div style="background:#1a1f26;border-radius:12px;padding:14px;margin:10px 0;border:1px solid #2a3038">
  <div style="font-size:11px;color:#999">{league} · {time}</div>
  <div style="font-size:15px;font-weight:bold;margin:2px 0 6px">{home_team} vs {away_team}</div>
  <div style="font-size:11px;color:#aaa;margin-bottom:6px">FH BTTS: <span style="color:{fh_btts_color}">{fh_btts}%</span> &nbsp;|&nbsp; Over 4.5: <span style="color:{over45_color}">{over45}%</span></div>
  <div style="font-size:11px;color:#a0e8a0">{flags}</div>
</div>"""


def render_combined_cards(predictions):
    if not predictions:
        return '<p style="text-align:center;color:#888">No fixtures on this date qualified.</p>'
    cards = ""
    for p in predictions:
        fh_hit = p["fh_btts"] >= SCANNER_FH_BTTS_MIN
        o45_hit = p["over45"] >= SCANNER_OVER45_MIN
        flags = " + ".join(f for f, hit in [("✓ FH BTTS flagged", fh_hit), ("✓ Over 4.5 flagged", o45_hit)] if hit)
        cards += COMBINED_CARD_TEMPLATE.format(
            league=p["league"], time=p["date"][:16].replace("T", " "),
            home_team=p["home_team"], away_team=p["away_team"],
            fh_btts=p["fh_btts"], over45=p["over45"], flags=flags,
            fh_btts_color="#a0e8a0" if fh_hit else "#aaa",
            over45_color="#a0e8a0" if o45_hit else "#aaa",
        )
    return cards


def make_combined_scanner_html(predictions, csv_name, date_label=None, prev_href=None, next_href=None):
    prev_link = f'<a href="{prev_href}" style="color:#7ec8ff;text-decoration:none;font-size:20px">◀</a>' if prev_href else '<span style="color:#444;font-size:20px">◀</span>'
    next_link = f'<a href="{next_href}" style="color:#7ec8ff;text-decoration:none;font-size:20px">▶</a>' if next_href else '<span style="color:#444;font-size:20px">▶</span>'
    date_bar = f"""
<div style="display:flex;align-items:center;justify-content:center;gap:20px;margin:10px 0 4px">
  {prev_link}
  <span style="font-size:15px;font-weight:bold">{date_label or ''}</span>
  {next_link}
</div>""" if date_label else ""

    return SCANNER_HTML_TEMPLATE.format(
        page_title="FH BTTS / Over 4.5 Daily Scanner", icon="🎯",
        subtitle=f"First-half BTTS ≥{SCANNER_FH_BTTS_MIN}% or Over 4.5 Goals ≥{SCANNER_OVER45_MIN}%",
        generated=datetime.now().strftime("%d %b %H:%M"),
        date_bar=date_bar, csv_name=csv_name,
        qualified_count=len(predictions),
        cards=render_combined_cards(predictions),
    )


def write_combined_scanner_csv(predictions, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "League", "HomeTeam", "AwayTeam", "FHBTTS", "Over45"])
        for p in predictions:
            writer.writerow([p["date"], p["league"], p["home_team"], p["away_team"],
                              p["fh_btts"], p["over45"]])


def build_combined_scanner(all_predictions, base_dir="docs/match-iq/scanners"):
    qualified = [p for p in all_predictions
                 if p["fh_btts"] >= SCANNER_FH_BTTS_MIN or p["over45"] >= SCANNER_OVER45_MIN]
    qualified.sort(key=lambda p: (p["date_key"], -max(p["fh_btts"], p["over45"])))

    out_dir = f"{base_dir}/fh-btts-over45"
    os.makedirs(out_dir, exist_ok=True)

    by_date = group_by_date(qualified)
    date_keys = list(by_date.keys())

    if not date_keys:
        with open(f"{out_dir}/index.html", "w") as f:
            f.write(make_combined_scanner_html([], csv_name="fh-btts-over45_predictions.csv"))
    else:
        for i, date_key in enumerate(date_keys):
            prev_href = date_page_filename(date_keys[i - 1]) if i > 0 else None
            next_href = date_page_filename(date_keys[i + 1]) if i < len(date_keys) - 1 else None
            page_html = make_combined_scanner_html(
                by_date[date_key], csv_name="fh-btts-over45_predictions.csv",
                date_label=format_date_label(date_key), prev_href=prev_href, next_href=next_href,
            )
            with open(f"{out_dir}/{date_page_filename(date_key)}", "w") as f:
                f.write(page_html)
        with open(f"{out_dir}/{date_page_filename(date_keys[0])}") as f:
            soonest_html = f.read()
        with open(f"{out_dir}/index.html", "w") as f:
            f.write(soonest_html)

    write_combined_scanner_csv(qualified, f"{out_dir}/fh-btts-over45_predictions.csv")
    print(f"  FH BTTS / Over 4.5 scanner: {len(qualified)} fixtures across {len(date_keys)} date(s)")


SCANNER_CONFIGS = [
    {
        "dir": "over25", "market_key": "over25", "min": SCANNER_OVER25_MIN,
        "title": "Over 2.5 Goals Daily Scanner", "icon": "📊", "badge_label": "O2.5",
        "subtitle_fmt": f"All matches with ≥{SCANNER_OVER25_MIN}% Over 2.5 probability",
    },
    {
        "dir": "btts", "market_key": "btts", "min": SCANNER_BTTS_MIN,
        "title": "BTTS Daily Scanner", "icon": "⚽", "badge_label": "BTTS",
        "subtitle_fmt": f"All matches with ≥{SCANNER_BTTS_MIN}% BTTS probability",
    },
    {
        "dir": "corners", "market_key": "corners_over105", "min": SCANNER_CORNERS_MIN,
        "title": "Over 10.5 Corners Daily Scanner", "icon": "🚩", "badge_label": "O10.5",
        "subtitle_fmt": f"All matches with ≥{SCANNER_CORNERS_MIN}% Over 10.5 corners probability",
    },
]


def build_daily_signals_scanners(all_predictions, base_dir="docs/match-iq/scanners"):
    """Generates the three Daily Signals scanner pages (Over 2.5, BTTS,
    Over 10.5 Corners), each filtered independently from the FULL
    unfiltered fixture list — not the main page's filtered set — with
    its own date-paginated pages and CSV export, mirroring the main
    Match IQ page's existing date-navigation pattern."""
    for cfg in SCANNER_CONFIGS:
        qualified = [p for p in all_predictions if _scanner_badge_value(p, cfg["market_key"]) >= cfg["min"]]
        qualified.sort(key=lambda p: (p["date_key"], -_scanner_badge_value(p, cfg["market_key"])))

        out_dir = f"{base_dir}/{cfg['dir']}"
        os.makedirs(out_dir, exist_ok=True)

        by_date = group_by_date(qualified)
        date_keys = list(by_date.keys())

        common = dict(page_title=cfg["title"], icon=cfg["icon"], subtitle=cfg["subtitle_fmt"],
                      market_key=cfg["market_key"], badge_label=cfg["badge_label"],
                      csv_name=f"{cfg['dir']}_predictions.csv")

        if not date_keys:
            with open(f"{out_dir}/index.html", "w") as f:
                f.write(make_scanner_html([], **common))
        else:
            for i, date_key in enumerate(date_keys):
                prev_href = date_page_filename(date_keys[i - 1]) if i > 0 else None
                next_href = date_page_filename(date_keys[i + 1]) if i < len(date_keys) - 1 else None
                page_html = make_scanner_html(
                    by_date[date_key], date_label=format_date_label(date_key),
                    prev_href=prev_href, next_href=next_href, **common,
                )
                with open(f"{out_dir}/{date_page_filename(date_key)}", "w") as f:
                    f.write(page_html)
            with open(f"{out_dir}/{date_page_filename(date_keys[0])}") as f:
                soonest_html = f.read()
            with open(f"{out_dir}/index.html", "w") as f:
                f.write(soonest_html)

        write_scanner_csv(qualified, f"{out_dir}/{cfg['dir']}_predictions.csv")
        print(f"  {cfg['title']}: {len(qualified)} fixtures across {len(date_keys)} date(s)")

    build_combined_scanner(all_predictions, base_dir)


if __name__ == "__main__":
    # Prefer an environment variable (safer for CI — command-line args can
    # end up visible in process listings/logs) but keep the command-line
    # argument working too, for convenience when testing locally.
    api_key = os.environ.get("THESTATSAPI_KEY")
    if not api_key and len(sys.argv) >= 2:
        api_key = sys.argv[1]
    if not api_key:
        print("Usage: python3 match_iq.py YOUR_API_KEY")
        print("  (or set the THESTATSAPI_KEY environment variable)")
        sys.exit(1)

    all_predictions = build_all_predictions(api_key)
    predictions = apply_main_filter(all_predictions)

    os.makedirs("docs/match-iq", exist_ok=True)
    by_date = group_by_date(predictions)
    date_keys = list(by_date.keys())

    if not date_keys:
        with open("docs/match-iq/match_iq_index.html", "w") as f:
            f.write(make_html([]))
    else:
        for i, date_key in enumerate(date_keys):
            prev_href = date_page_filename(date_keys[i - 1]) if i > 0 else None
            next_href = date_page_filename(date_keys[i + 1]) if i < len(date_keys) - 1 else None
            page_html = make_html(
                by_date[date_key],
                date_label=format_date_label(date_key),
                prev_href=prev_href, next_href=next_href,
            )
            with open(f"docs/match-iq/{date_page_filename(date_key)}", "w") as f:
                f.write(page_html)

        # index.html mirrors the soonest date, so the root URL lands
        # somewhere sensible and you navigate forward from there — same
        # pattern as Goal IQ's date navigation.
        with open(f"docs/match-iq/{date_page_filename(date_keys[0])}") as f:
            soonest_html = f.read()
        with open("docs/match-iq/match_iq_index.html", "w") as f:
            f.write(soonest_html)

    write_csv(predictions, "docs/match-iq/match_iq_predictions.csv")
    with open("docs/match-iq/match_iq.json", "w") as f:
        json.dump(predictions, f, indent=2, default=str)

    print(f"\nMain page — {len(predictions)} fixtures across {len(date_keys)} date(s).")

    print("\nBuilding Daily Signals scanners...")
    build_daily_signals_scanners(all_predictions)

    print(f"\nDone — {len(all_predictions)} total fixtures projected.")
