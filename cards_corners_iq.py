#!/usr/bin/env python3
"""
Cards & Corners IQ — built on TheStatsAPI (api.thestatsapi.com)
Same architecture and same data source as Match IQ, but focused ONLY on
per-TEAM corner and yellow-card totals -- not match totals (Match IQ
already covers combined Over 10.5 Corners), and not cards+reds combined
(per the design decision behind this tool: reds are rare enough in a
recency-weighted 7-game sample to be mostly noise, so this tracks YELLOW
CARDS ONLY -- a red card swing here would need its own, separately
validated model, not folded into this one).

WHY A SEPARATE SCRIPT rather than importing from match_iq.py: every tool
in this suite is self-contained on purpose (Euro Ice doesn't import Blue
Line's code even though both are hockey; Strike Zone doesn't import Blitz
IQ's even though both use recency-weighted Poisson). This one is no
different -- it duplicates match_iq.py's proven API/rate-limit layer
rather than importing it, so a change to Match IQ can never silently
break this tool (or vice versa) through a shared import. LEAGUE_SEARCH_
NAMES is kept in sync with match_iq.py BY HAND -- if you add/remove a
league there, mirror the change here too.

HOT FORM vs REAL STREAK (same distinction fixed across every other tool
in this suite after the Guardians-style edge case): Hot Form is an
average over the recent-games window -- can mask a bad MOST RECENT game.
Real Streak is a genuine CONSECUTIVE run above a threshold, computed by
walking backward from the most recent game and stopping at the first
break. Both are shown, clearly labeled, same as everywhere else.

Setup:
    pip3 install requests --break-system-packages
    python3 cards_corners_iq.py YOUR_API_KEY

Output:
    docs/cards-corners/index.html
    docs/cards-corners/cards_corners_predictions.csv
    docs/cards-corners/cards_corners.json
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
RECENT_GAMES = 7          # same window as Match IQ, for a comparable sample size
PRIOR_STRENGTH = 3        # same small-sample shrinkage weight as every sibling tool
FIXTURE_WINDOW_DAYS = 10

# Kept in sync BY HAND with match_iq.py's LEAGUE_SEARCH_NAMES -- see module
# docstring for why this isn't a shared import.
LEAGUE_SEARCH_NAMES = [
    "Premier League",
    "Championship",
    "Bundesliga",
    "Eredivisie",
    "Liga Portugal Betclic",
    "LaLiga",
    "UEFA Champions League",
    "MLS",
    "Swiss Super League",
    "Danish Superliga",
    "Eliteserien",
]

# ---------------------------------------------------------------------
# Hot Form / Real Streak thresholds -- STARTING GUESSES, not backtested.
# Corners: ~5+ in a match is a genuinely corner-heavy team; cards: ~2+
# yellows in a match is a genuinely card-heavy team for one side. Tune
# these once the results tracker (if/when built for this tool) shows
# whether they're calling anything real.
# ---------------------------------------------------------------------
CORNERS_HOT_FORM_MIN = 5.0
CORNERS_HOT_FORM_MIN_GAMES = 3
CORNERS_REAL_STREAK_THRESHOLD = 5
CORNERS_REAL_STREAK_MIN_LENGTH = 3

CARDS_HOT_FORM_MIN = 2.0
CARDS_HOT_FORM_MIN_GAMES = 3
CARDS_REAL_STREAK_THRESHOLD = 2
CARDS_REAL_STREAK_MIN_LENGTH = 3


def _headers(key):
    return {"Authorization": f"Bearer {key}"}


def _get(path, key, params=None, timeout=15):
    """Same adaptive rate-limiting as match_iq.py -- reads the real
    X-RateLimit-Remaining/Reset headers and only backs off when actually
    close to the limit, plus a flat pacing delay per call (2.0s) that
    match_iq.py found necessary in practice to avoid repeated 429s."""
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
    """Last RECENT_GAMES finished matches for a team, chronological
    OLDEST-FIRST (unlike match_iq.py's own get_team_form, which appends
    while iterating newest-first -- that ordering doesn't matter for a
    plain average, but it WOULD silently break Real Streak's "walk
    backward from the most recent game" logic here, so this fetch
    explicitly sorts ascending before building the lists). Pulls only
    corner_kicks (both sides) and yellow_cards (this team's side) --
    red_cards is deliberately not read at all, per this tool's cards
    definition (see module docstring)."""
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

    # ascending (oldest first) -- take the most recent RECENT_GAMES while
    # PRESERVING chronological order within that window
    matches = sorted(data["data"], key=lambda m: m["utc_date"])[-RECENT_GAMES:]
    if not matches:
        team_form_cache[cache_key] = None
        return None

    corners, corners_conceded, yellow_cards = [], [], []
    dates = []

    for m in matches:
        is_home = m["home_team"]["id"] == team_id
        stats = _get(f"/football/matches/{m['id']}/stats", key)
        if not stats or not stats.get("data"):
            continue
        ov = stats["data"].get("overview", {})

        def side_val(stat_key):
            item = ov.get(stat_key, {}).get("all")
            if not item:
                return None
            return item["home"] if is_home else item["away"]

        def opp_side_val(stat_key):
            item = ov.get(stat_key, {}).get("all")
            if not item:
                return None
            return item["away"] if is_home else item["home"]

        c = side_val("corner_kicks")
        cc = opp_side_val("corner_kicks")
        y = side_val("yellow_cards")

        # Only append to a metric's own list if that metric was actually
        # present for this match -- keeps corners_list/yellow_cards_list
        # each internally consistent (no placeholder zeros standing in
        # for missing data), even if that means the two lists end up
        # different lengths for the same team.
        if c is not None:
            corners.append(c)
        if cc is not None:
            corners_conceded.append(cc)
        if y is not None:
            yellow_cards.append(y)
        dates.append(m["utc_date"])

    if not corners and not yellow_cards:
        team_form_cache[cache_key] = None
        return None

    form = {
        "n_games": len(matches),
        "avg_corners": round(sum(corners) / len(corners), 2) if corners else None,
        "corners_list": corners,  # oldest -> newest
        "avg_corners_conceded": round(sum(corners_conceded) / len(corners_conceded), 2) if corners_conceded else None,
        "corners_conceded_list": corners_conceded,
        "avg_yellow_cards": round(sum(yellow_cards) / len(yellow_cards), 2) if yellow_cards else None,
        "yellow_cards_list": yellow_cards,  # oldest -> newest
    }
    team_form_cache[cache_key] = form
    return form


def shrink(value, n, league_avg, prior=PRIOR_STRENGTH):
    """Same small-sample protection as every sibling tool -- a 1-2 game
    sample leans mostly on the league average; by RECENT_GAMES games the
    team's own recent rate dominates."""
    if value is None:
        return league_avg
    return (n * value + prior * league_avg) / (n + prior)


# ---------------------------------------------------------------------
# League-average running samples -- same pattern as Match IQ's
# _lg_full_corners_avg()/_record_full_corners_sample(): built up live from
# whatever teams have been processed so far in THIS run, rather than a
# separate standings-based call, since neither corners nor cards are
# reported in TheStatsAPI's standings endpoint (only goals are).
# ---------------------------------------------------------------------
_corners_won_samples = []
_corners_conceded_samples = []
_cards_samples = []


def _lg_avg(samples, fallback):
    return sum(samples) / len(samples) if samples else fallback


def _record_samples(form):
    if form.get("avg_corners") is not None:
        _corners_won_samples.append(form["avg_corners"])
    if form.get("avg_corners_conceded") is not None:
        _corners_conceded_samples.append(form["avg_corners_conceded"])
    if form.get("avg_yellow_cards") is not None:
        _cards_samples.append(form["avg_yellow_cards"])


def project_team_corners(form, opp_form):
    """This team's expected corners = its own recency-weighted+season-
    shrunk corner-WINNING rate, adjusted by the OPPONENT's corner-
    CONCEDING rate relative to the league average -- same opponent-
    adjustment idea already validated in Match IQ's predict_full_corners,
    just kept at the single-team level instead of summed into a match
    total."""
    _record_samples(form)
    lg_won = _lg_avg(_corners_won_samples, 5.0)
    lg_conceded = _lg_avg(_corners_conceded_samples, 5.0)

    own_rate = shrink(form.get("avg_corners"), form["n_games"], lg_won)
    opp_conceded_rate = shrink(opp_form.get("avg_corners_conceded"), opp_form["n_games"], lg_conceded)
    opp_factor = (opp_conceded_rate / lg_conceded) if lg_conceded else 1.0

    lam = round(own_rate * opp_factor, 2)
    return {"lambda": lam, "corners_list": form.get("corners_list") or []}


def project_team_cards(form):
    """This team's expected yellow cards = its own recency-weighted+
    season-shrunk rate, with NO opponent adjustment -- a team's card
    count is mostly about ITS OWN discipline/tactical fouling and the
    referee, not meaningfully explained by the opponent's own card rate
    the way corners-conceded genuinely predicts corners allowed. Adding
    an unvalidated opponent factor here would be inventing a signal, not
    using one -- so this stays deliberately simpler than the corners
    projection."""
    _record_samples(form)
    lg_cards = _lg_avg(_cards_samples, 1.8)
    lam = round(shrink(form.get("avg_yellow_cards"), form["n_games"], lg_cards), 2)
    return {"lambda": lam, "yellow_cards_list": form.get("yellow_cards_list") or []}


def safe_line(lam, factor=0.72, round_to=0.5):
    if lam is None:
        return None
    raw = lam * factor
    line = math.floor(raw / round_to) * round_to
    return max(line, round_to)


def prob_over(lam, line):
    threshold = math.floor(line) + 1
    return 1 - poisson_cdf(threshold - 1, lam)


def hit_rate(values, line):
    """Model-free empirical cross-check from the same raw values feeding
    the projection -- same convention as every sibling tool."""
    if not values:
        return None
    hits = sum(1 for v in values if v > line)
    return {"hits": hits, "total": len(values)}


def format_history(values):
    return "/".join(str(v) for v in values) if values else None


# ---------------------------------------------------------------------
# Hot Form (average) vs Real Streak (genuine consecutive run) -- same
# fix already applied to Match IQ, Euro Ice, Strike Zone, Under IQ, and
# Orange Line after the Guardians-style edge case (high average, but the
# MOST RECENT game breaks the pattern). Both computed here for corners
# and cards independently.
# ---------------------------------------------------------------------

def _current_streak(values_oldest_first, threshold):
    """Walks backward from the most recent value, counting consecutive
    values >= threshold, stopping at the first break. values_oldest_first
    must genuinely be oldest->newest (see get_team_form's docstring) or
    this silently computes the wrong streak."""
    streak = 0
    for v in reversed(values_oldest_first):
        if v >= threshold:
            streak += 1
        else:
            break
    return streak


def build_hot_form_entries(team_rows, stat_key, min_avg, min_games, label, unit):
    """team_rows: list of {"team", "match", "league", "n_games", "avg", "list"}."""
    entries = []
    for t in team_rows:
        if t["n_games"] >= min_games and t["avg"] is not None and t["avg"] >= min_avg:
            entries.append({**t, "label": label, "unit": unit})
    entries.sort(key=lambda e: -e["avg"])
    return entries


def build_real_streak_entries(team_rows, threshold, min_length, label, unit):
    entries = []
    for t in team_rows:
        streak = _current_streak(t["list"], threshold)
        if streak >= min_length:
            entries.append({**t, "streak": streak, "threshold": threshold, "label": label, "unit": unit})
    entries.sort(key=lambda e: -e["streak"])
    return entries


def build_predictions(key):
    predictions = []
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

            h_corners = project_team_corners(h_form, a_form)
            a_corners = project_team_corners(a_form, h_form)
            h_cards = project_team_cards(h_form)
            a_cards = project_team_cards(a_form)

            predictions.append({
                "match_id": m["id"], "league": comp["name"],
                "date": m["utc_date"], "date_key": m["utc_date"][:10],
                "home_team": m["home_team"]["name"], "away_team": m["away_team"]["name"],
                "home_form": h_form, "away_form": a_form,
                "home_corners": h_corners, "away_corners": a_corners,
                "home_cards": h_cards, "away_cards": a_cards,
            })
    return predictions


def build_legs(predictions):
    legs = []
    for p in predictions:
        match_label = f"{p['home_team']} vs {p['away_team']}"
        for side, team_name, proj, list_key in (
            ("home", p["home_team"], p["home_corners"], "corners_list"),
            ("away", p["away_team"], p["away_corners"], "corners_list"),
        ):
            line = safe_line(proj["lambda"])
            if not line:
                continue
            legs.append({
                "match": match_label, "subject": team_name,
                "market": f"{team_name} Over {line} Corners",
                "prob": round(prob_over(proj["lambda"], line) * 100),
                "category": "Team Corners",
                "hit_rate": hit_rate(proj[list_key], line),
                "detail": f"{p['league']} · proj {proj['lambda']} corners",
                "history": format_history(proj[list_key]),
            })
        for side, team_name, proj, list_key in (
            ("home", p["home_team"], p["home_cards"], "yellow_cards_list"),
            ("away", p["away_team"], p["away_cards"], "yellow_cards_list"),
        ):
            line = safe_line(proj["lambda"])
            if not line:
                continue
            legs.append({
                "match": match_label, "subject": team_name,
                "market": f"{team_name} Over {line} Cards",
                "prob": round(prob_over(proj["lambda"], line) * 100),
                "category": "Team Cards",
                "hit_rate": hit_rate(proj[list_key], line),
                "detail": f"{p['league']} · proj {proj['lambda']} yellow cards",
                "history": format_history(proj[list_key]),
            })
    return legs


def _team_rows(predictions, side_key, form_key, stat_avg_key, stat_list_key):
    rows = []
    for p in predictions:
        team_name = p[f"{side_key}_team"]
        form = p[form_key]
        match_label = f"{p['home_team']} vs {p['away_team']}"
        avg = form.get(stat_avg_key)
        lst = form.get(stat_list_key) or []
        if avg is None:
            continue
        rows.append({
            "team": team_name, "match": match_label, "league": p["league"],
            "date": p["date"], "n_games": form["n_games"], "avg": avg, "list": lst,
        })
    return rows


def build_all_form_entries(predictions):
    corner_rows = (_team_rows(predictions, "home", "home_form", "avg_corners", "corners_list") +
                   _team_rows(predictions, "away", "away_form", "avg_corners", "corners_list"))
    card_rows = (_team_rows(predictions, "home", "home_form", "avg_yellow_cards", "yellow_cards_list") +
                 _team_rows(predictions, "away", "away_form", "avg_yellow_cards", "yellow_cards_list"))

    return {
        "corners_hot_form": build_hot_form_entries(
            corner_rows, "avg_corners", CORNERS_HOT_FORM_MIN, CORNERS_HOT_FORM_MIN_GAMES,
            "Corners Hot Form", "corners"),
        "corners_real_streak": build_real_streak_entries(
            corner_rows, CORNERS_REAL_STREAK_THRESHOLD, CORNERS_REAL_STREAK_MIN_LENGTH,
            "Real Corner Streak", "corners"),
        "cards_hot_form": build_hot_form_entries(
            card_rows, "avg_yellow_cards", CARDS_HOT_FORM_MIN, CARDS_HOT_FORM_MIN_GAMES,
            "Cards Hot Form", "cards"),
        "cards_real_streak": build_real_streak_entries(
            card_rows, CARDS_REAL_STREAK_THRESHOLD, CARDS_REAL_STREAK_MIN_LENGTH,
            "Real Card Streak", "cards"),
    }


# ---------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------

BUILDER_TEMPLATE = """<div class="builderPanel">
  <div class="builderTitle">🎯 Safest Bet Builder</div>
  <div id="builderCategoryToggles" class="builderToggles"></div>
  <div class="builderControls">
    <label>Target odds:</label>
    <input type="number" step="0.1" min="1.1" value="5.0" id="targetOdds">
    <label>Max legs:</label>
    <input type="number" step="1" min="2" value="8" id="maxLegs">
    <button class="builderBtn" onclick="buildSafest()">Build</button>
    <button class="builderBtnAlt" onclick="buildSafest()">🔀 Shuffle</button>
  </div>
  <div id="builderResult" class="builderResult">
    Untick a market you don't want considered, set a target odds and leg cap, then tap Build.
    Caps at 2 legs per team to avoid stacking a team's own Corners and Cards legs from the same match.
  </div>
</div>"""

STREAK_ROW = """<div style="display:flex;justify-content:space-between;font-size:12px;padding:6px 0;border-top:1px solid #3a2a20">
  <div><b>{team}</b><br><span style="color:#998">{match} · {league}</span></div>
  <div style="text-align:right"><span style="color:#7dd3a8;font-weight:bold">{value}</span><br><span style="color:#998">{sub}</span></div>
</div>"""

STREAK_PANEL = """<div style="background:#1a1310;border-radius:12px;padding:16px;margin:12px 0;border:1px solid #3a2a20">
  <div style="font-size:14px;font-weight:bold;margin-bottom:6px">{icon} {title}</div>
  <div style="font-size:11px;color:#998;margin-bottom:10px">{note}</div>
  {rows}
</div>"""

MATCH_CARD = """<div style="background:#14261a;border-radius:12px;padding:16px;margin:12px 0;border:1px solid #2a4a34">
  <div style="font-size:11px;color:#998;text-transform:uppercase;letter-spacing:.03em">{league}</div>
  <h3 style="margin:2px 0 10px 0;font-size:16px">{away} @ {home}</h3>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
    <div style="background:#0f1a12;border-radius:8px;padding:10px">
      <div style="font-size:11px;color:#998;margin-bottom:4px">{home} — Corners</div>
      <div style="font-size:20px;font-weight:800;color:#7dd3a8">{home_corners_lam}</div>
      <div style="font-size:10px;color:#998">last: {home_corners_hist}</div>
    </div>
    <div style="background:#0f1a12;border-radius:8px;padding:10px">
      <div style="font-size:11px;color:#998;margin-bottom:4px">{away} — Corners</div>
      <div style="font-size:20px;font-weight:800;color:#7dd3a8">{away_corners_lam}</div>
      <div style="font-size:10px;color:#998">last: {away_corners_hist}</div>
    </div>
    <div style="background:#0f1a12;border-radius:8px;padding:10px">
      <div style="font-size:11px;color:#998;margin-bottom:4px">{home} — Cards</div>
      <div style="font-size:20px;font-weight:800;color:#ffeb3b">{home_cards_lam}</div>
      <div style="font-size:10px;color:#998">last: {home_cards_hist}</div>
    </div>
    <div style="background:#0f1a12;border-radius:8px;padding:10px">
      <div style="font-size:11px;color:#998;margin-bottom:4px">{away} — Cards</div>
      <div style="font-size:20px;font-weight:800;color:#ffeb3b">{away_cards_lam}</div>
      <div style="font-size:10px;color:#998">last: {away_cards_hist}</div>
    </div>
  </div>
</div>"""

HTML_TEMPLATE = """<!DOCTYPE html><html><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Cards &amp; Corners IQ</title>
<style>
  :root{{--bg:#0b0f14; --panel:#121820; --panel2:#161d27; --border:#233040; --text:#e8edf2; --sub:#8b98a8; --amber:#facc15; --green:#22c55e;}}
  body{{margin:0; background:var(--bg); color:var(--text); font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; padding:16px; max-width:640px; margin:0 auto;}}
  h1{{font-size:20px; margin-bottom:4px;}}
  .sub{{color:var(--sub); font-size:13px; margin-bottom:18px;}}
  .builderPanel{{background:var(--panel); border:1px solid var(--border); border-radius:12px; padding:16px; margin-bottom:14px;}}
  .builderTitle{{font-size:15px; font-weight:800; margin-bottom:10px;}}
  .builderToggles{{display:flex; gap:10px; flex-wrap:wrap; margin-bottom:10px; font-size:12px;}}
  .builderToggles label{{display:flex; align-items:center; gap:4px; color:var(--text); cursor:pointer;}}
  .builderControls{{display:flex; gap:8px; align-items:center; margin-bottom:6px; flex-wrap:wrap;}}
  .builderControls label{{font-size:12px; color:var(--sub);}}
  .builderControls input{{width:70px; background:var(--panel2); border:1px solid var(--border); color:var(--text); border-radius:6px; padding:6px 8px; font-size:13px;}}
  .builderBtn{{background:var(--green); color:#04140a; font-weight:700; border:none; padding:7px 14px; border-radius:6px; font-size:13px; cursor:pointer;}}
  .builderBtnAlt{{background:var(--panel2); border:1px solid var(--border); color:var(--text); padding:7px 14px; border-radius:6px; font-size:13px; cursor:pointer;}}
  .builderResult{{font-size:12px; color:var(--sub);}}
  .legRow{{display:flex; justify-content:space-between; padding:5px 0; border-bottom:1px solid var(--border);}}
  .footnote{{font-size:11px; color:var(--sub); text-align:center; margin-top:20px; line-height:1.6;}}
</style></head>
<body>
  <h1>🚩 Cards &amp; Corners IQ</h1>
  <div class="sub">Per-team Corners &amp; Yellow Cards · generated {generated}</div>

  {builder}
  {corners_hot_form}
  {corners_real_streak}
  {cards_hot_form}
  {cards_real_streak}
  {cards}

  <div class="footnote">
    Corners: each team's own recency-weighted+season corner-winning rate, adjusted by the
    OPPONENT'S corner-conceding rate vs. league average. Cards: each team's own recency-weighted+
    season YELLOW card rate (red cards excluded entirely — see the page's build notes), with no
    opponent adjustment. Hot Form = average over the last {recent_games} games (can mask a bad most
    recent game). Real Streak = genuinely CONSECUTIVE recent games clearing the threshold, walking
    backward from the most recent game. Lines are set automatically below the model's projection for
    a safety margin.
  </div>

<script>
const LEGS = {legs_json};

function poissonCDF(threshold, lambda){{
  let p = Math.exp(-lambda), cum = p;
  for(let i=1;i<=threshold;i++){{ p = p*lambda/i; cum += p; }}
  return cum;
}}
function initToggles() {{
  const container = document.getElementById('builderCategoryToggles');
  const cats = [...new Set(LEGS.map(l => l.category))];
  container.innerHTML = cats.map(c => `
    <label><input type="checkbox" class="catToggle" value="${{c}}" checked> ${{c}}</label>
  `).join('');
}}
function shuffleArr(arr) {{
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
  const keys = Object.keys(bands).map(Number).sort((a,b) => b-a);
  let result = [];
  keys.forEach(k => {{ result = result.concat(shuffleArr(bands[k])); }});
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
  const subjectCount = {{}};
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
        const count = subjectCount[leg.subject] || 0;
        if (count >= 2) continue;
        chosen.push(leg);
        combinedOdds *= 100 / leg.prob;
        subjectCount[leg.subject] = count + 1;
        addedThisPass = true;
        break;
      }}
    }}
  }}

  const out = document.getElementById('builderResult');
  if (!chosen.length) {{ out.innerHTML = 'No legs available to build from.'; return; }}

  const rows = chosen.map(l => `
    <div class="legRow">
      <span>${{l.match}}<br><span style="color:var(--amber)">${{l.market}}</span> <span style="color:var(--sub)">· ${{l.category}}</span>
      ${{l.detail ? `<br><span style="color:var(--sub);font-size:10px">${{l.detail}}</span>` : ''}}
      ${{l.history ? `<br><span style="color:var(--sub);font-size:10px">last games: ${{l.history}}</span>` : ''}}</span>
      <span style="text-align:right"><span style="color:var(--amber);font-weight:bold">${{l.prob}}%</span>${{l.hit_rate ? `<br><span style="color:var(--sub);font-size:11px">${{l.hit_rate.hits}}/${{l.hit_rate.total}}</span>` : ''}}</span>
    </div>
  `).join('');

  const capNote = chosen.length >= maxLegs && combinedOdds < target
    ? ' (hit the leg cap before reaching target — raise Max legs or lower Target odds)'
    : (combinedOdds < target ? ' (ran out of legs before reaching target)' : '');

  out.innerHTML = `
    <div style="color:var(--text);font-size:13px;margin-bottom:6px">
      ${{chosen.length}} legs · est. combined odds ~<b>${{combinedOdds.toFixed(2)}}</b>${{capNote}}
    </div>
    ${{rows}}
    <div style="color:var(--sub);font-size:10px;margin-top:8px;line-height:1.4">
      Estimate multiplies each leg's fair odds (100/probability) — real sportsbook odds
      include their margin, so treat this as a ranking tool, not a firm price.
    </div>
  `;
}}
initToggles();
</script>
</body></html>"""


def _streak_panel(entries, icon, title, note, value_fmt, sub_fmt):
    if not entries:
        return ""
    rows = "".join(STREAK_ROW.format(
        team=e["team"], match=e["match"], league=e["league"],
        value=value_fmt(e), sub=sub_fmt(e),
    ) for e in entries)
    return STREAK_PANEL.format(icon=icon, title=title, note=note, rows=rows)


def make_html(predictions):
    legs = build_legs(predictions)
    builder = BUILDER_TEMPLATE if legs else ""

    forms = build_all_form_entries(predictions)

    corners_hot_form = _streak_panel(
        forms["corners_hot_form"], "📊", "Corners Hot Form",
        f"Average over the last {RECENT_GAMES} games — can mask a bad most recent game. Not a streak.",
        lambda e: f"{e['avg']} avg", lambda e: f"{e['n_games']}gm · {format_history(e['list'])}",
    )
    corners_real_streak = _streak_panel(
        forms["corners_real_streak"], "🔥", "Real Corner Streak",
        f"Genuinely CONSECUTIVE recent games with {CORNERS_REAL_STREAK_THRESHOLD}+ corners, walking back from the most recent.",
        lambda e: f"{e['streak']}+ straight", lambda e: format_history(e["list"]),
    )
    cards_hot_form = _streak_panel(
        forms["cards_hot_form"], "📊", "Cards Hot Form",
        f"Average over the last {RECENT_GAMES} games — can mask a bad most recent game. Not a streak.",
        lambda e: f"{e['avg']} avg", lambda e: f"{e['n_games']}gm · {format_history(e['list'])}",
    )
    cards_real_streak = _streak_panel(
        forms["cards_real_streak"], "🔥", "Real Card Streak",
        f"Genuinely CONSECUTIVE recent games with {CARDS_REAL_STREAK_THRESHOLD}+ yellow cards, walking back from the most recent.",
        lambda e: f"{e['streak']}+ straight", lambda e: format_history(e["list"]),
    )

    cards_html = "".join(MATCH_CARD.format(
        league=p["league"], home=p["home_team"], away=p["away_team"],
        home_corners_lam=p["home_corners"]["lambda"], away_corners_lam=p["away_corners"]["lambda"],
        home_corners_hist=format_history(p["home_corners"]["corners_list"]) or "—",
        away_corners_hist=format_history(p["away_corners"]["corners_list"]) or "—",
        home_cards_lam=p["home_cards"]["lambda"], away_cards_lam=p["away_cards"]["lambda"],
        home_cards_hist=format_history(p["home_cards"]["yellow_cards_list"]) or "—",
        away_cards_hist=format_history(p["away_cards"]["yellow_cards_list"]) or "—",
    ) for p in predictions)
    if not cards_html:
        cards_html = '<p style="text-align:center;color:var(--sub)">No usable matches today.</p>'

    return HTML_TEMPLATE.format(
        generated=datetime.now().strftime("%Y-%m-%d %H:%M"),
        recent_games=RECENT_GAMES,
        builder=builder, corners_hot_form=corners_hot_form, corners_real_streak=corners_real_streak,
        cards_hot_form=cards_hot_form, cards_real_streak=cards_real_streak, cards=cards_html,
        legs_json=json.dumps(legs),
    )


def write_csv(predictions, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "League", "Match", "Team", "Market", "Projected",
                          "RecentValues", "Line", "OverOdds", "UnderOdds", "ActualValue", "HitOrMiss"])
        for p in predictions:
            for team_name, proj, list_key, market in (
                (p["home_team"], p["home_corners"], "corners_list", "Corners"),
                (p["away_team"], p["away_corners"], "corners_list", "Corners"),
                (p["home_team"], p["home_cards"], "yellow_cards_list", "Cards"),
                (p["away_team"], p["away_cards"], "yellow_cards_list", "Cards"),
            ):
                writer.writerow([
                    p["date"], p["league"], f"{p['home_team']} vs {p['away_team']}", team_name, market,
                    proj["lambda"], "; ".join(str(v) for v in proj[list_key]),
                    "", "", "", "", "",
                ])


if __name__ == "__main__":
    api_key = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("THESTATSAPI_KEY")
    if not api_key:
        print("Pass your TheStatsAPI key as an argument, or set THESTATSAPI_KEY.")
        raise SystemExit(1)

    predictions = build_predictions(api_key)
    os.makedirs("docs/cards-corners", exist_ok=True)
    with open("docs/cards-corners/index.html", "w") as f:
        f.write(make_html(predictions))
    write_csv(predictions, "docs/cards-corners/cards_corners_predictions.csv")
    with open("docs/cards-corners/cards_corners.json", "w") as f:
        json.dump(predictions, f, indent=2, default=str)
    print(f"\nDone — {len(predictions)} matches processed.")

    try:
        import cards_corners_results_tracker
        cards_corners_results_tracker.run_results_tracker(predictions, api_key)
    except Exception as e:
        print(f"[!] results tracker failed (main run unaffected): {e}")
