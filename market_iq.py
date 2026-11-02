"""
Market IQ
---------
1X2 (home/draw/away) match-result model for English, Italian and Spanish football, built
from BOOKMAKER ODDS rather than this suite's usual stats-driven
approach (user request, 2026-10-29: "Can we use the bookmaker odds to
predict 1 x 2" -> "Only English leagues" -> "Let's run it via GitHub"
-> "Full site tool").

Unlike every other tool in this suite (Match IQ, Blitz IQ, Orange
Line, etc.), this one does NOT blend rolling form into a projection.
It takes a bookmaker's own priced-in probabilities and strips out
their margin ("de-vigging") to get the bookmaker's true implied
home/draw/away probabilities -- the same approach already used in
this user's horse racing model (de-vigged market odds + form). No
separate stats model is blended in here; this is a pure market-read
tool, one step below a full predictor, same spirit as Prop Finder
being a research tool rather than a prediction.

CONFIRMED LIVE (2026-10-29, via odds_1x2_diagnostic.py run through
GitHub Actions) before this was built:
  - GET /matches/{id}/odds is real and works with this account's key
    -- no special plan/tier gating hit (a gated plan would 401/403,
    not the clean 404 seen below).
  - Response shape: {"data": {"match_id": ..., "bookmakers": [
    {"bookmaker": "...", "markets": {"match_odds": {"home": {...},
    "draw": {...}, "away": {...}}}}]}}. Each side has "opening" and
    "last_seen" decimal-odds strings.
  - "opening" can be null while "last_seen" is populated -- this
    script always prefers last_seen, falling back to opening only if
    last_seen itself is missing.
  - Not every match has odds posted yet: a Championship fixture
    (West Brom vs Millwall) returned a clean 404 "Odds not found for
    this match". This is a PER-MATCH gap, not a per-league failure --
    League One and League Two both returned real odds for their own
    test matches. Handled here by skipping that match (with a
    [DIAG] print), never by crashing or guessing a price.
  - Only one bookmaker ("Paddy Power") has appeared in every example
    seen so far. If a match ever comes back with more than one,
    devig_match_odds() below averages each bookmaker's OWN de-vigged
    probabilities (never raw prices first) and prints a [DIAG] note,
    since that shape has never actually been observed live.

Usage:
    python3 market_iq.py

Set THESTATSAPI_KEY as an environment variable, or paste it into
API_KEY below. Writes a self-contained market_iq.html page (same
single-script-generates-its-own-HTML pattern as orange_line.py,
chosen over Prop Finder's script+JSON+client-filtered-HTML pattern
because there's no per-user filter control needed here -- just a
plain list of upcoming matches with their de-vigged prices).
"""

import os
import sys
import json
import math
import time
import hashlib
import html as html_lib
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

API_KEY = os.environ.get("THESTATSAPI_KEY", "PASTE_YOUR_KEY_HERE")
BASE_URL = "https://api.thestatsapi.com/api/football"

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

OUTPUT_HTML = Path(__file__).parent / "market_iq.html"

HEADERS = {"Authorization": f"Bearer {API_KEY}"}

# English leagues first (user request, 2026-10-29), then Italy and Spain
# (user request, 2026-11-02), in the order displayed on the page.
# "LaLiga" is TheStatsAPI's exact name (confirmed earlier via
# check_iberian_leagues.py, id comp_8814, country Spain) -- "La Liga" with
# a space does not match. "Serie A" is unverified against the live API:
# watch the first run's [DIAG]/[!] lines for it.
LEAGUES = [
    "Premier League",
    "Championship",
    "League One",
    "League Two",
    "Serie A",
    "LaLiga",
]

# Country hint used to pick the right competition when several share an
# exact name (there is more than one "Serie A" in the world). Compared
# case-insensitively against the competition's "country" field; if the
# field is missing the hint is ignored.
LEAGUE_COUNTRIES = {
    "serie a": "italy",
    "laliga": "spain",
}

# How far ahead to look for scheduled matches. Kept modest -- odds
# tend to firm up closer to kickoff, and a shorter window keeps the
# page focused on matches someone would actually act on soon.
DAYS_AHEAD = 10

# User request, 2026-10-29: "I use bet365" -- once more than one
# bookmaker started showing up per match (first confirmed live the
# same day, after only ever seeing Paddy Power alone during
# diagnostics), a plain cross-bookmaker average stopped being the
# right default for someone who actually prices against one specific
# book. When this bookmaker is present on a match, its own de-vigged
# probabilities are used alone instead of being blended with the
# others. Matched case-insensitively against each bookmaker's name
# (API diagnostics never confirmed a fixed casing/spelling for these).
PREFERRED_BOOKMAKER = "bet365"


# ----------------------------------------------------------------------
# Simple on-disk cache, same pattern as prop_finder.py/player_stat_model.py
# ----------------------------------------------------------------------

def _cache_path(key: str) -> Path:
    h = hashlib.sha256(key.encode()).hexdigest()[:20]
    return CACHE_DIR / f"{h}.json"


def cached_get(path: str, params: dict | None = None, ttl_hours: int = 12) -> dict:
    """GET from TheStatsAPI with a simple file cache. Shared CACHE_DIR
    with the rest of the suite, though GitHub Actions runs from a
    clean checkout each time so this mostly only helps local re-runs."""
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
    time.sleep(0.3)
    return data


def get_match_odds(match_id: str) -> dict | None:
    """GET /matches/{id}/odds. Not every match has odds posted yet
    (confirmed live: a clean 404 "Odds not found for this match") --
    that's a per-match data gap, so a 404 returns None instead of
    raising. Cached separately from cached_get() with a short TTL
    (odds move) and the 404 itself is cached too, so a re-run within
    the hour doesn't keep re-asking for odds that don't exist yet."""
    cache_key = f"/matches/{match_id}/odds::v1"
    cfile = _cache_path(cache_key)
    if cfile.exists():
        age_hours = (time.time() - cfile.stat().st_mtime) / 3600
        if age_hours < 1:
            cached = json.loads(cfile.read_text())
            return None if cached.get("__no_odds__") else cached

    resp = requests.get(f"{BASE_URL}/matches/{match_id}/odds", headers=HEADERS, timeout=20)
    if resp.status_code == 404:
        cfile.write_text(json.dumps({"__no_odds__": True}))
        return None
    resp.raise_for_status()
    data = resp.json()
    cfile.write_text(json.dumps(data))
    time.sleep(0.3)
    return data


# ----------------------------------------------------------------------
# TheStatsAPI calls -- same confirmed endpoints/shapes as the rest of the suite
# ----------------------------------------------------------------------

def find_competition(name: str) -> dict | None:
    """GET /competitions?search=... -- exact-name match preferred over
    a substring hit (same fix needed everywhere else in this suite: a
    "Premier League" search can rank a same-named foreign league
    first)."""
    data = cached_get("/competitions", {"search": name}, ttl_hours=24 * 30)
    results = data.get("data", [])
    if not results:
        print(f"  [!] no competition found for '{name}'")
        return None
    exact = [c for c in results if c.get("name", "").lower() == name.lower()]
    want = LEAGUE_COUNTRIES.get(name.lower())
    if want and exact:
        for comp in exact:
            ctry = comp.get("country")
            ctry = (ctry.get("name") if isinstance(ctry, dict) else ctry) or ""
            if str(ctry).lower() == want:
                return comp
        print(f"  [DIAG] '{name}': {len(exact)} exact-name result(s) but none with country "
              f"'{want}' (countries seen: {[c.get('country') for c in exact]}) -- using first")
    if exact:
        return exact[0]
    print(f"  [DIAG] no exact match for '{name}' among {len(results)} results -- "
          f"using first: {results[0].get('name')}")
    return results[0]


def get_upcoming_matches(competition_id: str, days_ahead: int = DAYS_AHEAD) -> list[dict]:
    today = date.today()
    data = cached_get("/matches", {
        "competition_id": competition_id,
        "date_from": today.isoformat(),
        "date_to": (today + timedelta(days=days_ahead)).isoformat(),
        "status": "scheduled",
        "per_page": 50,
    }, ttl_hours=6)
    matches = data.get("data", data.get("matches", []))
    matches.sort(key=lambda m: m.get("utc_date", m.get("date", "")))
    return matches


# ----------------------------------------------------------------------
# De-vigging
# ----------------------------------------------------------------------

def devig_match_odds(odds_response: dict) -> dict | None:
    """Turn a /matches/{id}/odds response into de-vigged true
    home/draw/away probabilities.

    Handles the two confirmed-live edge cases:
      - "opening" can be null while "last_seen" is populated -- prefer
        last_seen, fall back to opening only if last_seen is missing.
      - Several bookmakers can appear per match (confirmed live
        2026-10-29, after only ever seeing Paddy Power alone during
        diagnostics). When PREFERRED_BOOKMAKER is among them, its own
        de-vigged probabilities are used alone -- user request,
        2026-10-29: "I use bet365", so the fair price shown should
        reflect the book they actually bet on, not a blend. If
        PREFERRED_BOOKMAKER isn't on this match, each bookmaker's OWN
        de-vigged probabilities are averaged together instead -- never
        average raw prices first, which would let one bookmaker's
        margin bleed into another's true read.

    When PREFERRED_BOOKMAKER's own price is used AND at least one other
    bookmaker is also on the match, the result also carries a
    "market_*" comparison -- the average of every bookmaker present
    (PREFERRED_BOOKMAKER included), so the page can show Bet365's price
    next to what the wider market thinks (user request, 2026-10-30:
    "Yes do" -- add that comparison instead of leaving it to eyeball).
    A de-vigged number read alone carries no edge, since it's just that
    book's own margin stripped out; the gap between Bet365's price and
    the market average is the part worth looking at. No market_*
    fields are added when Bet365 is the only bookmaker on a match (the
    "average" source is already a one-sided view, or because there is
    no "rest of market" to compare it to) or when source is "average"
    (the primary number already IS the market average).

    Returns None (with a [DIAG] print) if the response doesn't match
    the confirmed shape, rather than guessing at a different one.
    """
    bookmakers = (odds_response or {}).get("data", {}).get("bookmakers", [])
    if not bookmakers:
        print("  [DIAG] odds response had no bookmakers[] -- unexpected shape, skipping")
        return None

    per_bookmaker = []
    for bk in bookmakers:
        match_odds = (bk.get("markets") or {}).get("match_odds")
        if not match_odds:
            continue

        prices = {}
        ok = True
        for side in ("home", "draw", "away"):
            side_odds = match_odds.get(side) or {}
            price = side_odds.get("last_seen")
            if price is None:
                price = side_odds.get("opening")
            if price is None:
                ok = False
                break
            try:
                price = float(price)
            except (TypeError, ValueError):
                ok = False
                break
            if price <= 1.0:
                ok = False
                break
            prices[side] = price
        if not ok:
            continue

        raw = {side: 1.0 / price for side, price in prices.items()}
        total = sum(raw.values())
        overround = total - 1.0
        fair = {side: v / total for side, v in raw.items()}
        per_bookmaker.append({"name": bk.get("bookmaker", "Unknown"), **fair,
                              "overround": overround, "prices": prices})

    if not per_bookmaker:
        return None

    names_used = [p["name"] for p in per_bookmaker]
    n = len(per_bookmaker)
    avg = {side: sum(p[side] for p in per_bookmaker) / n for side in ("home", "draw", "away")}
    avg_overround = sum(p["overround"] for p in per_bookmaker) / n

    preferred = next((p for p in per_bookmaker if p["name"].strip().lower() == PREFERRED_BOOKMAKER), None)

    if preferred is not None:
        result = {
            "home_prob": preferred["home"],
            "draw_prob": preferred["draw"],
            "away_prob": preferred["away"],
            "overround": preferred["overround"],
            "bookmakers": [preferred["name"]],
            "source": "preferred",
            # Raw decimal odds actually on offer at the preferred book --
            # the results tracker needs these to grade flagged bets at
            # the price that could really have been taken.
            "bet365_odds": dict(preferred["prices"]),
        }
        if n > 1:
            result.update({
                "market_home_prob": avg["home"],
                "market_draw_prob": avg["draw"],
                "market_away_prob": avg["away"],
                "market_overround": avg_overround,
                "market_bookmaker_count": n,
            })
        return result

    if n > 1:
        print(f"  [DIAG] {PREFERRED_BOOKMAKER!r} not among this match's bookmakers "
              f"({', '.join(names_used)}) -- averaging each bookmaker's own de-vigged "
              f"probabilities instead")

    return {
        "home_prob": avg["home"],
        "draw_prob": avg["draw"],
        "away_prob": avg["away"],
        "overround": avg_overround,
        "bookmakers": names_used,
        "source": "average",
    }


def flagged_sides(fair: dict) -> list[str]:
    """Sides where the market average rates the outcome at least
    MKT_GAP_HIGHLIGHT_PP more likely than Bet365 prices it -- the same
    rounded-gap rule the page uses to highlight a tile green, so the
    tracker logs exactly what the user sees flagged. Empty when there's
    no market comparison on the match."""
    if "market_home_prob" not in fair:
        return []
    return [side for side in ("home", "draw", "away")
            if round((fair[f"market_{side}_prob"] - fair[f"{side}_prob"]) * 100, 1)
            >= MKT_GAP_HIGHLIGHT_PP]


# ----------------------------------------------------------------------
# League table (user request, 2026-11-02: "add the league table standings
# to each fixture")
# ----------------------------------------------------------------------

def get_league_table(competition_id: str, league_name: str) -> dict:
    """{team_id: {"position", "points", "played"}} for the competition's
    current season, from /competitions/{id}/seasons/{season}/standings --
    same endpoint match_iq.py and prop_finder.py already use.

    Only team.id, goals_for, goals_against and matches_played are CONFIRMED
    row fields in this suite. position/points are read defensively
    (position, then rank; points, then 3*wins+draws) and, when position
    is missing, derived by sorting on points then goal difference. The
    actual row keys are printed once per league so the first live run
    confirms the real shape. Returns {} on any failure -- the odds page
    must never depend on this."""
    try:
        comp = cached_get(f"/competitions/{competition_id}", ttl_hours=24)
        season_id = comp.get("data", {}).get("current_season_id")
        if not season_id:
            print(f"  [DIAG] no current_season_id for {league_name} -- no table shown")
            return {}
        rows = cached_get(f"/competitions/{competition_id}/seasons/{season_id}/standings",
                          ttl_hours=6).get("data", [])
    except Exception as exc:
        print(f"  [DIAG] couldn't load {league_name} standings ({exc}) -- no table shown")
        return {}
    if rows:
        print(f"  [DIAG] {league_name} standings row keys: {sorted(rows[0].keys())}")
    return parse_table_rows(rows, league_name)


def parse_table_rows(rows: list, league_name: str = "") -> dict:
    table = {}
    for row in rows:
        team = row.get("team") or {}
        tid = team.get("id")
        if not tid:
            continue
        pts = row.get("points")
        if pts is None and row.get("wins") is not None and row.get("draws") is not None:
            pts = 3 * row["wins"] + row["draws"]
        pos = row.get("position", row.get("rank"))
        gd = row.get("goal_difference")
        if gd is None and row.get("goals_for") is not None and row.get("goals_against") is not None:
            gd = row["goals_for"] - row["goals_against"]
        table[tid] = {
            "position": pos, "points": pts, "played": row.get("matches_played"),
            # wins/draws/losses/goal_difference confirmed live (standings row
            # keys logged by the 2026-11-02 run).
            "wins": row.get("wins"), "draws": row.get("draws"), "losses": row.get("losses"),
            "gd": gd, "gf": row.get("goals_for"), "ga": row.get("goals_against"),
            "_sort": (-(pts if pts is not None else -1),
                      -((row.get("goals_for") or 0) - (row.get("goals_against") or 0)),
                      -(row.get("goals_for") or 0)),
        }

    if table and all(v["position"] is None for v in table.values()):
        if all(v["points"] is not None for v in table.values()):
            for i, tid in enumerate(sorted(table, key=lambda t: table[t]["_sort"]), start=1):
                table[tid]["position"] = i
            print(f"  [DIAG] {league_name}: no position field in standings rows -- derived from "
                  f"points then goal difference (head-to-head tiebreaks not applied)")
        else:
            print(f"  [DIAG] {league_name}: standings rows have neither position nor points -- no table shown")
            return {}
    for v in table.values():
        v.pop("_sort", None)
    return table


# ----------------------------------------------------------------------
# Table-based 1X2 model (user request, 2026-11-02: "can the model predict
# 1x2" -> "Yes"). Same independent-Poisson scoreline method Match IQ's
# predict_1x2() uses, fed by SEASON goals for/against from the standings
# rows this page already fetches (no extra API calls, works in all four
# leagues). It is deliberately simple and is shown NEXT TO Bet365's price,
# never instead of it: the results tracker grades it against the market,
# because bookmaker prices usually beat a table-only model.
# ----------------------------------------------------------------------

# How hard a team's own season goals per game is pulled toward the league
# average, expressed as "extra average games". Match IQ uses 3, but that is
# for a 7-game form window and a flat 3 is far too weak for ~9 games of
# season totals. Rule of thumb: regression games ~ (variance of one game's
# goals ~1.4) / (variance of true team strength ~0.3^2 = 0.09) ~ 15. 12 is
# a deliberate, slightly lighter choice. This is a judgement call, not a
# fitted value -- the results tracker's model-vs-market comparison is what
# should eventually tune it.
MODEL_PRIOR_GAMES = 12

# Standings give season-wide goals per game with NO home/away split, so
# home advantage is applied as a fixed assumption: roughly 25% more goals
# at home than away in English football, split evenly around the average
# (1.12 / 0.89, whose ratio is ~1.26). An assumption, not fitted.
HOME_GOALS_FACTOR = 1.12
AWAY_GOALS_FACTOR = 0.89

# A model gap this big vs Bet365's de-vigged probability is highlighted.
# Deliberately larger than the market-gap threshold: a table-only model is
# noisier than a cross-bookmaker average.
MODEL_GAP_HIGHLIGHT_PP = 5.0


def league_goal_rates(table: dict) -> tuple[float, float] | None:
    """(avg goals scored per team per game, avg conceded) across the
    league, or None without usable goals data. By symmetry the two are
    nearly equal; both are kept to mirror Match IQ's lg_scored/lg_conceded."""
    scored, conceded = [], []
    for t in table.values():
        if t.get("played") and t.get("gf") is not None and t.get("ga") is not None:
            scored.append(t["gf"] / t["played"])
            conceded.append(t["ga"] / t["played"])
    if not scored:
        return None
    return sum(scored) / len(scored), sum(conceded) / len(conceded)


def _shrunk_rate(total, played, league_avg):
    return (total + MODEL_PRIOR_GAMES * league_avg) / (played + MODEL_PRIOR_GAMES)


def poisson_1x2(exp_home: float, exp_away: float, max_goals: int = 10) -> dict:
    """Home/Draw/Away probabilities from independent Poisson goals (unrounded
    -- Match IQ's version rounds to whole percents). Ignores the small real-
    world correlation between the two teams' scoring and the Dixon-Coles
    low-score correction, so draws are, if anything, slightly under-priced."""
    pmf_h, pmf_a = [], []
    for k in range(max_goals + 1):
        pmf_h.append(math.exp(-exp_home) * exp_home ** k / math.factorial(k))
        pmf_a.append(math.exp(-exp_away) * exp_away ** k / math.factorial(k))
    home = draw = away = 0.0
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p = pmf_h[h] * pmf_a[a]
            if h > a:
                home += p
            elif h == a:
                draw += p
            else:
                away += p
    total = home + draw + away          # renormalise the (tiny) truncated tail
    return {"home_prob": home / total, "draw_prob": draw / total, "away_prob": away / total}


def model_1x2(home_row: dict | None, away_row: dict | None, rates) -> dict | None:
    """Opponent-adjusted expected goals, then Poisson -> 1X2. Same shape as
    Match IQ's predict_goals(): exp_home = home attack x (away defence /
    league avg), exp_away likewise, with a fixed home-advantage factor.
    Returns None when either team or the league has no goals data."""
    if not rates or not home_row or not away_row:
        return None
    for r in (home_row, away_row):
        if not r.get("played") or r.get("gf") is None or r.get("ga") is None:
            return None
    lg_scored, lg_conceded = rates
    if lg_scored <= 0 or lg_conceded <= 0:
        return None
    h_att = _shrunk_rate(home_row["gf"], home_row["played"], lg_scored)
    h_def = _shrunk_rate(home_row["ga"], home_row["played"], lg_conceded)
    a_att = _shrunk_rate(away_row["gf"], away_row["played"], lg_scored)
    a_def = _shrunk_rate(away_row["ga"], away_row["played"], lg_conceded)
    exp_home = HOME_GOALS_FACTOR * h_att * (a_def / lg_conceded)
    exp_away = AWAY_GOALS_FACTOR * a_att * (h_def / lg_conceded)
    out = poisson_1x2(exp_home, exp_away)
    out.update(exp_home=exp_home, exp_away=exp_away)
    return out


def model_flagged_sides(card: dict) -> list[str]:
    """Sides where the table model rates the outcome at least
    MODEL_GAP_HIGHLIGHT_PP more likely than Bet365 (or the average) prices
    it. Same rounded-gap convention as flagged_sides()."""
    m = card.get("model")
    if not m:
        return []
    return [side for side in ("home", "draw", "away")
            if round((m[f"{side}_prob"] - card[f"{side}_prob"]) * 100, 1) >= MODEL_GAP_HIGHLIGHT_PP]


MODEL_SKIP_GAP_PP = 15.0   # model this far from Bet365 on any side => "skip"


def model_skip(card: dict) -> bool:
    """True when the table model is so far from Bet365 that the blue lines
    are almost certainly the model missing something (squad quality,
    injuries), not an edge: any side off by >= MODEL_SKIP_GAP_PP, or the
    model and Bet365 disagreeing on the favourite."""
    m = card.get("model")
    if not m:
        return False
    sides = ("home", "draw", "away")
    if any(abs(m[f"{x}_prob"] - card[f"{x}_prob"]) * 100 >= MODEL_SKIP_GAP_PP for x in sides):
        return True
    return max(sides, key=lambda x: m[f"{x}_prob"]) != max(sides, key=lambda x: card[f"{x}_prob"])


def ordinal(n: int) -> str:
    n = int(n)
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


# ----------------------------------------------------------------------
# Scan
# ----------------------------------------------------------------------

def team_name(side: dict | str) -> str:
    if isinstance(side, dict):
        return side.get("name", "Unknown")
    return side or "Unknown"


def scan_league(league_name: str) -> list[dict]:
    comp = find_competition(league_name)
    if not comp:
        return []
    matches = get_upcoming_matches(comp["id"])
    print(f"{comp.get('name', league_name)}: {len(matches)} upcoming match(es) in the next {DAYS_AHEAD} days")
    table = get_league_table(comp["id"], comp.get("name", league_name))
    rates = league_goal_rates(table)

    cards = []
    for m in matches:
        match_id = m.get("id")
        home = team_name(m.get("home_team", {}))
        away = team_name(m.get("away_team", {}))
        kickoff = m.get("utc_date", m.get("date", ""))

        try:
            odds_data = get_match_odds(match_id)
        except Exception as exc:
            print(f"  [!] odds lookup failed for {home} vs {away}: {exc}")
            continue

        if odds_data is None:
            print(f"  [DIAG] no odds yet for {home} vs {away} ({kickoff}) -- skipping")
            continue

        fair = devig_match_odds(odds_data)
        if fair is None:
            print(f"  [DIAG] couldn't de-vig {home} vs {away} -- odds shape didn't match, skipping")
            continue

        h_id = (m.get("home_team") or {}).get("id") if isinstance(m.get("home_team"), dict) else None
        a_id = (m.get("away_team") or {}).get("id") if isinstance(m.get("away_team"), dict) else None
        model = model_1x2(table.get(h_id), table.get(a_id), rates)

        card = {
            "match_id": match_id,
            "league": comp.get("name", league_name),
            "home": home,
            "away": away,
            "kickoff": kickoff,
            "flagged": flagged_sides(fair),
            "home_table": table.get((m.get("home_team") or {}).get("id")) if isinstance(m.get("home_team"), dict) else None,
            "away_table": table.get((m.get("away_team") or {}).get("id")) if isinstance(m.get("away_team"), dict) else None,
            "model": model,
            **fair,
        }
        card["model_flagged"] = model_flagged_sides(card)
        cards.append(card)

    return cards


def run_all_leagues(league_names: list[str] = LEAGUES) -> list[dict]:
    all_cards = []
    for name in league_names:
        try:
            all_cards.extend(scan_league(name))
        except Exception as exc:
            print(f"Skipping league '{name}': {exc}")
            continue
    return all_cards


# ----------------------------------------------------------------------
# HTML
# ----------------------------------------------------------------------

PAGE_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Market IQ</title>
<style>
  :root{{
    --bg:#0b0f14; --panel:#121820; --panel2:#161d27; --border:#233040;
    --text:#e8edf2; --sub:#8b98a8; --amber:#facc15; --green:#34d399; --red:#f87171;
  }}
  body{{
    margin:0; background:var(--bg); color:var(--text);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
    padding:20px; max-width:640px; margin:0 auto;
  }}
  h1{{ font-size:24px; margin-bottom:4px; }}
  .sub{{ color:var(--sub); font-size:13px; margin-bottom:6px; }}
  .notice{{
    background:#1a1410; border:1px solid #3a2e18; border-radius:10px;
    padding:10px 12px; font-size:11.5px; color:#d8c9a8; line-height:1.5;
    margin-bottom:20px;
  }}
  h2.league{{
    font-size:14px; color:var(--amber); text-transform:uppercase;
    letter-spacing:0.04em; margin:26px 0 10px; border-bottom:1px solid var(--border);
    padding-bottom:6px;
  }}
  .card{{
    background:var(--panel); border:1px solid var(--border); border-radius:12px;
    padding:14px; margin-bottom:10px;
  }}
  .matchup{{ font-size:15px; font-weight:700; margin-bottom:2px; }}
  .kickoff{{ color:var(--sub); font-size:11px; margin-bottom:10px; }}
  .tbl{{ font-size:11px; color:var(--sub); background:var(--panel2); border-radius:8px;
    padding:6px 10px; margin:-2px 0 10px; }}
  .tbl .r{{ display:flex; justify-content:space-between; gap:10px; padding:1px 0; }}
  .tbl .n{{ overflow:hidden; text-overflow:ellipsis; white-space:nowrap; min-width:0; }}
  .tbl .r span:last-child{{ white-space:nowrap; flex-shrink:0; }}
  .tbl b{{ color:var(--text); font-weight:600; }}
  .probs{{ display:flex; gap:8px; margin-bottom:8px; }}
  .prob{{
    flex:1; background:var(--panel2); border-radius:8px; padding:8px 6px;
    text-align:center;
  }}
  .prob .label{{ color:var(--sub); font-size:10px; text-transform:uppercase; margin-bottom:3px; }}
  .prob .pct{{ font-size:16px; font-weight:700; }}
  .prob .fairodds{{ color:var(--sub); font-size:10px; margin-top:2px; }}
  .prob.fav .pct{{ color:var(--green); }}
  .prob .mkt{{ color:var(--sub); font-size:9.5px; margin-top:4px; border-top:1px solid var(--border); padding-top:3px; }}
  .prob .mkt.highlight{{ color:var(--green); font-weight:600; }}
  .skip{{ display:inline-block; margin:2px 0 8px; padding:2px 8px; border-radius:999px;
         font-size:11px; font-weight:700; color:#ffb4a8; background:rgba(255,120,100,.14);
         border:1px solid rgba(255,120,100,.4); }}
  .prob .mdl{{ color:var(--sub); font-size:9.5px; margin-top:2px; }}
  .prob .mdl.highlight{{ color:#7ec8ff; font-weight:600; }}
  .meta{{ color:var(--sub); font-size:10.5px; display:flex; justify-content:space-between; }}
  .empty{{ color:var(--sub); font-size:13px; padding:20px 0; }}
  .footer{{ color:var(--sub); font-size:11px; text-align:center; margin-top:28px; line-height:1.6; }}
</style>
</head>
<body>
  <h1>📈 Market IQ</h1>
  <div class="sub">Generated {generated_at}</div>
  <div class="notice">
    This one works differently from the rest of the suite: it doesn't
    blend rolling form into a projection. It takes a bookmaker's own
    1X2 odds and strips out their built-in margin ("de-vigging") to
    show the bookmaker's true implied Home/Draw/Away probabilities --
    a market read, not a stats-driven prediction. Prices shown are
    <b>Bet365's own</b> price where Bet365 has posted odds on that
    match; where they haven't, it falls back to the average of
    whichever bookmakers have (labelled "market avg" below). Each
    bookmaker's <b>last seen</b> price is used, falling back to their
    opening price only if no later price was recorded. Where other
    bookmakers are also on a match, each tile adds a small "mkt" line
    comparing Bet365's price to the market average -- a de-vigged
    number on its own carries no edge, since it's just that book's own
    margin removed, so a highlighted gap there (Bet365 paying longer
    odds than the wider market thinks fair) is the part worth a second
    look, not the percentage itself. Each tile also shows an "mdl" line (the table model): a simple table-based
    estimate (season goals for and against, plus a fixed home-advantage
    assumption) turned into Home/Draw/Away odds the same way Match IQ does
    it. It is NOT a bookmaker price and is usually less sharp than one, so
    a highlighted blue gap is only a prompt to look closer -- the Results
    page tracks whether the model actually beats the market. Tap a Home/Draw/Away tile
    on any Bet365-priced match to add it to the slip calculator at the bottom
    of the page. Not every
    upcoming match has odds posted yet -- those are left off rather
    than guessed at.
  </div>
  {filter_html}
  {sections_html}
  <div class="footer">
    <a href="results/index.html" style="color:#7ec8ff">📋 Results tracker — do the flagged gaps actually pay?</a><br><br>
    Runs on its own automated schedule via GitHub Actions.<br>
    De-vigged percentages reflect one bookmaker's priced-in view, not a
    guarantee of the outcome -- treat as a market snapshot, not a firm pick.
  </div>
  {slip_html}
</body>
</html>
"""

# Slip calculator (user request, 2026-10-31, after a 10-leg accumulator lost
# with 8 legs landing): tap tiles to build a slip and see the REAL combined
# chance of every leg landing next to the combined odds. Plain string, not a
# format template -- it is injected as-is, so CSS/JS braces stay single.
# One selection per match (picking another side replaces the first), so legs
# are separate matches and the independence assumption is a fair first
# approximation. "Chance" uses Bet365's own de-vigged probabilities; when
# every leg also has a market-average view, that is shown too.
SLIP_HTML = """<style>
  body{padding-bottom:90px}
  .prob[data-pick]{cursor:pointer;border:1px dashed #34414f;touch-action:manipulation;
    -webkit-tap-highlight-color:rgba(250,204,21,.25);user-select:none;-webkit-user-select:none}
  .prob[data-pick].picked{border:1px solid var(--amber);background:#1d2430}
  #slip{position:fixed;left:0;right:0;bottom:0;background:#0f151d;border-top:1px solid var(--border);
    padding:8px 16px 10px;z-index:10}
  #slip .inner{max-width:640px;margin:0 auto}
  #slip summary{cursor:pointer;font-size:13px;list-style:none}
  #slip summary b{color:var(--amber)}
  #slip .legs{font-size:11px;color:var(--sub);margin:8px 0}
  #slip .legs div{display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid var(--border)}
  #slip .legs button{background:none;border:0;color:var(--red);font-size:12px;cursor:pointer}
  #slip .row{display:flex;justify-content:space-between;font-size:12px;padding:3px 0}
  #slip .note{font-size:10px;color:var(--sub);line-height:1.5;margin-top:6px}
  #slip input{width:64px;background:var(--panel2);border:1px solid var(--border);color:var(--text);
    border-radius:6px;padding:3px 6px;font-size:12px}
  #slip .clear{background:none;border:1px solid var(--border);color:var(--sub);border-radius:6px;
    font-size:11px;padding:3px 8px;cursor:pointer;margin-top:6px}
</style>
<div id="slip"><div class="inner"><details>
  <summary id="slipSummary"><b>Slip calculator</b> · tap a Home / Draw / Away tile to add a leg</summary>
  <div id="slipBody" style="display:none">
  <div class="legs" id="slipLegs"></div>
  <div class="row"><span>Stake £ <input id="slipStake" type="number" min="0" step="1" value="10"></span><span id="slipReturn"></span></div>
  <div class="row"><span>Fair odds (Bet365's de-vigged view)</span><span id="slipFair"></span></div>
  <div class="row"><span>Expected return per £1 staked</span><span id="slipEV"></span></div>
  <div class="row" id="slipMktRow" style="display:none"><span>Market-average view: chance / expected return</span><span id="slipMkt"></span></div>
  <div class="note">Assumes legs are independent (one pick per match). A snapshot of prices, not a prediction -- every extra leg multiplies bet365's margin as well as the payout.</div>
  <button class="clear" id="slipClear">Clear slip</button>
  </div>
</details></div></div>
<script>
(function(){
  var picks = {}; // match id -> {side,label,odds,prob,mprob,el}
  var slip = document.getElementById('slip');
  function fmtPct(x){ return (x*100).toFixed(x < 0.1 ? 2 : 1) + '%'; }
  function oneIn(p){ var n = 1/p; return n >= 10 ? Math.round(n) : n.toFixed(1); }
  function render(){
    var ids = Object.keys(picks), n = ids.length;
    var details = document.querySelector('#slip details'), body = document.getElementById('slipBody');
    if(!n){
      document.getElementById('slipSummary').innerHTML =
        '<b>Slip calculator</b> · tap a Home / Draw / Away tile to add a leg';
      body.style.display = 'none';
      details.open = false;
      return;
    }
    body.style.display = 'block';
    var odds = 1, prob = 1, mprob = 1, haveM = true;
    ids.forEach(function(id){
      var k = picks[id]; odds *= k.odds; prob *= k.prob;
      if(k.mprob == null) haveM = false; else mprob *= k.mprob;
    });
    var stake = parseFloat(document.getElementById('slipStake').value) || 0;
    document.getElementById('slipSummary').innerHTML =
      '<b>' + n + ' leg' + (n>1?'s':'') + '</b> · odds <b>' + odds.toFixed(2) + '</b> · chance all land <b>' +
      fmtPct(prob) + '</b> (1 in ' + oneIn(prob) + ')  ▲';
    document.getElementById('slipReturn').textContent = 'returns £' + (stake*odds).toFixed(2);
    document.getElementById('slipFair').textContent = (1/prob).toFixed(2) + ' vs ' + odds.toFixed(2) + ' offered';
    document.getElementById('slipEV').textContent = '£' + (prob*odds).toFixed(2);
    var mrow = document.getElementById('slipMktRow');
    if(haveM){
      mrow.style.display = 'flex';
      document.getElementById('slipMkt').textContent = fmtPct(mprob) + ' / £' + (mprob*odds).toFixed(2);
    } else { mrow.style.display = 'none'; }
    var legs = document.getElementById('slipLegs'); legs.innerHTML = '';
    ids.forEach(function(id){
      var k = picks[id], row = document.createElement('div'), t = document.createElement('span'),
          b = document.createElement('button');
      t.textContent = k.label + ' @ ' + k.odds.toFixed(2);
      b.textContent = 'remove'; b.setAttribute('data-remove', id);
      row.appendChild(t); row.appendChild(b); legs.appendChild(row);
    });
  }
  function unpick(id){
    if(picks[id]){ picks[id].el.classList.remove('picked'); delete picks[id]; }
  }
  document.addEventListener('click', function(e){
    var rm = e.target.closest && e.target.closest('[data-remove]');
    if(rm){ unpick(rm.getAttribute('data-remove')); render(); return; }
    var tile = e.target.closest && e.target.closest('.prob[data-pick]');
    if(!tile) return;
    var id = tile.getAttribute('data-match'), side = tile.getAttribute('data-side');
    var same = picks[id] && picks[id].side === side;
    unpick(id);
    if(!same){
      var m = tile.getAttribute('data-mprob');
      picks[id] = {side: side, label: tile.getAttribute('data-label'),
        odds: parseFloat(tile.getAttribute('data-odds')), prob: parseFloat(tile.getAttribute('data-prob')),
        mprob: m === null || m === '' ? null : parseFloat(m), el: tile};
      tile.classList.add('picked');
    }
    render();
  });
  document.getElementById('slipStake').addEventListener('input', render);
  document.getElementById('slipClear').addEventListener('click', function(){
    Object.keys(picks).forEach(unpick); render();
  });
})();
</script>"""

LEAGUE_SECTION_TEMPLATE = """<section class="lsec" data-league="{league_key}"><h2 class="league">{league}</h2>
{cards_html}</section>"""

# League + date filter (user request, 2026-11-02). Plain string, not a
# .format() template, so CSS/JS braces are written normally. Chips are
# built by make_html(); this JS only shows/hides cards and sections.
FILTER_STYLE = """<style>
  #filters{margin:14px 0 6px;}
  #filters .row{display:flex; gap:6px; overflow-x:auto; padding:2px 0 8px; -webkit-overflow-scrolling:touch;}
  #filters .chip{flex:0 0 auto; padding:6px 12px; border-radius:999px; font-size:12.5px; cursor:pointer;
    color:var(--sub); background:var(--panel2); border:1px solid var(--border);
    font-family:inherit; touch-action:manipulation; -webkit-tap-highlight-color:transparent;}
  #filters .chip.on{color:#0b0f14; background:#7ec8ff; border-color:#7ec8ff; font-weight:700;}
  #filters .lbl{font-size:11px; color:var(--sub); margin:2px 0 2px; text-transform:uppercase; letter-spacing:.06em;}
  #noMatch{display:none; text-align:center; color:var(--sub); padding:24px 0; font-size:14px;}
</style>"""

FILTER_SCRIPT = """<script>
(function(){
  var sel = {league:'all', date:'all'};
  function apply(){
    var shown = 0;
    document.querySelectorAll('.lsec').forEach(function(sec){
      var leagueOk = sel.league==='all' || sec.getAttribute('data-league')===sel.league;
      var n = 0;
      sec.querySelectorAll('.card').forEach(function(c){
        var ok = leagueOk && (sel.date==='all' || c.getAttribute('data-date')===sel.date);
        c.style.display = ok ? '' : 'none';
        if (ok) n++;
      });
      var empty = sec.querySelector('.empty');
      if (empty) { empty.style.display = (leagueOk && sel.date==='all') ? '' : 'none'; if (leagueOk && sel.date==='all') n++; }
      sec.style.display = n ? '' : 'none';
      shown += n;
    });
    document.getElementById('noMatch').style.display = shown ? 'none' : 'block';
  }
  document.getElementById('filters').addEventListener('click', function(e){
    var b = e.target.closest('.chip'); if (!b) return;
    var kind = b.getAttribute('data-kind');
    sel[kind] = b.getAttribute('data-val');
    document.querySelectorAll('#filters .chip[data-kind="'+kind+'"]').forEach(function(x){
      x.classList.toggle('on', x===b);
    });
    apply();
  });
})();
</script>"""


def _chip(kind: str, val: str, label: str, on: bool = False) -> str:
    return (f'<button type="button" class="chip{" on" if on else ""}" data-kind="{kind}" '
            f'data-val="{html_lib.escape(val, quote=True)}">{html_lib.escape(label)}</button>')


def _card_date(card: dict) -> str:
    return (card.get("kickoff") or "")[:10]


def build_filter_html(cards: list[dict]) -> str:
    league_chips = [_chip("league", "all", "All leagues", True)] + [
        _chip("league", lg.lower(), lg) for lg in LEAGUES]
    dates = sorted({d for d in (_card_date(c) for c in cards) if d})
    date_chips = [_chip("date", "all", "All dates", True)]
    for d in dates:
        try:
            label = datetime.strptime(d, "%Y-%m-%d").strftime("%a %d %b")
        except ValueError:
            label = d
        date_chips.append(_chip("date", d, label))
    return (FILTER_STYLE + '<div id="filters"><div class="lbl">League</div><div class="row">'
            + "".join(league_chips) + '</div><div class="lbl">Date</div><div class="row">'
            + "".join(date_chips) + '</div></div>'
            '<div id="noMatch">No matches for that league and date.</div>')

CARD_TEMPLATE = """<div class="card"{card_attrs}>
  <div class="matchup">{home} vs {away}</div>
  <div class="kickoff">{kickoff_display}</div>
  {table_line}
  {skip_badge}
  <div class="probs">
    <div class="prob{home_fav}"{home_attrs}>
      <div class="label">Home</div>
      <div class="pct">{home_pct}%</div>
      <div class="fairodds">fair {home_fairodds}</div>
      {home_mkt}
      {home_mdl}
    </div>
    <div class="prob{draw_fav}"{draw_attrs}>
      <div class="label">Draw</div>
      <div class="pct">{draw_pct}%</div>
      <div class="fairodds">fair {draw_fairodds}</div>
      {draw_mkt}
      {draw_mdl}
    </div>
    <div class="prob{away_fav}"{away_attrs}>
      <div class="label">Away</div>
      <div class="pct">{away_pct}%</div>
      <div class="fairodds">fair {away_fairodds}</div>
      {away_mkt}
      {away_mdl}
    </div>
  </div>
  <div class="meta"><span>{bookmakers_display}</span><span>{overround_pct}% overround</span></div>
</div>"""

# One tile's "vs market" line, only rendered when a comparison is
# available (Bet365's own price exists AND at least one other
# bookmaker is also on the match -- see devig_match_odds' market_*
# fields). gap = market probability minus Bet365's own probability:
# positive means the wider market rates that side MORE likely than
# Bet365 prices it, i.e. Bet365 is paying a longer price than the
# market thinks is fair -- that gap, not the raw percentage, is what
# actually points at value. Flagged only past a small threshold so
# ordinary noise between books isn't highlighted as if it were a signal.
MKT_LINE = '<div class="mkt{highlight}">mkt {mkt_pct}% ({sign}{gap_pct})</div>'
MDL_LINE = '<div class="mdl{highlight}">mdl {mdl_pct}% ({sign}{gap_pct})</div>'
MKT_GAP_HIGHLIGHT_PP = 2.0


def _fmt_kickoff(raw: str) -> str:
    if not raw:
        return "Date TBC"
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return dt.strftime("%a %d %b, %H:%M UTC")
    except ValueError:
        return raw


def _fav_class(prob: float, home_p: float, draw_p: float, away_p: float) -> str:
    return " fav" if prob == max(home_p, draw_p, away_p) else ""


def _mdl_line(bet365_p: float, model_p: float | None) -> str:
    """One tile's "model" line: the table model's probability and its gap
    to Bet365's de-vigged figure (highlighted past MODEL_GAP_HIGHLIGHT_PP)."""
    if model_p is None:
        return ""
    gap_pct = round((model_p - bet365_p) * 100, 1) + 0.0
    return MDL_LINE.format(
        mdl_pct=round(model_p * 100, 1),
        sign="+" if gap_pct >= 0 else "",
        gap_pct=gap_pct,
        highlight=" highlight" if gap_pct >= MODEL_GAP_HIGHLIGHT_PP else "",
    )


def _mkt_line(bet365_p: float, market_p: float | None) -> str:
    """Build one tile's "vs market" line, or "" when there's nothing to
    compare against (see MKT_LINE's own comment for what the gap means)."""
    if market_p is None:
        return ""
    # "+ 0.0" normalises a negative zero (a tiny negative gap rounds to
    # -0.0, which would otherwise print as "+-0.0").
    gap_pct = round((market_p - bet365_p) * 100, 1) + 0.0
    return MKT_LINE.format(
        mkt_pct=round(market_p * 100, 1),
        sign="+" if gap_pct >= 0 else "",
        gap_pct=gap_pct,
        highlight=" highlight" if gap_pct >= MKT_GAP_HIGHLIGHT_PP else "",
    )


def _table_line(card: dict) -> str:
    """Two stacked rows (home, then away): team name on the left, league
    position / points / W-D-L / goal difference on the right (games played
    only as a fallback when W-D-L is missing). A team with no table
    data is simply omitted; both missing -> no block at all."""
    def row(team: str, t: dict | None) -> str:
        if not t or t.get("position") is None:
            return ""
        bits = [f"<b>{ordinal(t['position'])}</b>"]
        if t.get("points") is not None:
            bits.append(f"{t['points']} pts")
        if None not in (t.get("wins"), t.get("draws"), t.get("losses")):
            bits.append(f"{t['wins']}-{t['draws']}-{t['losses']}")   # W-D-L (sums to games played)
        elif t.get("played") is not None:
            bits.append(f"P{t['played']}")
        if t.get("gd") is not None:
            bits.append(f"{t['gd']:+d} GD")
        return (f'<div class="r"><span class="n">{html_lib.escape(team)}</span>'
                f'<span>{" · ".join(bits)}</span></div>')

    rows = row(card["home"], card.get("home_table")) + row(card["away"], card.get("away_table"))
    return f'<div class="tbl">{rows}</div>' if rows else ""


def build_card_html(card: dict) -> str:
    home_p, draw_p, away_p = card["home_prob"], card["draw_prob"], card["away_prob"]
    names = sorted(set(card.get("bookmakers", [])))
    if card.get("source") == "preferred":
        bookmakers_display = names[0] if names else "Unknown"
    elif len(names) > 1:
        bookmakers_display = f"Market avg ({len(names)} books)"
    else:
        bookmakers_display = names[0] if names else "Unknown"

    has_market = "market_home_prob" in card

    def attrs(side: str, prob: float) -> str:
        """data-* attributes that make a tile tappable for the slip
        calculator -- only when Bet365's own decimal odds and a match id
        exist (an 'average'-source card has no Bet365 price to bet at)."""
        odds = (card.get("bet365_odds") or {}).get(side)
        if not odds or not card.get("match_id"):
            return ""
        label = f"{card['home']} vs {card['away']} - {side.capitalize()}"
        mprob = card.get(f"market_{side}_prob") if has_market else None
        return (f' data-pick="1" data-match="{html_lib.escape(str(card["match_id"]), quote=True)}"'
                f' data-side="{side}" data-label="{html_lib.escape(label, quote=True)}"'
                f' data-odds="{odds}" data-prob="{prob}"'
                f' data-mprob="{"" if mprob is None else mprob}"')

    return CARD_TEMPLATE.format(
        home=card["home"], away=card["away"],
        kickoff_display=_fmt_kickoff(card.get("kickoff", "")),
        card_attrs=f' data-date="{_card_date(card)}"',
        table_line=_table_line(card),
        skip_badge=('<div class="skip">SKIP &middot; model far from Bet365 &ndash; ignore the mdl lines</div>'
                    if model_skip(card) else ""),
        home_pct=round(home_p * 100, 1), draw_pct=round(draw_p * 100, 1), away_pct=round(away_p * 100, 1),
        home_fairodds=round(1 / home_p, 2) if home_p > 0 else "-",
        draw_fairodds=round(1 / draw_p, 2) if draw_p > 0 else "-",
        away_fairodds=round(1 / away_p, 2) if away_p > 0 else "-",
        home_fav=_fav_class(home_p, home_p, draw_p, away_p),
        draw_fav=_fav_class(draw_p, home_p, draw_p, away_p),
        away_fav=_fav_class(away_p, home_p, draw_p, away_p),
        home_attrs=attrs("home", home_p), draw_attrs=attrs("draw", draw_p), away_attrs=attrs("away", away_p),
        home_mdl=_mdl_line(home_p, (card.get("model") or {}).get("home_prob")),
        draw_mdl=_mdl_line(draw_p, (card.get("model") or {}).get("draw_prob")),
        away_mdl=_mdl_line(away_p, (card.get("model") or {}).get("away_prob")),
        home_mkt=_mkt_line(home_p, card.get("market_home_prob") if has_market else None),
        draw_mkt=_mkt_line(draw_p, card.get("market_draw_prob") if has_market else None),
        away_mkt=_mkt_line(away_p, card.get("market_away_prob") if has_market else None),
        bookmakers_display=bookmakers_display,
        overround_pct=round(card.get("overround", 0) * 100, 1),
    )


def make_html(cards: list[dict]) -> str:
    by_league: dict[str, list[dict]] = {}
    for card in cards:
        by_league.setdefault(card["league"], []).append(card)

    sections = []
    for league_name in LEAGUES:
        matching_key = next((k for k in by_league if k.lower() == league_name.lower()), None)
        league_cards = by_league.get(matching_key, []) if matching_key else []
        if not league_cards:
            cards_html = '<div class="empty">No upcoming matches with posted odds right now.</div>'
        else:
            cards_html = "\n".join(build_card_html(c) for c in league_cards)
        sections.append(LEAGUE_SECTION_TEMPLATE.format(league=league_name, league_key=league_name.lower(), cards_html=cards_html))

    return PAGE_TEMPLATE.format(
        generated_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        filter_html=build_filter_html(cards),
        sections_html="\n".join(sections),
        slip_html=SLIP_HTML + FILTER_SCRIPT,
    )


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    if API_KEY == "PASTE_YOUR_KEY_HERE":
        print("Set THESTATSAPI_KEY (env var) or paste a key into API_KEY before running.", file=sys.stderr)
        sys.exit(1)

    cards = run_all_leagues()
    html = make_html(cards)
    OUTPUT_HTML.write_text(html)
    print(f"\nWrote {len(cards)} match(es) with odds across {len(LEAGUES)} league(s) -> {OUTPUT_HTML}")

    # Results tracker: logs flagged sides pre-kickoff, grades them later.
    # Wrapped so a tracker problem can never stop the page above from
    # being published.
    try:
        import market_iq_results_tracker
        market_iq_results_tracker.run_results_tracker(cards, API_KEY)
    except Exception as exc:
        print(f"[!] results tracker failed (page was still written): {exc}")


if __name__ == "__main__":
    main()
