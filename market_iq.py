"""
Market IQ
---------
1X2 (home/draw/away) match-result model for English football, built
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
import time
import hashlib
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

# Only English leagues (explicit user request, 2026-10-29), in the
# order they're displayed on the page.
LEAGUES = [
    "Premier League",
    "Championship",
    "League One",
    "League Two",
]

# How far ahead to look for scheduled matches. Kept modest -- odds
# tend to firm up closer to kickoff, and a shorter window keeps the
# page focused on matches someone would actually act on soon.
DAYS_AHEAD = 10


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
    for comp in results:
        if comp.get("name", "").lower() == name.lower():
            return comp
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
      - Every confirmed example so far has exactly one bookmaker. If
        more than one ever appears, each bookmaker's OWN de-vigged
        probabilities are averaged together -- never average raw
        prices first, which would let one bookmaker's margin bleed
        into another's true read.

    Returns None (with a [DIAG] print) if the response doesn't match
    the confirmed shape, rather than guessing at a different one.
    """
    bookmakers = (odds_response or {}).get("data", {}).get("bookmakers", [])
    if not bookmakers:
        print("  [DIAG] odds response had no bookmakers[] -- unexpected shape, skipping")
        return None

    per_bookmaker = []
    names_used = []
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
        per_bookmaker.append({**fair, "overround": overround})
        names_used.append(bk.get("bookmaker", "Unknown"))

    if not per_bookmaker:
        return None

    if len(per_bookmaker) > 1:
        print(f"  [DIAG] multiple bookmakers on one match ({', '.join(names_used)}) -- "
              f"never confirmed live before now; averaging each bookmaker's own "
              f"de-vigged probabilities")

    n = len(per_bookmaker)
    avg = {side: sum(p[side] for p in per_bookmaker) / n for side in ("home", "draw", "away")}
    avg_overround = sum(p["overround"] for p in per_bookmaker) / n

    return {
        "home_prob": avg["home"],
        "draw_prob": avg["draw"],
        "away_prob": avg["away"],
        "overround": avg_overround,
        "bookmakers": names_used,
    }


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

        cards.append({
            "league": comp.get("name", league_name),
            "home": home,
            "away": away,
            "kickoff": kickoff,
            **fair,
        })

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
  .probs{{ display:flex; gap:8px; margin-bottom:8px; }}
  .prob{{
    flex:1; background:var(--panel2); border-radius:8px; padding:8px 6px;
    text-align:center;
  }}
  .prob .label{{ color:var(--sub); font-size:10px; text-transform:uppercase; margin-bottom:3px; }}
  .prob .pct{{ font-size:16px; font-weight:700; }}
  .prob .fairodds{{ color:var(--sub); font-size:10px; margin-top:2px; }}
  .prob.fav .pct{{ color:var(--green); }}
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
    each bookmaker's <b>last seen</b> price (falling back to their
    opening price only if no later price was recorded). Not every
    upcoming match has odds posted yet -- those are left off rather
    than guessed at.
  </div>
  {sections_html}
  <div class="footer">
    Runs on its own automated schedule via GitHub Actions.<br>
    De-vigged percentages reflect one bookmaker's priced-in view, not a
    guarantee of the outcome -- treat as a market snapshot, not a firm pick.
  </div>
</body>
</html>
"""

LEAGUE_SECTION_TEMPLATE = """<h2 class="league">{league}</h2>
{cards_html}"""

CARD_TEMPLATE = """<div class="card">
  <div class="matchup">{home} vs {away}</div>
  <div class="kickoff">{kickoff_display}</div>
  <div class="probs">
    <div class="prob{home_fav}">
      <div class="label">Home</div>
      <div class="pct">{home_pct}%</div>
      <div class="fairodds">fair {home_fairodds}</div>
    </div>
    <div class="prob{draw_fav}">
      <div class="label">Draw</div>
      <div class="pct">{draw_pct}%</div>
      <div class="fairodds">fair {draw_fairodds}</div>
    </div>
    <div class="prob{away_fav}">
      <div class="label">Away</div>
      <div class="pct">{away_pct}%</div>
      <div class="fairodds">fair {away_fairodds}</div>
    </div>
  </div>
  <div class="meta"><span>{bookmakers}</span><span>{overround_pct}% overround</span></div>
</div>"""


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


def build_card_html(card: dict) -> str:
    home_p, draw_p, away_p = card["home_prob"], card["draw_prob"], card["away_prob"]
    bookmakers = ", ".join(sorted(set(card.get("bookmakers", [])))) or "Unknown"
    return CARD_TEMPLATE.format(
        home=card["home"], away=card["away"],
        kickoff_display=_fmt_kickoff(card.get("kickoff", "")),
        home_pct=round(home_p * 100, 1), draw_pct=round(draw_p * 100, 1), away_pct=round(away_p * 100, 1),
        home_fairodds=round(1 / home_p, 2) if home_p > 0 else "-",
        draw_fairodds=round(1 / draw_p, 2) if draw_p > 0 else "-",
        away_fairodds=round(1 / away_p, 2) if away_p > 0 else "-",
        home_fav=_fav_class(home_p, home_p, draw_p, away_p),
        draw_fav=_fav_class(draw_p, home_p, draw_p, away_p),
        away_fav=_fav_class(away_p, home_p, draw_p, away_p),
        bookmakers=bookmakers,
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
        sections.append(LEAGUE_SECTION_TEMPLATE.format(league=league_name, cards_html=cards_html))

    return PAGE_TEMPLATE.format(
        generated_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        sections_html="\n".join(sections),
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


if __name__ == "__main__":
    main()
