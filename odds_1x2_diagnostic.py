"""
Odds 1X2 Diagnostic
--------------------
ONE-OFF confirmation script, NOT a tool/model -- run this once, read the
output, then delete it (or leave it, it's harmless either way). Checks
three things before any real "predict 1X2 from bookmaker odds" model
gets built:

  1. Does /football/matches/{id}/odds actually work with YOUR key and
     plan, or does it 403/404 (the docs page that confirmed this
     endpoint exists didn't say whether odds access is gated to a
     specific pricing tier)?
  2. What does a REAL response actually look like -- the docs describe
     bookmakers[].markets.match_odds.{home,draw,away} with opening/
     last_seen prices, but that's from thestatsapi.com's own docs page,
     never confirmed against a live call the way every other endpoint
     in this suite's scripts has been.
  3. Does it return real, current odds for the English leagues (user
     request, 2026-10-29: "Only English leagues") -- Premier League,
     Championship, League One, League Two -- or is coverage thin/empty
     for some of them?

Usage:
    python3 odds_1x2_diagnostic.py

Set THESTATSAPI_KEY as an environment variable first (same key your
other scripts already use).
"""

import os
import sys
import json
from datetime import date, timedelta

import requests

API_KEY = os.environ.get("THESTATSAPI_KEY", "PASTE_YOUR_KEY_HERE")
BASE_URL = "https://api.thestatsapi.com/api/football"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

ENGLISH_LEAGUES = [
    "Premier League",
    "Championship",
    "League One",
    "League Two",
]


def get(path, params=None):
    r = requests.get(f"{BASE_URL}{path}", headers=HEADERS, params=params or {}, timeout=20)
    print(f"  GET {path} params={params} -> HTTP {r.status_code}")
    if r.status_code != 200:
        print(f"    [!] non-200 response body (first 500 chars): {r.text[:500]}")
        return None
    try:
        return r.json()
    except Exception as e:
        print(f"    [!] response wasn't valid JSON: {e}")
        print(f"    raw body (first 500 chars): {r.text[:500]}")
        return None


def find_competition(name):
    # Same exact-match-preferred pattern already confirmed in
    # player_stat_model.py's find_competition -- a substring search can
    # match the wrong country's league of the same name.
    data = get("/competitions", {"search": name})
    if not data:
        return None
    results = data.get("data", [])
    if not results:
        print(f"    [!] no competition found for '{name}'")
        return None
    for comp in results:
        if comp.get("name", "").lower() == name.lower():
            return comp
    print(f"    [DIAG] no exact match for '{name}' among {len(results)} results -- using first: "
          f"{results[0].get('name')}")
    return results[0]


def get_upcoming_matches(competition_id, days_ahead=10):
    today = date.today()
    data = get("/matches", {
        "competition_id": competition_id,
        "date_from": today.isoformat(),
        "date_to": (today + timedelta(days=days_ahead)).isoformat(),
        "status": "scheduled",
        "per_page": 5,
    })
    if not data:
        return []
    return data.get("data", data.get("matches", []))


def main():
    if API_KEY == "PASTE_YOUR_KEY_HERE":
        print("[!] Set THESTATSAPI_KEY before running this.")
        sys.exit(1)

    for league_name in ENGLISH_LEAGUES:
        print(f"\n=== {league_name} ===")
        comp = find_competition(league_name)
        if not comp:
            continue
        comp_id = comp.get("id")
        print(f"  competition_id={comp_id} ({comp.get('name')})")

        matches = get_upcoming_matches(comp_id)
        if not matches:
            print(f"  [!] no upcoming scheduled matches found for {league_name} in the next 10 days")
            continue
        print(f"  {len(matches)} upcoming match(es) found, checking odds on the first one:")

        m = matches[0]
        match_id = m.get("id")
        home = m.get("home_team", {}).get("name", "?") if isinstance(m.get("home_team"), dict) else m.get("home_team", "?")
        away = m.get("away_team", {}).get("name", "?") if isinstance(m.get("away_team"), dict) else m.get("away_team", "?")
        print(f"  match_id={match_id}: {home} vs {away} on {m.get('date', '?')}")

        odds_data = get(f"/matches/{match_id}/odds")
        if odds_data is None:
            print(f"  [!] odds endpoint failed for match {match_id} -- see HTTP status/body above")
            continue

        print("  [DIAG] raw odds response (pretty-printed, full):")
        print(json.dumps(odds_data, indent=2)[:3000])

    print("\n\nDone. Paste this whole output back so we know whether to build the real model.")


if __name__ == "__main__":
    main()
