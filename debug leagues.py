"""
Quick sanity check: list every league Highlightly's hockey API knows about,
so we can confirm what string/ID it actually expects for SHL (and spot any
other name mismatches before they silently return 0 fixtures).

Usage:
    export HIGHLIGHTLY_KEY=your-key-here
    python3 debug_leagues.py

    # optionally filter:
    python3 debug_leagues.py sweden
    python3 debug_leagues.py shl
"""

import os
import sys
import requests

BASE_URL = "https://hockey.highlightly.net"
API_KEY = os.environ.get("HIGHLIGHTLY_KEY")

if not API_KEY:
    print("ERROR: HIGHLIGHTLY_KEY is not set in this shell.")
    print("Run: export HIGHLIGHTLY_KEY=your-key-here")
    sys.exit(1)

# Also fetch today's matches for a given league to spot-check whether "0 fixtures"
# means the season hasn't started, or the request is malformed. Call as:
#   python3 debug_leagues.py --matches SHL 2026-09-18
if len(sys.argv) > 1 and sys.argv[1] == "--matches":
    league_name = sys.argv[2] if len(sys.argv) > 2 else "SHL"
    date = sys.argv[3] if len(sys.argv) > 3 else None

    def check_matches():
        headers = {"x-rapidapi-key": API_KEY}
        params = {"leagueName": league_name}
        if date:
            params["date"] = date
        r = requests.get(f"{BASE_URL}/matches", headers=headers, params=params)
        r.raise_for_status()
        resp = r.json()
        matches = resp.get("data", [])
        print(f"leagueName={league_name!r} date={date!r} -> {len(matches)} match(es)")
        for m in matches:
            print(
                f"  {m['date']}  {m['homeTeam']['name']} vs {m['awayTeam']['name']}  "
                f"[{m['state']['description']}]  season={m['league'].get('season')}"
            )
        if not matches:
            print("  (empty - try omitting 'date' to see if ANY matches exist for this league at all)")

    check_matches()
    sys.exit(0)


# Confirm mode: checks all 4 target leagues by name and shows EVERY result
# Highlightly returns for that name, with country attached. This is the
# direct fix for the Extraliga (Czech vs Belarus) mix-up — if a name has
# more than one country-match, this prints all of them so the ambiguity
# is visible up front instead of silently picking the wrong one later.
#   python3 debug_leagues.py --confirm
TARGET_LEAGUES = [
    ("SHL", "Sweden"),
    ("Extraliga", "Czech Republic"),   # expected — confirm exact string below
    ("DEL", "Germany"),
    ("National League", "Switzerland"),  # bet365 shows "Switzerland NLA";
                                          # Highlightly's own name TBD - this
                                          # run is what confirms it
]

if len(sys.argv) > 1 and sys.argv[1] == "--confirm":
    def _get_confirm(path, params=None):
        headers = {"x-rapidapi-key": API_KEY}
        r = requests.get(f"{BASE_URL}{path}", headers=headers, params=params or {})
        r.raise_for_status()
        return r.json()

    print("Confirming exact league name + country for each target...\n")
    for name, expected_country in TARGET_LEAGUES:
        print(f"Searching leagueName={name!r} (expecting country={expected_country!r})")
        try:
            data = _get_confirm("/leagues", {"leagueName": name})
        except requests.HTTPError as e:
            print(f"  [!] request failed: {e}")
            continue
        results = data.get("data", []) if isinstance(data, dict) else data
        if not results:
            print("  (no results at all for this name - try a broader/partial name)")
            continue
        for lg in results:
            lg_name = lg.get("name", "")
            country_obj = lg.get("country") or {}
            country = country_obj.get("name", "") if isinstance(country_obj, dict) else str(country_obj)
            lg_id = lg.get("id", "")
            flag = "  <-- MATCHES expected country" if country.lower() == expected_country.lower() else "  *** DIFFERENT COUNTRY - do not use blindly ***"
            print(f"  id={lg_id}  name={lg_name!r}  country={country!r}{flag}")
        print()
    sys.exit(0)



    # Per the official docs, the required header is x-rapidapi-key
    # (x-rapidapi-host is only needed if you're calling through RapidAPI's
    # host instead of hockey.highlightly.net directly).
    headers = {"x-rapidapi-key": API_KEY}
    r = requests.get(f"{BASE_URL}{path}", headers=headers, params=params or {})
    r.raise_for_status()
    return r.json()


def main():
    query = sys.argv[1].lower() if len(sys.argv) > 1 else None

    print("Fetching full league list from /leagues (no filter)...\n")
    data = _get("/leagues")

    # /leagues always returns {"data": [...], "pagination": {...}, "plan": {...}}
    leagues = data.get("data", []) if isinstance(data, dict) else data

    if not leagues:
        print("No leagues returned at all - check the endpoint/auth, not just the name.")
        return

    print(f"{len(leagues)} leagues total (this page; check 'pagination' if truncated).\n")
    print(f"{'ID':<10} {'Name':<30} {'Country'}")
    print("-" * 60)

    for lg in leagues:
        name = str(lg.get("name") or "")
        country_obj = lg.get("country") or {}
        country = country_obj.get("name", "") if isinstance(country_obj, dict) else str(country_obj)
        lg_id = str(lg.get("id") or "")
        seasons = [s.get("season") for s in lg.get("seasons", [])]

        if query and query not in name.lower() and query not in country.lower():
            continue

        print(f"{lg_id:<10} {name:<30} {country:<15} seasons={seasons}")

    if query:
        print(f"\n(filtered to rows matching '{query}')")
    else:
        print(
            "\nNote: /leagues defaults to limit=100 - if SHL/DEL/etc aren't shown above, "
            "re-run with a query filter, or add offset=100, 200... to page through the rest."
        )


if __name__ == "__main__":
    main()
