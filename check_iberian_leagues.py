#!/usr/bin/env python3
"""
Debug script for Primeira Liga and La Liga -- both started failing to
resolve via TheStatsAPI's competition search (previously worked fine).
Dumps the raw search results for several name variants of each, plus a
broad country-based search, to find out what changed and pin down the
real current name(s).

Reads THESTATSAPI_KEY from the environment (same secret already used by
Match IQ's own workflow) -- designed to run via GitHub Actions, not a
local terminal.
"""

import os
import sys
import json
import requests

BASE = "https://api.thestatsapi.com/api"

PORTUGAL_NAMES = ["Primeira Liga", "Liga Portugal", "Liga Portugal Betclic", "Primeira Divisao"]
SPAIN_NAMES = ["La Liga", "LaLiga", "LALIGA EA SPORTS", "Primera Division"]


def _get(path, key, params=None):
    r = requests.get(f"{BASE}{path}", headers={"Authorization": f"Bearer {key}"},
                      params=params or {}, timeout=15)
    if r.status_code != 200:
        print(f"  [!] {r.status_code} on {path}: {r.text[:200]}")
        return None
    return r.json()


def check_name(name, key):
    print(f"\n  Searching: {name!r}")
    data = _get("/football/competitions", key, params={"search": name, "per_page": 5})
    if not data or not data.get("data"):
        print("    NOT FOUND")
        return
    for c in data["data"]:
        print(f"    id={c.get('id')}  name={c.get('name')!r}  country={c.get('country')}")


def main():
    key = os.environ.get("THESTATSAPI_KEY") or (sys.argv[1] if len(sys.argv) > 1 else None)
    if not key:
        print("Set THESTATSAPI_KEY or pass it as an argument.")
        sys.exit(1)

    print("#" * 70)
    print("# PORTUGAL")
    print("#" * 70)
    for name in PORTUGAL_NAMES:
        check_name(name, key)

    print("\n" + "#" * 70)
    print("# SPAIN")
    print("#" * 70)
    for name in SPAIN_NAMES:
        check_name(name, key)

    print("\n" + "#" * 70)
    print("# BROAD 'Portugal' / 'Spain' searches (in case the name changed entirely)")
    print("#" * 70)
    for country_term in ["Portugal", "Spain"]:
        print(f"\n  Searching: {country_term!r}")
        data = _get("/football/competitions", key, params={"search": country_term, "per_page": 10})
        if not data or not data.get("data"):
            print("    NOT FOUND")
            continue
        for c in data["data"]:
            print(f"    id={c.get('id')}  name={c.get('name')!r}  country={c.get('country')}")


if __name__ == "__main__":
    main()
