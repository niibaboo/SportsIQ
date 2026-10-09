#!/usr/bin/env python3
"""
Player Stat Model Results Tracker
--------------------------------------------------------------
Same architecture as Match IQ / Euro Ice's trackers, adapted for
Player Stat Model's per-PLAYER-prop leg structure. Logs every leg that
carries a resolvable match_id (only legs built from the automated
daily fixture scan do — see build_legs()/scan_team() in
player_stat_model.py; watchlist/manual lookups aren't tied to one
fixture and are skipped here, the same way Euro Ice skips Game Total
legs it can't verify a single side of).

On later runs, once a logged pick's match has had time to finish,
looks the player's row back up via TheStatsAPI's
/matches/{match_id}/player-stats endpoint (the same endpoint
player_stat_model.py already uses for rolling form) and marks hit/miss
against the real match stat.

Designed to be imported and called from player_stat_model.py — see
run_results_tracker(). Not run standalone.

Output:
    docs/player-stat-model/results/log.json    -- the full log
    docs/player-stat-model/results/index.html  -- dashboard: overall +
                                                   per-market + per-league
                                                   win rate, recent history
"""

import os
import json
import hashlib
import requests
from datetime import datetime, timezone, date

BASE_URL = "https://api.thestatsapi.com/api/football"
LOG_PATH = "docs/player-stat-model/results/log.json"
DASHBOARD_PATH = "docs/player-stat-model/results/index.html"

# prop_key -> (stat extractor over a /matches/{id}/player-stats row,
# line, comparator). Mirrors STAT_FIELDS + PROP_LINE_MAP in
# player_stat_model.py exactly — same field paths, same "to be
# carded"/"to score"/"to assist"/"goal or assist" >=1 semantics —
# duplicated here (not imported) so this module has no circular
# dependency on player_stat_model.py, same pattern already used by
# euro_ice_results_tracker.py (its own _get()/parse_score() rather
# than importing euro_ice.py).
def _shots(row): return row.get("shooting", {}).get("total_shots", 0)
def _sot(row): return row.get("shooting", {}).get("shots_on_target", 0)
def _cards(row): return row.get("general", {}).get("yellow_cards", 0) + row.get("general", {}).get("red_cards", 0)
def _goals(row): return row.get("shooting", {}).get("goals", 0)
def _assists(row): return row.get("passing", {}).get("assists", 0)
def _tackles(row): return row.get("defending", {}).get("tackles", 0)
def _fouls(row): return row.get("general", {}).get("fouls", 0)
def _goal_or_assist(row): return _goals(row) + _assists(row)

PROP_VERIFY_MAP = {
    "shots_over_1.5": (_shots, 1.5),
    "shots_over_2.5": (_shots, 2.5),
    "sot_over_0.5": (_sot, 0.5),
    "sot_over_1.5": (_sot, 1.5),
    "to_be_carded": (_cards, 0.5),
    "to_score": (_goals, 0.5),
    "to_assist": (_assists, 0.5),
    "goal_or_assist": (_goal_or_assist, 0.5),
    "tackles_over_1.5": (_tackles, 1.5),
    "tackles_over_2.5": (_tackles, 2.5),
    "fouls_over_1.5": (_fouls, 1.5),
    "fouls_over_2.5": (_fouls, 2.5),
}

# stat_key (as used by STREAK_CONFIG / build_hot_form_entries /
# build_real_streak_entries in player_stat_model.py) -> same extractor
# functions above. Hot Form / Real Streak entries carry a stat_key, not
# a prop_key, since they're not tied to one of the fixed prop lines --
# kept as a separate map rather than folding into PROP_VERIFY_MAP so a
# stat_key never accidentally resolves against the wrong (prop-line)
# comparison semantics (see verify_pending_results' >= vs > distinction
# below).
STAT_KEY_EXTRACTOR = {
    "shots": _shots,
    "shots_on_target": _sot,
    "tackles": _tackles,
    "fouls": _fouls,
    "goal_or_assist": _goal_or_assist,
}

SCANNER_LABELS = {
    "prop": "Prop Picks",
    "player_hot_form": "Hot Form",
    "player_real_streak": "Real Streak",
}


def _get_match_player_row(match_id, player_id, headers):
    try:
        r = requests.get(f"{BASE_URL}/matches/{match_id}/player-stats",
                          headers=headers, timeout=15)
    except Exception as e:
        print(f"    [!] verification request failed: match {match_id} ({e})")
        return None
    if r.status_code != 200:
        # Normal while the match hasn't been played/ingested yet — not
        # an error worth logging loudly, leave pending for next run.
        return None
    data = r.json()
    rows = data.get("data", data.get("player_stats", []))
    for row in rows:
        if str(row.get("player_id")) == str(player_id):
            return row
    return None


def _entry_id(player_id, key, match_id):
    """key is a prop_key for a prop leg, or "{scanner}|{stat_key}" for a
    Hot Form/Real Streak entry -- the scanner prefix is what stops a
    player qualifying for BOTH Hot Form and Real Streak in the same stat
    on the same match from colliding onto one log entry (same bug class
    already caught and fixed in Orange Line's tracker: without something
    stat/scanner-specific in the id, same player+match+prop collapses
    multiple real picks into "already logged" duplicates)."""
    raw = f"{player_id}|{key}|{match_id}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def load_log():
    if not os.path.exists(LOG_PATH):
        return []
    try:
        with open(LOG_PATH) as f:
            return json.load(f)
    except Exception as e:
        print(f"  [!] Couldn't read existing results log ({e}) -- starting fresh.")
        return []


def save_log(entries):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "w") as f:
        json.dump(entries, f, indent=2, default=str)


def log_todays_signals(legs, hot_form, real_streak, log):
    """Only legs/entries carrying a match_id (built from the automated
    daily fixture scan) can be verified against a specific later result
    — watchlist/manual-lookup legs have no single fixture to check and
    are skipped, same reasoning Euro Ice uses to skip Game Total legs."""
    existing_ids = {e["id"] for e in log}
    added = 0
    for leg in legs:
        if not leg.get("match_id") or not leg.get("prop_key"):
            continue
        eid = _entry_id(leg.get("player_id"), leg["prop_key"], leg["match_id"])
        if eid in existing_ids:
            continue
        date_key = (leg.get("match_date") or "")[:10]
        log.append({
            "id": eid,
            "scanner": "prop",
            "player": leg["player"],
            "player_id": leg.get("player_id"),
            "prop_key": leg["prop_key"],
            "market": leg["market"],
            "category": leg["category"],
            "prob": leg["prob"],
            "detail": leg.get("detail"),
            "opponent": leg.get("opponent"),
            "match_id": leg["match_id"],
            "match_date": leg.get("match_date"),
            "date_key": date_key,
            "logged_at": datetime.now(timezone.utc).isoformat(),
            "status": "pending", "result": None, "actual": None,
        })
        existing_ids.add(eid)
        added += 1

    def _add_form(entries, scanner):
        nonlocal added
        for e in entries or []:
            if not e.get("match_id") or not e.get("stat_key"):
                continue
            # scanner-prefixed key -- see _entry_id's docstring for why.
            eid = _entry_id(e.get("player_id"), f"{scanner}|{e['stat_key']}", e["match_id"])
            if eid in existing_ids:
                continue
            date_key = (e.get("match_date") or "")[:10]
            value = e.get("avg") if scanner == "player_hot_form" else e.get("streak")
            kind = "Hot Form" if scanner == "player_hot_form" else "Real Streak"
            log.append({
                "id": eid,
                "scanner": scanner,
                "player": e["player"],
                "player_id": e.get("player_id"),
                "stat_key": e["stat_key"],
                "market": f"{e['player']} {e['label']} {kind}",
                "category": e["label"],
                "value": value,
                "threshold": e.get("threshold"),
                "detail": f"{e.get('team', 'Unknown team')} · {e.get('league', 'Unknown league')}",
                "opponent": e.get("opponent"),
                "match_id": e["match_id"],
                "match_date": e.get("match_date"),
                "date_key": date_key,
                "logged_at": datetime.now(timezone.utc).isoformat(),
                "status": "pending", "result": None, "actual": None,
            })
            existing_ids.add(eid)
            added += 1

    _add_form(hot_form, "player_hot_form")
    _add_form(real_streak, "player_real_streak")

    print(f"  Results log: {added} new pick(s) logged, {len(log)} total in log")
    return log


def verify_pending_results(log, headers, max_checks=80):
    """Only attempts entries whose match date has already passed —
    same guard other trackers use, since /player-stats for a match
    that hasn't been played yet correctly returns nothing to find."""
    today = date.today().isoformat()
    checked = 0
    updated = 0

    for entry in log:
        if entry["status"] != "pending":
            continue
        if entry["date_key"] >= today:
            continue
        if checked >= max_checks:
            break
        checked += 1

        try:
            row = _get_match_player_row(entry["match_id"], entry["player_id"], headers)
        except Exception as e:
            print(f"    [!] verification error for entry {entry['id']}: {e}")
            row = None

        if row is None:
            continue  # match not finished/ingested yet — leave pending

        scanner = entry.get("scanner", "prop")
        if scanner == "prop":
            extractor, line = PROP_VERIFY_MAP.get(entry["prop_key"], (None, None))
            if extractor is None:
                continue
            actual = extractor(row)
            result = "hit" if actual > line else "miss"
        else:
            # Hot Form / Real Streak both ask "did THIS game clear the
            # threshold too" — >= matches build_hot_form_entries()'s and
            # build_real_streak_entries()'s own qualifying comparison in
            # player_stat_model.py, not the prop legs' plain > used for
            # fixed prop lines above.
            extractor = STAT_KEY_EXTRACTOR.get(entry.get("stat_key"))
            if extractor is None:
                continue
            actual = extractor(row)
            result = "hit" if actual >= entry["threshold"] else "miss"

        entry["status"] = "verified"
        entry["result"] = result
        entry["actual"] = actual
        entry["verified_at"] = datetime.now(timezone.utc).isoformat()
        updated += 1

    print(f"  Results verification: checked {checked} pending entries, {updated} newly verified")
    return log


def build_results_dashboard(log):
    verified = [e for e in log if e["status"] == "verified"]
    pending = [e for e in log if e["status"] == "pending"]

    # Only prop-scanner entries carry a prop_key -- Hot Form/Real Streak
    # entries use stat_key instead (see log_todays_signals), so this
    # grouping is scoped to scanner == "prop" rather than assuming every
    # verified entry has a prop_key.
    by_prop = {}
    for e in verified:
        if e.get("scanner", "prop") != "prop":
            continue
        d = by_prop.setdefault(e["prop_key"], {"hit": 0, "miss": 0})
        d[e["result"]] += 1

    # By-category breakdown (same grouping build_legs() already uses:
    # Shots, Shots on Target, Cards, Goals, Assists, Tackles, Fouls,
    # Goal or Assist) — coarser than per-prop, easier to read at a glance.
    # Hot Form/Real Streak entries share these same category labels
    # (Shots, Tackles, etc. — see STREAK_CONFIG), so this mixes prop
    # picks and streak picks together here; the By Scanner breakdown
    # below is what keeps them apart.
    by_category = {}
    for e in verified:
        d = by_category.setdefault(e.get("category") or "Unknown", {"hit": 0, "miss": 0})
        d[e["result"]] += 1

    # By-scanner breakdown — Prop Picks (the fixed Poisson-priced lines)
    # vs. Hot Form (rolling average clears a line) vs. Real Streak
    # (genuinely consecutive games clearing a line). Kept separate from
    # By Category above since all three scanners can tag the same
    # category label (e.g. "Shots").
    by_scanner = {}
    for e in verified:
        d = by_scanner.setdefault(e.get("scanner", "prop"), {"hit": 0, "miss": 0})
        d[e["result"]] += 1

    # OVERALL covers every verified entry regardless of scanner -- summing
    # by_prop here would silently exclude Hot Form/Real Streak picks from
    # the headline number now that by_prop is scoped to scanner == "prop".
    total_hit = sum(1 for e in verified if e["result"] == "hit")
    total_miss = sum(1 for e in verified if e["result"] == "miss")
    total = total_hit + total_miss
    overall_pct = round(100 * total_hit / total) if total else None

    def _rows(grouping):
        out = ""
        for label, d in sorted(grouping.items(), key=lambda kv: -(kv[1]["hit"] + kv[1]["miss"])):
            n = d["hit"] + d["miss"]
            pct = round(100 * d["hit"] / n) if n else None
            pct_str = f"{pct}%" if pct is not None else "—"
            out += f"""<div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid var(--border)">
  <span>{label}</span>
  <span style="color:var(--green);font-weight:bold">{pct_str}</span>
  <span style="color:var(--sub);font-size:12px">{d['hit']}/{n}</span>
</div>"""
        return out or '<p style="color:var(--sub);font-size:12px">No verified picks yet.</p>'

    category_rows = _rows(by_category)
    scanner_rows = _rows({SCANNER_LABELS.get(k, k): v for k, v in by_scanner.items()})

    recent = sorted(verified, key=lambda e: e.get("verified_at", ""), reverse=True)[:30]
    recent_rows = ""
    for e in recent:
        color = "#22c55e" if e["result"] == "hit" else "#ef4444"
        recent_rows += f"""<div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid var(--border);font-size:12px">
  <span>{e['market']}</span>
  <span style="color:{color};font-weight:bold">{e['result'].upper()}</span>
</div>"""

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Results — Player Stat Model</title>
<style>:root{{--bg:#0b0f14;--panel:#121820;--border:#233040;--text:#e8edf2;--sub:#8b98a8;--green:#22c55e;}}</style></head>
<body style="background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;padding:16px;max-width:640px;margin:0 auto">
<p style="text-align:center;margin-bottom:6px"><a href="../index.html" style="color:#7ec8ff;text-decoration:none;font-size:12px">← Player Stat Model</a></p>
<h2 style="text-align:center;margin-bottom:2px">📊 Results Tracker</h2>
<p style="text-align:center;color:var(--sub);font-size:11px;margin-top:0">{datetime.now().strftime("%d %b %H:%M")} · every pick from the daily fixture scan, auto-verified against real match stats</p>

<div style="background:var(--panel);border-radius:12px;padding:16px;margin:14px 0;border:1px solid var(--border);text-align:center">
  <div style="font-size:11px;color:var(--sub)">OVERALL</div>
  <div style="font-size:32px;font-weight:bold;color:var(--green)">{overall_pct if overall_pct is not None else "—"}{"%" if overall_pct is not None else ""}</div>
  <div style="font-size:12px;color:var(--sub)">{total_hit}/{total} verified picks · {len(pending)} pending (match not finished yet)</div>
</div>

<div style="background:var(--panel);border-radius:12px;padding:16px;margin:14px 0;border:1px solid var(--border)">
  <div style="font-weight:bold;margin-bottom:8px">By Scanner</div>
  {scanner_rows}
</div>

<div style="background:var(--panel);border-radius:12px;padding:16px;margin:14px 0;border:1px solid var(--border)">
  <div style="font-weight:bold;margin-bottom:8px">By Category</div>
  {category_rows}
</div>

<div style="background:var(--panel);border-radius:12px;padding:16px;margin:14px 0;border:1px solid var(--border)">
  <div style="font-weight:bold;margin-bottom:8px">Recent Results</div>
  {recent_rows or '<p style="color:var(--sub);font-size:12px">Nothing verified yet — check back after a few days of picks have had time to play out.</p>'}
</div>

<div style="font-size:11px;color:var(--sub);text-align:center;margin-top:20px;line-height:1.6">
  Only picks from the automated daily fixture scan are tracked here — a
  watchlist or single-player lookup isn't tied to one specific match, so
  there's nothing to verify it against. Sample sizes are small early on;
  treat percentages with real caution until there's a few weeks of picks.
</div>
</body></html>"""

    os.makedirs(os.path.dirname(DASHBOARD_PATH), exist_ok=True)
    with open(DASHBOARD_PATH, "w") as f:
        f.write(html)
    print(f"  Results dashboard: {total} verified, {overall_pct}% overall" if total else "  Results dashboard: no verified picks yet")


def run_results_tracker(legs, hot_form, real_streak, headers):
    """Single entry point called from player_stat_model.py's --auto path."""
    print("\nRunning results tracker...")
    log = load_log()
    log = log_todays_signals(legs, hot_form, real_streak, log)
    log = verify_pending_results(log, headers)
    save_log(log)
    build_results_dashboard(log)
