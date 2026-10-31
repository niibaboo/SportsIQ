#!/usr/bin/env python3
"""
Prop Finder Results Tracker
--------------------------------------------------------------
Same idea as every other tracker in this suite: log the picks BEFORE the
games are played, grade them later against what actually happened -- an
honest track record, not a screenshot.

WHY THIS EXISTS (user request, 2026-10-31): a long shots-on-target
accumulator lost with 7 of its 9 legs landing, and the question was how
often Prop Finder's picks really hit. That per-leg hit rate is the number
that decides how many legs a slip can realistically carry, so this
tracker measures it, per stat and per screen.

WHAT GETS LOGGED: on every Prop Finder run, every player/stat that
qualifies for the page's own "100%" screen (MIN_PERFECT_GAMES+ fetched
games, every one clearing the line) or "Real Streak" screen (a current
run of MIN_STREAK+ consecutive games clearing the line), at the page's
DEFAULT line for that stat (duplicated from prop_finder.html's
DEFAULT_LINES -- the page lets the user type any line, but a tracker
needs a fixed one). Each pick also stores the BASE RATE: how often ANY
player-game in the scanned data clears that same line. A pick only means
something if it beats that.

HOW RESULTS ARE VERIFIED -- NO EXTRA API CALLS: Prop Finder already
stores each player's per-game log, so a pick is graded from the next run's
data. A pick is graded on the player's EARLIEST game dated after the last
game the pick was based on (hit = the stat cleared the line). Picks with no
newer game in the data stay pending, and are voided after 30 days
(injury, transfer, dropped from the squad scan).

KNOWN LIMITATION: the game log only holds games a player appeared in
(0 minutes is dropped), so a missed match is invisible. A pick whose
player sat out the next match is graded on the next game he DID play. That
slightly misstates "next match", but it measures the same thing -- how
often a form pick clears -- and a bet on a non-playing player is normally
voided anyway.

Output:
    docs/prop-finder/results/log.json
    docs/prop-finder/results/index.html

Usage (called automatically from prop_finder.py's main block):
    import prop_finder_results_tracker
    prop_finder_results_tracker.run_results_tracker(players, stat_labels)
"""

import os
import json
import hashlib
import html as html_lib
from datetime import datetime, timezone, timedelta

LOG_PATH = "docs/prop-finder/results/log.json"
DASHBOARD_PATH = "docs/prop-finder/results/index.html"

# Duplicated from prop_finder.html (per this suite's self-contained-tool
# convention). If those defaults change there, change them here too.
DEFAULT_LINES = {
    "goals": 0.5,
    "shots": 1.5,
    "shots_on_target": 0.5,
    "tackles": 1.5,
    "fouls": 1.5,
    "cards": 0.5,
}
MIN_STREAK = 2
MIN_PERFECT_GAMES = 3

VOID_AFTER_DAYS = 30

CATEGORY_LABEL = {"perfect": "100%", "streak": "Real Streak"}


def load_log():
    if not os.path.exists(LOG_PATH):
        return []
    try:
        with open(LOG_PATH) as f:
            return json.load(f)
    except Exception as e:
        print(f"  [!] couldn't read existing results log ({e}) -- starting fresh. "
              f"The old file is still on disk if this needs investigating.")
        return []


def save_log(entries):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "w") as f:
        json.dump(entries, f, indent=2, default=str)


def _entry_id(category, player_id, stat, last_game_date):
    raw = f"{category}|{player_id}|{stat}|{last_game_date}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def base_rates(players):
    """{stat: fraction of ALL player-games in the data that clear that
    stat's default line} -- the bar a pick has to beat to mean anything."""
    out = {}
    for stat, line in DEFAULT_LINES.items():
        total = hits = 0
        for p in players:
            for g in p.get("game_log", []):
                total += 1
                if (g.get(stat) or 0) >= line:
                    hits += 1
        out[stat] = (hits / total) if total else None
    return out


def current_picks(player, stat, line):
    """Mirrors the page's two screens for one player/stat. game_log is
    newest-first. Returns a list of (category, last_game_date, size)."""
    games = player.get("game_log", [])
    if not games:
        return []
    last_date = games[0].get("date")
    if not last_date:
        return []
    picks = []
    if len(games) >= MIN_PERFECT_GAMES and all((g.get(stat) or 0) >= line for g in games):
        picks.append(("perfect", last_date, len(games)))
    run = 0
    for g in games:
        if (g.get(stat) or 0) >= line:
            run += 1
        else:
            break
    if run >= MIN_STREAK:
        picks.append(("streak", last_date, run))
    return picks


def log_todays_picks(players, log, now=None):
    now = now or datetime.now(timezone.utc)
    existing = {e["id"] for e in log}
    rates = base_rates(players)
    added = 0
    for p in players:
        for stat, line in DEFAULT_LINES.items():
            for category, last_date, size in current_picks(p, stat, line):
                eid = _entry_id(category, p.get("player_id"), stat, last_date)
                if eid in existing:
                    continue
                log.append({
                    "id": eid, "category": category, "player_id": p.get("player_id"),
                    "player": p.get("name"), "team": p.get("team"), "league": p.get("league"),
                    "stat": stat, "line": line, "size": size,
                    "last_game_date": last_date, "base_rate": rates.get(stat),
                    "logged_at": now.isoformat(), "status": "pending",
                    "result": None, "actual": None, "graded_game_date": None,
                })
                existing.add(eid)
                added += 1
    print(f"  Results log: {added} new pick(s) logged, {len(log)} total in log")
    return log


def verify_pending_results(players, log, now=None):
    now = now or datetime.now(timezone.utc)
    by_player = {str(p.get("player_id")): p for p in players}
    void_before = (now - timedelta(days=VOID_AFTER_DAYS)).isoformat()
    graded = voided = 0
    for e in log:
        if e["status"] != "pending":
            continue
        p = by_player.get(str(e["player_id"]))
        newer = [g for g in (p.get("game_log", []) if p else [])
                 if g.get("date") and g["date"] > e["last_game_date"]]
        if newer:
            g = min(newer, key=lambda x: x["date"])
            actual = g.get(e["stat"]) or 0
            e.update(status="verified", actual=actual, graded_game_date=g["date"],
                     result="hit" if actual >= e["line"] else "miss",
                     verified_at=now.isoformat())
            graded += 1
        elif e["logged_at"] < void_before:
            e["status"] = "void"
            voided += 1
    print(f"  Results verification: {graded} newly graded, {voided} voided")
    return log


def summarize(log):
    """Pure stats (kept apart from HTML so it can be tested)."""
    verified = [e for e in log if e["status"] == "verified"]
    groups = {}
    for e in verified:
        for key in (("all", e["category"]), (e["stat"], e["category"])):
            d = groups.setdefault(key, {"n": 0, "hits": 0, "base": 0.0, "base_n": 0})
            d["n"] += 1
            d["hits"] += e["result"] == "hit"
            if e.get("base_rate") is not None:
                d["base"] += e["base_rate"]
                d["base_n"] += 1
    for d in groups.values():
        d["rate"] = d["hits"] / d["n"]
        d["base_rate"] = (d["base"] / d["base_n"]) if d["base_n"] else None
    return {"groups": groups, "verified": verified,
            "pending": sum(1 for e in log if e["status"] == "pending")}


def _pct(x):
    return "—" if x is None else f"{x * 100:.0f}%"


def build_results_dashboard(log, stat_labels=None):
    s = summarize(log)
    esc = html_lib.escape
    stat_labels = stat_labels or {}
    g = s["groups"]

    def card(category):
        d = g.get(("all", category))
        title = CATEGORY_LABEL[category]
        if not d:
            return f'<div style="flex:1;text-align:center"><div style="font-size:11px;color:#888">{title}</div><div style="font-size:26px;color:#8b98a8">—</div></div>'
        return (f'<div style="flex:1;text-align:center"><div style="font-size:11px;color:#888">{title}</div>'
                f'<div style="font-size:26px;font-weight:bold;color:#a0e8a0">{_pct(d["rate"])}</div>'
                f'<div style="font-size:11px;color:#8b98a8">{d["hits"]}/{d["n"]} · base {_pct(d["base_rate"])}</div></div>')

    # Parlay maths off the per-leg hit rate (assumes legs are independent,
    # which same-match legs are not -- stated on the page).
    parlay = ""
    best = g.get(("all", "perfect")) or g.get(("all", "streak"))
    if best and best["n"] >= 1:
        r = best["rate"]
        cells = "".join(
            f'<div style="flex:1;text-align:center"><div style="font-size:11px;color:#888">{n} legs</div>'
            f'<div style="font-size:18px;font-weight:bold">{r ** n * 100:.0f}%</div></div>'
            for n in (2, 3, 5, 9))
        parlay = f"""<div style="background:#1a1f26;border-radius:12px;padding:16px;margin:14px 0;border:1px solid #2a3038">
  <div style="font-weight:bold;margin-bottom:2px">Chance every leg hits</div>
  <div style="font-size:11px;color:#8b98a8;margin-bottom:10px">At a {_pct(r)} per-leg hit rate. Assumes legs are independent — several players from one match tend to rise and fall together, so real odds are usually worse.</div>
  <div style="display:flex">{cells}</div>
</div>"""

    rows = ""
    for stat in DEFAULT_LINES:
        for category in ("perfect", "streak"):
            d = g.get((stat, category))
            if not d:
                continue
            edge = (d["rate"] - d["base_rate"]) * 100 if d["base_rate"] is not None else None
            edge_s = "" if edge is None else f' · {edge:+.0f}pp vs base'
            rows += f"""<div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #233040">
  <div><b>{esc(stat_labels.get(stat, stat))} {DEFAULT_LINES[stat]}+</b> <span style="color:#8b98a8;font-size:11px">{CATEGORY_LABEL[category]}</span><br><span style="color:#8b98a8;font-size:11px">base rate {_pct(d['base_rate'])}{edge_s}</span></div>
  <div style="text-align:right"><span style="color:#a0e8a0;font-weight:bold;font-size:16px">{_pct(d['rate'])}</span><br><span style="color:#8b98a8;font-size:11px">{d['hits']}/{d['n']}</span></div>
</div>"""

    recent = sorted(s["verified"], key=lambda e: e.get("verified_at", ""), reverse=True)[:30]
    recent_rows = ""
    for e in recent:
        color = "#22c55e" if e["result"] == "hit" else "#ef4444"
        recent_rows += f"""<div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #233040;font-size:12px">
  <div><b>{esc(str(e['player']))}</b> <span style="color:#8b98a8">{esc(str(e['team']))}</span><br><span style="color:#8b98a8">{esc(stat_labels.get(e['stat'], e['stat']))} {e['line']}+ · {CATEGORY_LABEL[e['category']]} ({e['size']}) · actual {e['actual']}</span></div>
  <span style="color:{color};font-weight:bold">{e['result'].upper()}</span>
</div>"""

    empty_recent = ('<p style="color:#888;font-size:12px">Nothing graded yet — a pick is graded on the '
                    "player's next game after the run that logged it.</p>")
    n_ver = len(s["verified"])
    note = ("Far too few graded picks to conclude anything yet — treat every number here as noise until there are a few hundred."
            if n_ver < 200 else "Enough graded picks to start trusting the pattern.")
    page = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Results — Prop Finder</title></head>
<body style="background:#0b0f14;color:white;font-family:Arial;padding:12px;max-width:600px;margin:auto">
<h2 style="text-align:center;margin-bottom:2px">📋 Prop Finder Results</h2>
<p style="text-align:center;color:#888;font-size:11px;margin-top:0">{datetime.now().strftime("%d %b %H:%M")} · {n_ver} graded · {s['pending']} pending</p>

<div style="background:#1a1f26;border-radius:12px;padding:16px;margin:14px 0;border:1px solid #2a3038">
  <div style="font-weight:bold;margin-bottom:8px">Per-leg hit rate, next game</div>
  <div style="display:flex">{card('perfect')}{card('streak')}</div>
  <div style="font-size:11px;color:#8b98a8;margin-top:10px;line-height:1.5">"Base" is how often ANY player-game clears the same line. A screen only adds value if it beats that. {note}</div>
</div>
{parlay}
<div style="background:#1a1f26;border-radius:12px;padding:16px;margin:14px 0;border:1px solid #2a3038">
  <div style="font-weight:bold;margin-bottom:8px">By Stat</div>
  {rows or '<p style="color:#888;font-size:12px">No graded picks yet.</p>'}
</div>
<div style="background:#1a1f26;border-radius:12px;padding:16px;margin:14px 0;border:1px solid #2a3038">
  <div style="font-weight:bold;margin-bottom:8px">Recent Results</div>
  {recent_rows or empty_recent}
</div>
<div style="font-size:11px;color:#8b98a8;text-align:center;margin-top:20px;line-height:1.6">
  Picks are logged at the page's default lines before the games are played and graded on the player's
  next recorded game. Games a player missed are invisible, so a pick can grade on a later game than the
  very next match.<br><a href="../index.html" style="color:#7ec8ff">← Back to Prop Finder</a>
</div>
</body></html>"""

    os.makedirs(os.path.dirname(DASHBOARD_PATH), exist_ok=True)
    with open(DASHBOARD_PATH, "w") as f:
        f.write(page)
    print(f"  Results dashboard: {n_ver} graded pick(s)" if n_ver else "  Results dashboard: nothing graded yet")


def run_results_tracker(players, stat_labels=None):
    """Single entry point called from prop_finder.py's main block."""
    print("\nRunning results tracker...")
    log = load_log()
    # Grade first so a pick logged on a previous run is settled by THIS
    # run's newer games before today's picks are added.
    log = verify_pending_results(players, log)
    log = log_todays_picks(players, log)
    save_log(log)
    build_results_dashboard(log, stat_labels)
