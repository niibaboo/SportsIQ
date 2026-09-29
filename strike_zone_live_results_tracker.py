#!/usr/bin/env python3
"""
Strike Zone Live — Results Tracker

Same pattern as Blitz IQ Live's tracker: log each qualifying in-game signal
once (the first time it hits >=70% for a game), verify it later against
what actually happened, build a win-rate dashboard.

  - Next Run signals ("Home Next Run" / "Away Next Run") are verified from
    the FINAL linescore's inning-by-inning breakdown: build the sequence of
    half-innings (top of 1, bottom of 1, top of 2, ...) with runs scored in
    each, find the first half STRICTLY AFTER the logged half-inning that
    has runs > 0, and check which side that half belonged to. If no later
    half-inning scored, it's "void" (nothing to grade), not a loss.

  - Game Total Pace signals ("Game Total Over" / "Game Total Under") are
    verified against the pregame projection (sum of both teams' lambdas
    from build_slate.py) vs. the actual final combined runs. An exact
    push is void.

  - Inning Winner / Inning Most Hits signals (e.g. "5th Inning Winner:
    Atlanta Braves") are verified against that SPECIFIC inning's entry
    in the final linescore's innings array (runs for Winner, hits for
    Most Hits) -- home vs away in that inning alone, not the whole game.
    An exact tie in that inning is void ONLY if the signal itself wasn't
    predicting a tie (a correctly-predicted tie is a hit). If the game
    ended before reaching that inning, it's void -- nothing to grade.

Wrapped in try/except by strike_zone_live.py's __main__, same as every
other tracker in the suite.
"""

import os
import json
import time
from datetime import datetime, timezone

import requests

BASE = "https://statsapi.mlb.com/api/v1"
LOG_PATH = "docs/strike-zone-live/results_log.json"
DASHBOARD_PATH = "docs/strike-zone-live/results/index.html"
MAX_CHECKS_PER_RUN = 40


def _get(url, params=None, timeout=15):
    try:
        r = requests.get(url, params=params or {}, timeout=timeout)
        time.sleep(0.5)
        if r.status_code != 200:
            print(f"    [!] {r.status_code} on {url}")
            return None
        return r.json()
    except Exception as e:
        time.sleep(0.5)
        print(f"    [!] request failed: {url} ({e})")
        return None


def load_log():
    if not os.path.exists(LOG_PATH):
        return {}
    try:
        with open(LOG_PATH) as f:
            return json.load(f)
    except Exception as e:
        print(f"  [!] couldn't load existing log, starting fresh: {e}")
        return {}


def save_log(log):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, 'w') as f:
        json.dump(log, f, indent=2, default=str)


def _entry_id(game_pk, label):
    return f"{game_pk}_{label.replace(' ', '_')}"


def log_todays_signals(live_games, log):
    added = 0
    for g in live_games:
        signal = g.get('model_signal') or {}
        if not signal.get('qualifies') or not signal.get('label'):
            continue

        eid = _entry_id(g['game_pk'], signal['label'])
        if eid in log:
            continue

        # half-innings elapsed AT LOG TIME, from inning/inning_state, using
        # the same convention as strike_zone_live.py's half_innings_elapsed
        inning, state = g.get('inning'), g.get('inning_state')
        base = ((inning or 1) - 1) * 2
        if state == 'Top':
            logged_half_index = base
        elif state in ('Middle', 'Bottom'):
            logged_half_index = base + 1
        elif state == 'End':
            logged_half_index = base + 2
        else:
            logged_half_index = base

        log[eid] = {
            'id': eid,
            'game_pk': g['game_pk'],
            'home_team': g['home_team'],
            'away_team': g['away_team'],
            'label': signal['label'],
            'prob': signal['prob'],
            'logged_at': datetime.now(timezone.utc).isoformat(),
            'logged_inning': inning,
            'logged_inning_state': state,
            'logged_half_index': logged_half_index,
            'logged_home_runs': g.get('home_runs'),
            'logged_away_runs': g.get('away_runs'),
            'pregame_total': (g.get('pregame_projection') or {}).get('pregame_total'),
            'status': 'pending',
        }
        added += 1
    if added:
        print(f"  Logged {added} new signal(s)")


def _build_half_inning_sequence(innings):
    """From MLB's linescore.innings array (each with away.runs/home.runs),
    build [(half_index, side, runs), ...] in chronological order: top of
    inning 1 is half_index 0 (away bats), bottom of inning 1 is half_index
    1 (home bats), and so on."""
    sequence = []
    for i, inn in enumerate(innings or []):
        away_runs = (inn.get('away') or {}).get('runs')
        home_runs = (inn.get('home') or {}).get('runs')
        if away_runs is not None:
            sequence.append((i * 2, 'away', away_runs))
        if home_runs is not None:
            sequence.append((i * 2 + 1, 'home', home_runs))
    return sequence


def _verify_next_run_entry(entry, linescore):
    logged_half_index = entry.get('logged_half_index')
    if logged_half_index is None:
        return 'void'

    predicted_side = 'home' if 'Home' in entry['label'] else 'away'
    sequence = _build_half_inning_sequence(linescore.get('innings'))

    for half_index, side, runs in sequence:
        if half_index > logged_half_index and runs and runs > 0:
            return 'hit' if side == predicted_side else 'miss'

    return 'void'  # nobody scored again after the signal fired


def _verify_inning_market_entry(entry, linescore):
    """Grades a "{Nth} Inning Winner: <team>" or "{Nth} Inning Most Hits:
    <team>" signal against that SPECIFIC inning's linescore entry, not
    the game total. entry['logged_inning'] is the inning number the
    signal was about (1-indexed, matching MLB's inning numbering)."""
    logged_inning = entry.get('logged_inning')
    if not logged_inning:
        return 'void'

    innings = linescore.get('innings') or []
    idx = logged_inning - 1
    if idx < 0 or idx >= len(innings):
        return 'void'  # game ended before/without reaching that inning

    inn = innings[idx]
    stat_key = 'hits' if 'Most Hits' in entry['label'] else 'runs'
    home_val = (inn.get('home') or {}).get(stat_key)
    away_val = (inn.get('away') or {}).get(stat_key)
    if home_val is None or away_val is None:
        return 'void'

    if home_val > away_val:
        actual_side = 'home'
    elif away_val > home_val:
        actual_side = 'away'
    else:
        actual_side = 'tie'

    if 'Tie' in entry['label']:
        predicted_side = 'tie'
    elif entry.get('home_team') and entry['home_team'] in entry['label']:
        predicted_side = 'home'
    elif entry.get('away_team') and entry['away_team'] in entry['label']:
        predicted_side = 'away'
    else:
        return 'void'  # couldn't determine which side was predicted

    return 'hit' if predicted_side == actual_side else 'miss'


def _verify_game_total_entry(entry, final_home, final_away):
    pregame_total = entry.get('pregame_total')
    if final_home is None or final_away is None or pregame_total is None:
        return 'void'

    final_total = final_home + final_away
    if final_total == pregame_total:
        return 'void'

    if 'Over' in entry['label']:
        return 'hit' if final_total > pregame_total else 'miss'
    return 'hit' if final_total < pregame_total else 'miss'


def verify_pending_results(log):
    pending = [e for e in log.values() if e['status'] == 'pending']
    checked = 0
    graded = 0

    for entry in pending:
        if checked >= MAX_CHECKS_PER_RUN:
            break
        checked += 1

        data = _get(f"{BASE}/schedule", params={"gamePk": entry['game_pk'], "hydrate": "linescore"})
        if not data:
            continue
        try:
            game = data["dates"][0]["games"][0]
        except (KeyError, IndexError):
            continue

        if game.get('status', {}).get('abstractGameState') != 'Final':
            continue  # still in progress -- try again next run

        linescore = game.get('linescore') or {}
        teams = linescore.get('teams') or {}
        try:
            final_home = float((teams.get('home') or {}).get('runs'))
            final_away = float((teams.get('away') or {}).get('runs'))
        except (TypeError, ValueError):
            final_home, final_away = None, None

        if 'Next Run' in entry['label']:
            result = _verify_next_run_entry(entry, linescore)
        elif 'Inning Winner' in entry['label'] or 'Inning Most Hits' in entry['label']:
            result = _verify_inning_market_entry(entry, linescore)
        else:
            result = _verify_game_total_entry(entry, final_home, final_away)

        entry['status'] = result
        entry['final_home_runs'] = final_home
        entry['final_away_runs'] = final_away
        graded += 1

    if checked:
        print(f"  Checked {checked} pending signal(s), graded {graded}")


DASHBOARD_TEMPLATE = """<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Strike Zone Live — Results</title></head>
<body style="background:#0b0f14;color:white;font-family:Arial;padding:12px;max-width:700px;margin:auto">
<h2 style="text-align:center">📊 Strike Zone Live — Results Tracker</h2>
<p style="text-align:center;color:#888;font-size:11px">Every qualifying live signal (≥70%), logged the instant it fired and graded against what actually happened · {generated}</p>
<p style="text-align:center;margin-bottom:16px"><a href="../index.html" style="color:#7ec8ff;text-decoration:none;font-size:12px">⚡ Back to Live Signals</a></p>
{summary}
{rows}
<p style="text-align:center;color:#666;font-size:10px;margin-top:20px">
Next Run signals grade against the actual next half-inning with a run after the signal fired; if nobody scores again, it's marked Void (not a loss). Game Total signals grade against the pregame run projection; an exact push is Void. Inning Winner / Inning Most Hits signals grade against that SPECIFIC inning's actual runs/hits; Void means the game ended before reaching that inning.
</p>
</body></html>"""

CATEGORY_SUMMARY_ROW = """<div style="display:flex;justify-content:space-between;padding:6px 0;border-top:1px solid #232a33">
  <div>{label}</div>
  <div><span style="color:#a0e8a0;font-weight:bold">{hits}</span>/<span style="color:#888">{total}</span>
  <span style="color:#666"> ({rate}%)</span></div>
</div>"""

ENTRY_ROW = """<div style="background:#1a1f26;border-radius:10px;padding:12px;margin:8px 0;border:1px solid #2a3038;display:flex;justify-content:space-between">
  <div>
    <div style="font-size:13px;font-weight:bold">{away_team} @ {home_team}</div>
    <div style="color:#888;font-size:11px">{label} · {prob}% · logged inning {inning} ({state})</div>
  </div>
  <div style="text-align:right"><span style="color:{color};font-weight:bold;font-size:13px">{status_label}</span></div>
</div>"""

STATUS_COLORS = {'hit': '#a0e8a0', 'miss': '#ff6b6b', 'void': '#888', 'pending': '#ffeb3b'}


def build_results_dashboard(log):
    entries = list(log.values())
    graded = [e for e in entries if e['status'] in ('hit', 'miss')]

    def _category(label):
        if 'Next Run' in label:
            return 'Next Run'
        if 'Inning Winner' in label:
            return 'Inning Winner'
        if 'Inning Most Hits' in label:
            return 'Inning Most Hits'
        return 'Game Total Pace'

    by_category = {}
    for e in graded:
        by_category.setdefault(_category(e['label']), []).append(e)

    summary_rows = ""
    for cat, cat_entries in sorted(by_category.items()):
        hits = sum(1 for e in cat_entries if e['status'] == 'hit')
        total = len(cat_entries)
        rate = round(100 * hits / total) if total else 0
        summary_rows += CATEGORY_SUMMARY_ROW.format(label=cat, hits=hits, total=total, rate=rate)

    overall_hits = sum(1 for e in graded if e['status'] == 'hit')
    overall_total = len(graded)
    overall_rate = round(100 * overall_hits / overall_total) if overall_total else 0
    summary = f"""<div style="background:#1a1f26;border-radius:12px;padding:16px;margin:12px 0;border:1px solid #2a3038">
      <div style="font-size:15px;font-weight:bold;margin-bottom:6px">Overall: {overall_hits}/{overall_total} ({overall_rate}%)</div>
      {summary_rows or '<div style="color:#888;font-size:12px">No graded signals yet.</div>'}
    </div>"""

    entries_sorted = sorted(entries, key=lambda e: e.get('logged_at', ''), reverse=True)[:50]
    rows = "".join(ENTRY_ROW.format(
        away_team=e['away_team'], home_team=e['home_team'],
        label=e['label'], prob=e['prob'],
        inning=e.get('logged_inning', '?'), state=e.get('logged_inning_state', '?'),
        color=STATUS_COLORS.get(e['status'], '#888'),
        status_label=e['status'].upper(),
    ) for e in entries_sorted)
    if not rows:
        rows = '<p style="text-align:center;color:#888">No signals logged yet.</p>'

    os.makedirs(os.path.dirname(DASHBOARD_PATH), exist_ok=True)
    with open(DASHBOARD_PATH, 'w') as f:
        f.write(DASHBOARD_TEMPLATE.format(
            generated=datetime.now().strftime('%d %b %H:%M'),
            summary=summary, rows=rows,
        ))


def run_results_tracker(live_games):
    log = load_log()
    log_todays_signals(live_games, log)
    verify_pending_results(log)
    save_log(log)
    build_results_dashboard(log)
