#!/usr/bin/env python3
"""
Blitz IQ Live — Results Tracker

Same pattern as the suite's other results trackers (log every qualifying
pick as "pending" before the outcome is known, verify later against the
real result, build a win-rate dashboard) but adapted for IN-GAME signals
rather than pregame ones:

  - Next Score signals ("Home Next Score" / "Away Next Score") are logged
    the moment they first qualify (>=70%) for a game, with the game clock
    at that instant. Verified once the game is final, by finding the next
    actual scoring play (via ESPN's summary endpoint) that happened AFTER
    that logged instant and checking which team scored it. If neither team
    scores again after the signal fired, there's nothing to grade — marked
    "void", not counted as a win or loss.

  - Game Total Pace signals ("Game Total Over" / "Game Total Under") are
    logged the same way, and verified against the model's own PREGAME
    total projection vs. the actual final combined score (a tie is void).

Each (game, label) pair is logged only ONCE per game — the first time it
qualifies — even though the live script re-evaluates every 10 minutes and
the signal can flip direction as the game progresses. That matches how
someone would actually act on it: the first time the model flags
something, not every re-confirmation of it.

Wrapped in try/except by blitz_iq_live.py's __main__, same as every other
tracker in the suite — a tracker failure never breaks the main live run.
"""

import os
import json
import time
from datetime import datetime, timezone

import requests

LOG_PATH = "docs/blitz-iq-live/results_log.json"
DASHBOARD_PATH = "docs/blitz-iq-live/results/index.html"
SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary"
MAX_CHECKS_PER_RUN = 40  # generous cap; ESPN's API has no published quota to protect
GAME_MINUTES = 60


def _get(url, params=None, timeout=15):
    try:
        r = requests.get(url, params=params or {}, timeout=timeout)
        time.sleep(1.0)
        if r.status_code != 200:
            print(f"    [!] {r.status_code} on {url}")
            return None
        return r.json()
    except Exception as e:
        time.sleep(1.0)
        print(f"    [!] request failed: {url} ({e})")
        return None


def elapsed_minutes(period, clock_display):
    """Same formula as blitz_iq_live.py's elapsed_minutes(), duplicated
    here (rather than imported) so this tracker stays a standalone module
    like the rest of the suite's trackers -- takes period/clock directly
    instead of a status dict."""
    if period is None:
        return None
    clock_display = clock_display or '15:00'
    try:
        mins, secs = clock_display.split(':')
        remaining_in_period = int(mins) + int(secs) / 60.0
    except (ValueError, AttributeError):
        remaining_in_period = 15.0

    if period <= 4:
        return (period - 1) * 15 + (15 - remaining_in_period)
    ot_period_num = period - 4
    return 60 + (ot_period_num - 1) * 10 + (10 - remaining_in_period)


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


def _entry_id(game_id, label):
    return f"{game_id}_{label.replace(' ', '_')}"


def log_todays_signals(live_games, log):
    """Log each currently-qualifying model_signal once per (game, label) --
    if it's already in the log (from an earlier run this same game), it's
    left alone rather than re-logged or overwritten."""
    added = 0
    for g in live_games:
        signal = g.get('model_signal') or {}
        if not signal.get('qualifies') or not signal.get('label'):
            continue

        eid = _entry_id(g['game_id'], signal['label'])
        if eid in log:
            continue

        log[eid] = {
            'id': eid,
            'game_id': g['game_id'],
            'home_id': g.get('home_id'),
            'away_id': g.get('away_id'),
            'home_team': g['home_team'],
            'away_team': g['away_team'],
            'label': signal['label'],
            'prob': signal['prob'],
            'logged_at': datetime.now(timezone.utc).isoformat(),
            'logged_period': g.get('period'),
            'logged_clock': g.get('display_clock'),
            'logged_home_score': g.get('home_score'),
            'logged_away_score': g.get('away_score'),
            'pregame_total': (g.get('pregame_projection') or {}).get('exp_total'),
            'status': 'pending',
        }
        added += 1
    if added:
        print(f"  Logged {added} new signal(s)")


def _is_final(summary):
    try:
        return bool(summary['header']['competitions'][0]['status']['type']['completed'])
    except (KeyError, IndexError, TypeError):
        return False


def _final_score(summary):
    try:
        competitors = summary['header']['competitions'][0]['competitors']
        home = next(c for c in competitors if c.get('homeAway') == 'home')
        away = next(c for c in competitors if c.get('homeAway') == 'away')
        return float(home['score']), float(away['score'])
    except (KeyError, StopIteration, TypeError, ValueError):
        return None, None


def _scoring_plays(summary):
    """Returns a list of (elapsed_minutes, team_id) for every scoring play
    in the game, in whatever order ESPN provides -- sorted here by elapsed
    time to guarantee chronological order regardless of feed order."""
    plays = []
    for p in summary.get('scoringPlays', []) or []:
        period = (p.get('period') or {}).get('number')
        clock = (p.get('clock') or {}).get('displayValue')
        team_id = (p.get('team') or {}).get('id')
        elapsed = elapsed_minutes(period, clock)
        if elapsed is not None and team_id is not None:
            plays.append((elapsed, str(team_id)))
    plays.sort(key=lambda x: x[0])
    return plays


def _verify_next_score_entry(entry, summary):
    logged_elapsed = elapsed_minutes(entry.get('logged_period'), entry.get('logged_clock'))
    if logged_elapsed is None:
        return 'void', None

    predicted_team_id = str(entry['home_id']) if 'Home' in entry['label'] else str(entry['away_id'])

    for elapsed, team_id in _scoring_plays(summary):
        if elapsed > logged_elapsed + 0.01:  # small epsilon past the logged instant
            return ('hit' if team_id == predicted_team_id else 'miss'), team_id

    return 'void', None  # neither team scored again after the signal fired


def _verify_game_total_entry(entry):
    final_home, final_away = entry.get('_final_home'), entry.get('_final_away')
    pregame_total = entry.get('pregame_total')
    if final_home is None or final_away is None or pregame_total is None:
        return 'void'

    final_total = final_home + final_away
    if final_total == pregame_total:
        return 'void'  # push -- neither Over nor Under

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

        summary = _get(SUMMARY_URL, params={'event': entry['game_id']})
        if not summary or not _is_final(summary):
            continue  # game still in progress (or fetch failed) -- try again next run

        final_home, final_away = _final_score(summary)
        entry['_final_home'], entry['_final_away'] = final_home, final_away

        if 'Next Score' in entry['label']:
            result, scorer_team_id = _verify_next_score_entry(entry, summary)
            entry['scorer_team_id'] = scorer_team_id
        else:
            result = _verify_game_total_entry(entry)

        entry['status'] = result
        entry['final_home_score'] = final_home
        entry['final_away_score'] = final_away
        entry.pop('_final_home', None)
        entry.pop('_final_away', None)
        graded += 1

    if checked:
        print(f"  Checked {checked} pending signal(s), graded {graded}")


DASHBOARD_TEMPLATE = """<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Blitz IQ Live — Results</title></head>
<body style="background:#0b0f14;color:white;font-family:Arial;padding:12px;max-width:700px;margin:auto">
<h2 style="text-align:center">📊 Blitz IQ Live — Results Tracker</h2>
<p style="text-align:center;color:#888;font-size:11px">Every qualifying live signal (≥70%), logged the instant it fired and graded against what actually happened · {generated}</p>
<p style="text-align:center;margin-bottom:16px"><a href="../index.html" style="color:#7ec8ff;text-decoration:none;font-size:12px">⚡ Back to Live Signals</a></p>
{summary}
{rows}
<p style="text-align:center;color:#666;font-size:10px;margin-top:20px">
Next Score signals grade against the actual next scoring play after the signal fired; if neither team scores again, it's marked Void (not a loss). Game Total signals grade against the model's own pregame total projection; an exact push is Void.
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
    <div style="color:#888;font-size:11px">{label} · {prob}% · logged Q{period} {clock}</div>
  </div>
  <div style="text-align:right"><span style="color:{color};font-weight:bold;font-size:13px">{status_label}</span></div>
</div>"""

STATUS_COLORS = {'hit': '#a0e8a0', 'miss': '#ff6b6b', 'void': '#888', 'pending': '#ffeb3b'}


def build_results_dashboard(log):
    entries = list(log.values())
    graded = [e for e in entries if e['status'] in ('hit', 'miss')]

    by_category = {}
    for e in graded:
        cat = 'Next Score' if 'Next Score' in e['label'] else 'Game Total Pace'
        by_category.setdefault(cat, []).append(e)

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
        period=e.get('logged_period', '?'), clock=e.get('logged_clock', '?'),
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
