#!/usr/bin/env python3
"""
Market IQ Results Tracker
--------------------------------------------------------------
Same pattern as every other tracker in this suite: log the picks BEFORE
kickoff, on later runs fetch the REAL result for anything that has since
finished, and grade it -- an honest track record, not a screenshot.

WHY THIS EXISTS (user request, 2026-10-30, after asking for an auto-bet
model): Market IQ's de-vigged Bet365 price carries no edge on its own.
The only candidate signal is the gap between Bet365's price and the wider
market average (a positive gap = the market rates a side MORE likely than
Bet365 prices it, i.e. Bet365 is paying a longer price than the field
thinks fair). Before anything gets built on that gap -- alerts, staking,
anything -- this tracker answers whether it actually pays.

WHAT GETS LOGGED: one entry per match that Bet365 has priced, holding
Bet365's decimal odds for Home/Draw/Away, Bet365's own de-vigged
probabilities, the market-average probabilities (when other bookmakers
were also on the match), and which sides were FLAGGED (gap >= the same
MKT_GAP_HIGHLIGHT_PP threshold the page highlights green). While a match
is still in the future its snapshot is refreshed on each run (latest
pre-kickoff price); once kickoff passes it is frozen.

HOW IT'S SCORED: every flagged side is a flat 1-unit bet at Bet365's
decimal odds (win = +odds-1, loss = -1). The key honesty check is
CALIBRATION: if the flags are a real signal, flagged sides should win
about as often as the MARKET probability says (higher than Bet365's own
implied probability). If the actual hit rate sits at or below Bet365's
implied rate, the gap is noise. Expect noise for a long time -- a few
dozen bets say almost nothing about ROI. The dashboard shows the sample
size so nobody reads a 10-bet run as proof.

HOW RESULTS ARE VERIFIED: TheStatsAPI /football/matches/{id} -- graded
only once status == "finished" AND score.home/score.away are populated
(same score shape match_iq.py reads). A finished match whose score never
populates stays pending (with a [DIAG] line) rather than being guessed.
Matches still unfinished 14 days after kickoff are marked "void"
(postponed/abandoned) and excluded from every stat.

Usage (called automatically from market_iq.py's main block):
    import market_iq_results_tracker
    market_iq_results_tracker.run_results_tracker(cards, api_key)

Output:
    docs/market-iq/results/log.json
    docs/market-iq/results/index.html
"""

import os
import json
import html as html_lib
import requests
from datetime import datetime, timezone, timedelta

BASE = "https://api.thestatsapi.com/api"
LOG_PATH = "docs/market-iq/results/log.json"
DASHBOARD_PATH = "docs/market-iq/results/index.html"

# Caps fresh API calls per run so a backlog can't blow the metered tier.
MAX_CHECKS_PER_RUN = 80

# A match still unfinished this many days after kickoff is treated as
# postponed/abandoned and voided.
VOID_AFTER_DAYS = 14

SIDES = ("home", "draw", "away")


def _headers(key):
    return {"Authorization": f"Bearer {key}"}


def _get(path, key, params=None, timeout=15):
    try:
        r = requests.get(f"{BASE}{path}", headers=_headers(key), params=params or {}, timeout=timeout)
    except Exception as e:
        print(f"    [!] verification request failed: {path} ({e})")
        return None
    if r.status_code != 200:
        print(f"    [!] {r.status_code} on {path}: {r.text[:150]}")
        return None
    return r.json()


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


def _kickoff_passed(kickoff, now):
    """True once the match's kickoff is in the past. Unparseable kickoff
    counts as passed so a snapshot is never refreshed on a guess."""
    try:
        dt = datetime.fromisoformat(str(kickoff).replace("Z", "+00:00"))
    except ValueError:
        return True
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt <= now


def log_todays_signals(cards, log, now=None):
    """One entry per Bet365-priced match, keyed by match_id. Refreshes the
    snapshot while the match is still upcoming; frozen after kickoff.
    Cards without a match_id or Bet365 odds are skipped (nothing to
    grade or no price to bet at)."""
    now = now or datetime.now(timezone.utc)
    by_id = {e["match_id"]: e for e in log}
    added = refreshed = 0

    for c in cards:
        mid = c.get("match_id")
        odds = c.get("bet365_odds")
        if not mid or not odds or c.get("source") != "preferred":
            continue

        has_market = "market_home_prob" in c
        snapshot = {
            "match_id": mid, "league": c.get("league"),
            "home_team": c.get("home"), "away_team": c.get("away"),
            "kickoff": c.get("kickoff"), "date_key": str(c.get("kickoff", ""))[:10],
            "odds": {s: odds[s] for s in SIDES},
            "bet365_prob": {s: c[f"{s}_prob"] for s in SIDES},
            "market_prob": ({s: c[f"market_{s}_prob"] for s in SIDES} if has_market else None),
            "market_bookmaker_count": c.get("market_bookmaker_count"),
            "flagged": list(c.get("flagged", [])) if has_market else [],
        }

        existing = by_id.get(mid)
        if existing is None:
            snapshot.update({"status": "pending", "result": None, "score": None,
                             "logged_at": now.isoformat()})
            log.append(snapshot)
            by_id[mid] = snapshot
            added += 1
        elif existing["status"] == "pending" and not _kickoff_passed(existing.get("kickoff"), now):
            existing.update(snapshot)
            existing["updated_at"] = now.isoformat()
            refreshed += 1

    print(f"  Results log: {added} new match(es) logged, {refreshed} refreshed, {len(log)} total in log")
    return log


def _verify_entry(entry, key):
    """Returns {"result": "home"/"draw"/"away", "score": "h-a"} once the
    match is finished with a real score, else None (stays pending)."""
    data = _get(f"/football/matches/{entry['match_id']}", key)
    m = data.get("data") if data else None
    if not m or m.get("status") != "finished":
        return None
    score = m.get("score") or {}
    h, a = score.get("home"), score.get("away")
    if h is None or a is None:
        print(f"    [DIAG] match {entry['match_id']} is finished but has no score.home/away "
              f"(score field was: {score!r}) -- leaving pending")
        return None
    result = "home" if h > a else "away" if a > h else "draw"
    return {"result": result, "score": f"{h}-{a}"}


def verify_pending_results(log, key, now=None, max_checks=MAX_CHECKS_PER_RUN):
    now = now or datetime.now(timezone.utc)
    today = now.date().isoformat()
    void_before = (now - timedelta(days=VOID_AFTER_DAYS)).date().isoformat()
    checked = updated = voided = 0

    for entry in log:
        if entry["status"] != "pending":
            continue
        if entry["date_key"] >= today:
            continue  # match hasn't been played yet
        if checked >= max_checks:
            break
        checked += 1

        try:
            res = _verify_entry(entry, key)
        except Exception as e:
            print(f"    [!] verification error for match {entry['match_id']}: {e}")
            res = None

        if res:
            entry.update(status="verified", result=res["result"], score=res["score"],
                         verified_at=now.isoformat())
            updated += 1
        elif entry["date_key"] < void_before:
            entry["status"] = "void"
            voided += 1

    print(f"  Results verification: checked {checked} pending, {updated} newly verified, {voided} voided")
    return log


def summarize(log):
    """Pure stats, kept separate from HTML so it can be tested directly."""
    verified = [e for e in log if e["status"] == "verified"]
    bets = []
    for e in verified:
        if not e.get("market_prob"):
            continue
        for side in e.get("flagged", []):
            won = e["result"] == side
            odds = e["odds"][side]
            bets.append({
                "entry": e, "side": side, "odds": odds, "won": won,
                "profit": (odds - 1.0) if won else -1.0,
                "bet365_prob": e["bet365_prob"][side], "market_prob": e["market_prob"][side],
            })
    n = len(bets)
    out = {"n": n, "pending": sum(1 for e in log if e["status"] == "pending"),
           "verified_matches": len(verified), "bets": bets}
    if n:
        wins = sum(1 for b in bets if b["won"])
        out.update(
            wins=wins, hit_rate=wins / n,
            exp_bet365=sum(b["bet365_prob"] for b in bets) / n,
            exp_market=sum(b["market_prob"] for b in bets) / n,
            avg_odds=sum(b["odds"] for b in bets) / n,
            profit=sum(b["profit"] for b in bets), roi=sum(b["profit"] for b in bets) / n,
        )
    return out


def _pct(x):
    return f"{x * 100:.1f}%"


def build_results_dashboard(log):
    s = summarize(log)
    esc = html_lib.escape

    if s["n"]:
        sign = "+" if s["profit"] >= 0 else ""
        roi_color = "#22c55e" if s["profit"] >= 0 else "#ef4444"
        verdict_gap = s["hit_rate"] - s["exp_bet365"]
        summary = f"""<div style="background:#1a1f26;border-radius:12px;padding:16px;margin:14px 0;border:1px solid #2a3038;text-align:center">
  <div style="font-size:11px;color:#888">FLAGGED SIDES · 1 UNIT FLAT AT BET365 ODDS</div>
  <div style="font-size:32px;font-weight:bold;color:{roi_color}">{sign}{s['roi'] * 100:.1f}% ROI</div>
  <div style="font-size:12px;color:#8b98a8">{s['wins']}/{s['n']} won · {sign}{s['profit']:.2f} units · avg odds {s['avg_odds']:.2f}</div>
</div>
<div style="background:#1a1f26;border-radius:12px;padding:16px;margin:14px 0;border:1px solid #2a3038">
  <div style="font-weight:bold;margin-bottom:8px">Is the gap a real signal?</div>
  <div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #233040"><span>Actual hit rate</span><b>{_pct(s['hit_rate'])}</b></div>
  <div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #233040"><span>Bet365 implied (de-vigged)</span><b>{_pct(s['exp_bet365'])}</b></div>
  <div style="display:flex;justify-content:space-between;padding:6px 0"><span>Market-average implied</span><b>{_pct(s['exp_market'])}</b></div>
  <div style="font-size:11px;color:#8b98a8;margin-top:8px;line-height:1.5">
    If the flags are real, the actual hit rate should sit near the market figure and above
    Bet365's. Right now it is {verdict_gap * 100:+.1f}pp vs Bet365's implied rate over {s['n']} bet(s).
    {'That is far too few bets to conclude anything -- treat it as noise until there are well over 100.' if s['n'] < 100 else 'Enough bets to start taking the pattern seriously, though ROI is still very noisy.'}
  </div>
</div>"""
    else:
        summary = f"""<div style="background:#1a1f26;border-radius:12px;padding:16px;margin:14px 0;border:1px solid #2a3038;text-align:center">
  <div style="font-size:32px;font-weight:bold;color:#8b98a8">—</div>
  <div style="font-size:12px;color:#8b98a8">No graded flagged bets yet · {s['pending']} match(es) pending</div>
</div>"""

    recent = sorted(s["bets"], key=lambda b: b["entry"].get("verified_at", ""), reverse=True)[:30]
    rows = ""
    for b in recent:
        e = b["entry"]
        color = "#22c55e" if b["won"] else "#ef4444"
        rows += f"""<div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #233040;font-size:12px">
  <div><b>{esc(str(e['home_team']))} vs {esc(str(e['away_team']))}</b><br><span style="color:#8b98a8">backed {b['side']} @ {b['odds']:.2f} · Bet365 {_pct(b['bet365_prob'])} vs mkt {_pct(b['market_prob'])} · final {esc(str(e['score']))}</span></div>
  <span style="color:{color};font-weight:bold">{'WON' if b['won'] else 'LOST'}</span>
</div>"""

    page = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Results — Market IQ</title></head>
<body style="background:#0b0f14;color:white;font-family:Arial;padding:12px;max-width:600px;margin:auto">
<h2 style="text-align:center;margin-bottom:2px">📋 Market IQ Results</h2>
<p style="text-align:center;color:#888;font-size:11px;margin-top:0">{datetime.now().strftime("%d %b %H:%M")} · {s['verified_matches']} match(es) graded · {s['pending']} pending</p>
{summary}
<div style="background:#1a1f26;border-radius:12px;padding:16px;margin:14px 0;border:1px solid #2a3038">
  <div style="font-weight:bold;margin-bottom:8px">Recent Flagged Bets</div>
  {rows or '<p style="color:#888;font-size:12px">Nothing graded yet — check back after flagged matches have been played.</p>'}
</div>
<div style="font-size:11px;color:#8b98a8;text-align:center;margin-top:20px;line-height:1.6">
  A side is flagged when the market average rates it 2pp+ more likely than Bet365 prices it.
  Every match is logged before kickoff and graded against the real final score; matches
  unfinished 14 days after kickoff are voided. Flat 1-unit staking, no cherry-picking.<br>
  <a href="../index.html" style="color:#7ec8ff">← Back to Market IQ</a>
</div>
</body></html>"""

    os.makedirs(os.path.dirname(DASHBOARD_PATH), exist_ok=True)
    with open(DASHBOARD_PATH, "w") as f:
        f.write(page)
    print(f"  Results dashboard: {s['n']} graded flagged bet(s)" if s["n"] else "  Results dashboard: no graded flagged bets yet")


def run_results_tracker(cards, key):
    """Single entry point called from market_iq.py's main block."""
    print("\nRunning results tracker...")
    log = load_log()
    log = log_todays_signals(cards, log)
    log = verify_pending_results(log, key)
    save_log(log)
    build_results_dashboard(log)
