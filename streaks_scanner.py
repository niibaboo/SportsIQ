#!/usr/bin/env python3
"""
Streaks Scanner — cross-suite Real Streak aggregator

Every sibling predictor in this repo that has a Hot Form / Real Streak
screen already logs its Real Streak picks (pending AND settled) into its
own results tracker's log.json. This script doesn't re-run any model or
call any sport API itself — it just reads the log.json each tool's own
results tracker has already written, pulls out the Real Streak entries
specifically (Hot Form is intentionally left out: it's an average that
can mask a bad most recent game, whereas Real Streak is a genuine
CONSECUTIVE run, which is the stronger/rarer signal worth surfacing on
its own page), and builds one combined dashboard: today's live Real
Streak picks across every sport in one place, plus each tool's Real
Streak track record side by side.

Run this AFTER (or independently of, on its own schedule) the other
tools' own workflows -- it only reads what's already on disk, so it's
safe to run on a tighter schedule than any individual sport's own
scanner; it'll just reflect whatever each log currently has.

Setup:
    python3 streaks_scanner.py

Output:
    docs/streaks/index.html
"""

import os
import json
from datetime import datetime, timezone

OUTPUT_PATH = "docs/streaks/index.html"

# One entry per sibling tool that has a Real Streak screen. log_path is
# wherever that tool's OWN results tracker already writes its log.json
# (see each file's LOG_PATH constant) -- this script never writes to
# those files, only reads them.
SOURCES = [
    {"label": "Blue Line", "sport": "NHL", "log_path": "docs/blue-line/results/log.json",
     "page": "../blue-line/index.html"},
    {"label": "Blitz IQ", "sport": "NFL", "log_path": "docs/blitz-iq/results/log.json",
     "page": "../blitz-iq/index.html"},
    {"label": "Match IQ", "sport": "Soccer", "log_path": "docs/match-iq/results/log.json",
     "page": "../match_iq_index.html"},
    {"label": "Player Stat Model", "sport": "Soccer Props", "log_path": "docs/player-stat-model/results/log.json",
     "page": "../player-stat-model/index.html"},
    {"label": "Orange Line", "sport": "NBA", "log_path": "docs/orange-line/results/log.json",
     "page": "../orange-line/index.html"},
    {"label": "Euro Ice", "sport": "SHL/Czech", "log_path": "docs/euro-ice/results/log.json",
     "page": "../euro-ice/index.html"},
    {"label": "EuroLeague IQ", "sport": "EuroLeague", "log_path": "docs/results/log.json",
     "page": "../euroleague-iq/index.html"},
    {"label": "Strike Zone", "sport": "MLB", "log_path": "docs/strike-zone/results/log.json",
     "page": "../strike-zone/index.html"},
]


def _is_real_streak_scanner(scanner):
    """Every tool names its Real Streak scanners a little differently
    (real_streak, sog_real_streak, team_real_streak, player_real_streak,
    real_run_streak, real_hit_streak, real_k_streak, ...) but they all
    contain both "real" and "streak" -- Hot Form scanners never do
    (hot_form, sog_hot_form, player_hot_form, ...), so this one check
    catches every tool's Real Streak scanner without hardcoding each
    tool's exact naming scheme, and keeps working if a tool adds a new
    Real Streak stat later without this script needing an update."""
    s = (scanner or "").lower()
    return "real" in s and "streak" in s


def _load_log(path):
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def _subject(e):
    """Most trackers store the streaking team/player under "subject",
    "player" or "team". A couple (EuroLeague IQ, Orange Line) only store
    a combined "{team} vs {opponent}" string under "match" -- split that
    on " vs " and take the team side rather than falling through to
    "Unknown" (user report, 2026-10-20: EuroLeague IQ rows were showing
    "Unknown" as the subject for exactly this reason)."""
    subject = e.get("subject") or e.get("player") or e.get("team")
    if subject:
        return subject
    match = e.get("match")
    if match and " vs " in match:
        return match.split(" vs ", 1)[0]
    return match or "Unknown"


def _detail(e):
    """Surfaces each entry's "detail" field when it has one -- several
    trackers (EuroLeague IQ, Player Stat Model) pack real context in
    here (the actual threshold cleared, team/league), which the plain
    subject/market/value row was dropping entirely (user report,
    2026-10-20: "not much information on the streak"). The couple of
    trackers that only ever stash a raw "is_home=True/False" flag in
    this field get it turned into a plain "Home"/"Away" label instead
    of showing the raw Python repr."""
    detail = e.get("detail")
    if not detail:
        return None
    if detail == "is_home=True":
        return "Home"
    if detail == "is_home=False":
        return "Away"
    return detail


def _streak_games_str(e):
    """Normalizes each tool's per-game streak-value history into a single
    "oldest→newest" slash-joined string, matching the convention each
    tool's own page already uses (user request, 2026-10-20: "can it add
    the exact last stats from the original page?"). Every tracker but
    Player Stat Model stores this as a raw list under "streak_games";
    Player Stat Model pre-joins it into a string already. Missing/None
    (older log entries logged before this field existed) returns None so
    the caller can skip the line entirely rather than show "None"."""
    sg = e.get("streak_games")
    if not sg:
        return None
    if isinstance(sg, str):
        return sg
    try:
        return "/".join(str(int(v)) if float(v) == int(v) else str(v) for v in sg)
    except (TypeError, ValueError):
        return "/".join(str(v) for v in sg)


def _date_key(e):
    return (e.get("date_key") or e.get("match_date") or e.get("game_date")
            or e.get("date") or "")[:10]


def collect():
    """Returns (live, by_source) where live is every pending Real Streak
    entry across all tools (newest first) and by_source is a dict of
    label -> {"hit": n, "miss": n, "pending": n}."""
    live = []
    by_source = {}
    for src in SOURCES:
        log = _load_log(src["log_path"])
        stats = {"hit": 0, "miss": 0, "pending": 0}
        for e in log:
            if not _is_real_streak_scanner(e.get("scanner")):
                continue
            if e.get("status") == "pending":
                stats["pending"] += 1
                live.append({
                    "source": src["label"], "sport": src["sport"],
                    "subject": _subject(e), "market": e.get("market", ""),
                    "value": e.get("value"), "detail": _detail(e),
                    "threshold": e.get("threshold"), "date": _date_key(e),
                    "streak_games": _streak_games_str(e),
                })
            elif e.get("result") == "hit":
                stats["hit"] += 1
            elif e.get("result") == "miss":
                stats["miss"] += 1
        by_source[src["label"]] = stats

    live.sort(key=lambda e: e["date"], reverse=True)
    return live, by_source


def build_html(live, by_source):
    total_hit = sum(s["hit"] for s in by_source.values())
    total_miss = sum(s["miss"] for s in by_source.values())
    total_settled = total_hit + total_miss
    overall_pct = round(100 * total_hit / total_settled) if total_settled else None

    source_rows = ""
    for src in SOURCES:
        stats = by_source.get(src["label"], {"hit": 0, "miss": 0, "pending": 0})
        settled = stats["hit"] + stats["miss"]
        pct = round(100 * stats["hit"] / settled) if settled else None
        pct_str = f"{pct}%" if pct is not None else "—"
        source_rows += f"""<a href="{src['page']}" style="text-decoration:none;color:inherit">
<div style="display:flex;justify-content:space-between;align-items:center;padding:10px 0;border-bottom:1px solid #233040">
  <div>
    <div style="font-weight:600">{src['label']}</div>
    <div style="color:#8b98a8;font-size:11px">{src['sport']}</div>
  </div>
  <div style="text-align:right">
    <div style="color:#a0e8a0;font-weight:bold">{pct_str}</div>
    <div style="color:#8b98a8;font-size:11px">{stats['hit']}W / {stats['miss']}L · {stats['pending']} live</div>
  </div>
</div></a>"""

    # Tool filter + date filter on "Live Real Streaks" (user request,
    # 2026-10-20) -- pending entries can sit in a tool's log for weeks
    # if a game never got verified (API hiccup, postponed fixture, etc),
    # so without a filter this list mixes genuinely live picks with
    # month-old stragglers. Both filters are client-side, same pattern
    # as the date filter already on the Over IQ dashboard, and combine
    # with AND logic (a row shows only when it passes both).
    live_rows = ""
    for e in live:
        value_str = f" · streak {e['value']}" if e.get("value") is not None else ""
        # Threshold shown only when the market text doesn't already spell
        # it out itself (Orange Line/EuroLeague IQ already embed it, e.g.
        # "...≥2 REB" or "...above league avg (18+)" -- repeating it would
        # just be noise).
        threshold = e.get("threshold")
        already_shown = str(threshold) in e["market"] or str(threshold) in (e.get("detail") or "")
        threshold_str = f" (needs {threshold}+)" if threshold is not None and not already_shown else ""
        detail_line = (f'<div style="font-size:11px;color:#8b98a8;margin-top:2px">{e["detail"]}</div>'
                        if e.get("detail") else "")
        streak_line = (f'<div style="font-size:11px;color:#f59e0b;margin-top:2px">Last games (old→new): {e["streak_games"]}</div>'
                        if e.get("streak_games") else "")
        live_rows += f"""<div class="live-row" data-source="{e['source']}" data-date="{e['date']}" style="padding:8px 0;border-bottom:1px solid #1a2029">
  <div style="display:flex;justify-content:space-between;font-size:12px;color:#8b98a8;margin-bottom:2px">
    <span>{e['source']} · {e['sport']}</span>
    <span>{e['date']}</span>
  </div>
  <div style="font-size:13px">{e['subject']} — {e['market']}{value_str}{threshold_str}</div>
  {detail_line}
  {streak_line}
</div>"""
    if not live_rows:
        live_rows = '<p style="color:#888;font-size:12px">No live Real Streak picks right now across any tool.</p>'

    tools_with_live = sorted({e["source"] for e in live})
    tool_filter_bar = ""
    if tools_with_live:
        tool_buttons = ''.join(
            f'<button class="tool-filter-btn" data-tool="{t}" onclick="filterLive(\'tool\',\'{t}\',this)" '
            f'style="background:#1a2433;border:1px solid #233040;color:#7ec8ff;padding:5px 10px;border-radius:999px;font-size:11px;cursor:pointer">{t}</button>'
            for t in tools_with_live
        )
        tool_filter_bar = f"""<div style="display:flex;gap:6px;margin-bottom:8px;flex-wrap:wrap">
          <button class="tool-filter-btn active-filter" data-tool="all" onclick="filterLive('tool','all',this)" style="background:#7ec8ff;border:1px solid #7ec8ff;color:#0b0f14;padding:5px 10px;border-radius:999px;font-size:11px;font-weight:700;cursor:pointer">All tools</button>
          {tool_buttons}
        </div>"""

    date_filter_bar = """<div style="display:flex;gap:6px;margin-bottom:10px;flex-wrap:wrap">
          <button class="date-filter-btn" data-range="7" onclick="filterLive('date',7,this)" style="background:#1a2433;border:1px solid #233040;color:#f59e0b;padding:6px 12px;border-radius:999px;font-size:12px;cursor:pointer">Last 7 Days</button>
          <button class="date-filter-btn" data-range="30" onclick="filterLive('date',30,this)" style="background:#1a2433;border:1px solid #233040;color:#f59e0b;padding:6px 12px;border-radius:999px;font-size:12px;cursor:pointer">Last 30 Days</button>
          <button class="date-filter-btn active-filter" data-range="all" onclick="filterLive('date','all',this)" style="background:#f59e0b;border:1px solid #f59e0b;color:#0b0f14;padding:6px 12px;border-radius:999px;font-size:12px;font-weight:700;cursor:pointer">All</button>
        </div>
        <script>
        var _liveFilters = {tool: 'all', date: 'all'};
        function filterLive(kind, val, btn) {
          _liveFilters[kind] = val;
          var rows = document.querySelectorAll('.live-row');
          var cutoff = null;
          if (_liveFilters.date !== 'all') {
            cutoff = new Date();
            cutoff.setHours(0,0,0,0);
            cutoff.setDate(cutoff.getDate() - _liveFilters.date);
          }
          rows.forEach(function(row) {
            var toolOk = (_liveFilters.tool === 'all') || (row.getAttribute('data-source') === _liveFilters.tool);
            var dateOk = true;
            if (cutoff !== null) {
              var d = new Date(row.getAttribute('data-date') + 'T00:00:00');
              dateOk = (d >= cutoff);
            }
            row.style.display = (toolOk && dateOk) ? 'block' : 'none';
          });
          var btnClass = (kind === 'tool') ? 'tool-filter-btn' : 'date-filter-btn';
          var activeColor = (kind === 'tool') ? '#7ec8ff' : '#f59e0b';
          document.querySelectorAll('.' + btnClass).forEach(function(b) {
            b.style.background = '#1a2433'; b.style.color = activeColor; b.style.fontWeight = 'normal';
          });
          btn.style.background = activeColor; btn.style.color = '#0b0f14'; btn.style.fontWeight = '700';
        }
        </script>"""

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Real Streaks — All Models</title></head>
<body style="background:#0b0f14;color:white;font-family:Arial;padding:12px;max-width:600px;margin:auto">
<p style="text-align:center;margin-bottom:6px"><a href="../index.html" style="color:#7ec8ff;text-decoration:none;font-size:12px">← Predictor Suite</a></p>
<h2 style="text-align:center;margin-bottom:2px">🔥 Real Streaks — All Models</h2>
<p style="text-align:center;color:#888;font-size:11px;margin-top:0">{datetime.now(timezone.utc).strftime("%d %b %H:%M UTC")} · every Real Streak pick, every tool, one page · genuine CONSECUTIVE runs only, not Hot Form averages</p>

<div style="background:#1a1f26;border-radius:12px;padding:16px;margin:14px 0;border:1px solid #2a3038;text-align:center">
  <div style="font-size:11px;color:#888">OVERALL REAL STREAK RECORD</div>
  <div style="font-size:32px;font-weight:bold;color:#a0e8a0">{overall_pct if overall_pct is not None else "—"}{"%" if overall_pct is not None else ""}</div>
  <div style="font-size:12px;color:#8b98a8">{total_hit}/{total_settled} verified · {len(live)} live pick(s) right now</div>
</div>

<div style="background:#1a1f26;border-radius:12px;padding:16px;margin:14px 0;border:1px solid #2a3038">
  <div style="font-weight:bold;margin-bottom:8px">By tool</div>
  {source_rows}
</div>

<div style="background:#1a1f26;border-radius:12px;padding:16px;margin:14px 0;border:1px solid #2a3038">
  <div style="font-weight:bold;margin-bottom:8px">Live Real Streaks right now</div>
  {tool_filter_bar}
  {date_filter_bar}
  {live_rows}
</div>

<p style="text-align:center;color:#666;font-size:10px;margin-top:20px">This page only reads each tool's own results log -- it doesn't fetch anything itself, so it's exactly as fresh as the last time each individual tool ran. A tool that hasn't run yet today just won't have today's picks here until it does.</p>
</body></html>"""


def main():
    live, by_source = collect()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        f.write(build_html(live, by_source))
    total_live = len(live)
    print(f"Streaks Scanner: {total_live} live Real Streak pick(s) across {len(SOURCES)} tools -- written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
