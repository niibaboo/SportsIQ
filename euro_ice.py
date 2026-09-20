#!/usr/bin/env python3
"""
Euro Ice — Team & Game Totals (SHL / Czech Extraliga / DEL)
--------------------------------------------------------------
Team-level goals over/under predictor for SHL (Sweden), Czech Extraliga,
and DEL (Germany), built on Highlightly's Hockey API free tier:
  https://highlightly.net/hockey-api/documentation/

SCOPE NOTE — read before assuming this matches Blue Line's coverage:
Highlightly's API has NO player-level endpoints anywhere (confirmed by
reading every documented endpoint: Countries, Highlights, Leagues,
Matches, Teams, Bookmakers, Odds, Last Five Games, Head-2-Head,
Standings — all team/match-level only). This tool can only ever cover
TEAM TOTALS and GAME TOTALS, never player props (goalscorer, points,
shots on goal). If you need those for these leagues, this data source
can't provide them.

UNVERIFIED ASSUMPTIONS — this was built entirely from Highlightly's
documentation, not tested against a live API key. Everything below is
flagged inline where it matters, but the two biggest ones:
  1. state.score.current's format ("4 - 3") is parsed as HOME - AWAY.
     Documentation's own example is unhelpful (both teams named
     identically in the sample), so this order is a reasonable guess,
     not a confirmed fact. If projections come out looking backwards
     (a team's "goals for" looks suspiciously like their "goals
     against"), this is the first place to check.
  2. /teams/statistics/{id} requires a fromDate param whose exact
     semantics aren't fully documented (season-to-date? rolling
     window?). Defaulted to a guessed season-start date — verify
     against what a real response actually contains before trusting
     season_gpg numbers.
  3. RESOLVED (2026-09-18, via debug_leagues.py --confirm against live
     data): league IDs are now hardcoded in LEAGUE_TARGETS below rather
     than resolved by name at runtime. "Extraliga" turned out to be
     genuinely ambiguous — Highlightly has three leagues by that exact
     name (Belarus, Czech Republic, Slovakia); an earlier version of
     this script matched Belarus by mistake since name-only lookup has
     no way to disambiguate identically-named leagues in different
     countries. All four current leagues (SHL, Czech Extraliga, DEL,
     Switzerland National League) have confirmed IDs now — see
     LEAGUE_TARGETS.

Usage:
    pip3 install requests --break-system-packages
    export HIGHLIGHTLY_KEY=your_key_here
    python3 euro_ice.py               # today's fixtures across all 3 leagues
    python3 euro_ice.py 2026-10-05    # a specific date
    python3 euro_ice.py --auto        # non-interactive, for GitHub Actions —
                                       # same as running with no date arg,
                                       # scans today and always exits 0

Output:
    docs/euro-ice/index.html
    docs/euro-ice/euro_ice.json
"""

import os
import sys
import math
import json
import requests
from datetime import date, datetime

BASE = "https://hockey.highlightly.net"
# SECURITY: no hardcoded fallback key. The old default here was a real,
# committed API key sitting in a PUBLIC repo — rotate that key on
# Highlightly's dashboard if this file was ever pushed with one baked in.
API_KEY = os.environ.get("HIGHLIGHTLY_KEY")

# UNCONFIRMED exact names — see module docstring point 3. Watch the
# first run's "couldn't find league" warnings closely.
# CONFIRMED via debug_leagues.py --confirm against live Highlightly data
# (2026-09-18). "Extraliga" alone is genuinely ambiguous — Highlightly has
# THREE leagues by that exact name (Belarus id=1635, Czech Republic
# id=9294, Slovakia id=78225); the old name-only lookup silently picked
# whichever came first and matched Belarus. IDs are hardcoded here
# instead of resolved by name+country at runtime, which removes that
# ambiguity risk entirely rather than just filtering it correctly.
LEAGUE_TARGETS = [
    {"id": 40781, "name": "SHL", "country": "Sweden"},
    {"id": 9294, "name": "Extraliga", "country": "Czech Republic"},
    {"id": 16953, "name": "DEL", "country": "Germany"},
    {"id": 44185, "name": "National League", "country": "Switzerland"},
]

RECENT_WEIGHT = 0.65
DEFAULT_LINE_FACTOR = 0.72  # same safety-margin convention as every other tool
FINISHED_STATES = {"Finished", "Finished after penalties", "Finished after over time"}


def _get(path, params=None):
    if not API_KEY:
        print("Set the HIGHLIGHTLY_KEY environment variable first.")
        raise SystemExit(1)
    headers = {"x-rapidapi-key": API_KEY}
    r = requests.get(f"{BASE}{path}", headers=headers, params=params or {})
    r.raise_for_status()
    return r.json()


def poisson_pmf(k, lam):
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def prob_over(lam, line):
    threshold = math.floor(line) + 1
    cum = sum(poisson_pmf(i, lam) for i in range(threshold))
    return 1 - cum


def hit_rate(values, line):
    """(hits, total) from raw per-game values — same model-free
    empirical cross-check used everywhere else in this suite."""
    if not values:
        return None
    hits = sum(1 for v in values if v > line)
    return {"hits": hits, "total": len(values)}


def safe_line(lam, factor=DEFAULT_LINE_FACTOR, round_to=0.5):
    if lam is None:
        return None
    raw = lam * factor
    line = math.floor(raw / round_to) * round_to
    return max(line, round_to)


def win_probs_and_scores(lh, la):
    """Regulation-time home/away/tie win probabilities plus the top-2
    most likely correct scores, from a Poisson grid over each team's
    projected goals — same approach as Blue Line's per-game card, minus
    OT/SO (no way to model that from goals-only data, so these are NOT
    true moneyline probabilities, same caveat as the sibling tools)."""
    ph = pa = pt = 0.0
    for i in range(10):
        for j in range(10):
            p = poisson_pmf(i, lh) * poisson_pmf(j, la)
            if i > j:
                ph += p
            elif j > i:
                pa += p
            else:
                pt += p
    scores = []
    for i in range(7):
        for j in range(7):
            scores.append(((j, i), poisson_pmf(j, la) * poisson_pmf(i, lh)))
    scores.sort(key=lambda x: x[1], reverse=True)
    return ph, pa, pt, scores[:2]


def find_league(name):
    """name -> league dict, preferring an exact case-insensitive name
    match over the first search result. NOT used by build_legs_and_cards()
    anymore — see LEAGUE_TARGETS above: "Extraliga" turned out to match
    THREE different countries by exact name, so even the "prefer exact
    match" protection here isn't enough on its own when multiple leagues
    share the identical name. Kept for reference and for debug tooling
    (this is the same lookup debug_leagues.py's --confirm mode uses)."""
    data = _get("/leagues", {"leagueName": name})
    results = data.get("data", [])
    if not results:
        return None
    for lg in results:
        if lg.get("name", "").lower() == name.lower():
            return lg
    return results[0]


def parse_score(score_str):
    """'4 - 3' -> (4, 3), assumed HOME - AWAY. Returns None on anything
    unparseable rather than guessing — see module docstring point 1
    for why this order is not fully confirmed."""
    if not score_str:
        return None
    parts = score_str.replace(" ", "").split("-")
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def get_last_five(team_id):
    """Raw last-5 FINISHED games for a team, oldest first. Highlightly's
    own docs say unfinished games are never included in this endpoint's
    response, so no extra state filtering is needed here — unlike the
    general /matches endpoint, which mixes finished and upcoming games
    together and needs explicit filtering (see get_upcoming_matches)."""
    games = _get("/last-five-games", {"teamId": team_id})
    parsed = []
    for g in games:
        score = parse_score(g.get("state", {}).get("score", {}).get("current"))
        if not score:
            continue
        home_goals, away_goals = score
        is_home = g.get("homeTeam", {}).get("id") == team_id
        gf = home_goals if is_home else away_goals
        parsed.append({"date": g.get("date") or "", "gf": gf})
    parsed.sort(key=lambda x: x["date"])
    return parsed


def get_team_season_stats(team_id, from_date):
    """Season aggregate via /teams/statistics/{id}?fromDate=... — see
    module docstring point 2 on fromDate's unconfirmed semantics."""
    try:
        data = _get(f"/teams/statistics/{team_id}", {"fromDate": from_date})
    except requests.HTTPError:
        return None
    if not data:
        return None
    return data[0] if isinstance(data, list) else data


def cap_outliers(values, multiplier=1.5, min_cap=3):
    """Clip any single game's goal count before it enters the
    recency-weighted average, so one freak blowout doesn't single-
    handedly drag a team's projection up. Cap is set relative to the
    sample's own median (1.5x by default) rather than a fixed number,
    so it scales with how the team's actually been playing — a team
    with a median of 2 gets capped near 3, a high-scoring team with a
    median of 5 gets capped near 7.5, not squashed to the same ceiling.
    min_cap keeps the cap from collapsing to something tiny (e.g. a
    median of 1 shouldn't cap everything at 1.5). Needs at least 3
    games to compute a median worth trusting; below that, values pass
    through unchanged — capping off 1-2 data points isn't a real
    outlier detection, just noise.

    NOTE: this only affects the RECENCY-WEIGHTED projection input.
    last5_gf (used for display and for hit_rate()'s empirical over/under
    count) still holds the real, uncapped scores — capping is a
    modeling choice for the lambda estimate, not a rewrite of history.
    """
    if len(values) < 3:
        return values
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    median = sorted_vals[n // 2] if n % 2 == 1 else (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
    cap = max(median * multiplier, min_cap)
    return [min(v, cap) for v in values]


def project_team_goals(team_id, from_date, weight=RECENT_WEIGHT):
    """Blended recency-weighted + season-average goals-for projection —
    same architecture as Strike Zone's project_team_runs()."""
    season = get_team_season_stats(team_id, from_date)
    last5 = get_last_five(team_id)

    season_gpg = None
    if season:
        total = season.get("total", {})
        played = total.get("games", {}).get("played")
        scored = total.get("goals", {}).get("scored")
        if played:
            season_gpg = scored / played

    # last5_gf stays RAW (real scores) for display + hit_rate(); only the
    # values feeding the weighted-average projection get outlier-capped.
    gf_values = [g["gf"] for g in last5]
    capped_values = cap_outliers(gf_values)
    recent_gpg = None
    if len(capped_values) >= 2:
        n = len(capped_values)
        wts = [1.4 ** i for i in range(n)]
        recent_gpg = sum(w * g for w, g in zip(wts, capped_values)) / sum(wts)
    elif capped_values:
        recent_gpg = capped_values[0]

    if season_gpg is None and recent_gpg is None:
        return None
    if season_gpg is None:
        blended = recent_gpg
    elif recent_gpg is None:
        blended = season_gpg
    else:
        blended = weight * recent_gpg + (1 - weight) * season_gpg

    return {
        "lambda": round(blended, 2),
        "last5_gf": gf_values,
        "season_gpg": round(season_gpg, 2) if season_gpg is not None else None,
    }


def get_upcoming_matches(league_id, target_date):
    data = _get("/matches", {"leagueId": league_id, "date": target_date.isoformat()})
    matches = data.get("data", [])
    return [m for m in matches if m.get("state", {}).get("description") == "Not started"]


def season_start_guess(target_date):
    """European hockey seasons typically run Sept–April/May. If scanning
    in the first half of the calendar year, last season started the
    PREVIOUS September. A heuristic, not confirmed against how
    Highlightly itself defines season boundaries — see module docstring
    point 2."""
    year = target_date.year if target_date.month >= 7 else target_date.year - 1
    return f"{year}-09-01"


def render_match_card(league_name, home_name, away_name, lh, la, tot, o55,
                       ph, pa, pt, top2, home_proj, away_proj):
    """Per-match card: win probability bar + top-2 correct-score picks,
    matching Blue Line's card layout. No props section — Highlightly has
    no player-level data (see module docstring SCOPE NOTE) — replaced
    here with a last-5-games line for each team, since that data does
    exist for this source and Blue Line's doesn't have an equivalent to
    show. Correct score trimmed from the original top-9 grid down to the
    top-2 picks — nine near-identical single-digit percentages was more
    choice than useful signal."""

    def render_scores():
        return "".join(
            f"<div style='background:var(--panel2);border-radius:8px;padding:8px;text-align:center'>"
            f"<div style='font-size:12px;color:var(--sub)'>{away_name} {a}-{h} {home_name}</div>"
            f"<div style='font-weight:700;margin-top:2px'>{p*100:.1f}%</div></div>"
            for (a, h), p in top2
        )

    home_hist = "/".join(str(v) for v in home_proj["last5_gf"]) or "—"
    away_hist = "/".join(str(v) for v in away_proj["last5_gf"]) or "—"

    win_bar = f"""<div style="margin:10px 0 6px 0">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;font-size:12px;margin-bottom:4px">
        <div>
          <div>{away_name} {pa*100:.0f}%</div>
          <div style="font-size:10px;color:var(--sub);margin-top:2px">last 5 (old→new): {away_hist}</div>
        </div>
        <span>Tie {pt*100:.0f}%</span>
        <div style="text-align:right">
          <div>{home_name} {ph*100:.0f}%</div>
          <div style="font-size:10px;color:var(--sub);margin-top:2px">last 5 (old→new): {home_hist}</div>
        </div>
      </div>
      <div style="display:flex;height:10px;border-radius:999px;overflow:hidden;background:var(--panel2)">
        <div style="width:{pa*100:.1f}%;background:#ff4d5a"></div>
        <div style="width:{pt*100:.1f}%;background:#5a5f7a"></div>
        <div style="width:{ph*100:.1f}%;background:#4ea1ff"></div>
      </div>
    </div>"""


    return f"""<div class="builderPanel">
      <div style="font-size:11px;color:var(--sub);text-transform:uppercase;letter-spacing:.03em">{league_name}</div>
      <h3 style="margin:2px 0 4px 0;font-size:17px">{away_name} @ {home_name} — Total {tot:.2f}</h3>
      <p style="margin:0;color:var(--sub);font-size:13px">Proj: {away_name} {la:.2f} - {lh:.2f} {home_name} | O5.5 {o55*100:.0f}%</p>
      {win_bar}
      <div style="margin-top:12px">
        <div style="font-size:12px;color:var(--sub);margin-bottom:6px">Correct Score</div>
        <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:6px">{render_scores()}</div>
      </div>
    </div>"""


def build_legs_and_cards(target_date):
    """Returns (legs, cards_html). legs feeds the Safest Bet Builder;
    cards_html is the per-match win-prob/correct-score breakdown that
    was previously missing entirely from this tool's output."""
    legs = []
    cards = ""
    from_date = season_start_guess(target_date)

    for league in LEAGUE_TARGETS:
        print(f"Scanning {league['name']} ({league['country']})...")
        matches = get_upcoming_matches(league["id"], target_date)
        print(f"  {len(matches)} fixtures found")

        for m in matches:
            home, away = m["homeTeam"], m["awayTeam"]
            match_label = f"{home['name']} vs {away['name']}"

            home_proj = project_team_goals(home["id"], from_date)
            away_proj = project_team_goals(away["id"], from_date)

            for team, proj in ((home, home_proj), (away, away_proj)):
                if not proj:
                    continue
                line = safe_line(proj["lambda"])
                if not line:
                    continue
                prob = prob_over(proj["lambda"], line)
                legs.append({
                    "match": match_label,
                    "subject": team["name"],
                    "market": f"{team['name']} Over {line} Goals",
                    "prob": round(prob * 100),
                    "hit_rate": hit_rate(proj["last5_gf"], line),
                    "category": f"{league['name']} Team Total",
                    "detail": f"proj {proj['lambda']} goals" +
                              (f" · season {proj['season_gpg']}/gm" if proj["season_gpg"] is not None else ""),
                    "history": "/".join(str(v) for v in proj["last5_gf"]) or None,
                })

            if home_proj and away_proj:
                total_lambda = home_proj["lambda"] + away_proj["lambda"]
                line = safe_line(total_lambda)
                if line:
                    prob = prob_over(total_lambda, line)
                    legs.append({
                        "match": match_label,
                        "subject": match_label,
                        "market": f"Game Over {line} Total Goals",
                        "prob": round(prob * 100),
                        # No hit_rate here deliberately — same reasoning
                        # as Blitz IQ's Game Total leg: this combines
                        # two teams' SEPARATE scoring histories, not a
                        # real shared head-to-head record, so there's
                        # no genuine paired history to count against.
                        "hit_rate": None,
                        "category": f"{league['name']} Game Total",
                        "detail": f"proj {round(total_lambda, 2)} goals combined",
                        "history": None,
                    })

                # NEW: build the per-match card (win prob bar + correct
                # score grid) that this tool was previously missing.
                lh, la = home_proj["lambda"], away_proj["lambda"]
                tot = lh + la
                o55 = prob_over(tot, 5.5)
                ph, pa, pt, top2 = win_probs_and_scores(lh, la)
                cards += render_match_card(
                    league["name"], home["name"], away["name"],
                    lh, la, tot, o55, ph, pa, pt, top2, home_proj, away_proj,
                )

    return legs, cards


HTML_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Euro Ice — {date}</title>
<style>
  :root{{--bg:#0b0f14; --panel:#121820; --panel2:#161d27; --border:#233040; --text:#e8edf2; --sub:#8b98a8; --amber:#facc15; --green:#22c55e;}}
  body{{margin:0; background:var(--bg); color:var(--text); font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; padding:16px; max-width:640px; margin:0 auto;}}
  h1{{font-size:20px; margin-bottom:4px;}}
  .sub{{color:var(--sub); font-size:13px; margin-bottom:18px;}}
  .builderPanel{{background:var(--panel); border:1px solid var(--border); border-radius:12px; padding:16px; margin-bottom:14px;}}
  .builderTitle{{font-size:15px; font-weight:800; margin-bottom:10px;}}
  .builderToggles{{display:flex; gap:10px; flex-wrap:wrap; margin-bottom:10px; font-size:12px;}}
  .builderToggles label{{display:flex; align-items:center; gap:4px; color:var(--text); cursor:pointer;}}
  .builderControls{{display:flex; gap:8px; align-items:center; margin-bottom:6px; flex-wrap:wrap;}}
  .builderControls label{{font-size:12px; color:var(--sub);}}
  .builderControls input{{width:70px; background:var(--panel2); border:1px solid var(--border); color:var(--text); border-radius:6px; padding:6px 8px; font-size:13px;}}
  .builderBtn{{background:var(--green); color:#04140a; font-weight:700; border:none; padding:7px 14px; border-radius:6px; font-size:13px; cursor:pointer;}}
  .builderBtnAlt{{background:var(--panel2); border:1px solid var(--border); color:var(--text); padding:7px 14px; border-radius:6px; font-size:13px; cursor:pointer;}}
  .builderResult{{font-size:12px; color:var(--sub);}}
  .legRow{{display:flex; justify-content:space-between; padding:5px 0; border-bottom:1px solid var(--border);}}
  .footnote{{font-size:11px; color:var(--sub); text-align:center; margin-top:20px; line-height:1.6;}}
</style></head>
<body>
  <h1>🧊 Euro Ice — Team &amp; Game Totals</h1>
  <div class="sub">SHL · Czech Extraliga · DEL — {date} · generated {generated}</div>

  <div id="builderPanel" class="builderPanel">
    <div class="builderTitle">🎯 Safest Bet Builder</div>
    <div id="builderCategoryToggles" class="builderToggles"></div>
    <div class="builderControls">
      <label>Target odds:</label>
      <input type="number" step="0.1" min="1.1" value="5.0" id="targetOdds">
      <label>Max legs:</label>
      <input type="number" step="1" min="2" value="8" id="maxLegs">
      <button class="builderBtn" onclick="buildSafest()">Build</button>
      <button class="builderBtnAlt" onclick="buildSafest()">🔀 Shuffle</button>
    </div>
    <div id="builderResult" class="builderResult">
      Untick any market you don't want considered, set a target odds and leg cap, then tap
      Build. Caps at 2 legs per team/matchup to avoid stacking a team's own Team Total against
      the same game's Game Total. Team-level only — no player props are available from this
      data source. Tap Shuffle for a fresh pick without changing your settings.
    </div>
  </div>

  {cards}

  <div class="footnote">
    Team-totals lambda blends a recency-weighted last-5-games rate ({weight}% recent) with
    season-to-date average, then prices with a Poisson distribution. Lines are set
    automatically below the model's projection for a safety margin. Game Total combines two
    teams' own separate scoring histories, not real head-to-head data — treat it with more
    caution than the single-team legs. Win probability / correct score are regulation-time
    only (no OT/SO modeling from goals-only data). This tool covers goals only; no player
    props are available from Highlightly's free tier.
  </div>

<script>
const LEGS = {legs_json};

function poissonCDF(threshold, lambda){{
  let p = Math.exp(-lambda), cum = p;
  for(let i=1;i<=threshold;i++){{ p = p*lambda/i; cum += p; }}
  return cum;
}}
function initToggles() {{
  const container = document.getElementById('builderCategoryToggles');
  const cats = [...new Set(LEGS.map(l => l.category))];
  container.innerHTML = cats.map(c => `
    <label><input type="checkbox" class="catToggle" value="${{c}}" checked> ${{c}}</label>
  `).join('');
}}
function shuffleArr(arr) {{
  for (let i = arr.length - 1; i > 0; i--) {{
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }}
  return arr;
}}
function tieredShuffle(legs, bandSize) {{
  const bands = {{}};
  legs.forEach(l => {{
    const band = Math.floor(l.prob / bandSize);
    (bands[band] = bands[band] || []).push(l);
  }});
  const keys = Object.keys(bands).map(Number).sort((a,b) => b-a);
  let result = [];
  keys.forEach(k => {{ result = result.concat(shuffleArr(bands[k])); }});
  return result;
}}
function buildSafest() {{
  const target = parseFloat(document.getElementById('targetOdds').value) || 5.0;
  const maxLegs = parseInt(document.getElementById('maxLegs').value) || 8;
  const activeCats = [...document.querySelectorAll('.catToggle:checked')].map(el => el.value);

  const byCategory = {{}};
  LEGS.filter(l => l.prob > 0 && activeCats.includes(l.category)).forEach(l => {{
    (byCategory[l.category] = byCategory[l.category] || []).push(l);
  }});
  const categories = Object.keys(byCategory);
  categories.forEach(c => {{ byCategory[c] = tieredShuffle(byCategory[c], 5); }});
  const cursor = {{}};
  categories.forEach(c => cursor[c] = 0);

  const chosen = [];
  const subjectCount = {{}};
  let combinedOdds = 1;
  let addedThisPass = true;

  while (addedThisPass && combinedOdds < target && chosen.length < maxLegs) {{
    addedThisPass = false;
    for (const cat of categories) {{
      if (combinedOdds >= target || chosen.length >= maxLegs) break;
      const arr = byCategory[cat];
      while (cursor[cat] < arr.length) {{
        const leg = arr[cursor[cat]];
        cursor[cat]++;
        const count = subjectCount[leg.subject] || 0;
        if (count >= 2) continue;
        chosen.push(leg);
        combinedOdds *= 100 / leg.prob;
        subjectCount[leg.subject] = count + 1;
        addedThisPass = true;
        break;
      }}
    }}
  }}

  const out = document.getElementById('builderResult');
  if (!chosen.length) {{ out.innerHTML = 'No legs available to build from.'; return; }}

  const rows = chosen.map(l => `
    <div class="legRow">
      <span>${{l.match}}<br><span style="color:var(--amber)">${{l.market}}</span> <span style="color:var(--sub)">· ${{l.category}}</span>
      ${{l.detail ? `<br><span style="color:var(--sub);font-size:10px">${{l.detail}}</span>` : ''}}
      ${{l.history ? `<br><span style="color:var(--sub);font-size:10px">last games: ${{l.history}}</span>` : ''}}</span>
      <span style="text-align:right"><span style="color:var(--amber);font-weight:bold">${{l.prob}}%</span>${{l.hit_rate ? `<br><span style="color:var(--sub);font-size:11px">${{l.hit_rate.hits}}/${{l.hit_rate.total}}</span>` : ''}}</span>
    </div>
  `).join('');

  const capNote = chosen.length >= maxLegs && combinedOdds < target
    ? ' (hit the leg cap before reaching target — raise Max legs or lower Target odds)'
    : (combinedOdds < target ? ' (ran out of legs before reaching target)' : '');

  out.innerHTML = `
    <div style="color:var(--text);font-size:13px;margin-bottom:6px">
      ${{chosen.length}} legs · est. combined odds ~<b>${{combinedOdds.toFixed(2)}}</b>${{capNote}}
    </div>
    ${{rows}}
    <div style="color:var(--sub);font-size:10px;margin-top:8px;line-height:1.4">
      Estimate multiplies each leg's fair odds (100/probability) — real sportsbook odds
      include their margin, so treat this as a ranking tool, not a firm price.
    </div>
  `;
}}
initToggles();
</script>
</body></html>
"""


def render_html(legs, cards, target_date):
    return HTML_TEMPLATE.format(
        date=target_date.isoformat(),
        generated=datetime.now().strftime("%Y-%m-%d %H:%M"),
        weight=int(RECENT_WEIGHT * 100),
        cards=cards or "<p style='color:var(--sub);text-align:center'>No matchups had enough data for a full card today.</p>",
        legs_json=json.dumps(legs),
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] != "--auto":
        target = date.fromisoformat(sys.argv[1])
    else:
        target = date.today()

    print(f"Fetching Euro Ice slate for {target.isoformat()}…")
    legs, cards = build_legs_and_cards(target)

    if not legs:
        print("No usable legs today — either no fixtures found for the configured "
              "leagues, or no team had enough data to project. Nothing to publish; "
              "docs/ left as whatever the last successful run published.")
        raise SystemExit(0)

    html = render_html(legs, cards, target)
    os.makedirs("docs/euro-ice", exist_ok=True)
    with open("docs/euro-ice/index.html", "w") as f:
        f.write(html)
    with open("docs/euro-ice/euro_ice.json", "w") as f:
        json.dump(legs, f, indent=2, default=str)
    print(f"\nDone. {len(legs)} legs written to docs/euro-ice/index.html")
