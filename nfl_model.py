#!/usr/bin/env python3
"""
Blitz IQ — NFL Team Points Over/Under Predictor
Same architecture as the MLB (Strike Zone) and soccer (Goal IQ) tools, adapted
for NFL scoring:

- Uses ESPN's public (undocumented but documented-elsewhere and no-key-needed)
  API — no signup required.
- Team point totals are modeled with a NORMAL distribution, not Poisson.
  Poisson fits well for low-count events like goals or strikeouts; NFL scores
  are larger, non-1-point increments (3, 6, 7, 8...) and behave much closer
  to a bell curve in practice.
- Recency-weighted last-5-games average, shrunk toward the league average
  for small samples (same small-sample protection built for MLB/soccer,
  since NFL teams only play ~17 games/season — "last 5" is meaningfully
  more of the season than in MLB or soccer).

Setup:
    pip3 install requests --break-system-packages
    python3 nfl_model.py

Output:
    docs/index.html, docs/blitz_iq_predictions.csv

Note: ESPN doesn't publish an official rate limit for this endpoint, but
"excessive requests may be blocked" per their own docs — this script paces
itself conservatively (1.2s between calls) to stay well clear of that,
rather than assuming no limit means no risk.
"""

import os
import sys
import time
import math
import csv
import json
from datetime import datetime, timedelta
import requests

BASE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
REQUEST_DELAY = 1.2  # no official ESPN rate limit is published — paced conservatively anyway
RECENT_GAMES = 5
PRIOR_STRENGTH = 3  # same "worth" of league-average games blended in for small samples as MLB/soccer
DEFAULT_TEAM_STD = 10.0  # rough per-team point stdev; used until we compute a real one from data


def _get(url, params=None, timeout=15):
    try:
        r = requests.get(url, params=params or {}, timeout=timeout)
        time.sleep(REQUEST_DELAY)
        if r.status_code != 200:
            print(f"  [!] {r.status_code} on {url}")
            return None
        return r
    except Exception as e:
        time.sleep(REQUEST_DELAY)
        print(f"  [!] request failed: {url} ({e})")
        return None


def norm_cdf(x, mean, std):
    """Standard normal CDF via the error function — no scipy dependency."""
    if std <= 0:
        return 1.0 if x >= mean else 0.0
    z = (x - mean) / (std * math.sqrt(2))
    return 0.5 * (1 + math.erf(z))


def get_teams():
    r = _get(f"{BASE}/teams", params={"limit": 40})
    if r is None:
        return []
    try:
        return r.json()['sports'][0]['leagues'][0]['teams']
    except Exception as e:
        print(f"  [!] couldn't parse teams response: {e}")
        return []


team_form_cache = {}


def _extract_completed_games(events, team_id):
    """Shared parsing logic for a schedule response — pulls each completed
    game's scored/allowed for one team."""
    completed = [e for e in events if e.get('competitions', [{}])[0].get('status', {})
                 .get('type', {}).get('completed')]
    completed.sort(key=lambda e: e.get('date', ''))
    recent = completed[-RECENT_GAMES:]
    scored, allowed = [], []
    for e in recent:
        comp = e['competitions'][0]
        competitors = comp.get('competitors', [])
        me = next((c for c in competitors if str(c['team']['id']) == str(team_id)), None)
        opp = next((c for c in competitors if str(c['team']['id']) != str(team_id)), None)
        if not me or not opp:
            continue
        try:
            scored.append(float(me['score']['value']))
            allowed.append(float(opp['score']['value']))
        except (KeyError, TypeError, ValueError):
            continue
    return scored, allowed


def get_team_form(team_id):
    """Last N completed games for a team — points scored and allowed.

    Early in a new season (Week 1, before most/all games have kicked off),
    a team can have ZERO completed games this season — that's not a parse
    bug, it's genuinely no data yet. In that case, fall back to the tail
    end of LAST season, so the tool isn't dead for the whole first week or
    two of the year. A form dict is tagged with which source it actually
    used ('current' or 'prior_season') so that's visible on the page rather
    than silently blending two different seasons' games without saying so.
    """
    if team_id in team_form_cache:
        return team_form_cache[team_id]

    r = _get(f"{BASE}/teams/{team_id}/schedule")
    scored, allowed, source = [], [], 'current'
    if r is not None:
        try:
            events = r.json().get('events', [])
            scored, allowed = _extract_completed_games(events, team_id)
        except Exception as e:
            print(f"  [!] couldn't parse schedule for team {team_id}: {e}")

    if not scored:
        current_year = datetime.now().year
        prior_season = current_year - 1
        r2 = _get(f"{BASE}/teams/{team_id}/schedule", params={"season": prior_season})
        if r2 is not None:
            try:
                events2 = r2.json().get('events', [])
                scored, allowed = _extract_completed_games(events2, team_id)
                source = 'prior_season'
            except Exception as e:
                print(f"  [!] couldn't parse prior-season schedule for team {team_id}: {e}")

    if not scored:
        return None

    n = len(scored)
    form = {
        'avg_scored': round(sum(scored) / n, 1),
        'avg_allowed': round(sum(allowed) / n, 1),
        'n_games': n,
        'source': source,
        'scored_list': scored,
        'allowed_list': allowed,
    }
    team_form_cache[team_id] = form
    return form


def recency_weighted(values):
    n = len(values)
    if n == 0:
        return None
    wts = [1.3 ** i for i in range(n)]
    return sum(w * v for w, v in zip(wts, values)) / sum(wts)


def shrink(value, n, league_avg, prior=PRIOR_STRENGTH):
    return (n * value + prior * league_avg) / (n + prior)


def league_averages(all_forms):
    scored = [f['avg_scored'] for f in all_forms if f]
    allowed = [f['avg_allowed'] for f in all_forms if f]
    lg_scored = sum(scored) / len(scored) if scored else 22.0
    lg_allowed = sum(allowed) / len(allowed) if allowed else 22.0
    return lg_scored, lg_allowed


def predict(h_form, a_form, lg_scored, lg_allowed):
    h_recent_scored = recency_weighted(h_form['scored_list'])
    h_recent_allowed = recency_weighted(h_form['allowed_list'])
    a_recent_scored = recency_weighted(a_form['scored_list'])
    a_recent_allowed = recency_weighted(a_form['allowed_list'])

    h_scored = shrink(h_recent_scored, h_form['n_games'], lg_scored)
    h_allowed = shrink(h_recent_allowed, h_form['n_games'], lg_allowed)
    a_scored = shrink(a_recent_scored, a_form['n_games'], lg_scored)
    a_allowed = shrink(a_recent_allowed, a_form['n_games'], lg_allowed)

    exp_home = h_scored * (a_allowed / lg_allowed)
    exp_away = a_scored * (h_allowed / lg_allowed)
    exp_total = round(exp_home + exp_away, 1)

    total_std = math.sqrt(DEFAULT_TEAM_STD ** 2 + DEFAULT_TEAM_STD ** 2)

    return {
        'exp_home': round(exp_home, 1), 'exp_away': round(exp_away, 1),
        'exp_total': exp_total, 'total_std': round(total_std, 1),
    }


def over_under_prob(exp_total, total_std, line):
    p_under = norm_cdf(line, exp_total, total_std)
    return round((1 - p_under) * 100), round(p_under * 100)


def get_week_scoreboard():
    r = _get(f"{BASE}/scoreboard")
    if r is None:
        return []
    try:
        return r.json().get('events', [])
    except Exception as e:
        print(f"  [!] couldn't parse scoreboard: {e}")
        return []


PLAYER_STAT_CONFIG = {
    'QB': {'stat': 'YDS', 'label': 'Passing Yards', 'dist': 'normal', 'std': 55, 'prior': 235},
    'RB': {'stat': 'YDS', 'label': 'Rushing Yards', 'dist': 'normal', 'std': 28, 'prior': 60},
    'WR': {'stat': 'REC', 'label': 'Receptions', 'dist': 'poisson', 'std': None, 'prior': 4.0},
    'TE': {'stat': 'REC', 'label': 'Receptions', 'dist': 'poisson', 'std': None, 'prior': 3.5},
}

depth_chart_cache = {}


def get_starters(team_id):
    if team_id in depth_chart_cache:
        return depth_chart_cache[team_id]
    r = _get(f"{BASE}/teams/{team_id}/depthcharts")
    starters = {}
    if r is None:
        print(f"    [!] depth chart fetch failed for team {team_id} (no response)")
        depth_chart_cache[team_id] = starters
        return starters
    try:
        data = r.json()
        top_keys = list(data.keys())
        depthchart_raw = data.get('depthchart')
        if isinstance(depthchart_raw, list):
            groups = depthchart_raw
        elif isinstance(depthchart_raw, dict):
            groups = depthchart_raw.get('items') or depthchart_raw.get('athletes') or []
            if not groups:
                print(f"    [!] depth chart for team {team_id}: 'depthchart' is a dict with keys {list(depthchart_raw.keys())} — none matched expected wrapper keys")
        else:
            groups = []
        if not groups:
            print(f"    [!] depth chart for team {team_id}: 'depthchart' key present but empty/unrecognized shape (type={type(depthchart_raw).__name__}). Top-level keys were: {top_keys}")
        for group in groups:
            positions = group.get('positions', {})
            for pos_key, pos_data in positions.items():
                pos_abbr = (pos_data.get('position', {}).get('abbreviation')
                            or pos_key or '').upper()
                if pos_abbr not in PLAYER_STAT_CONFIG:
                    continue
                if pos_abbr in starters:
                    continue
                slots = pos_data.get('athletes', [])
                if slots:
                    athlete = slots[0].get('athlete', slots[0])
                    starters[pos_abbr] = {
                        'id': athlete.get('id'), 'name': athlete.get('displayName', athlete.get('fullName', '?')),
                    }
        if groups and not starters:
            sample = groups[0]
            print(f"    [!] depth chart for team {team_id}: found {len(groups)} group(s) but matched 0 of QB/RB/WR/TE. "
                  f"First group's keys: {list(sample.keys()) if isinstance(sample, dict) else type(sample).__name__}, "
                  f"positions sub-keys (if any): {list(sample.get('positions', {}).keys()) if isinstance(sample, dict) else 'n/a'}")
    except Exception as e:
        print(f"  [!] couldn't parse depth chart for team {team_id}: {e}")
    depth_chart_cache[team_id] = starters
    return starters


player_gamelog_cache = {}


def _fetch_gamelog_values(athlete_id, stat_key, season):
    r = _get(f"https://site.web.api.espn.com/apis/common/v3/sports/football/nfl/athletes/{athlete_id}/gamelog",
             params={"season": season})
    values = []
    if r is None:
        print(f"    [!] gamelog fetch failed for athlete {athlete_id}, season {season} (no response)")
        return values
    try:
        data = r.json()
        top_keys = list(data.keys())
        if top_keys == ['filters']:
            print(f"    [!] gamelog for athlete {athlete_id}, season {season}: still only got 'filters' even with season param. Filters content: {data.get('filters')}")
        season_data = data.get('seasonTypes', [])
        if not season_data and top_keys != ['filters']:
            print(f"    [!] gamelog for athlete {athlete_id}, season {season}: no 'seasonTypes' key. Top-level keys were: {top_keys}")

        root_labels = data.get('labels') or data.get('names') or data.get('displayNames')
        if season_data and not root_labels:
            print(f"    [DIAG] athlete {athlete_id}, season {season}: no root-level labels/names/displayNames found. Full top-level keys: {top_keys}")

        found_any_label_set = bool(root_labels)
        printed_sample = False
        for st in season_data:
            for cat in st.get('categories', []):
                if not printed_sample:
                    cat_keys = list(cat.keys())
                    events_sample = cat.get('events', [])[:1]
                    totals_sample = cat.get('totals')
                    print(f"    [DIAG] athlete {athlete_id}, season {season}: category top-level keys: {cat_keys}, root_labels sample: {str(root_labels)[:200]}")
                    print(f"    [DIAG] first event (raw, truncated to 500 chars): {str(events_sample)[:500]}")
                    print(f"    [DIAG] category 'totals' (raw, truncated to 500 chars): {str(totals_sample)[:500]}")
                    printed_sample = True
                if not root_labels:
                    continue
                for game in cat.get('events', []):
                    stats = game.get('stats', [])
                    if stat_key in root_labels:
                        idx = root_labels.index(stat_key)
                        try:
                            values.append(float(stats[idx]))
                        except (IndexError, ValueError, TypeError):
                            continue
        if season_data and not values:
            print(f"    [!] gamelog for athlete {athlete_id}, season {season}, stat '{stat_key}': parsed categories but found no matching values (found_any_labels={found_any_label_set}) — stat_key likely doesn't match ESPN's actual label name")
    except Exception as e:
        print(f"  [!] couldn't parse gamelog for athlete {athlete_id}, season {season}: {e}")
    return values


def get_player_gamelog(athlete_id, stat_key):
    cache_key = (athlete_id, stat_key)
    if cache_key in player_gamelog_cache:
        return player_gamelog_cache[cache_key]

    current_year = datetime.now().year
    values = _fetch_gamelog_values(athlete_id, stat_key, current_year)
    if not values:
        values = _fetch_gamelog_values(athlete_id, stat_key, current_year - 1)

    values = values[-RECENT_GAMES:]
    player_gamelog_cache[cache_key] = values
    return values


def project_player_stat(pos_abbr, athlete_id, name):
    cfg = PLAYER_STAT_CONFIG[pos_abbr]
    values = get_player_gamelog(athlete_id, cfg['stat'])
    if not values:
        return None
    n = len(values)
    recent = recency_weighted(values)
    shrunk = shrink(recent, n, cfg['prior'])
    return {
        'name': name, 'position': pos_abbr, 'label': cfg['label'],
        'dist': cfg['dist'], 'std': cfg['std'],
        'projected': round(shrunk, 1), 'n_games': n, 'recent_values': values,
        'athlete_id': athlete_id,  # NEW: needed later to look up this
                                     # player's actual boxscore line for
                                     # the results tracker
    }


def get_team_player_props(team_id):
    starters = get_starters(team_id)
    if not starters:
        print(f"    no starters identified for team {team_id} — 0 player props possible for this team")
    props = []
    for pos_abbr, athlete in starters.items():
        if not athlete.get('id'):
            continue
        proj = project_player_stat(pos_abbr, athlete['id'], athlete['name'])
        if proj:
            props.append(proj)
        else:
            print(f"    {athlete['name']} ({pos_abbr}): no gamelog data found")
    return props


def player_over_under_prob(proj, line):
    if proj['dist'] == 'poisson':
        p_under = poisson_cdf(math.floor(line), proj['projected'])
    else:
        p_under = norm_cdf(line, proj['projected'], proj['std'])
    return round((1 - p_under) * 100), round(p_under * 100)


def poisson_pmf(k, lam):
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def poisson_cdf(k, lam):
    return sum(poisson_pmf(i, lam) for i in range(int(k) + 1))


def normal_prop(mean, std, factor=0.72, round_to=0.5):
    if mean is None or std is None:
        return None
    raw_line = mean * factor
    line = math.floor(raw_line / round_to) * round_to
    if line < round_to:
        line = round_to
    prob_over = 1 - norm_cdf(line, mean, std)
    return {"line": line, "prob": round(prob_over * 100), "avg": mean}


def poisson_prop(mean, factor=0.72):
    if mean is None:
        return None
    raw_line = mean * factor
    line = math.floor(raw_line * 2) / 2
    if line < 0.5:
        line = 0.5
    threshold = int(math.floor(line)) + 1
    prob = 1 - poisson_cdf(threshold - 1, mean)
    return {"line": line, "prob": round(prob * 100), "avg": mean}


def hit_rate(values, line):
    if not values:
        return None
    hits = sum(1 for v in values if v > line)
    return {"hits": hits, "total": len(values)}


def format_history(lst):
    if not lst:
        return None
    return "/".join(str(v) for v in lst)


def build_legs(predictions):
    legs = []
    for p in predictions:
        match_label = p["match"]
        hf, af = p["home_form"], p["away_form"]
        game_id = p.get("game_id")  # NEW
        game_date = (p.get("date") or "")[:10]  # NEW

        home_total = normal_prop(p["exp_home"], DEFAULT_TEAM_STD)
        if home_total:
            legs.append({
                "match": match_label,
                "market": f"{p['home_team']} Over {home_total['line']} Points",
                "prob": home_total["prob"], "category": "Team Total",
                "hit_rate": hit_rate(hf.get("scored_list"), home_total["line"]),
                "detail": f"proj {home_total['avg']} pts ({hf['n_games']}gm{' · last season' if hf.get('source') == 'prior_season' else ''})",
                "history": format_history(hf.get("scored_list")),
                "game_id": game_id, "game_date": game_date, "is_home": True, "line": home_total["line"],
            })
        away_total = normal_prop(p["exp_away"], DEFAULT_TEAM_STD)
        if away_total:
            legs.append({
                "match": match_label,
                "market": f"{p['away_team']} Over {away_total['line']} Points",
                "prob": away_total["prob"], "category": "Team Total",
                "hit_rate": hit_rate(af.get("scored_list"), away_total["line"]),
                "detail": f"proj {away_total['avg']} pts ({af['n_games']}gm{' · last season' if af.get('source') == 'prior_season' else ''})",
                "history": format_history(af.get("scored_list")),
                "game_id": game_id, "game_date": game_date, "is_home": False, "line": away_total["line"],
            })

        game_total = normal_prop(p["exp_total"], p["total_std"])
        if game_total:
            legs.append({
                "match": match_label,
                "market": f"Game Over {game_total['line']} Total Points",
                "prob": game_total["prob"], "category": "Game Total",
                "hit_rate": None,
                "detail": f"proj {game_total['avg']} pts ({hf['n_games']}v{af['n_games']}gm)",
                "history": None,
                "game_id": game_id, "game_date": game_date, "line": game_total["line"],
            })

        for team_name, props in [(p["home_team"], p.get("home_props") or []),
                                   (p["away_team"], p.get("away_props") or [])]:
            for prop in props:
                if prop["dist"] == "poisson":
                    result = poisson_prop(prop["projected"])
                else:
                    result = normal_prop(prop["projected"], prop["std"])
                if not result:
                    continue
                legs.append({
                    "match": match_label,
                    "market": f"{prop['name']} Over {result['line']} {prop['label']}",
                    "prob": result["prob"], "category": prop["label"],
                    "hit_rate": hit_rate(prop.get("recent_values"), result["line"]),
                    "detail": f"proj {result['avg']} ({prop['n_games']}gm)",
                    "history": format_history(prop.get("recent_values")),
                    "game_id": game_id, "game_date": game_date, "line": result["line"],
                    "athlete_id": prop.get("athlete_id"), "stat_key": PLAYER_STAT_CONFIG[prop["position"]]["stat"],
                    "is_home": team_name == p["home_team"],
                })
    return legs


BUILDER_TEMPLATE = """
<div style="background:#1a1f26;border-radius:12px;padding:16px;margin:12px 0;border:1px solid #2a3038">
  <div style="font-size:14px;font-weight:bold;margin-bottom:10px">🎯 Safest Bet Builder</div>
  <div id="categoryToggles" style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:10px;font-size:12px"></div>
  <div style="display:flex;gap:8px;align-items:center;margin-bottom:6px;flex-wrap:wrap">
    <label style="font-size:12px;color:#aaa">Target odds:</label>
    <input id="targetOdds" type="number" step="0.1" min="1.1" value="5.0"
      style="width:70px;background:#0f1318;border:1px solid #333;color:white;border-radius:6px;padding:6px 8px;font-size:13px">
    <label style="font-size:12px;color:#aaa">Max legs:</label>
    <input id="maxLegs" type="number" step="1" min="2" value="8"
      style="width:55px;background:#0f1318;border:1px solid #333;color:white;border-radius:6px;padding:6px 8px;font-size:13px">
    <button onclick="buildSafest()"
      style="background:#3a7d7a;border:none;color:white;padding:7px 14px;border-radius:6px;font-size:13px;cursor:pointer">
      Build
    </button>
    <button onclick="buildSafest()"
      style="background:#2a3038;border:1px solid #444;color:white;padding:7px 14px;border-radius:6px;font-size:13px;cursor:pointer">
      🔀 Shuffle
    </button>
  </div>
  <div id="builderResult" style="font-size:12px;color:#888">
    Untick any market type you don't want considered, set a target odds and
    leg cap, then tap Build. It rotates through whichever categories are
    ticked, groups near-tied legs and shuffles within each group so it draws
    from more of the week's games rather than always the exact same few, and
    caps at 2 legs per game to avoid stacking correlated legs from one
    matchup. Tap Shuffle for a fresh pick among equally-safe options without
    changing your settings.
  </div>
</div>
<script>
const LEGS = {legs_json};

function initCategoryToggles() {{
  const container = document.getElementById('categoryToggles');
  const cats = [...new Set(LEGS.map(l => l.category))];
  container.innerHTML = cats.map(c => `
    <label style="display:flex;align-items:center;gap:4px;color:#ccc;cursor:pointer">
      <input type="checkbox" class="catToggle" value="${{c}}" checked>
      ${{c}}
    </label>
  `).join('');
}}
initCategoryToggles();

function shuffle(arr) {{
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
  const bandKeys = Object.keys(bands).map(Number).sort((a, b) => b - a);
  let result = [];
  bandKeys.forEach(b => {{ result = result.concat(shuffle(bands[b])); }});
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
  const matchCount = {{}};
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
        const count = matchCount[leg.match] || 0;
        if (count >= 2) continue;
        chosen.push(leg);
        combinedOdds *= 100 / leg.prob;
        matchCount[leg.match] = count + 1;
        addedThisPass = true;
        break;
      }}
    }}
  }}

  const el = document.getElementById('builderResult');
  if (!chosen.length) {{
    el.innerHTML = 'No legs available to build from.';
    return;
  }}

  const rows = chosen.map(l =>
    `<div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #2a3038">
       <span>${{l.match}}<br><span style="color:#7ec8ff">${{l.market}}</span> <span style="color:#555">· ${{l.category}}</span>
       ${{l.detail ? `<br><span style="color:#666;font-size:10px">${{l.detail}}</span>` : ''}}
       ${{l.history ? `<br><span style="color:#555;font-size:10px">last games: ${{l.history}}</span>` : ''}}</span>
       <span style="text-align:right"><span style="color:#ffeb3b;font-weight:bold">${{l.prob}}%</span>${{l.hit_rate ? `<br><span style="color:#888;font-size:11px">${{l.hit_rate.hits}}/${{l.hit_rate.total}}</span>` : ''}}</span>
     </div>`
  ).join('');

  const capNote = chosen.length >= maxLegs && combinedOdds < target
    ? ' (hit the leg cap before reaching target — raise Max legs or lower Target odds)'
    : (combinedOdds < target ? ' (ran out of legs before reaching target)' : '');

  el.innerHTML = `
    <div style="color:white;font-size:13px;margin-bottom:6px">
      ${{chosen.length}} legs · est. combined odds ~<b>${{combinedOdds.toFixed(2)}}</b>${{capNote}}
    </div>
    ${{rows}}
    <div style="color:#666;font-size:10px;margin-top:8px;line-height:1.4">
      Estimate multiplies each leg's fair odds (100/probability) — real
      sportsbook odds include their margin and legs within the same game
      aren't fully independent, so treat this as a ranking tool, not a firm
      price. Team/game totals use a Normal-distribution projection; WR/TE
      receptions use Poisson. All lines are set automatically below the
      model's projection for a safety margin.
    </div>
  `;
}}
</script>
"""


def build_predictions():
    print("Fetching teams…")
    teams = get_teams()
    if not teams:
        print("No teams returned — aborting.")
        return []

    print(f"Fetching form for {len(teams)} teams…")
    all_forms = {}
    for t in teams:
        tid = t['team']['id']
        form = get_team_form(tid)
        all_forms[tid] = form
        tag = f" (source: {form['source']})" if form else " (no data)"
        print(f"  {t['team']['displayName']}{tag}")

    lg_scored, lg_allowed = league_averages(all_forms.values())
    print(f"League averages: {lg_scored:.1f} scored/game, {lg_allowed:.1f} allowed/game")

    print("Fetching this week's scoreboard…")
    events = get_week_scoreboard()
    print(f"{len(events)} games this week")

    predictions = []
    for e in events:
        comp = e.get('competitions', [{}])[0]
        competitors = comp.get('competitors', [])
        home = next((c for c in competitors if c.get('homeAway') == 'home'), None)
        away = next((c for c in competitors if c.get('homeAway') == 'away'), None)
        if not home or not away:
            continue
        if comp.get('status', {}).get('type', {}).get('completed'):
            continue

        h_id, a_id = home['team']['id'], away['team']['id']
        h_form, a_form = all_forms.get(h_id), all_forms.get(a_id)
        if not h_form or not a_form:
            print(f"  skipping {away['team']['displayName']} @ {home['team']['displayName']}: missing form data")
            continue

        proj = predict(h_form, a_form, lg_scored, lg_allowed)
        print(f"  Player props: {away['team']['displayName']} @ {home['team']['displayName']}")
        home_props = get_team_player_props(h_id)
        away_props = get_team_player_props(a_id)
        predictions.append({
            'date': e.get('date', ''),
            'game_id': e.get('id'),  # NEW: needed later to look up the real
                                       # final result for the results tracker
            'match': f"{away['team']['displayName']} @ {home['team']['displayName']}",
            'home_team': home['team']['displayName'], 'away_team': away['team']['displayName'],
            'home_form': h_form, 'away_form': a_form,
            'home_props': home_props, 'away_props': away_props,
            **proj,
        })

    predictions.sort(key=lambda x: x['exp_total'], reverse=True)
    return predictions


HTML_TEMPLATE = """<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Blitz IQ — NFL</title></head>
<body style="background:#0b0f14;color:white;font-family:Arial;padding:12px;max-width:600px;margin:auto">
<h2 style="text-align:center">🏈 BLITZ IQ — NFL Team Points</h2>
<p style="text-align:center;color:#888;font-size:11px">Recency-weighted scoring/allowed rates, Normal-distribution projected · {generated}</p>
<p style="text-align:center;margin-bottom:16px"><a href="blitz_iq_predictions.csv" download style="background:#222;border:1px solid #444;color:white;padding:8px 14px;border-radius:8px;text-decoration:none;font-size:13px">⬇ Download CSV</a></p>
<p style="text-align:center;margin-bottom:16px"><a href="results/index.html" style="color:#ffeb3b;text-decoration:none;font-size:12px">📊 Results Tracker</a></p>
{builder}
{cards}
<p style="text-align:center;color:#666;font-size:10px;margin-top:20px">Enter your book's Over/Under line and odds to compute an edge the same way as the MLB/soccer tools — this page shows the model's own projection only.</p>
</body></html>"""

CARD_TEMPLATE = """<div style="background:#1a1f26;border-radius:12px;padding:16px;margin:12px 0;border:1px solid #2a3038">
  <div style="font-size:11px;color:#999;margin-bottom:4px">{date}</div>
  <div style="font-size:17px;font-weight:bold;margin-bottom:10px">{match}</div>
  <div style="display:flex;justify-content:space-between;text-align:center">
    <div><div style="color:#aaa;font-size:11px">{away_team}</div><div style="color:#ffeb3b;font-size:20px;font-weight:bold">{exp_away}</div></div>
    <div><div style="color:#aaa;font-size:11px">TOTAL</div><div style="color:#7ec8ff;font-size:22px;font-weight:bold">{exp_total}</div></div>
    <div><div style="color:#aaa;font-size:11px">{home_team}</div><div style="color:#ffeb3b;font-size:20px;font-weight:bold">{exp_home}</div></div>
  </div>
  <div style="background:#0f1318;border-radius:8px;padding:8px;margin-top:10px;display:flex;justify-content:space-between;font-size:11px">
    <div>{away_team}: {away_scored} scored/gm • {away_allowed} allowed/gm ({away_n}gm{away_source_tag})</div>
  </div>
  <div style="background:#0f1318;border-radius:8px;padding:8px;margin-top:6px;font-size:11px">
    {home_team}: {home_scored} scored/gm • {home_allowed} allowed/gm ({home_n}gm{home_source_tag})
  </div>
  {player_props_html}
</div>"""

PLAYER_PROP_ROW = """<div style="display:flex;justify-content:space-between;font-size:11px;padding:5px 0;border-top:1px solid #232a33">
  <div>{name} ({position}) — {label}</div>
  <div style="color:#c792ea;font-weight:bold">{projected} <span style="color:#666;font-weight:normal">({n_games}gm)</span></div>
</div>"""


def player_props_section(team_label, props):
    if not props:
        return ""
    rows = "".join(PLAYER_PROP_ROW.format(**p) for p in props)
    return f'<div style="margin-top:8px"><div style="color:#888;font-size:10px;text-transform:uppercase;margin-bottom:2px">{team_label} Player Props</div>{rows}</div>'


def make_html(predictions):
    cards = "".join(CARD_TEMPLATE.format(
        date=p['date'][:16].replace('T', ' '), match=p['match'],
        away_team=p['away_team'], home_team=p['home_team'],
        exp_away=p['exp_away'], exp_home=p['exp_home'], exp_total=p['exp_total'],
        away_scored=p['away_form']['avg_scored'], away_allowed=p['away_form']['avg_allowed'],
        away_n=p['away_form']['n_games'],
        away_source_tag=' · last season' if p['away_form'].get('source') == 'prior_season' else '',
        home_scored=p['home_form']['avg_scored'], home_allowed=p['home_form']['avg_allowed'],
        home_n=p['home_form']['n_games'],
        home_source_tag=' · last season' if p['home_form'].get('source') == 'prior_season' else '',
        player_props_html=(
            player_props_section(p['away_team'], p.get('away_props', []))
            + player_props_section(p['home_team'], p.get('home_props', []))
        ),
    ) for p in predictions)
    if not cards:
        cards = '<p style="text-align:center;color:#888">No upcoming games with usable form data this week.</p>'

    legs = build_legs(predictions)
    builder = BUILDER_TEMPLATE.format(legs_json=json.dumps(legs)) if legs else ""

    return HTML_TEMPLATE.format(
        generated=datetime.now().strftime('%d %b %H:%M'), builder=builder, cards=cards,
    )


def write_csv(predictions, path):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Date', 'Match', 'HomeTeam', 'AwayTeam',
            'HomeScoredAvg', 'HomeAllowedAvg', 'HomeSampleSize',
            'AwayScoredAvg', 'AwayAllowedAvg', 'AwaySampleSize',
            'ExpHome', 'ExpAway', 'ExpTotal', 'TotalStd',
            'Line', 'OverOdds', 'UnderOdds',
            'ActualHomeScore', 'ActualAwayScore', 'HitOrMiss',
        ])
        for p in predictions:
            writer.writerow([
                p['date'], p['match'], p['home_team'], p['away_team'],
                p['home_form']['avg_scored'], p['home_form']['avg_allowed'], p['home_form']['n_games'],
                p['away_form']['avg_scored'], p['away_form']['avg_allowed'], p['away_form']['n_games'],
                p['exp_home'], p['exp_away'], p['exp_total'], p['total_std'],
                '', '', '',
                '', '', '',
            ])


def write_player_props_csv(predictions, path):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Date', 'Match', 'Team', 'Player', 'Position', 'PropType',
            'Projected', 'SampleSize', 'RecentValues',
            'Line', 'OverOdds', 'UnderOdds', 'ActualValue', 'HitOrMiss',
        ])
        for p in predictions:
            for team_label, props in [(p['away_team'], p.get('away_props', [])),
                                       (p['home_team'], p.get('home_props', []))]:
                for prop in props:
                    writer.writerow([
                        p['date'], p['match'], team_label, prop['name'], prop['position'], prop['label'],
                        prop['projected'], prop['n_games'], '; '.join(str(v) for v in prop['recent_values']),
                        '', '', '', '', '',
                    ])


if __name__ == "__main__":
    predictions = build_predictions()
    os.makedirs('docs/blitz-iq', exist_ok=True)
    with open('docs/blitz-iq/index.html', 'w') as f:
        f.write(make_html(predictions))
    write_csv(predictions, 'docs/blitz-iq/blitz_iq_predictions.csv')
    write_player_props_csv(predictions, 'docs/blitz-iq/blitz_iq_player_props.csv')
    with open('docs/blitz-iq/blitz_iq.json', 'w') as f:
        json.dump(predictions, f, indent=2, default=str)

    try:
        import blitz_iq_results_tracker
        blitz_iq_results_tracker.run_results_tracker(build_legs(predictions))
    except Exception as e:
        print(f"[!] Results tracker failed, but the rest of this run succeeded: {e}")

    print(f"\nDone — {len(predictions)} games projected.")
