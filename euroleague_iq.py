#!/usr/bin/env python3
"""
EuroLeague IQ — EuroLeague Basketball Team Points + Player Props Predictor
--------------------------------------------------------------
Same architecture/sibling as Blitz IQ (NFL), Orange Line (NBA), Match IQ,
etc: recency-weighted recent form, small-sample shrinkage toward a league
average, Normal distribution for team point totals, a Safest Bet Builder,
and a results tracker.

DATA SOURCE: EuroLeague's own public v2 REST API --
  https://api-live.euroleague.net/v2/competitions/E/seasons/{seasonCode}/...
This is NOT officially documented by EuroLeague, but it's a genuinely
public, keyless, unauthenticated API confirmed via its own Swagger page
(api-live.euroleague.net/swagger) and used in production by several
open-source clients (e.g. github.com/aimon7/euroleague-api, whose TypeScript
source + response fixtures this file's field names/URLs were confirmed
against). Season code format is "E<year>" where <year> is the season's
START year, e.g. "E2026" for the 2026-27 season -- EuroCup uses "U"
instead of "E".

IMPORTANT CAVEAT: this session's sandbox can't reach external hosts other
than GitHub/package registries (confirmed -- both ESPN and this API were
blocked by the sandbox's own egress proxy when tested directly), so the
endpoints/field names below are confirmed from the open-source client's
*documented, tested* source and fixtures, not a live response fetched in
this session. If something comes back empty or [!]/[DIAG] prints show up
in the Actions log, that's the first place to look -- same defensive style
as every other tool in this suite that works against an undocumented API.

Team form: walks each team's own last RECENT_GAMES *played* games from the
season schedule, pulling each game's final score from the box score's
local/road 'total.points' (NOT ESPN -- this is EuroLeague's own confirmed
field name). Normal-distribution team totals, same as Blitz IQ.

Player props: EuroLeague's API has no simple "starters" endpoint like
ESPN's NFL depth chart, so instead of a separate roster fetch, a team's
featured players are derived straight from the SAME recent box scores
already being pulled for team form -- whichever players actually appeared
in at least PLAYER_MIN_GAMES of the last RECENT_GAMES games, ranked by
average points and capped to PLAYER_POOL_SIZE. This is arguably more
robust than a static depth chart (it reflects actual recent rotation/
minutes, not a preseason roster slot). Props offered: Points (normal),
Rebounds (poisson), Assists (poisson) -- same distribution split Blitz IQ
uses for yardage-like vs. count-like stats.

Output:
    docs/euroleague-iq/index.html         -- predictions page
    docs/euroleague_iq_predictions.csv    -- team totals
    docs/euroleague_iq_player_props.csv   -- player props
    docs/euroleague_iq.json               -- raw predictions

Lives inside the main sportsiq monorepo alongside every other sibling
tool, so (fixed 2026-10-20) the predictions page is nested under its own
docs/euroleague-iq/ subfolder like Blue Line, Blitz IQ, Match IQ etc. --
it used to write straight to docs/index.html, which is the shared
Predictor Suite landing page every tool links from, so each EuroLeague
IQ run was silently overwriting that landing page (wiping out every
other tool's nav card) until something else rewrote it back.

Hot Form / Real Streak: team-level scoring-form screens, same concept as
Euro Ice's Goal Streak panel but relative to a freshly-computed league
average rather than a fixed absolute number -- EuroLeague's scoring
environment isn't something this session can calibrate against live data,
so HOT_FORM_MARGIN and the Real Streak threshold are both set off
lg_scored (this run's own league-average points) instead of a guessed
constant. Hot Form = last RECENT_GAMES avg scored beats league average by
HOT_FORM_MARGIN+; Real Streak = a genuine CONSECUTIVE run of games each
beating league average, length >= REAL_STREAK_MIN_LENGTH.
"""

import os
import csv
import json
import html as html_lib
import math
import time
import requests
from datetime import datetime, timezone, timedelta

BASE = "https://api-live.euroleague.net/v2/competitions/E"
REQUEST_DELAY = 1.0          # paced, same reasoning as every other tool here --
                               # be polite to an undocumented API, especially
                               # one with no published rate limit.
RECENT_GAMES = 5
PRIOR_STRENGTH = 3
DEFAULT_TEAM_STD = 9.0        # points -- EuroLeague games run lower-scoring
                               # and lower-variance than NBA, this is a
                               # starting estimate, not a fitted value.
UPCOMING_WINDOW_DAYS = 8       # how far ahead to scan the schedule for games
                               # to predict -- EuroLeague doesn't expose a
                               # simple "this week" endpoint the way ESPN's
                               # NFL scoreboard does, so this filters the
                               # full season schedule by date instead.
PLAYER_MIN_GAMES = 2           # a player needs to have appeared in at least
                               # this many of the last RECENT_GAMES games to
                               # be considered a "featured" player worth a prop
PLAYER_POOL_SIZE = 5           # cap per team, same rough scale as Blitz IQ's
                               # one-starter-per-position (QB/RB/WR/TE = 4)
HOT_FORM_MARGIN = 6.0          # points above this run's own league-average
                               # scored (lg_scored) needed to count as
                               # "hot" over the last RECENT_GAMES games --
                               # relative, not a fixed absolute number, same
                               # reasoning as the module docstring above.
REAL_STREAK_MIN_LENGTH = 3     # shortest CONSECUTIVE run (most recent game
                               # backward, no break) of games each beating
                               # league-average scored that counts as "a
                               # streak" -- same bar Euro Ice's Real Streak
                               # panel uses.


def current_season_year():
    """EuroLeague's own JS client computes this the same way: a season
    "E2026" runs Oct 2026 - May/June 2027, so from July onward we're
    already in next season's year; before July we're still finishing the
    season that started the PREVIOUS year."""
    today = datetime.now(timezone.utc)
    return today.year if today.month >= 7 else today.year - 1


def season_code(year=None):
    return f"E{year if year is not None else current_season_year()}"


def _get(url, params=None):
    time.sleep(REQUEST_DELAY)
    try:
        r = requests.get(url, params=params, timeout=20)
    except Exception as e:
        print(f"  [!] request failed: {url} ({e})")
        return None
    if r.status_code != 200:
        print(f"  [!] {r.status_code} on {url}")
        return None
    return r


def norm_cdf(x, mean, std):
    if std is None or std <= 0:
        return 1.0 if x >= mean else 0.0
    return 0.5 * (1 + math.erf((x - mean) / (std * math.sqrt(2))))


schedule_cache = {}


def get_schedule(season):
    """Full season schedule -- every game, played and upcoming. One call
    covers the whole season, cached per season so team-form/player-prop
    lookups across every game on the slate share this single fetch
    instead of each team re-requesting it."""
    if season in schedule_cache:
        return schedule_cache[season]
    r = _get(f"{BASE}/seasons/{season}/games")
    games = []
    if r is not None:
        try:
            games = r.json().get('data', [])
        except Exception as e:
            print(f"  [!] couldn't parse schedule for season {season}: {e}")
    schedule_cache[season] = games
    return games


game_stats_cache = {}


def get_game_stats(season, game_code):
    """Per-game box score -- the rich v2 'stats' endpoint (points,
    totalRebounds, assistances, etc. per player, confirmed field names),
    not the legacy live-feed boxscore which only carries raw points.
    Cached per (season, game_code) since a team's own last-N-games lookup
    and its opponent's both end up requesting some of the same games."""
    key = (season, game_code)
    if key in game_stats_cache:
        return game_stats_cache[key]
    r = _get(f"{BASE}/seasons/{season}/games/{game_code}/stats")
    stats = None
    if r is not None:
        try:
            stats = r.json()
        except Exception as e:
            print(f"    [!] couldn't parse game stats for game {game_code}: {e}")
    game_stats_cache[key] = stats
    return stats


def _team_games(schedule, tv_code):
    """This team's PLAYED games, oldest-first by date -- same ordering
    convention recency_weighted() relies on everywhere else in this
    suite (higher weight for a later index)."""
    games = []
    for g in schedule:
        if not g.get('played'):
            continue
        local = (g.get('local') or {}).get('club') or {}
        road = (g.get('road') or {}).get('club') or {}
        if local.get('tvCode') != tv_code and road.get('tvCode') != tv_code:
            continue
        games.append(g)
    games.sort(key=lambda g: g.get('date') or '')
    return games


def recency_weighted(values):
    n = len(values)
    if n == 0:
        return None
    wts = [1.3 ** i for i in range(n)]
    return sum(w * v for w, v in zip(wts, values)) / sum(wts)


def shrink(value, n, league_avg, prior=PRIOR_STRENGTH):
    return (n * value + prior * league_avg) / (n + prior)


def _current_scoring_streak(scored_list, threshold):
    """scored_list is oldest->newest (recency_weighted()'s convention
    everywhere in this file). Walks backward from the most recent game,
    counting consecutive games that beat threshold, stopping at the first
    break -- same approach as Euro Ice's _current_goal_streak."""
    streak = 0
    for v in reversed(scored_list):
        if v > threshold:
            streak += 1
        else:
            break
    return streak


team_form_cache = {}


# Team rebounds / assists / 3PT (user request, 2026-11-16: "update like the new
# Orange Line"). Unlike the NBA/ESPN version, this API's game-stats payload has
# a real team 'total' block (points is confirmed there), so each team's actual
# box-score totals are used where the key exists -- no summing of a rotation.
# Rebounds/assists use the same field names the player rows are confirmed to
# use (totalRebounds / assistances); the 3-pointers-made key is NOT confirmed,
# so several likely names are tried and the available keys are printed ([DIAG])
# when none match, in which case 3PT is simply left out.
TEAM_STAT_SPECS = [
    ('reb', 'Rebounds', 2.0, ('totalRebounds',)),
    ('ast', 'Assists', 2.0, ('assistances',)),
    ('tpm', '3-Pointers Made', 1.5, ('fieldGoalsMade3', 'threePointersMade', 'fieldGoalsMadeThree', 'fieldGoalsMade3Points')),
]
TEAM_STAT_MIN_GAMES = 3        # need at least this many games with the stat for a projection
TEAM_STAT_STD_COEF = 0.9       # sd = coef * sqrt(mean); an UNCALIBRATED starting value
_diag_printed = set()


def team_stat_from_block(block, keys):
    """One team's box-score total for a stat, or None. Prefers the official
    'total' block; falls back to summing the player rows (needs 5+ players
    reporting, and omits team rebounds, so it can run a little low)."""
    block = block or {}
    total = block.get('total') or {}
    for k in keys:
        v = total.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    players = block.get('players') or []
    for k in keys:
        vals = []
        for p in players:
            v = (p.get('stats') or {}).get(k)
            try:
                if v is not None:
                    vals.append(float(v))
            except (TypeError, ValueError):
                pass
        if len(vals) >= 5:
            return float(sum(vals))
    return None


def _diag_missing_stat(code, keys, block):
    if code in _diag_printed:
        return
    _diag_printed.add(code)
    print(f"    [DIAG] team stat '{code}': none of {list(keys)} found in a team's total block or player rows. "
          f"Keys in total block: {sorted((block or {}).get('total', {}).keys())}")


def get_team_form(tv_code, schedule, season):
    """Last RECENT_GAMES played games' scored/allowed points, pulled from
    each game's box score 'total.points' on whichever side (local/road)
    this team was in that specific game."""
    if (season, tv_code) in team_form_cache:
        return team_form_cache[(season, tv_code)]

    recent = _team_games(schedule, tv_code)[-RECENT_GAMES:]
    scored, allowed = [], []
    stat_vals = {code: [] for code, _l, _s, _k in TEAM_STAT_SPECS}
    for g in recent:
        gc = g.get('gameCode')
        if gc is None:
            continue
        stats = get_game_stats(season, gc)
        if not stats:
            continue
        local_code = ((g.get('local') or {}).get('club') or {}).get('tvCode')
        side, opp_side = ('local', 'road') if local_code == tv_code else ('road', 'local')
        pts = (stats.get(side) or {}).get('total', {}).get('points')
        opp_pts = (stats.get(opp_side) or {}).get('total', {}).get('points')
        if pts is None or opp_pts is None:
            print(f"    [!] game {gc}: couldn't find total.points for one or both sides "
                  f"(side={side}) -- box score shape may not match what was confirmed "
                  f"against the open-source client's fixtures.")
            continue
        scored.append(float(pts))
        allowed.append(float(opp_pts))
        for code, _label, _step, keys in TEAM_STAT_SPECS:
            v = team_stat_from_block(stats.get(side), keys)
            if v is None:
                _diag_missing_stat(code, keys, stats.get(side))
            else:
                stat_vals[code].append(v)

    if not scored:
        team_form_cache[(season, tv_code)] = None
        return None

    n = len(scored)
    form = {
        'avg_scored': round(sum(scored) / n, 1),
        'avg_allowed': round(sum(allowed) / n, 1),
        'n_games': n,
        'scored_list': scored,
        'allowed_list': allowed,
        # Only kept when EVERY counted game has the stat, so the chips line up
        # with the points chips game for game.
        'stat_lists': {c: v for c, v in stat_vals.items() if len(v) == n},
    }
    team_form_cache[(season, tv_code)] = form
    return form


def league_averages(all_forms):
    scored = [f['avg_scored'] for f in all_forms if f]
    allowed = [f['avg_allowed'] for f in all_forms if f]
    lg_scored = sum(scored) / len(scored) if scored else 80.0
    lg_allowed = sum(allowed) / len(allowed) if allowed else 80.0
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


# stat_key (box score field) -> prop config. Points uses Normal (continuous,
# higher-variance, same treatment as Blitz IQ's QB Passing Yards); Rebounds/
# Assists use Poisson (lower counts, same treatment as Blitz IQ's WR/TE
# Receptions).
PLAYER_PROP_CONFIG = {
    'points':      {'label': 'Points',   'dist': 'normal',  'std': 6.0, 'prior': 11.0},
    'totalRebounds': {'label': 'Rebounds', 'dist': 'poisson', 'std': None, 'prior': 4.0},
    'assistances': {'label': 'Assists',  'dist': 'poisson', 'std': None, 'prior': 2.5},
}


def get_team_player_pool(tv_code, games, season):
    """Builds each player's recent per-stat gamelog straight from the same
    box scores team form already fetched (shared via game_stats_cache --
    no extra API calls). Returns the top PLAYER_POOL_SIZE players by
    average points among those who appeared in at least PLAYER_MIN_GAMES
    of the last RECENT_GAMES games -- a usage-based stand-in for "starters"
    since this API has no separate depth-chart endpoint."""
    per_player = {}
    for g in games:
        gc = g.get('gameCode')
        if gc is None:
            continue
        stats = get_game_stats(season, gc)
        if not stats:
            continue
        local_code = ((g.get('local') or {}).get('club') or {}).get('tvCode')
        side = 'local' if local_code == tv_code else 'road'
        team_block = stats.get(side) or {}
        for p in team_block.get('players', []):
            person = ((p.get('player') or {}).get('person')) or {}
            code = person.get('code')
            name = person.get('name')
            if not code:
                continue
            st = p.get('stats') or {}
            entry = per_player.setdefault(code, {'name': name, 'points': [], 'totalRebounds': [], 'assistances': []})
            for stat_key in PLAYER_PROP_CONFIG:
                v = st.get(stat_key)
                if v is not None:
                    try:
                        entry[stat_key].append(float(v))
                    except (TypeError, ValueError):
                        continue

    pool = [(code, e) for code, e in per_player.items() if len(e['points']) >= PLAYER_MIN_GAMES]
    pool.sort(key=lambda kv: -(sum(kv[1]['points']) / len(kv[1]['points'])))
    return pool[:PLAYER_POOL_SIZE]


def project_player_prop(stat_key, values):
    cfg = PLAYER_PROP_CONFIG[stat_key]
    if not values:
        return None
    n = len(values)
    recent = recency_weighted(values)
    shrunk = shrink(recent, n, cfg['prior'])
    return {
        'label': cfg['label'], 'dist': cfg['dist'], 'std': cfg['std'],
        'projected': round(shrunk, 1), 'n_games': n, 'recent_values': values,
        'stat_key': stat_key,
    }


def get_team_player_props(tv_code, games, season):
    pool = get_team_player_pool(tv_code, games, season)
    props = []
    for code, entry in pool:
        for stat_key in PLAYER_PROP_CONFIG:
            proj = project_player_prop(stat_key, entry.get(stat_key) or [])
            if proj:
                props.append({**proj, 'name': entry['name'], 'player_code': code})
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


def normal_prop(mean, std, factor=0.86, round_to=0.5):
    # factor is higher than Blitz IQ's 0.72 -- EuroLeague team totals run
    # much lower-variance relative to the mean than NFL yardage stats, so
    # a line that far below the projection would almost always hit and
    # tell the user nothing; 0.86 keeps a safety margin without being
    # trivial. Same reasoning applies per-stat below.
    if mean is None or std is None:
        return None
    raw_line = mean * factor
    line = math.floor(raw_line / round_to) * round_to
    if line < round_to:
        line = round_to
    prob_over = 1 - norm_cdf(line, mean, std)
    return {'line': line, 'prob': round(prob_over * 100), 'avg': round(mean, 1)}


def poisson_prop(mean, factor=0.75, round_to=0.5):
    if mean is None or mean <= 0:
        return None
    raw_line = mean * factor
    line = math.floor(raw_line / round_to) * round_to
    if line < round_to:
        line = round_to
    prob_over = 1 - poisson_cdf(math.floor(line), mean)
    return {'line': line, 'prob': round(prob_over * 100), 'avg': round(mean, 1)}


def hit_rate(values, line):
    if not values:
        return None
    hits = sum(1 for v in values if v > line)
    return {"hits": hits, "total": len(values)}


def format_history(lst):
    if not lst:
        return None
    return "/".join(str(v) for v in lst)


def get_upcoming_games(schedule, days_ahead=UPCOMING_WINDOW_DAYS):
    today = datetime.now(timezone.utc).date()
    upcoming = []
    for g in schedule:
        if g.get('played'):
            continue
        date_str = g.get('date')
        if not date_str:
            continue
        try:
            gd = datetime.fromisoformat(date_str[:10]).date()
        except ValueError:
            continue
        if today <= gd <= today + timedelta(days=days_ahead):
            upcoming.append(g)
    upcoming.sort(key=lambda g: g.get('date') or '')
    return upcoming


def build_predictions():
    season = current_season_year()
    code = season_code(season)
    schedule = get_schedule(code)
    if not schedule:
        print(f"  [!] empty schedule for season {code} -- nothing to predict")
        return [], [], [], None

    upcoming = get_upcoming_games(schedule)
    print(f"  {len(upcoming)} upcoming game(s) in the next {UPCOMING_WINDOW_DAYS} days")

    predictions = []
    all_forms = []
    team_forms_by_game = []
    for g in upcoming:
        local = (g.get('local') or {}).get('club') or {}
        road = (g.get('road') or {}).get('club') or {}
        h_code, a_code = local.get('tvCode'), road.get('tvCode')
        if not h_code or not a_code:
            continue
        h_form = get_team_form(h_code, schedule, code)
        a_form = get_team_form(a_code, schedule, code)
        if not h_form or not a_form:
            print(f"  skipping {road.get('name')} @ {local.get('name')}: missing form data "
                  f"(no played games yet this season for one side -- normal early in a new season)")
            continue
        all_forms.extend([h_form, a_form])
        team_forms_by_game.append((g, local, road, h_code, a_code, h_form, a_form))

    lg_scored, lg_allowed = league_averages(all_forms)
    # Fresh per-run threshold for Real Streak -- see HOT_FORM_MARGIN/
    # REAL_STREAK_MIN_LENGTH comments: no safe fixed number to hardcode, so
    # this is just this run's own league-average points, rounded.
    real_streak_threshold = round(lg_scored, 1)

    streak_entries = []
    real_streak_entries = []

    for g, local, road, h_code, a_code, h_form, a_form in team_forms_by_game:
        proj = predict(h_form, a_form, lg_scored, lg_allowed)
        print(f"  Player props: {road.get('name')} @ {local.get('name')}")
        home_games = _team_games(schedule, h_code)[-RECENT_GAMES:]
        away_games = _team_games(schedule, a_code)[-RECENT_GAMES:]
        home_props = get_team_player_props(h_code, home_games, code)
        away_props = get_team_player_props(a_code, away_games, code)
        predictions.append({
            'date': g.get('date', ''),
            'season': code, 'game_code': g.get('gameCode'),
            'match': f"{road.get('name')} @ {local.get('name')}",
            'home_team': local.get('name'), 'away_team': road.get('name'),
            'home_tv_code': h_code, 'away_tv_code': a_code,
            'home_form': h_form, 'away_form': a_form,
            'home_props': home_props, 'away_props': away_props,
            **proj,
        })

        for team, team_name, opp_name, is_home, form in (
            (local, local.get('name'), road.get('name'), True, h_form),
            (road, road.get('name'), local.get('name'), False, a_form),
        ):
            scored_list = form['scored_list']
            if form['avg_scored'] - lg_scored >= HOT_FORM_MARGIN:
                streak_entries.append({
                    'team': team_name, 'opponent': opp_name, 'is_home': is_home,
                    'date': g.get('date', ''), 'last5_avg': form['avg_scored'],
                    'last5_scored': scored_list, 'lg_scored': round(lg_scored, 1),
                    # Verification threshold (results tracker): did this team's
                    # ACTUAL points in the predicted match beat this run's
                    # league-average scored -- same "did the form continue"
                    # check Euro Ice's Hot Form/Real Streak panels use.
                    'threshold': real_streak_threshold,
                    'home_name': local.get('name'), 'away_name': road.get('name'),
                    'season': code, 'game_code': g.get('gameCode'),
                })

            streak_len = _current_scoring_streak(scored_list, real_streak_threshold)
            if streak_len >= REAL_STREAK_MIN_LENGTH:
                real_streak_entries.append({
                    'team': team_name, 'opponent': opp_name, 'is_home': is_home,
                    'date': g.get('date', ''), 'streak_len': streak_len,
                    'streak_games': scored_list[-streak_len:],  # already old->new
                    'full_sample': streak_len >= len(scored_list),
                    'threshold': real_streak_threshold,
                    'home_name': local.get('name'), 'away_name': road.get('name'),
                    'season': code, 'game_code': g.get('gameCode'),
                })

    # Chronological, not by exp_total -- EuroLeague round nights span several
    # calendar days within the UPCOMING_WINDOW_DAYS scan window, and sorting
    # by projected total mixed games from different days together with no
    # way to tell which was "tonight" vs "in a week" (user feedback: "it's
    # all over the place"). Date headers in make_html() group same-day games.
    predictions.sort(key=lambda x: x.get('date') or '')
    streak_entries.sort(key=lambda e: -e['last5_avg'])
    real_streak_entries.sort(key=lambda e: -e['streak_len'])
    return predictions, streak_entries, real_streak_entries, round(lg_scored, 1)


def build_legs(predictions):
    legs = []
    for p in predictions:
        match_label = p["match"]
        hf, af = p["home_form"], p["away_form"]
        game_code, season = p.get("game_code"), p.get("season")
        game_date = (p.get("date") or "")[:10]

        home_total = normal_prop(p["exp_home"], DEFAULT_TEAM_STD)
        if home_total:
            legs.append({
                "match": match_label,
                "market": f"{p['home_team']} Over {home_total['line']} Points",
                "prob": home_total["prob"], "category": "Team Total",
                "hit_rate": hit_rate(hf.get("scored_list"), home_total["line"]),
                "detail": f"proj {home_total['avg']} pts ({hf['n_games']}gm)",
                "history": format_history(hf.get("scored_list")),
                "game_code": game_code, "season": season, "game_date": game_date,
                "is_home": True, "line": home_total["line"],
            })
        away_total = normal_prop(p["exp_away"], DEFAULT_TEAM_STD)
        if away_total:
            legs.append({
                "match": match_label,
                "market": f"{p['away_team']} Over {away_total['line']} Points",
                "prob": away_total["prob"], "category": "Team Total",
                "hit_rate": hit_rate(af.get("scored_list"), away_total["line"]),
                "detail": f"proj {away_total['avg']} pts ({af['n_games']}gm)",
                "history": format_history(af.get("scored_list")),
                "game_code": game_code, "season": season, "game_date": game_date,
                "is_home": False, "line": away_total["line"],
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
                "game_code": game_code, "season": season, "game_date": game_date,
                "line": game_total["line"],
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
                    "game_code": game_code, "season": season, "game_date": game_date,
                    "line": result["line"], "player_code": prop.get("player_code"),
                    "stat_key": prop.get("stat_key"),
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
      price. Team/game totals and Points use a Normal-distribution
      projection; Rebounds/Assists use Poisson. All lines are set
      automatically below the model's projection for a safety margin.
    </div>
  `;
}}
</script>"""


STREAK_ENTRY_TEMPLATE = """<div style="background:#161d27;border-radius:8px;padding:10px 12px;margin:8px 0;display:flex;gap:10px;align-items:flex-start">
  <div style="min-width:56px;text-align:center;background:#0f1318;border:1px solid #2a3038;border-radius:8px;padding:6px 4px;flex-shrink:0">
    <div style="font-size:9px;color:#888">L5 AVG</div>
    <div style="font-size:17px;font-weight:bold;color:#22c55e">{last5_avg}</div>
  </div>
  <div style="flex:1;min-width:0">
    <div style="font-size:14px;font-weight:bold;margin:1px 0 4px">{team} <span style="color:#888;font-weight:normal;font-size:11px">({home_away})</span> vs {opponent}</div>
    <div style="font-size:10px;color:#cbd5e1">last {n} (old→new): {last5_str} · league avg {lg_scored}</div>
  </div>
</div>"""

REAL_STREAK_ENTRY_TEMPLATE = """<div style="background:#161d27;border-radius:8px;padding:10px 12px;margin:8px 0;display:flex;gap:10px;align-items:flex-start">
  <div style="min-width:56px;text-align:center;background:#0f1318;border:1px solid #f59e0b;border-radius:8px;padding:6px 4px;flex-shrink:0">
    <div style="font-size:9px;color:#888">STREAK</div>
    <div style="font-size:17px;font-weight:bold;color:#f59e0b">{streak_len}{plus}</div>
  </div>
  <div style="flex:1;min-width:0">
    <div style="font-size:14px;font-weight:bold;margin:1px 0 4px">{team} <span style="color:#888;font-weight:normal;font-size:11px">({home_away})</span> vs {opponent}</div>
    <div style="font-size:10px;color:#cbd5e1">{streak_len} straight above league avg ({threshold}+) (old→new): {streak_str}</div>
  </div>
</div>"""

STREAK_PANEL_TEMPLATE = """<div style="background:#1a1f26;border-radius:12px;padding:16px;margin:12px 0;border:1px solid #2a3038">
  <div style="font-size:15px;font-weight:800;margin-bottom:10px">🔥 Hot Form &amp; Streaks</div>
  <div style="font-size:11px;color:#888;margin-bottom:10px">
    Raw recent-FORM screens, not probabilistic predictions like the Team/Game Totals above.
    Both are measured against this run's own league-average scoring (currently {lg_scored} pts) --
    Hot Form and Real Streak measure genuinely different things, a team can appear in one, both,
    or neither.
  </div>
  <div style="font-size:12px;font-weight:700;margin:8px 0 2px">📊 Hot Form <span style="color:#888;font-weight:normal;font-size:10px">(last {n} avg ≥{margin}+ above league average)</span></div>
  {hot_form_entries}
  <div style="font-size:12px;font-weight:700;margin:14px 0 2px">🔥 Real Streak <span style="color:#888;font-weight:normal;font-size:10px">(≥{min_streak}+ CONSECUTIVE games above league average, no break)</span></div>
  {streak_entries}
</div>"""


def render_streak_panel(hot_form_entries, real_streak_entries, lg_scored):
    if not hot_form_entries and not real_streak_entries:
        return ""
    hot_form_html = "".join(
        STREAK_ENTRY_TEMPLATE.format(
            last5_avg=e["last5_avg"], team=e["team"],
            home_away="Home" if e["is_home"] else "Away", opponent=e["opponent"],
            n=len(e["last5_scored"]),
            last5_str="/".join(str(v) for v in e["last5_scored"]),  # already oldest-first
            lg_scored=e["lg_scored"],
        )
        for e in hot_form_entries
    ) or '<p style="color:#888;font-size:11px">None currently.</p>'

    streak_html = "".join(
        REAL_STREAK_ENTRY_TEMPLATE.format(
            streak_len=e["streak_len"], plus="+" if e["full_sample"] else "",
            team=e["team"], home_away="Home" if e["is_home"] else "Away",
            opponent=e["opponent"], threshold=lg_scored,
            streak_str="/".join(str(v) for v in e["streak_games"]),
        )
        for e in real_streak_entries
    ) or '<p style="color:#888;font-size:11px">None currently.</p>'

    return STREAK_PANEL_TEMPLATE.format(
        lg_scored=lg_scored, n=RECENT_GAMES, margin=HOT_FORM_MARGIN,
        min_streak=REAL_STREAK_MIN_LENGTH,
        hot_form_entries=hot_form_html, streak_entries=streak_html,
    )


HTML_TEMPLATE = """<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>EuroLeague IQ</title></head>
<body style="background:#0b0f14;color:white;font-family:Arial;padding:12px;max-width:600px;margin:auto">
<h2 style="text-align:center">🏀 EUROLEAGUE IQ</h2>
<p style="text-align:center;color:#888;font-size:11px">Recency-weighted scoring/allowed rates, Normal-distribution projected · {generated}</p>
<p style="text-align:center;margin-bottom:16px"><a href="euroleague_iq_predictions.csv" download style="background:#222;border:1px solid #444;color:white;padding:8px 14px;border-radius:8px;text-decoration:none;font-size:13px">⬇ Download CSV</a></p>
<p style="text-align:center;margin-bottom:16px">
  <a href="results/index.html" style="color:#ffeb3b;text-decoration:none;font-size:12px">📊 Results Tracker</a>
</p>
{builder}
{streak_panel}
{volume_panel}
{cards}
<p style="text-align:center;color:#666;font-size:10px;margin-top:20px">Enter your book's Over/Under line and odds to compute an edge the same way as the other tools in this suite — this page shows the model's own projection only. Player props are built from whichever players actually featured in each team's last {recent_games} games (no fixed "starters" list), so a new or returning player may take a run or two to show up.</p>
</body></html>"""

CARD_TEMPLATE = """<div class="gamecard" data-volume="{volume_level}" data-home="{home_attr}" data-away="{away_attr}" style="background:#1a1f26;border-radius:12px;padding:16px;margin:12px 0;border:1px solid #2a3038">
  <div style="font-size:11px;color:#999;margin-bottom:4px">{date}</div>{volume_badge}
  <div style="font-size:17px;font-weight:bold;margin-bottom:10px">{match}</div>
  <div style="display:flex;justify-content:space-between;text-align:center">
    <div><div style="color:#aaa;font-size:11px">{away_team}</div><div style="color:#ffeb3b;font-size:20px;font-weight:bold">{exp_away}</div></div>
    <div><div style="color:#aaa;font-size:11px">TOTAL</div><div style="color:#7ec8ff;font-size:22px;font-weight:bold">{exp_total}</div></div>
    <div><div style="color:#aaa;font-size:11px">{home_team}</div><div style="color:#ffeb3b;font-size:20px;font-weight:bold">{exp_home}</div></div>
  </div>
  <div style="background:#0f1318;border-radius:8px;padding:8px;margin-top:10px;display:flex;justify-content:space-between;font-size:11px">
    <div>{away_team}: {away_scored} scored/gm • {away_allowed} allowed/gm ({away_n}gm)</div>
  </div>
  <div style="background:#0f1318;border-radius:8px;padding:8px;margin-top:6px;font-size:11px">
    {home_team}: {home_scored} scored/gm • {home_allowed} allowed/gm ({home_n}gm)
  </div>
  {recent_html}{team_stats_html}
  {player_props_html}
</div>"""

PLAYER_PROP_ROW = """<div style="display:flex;justify-content:space-between;font-size:11px;padding:5px 0;border-top:1px solid #232a33">
  <div>{name} — {label}</div>
  <div style="color:#c792ea;font-weight:bold">{projected} <span style="color:#666;font-weight:normal">({n_games}gm)</span></div>
</div>"""


# --- Volume teams (user request, 2026-11-05; same idea as Orange Line) -----
# "Pick just a few teams with more points, rebounds and assists volume per
# game so I concentrate on their games only." Built ONLY from data this
# script already fetches (no extra API calls): a team's PTS/REB/AST are the
# sums of its featured players' projections (the same top-PLAYER_POOL_SIZE
# scorers the cards show, so REB/AST are those scorers' numbers, not a full
# roster), and pace is the scoring environment (avg scored + allowed) vs
# the average of the teams in the listed games -- there is no true
# possessions count here. Display only: no projection is changed by it.
VOLUME_TOP_PLAYERS = PLAYER_POOL_SIZE
VOLUME_MIN_TEAMS = 4
VOLUME_TOP_SHARE = 0.4    # top 40% of the listed teams, min 2, max 8


def _team_volume(props, form):
    by_stat = {'points': {}, 'totalRebounds': {}, 'assistances': {}}
    for p in props or []:
        if p.get('stat_key') in by_stat:
            by_stat[p['stat_key']][p.get('player_code') or p.get('name')] = p['projected']
    common = set(by_stat['points']) & set(by_stat['totalRebounds']) & set(by_stat['assistances'])
    if len(common) < 3 or not form:
        return None
    return {
        'pts': sum(by_stat['points'][c] for c in common),
        'reb': sum(by_stat['totalRebounds'][c] for c in common),
        'ast': sum(by_stat['assistances'][c] for c in common),
        'env': form['avg_scored'] + form['avg_allowed'],
    }


def build_volume_teams(predictions):
    """Returns (ranked_teams, game_tags) -- see Orange Line's version.
    game_tags: {game_code: (level, text)}, level 2 = both teams top-volume,
    1 = one team."""
    teams = {}
    for p in predictions:
        for side, props_key, form_key in (('home_team', 'home_props', 'home_form'),
                                          ('away_team', 'away_props', 'away_form')):
            v = _team_volume(p.get(props_key), p.get(form_key))
            if v:
                v.update(team=p[side], league='EuroLeague')
                teams[p[side]] = v
    rows = list(teams.values())
    ranked = []
    if len(rows) >= VOLUME_MIN_TEAMS:
        for key in ('pts', 'reb', 'ast', 'env'):
            vals = [r[key] for r in rows]
            mean = sum(vals) / len(vals)
            sd = (sum((x - mean) ** 2 for x in vals) / len(vals)) ** 0.5 or 1.0
            for r in rows:
                r['z_' + key] = (r[key] - mean) / sd
        mean_env = sum(r['env'] for r in rows) / len(rows)
        for r in rows:
            r['env_pct'] = (r['env'] / mean_env - 1) * 100 if mean_env else 0.0
            r['score'] = sum(r['z_' + k] for k in ('pts', 'reb', 'ast', 'env')) / 4
        rows.sort(key=lambda r: r['score'], reverse=True)
        k = min(8, max(2, round(len(rows) * VOLUME_TOP_SHARE)))
        for i, r in enumerate(rows):
            r['rank'] = i + 1
            r['top'] = i < k
        ranked = rows
    top_names = {r['team'] for r in ranked if r['top']}
    known = {r['team'] for r in ranked}
    tags = {}
    for p in predictions:
        if p['home_team'] not in known and p['away_team'] not in known:
            continue
        hot = [t for t in (p['home_team'], p['away_team']) if t in top_names]
        if len(hot) == 2:
            tags[p.get('game_code')] = (2, "Volume game · both teams top-volume")
        elif len(hot) == 1:
            tags[p.get('game_code')] = (1, f"Volume side · {hot[0]}")
    return ranked, tags


VOLUME_PANEL_TEMPLATE = """<div id="volPanel" data-k='{k_json}' style="background:#1a1f26;border-radius:12px;padding:16px;margin:12px 0;border:1px solid #2a3038">
  <div style="font-weight:bold;color:#7ec8ff;margin-bottom:2px">📈 Volume teams</div>
  <div style="font-size:11px;color:#998;margin-bottom:8px">Teams in the games listed below, ranked among themselves. Pick what to rank by; the 🔥 marks the top {share}% on that measure. PTS / REB / AST are the projected totals for each team's top {players} scorers; pace is the scoring environment vs the slate average.</div>
  <div style="font-size:11px;color:#998;margin-bottom:4px">Rank by:</div>
  <div id="volSort" style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px">
    <button type="button" class="vs" data-key="score">Overall</button>
    <button type="button" class="vs" data-key="pts">Points</button>
    <button type="button" class="vs" data-key="reb">Rebounds</button>
    <button type="button" class="vs" data-key="ast">Assists</button>
    <button type="button" class="vs" data-key="pace">Pace</button>
  </div>
  <div id="volRows">{rows}</div>
  <div style="font-size:11px;color:#998;margin:10px 0 6px">Show games:</div>
  <div id="volFilter" style="display:flex;gap:6px;flex-wrap:wrap">
    <button type="button" class="vf" data-min="0">All</button>
    <button type="button" class="vf" data-min="1">Any volume team</button>
    <button type="button" class="vf" data-min="2">Both volume teams</button>
  </div>
  <div style="font-size:10px;color:#776;margin-top:8px;line-height:1.5">Bookmakers already raise lines in high-volume games, so this narrows where to look; it does not on its own find value. Only the game cards below are filtered. Game badges follow the measure you rank by.</div>
</div>
<script>
(function(){
  var panel = document.getElementById('volPanel'); if (!panel) return;
  var K = JSON.parse(panel.getAttribute('data-k'));
  var sortKey = 'score', minLevel = 0;
  var rowsBox = document.getElementById('volRows');
  function pill(b, on){
    b.style.background = on ? '#7ec8ff' : '#222'; b.style.color = on ? '#0b0f14' : '#e6eaf0';
    b.style.fontWeight = on ? 'bold' : 'normal'; b.style.border = on ? '0' : '1px solid #444';
    b.style.borderRadius = '999px'; b.style.padding = '6px 12px'; b.style.fontSize = '12px';
  }
  var STATKEY = {score:null, pts:'pts', reb:'reb', ast:'ast', pace:'pace'};
  function render(){
    var rows = Array.prototype.slice.call(rowsBox.querySelectorAll('.vrow'));
    var leagues = {};
    rows.forEach(function(r){ (leagues[r.getAttribute('data-league')] = leagues[r.getAttribute('data-league')] || []).push(r); });
    var top = {}, ordered = [];
    Object.keys(leagues).forEach(function(lg){
      var g = leagues[lg].sort(function(a, b){ return parseFloat(b.getAttribute('data-' + sortKey)) - parseFloat(a.getAttribute('data-' + sortKey)); });
      g.forEach(function(r, i){
        var isTop = i < (K[lg] || 0);
        r.querySelector('.vrank').textContent = (i + 1) + '.';
        r.querySelector('.vflame').style.display = isTop ? '' : 'none';
        r.style.background = isTop ? '#1f2a38' : 'transparent'; r.style.color = isTop ? '#e6eaf0' : '#998';
        if (isTop) top[r.getAttribute('data-team')] = true;
        r.querySelectorAll('.vstat').forEach(function(sp){
          sp.style.color = (STATKEY[sortKey] === sp.getAttribute('data-k')) ? '#7ec8ff' : ''; });
        ordered.push(r);
      });
    });
    ordered.forEach(function(r){ rowsBox.appendChild(r); });
    document.querySelectorAll('.gamecard').forEach(function(c){
      var h = c.getAttribute('data-home'), a = c.getAttribute('data-away');
      var level = (top[h] ? 1 : 0) + (top[a] ? 1 : 0);
      c.setAttribute('data-volume', level);
      var badge = c.querySelector('.vbadge');
      if (badge) {
        badge.style.display = level ? 'inline-block' : 'none';
        badge.textContent = level === 2 ? '🔥 Volume game · both teams top-volume'
          : level === 1 ? '🔥 Volume side · ' + (top[h] ? h : a) : '';
      }
      c.style.display = level >= minLevel ? '' : 'none';
    });
    // hide a date header when none of its games are showing
    var hdr = null, any = false;
    function flush(){ if (hdr) hdr.style.display = any ? '' : 'none'; }
    Array.prototype.slice.call(document.querySelectorAll('.vdate, .gamecard')).forEach(function(el){
      if (el.classList.contains('vdate')) { flush(); hdr = el; any = false; }
      else if (el.style.display !== 'none') any = true;
    });
    flush();
  }
  document.getElementById('volSort').addEventListener('click', function(e){
    var b = e.target.closest('.vs'); if (!b) return;
    sortKey = b.getAttribute('data-key');
    document.querySelectorAll('#volSort .vs').forEach(function(x){ pill(x, x === b); });
    render();
  });
  document.getElementById('volFilter').addEventListener('click', function(e){
    var b = e.target.closest('.vf'); if (!b) return;
    minLevel = parseInt(b.getAttribute('data-min'), 10);
    document.querySelectorAll('#volFilter .vf').forEach(function(x){ pill(x, x === b); });
    render();
  });
  document.querySelectorAll('#volSort .vs').forEach(function(x){ pill(x, x.getAttribute('data-key') === 'score'); });
  document.querySelectorAll('#volFilter .vf').forEach(function(x){ pill(x, x.getAttribute('data-min') === '0'); });
  render();
})();
</script>"""


def render_volume_panel(ranked):
    if not ranked:
        return ""
    multi = len({x['league'] for x in ranked}) > 1
    k_by_league = {}
    for r in ranked:
        k_by_league[r['league']] = k_by_league.get(r['league'], 0) + (1 if r['top'] else 0)
    rows = ""
    for r in ranked:
        hi = r['top']
        flame_style = '' if hi else ' style="display:none"'
        lg = ' <span style="color:#776">' + html_lib.escape(r['league']) + '</span>' if multi else ''
        rows += (f'<div class="vrow" data-team="{html_lib.escape(r["team"], quote=True)}" data-league="{html_lib.escape(r["league"], quote=True)}" '
                 f'data-score="{r["score"]:.4f}" data-pts="{r["pts"]:.2f}" data-reb="{r["reb"]:.2f}" '
                 f'data-ast="{r["ast"]:.2f}" data-pace="{r["env_pct"]:.2f}" '
                 f'style="display:flex;justify-content:space-between;gap:8px;padding:5px 8px;margin:2px 0;border-radius:6px;font-size:12px;'
                 f'background:{"#1f2a38" if hi else "transparent"};color:{"#e6eaf0" if hi else "#998"}">'
                 f'<span><b class="vrank">{r["rank"]}.</b> {html_lib.escape(r["team"])}<span class="vflame"{flame_style}> 🔥</span>{lg}</span>'
                 f'<span style="white-space:nowrap"><span class="vstat" data-k="pts">{r["pts"]:.0f} pts</span> · '
                 f'<span class="vstat" data-k="reb">{r["reb"]:.0f} reb</span> · '
                 f'<span class="vstat" data-k="ast">{r["ast"]:.0f} ast</span> · '
                 f'<span class="vstat" data-k="pace">{r["env_pct"]:+.1f}% pace</span></span></div>')
    # str.replace, not .format: the template carries JS braces.
    return (VOLUME_PANEL_TEMPLATE.replace("{players}", str(VOLUME_TOP_PLAYERS))
            .replace("{share}", str(round(VOLUME_TOP_SHARE * 100)))
            .replace("{k_json}", html_lib.escape(json.dumps(k_by_league), quote=True).replace("&quot;", '"'))
            .replace("{rows}", rows))


def _stat_sd(mean):
    return TEAM_STAT_STD_COEF * math.sqrt(max(mean, 0.0))


def team_stat_mean(form, code):
    """Recency-weighted mean of the team's own box-score totals, or None."""
    vals = ((form or {}).get('stat_lists') or {}).get(code)
    if not vals or len(vals) < TEAM_STAT_MIN_GAMES:
        return None
    return recency_weighted(vals)


def stat_1x2(mu_home, mu_away):
    """(P home more, P tie, P away more) for integer-valued team totals;
    normal approximation with a continuity correction (tie = P(|D| < 0.5))."""
    sd = math.sqrt(_stat_sd(mu_home) ** 2 + _stat_sd(mu_away) ** 2)
    mu = mu_home - mu_away
    p_away = norm_cdf(-0.5, mu, sd)
    p_home = 1.0 - norm_cdf(0.5, mu, sd)
    return p_home, max(0.0, 1.0 - p_home - p_away), p_away


def stat_over_ladder(mu_home, mu_away, step, n_lines=3):
    """Combined over/under at n_lines half-point lines around the projection."""
    mu = mu_home + mu_away
    sd = math.sqrt(_stat_sd(mu_home) ** 2 + _stat_sd(mu_away) ** 2)
    main = math.floor(mu) + 0.5
    out = []
    for i in range(-(n_lines // 2), n_lines // 2 + 1):
        line = main + i * step
        if line < 0.5:
            continue
        over = 1.0 - norm_cdf(line, mu, sd)
        out.append((line, over, 1.0 - over))
    return mu, out


def _half(x):
    return round(x * 2) / 2


def _fmt_num(v):
    return str(int(v)) if float(v).is_integer() else f"{v:.1f}"


def _over_under_run(values, line):
    kind, n = None, 0
    for v in reversed(values):
        k = 'over' if v > line else ('under' if v < line else None)
        if k is None or (kind is not None and k != kind):
            break
        kind, n = k, n + 1
    return kind, n


def _chips_row(label, values, line):
    if not values:
        return ""
    chips = ""
    for v in values:
        bg, fg = ("#16321f", "#7ee09a") if v > line else (("#3a1f1f", "#f08a8a") if v < line else ("#2a2f36", "#aaa"))
        chips += (f'<span style="display:inline-block;min-width:24px;text-align:center;padding:1px 4px;margin:1px;'
                  f'border-radius:4px;background:{bg};color:{fg}">{_fmt_num(v)}</span>')
    overs = sum(1 for v in values if v > line)
    kind, n = _over_under_run(values, line)
    run = f" · <b style=\"color:{'#7ee09a' if kind == 'over' else '#f08a8a'}\">{n} straight {kind}</b>" if kind and n >= 2 else ""
    return (f'<div style="margin-bottom:6px"><div style="color:#dde">{label} <span style="color:#778">vs {_fmt_num(line)}</span></div>'
            f'<div>{chips}</div>'
            f'<div style="color:#99a">avg {sum(values) / len(values):.1f} · over {overs}/{len(values)}{run}</div></div>')


def recent_totals_html(p):
    """Last-games chips for each team's points and the game total, vs tonight's projection."""
    hf, af = p.get('home_form'), p.get('away_form')
    if not hf or not af or p.get('exp_total') is None:
        return ""
    def game_totals(form):
        return [a + b for a, b in zip(form.get('scored_list') or [], form.get('allowed_list') or [])]
    game_line = _half(p['exp_total'])
    n = max(len(hf.get('scored_list') or []), len(af.get('scored_list') or []))
    rows = (_chips_row(f"{p['away_team']} points", af.get('scored_list'), _half(p['exp_away']))
            + _chips_row(f"{p['home_team']} points", hf.get('scored_list'), _half(p['exp_home']))
            + _chips_row(f"{p['away_team']} game totals", game_totals(af), game_line)
            + _chips_row(f"{p['home_team']} game totals", game_totals(hf), game_line))
    if not rows:
        return ""
    return (f'<div style="background:#0f1318;border-radius:8px;padding:8px 10px;margin-top:6px;font-size:11px">'
            f'<div style="color:#778;text-transform:uppercase;font-size:9px;margin-bottom:5px">Last {n} games · oldest → latest · green = over tonight\'s projection</div>'
            f'{rows}</div>')


def team_stats_html(p):
    away_team, home_team = p['away_team'], p['home_team']
    blocks = ""
    for code, label, step, _keys in TEAM_STAT_SPECS:
        mh = team_stat_mean(p.get('home_form'), code)
        ma = team_stat_mean(p.get('away_form'), code)
        if mh is None or ma is None:
            continue
        ph, pt, pa = stat_1x2(mh, ma)
        best = max((ph, 'h'), (pt, 't'), (pa, 'a'))[1]
        def cell(txt, val, tag):
            hi = tag == best
            return (f'<div style="flex:1;text-align:center;padding:4px 2px;border-radius:6px;'
                    f'background:{"#1c2a3a" if hi else "#151a21"};border:1px solid {"#36506e" if hi else "#232a33"}">'
                    f'<div style="color:#99a;font-size:10px">{txt}</div>'
                    f'<div style="color:{"#ffeb3b" if hi else "#dde"};font-weight:bold">{val * 100:.0f}%</div></div>')
        mu, ladder = stat_over_ladder(mh, ma, step)
        lad = "".join(
            f'<div style="flex:1;text-align:center;padding:4px 2px;border-radius:6px;background:#151a21;border:1px solid #232a33">'
            f'<div style="color:#99a;font-size:10px">{line:g}</div>'
            f'<div style="color:#7ee09a">O {over * 100:.0f}%</div>'
            f'<div style="color:#f08a8a">U {under * 100:.0f}%</div></div>'
            for line, over, under in ladder)
        hist_a = p['away_form']['stat_lists'][code]
        hist_h = p['home_form']['stat_lists'][code]
        recent = ('<div style="color:#778;font-size:9px;text-transform:uppercase;margin:6px 0 3px">'
                  "Last games · official team totals · oldest → latest · green = over this team's projection</div>"
                  + _chips_row(f"{away_team} {label.lower()}", hist_a, _half(ma))
                  + _chips_row(f"{home_team} {label.lower()}", hist_h, _half(mh)))
        blocks += (f'<div style="margin-bottom:10px"><div style="color:#dde;margin-bottom:4px"><b>{label}</b> '
                   f'<span style="color:#778">{away_team} {ma:.1f} · {home_team} {mh:.1f} · total {mu:.1f}</span></div>'
                   f'<div style="display:flex;gap:4px;margin-bottom:4px">{cell(away_team + " more", pa, "a")}{cell("Tie", pt, "t")}{cell(home_team + " more", ph, "h")}</div>'
                   f'<div style="display:flex;gap:4px">{lad}</div>{recent}</div>')
    if not blocks:
        return ""
    return ('<div style="background:#0f1318;border-radius:8px;padding:8px 10px;margin-top:6px;font-size:11px">'
            '<div style="color:#778;text-transform:uppercase;font-size:9px;margin-bottom:6px">Team rebounds · assists · 3PT — which team gets more, and combined over/under (model only)</div>'
            f'{blocks}'
            '<div style="color:#778;font-size:10px;line-height:1.5">Each projection is the team\'s own recency-weighted average of its last '
            f'games\' box-score totals (needs {TEAM_STAT_MIN_GAMES}+ games). It ignores the opponent, injuries and lineup news, and the spread (sd = {TEAM_STAT_STD_COEF} x sqrt(mean)) is an uncalibrated '
            'estimate. There is no bookmaker line: the lines sit around the model\'s own projection, so check them against your bookmaker\'s real line. '
            'The Results tracker records these picks so the 50% lines can be checked against what actually happens.</div></div>')


def build_team_stat_entries(predictions):
    """Tracker entries for CALIBRATION: combined over/under at the middle line
    (value = model P(over)) and the 1X2 top pick (value = model P(pick))."""
    out = []
    for p in predictions:
        if p.get('game_code') is None:
            continue
        date_key = (p.get('date') or '')[:10]
        for code, label, step, _keys in TEAM_STAT_SPECS:
            mh = team_stat_mean(p.get('home_form'), code)
            ma = team_stat_mean(p.get('away_form'), code)
            if mh is None or ma is None:
                continue
            base = {'match': p['match'], 'season': p['season'], 'game_code': p['game_code'],
                    'game_date': date_key, 'stat_key': code}
            mu, ladder = stat_over_ladder(mh, ma, step)
            mid = min(ladder, key=lambda r: abs(r[0] - mu)) if ladder else None
            if mid:
                out.append(dict(base, scanner='team_stat_ou', market=f'Team {label} Over {mid[0]:g} (combined)',
                                prob=mid[1], line=mid[0]))
            ph, pt, pa = stat_1x2(mh, ma)
            pick, prob = max(('home', ph), ('tie', pt), ('away', pa), key=lambda t: t[1])
            team = {'home': p['home_team'], 'away': p['away_team'], 'tie': 'Tie'}[pick]
            out.append(dict(base, scanner='team_stat_1x2', market=f'Team {label} 1X2: {team}',
                            prob=prob, line=None, pick=pick))
    return out


def player_props_section(team_label, props):
    if not props:
        return ""
    rows = "".join(PLAYER_PROP_ROW.format(**p) for p in props)
    return f'<div style="margin-top:8px"><div style="color:#888;font-size:10px;text-transform:uppercase;margin-bottom:2px">{team_label} Player Props</div>{rows}</div>'


def make_html(predictions, streak_entries=None, real_streak_entries=None, lg_scored=None):
    # Group into same-day sections with a date header, rather than one flat
    # list. build_predictions() already sorts chronologically, but sort
    # again here defensively so this function produces a correctly grouped
    # page even if it's ever called with an unsorted list.
    predictions = sorted(predictions, key=lambda x: x.get('date') or '')

    volume_ranked, volume_tags = build_volume_teams(predictions)

    def _vol(p):
        level, text = volume_tags.get(p.get('game_code'), (0, ''))
        badge = (f'<div class="vbadge" style="display:{"inline-block" if level else "none"};font-size:11px;color:#7ec8ff;'
                 f'background:#1f2a38;border:1px solid #36506e;border-radius:999px;padding:2px 8px;margin-bottom:6px">'
                 f'{"🔥 " + text if level else ""}</div>')
        return level, badge

    cards = ""
    last_date_key = None
    for p in predictions:
        date_key = (p.get('date') or '')[:10]
        if date_key != last_date_key:
            try:
                label = datetime.fromisoformat(date_key).strftime('%a %d %b') if date_key else "Date TBC"
            except ValueError:
                label = date_key or "Date TBC"
            cards += f'<div class="vdate" style="color:#7ec8ff;font-size:12px;font-weight:bold;margin:18px 0 8px;padding-bottom:4px;border-bottom:1px solid #2a3038">{label}</div>'
            last_date_key = date_key
        cards += CARD_TEMPLATE.format(
            volume_level=_vol(p)[0], volume_badge=_vol(p)[1],
            recent_html=recent_totals_html(p), team_stats_html=team_stats_html(p),
            home_attr=html_lib.escape(p['home_team'], quote=True), away_attr=html_lib.escape(p['away_team'], quote=True),
            date=p['date'][:16].replace('T', ' '), match=p['match'],
            away_team=p['away_team'], home_team=p['home_team'],
            exp_away=p['exp_away'], exp_home=p['exp_home'], exp_total=p['exp_total'],
            away_scored=p['away_form']['avg_scored'], away_allowed=p['away_form']['avg_allowed'],
            away_n=p['away_form']['n_games'],
            home_scored=p['home_form']['avg_scored'], home_allowed=p['home_form']['avg_allowed'],
            home_n=p['home_form']['n_games'],
            player_props_html=(
                player_props_section(p['away_team'], p.get('away_props', []))
                + player_props_section(p['home_team'], p.get('home_props', []))
            ),
        )
    if not cards:
        cards = '<p style="text-align:center;color:#666">No upcoming games with enough form data to project right now.</p>'

    legs = build_legs(predictions)
    builder = BUILDER_TEMPLATE.format(legs_json=json.dumps(legs)) if legs else ""

    streak_panel = render_streak_panel(
        streak_entries or [], real_streak_entries or [],
        lg_scored if lg_scored is not None else "—",
    )

    return HTML_TEMPLATE.format(
        generated=datetime.now().strftime("%d %b %H:%M"),
        builder=builder, streak_panel=streak_panel, volume_panel=render_volume_panel(volume_ranked),
        cards=cards, recent_games=RECENT_GAMES,
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
            'Date', 'Match', 'Team', 'Player', 'PropType',
            'Projected', 'SampleSize', 'RecentValues',
            'Line', 'OverOdds', 'UnderOdds', 'ActualValue', 'HitOrMiss',
        ])
        for p in predictions:
            for team_label, props in [(p['away_team'], p.get('away_props', [])),
                                       (p['home_team'], p.get('home_props', []))]:
                for prop in props:
                    writer.writerow([
                        p['date'], p['match'], team_label, prop['name'], prop['label'],
                        prop['projected'], prop['n_games'], '; '.join(str(v) for v in prop['recent_values']),
                        '', '', '', '', '',
                    ])


if __name__ == "__main__":
    predictions, streak_entries, real_streak_entries, lg_scored = build_predictions()
    os.makedirs('docs/euroleague-iq', exist_ok=True)
    with open('docs/euroleague-iq/index.html', 'w') as f:
        f.write(make_html(predictions, streak_entries, real_streak_entries, lg_scored))
    write_csv(predictions, 'docs/euroleague_iq_predictions.csv')
    write_player_props_csv(predictions, 'docs/euroleague_iq_player_props.csv')
    with open('docs/euroleague_iq.json', 'w') as f:
        json.dump(predictions, f, indent=2, default=str)

    try:
        import euroleague_iq_results_tracker
        euroleague_iq_results_tracker.run_results_tracker(
            build_legs(predictions), streak_entries, real_streak_entries, lg_scored,
            team_stat_entries=build_team_stat_entries(predictions),
        )
    except Exception as e:
        print(f"[!] Results tracker failed, but the rest of this run succeeded: {e}")
