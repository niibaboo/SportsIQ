#!/usr/bin/env python3
import math, requests, sys, json
from datetime import datetime, timezone
BASE="https://api-web.nhle.com/v1"
H={"User-Agent":"Mozilla/5.0"}
CLUB_CACHE={} # memoize club-stats
SCHEDULE_CACHE={} # memoize club-schedule-season (for Hot Form/Real Streak)
GAMELOG_CACHE={} # memoize player/{id}/game-log/now (for skater SOG Hot Form/Real Streak)

# Hot Form / Real Streak thresholds -- NHL teams average roughly 3.0-3.3
# goals/game league-wide (lower than Euro Ice's international-hockey pool,
# which is why that file's bars of 3.5 avg / 4-per-game are tightened here
# to 3.0/3). Starting point only, same as every other tool's thresholds in
# this suite -- watch real results and raise/lower once there's evidence
# (see Under IQ's TEAM_UNDER15_MIN, which got the same treatment).
HOT_FORM_MIN = 3.0          # avg goals over last N games to count as "hot"
HOT_FORM_MIN_GAMES = 5
REAL_STREAK_THRESHOLD = 3   # goals needed in a game to extend the streak
REAL_STREAK_MIN_LENGTH = 3  # min consecutive games to show a streak

# Same Hot Form/Real Streak treatment, applied to skater Shots on Goal
# instead of team goals -- scoped (by user choice) to just the 5 skaters
# per team already shown in the Props section, not the whole roster, so
# this doesn't add a per-player API call for every name on a club. Starting
# thresholds only, same caveat as above -- watch real results and adjust.
SOG_HOT_FORM_MIN = 3.0          # avg shots-on-goal over last N games
SOG_HOT_FORM_MIN_GAMES = 5
SOG_REAL_STREAK_THRESHOLD = 2   # SOG needed in a game to extend the streak
SOG_REAL_STREAK_MIN_LENGTH = 3  # min consecutive games to show a streak

# Goalie Saves lines -- fixed thresholds rather than a computed line (same
# approach as skater SOG's O1.5/O2.5 above), picked to bracket a typical
# starter's workload (bet365's own Saves milestone market runs whole
# numbers around 25/30/35 for a busy night -- these sit either side of
# a normal ~28-save game).
GOALIE_SAVES_LINE_LOW = 24.5
GOALIE_SAVES_LINE_HIGH = 29.5

# Same Hot Form/Real Streak treatment again, this time for the starting
# goalie's saves -- only the 1 goalie shown per team (get_goalie_props
# already narrows to the top-projected starter), so one extra API call
# per team, not per roster slot. Starting thresholds only, same caveat.
SAVES_HOT_FORM_MIN = 28.0         # avg saves over last N games to count as "hot"
SAVES_HOT_FORM_MIN_GAMES = 5
SAVES_REAL_STREAK_THRESHOLD = 25  # saves needed in a game to extend the streak
SAVES_REAL_STREAK_MIN_LENGTH = 3  # min consecutive games to show a streak

def fetch(u):
  return requests.get(u, headers=H, timeout=20).json()

def pois(k,lam):
  return math.exp(-lam)*lam**k/math.factorial(k)

def over_prob(line,lam):
  thr=math.floor(line)+1
  return 1-sum(pois(i,lam) for i in range(thr))

def under_prob(line,lam):
  # line is always an X.5 half-line (from safe_line/safe_line_under), so
  # there's never a push -- exactly 1-over_prob, no edge case to handle.
  return 1-over_prob(line,lam)

def safe_line(lam, factor=0.55):
  raw=lam*factor
  line=math.floor(raw*2)/2
  if line<0.5: line=0.5
  return line

# Team Total Under counterpart (user request, 2026-10-18). Deliberately
# NOT the same line as safe_line() above with the comparison flipped --
# safe_line's factor=0.55 picks a line intentionally LOW relative to the
# projection so the Over is an easy clear (same shape as the Over1.5/
# Over IQ "safest pick" lines elsewhere in this suite); using that same
# low line for Under would make the Under leg a near-guaranteed MISS,
# not a real prediction. factor=1.45 is the mirror-image heuristic:
# scales the line UP from the projection by roughly the same distance
# 0.55 scales it down, so Under also lands on a line that's genuinely
# likely to hold, not just the arithmetic inverse of a line picked to
# favor Over.
def safe_line_under(lam, factor=1.45, max_line=5.5):
  raw=lam*factor
  line=math.floor(raw*2)/2
  if line<0.5: line=0.5
  # bet365 (and books generally) only lists NHL team total goals up to
  # 5.5 -- a hot team's projection * 1.45 can float past that (e.g. lam
  # 5.2 -> raw 7.5), landing on a line nobody can actually bet (user
  # report, 2026-10-18: Blue Line showed "U7.5" for Carolina but bet365's
  # own market tops out at 5.5). Cap it so the prediction always matches
  # a real, bettable line.
  if line>max_line: line=max_line
  return line

# Schedule and standings are fetched independently now -- a bad/thin
# standings sample (e.g. preseason, when every team shows 0 GP for the new
# season) used to be caught by one shared try/except that also swallowed
# the schedule fetch, so a standings-side ZeroDivisionError silently wiped
# out games we'd already found. That's wrong: the schedule doesn't depend
# on standings being usable, so it shouldn't die with it.

# FIXED: this used to flatten every gameWeek day into one list, then
# keep only games whose startTimeUTC STRING started with the FIRST
# game's UTC date. NHL games run late US evening, so a single night's
# slate routinely crosses the UTC midnight boundary partway through
# (confirmed with live bet365 evidence: 23:07/23:22/23:37/23:52 UK time
# on one date, then 00:07 UK time -- same NHL night, but a different UTC
# calendar date) -- every game on the far side of that rollover was
# silently dropped, even though it belonged to the same scheduled slate.
# /schedule/now's gameWeek[0] is the NHL API's OWN "today" bucket,
# already correctly grouped by schedule date server-side -- using its
# games directly instead of re-deriving "today" from a raw timestamp
# substring sidesteps the UTC-rollover bug entirely rather than
# patching the date-matching logic.
games=[]
try:
  sched=fetch(f"{BASE}/schedule/now")
  game_weeks=sched.get("gameWeek",[])
  games=game_weeks[0].get("games",[]) if game_weeks else []
  # EXTENDED (user request, 2026-10-22 -- "more games in UK midnight
  # time" than the page showed): this used to stop at gameWeek[0], NHL's
  # own "today" bucket. Blue Line only runs once a day (09:00 UTC, see
  # blue_line.yml), so by UK midnight that night's US slate is often
  # already deep in or over, and the NEXT scheduled NHL day's games --
  # the ones bet365 still has lines open for deep into UK night/next
  # morning -- weren't included at all until the following day's run.
  # gameWeek[1] is that next day, same NHL-API-grouped bucket gameWeek[0]
  # already relies on, so this sidesteps re-deriving "tomorrow" from a
  # raw timestamp the same way the gameWeek[0] fix avoided doing for
  # "today". Each game already carries its own date downstream
  # (game_date = startTimeUTC[:10]), so nothing else needs to change to
  # keep next-day picks distinguishable from tonight's.
  if len(game_weeks) > 1:
    games = games + game_weeks[1].get("games", [])
except Exception as e:
  print(f"schedule error: {e}", file=sys.stderr)
  games=[]

MIN_AVG_GP = 3  # avg games played per team before we trust this season's own numbers

def _avg_gp(st):
  return sum(t.get("gamesPlayed",0) for t in st.values())/len(st) if st else 0

standings={}
using_fallback_standings=False
try:
  sd=fetch(f"{BASE}/standings/now")
  cur={r["teamAbbrev"]["default"]:r for r in sd["standings"]}
  if _avg_gp(cur) < MIN_AVG_GP:
    raise ValueError(f"current season too thin (avg {_avg_gp(cur):.1f} GP/team) -- likely preseason")
  standings=cur
except Exception as e:
  print(f"current-season standings unusable ({e}); falling back to last season's final standings", file=sys.stderr)
  try:
    # Mid-April of the most recently *completed* regular season. NHL seasons
    # span two calendar years, so from Aug-Dec we want April of this same
    # year (that season just finished); from Jan-Jul the prior season ended
    # in April of the previous year.
    now=datetime.now(timezone.utc)
    fallback_year = now.year if now.month >= 8 else now.year - 1
    sd=fetch(f"{BASE}/standings/{fallback_year}-04-15")
    standings={r["teamAbbrev"]["default"]:r for r in sd["standings"]}
    using_fallback_standings=True
  except Exception as e2:
    print(f"fallback standings also failed: {e2}", file=sys.stderr)
    standings={}

avg_h=avg_r=3.0
if standings:
  try:
    avg_h=sum(t["homeGoalsFor"] for t in standings.values())/sum(t["homeGamesPlayed"] for t in standings.values())
    avg_r=sum(t["roadGoalsFor"] for t in standings.values())/sum(t["roadGamesPlayed"] for t in standings.values())
  except ZeroDivisionError as e:
    print(f"standings averages error even after fallback: {e}", file=sys.stderr)
    standings={}

def pred(home,away):
  h=standings[home]; a=standings[away]
  raw_h = (h["homeGoalsFor"]/max(h["homeGamesPlayed"],1)) * (a["roadGoalsAgainst"]/max(a["roadGamesPlayed"],1)) / avg_r
  raw_a = (a["roadGoalsFor"]/max(a["roadGamesPlayed"],1)) * (h["homeGoalsAgainst"]/max(h["homeGamesPlayed"],1)) / avg_h
  lh = min(max(raw_h, 0.8), 5.5)
  la = min(max(raw_a, 0.8), 5.5)
  tot = lh+la
  if tot < 4.5:
    scale = 5.2 / tot
    lh *= scale; la *= scale
  if tot > 8.0:
    scale = 7.2 / tot
    lh *= scale; la *= scale
  return lh,la

def get_props(team, opp, is_home):
  if team not in standings: return []
  if team in CLUB_CACHE:
    data=CLUB_CACHE[team]
  else:
    try:
      data=fetch(f"{BASE}/club-stats/{team}/now")
      CLUB_CACHE[team]=data
    except Exception as e:
      print(f"club-stats {team} failed: {e}", file=sys.stderr)
      return []

  try:
    lh,la=pred(opp if not is_home else team, team if not is_home else opp) if opp in standings and team in standings else (3,3)
    team_lam = lh if is_home else la
    season_gf = standings[team]["goalFor"]/max(standings[team]["gamesPlayed"],1)
    pace = team_lam/max(season_gf,0.1)
    shot_fac = 1+0.5*(pace-1)
    props=[]
    # FIXED: this required gp<10 to be excluded, i.e. 10+ games played
    # THIS SEASON before a skater's props would show at all. club-stats
    # resets every new season, so for the first ~6 weeks of an 82-game
    # season (confirmed live: mid-October, teams only a handful of games
    # in) literally no skater on any team clears that bar -- every match
    # card's Props sections showed "No data" across the board, which is
    # exactly what was reported. Lowered to the same MIN_AVG_GP=3 floor
    # already used for the team-level standings fallback above -- low
    # enough to clear early in the season, still high enough to keep a
    # true 1-game call-up's noisy per-game rate from dominating.
    for p in data.get("skaters",[]):
      gp=p.get("gamesPlayed",0)
      if gp<3: continue
      gpg=p["goals"]/gp; ppg=p["points"]/gp; spg=p["shots"]/gp
      lam_g=gpg*pace; lam_p=ppg*pace; lam_s=spg*shot_fac
      props.append({
        "player_id": p.get("playerId"),  # NEW: needed to match this player's
                                            # actual boxscore line later for
                                            # the results tracker -- name-only
                                            # matching is too fragile
        "name": f"{p['firstName']['default']} {p['lastName']['default']}",
        "pos": p.get("positionCode",""),
        "any": 1-pois(0,lam_g),
        "pts": 1-pois(0,lam_p),
        "sog": lam_s,
        "o1": over_prob(1.5,lam_s),
        "o2": over_prob(2.5,lam_s)
      })
    props.sort(key=lambda x:x["any"], reverse=True)
    top=props[:5]
    # Hot Form/Real Streak is only fetched for these 5 shown skaters, not
    # the whole roster (scoped this way deliberately -- one extra API call
    # per name shown, not per roster slot).
    for x in top:
      x["sog_recent"]=get_player_recent_sog(x.get("player_id"))
    return top
  except Exception as e:
    print(f"get_props {team} logic error: {e}", file=sys.stderr)
    return []

def get_goalie_props(team, opp):
  """Projected saves for `team`'s likely starter, same club-stats call
  get_props already fetches/caches (just a different list on the same
  response -- club-stats returns "goalies" alongside "skaters").

  Unlike skater props (top 5 shown, several play every game), only the
  TOP-PROJECTED goalie is returned -- an NHL team starts one goalie a
  night, so showing a backup's season average for a game they likely
  won't play would be misleading, not just extra.

  Saves scale with how many shots the OPPONENT generates, not this
  team's own pace -- so the pace factor here is the opponent's own
  scoring rate relative to the league average (avg_h/avg_r, already
  computed above for pred()), same relative-pace idea as shot_fac in
  get_props just pointed at the other team."""
  if team not in standings: return []
  if team in CLUB_CACHE:
    data=CLUB_CACHE[team]
  else:
    try:
      data=fetch(f"{BASE}/club-stats/{team}/now")
      CLUB_CACHE[team]=data
    except Exception as e:
      print(f"club-stats {team} failed: {e}", file=sys.stderr)
      return []

  try:
    league_avg_gf = (avg_h+avg_r)/2
    opp_gf = standings[opp]["goalFor"]/max(standings[opp]["gamesPlayed"],1) if opp in standings else league_avg_gf
    opp_pace = opp_gf/max(league_avg_gf,0.1)
    props=[]
    for p in data.get("goalies",[]):
      gp=p.get("gamesPlayed",0)
      if gp<3: continue
      saves=p.get("saves")
      if saves is None: continue
      spg=saves/gp
      lam=spg*opp_pace
      props.append({
        "player_id": p.get("playerId"),
        "name": f"{p['firstName']['default']} {p['lastName']['default']}",
        "pos": "G",
        "saves_proj": lam,
        "o_low": over_prob(GOALIE_SAVES_LINE_LOW,lam),
        "o_high": over_prob(GOALIE_SAVES_LINE_HIGH,lam),
      })
    props.sort(key=lambda x:x["saves_proj"], reverse=True)
    top=props[:1]
    for x in top:
      x["saves_recent"]=get_player_recent_saves(x.get("player_id"))
    return top
  except Exception as e:
    print(f"get_goalie_props {team} logic error: {e}", file=sys.stderr)
    return []

def _get_player_gamelog(player_id):
  """Shared fetch+cache for /player/{id}/game-log/now -- one request per
  player_id regardless of how many stat keys get pulled from it (a
  skater's sog and a goalie's saves never collide on the same
  player_id, so one cache serves both get_player_recent_sog and
  get_player_recent_saves below).

  UNVERIFIED ENDPOINT: this sandbox has no network path to
  api-web.nhle.com to confirm its exact response shape. The field names
  used by callers below (gameLog[], gameDate, shots/saves) match
  multiple independent public references for this endpoint, same
  confidence level as the club-schedule-season endpoint above -- not
  confirmed live. If a player's Hot Form/Real Streak entries never show
  up, check stderr for a "game-log" warning first."""
  if not player_id:
    return None
  if player_id in GAMELOG_CACHE:
    return GAMELOG_CACHE[player_id]
  try:
    log=fetch(f"{BASE}/player/{player_id}/game-log/now")
    GAMELOG_CACHE[player_id]=log
    return log
  except Exception as e:
    print(f"game-log {player_id} failed: {e}", file=sys.stderr)
    return None

def _recent_stat(player_id, stat_key, n_games):
  """Last n_games games' value of stat_key for a player, oldest-first
  (same convention as get_team_recent_goals). Sorting by gameDate
  ourselves rather than trusting the API's own order, since community
  docs disagree on whether gameLog comes back newest-first or
  oldest-first."""
  log=_get_player_gamelog(player_id)
  if not log:
    return []
  try:
    entries=[g for g in log.get("gameLog",[]) if g.get(stat_key) is not None]
    entries.sort(key=lambda g: g.get("gameDate") or "")
    last=entries[-n_games:]
    return [g[stat_key] for g in last]
  except Exception as e:
    print(f"game-log {player_id} parse error: {e}", file=sys.stderr)
    return []

def get_player_recent_sog(player_id):
  """Last SOG_HOT_FORM_MIN_GAMES games' shots-on-goal for a skater."""
  return _recent_stat(player_id, "shots", SOG_HOT_FORM_MIN_GAMES)

def get_player_recent_saves(player_id):
  """Last SAVES_HOT_FORM_MIN_GAMES games' saves for a goalie.

  Field-name hedge: unlike skater shots (confirmed indirectly -- the
  live site is showing real Player SOG Hot Form/Real Streak data, so
  "shots" is the right key), a goalie's game-log entry shape was never
  confirmed live at all. The NHL boxscore endpoint confirms "saves" as
  a flat count (already relied on by the results tracker), but
  game-log is a different endpoint and might report shotsAgainst/
  goalsAgainst instead. Tries "saves" first; if that field is simply
  absent from every entry (not just this player having no saves, which
  would be a real 0 and still match), falls back to deriving it from
  shotsAgainst-goalsAgainst. If NEITHER shape matches, logs once to
  stderr so a quiet "no Goalie Saves panel" has a concrete next step
  instead of just silently returning nothing forever."""
  log=_get_player_gamelog(player_id)
  if not log:
    return []
  entries=log.get("gameLog",[])
  if any(g.get("saves") is not None for g in entries):
    return _recent_stat(player_id, "saves", SAVES_HOT_FORM_MIN_GAMES)
  derived=[g for g in entries if g.get("shotsAgainst") is not None and g.get("goalsAgainst") is not None]
  if derived:
    print(f"game-log {player_id}: no 'saves' field on goalie entries, deriving from shotsAgainst-goalsAgainst instead", file=sys.stderr)
    derived.sort(key=lambda g: g.get("gameDate") or "")
    last=derived[-SAVES_HOT_FORM_MIN_GAMES:]
    return [g["shotsAgainst"]-g["goalsAgainst"] for g in last]
  if entries:
    print(f"game-log {player_id}: no usable saves field found on goalie entries (checked 'saves' and shotsAgainst/goalsAgainst) -- keys seen: {sorted(entries[0].keys())}", file=sys.stderr)
  return []

def get_team_recent_goals(team):
  """Last HOT_FORM_MIN_GAMES FINISHED games' goals-for for `team`,
  oldest-first (same convention as Euro Ice's get_last_five -- a
  backward walk over this list from the end is a walk from most
  recent to oldest, which is what _current_goal_streak needs).

  UNVERIFIED ENDPOINT: /club-schedule-season/{team}/now isn't called
  anywhere else in this file, and this sandbox has no network path to
  api-web.nhle.com to confirm its exact response shape. The field
  names below (games[], gameState, gameDate, homeTeam/awayTeam.abbrev
  + .score) match the shape every OTHER NHL schedule-family endpoint
  already used in this file (/schedule/now) and in the results
  tracker (/gamecenter/{id}/boxscore) uses, so this is a reasonable
  bet, not a confirmed one -- if Hot Form/Real Streak entries never
  show up on a live run, check stderr for a "club-schedule" warning
  first."""
  if team in SCHEDULE_CACHE:
    sched=SCHEDULE_CACHE[team]
  else:
    try:
      sched=fetch(f"{BASE}/club-schedule-season/{team}/now")
      SCHEDULE_CACHE[team]=sched
    except Exception as e:
      print(f"club-schedule-season {team} failed: {e}", file=sys.stderr)
      return []

  try:
    finished=[g for g in sched.get("games",[])
              if g.get("gameState") in ("OFF","FINAL")
              and g.get("homeTeam",{}).get("score") is not None
              and g.get("awayTeam",{}).get("score") is not None]
    finished.sort(key=lambda g: g.get("gameDate") or g.get("startTimeUTC") or "")
    last=finished[-HOT_FORM_MIN_GAMES:]
    out=[]
    for g in last:
      is_home=g.get("homeTeam",{}).get("abbrev")==team
      gf=g["homeTeam"]["score"] if is_home else g["awayTeam"]["score"]
      out.append(gf)
    return out
  except Exception as e:
    print(f"club-schedule-season {team} parse error: {e}", file=sys.stderr)
    return []

def _current_goal_streak(goals_oldest_first, threshold=REAL_STREAK_THRESHOLD):
  """Walk backward (most-recent-first) and stop at the first game
  under the threshold -- a genuine consecutive run, not an average."""
  streak=0
  for g in reversed(goals_oldest_first):
    if g>=threshold:
      streak+=1
    else:
      break
  return streak

def render_streak_panel(hot_form_entries, real_streak_entries):
  if not hot_form_entries and not real_streak_entries:
    return ""
  def row(e, kind):
    vs=f"{'vs' if e['is_home'] else '@'} {e['opponent']}"
    if kind=="hot":
      return f"""<div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #1e3a6a;font-size:13px">
  <span>{e['team']} {vs}</span><span style="color:#4ea1ff;font-weight:bold">{e['last5_avg']} gpg (L{HOT_FORM_MIN_GAMES})</span>
</div>"""
    plus="+" if e["full_sample"] else ""
    return f"""<div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #1e3a6a;font-size:13px">
  <span>{e['team']} {vs}</span><span style="color:#2a6f6a;font-weight:bold">{e['streak_len']}{plus} straight ({'/'.join(str(g) for g in e['streak_games'])})</span>
</div>"""
  hot_rows="".join(row(e,"hot") for e in hot_form_entries) or '<p style="color:#8aa;font-size:12px">None today.</p>'
  streak_rows="".join(row(e,"streak") for e in real_streak_entries) or '<p style="color:#8aa;font-size:12px">None today.</p>'
  return f"""
<div style="background:#0f1e3a;border:1px solid #1e3a6a;border-radius:14px;padding:16px;margin:18px 0">
  <div style="font-weight:bold;margin-bottom:2px">📊 Hot Form</div>
  <div style="color:#8aa;font-size:11px;margin-bottom:8px">Averaging {HOT_FORM_MIN}+ goals over their last {HOT_FORM_MIN_GAMES} games</div>
  {hot_rows}
</div>
<div style="background:#0f1e3a;border:1px solid #1e3a6a;border-radius:14px;padding:16px;margin:18px 0">
  <div style="font-weight:bold;margin-bottom:2px">🔥 Real Streak</div>
  <div style="color:#8aa;font-size:11px;margin-bottom:8px">Consecutive games scoring {REAL_STREAK_THRESHOLD}+ goals (oldest→newest shown)</div>
  {streak_rows}
</div>"""

def render_player_streak_panel(
    hot_form_entries, real_streak_entries,
    hot_title="📊 Player SOG Hot Form",
    hot_note=f"Averaging {SOG_HOT_FORM_MIN}+ shots on goal over their last {SOG_HOT_FORM_MIN_GAMES} games (top props skaters only)",
    hot_unit=f"SOG/g (L{SOG_HOT_FORM_MIN_GAMES})",
    streak_title="🔥 Player SOG Real Streak",
    streak_note=f"Consecutive games with {SOG_REAL_STREAK_THRESHOLD}+ shots on goal (oldest→newest shown)",
):
  """Same layout as render_streak_panel above, but for a per-player stat
  (skater SOG by default, also reused for goalie saves below via the
  hot_*/streak_* overrides) -- separate from render_streak_panel since
  rows need a player name alongside the team, not just the team."""
  if not hot_form_entries and not real_streak_entries:
    return ""
  def row(e, kind):
    vs=f"{'vs' if e['is_home'] else '@'} {e['opponent']}"
    who=f"{e['player']} ({e['team']} {vs})"
    if kind=="hot":
      return f"""<div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #1e3a6a;font-size:13px">
  <span>{who}</span><span style="color:#4ea1ff;font-weight:bold">{e['last5_avg']} {hot_unit}</span>
</div>"""
    plus="+" if e["full_sample"] else ""
    return f"""<div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #1e3a6a;font-size:13px">
  <span>{who}</span><span style="color:#2a6f6a;font-weight:bold">{e['streak_len']}{plus} straight ({'/'.join(str(g) for g in e['streak_games'])})</span>
</div>"""
  hot_rows="".join(row(e,"hot") for e in hot_form_entries) or '<p style="color:#8aa;font-size:12px">None today.</p>'
  streak_rows="".join(row(e,"streak") for e in real_streak_entries) or '<p style="color:#8aa;font-size:12px">None today.</p>'
  return f"""
<div style="background:#0f1e3a;border:1px solid #1e3a6a;border-radius:14px;padding:16px;margin:18px 0">
  <div style="font-weight:bold;margin-bottom:2px">{hot_title}</div>
  <div style="color:#8aa;font-size:11px;margin-bottom:8px">{hot_note}</div>
  {hot_rows}
</div>
<div style="background:#0f1e3a;border:1px solid #1e3a6a;border-radius:14px;padding:16px;margin:18px 0">
  <div style="font-weight:bold;margin-bottom:2px">{streak_title}</div>
  <div style="color:#8aa;font-size:11px;margin-bottom:8px">{streak_note}</div>
  {streak_rows}
</div>"""

all_legs=[]
hot_form_entries=[]
real_streak_entries=[]
player_hot_form_entries=[]
player_real_streak_entries=[]
goalie_hot_form_entries=[]
goalie_real_streak_entries=[]

BUILDER_TEMPLATE = """
<div style="background:#0f1e3a;border:1px solid #1e3a6a;border-radius:14px;padding:16px;margin:18px 0">
  <div style="font-size:15px;font-weight:700;margin-bottom:10px">🎯 Safest Bet Builder</div>
  <div id="categoryToggles" style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:10px;font-size:12px"></div>
  <div style="display:flex;gap:8px;align-items:center;margin-bottom:6px;flex-wrap:wrap">
    <label style="font-size:12px;color:#8aa">Target odds:</label>
    <input id="targetOdds" type="number" step="0.1" min="1.1" value="5.0"
      style="width:70px;background:#0a1730;border:1px solid #1e3a6a;color:white;border-radius:6px;padding:6px 8px;font-size:13px">
    <label style="font-size:12px;color:#8aa">Max legs:</label>
    <input id="maxLegs" type="number" step="1" min="2" value="8"
      style="width:55px;background:#0a1730;border:1px solid #1e3a6a;color:white;border-radius:6px;padding:6px 8px;font-size:13px">
    <button onclick="buildSafest()"
      style="background:#2a6f6a;border:none;color:white;padding:7px 14px;border-radius:6px;font-size:13px;cursor:pointer">
      Build
    </button>
    <button onclick="buildSafest()"
      style="background:#1e3a6a;border:1px solid #2a4a7a;color:white;padding:7px 14px;border-radius:6px;font-size:13px;cursor:pointer">
      🔀 Shuffle
    </button>
  </div>
  <div id="builderResult" style="font-size:12px;color:#8aa">
    Untick any market type you don't want considered, set a target odds and
    leg cap, then tap Build. It rotates through whichever categories are
    ticked, groups near-tied legs and shuffles within each group so it draws
    from more of the day's games rather than always the exact same few, and
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
    `<div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #1e3a6a">
       <span>${{l.match}}<br><span style="color:#4ea1ff">${{l.market}}</span> <span style="color:#5a6a8a">· ${{l.category}}</span>
       ${{l.detail ? `<br><span style="color:#5a6a8a;font-size:10px">${{l.detail}}</span>` : ''}}</span>
       <span style="color:#ffeb3b;font-weight:bold">${{l.prob}}%</span>
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
    <div style="color:#5a6a8a;font-size:10px;margin-top:8px;line-height:1.4">
      Estimate multiplies each leg's fair odds (100/probability) — real
      sportsbook odds include their margin and legs within the same game
      aren't fully independent, so treat this as a ranking tool, not a firm
      price. Team/game totals and shots-on-goal use Poisson projections with
      the line set below the model's expectation for a safety margin;
      anytime-goal and to-record-a-point use the model's own at-least-one
      probabilities directly.
    </div>
  `;
}}
</script>
"""

cards=""
for g in games:
  try:
    home=g["homeTeam"]["abbrev"]; away=g["awayTeam"]["abbrev"]
    game_date = g.get("startTimeUTC", "")[:10]
    # Shown on the card now that `games` can span two NHL schedule days
    # (today + tomorrow, see the gameWeek[1] extension above) -- without
    # this, a tonight's-slate card and a tomorrow's-slate card look
    # identical. Falls back to the raw date string if parsing fails
    # rather than dropping the card.
    try:
      date_label = datetime.strptime(game_date, "%Y-%m-%d").strftime("%a %b %-d")
    except ValueError:
      date_label = game_date
    lh,la=pred(home,away)
    tot=lh+la
    o55=1-sum(pois(i,tot) for i in range(6))
    ph=pa=pt=0
    for i in range(0,10):
      for j in range(0,10):
        p=pois(i,lh)*pois(j,la)
        if i>j: ph+=p
        elif j>i: pa+=p
        else: pt+=p
    scores=[]
    for i in range(0,7):
      for j in range(0,7):
        scores.append(((j,i), pois(j,la)*pois(i,lh)))
    scores.sort(key=lambda x:x[1], reverse=True)
    top3=scores[:3]

    props_h=get_props(home,away,True)
    props_a=get_props(away,home,False)
    goalie_h=get_goalie_props(home,away)
    goalie_a=get_goalie_props(away,home)
    def render_props(lst):
      return "".join([f"<div style='display:flex;justify-content:space-between;gap:12px;font-size:13px;padding:6px 0;border-bottom:1px solid #1e3a6a'><span style='min-width:120px'>{x['name']} ({x['pos']})</span><span style='text-align:right'>Goal {x['any']*100:.0f}% | 1+Pt {x['pts']*100:.0f}% | SOG {x['sog']:.1f} O1.5 {x['o1']*100:.0f}%</span></div>" for x in lst])
    def render_goalie(lst):
      return "".join([f"<div style='display:flex;justify-content:space-between;gap:12px;font-size:13px;padding:6px 0;border-bottom:1px solid #1e3a6a'><span style='min-width:120px'>{x['name']} (G)</span><span style='text-align:right'>Proj {x['saves_proj']:.1f} saves | O{GOALIE_SAVES_LINE_LOW} {x['o_low']*100:.0f}% | O{GOALIE_SAVES_LINE_HIGH} {x['o_high']*100:.0f}%</span></div>" for x in lst])
    def render_scores():
      return "".join([f"<div style='background:#0a1730;border-radius:8px;padding:8px;text-align:center'><div style='font-size:12px;color:#8aa'>{away} {a}-{h} {home}</div><div style='font-weight:700;margin-top:2px'>{p*100:.1f}%</div></div>" for (a,h),p in top3])

    # Last-5-games goals history, same source feeding Hot Form/Real Streak
    # below -- shown on the card itself too, matching Euro Ice's card layout
    # (its render_match_card puts this exact "last 5 (old->new)" line under
    # each team's win%, in place of a props section it doesn't have data for).
    recent_goals_by_team = {home: get_team_recent_goals(home), away: get_team_recent_goals(away)}
    home_hist = "/".join(str(v) for v in recent_goals_by_team[home]) or "—"
    away_hist = "/".join(str(v) for v in recent_goals_by_team[away]) or "—"

    win_bar=f"""<div style="margin:10px 0 6px 0"><div style="display:flex;justify-content:space-between;align-items:flex-start;font-size:12px;margin-bottom:4px"><div><div>{away} {pa*100:.0f}%</div><div style="font-size:10px;color:#cbd5e1;margin-top:2px">last 5 (old→new): {away_hist}</div></div><span>Tie {pt*100:.0f}%</span><div style="text-align:right"><div>{home} {ph*100:.0f}%</div><div style="font-size:10px;color:#cbd5e1;margin-top:2px">last 5 (old→new): {home_hist}</div></div></div><div style="display:flex;height:10px;border-radius:999px;overflow:hidden;background:#0a1730"><div style="width:{pa*100:.1f}%;background:#ff4d5a"></div><div style="width:{pt*100:.1f}%;background:#5a5f7a"></div><div style="width:{ph*100:.1f}%;background:#4ea1ff"></div></div></div>"""

    # Team Total Over/Under summary shown directly on the card (user
    # request, 2026-10-18 -- previously Team Total only appeared inside
    # the Safest Bet Builder below, easy to miss). Same lines the builder
    # itself uses (safe_line/safe_line_under), just surfaced here too so
    # the prediction is visible without opening the builder.
    home_o_line, away_o_line = safe_line(lh), safe_line(la)
    home_u_line, away_u_line = safe_line_under(lh), safe_line_under(la)
    team_total_line=f"""<p style="margin:4px 0 0;color:#8aa;font-size:12px">Team Total — {away} O{away_o_line} {over_prob(away_o_line,la)*100:.0f}% / U{away_u_line} {under_prob(away_u_line,la)*100:.0f}% · {home} O{home_o_line} {over_prob(home_o_line,lh)*100:.0f}% / U{home_u_line} {under_prob(home_u_line,lh)*100:.0f}%</p>"""

    cards+=f"""<div style="background:#0f1e3a;border:1px solid #1e3a6a;border-radius:14px;padding:16px;margin:18px 0"><h3 style="margin:0 0 4px 0">{away} @ {home} <span style="color:#7ec8ff;font-weight:normal;font-size:13px">· {date_label}</span> — Total {tot:.2f}</h3><p style="margin:0;color:#b7c5e6;font-size:13px">Proj: {away} {la:.2f} - {lh:.2f} {home} | O5.5 {o55*100:.0f}%</p>{team_total_line}{win_bar}<div style="margin-top:12px"><div style="font-size:12px;color:#8aa;margin-bottom:6px">Correct Score</div><div style="display:grid;grid-template-columns:repeat(3,1fr);gap:6px">{render_scores()}</div></div><details style="margin-top:12px"><summary style="cursor:pointer;color:#4ea1ff;font-size:13px">{home} Props</summary><div style="margin-top:8px">{render_props(props_h) or 'No data'}</div></details><details style="margin-top:8px"><summary style="cursor:pointer;color:#4ea1ff;font-size:13px">{away} Props</summary><div style="margin-top:8px">{render_props(props_a) or 'No data'}</div></details><details style="margin-top:8px"><summary style="cursor:pointer;color:#4ea1ff;font-size:13px">Goalie Saves</summary><div style="margin-top:8px">{render_goalie(goalie_h+goalie_a) or 'No data'}</div></details></div>"""

    match_label = f"{away} @ {home}"
    game_id = g.get("id")  # NEW: needed later to look up the real final
                             # result for the results tracker

    for team_label, opp_label, is_home_flag in [(home, away, True), (away, home, False)]:
      recent_goals = recent_goals_by_team[team_label]
      if len(recent_goals) >= HOT_FORM_MIN_GAMES:
        avg5 = round(sum(recent_goals)/len(recent_goals), 2)
        if avg5 >= HOT_FORM_MIN:
          hot_form_entries.append({
            "team": team_label, "opponent": opp_label, "is_home": is_home_flag,
            "last5_avg": avg5, "game_id": game_id, "date": game_date,
          })
      streak_len = _current_goal_streak(recent_goals)
      if streak_len >= REAL_STREAK_MIN_LENGTH:
        real_streak_entries.append({
          "team": team_label, "opponent": opp_label, "is_home": is_home_flag,
          "streak_len": streak_len, "streak_games": recent_goals[-streak_len:],
          "full_sample": streak_len >= len(recent_goals),
          "game_id": game_id, "date": game_date,
        })

    # Skater SOG Hot Form/Real Streak -- props already carry sog_recent
    # (oldest-first) from get_props, only for the 5 skaters shown per team.
    for team_label, opp_label, is_home_flag, props in [
      (home, away, True, props_h), (away, home, False, props_a)
    ]:
      for x in props:
        sog_hist = x.get("sog_recent") or []
        if len(sog_hist) >= SOG_HOT_FORM_MIN_GAMES:
          avg5 = round(sum(sog_hist)/len(sog_hist), 2)
          if avg5 >= SOG_HOT_FORM_MIN:
            player_hot_form_entries.append({
              "player": x["name"], "player_id": x.get("player_id"), "pos": x["pos"],
              "team": team_label, "opponent": opp_label, "is_home": is_home_flag,
              "last5_avg": avg5, "game_id": game_id, "date": game_date,
            })
        streak_len = _current_goal_streak(sog_hist, SOG_REAL_STREAK_THRESHOLD)
        if streak_len >= SOG_REAL_STREAK_MIN_LENGTH:
          player_real_streak_entries.append({
            "player": x["name"], "player_id": x.get("player_id"), "pos": x["pos"],
            "team": team_label, "opponent": opp_label, "is_home": is_home_flag,
            "streak_len": streak_len, "streak_games": sog_hist[-streak_len:],
            "full_sample": streak_len >= len(sog_hist),
            "game_id": game_id, "date": game_date,
          })

    # Goalie Saves Hot Form/Real Streak -- goalie_h/goalie_a already carry
    # saves_recent (oldest-first) from get_goalie_props, only for the 1
    # top-projected starter shown per team.
    for team_label, opp_label, is_home_flag, goalies in [
      (home, away, True, goalie_h), (away, home, False, goalie_a)
    ]:
      for x in goalies:
        saves_hist = x.get("saves_recent") or []
        if len(saves_hist) >= SAVES_HOT_FORM_MIN_GAMES:
          avg5 = round(sum(saves_hist)/len(saves_hist), 2)
          if avg5 >= SAVES_HOT_FORM_MIN:
            goalie_hot_form_entries.append({
              "player": x["name"], "player_id": x.get("player_id"), "pos": "G",
              "team": team_label, "opponent": opp_label, "is_home": is_home_flag,
              "last5_avg": avg5, "game_id": game_id, "date": game_date,
            })
        streak_len = _current_goal_streak(saves_hist, SAVES_REAL_STREAK_THRESHOLD)
        if streak_len >= SAVES_REAL_STREAK_MIN_LENGTH:
          goalie_real_streak_entries.append({
            "player": x["name"], "player_id": x.get("player_id"), "pos": "G",
            "team": team_label, "opponent": opp_label, "is_home": is_home_flag,
            "streak_len": streak_len, "streak_games": saves_hist[-streak_len:],
            "full_sample": streak_len >= len(saves_hist),
            "game_id": game_id, "date": game_date,
          })

    for team_label, lam in [(home, lh), (away, la)]:
      line = safe_line(lam)
      all_legs.append({
        "match": match_label, "market": f"{team_label} Over {line} Goals",
        "prob": round(over_prob(line, lam)*100), "category": "Team Total",
        "detail": f"proj {lam:.2f} goals",
        "game_id": game_id, "game_date": game_date,
        "is_home": team_label == home, "line": line, "direction": "over",
      })
      under_line = safe_line_under(lam)
      all_legs.append({
        "match": match_label, "market": f"{team_label} Under {under_line} Goals",
        "prob": round(under_prob(under_line, lam)*100), "category": "Team Total",
        "detail": f"proj {lam:.2f} goals",
        "game_id": game_id, "game_date": game_date,
        "is_home": team_label == home, "line": under_line, "direction": "under",
      })
    all_legs.append({
      "match": match_label, "market": "Game Over 5.5 Total Goals",
      "prob": round(o55*100), "category": "Game Total",
      "detail": f"proj {tot:.2f} goals",
      "game_id": game_id, "game_date": game_date, "line": 5.5,
    })
    for team_label, props in [(home, props_h), (away, props_a)]:
      for x in props:
        all_legs.append({
          "match": match_label, "market": f"{x['name']} Anytime Goal",
          "prob": round(x["any"]*100), "category": "Anytime Goalscorer",
          "detail": f"{team_label} · {x['pos']}",
          "game_id": game_id, "game_date": game_date, "player_id": x.get("player_id"),
          "is_home": team_label == home,
        })
        all_legs.append({
          "match": match_label, "market": f"{x['name']} Over 0.5 Points",
          "prob": round(x["pts"]*100), "category": "To Record a Point",
          "detail": f"{team_label} · {x['pos']}",
          "game_id": game_id, "game_date": game_date, "player_id": x.get("player_id"),
          "is_home": team_label == home,
        })
        all_legs.append({
          "match": match_label, "market": f"{x['name']} Over 1.5 SOG",
          "prob": round(x["o1"]*100), "category": "Shots on Goal",
          "detail": f"{team_label} · {x['pos']} · proj {x['sog']:.1f} SOG",
          "game_id": game_id, "game_date": game_date, "player_id": x.get("player_id"), "line": 1.5,
          "is_home": team_label == home,
        })
    for team_label, goalies in [(home, goalie_h), (away, goalie_a)]:
      for x in goalies:
        for line, prob_key in [(GOALIE_SAVES_LINE_LOW, "o_low"), (GOALIE_SAVES_LINE_HIGH, "o_high")]:
          all_legs.append({
            "match": match_label, "market": f"{x['name']} Over {line} Saves",
            "prob": round(x[prob_key]*100), "category": "Goalie Saves",
            "detail": f"{team_label} · G · proj {x['saves_proj']:.1f} saves",
            "game_id": game_id, "game_date": game_date, "player_id": x.get("player_id"), "line": line,
            "is_home": team_label == home,
          })
  except Exception as e: print(f"game loop {e}", file=sys.stderr); continue

builder = BUILDER_TEMPLATE.format(legs_json=json.dumps(all_legs)) if all_legs else ""
streak_panel = render_streak_panel(hot_form_entries, real_streak_entries)
player_streak_panel = render_player_streak_panel(player_hot_form_entries, player_real_streak_entries)
goalie_streak_panel = render_player_streak_panel(
  goalie_hot_form_entries, goalie_real_streak_entries,
  hot_title="📊 Goalie Saves Hot Form",
  hot_note=f"Averaging {SAVES_HOT_FORM_MIN}+ saves over their last {SAVES_HOT_FORM_MIN_GAMES} games (top-projected starter only)",
  hot_unit=f"saves/g (L{SAVES_HOT_FORM_MIN_GAMES})",
  streak_title="🔥 Goalie Saves Real Streak",
  streak_note=f"Consecutive games with {SAVES_REAL_STREAK_THRESHOLD}+ saves (oldest→newest shown)",
)

fallback_note = ('<p style="margin:4px 0 0;color:#ffb84e;font-size:12px">⚠️ Early season — projections use last season\'s final standings '
                  'until this season has a few games of its own on the board.</p>') if using_fallback_standings else ""

html=f"""<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Blue Line v2 Clean</title><style>body{{background:#081229;color:#fff;font-family:-apple-system,system-ui,sans-serif;padding:16px;max-width:800px;margin:0 auto}}h1{{color:#4ea1ff;font-size:22px}}</style></head><body><h1>🔵 Blue Line v2 — Patched</h1><p style="color:#8aa;font-size:12px">Last: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} | Games: {len(games)}</p>{fallback_note}<p style="margin:4px 0 0"><a href="results/index.html" style="color:#ffeb3b;text-decoration:none;font-size:12px">📊 Results Tracker</a></p>{builder}{streak_panel}{player_streak_panel}{goalie_streak_panel}{cards or '<p>No games today — model ready.</p>'}<p style="font-size:11px;color:#5a6a8a;margin-top:24px">Fixes: cached club-stats, logged excepts, utcnow→now(utc), total clamp 4.5-8.0. Added: Safest Bet Builder (goals/props markets only — no moneyline, since ph/pa are regulation-time only and would understate a favorite's true win odds through OT/SO). Games and standings are now fetched/validated independently so a thin preseason sample can't wipe out a valid schedule.</p></body></html>"""
import os as _os; _os.makedirs("docs/blue-line", exist_ok=True)
with open("docs/blue-line/index.html","w") as f: f.write(html)

try:
  import results_tracker_blue_line
  results_tracker_blue_line.run_results_tracker(
    all_legs, hot_form_entries, real_streak_entries,
    player_hot_form_entries, player_real_streak_entries,
    goalie_hot_form_entries, goalie_real_streak_entries,
  )
except Exception as e:
  # Results tracking sits on top of everything above, which has already
  # succeeded by this point -- a failure here should never take down an
  # otherwise-successful run.
  print(f"[!] Results tracker failed, but the rest of this run succeeded: {e}", file=sys.stderr)

print(f"Done {len(games)}")
