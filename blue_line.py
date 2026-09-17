#!/usr/bin/env python3
import math, requests, sys, json
from datetime import datetime, timezone
BASE="https://api-web.nhle.com/v1"
H={"User-Agent":"Mozilla/5.0"}
CLUB_CACHE={} # memoize club-stats

def fetch(u):
  return requests.get(u, headers=H, timeout=20).json()

def pois(k,lam):
  return math.exp(-lam)*lam**k/math.factorial(k)

def over_prob(line,lam):
  thr=math.floor(line)+1
  return 1-sum(pois(i,lam) for i in range(thr))

def safe_line(lam, factor=0.55):
  # Same "safety margin" line-setting as the soccer/NFL siblings, but with
  # a lower factor than their 0.72 default. Team goals are a low-count
  # Poisson stat (lambda ~2-4), so 0.72 was landing the line too close to
  # the mean — team totals were coming out at 53-76% in testing, notably
  # weaker than the 80-90%+ range shots/passing-yards-type legs get
  # elsewhere. 0.55 pushes the line down further for more real margin;
  # a team projected at only ~2 goals will still show a thinner number
  # than a team projected at 4 — that's the model being honest about a
  # genuinely thinner safety margin, not something to paper over.
  raw=lam*factor
  line=math.floor(raw*2)/2
  if line<0.5: line=0.5
  return line

# standings + projections
try:
  sd=fetch(f"{BASE}/standings/now")
  standings={r["teamAbbrev"]["default"]:r for r in sd["standings"]}
  avg_h=sum(t["homeGoalsFor"] for t in standings.values())/sum(t["homeGamesPlayed"] for t in standings.values())
  avg_r=sum(t["roadGoalsFor"] for t in standings.values())/sum(t["roadGamesPlayed"] for t in standings.values())

  def pred(home,away):
    h=standings[home]; a=standings[away]
    # FIXED: combined-total clamp - was double-clamping to 6 which killed totals
    raw_h = (h["homeGoalsFor"]/max(h["homeGamesPlayed"],1)) * (a["roadGoalsAgainst"]/max(a["roadGamesPlayed"],1)) / avg_r
    raw_a = (a["roadGoalsFor"]/max(a["roadGamesPlayed"],1)) * (h["homeGoalsAgainst"]/max(h["homeGamesPlayed"],1)) / avg_h
    lh = min(max(raw_h, 0.8), 5.5) # was 0.5-6, now 0.8-5.5 more realistic
    la = min(max(raw_a, 0.8), 5.5)
    # re-normalize total so we don't get 1-1 spin nobody trusts
    tot = lh+la
    if tot < 4.5:
      scale = 5.2 / tot
      lh *= scale; la *= scale
    if tot > 8.0:
      scale = 7.2 / tot
      lh *= scale; la *= scale
    return lh,la

  sched=fetch(f"{BASE}/schedule/now")
  games=[]
  for wk in sched.get("gameWeek",[]):
    for g in wk.get("games",[]): games.append(g)
  if games:
    first=games[0]["startTimeUTC"][:10]
    games=[g for g in games if g["startTimeUTC"].startswith(first)]
except Exception as e:
  print(f"standings/schedule error: {e}", file=sys.stderr)
  games=[]; standings={}

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
    for p in data.get("skaters",[]):
      gp=p.get("gamesPlayed",0)
      if gp<10: continue
      gpg=p["goals"]/gp; ppg=p["points"]/gp; spg=p["shots"]/gp
      lam_g=gpg*pace; lam_p=ppg*pace; lam_s=spg*shot_fac
      props.append({
        "name": f"{p['firstName']['default']} {p['lastName']['default']}",
        "pos": p.get("positionCode",""),
        "any": 1-pois(0,lam_g),
        "pts": 1-pois(0,lam_p),
        "sog": lam_s,
        "o1": over_prob(1.5,lam_s),
        "o2": over_prob(2.5,lam_s)
      })
    props.sort(key=lambda x:x["any"], reverse=True)
    return props[:5]
  except Exception as e:
    print(f"get_props {team} logic error: {e}", file=sys.stderr)
    return []

# ---------------------------------------------------------------------------
# Safest Bet Builder — same pattern as Corner Flag/Match IQ (soccer) and
# Blitz IQ (NFL): every market below gets a real probability, tagged with a
# category, collected into all_legs as the game loop runs. NOTE: unlike the
# soccer/NFL siblings, this script has no per-game history list (club-stats
# is a season aggregate, not a game log) so legs here carry no "last games"
# sequence — detail only.
#
# Deliberately NOT included: moneyline/win-market legs. ph/pa/pt below are
# REGULATION-time probabilities only (the Poisson grid has no way to model
# overtime/shootout), so they'd understate a favorite's true moneyline win
# probability, which includes OT/SO. Including them as "safe" legs would be
# misleading in a way the goals/props markets aren't.
# ---------------------------------------------------------------------------
all_legs=[]

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
    top9=scores[:9]

    props_h=get_props(home,away,True)
    props_a=get_props(away,home,False)
    def render_props(lst):
      return "".join([f"<div style='display:flex;justify-content:space-between;gap:12px;font-size:13px;padding:6px 0;border-bottom:1px solid #1e3a6a'><span style='min-width:120px'>{x['name']} ({x['pos']})</span><span style='text-align:right'>Goal {x['any']*100:.0f}% | 1+Pt {x['pts']*100:.0f}% | SOG {x['sog']:.1f} O1.5 {x['o1']*100:.0f}%</span></div>" for x in lst])
    def render_scores():
      return "".join([f"<div style='background:#0a1730;border-radius:8px;padding:8px;text-align:center'><div style='font-size:12px;color:#8aa'>{away} {a}-{h} {home}</div><div style='font-weight:700;margin-top:2px'>{p*100:.1f}%</div></div>" for (a,h),p in top9])

    win_bar=f"""<div style="margin:10px 0 6px 0"><div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:4px"><span>{away} {pa*100:.0f}%</span><span>Tie {pt*100:.0f}%</span><span>{home} {ph*100:.0f}%</span></div><div style="display:flex;height:10px;border-radius:999px;overflow:hidden;background:#0a1730"><div style="width:{pa*100:.1f}%;background:#ff4d5a"></div><div style="width:{pt*100:.1f}%;background:#5a5f7a"></div><div style="width:{ph*100:.1f}%;background:#4ea1ff"></div></div></div>"""

    cards+=f"""<div style="background:#0f1e3a;border:1px solid #1e3a6a;border-radius:14px;padding:16px;margin:18px 0"><h3 style="margin:0 0 4px 0">{away} @ {home} — Total {tot:.2f}</h3><p style="margin:0;color:#b7c5e6;font-size:13px">Proj: {away} {la:.2f} - {lh:.2f} {home} | O5.5 {o55*100:.0f}%</p>{win_bar}<div style="margin-top:12px"><div style="font-size:12px;color:#8aa;margin-bottom:6px">Correct Score</div><div style="display:grid;grid-template-columns:repeat(3,1fr);gap:6px">{render_scores()}</div></div><details style="margin-top:12px"><summary style="cursor:pointer;color:#4ea1ff;font-size:13px">{home} Props</summary><div style="margin-top:8px">{render_props(props_h) or 'No data'}</div></details><details style="margin-top:8px"><summary style="cursor:pointer;color:#4ea1ff;font-size:13px">{away} Props</summary><div style="margin-top:8px">{render_props(props_a) or 'No data'}</div></details></div>"""

    # --- Safest Bet Builder legs for this game ---
    match_label = f"{away} @ {home}"
    for team_label, lam in [(home, lh), (away, la)]:
      line = safe_line(lam)
      all_legs.append({
        "match": match_label, "market": f"{team_label} Over {line} Goals",
        "prob": round(over_prob(line, lam)*100), "category": "Team Total",
        "detail": f"proj {lam:.2f} goals",
      })
    all_legs.append({
      "match": match_label, "market": "Game Over 5.5 Total Goals",
      "prob": round(o55*100), "category": "Game Total",
      "detail": f"proj {tot:.2f} goals",
    })
    for team_label, props in [(home, props_h), (away, props_a)]:
      for x in props:
        all_legs.append({
          "match": match_label, "market": f"{x['name']} Anytime Goal",
          "prob": round(x["any"]*100), "category": "Anytime Goalscorer",
          "detail": f"{team_label} · {x['pos']}",
        })
        all_legs.append({
          "match": match_label, "market": f"{x['name']} Over 0.5 Points",
          "prob": round(x["pts"]*100), "category": "To Record a Point",
          "detail": f"{team_label} · {x['pos']}",
        })
        all_legs.append({
          "match": match_label, "market": f"{x['name']} Over 1.5 SOG",
          "prob": round(x["o1"]*100), "category": "Shots on Goal",
          "detail": f"{team_label} · {x['pos']} · proj {x['sog']:.1f} SOG",
        })
  except Exception as e: print(f"game loop {e}", file=sys.stderr); continue

builder = BUILDER_TEMPLATE.format(legs_json=json.dumps(all_legs)) if all_legs else ""

html=f"""<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Blue Line v2 Clean</title><style>body{{background:#081229;color:#fff;font-family:-apple-system,system-ui,sans-serif;padding:16px;max-width:800px;margin:0 auto}}h1{{color:#4ea1ff;font-size:22px}}</style></head><body><h1>🔵 Blue Line v2 — Patched</h1><p style="color:#8aa;font-size:12px">Last: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} | Games: {len(games)}</p>{builder}{cards or '<p>No games today — model ready.</p>'}<p style="font-size:11px;color:#5a6a8a;margin-top:24px">Fixes: cached club-stats, logged excepts, utcnow→now(utc), total clamp 4.5-8.0. Added: Safest Bet Builder (goals/props markets only — no moneyline, since ph/pa are regulation-time only and would understate a favorite's true win odds through OT/SO).</p></body></html>"""
import os as _os; _os.makedirs("docs/blue-line", exist_ok=True)
with open("docs/blue-line/index.html","w") as f: f.write(html)
print(f"Done {len(games)}")
