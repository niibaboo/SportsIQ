#!/usr/bin/env python3
"""
Strike Zone — Daily Slate Builder
Fetches today's (or a given date's) MLB games + probable pitchers from the
public MLB Stats API, runs each pitcher through the same TBF-based /
recency-weighted / Poisson K-prop model as the web calculator, and writes
a static HTML report you can open in any browser.

Usage:
    pip3 install requests --break-system-packages
    python3 build_slate.py                # today
    python3 build_slate.py 2026-09-05     # specific date

Output:
    slate_report.html  (in the same folder — open it in your browser)
"""

import sys
import math
import json
from datetime import date, datetime
from zoneinfo import ZoneInfo
import requests

UK_TZ = ZoneInfo("Europe/London")

BASE = "https://statsapi.mlb.com/api/v1"
LEAGUE_AVG_K_PCT = 22.1
LEAGUE_AVG_ERA = 4.00
BF_PER_IP = 4.3
RECENT_WEIGHT = 0.65


def bb_ip_to_decimal(ip_str):
    """MLB reports IP in baseball notation: '5.1' = 5 1/3, '5.2' = 5 2/3."""
    s = str(ip_str)
    if "." not in s:
        return float(s)
    whole, frac = s.split(".")
    w = float(whole) if whole else 0.0
    if frac == "1":
        return w + 1 / 3
    if frac == "2":
        return w + 2 / 3
    return w + (float("0." + frac) if frac else 0.0)


def poisson_pmf(k, lam):
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def winsorize_iqr(values, cap_multiplier=1.5):
    """Caps any single game at 1.5x the median of the last-5 sample before
    recency-weighting. Built for team runs/hits, where a single blowout game
    (e.g. a 12-run outburst) can otherwise dominate a small-sample weighted
    average and produce a wildly overconfident projection.

    (Note: an IQR/Tukey-fence approach was tried first but turned out too
    loose on a 5-value sample to reliably catch real outliers — a median
    cap is simpler and more predictable at this sample size.)
    Only the high end is capped, since a big scoring game is the case that
    actually distorts these projections; a shutout (0) is a normal, bounded
    outcome that doesn't need correcting the same way.
    """
    if len(values) < 3:
        return list(values)
    med = sorted(values)[len(values) // 2]
    cap = med * cap_multiplier
    return [min(v, cap) for v in values]


def get_season(season_year):
    return season_year


team_k_cache = {}


def get_team_k_pct(team_id, season):
    if team_id in team_k_cache:
        return team_k_cache[team_id]
    r = requests.get(f"{BASE}/teams/{team_id}/stats",
                      params={"stats": "season", "group": "hitting", "season": season})
    r.raise_for_status()
    data = r.json()
    splits = data.get("stats", [{}])[0].get("splits", [])
    if not splits:
        return None
    stat = splits[0]["stat"]
    so = stat.get("strikeOuts")
    pa = stat.get("plateAppearances")
    if pa is None:
        pa = (float(stat.get("atBats", 0)) + float(stat.get("baseOnBalls", 0)) +
              float(stat.get("hitByPitch", 0)) + float(stat.get("sacFlies", 0)) +
              float(stat.get("sacBunts", 0)))
    if not pa:
        return None
    kpct = (so / pa) * 100
    team_k_cache[team_id] = kpct
    return kpct


team_hitting_cache = {}
team_pitching_stat_cache = {}
team_gamelog_cache = {}
LEAGUE_AVG_HITS9 = 8.5


def get_team_hitting_stat(team_id, season):
    if team_id in team_hitting_cache:
        return team_hitting_cache[team_id]
    r = requests.get(f"{BASE}/teams/{team_id}/stats",
                      params={"stats": "season", "group": "hitting", "season": season})
    r.raise_for_status()
    splits = r.json().get("stats", [{}])[0].get("splits", [])
    stat = splits[0]["stat"] if splits else None
    team_hitting_cache[team_id] = stat
    return stat


def get_team_pitching_stat(team_id, season):
    """Full team-level season pitching stat dict (ERA, hitsPer9Inn, etc.),
    used as a bullpen-quality proxy for the innings the opposing starter
    doesn't cover."""
    if team_id in team_pitching_stat_cache:
        return team_pitching_stat_cache[team_id]
    r = requests.get(f"{BASE}/teams/{team_id}/stats",
                      params={"stats": "season", "group": "pitching", "season": season})
    r.raise_for_status()
    splits = r.json().get("stats", [{}])[0].get("splits", [])
    stat = splits[0]["stat"] if splits else None
    team_pitching_stat_cache[team_id] = stat
    return stat


def get_team_pitching_era(team_id, season):
    stat = get_team_pitching_stat(team_id, season)
    return float(stat["era"]) if stat and stat.get("era") else None


def get_team_pitching_hits9(team_id, season):
    stat = get_team_pitching_stat(team_id, season)
    return float(stat["hitsPer9Inn"]) if stat and stat.get("hitsPer9Inn") else None


def get_team_gamelog_splits(team_id, season):
    """Cached per-game hitting log for a team — shared source for both the
    runs and hits last-5 projections so we only fetch it once per team."""
    if team_id in team_gamelog_cache:
        return team_gamelog_cache[team_id]
    r = requests.get(f"{BASE}/teams/{team_id}/stats",
                      params={"stats": "gameLog", "group": "hitting", "season": season})
    r.raise_for_status()
    splits = r.json().get("stats", [{}])[0].get("splits", [])
    splits.sort(key=lambda s: s["date"])
    team_gamelog_cache[team_id] = splits
    return splits


def get_team_runs_last5(team_id, season):
    splits = get_team_gamelog_splits(team_id, season)
    return [s["stat"]["runs"] for s in splits[-5:]]


def get_team_hits_last5(team_id, season):
    splits = get_team_gamelog_splits(team_id, season)
    return [s["stat"]["hits"] for s in splits[-5:]]


def project_team_runs(team_id, season, opp_starter_era, starter_proj_ip, opp_team_id, weight=RECENT_WEIGHT):
    """Expected team runs = blended (season + recency) offense rate,
    adjusted for the specific opposing starter's quality for the innings
    he's projected to pitch, and the opposing team's overall staff ERA
    (as an approximation for bullpen quality) for the remaining innings.
    This is a coarser model than the pitcher props — it can't see bullpen
    matchups, park factors, or lineup-specific splits, so treat edges here
    with more skepticism than the K/outs props.
    """
    hstat = get_team_hitting_stat(team_id, season)
    if not hstat:
        return None
    games = hstat.get("gamesPlayed") or 1
    season_rpg = hstat.get("runs", 0) / games

    last5_runs = get_team_runs_last5(team_id, season)
    if len(last5_runs) >= 2:
        n = len(last5_runs)
        clipped_runs = winsorize_iqr(last5_runs)
        wts = [1.4 ** i for i in range(n)]
        recent_rpg = sum(w * r for w, r in zip(wts, clipped_runs)) / sum(wts)
    else:
        recent_rpg = season_rpg

    blended_rpg = weight * recent_rpg + (1 - weight) * season_rpg

    starter_share = max(0.0, min(1.0, (starter_proj_ip or 5.5) / 9))
    bullpen_share = 1 - starter_share
    opp_team_era = get_team_pitching_era(opp_team_id, season) or LEAGUE_AVG_ERA
    starter_adj = (opp_starter_era / LEAGUE_AVG_ERA) if opp_starter_era else 1.0
    bullpen_adj = (opp_team_era / LEAGUE_AVG_ERA)
    run_factor = starter_share * starter_adj + bullpen_share * bullpen_adj

    lam = blended_rpg * run_factor
    lo = hi = 0
    cum = 0.0
    for i in range(30):
        cum += poisson_pmf(i, lam)
        if cum >= 0.10 and lo == 0:
            lo = i
        if cum >= 0.90:
            hi = i
            break

    l5_str = "·".join(str(r) for r in last5_runs) if last5_runs else "—"
    return {"lambda": round(lam, 2), "lo": lo, "hi": hi, "l5_str": l5_str,
            "last5_runs": last5_runs,
            "season_rpg": round(season_rpg, 2), "run_factor": round(run_factor, 2)}


def project_team_hits(team_id, season, opp_starter_hits9, starter_proj_ip, opp_team_id, weight=RECENT_WEIGHT):
    """Same structure as project_team_runs but for team hits allowed —
    generally more reliable than runs since it doesn't depend on hit
    *sequencing* (stranding runners doesn't erase a hit the way it erases
    a run), though it's still a whole-lineup, multi-pitcher stat, so treat
    it as less reliable than the single-pitcher K/outs props.
    """
    hstat = get_team_hitting_stat(team_id, season)
    if not hstat:
        return None
    games = hstat.get("gamesPlayed") or 1
    season_hpg = hstat.get("hits", 0) / games

    last5_hits = get_team_hits_last5(team_id, season)
    if len(last5_hits) >= 2:
        n = len(last5_hits)
        clipped_hits = winsorize_iqr(last5_hits)
        wts = [1.4 ** i for i in range(n)]
        recent_hpg = sum(w * h for w, h in zip(wts, clipped_hits)) / sum(wts)
    else:
        recent_hpg = season_hpg

    blended_hpg = weight * recent_hpg + (1 - weight) * season_hpg

    starter_share = max(0.0, min(1.0, (starter_proj_ip or 5.5) / 9))
    bullpen_share = 1 - starter_share
    opp_team_hits9 = get_team_pitching_hits9(opp_team_id, season) or LEAGUE_AVG_HITS9
    starter_adj = (opp_starter_hits9 / LEAGUE_AVG_HITS9) if opp_starter_hits9 else 1.0
    bullpen_adj = (opp_team_hits9 / LEAGUE_AVG_HITS9)
    hit_factor = starter_share * starter_adj + bullpen_share * bullpen_adj

    lam = blended_hpg * hit_factor
    lo = hi = 0
    cum = 0.0
    for i in range(30):
        cum += poisson_pmf(i, lam)
        if cum >= 0.10 and lo == 0:
            lo = i
        if cum >= 0.90:
            hi = i
            break

    l5_str = "·".join(str(h) for h in last5_hits) if last5_hits else "—"
    return {"lambda": round(lam, 2), "lo": lo, "hi": hi, "l5_str": l5_str,
            "last5_hits": last5_hits,
            "season_hpg": round(season_hpg, 2), "hit_factor": round(hit_factor, 2)}


def get_pitcher_data(pitcher_id, season):
    r = requests.get(f"{BASE}/people/{pitcher_id}/stats",
                      params={"stats": "season", "group": "pitching", "season": season})
    r.raise_for_status()
    sdata = r.json()
    splits = sdata.get("stats", [{}])[0].get("splits", [])
    if not splits:
        return None
    season_stat = splits[0]["stat"]

    r2 = requests.get(f"{BASE}/people/{pitcher_id}/stats",
                       params={"stats": "gameLog", "group": "pitching", "season": season})
    r2.raise_for_status()
    ldata = r2.json()
    lsplits = ldata.get("stats", [{}])[0].get("splits", [])
    starts = [s for s in lsplits if s["stat"].get("gamesStarted") in (1, "1")]
    starts.sort(key=lambda s: s["date"])
    last5 = starts[-5:]
    last5_parsed = []
    for s in last5:
        raw_ip = bb_ip_to_decimal(s["stat"]["inningsPitched"])
        # SANITY CLAMP: a single MLB start cannot exceed 9 innings (a
        # complete game) under any normal circumstance — modern bullpen
        # usage makes even that vanishingly rare. Observed live: a
        # corrupted/misparsed gameLog entry produced a ~17.8 IP average
        # for one pitcher, which then inflated both his K and outs
        # projections to physically impossible numbers (13+ projected
        # strikeouts against a real recent average of ~2/start). This
        # mirrors winsorize_iqr()'s outlier protection for team
        # runs/hits — that existed already, this same protection had
        # just never been applied to pitcher IP data specifically.
        ip = min(raw_ip, 9.0)
        if raw_ip > 9.0:
            print(f"    [!] {s['date']}: raw inningsPitched={s['stat']['inningsPitched']!r} "
                  f"parsed to {raw_ip} IP — implausible for a single start, clamped to 9.0. "
                  f"Worth checking this game's raw MLB Stats API data directly.")
        last5_parsed.append({"k": s["stat"]["strikeOuts"], "ip": ip, "date": s["date"]})
    return {"season": season_stat, "last5": last5_parsed}


def project(season_stat, last5, opp_k_pct, weight=RECENT_WEIGHT, bf_per_ip=BF_PER_IP):
    ip = bb_ip_to_decimal(season_stat["inningsPitched"])
    k = season_stat["strikeOuts"]
    gs = season_stat.get("gamesStarted") or 1
    bb9 = season_stat.get("walksPer9Inn")
    bb9 = float(bb9) if bb9 is not None else None

    season_bf = ip * bf_per_ip
    season_k_rate = k / season_bf if season_bf else 0

    recent_k_rate = season_k_rate
    avg_recent_ip = ip / gs
    if len(last5) >= 2:
        n = len(last5)
        wts = [1.4 ** i for i in range(n)]
        num_ = sum(w * s["k"] for w, s in zip(wts, last5))
        den_ = sum(w * (s["ip"] * bf_per_ip) for w, s in zip(wts, last5))
        recent_k_rate = num_ / den_ if den_ else season_k_rate
        avg_recent_ip = sum(s["ip"] for s in last5) / n

    # SANITY CLAMP, take two: the per-game clamp added in get_pitcher_data()
    # only guards the len(last5) >= 2 branch above. When a pitcher has
    # FEWER than 2 qualifying starts in this season's game log, this
    # function falls back to season IP / games started (line above) —
    # a completely different code path that never touches last5 at all,
    # so the earlier fix couldn't reach it. Observed live: this exact
    # fallback produced ~17.8 "IP per start" for a pitcher with very few
    # logged starts, surviving the first fix untouched. Clamping the
    # final avg_recent_ip here, after both branches resolve, covers
    # both paths with one guard instead of duplicating it.
    avg_recent_ip = min(avg_recent_ip, 9.0)

    blended = weight * recent_k_rate + (1 - weight) * season_k_rate
    opp_adj = (opp_k_pct / LEAGUE_AVG_K_PCT) if opp_k_pct else 1.0

    control_adj = 1.0
    if bb9 is not None:
        control_adj = 1 - max(0, bb9 - 3.0) * 0.03 + max(0, 2.0 - bb9) * 0.015
        control_adj = max(0.75, min(1.08, control_adj))

    proj_ip = avg_recent_ip * control_adj
    proj_bf = proj_ip * bf_per_ip
    adj_rate = blended * opp_adj
    lam = adj_rate * proj_bf

    lo = hi = 0
    cum = 0.0
    for i in range(40):
        cum += poisson_pmf(i, lam)
        if cum >= 0.10 and lo == 0:
            lo = i
        if cum >= 0.90:
            hi = i
            break

    l5_str = "·".join(str(s["k"]) for s in last5) if last5 else "—"
    l5_vs_season = ((recent_k_rate / season_k_rate - 1) * 100) if season_k_rate else 0

    # Outs-recorded projection reuses the same projected IP — outs = IP * 3.
    outs_lambda = proj_ip * 3
    outs_lo = outs_hi = 0
    cum = 0.0
    for i in range(60):
        cum += poisson_pmf(i, outs_lambda)
        if cum >= 0.10 and outs_lo == 0:
            outs_lo = i
        if cum >= 0.90:
            outs_hi = i
            break

    return {
        "lambda": round(lam, 2), "lo": lo, "hi": hi, "proj_ip": round(proj_ip, 1),
        "outs_lambda": round(outs_lambda, 2), "outs_lo": outs_lo, "outs_hi": outs_hi,
        "season_era": season_stat.get("era"), "season_k9": season_stat.get("strikeoutsPer9Inn"),
        "season_hits9": season_stat.get("hitsPer9Inn"),
        "bb9": bb9, "whip": season_stat.get("whip"),
        "l5_str": l5_str, "l5_vs_season": round(l5_vs_season, 0),
        "k_last5": [s["k"] for s in last5],
        "outs_last5": [round(s["ip"] * 3) for s in last5],
    }


def build_slate(target_date):
    season = target_date.year
    r = requests.get(f"{BASE}/schedule", params={
        "sportId": 1, "date": target_date.isoformat(), "hydrate": "probablePitcher,team"
    })
    r.raise_for_status()
    data = r.json()
    games = data.get("dates", [{}])[0].get("games", [])

    slate = []
    for g in games:
        away, home = g["teams"]["away"], g["teams"]["home"]
        game_time = g.get("gameDate", "")
        try:
            utc_dt = datetime.fromisoformat(game_time.replace("Z", "+00:00"))
            uk_dt = utc_dt.astimezone(UK_TZ)
            t = uk_dt.strftime("%I:%M %p").lstrip("0") + f" {uk_dt.tzname()}"
        except Exception:
            t = ""
        entry = {"away": away["team"]["name"], "home": home["team"]["name"], "time": t, "pitchers": []}

        for side_name, side, opp in (("away", away, home), ("home", home, away)):
            prob = side.get("probablePitcher")
            if not prob:
                entry["pitchers"].append({"side": side_name, "name": None})
                continue
            print(f"  Fetching {prob['fullName']}…")
            try:
                pdata = get_pitcher_data(prob["id"], season)
                opp_kpct = None
                try:
                    opp_kpct = get_team_k_pct(opp["team"]["id"], season)
                except Exception:
                    pass
                if pdata:
                    proj = project(pdata["season"], pdata["last5"], opp_kpct)
                    entry["pitchers"].append({
                        "side": side_name, "name": prob["fullName"],
                        "team": side["team"]["name"], "opp": opp["team"]["name"],
                        "opp_kpct": round(opp_kpct, 1) if opp_kpct else None,
                        **proj
                    })
                else:
                    entry["pitchers"].append({"side": side_name, "name": prob["fullName"], "no_stats": True})
            except Exception as e:
                entry["pitchers"].append({"side": side_name, "name": prob["fullName"], "error": str(e)})

        slate.append(entry)

        # Team run + hit projections: each team's offense vs. the OPPOSING starter (+ that team's bullpen)
        entry["team_runs"] = {}
        entry["team_hits"] = {}
        pitcher_by_side = {p.get("side"): p for p in entry["pitchers"]}
        for side_name, side, opp in (("away", away, home), ("home", home, away)):
            opp_side = "home" if side_name == "away" else "away"
            opp_pitcher = pitcher_by_side.get(opp_side, {})
            opp_era = opp_pitcher.get("season_era")
            opp_era = float(opp_era) if opp_era not in (None, "-") else None
            opp_hits9 = opp_pitcher.get("season_hits9")
            opp_hits9 = float(opp_hits9) if opp_hits9 not in (None, "-") else None
            opp_proj_ip = opp_pitcher.get("proj_ip")

            try:
                tr = project_team_runs(
                    side["team"]["id"], season, opp_era, opp_proj_ip, opp["team"]["id"]
                )
                if tr:
                    entry["team_runs"][side_name] = {
                        "team": side["team"]["name"], "opp": opp["team"]["name"], **tr
                    }
            except Exception as e:
                entry["team_runs"][side_name] = {"team": side["team"]["name"], "error": str(e)}

            try:
                th = project_team_hits(
                    side["team"]["id"], season, opp_hits9, opp_proj_ip, opp["team"]["id"]
                )
                if th:
                    entry["team_hits"][side_name] = {
                        "team": side["team"]["name"], "opp": opp["team"]["name"], **th
                    }
            except Exception as e:
                entry["team_hits"][side_name] = {"team": side["team"]["name"], "error": str(e)}

    return slate


HTML_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Strike Zone — Slate for {date}</title>
<style>
  :root{{--bg:#0b0f14; --panel:#121820; --panel2:#161d27; --border:#233040; --text:#e8edf2; --sub:#8b98a8; --yellow:#facc15; --green:#22c55e;}}
  body{{margin:0; background:var(--bg); color:var(--text); font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; padding:16px; max-width:640px; margin:0 auto;}}
  h1{{font-size:20px; margin-bottom:4px;}}
  .sub{{color:var(--sub); font-size:13px; margin-bottom:18px;}}
  .gameGroup{{margin-bottom:14px; border:1px solid var(--border); border-radius:12px; overflow:hidden; background:var(--panel);}}
  .gameHead{{background:var(--panel2); padding:10px 14px; font-size:13px; color:var(--sub); display:flex; justify-content:space-between;}}
  .pitcherRow{{display:flex; flex-direction:column; gap:8px; padding:12px 14px; border-top:1px solid var(--border);}}
  .pTop{{display:flex; justify-content:space-between; align-items:center;}}
  .pName{{font-weight:700; font-size:15px;}}
  .pMeta{{font-size:11px; color:var(--sub); margin-top:2px;}}
  .pProj{{text-align:right;}}
  .pProjNum{{font-size:20px; font-weight:800; color:var(--yellow);}}
  .pProjSub{{font-size:11px; color:var(--sub);}}
  .noPitcher{{padding:12px 14px; color:var(--sub); font-size:12px; font-style:italic;}}
  .footnote{{font-size:11px; color:var(--sub); text-align:center; margin-top:20px; line-height:1.6;}}
  .edgeRow{{display:flex; gap:6px; flex-wrap:wrap; align-items:center; border-top:1px dashed var(--border); padding-top:8px;}}
  .edgeRow input{{width:76px; background:var(--panel2); border:1px solid var(--border); color:var(--text); border-radius:6px; padding:6px 4px; font-size:12px; -moz-appearance:textfield; appearance:textfield;}}
  .edgeRow input::-webkit-outer-spin-button, .edgeRow input::-webkit-inner-spin-button{{-webkit-appearance:none; margin:0;}}
  .edgeRow input::placeholder{{color:#5a6676;}}
  .edgeBtn{{background:var(--green); color:#04140a; border:none; border-radius:6px; padding:6px 12px; font-size:12px; font-weight:700; cursor:pointer;}}
  .edgeOut{{font-size:12px; font-weight:700; color:var(--sub); flex-basis:100%;}}
  .propLabel{{font-size:11px; color:var(--sub); text-transform:uppercase; letter-spacing:.04em; margin-top:4px;}}
  .edgeBadge{{display:inline-block; font-size:10px; font-weight:800; letter-spacing:.03em; padding:2px 7px; border-radius:5px; margin-right:4px;}}
  .badge-skip{{background:#232b35; color:var(--sub);}}
  .badge-lean{{background:#2a2a12; color:var(--yellow);}}
  .badge-play{{background:var(--green-dim, #16321f); color:var(--green);}}
  .badge-verify{{background:#3a1a1a; color:var(--red, #ef4444);}}
  .topBar{{display:flex; justify-content:space-between; align-items:center; margin-bottom:14px; flex-wrap:wrap; gap:8px;}}
  .downloadBtn{{background:var(--panel2); border:1px solid var(--border); color:var(--text); font-size:13px; font-weight:600; padding:8px 14px; border-radius:8px; cursor:pointer;}}
  .downloadBtn:active{{opacity:.8;}}
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
</style></head>
<body>
<div class="topBar">
  <div>
    <h1>⚾ Strike Zone — Daily Slate</h1>
    <div class="sub">{date} · generated {generated}</div>
  </div>
  <button class="downloadBtn" onclick="exportCSV()">⬇ Download CSV</button>
</div>
{builder_html}
{games_html}
<div class="footnote">Projections blend season K-rate/batter-faced with a recency-weighted last-5 rate ({weight}% recent), adjust for opponent K% and BB/9-driven outing length, then use a Poisson distribution for the range. Enter a book's line/odds under any pitcher to compute a de-vigged edge — that math runs entirely in your browser, no data leaves the page.</div>
<script>
const REPORT_DATE = "{date}";
const LEGS = {legs_json};

function poissonCDF(threshold, lambda){{
  let p = Math.exp(-lambda), cum = p;
  for(let i=1;i<=threshold;i++){{ p = p*lambda/i; cum += p; }}
  return cum;
}}
function classifyEdge(kind, bestEdge){{
  // Pitcher props (K, outs) get more benefit of the doubt than whole-team
  // props (runs, hits) — a whole-lineup stat has more blind spots, so it
  // needs a bigger edge to earn the same confidence label.
  const isTeamProp = (kind === 'runs_lambda' || kind === 'hits_lambda');
  const t = isTeamProp ? {{skip:8, lean:15, play:25}} : {{skip:5, lean:12, play:20}};
  if(bestEdge < t.skip)  return {{label:'SKIP',    cls:'badge-skip'}};
  if(bestEdge < t.lean)  return {{label:'LEAN',    cls:'badge-lean'}};
  if(bestEdge < t.play)  return {{label:'PLAY',    cls:'badge-play'}};
  return {{label:'⚠ VERIFY', cls:'badge-verify'}};
}}

function calcEdge(btn){{
  const kind = btn.dataset.kind;
  const row = btn.closest('.pitcherRow');
  const lambda = parseFloat(row.dataset[kind]);
  const wrap = btn.closest('.edgeRow');
  const line = parseFloat(wrap.querySelector('.lineInput').value);
  const overOdds = parseFloat(wrap.querySelector('.overInput').value);
  const underOdds = parseFloat(wrap.querySelector('.underInput').value);
  const out = wrap.querySelector('.edgeOut');
  if(isNaN(lambda) || isNaN(line)){{ out.textContent = 'Enter a line first.'; out.innerHTML = out.textContent; return; }}
  const threshold = Math.floor(line);
  const pUnder = poissonCDF(threshold, lambda);
  const pOver = 1 - pUnder;
  let text = `Model: Over ${{(pOver*100).toFixed(1)}}% · Under ${{(pUnder*100).toFixed(1)}}%`;
  let badgeHtml = '';
  if(!isNaN(overOdds) && !isNaN(underOdds) && overOdds>0 && underOdds>0){{
    const rawOver = 1/overOdds, rawUnder = 1/underOdds;
    const overround = rawOver + rawUnder;
    const mktOver = rawOver/overround, mktUnder = rawUnder/overround;
    const edgeOver = (pOver - mktOver)*100;
    const edgeUnder = (pUnder - mktUnder)*100;
    const bestEdge = Math.max(edgeOver, edgeUnder);
    const pick = edgeOver >= edgeUnder
      ? `Over edge ${{edgeOver>=0?'+':''}}${{edgeOver.toFixed(1)}}%`
      : `Under edge ${{edgeUnder>=0?'+':''}}${{edgeUnder.toFixed(1)}}%`;
    text += ` · ${{pick}}`;
    const {{label, cls}} = classifyEdge(kind, bestEdge);
    badgeHtml = `<span class="edgeBadge ${{cls}}">${{label}}</span> `;
    out.style.color = bestEdge >= 8 ? 'var(--green)' : (bestEdge >= 3 ? 'var(--yellow)' : 'var(--sub)');
  }} else {{
    out.style.color = 'var(--sub)';
  }}
  out.innerHTML = badgeHtml + text.replace(/&/g,'&amp;').replace(/</g,'&lt;');
}}

function csvEscape(v){{
  return `"${{String(v==null?'':v).replace(/"/g,'""')}}"`;
}}

function exportCSV(){{
  const header = ['Date','Matchup','GameTime','PlayerOrTeam','PropType','ProjectedMean','Line','OverOdds','UnderOdds','EdgeLabel','ModelResult','ActualResult','HitOrMiss'];
  const rows = [header];
  document.querySelectorAll('.gameGroup').forEach(game => {{
    const spans = game.querySelectorAll('.gameHead span');
    const matchup = spans[0] ? spans[0].textContent.trim() : '';
    const gameTime = spans[1] ? spans[1].textContent.trim() : '';
    game.querySelectorAll('.pitcherRow').forEach(row => {{
      const nameEl = row.querySelector('.pName');
      const name = nameEl ? nameEl.textContent.trim() : '';
      row.querySelectorAll('.edgeRow').forEach(er => {{
        const btn = er.querySelector('.edgeBtn');
        const kind = btn ? btn.dataset.kind : '';
        const lam = kind ? row.dataset[kind] : '';
        const labelEl = er.previousElementSibling;
        const propLabel = labelEl ? labelEl.textContent.split('(')[0].trim() : '';
        const line = er.querySelector('.lineInput').value;
        const overOdds = er.querySelector('.overInput').value;
        const underOdds = er.querySelector('.underInput').value;
        const modelResultEl = er.querySelector('.edgeOut');
        const badgeEl = modelResultEl ? modelResultEl.querySelector('.edgeBadge') : null;
        const edgeLabel = badgeEl ? badgeEl.textContent.trim() : '';
        const modelResult = modelResultEl ? modelResultEl.textContent.replace(edgeLabel, '').trim() : '';
        rows.push([REPORT_DATE, matchup, gameTime, name, propLabel, lam, line, overOdds, underOdds, edgeLabel, modelResult, '', '']);
      }});
    }});
  }});
  const csv = rows.map(r => r.map(csvEscape).join(',')).join('\\n');
  const blob = new Blob([csv], {{type:'text/csv;charset=utf-8;'}});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `strike_zone_${{REPORT_DATE}}.csv`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}}

// ---- Safest Bet Builder ----
function initBuilderToggles() {{
  const container = document.getElementById('builderToggles');
  if (!container) return;
  const cats = [...new Set(LEGS.map(l => l.category))];
  container.innerHTML = cats.map(c => `
    <label><input type="checkbox" class="szCatToggle" value="${{c}}" checked> ${{c}}</label>
  `).join('');
}}
function szShuffle(arr) {{
  for (let i = arr.length - 1; i > 0; i--) {{
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }}
  return arr;
}}
function szTieredShuffle(legs, bandSize) {{
  const bands = {{}};
  legs.forEach(l => {{
    const band = Math.floor(l.prob / bandSize);
    (bands[band] = bands[band] || []).push(l);
  }});
  const bandKeys = Object.keys(bands).map(Number).sort((a, b) => b - a);
  let result = [];
  bandKeys.forEach(b => {{ result = result.concat(szShuffle(bands[b])); }});
  return result;
}}
function buildSafestSZ() {{
  const target = parseFloat(document.getElementById('szTargetOdds').value) || 5.0;
  const maxLegs = parseInt(document.getElementById('szMaxLegs').value) || 8;
  const activeCats = [...document.querySelectorAll('.szCatToggle:checked')].map(el => el.value);

  const byCategory = {{}};
  LEGS.filter(l => l.prob > 0 && activeCats.includes(l.category)).forEach(l => {{
    (byCategory[l.category] = byCategory[l.category] || []).push(l);
  }});
  const categories = Object.keys(byCategory);
  categories.forEach(c => {{ byCategory[c] = szTieredShuffle(byCategory[c], 5); }});
  const cursor = {{}};
  categories.forEach(c => cursor[c] = 0);

  const chosen = [];
  const subjectCount = {{}};  // capped per pitcher/team, not per game
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
      <span>${{l.match}}<br><span style="color:var(--yellow)">${{l.market}}</span> <span style="color:var(--sub)">· ${{l.category}}</span>
      ${{l.detail ? `<br><span style="color:var(--sub);font-size:10px">${{l.detail}}</span>` : ''}}
      ${{l.history ? `<br><span style="color:var(--sub);font-size:10px">last games: ${{l.history}}</span>` : ''}}</span>
      <span style="text-align:right"><span style="color:var(--yellow);font-weight:bold">${{l.prob}}%</span>${{l.hit_rate ? `<br><span style="color:var(--sub);font-size:11px">${{l.hit_rate.hits}}/${{l.hit_rate.total}}</span>` : ''}}</span>
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
      include their margin and legs from the same pitcher/team aren't fully independent,
      so treat this as a ranking tool, not a firm price. All lines are set automatically
      below the model's projection for a safety margin.
    </div>
  `;
}}
initBuilderToggles();
</script>
</body></html>
"""

BUILDER_TEMPLATE = """<div class="builderPanel">
  <div class="builderTitle">🎯 Safest Bet Builder</div>
  <div class="builderToggles" id="builderToggles"></div>
  <div class="builderControls">
    <label>Target odds:</label>
    <input type="number" step="0.1" min="1.1" value="5.0" id="szTargetOdds">
    <label>Max legs:</label>
    <input type="number" step="1" min="2" value="8" id="szMaxLegs">
    <button class="builderBtn" onclick="buildSafestSZ()">Build</button>
    <button class="builderBtnAlt" onclick="buildSafestSZ()">🔀 Shuffle</button>
  </div>
  <div class="builderResult" id="builderResult">
    Untick any market you don't want considered, set a target odds and leg cap, then tap
    Build. Rotates through ticked categories, groups near-tied legs and shuffles within
    each group so it draws from more of today's slate rather than the same few
    pitchers/teams, and caps at 2 legs per pitcher/team to avoid stacking correlated
    lines (e.g. Strikeouts and Outs Recorded for the same starter). Tap Shuffle for a
    fresh pick without changing your settings.
  </div>
</div>"""

GAME_TEMPLATE = """<div class="gameGroup">
  <div class="gameHead"><span>{away} @ {home}</span><span>{time}</span></div>
  {team_run_rows}
  {pitcher_rows}
</div>"""

TEAM_RUN_ROW = """<div class="pitcherRow" data-runs_lambda="{lam}">
  <div class="pTop">
    <div>
      <div class="pName">{team} — Total Runs</div>
      <div class="pMeta">vs {opp} · L5 runs: {l5_str} · season {season_rpg}/gm · pitching-adj ×{run_factor}</div>
    </div>
    <div class="pProj">
      <div class="pProjNum">{lam}</div>
      <div class="pProjSub">{lo}–{hi} range</div>
    </div>
  </div>
  <div class="propLabel">Team Total Runs (full game)</div>
  <div class="edgeRow">
    <input type="number" step="0.5" class="lineInput" placeholder="Line">
    <input type="number" step="0.01" class="overInput" placeholder="Over odds">
    <input type="number" step="0.01" class="underInput" placeholder="Under odds">
    <button class="edgeBtn" data-kind="runs_lambda" onclick="calcEdge(this)">Edge</button>
    <div class="edgeOut"></div>
  </div>
</div>"""

TEAM_HIT_ROW = """<div class="pitcherRow" data-hits_lambda="{lam}">
  <div class="pTop">
    <div>
      <div class="pName">{team} — Total Hits</div>
      <div class="pMeta">vs {opp} · L5 hits: {l5_str} · season {season_hpg}/gm · pitching-adj ×{hit_factor}</div>
    </div>
    <div class="pProj">
      <div class="pProjNum">{lam}</div>
      <div class="pProjSub">{lo}–{hi} range</div>
    </div>
  </div>
  <div class="propLabel">Team Total Hits (full game)</div>
  <div class="edgeRow">
    <input type="number" step="0.5" class="lineInput" placeholder="Line">
    <input type="number" step="0.01" class="overInput" placeholder="Over odds">
    <input type="number" step="0.01" class="underInput" placeholder="Under odds">
    <button class="edgeBtn" data-kind="hits_lambda" onclick="calcEdge(this)">Edge</button>
    <div class="edgeOut"></div>
  </div>
</div>"""

PITCHER_ROW = """<div class="pitcherRow" data-lambda="{lam}" data-outs_lambda="{outs_lam}">
  <div class="pTop">
    <div>
      <div class="pName">{name}</div>
      <div class="pMeta">{team} vs {opp} · L5 Ks: {l5_str} ({l5_delta}) · BB/9 {bb9}</div>
    </div>
    <div class="pProj">
      <div class="pProjNum">{lam}</div>
      <div class="pProjSub">{lo}–{hi} range · {proj_ip} IP</div>
    </div>
  </div>
  <div class="propLabel">Strikeouts</div>
  <div class="edgeRow">
    <input type="number" step="0.5" class="lineInput" placeholder="Line">
    <input type="number" step="0.01" class="overInput" placeholder="Over odds">
    <input type="number" step="0.01" class="underInput" placeholder="Under odds">
    <button class="edgeBtn" data-kind="lambda" onclick="calcEdge(this)">Edge</button>
    <div class="edgeOut"></div>
  </div>
  <div class="propLabel">Outs Recorded <span class="pProjSub">(proj {outs_lam} · {outs_lo}–{outs_hi} range)</span></div>
  <div class="edgeRow">
    <input type="number" step="0.5" class="lineInput" placeholder="Line">
    <input type="number" step="0.01" class="overInput" placeholder="Over odds">
    <input type="number" step="0.01" class="underInput" placeholder="Under odds">
    <button class="edgeBtn" data-kind="outs_lambda" onclick="calcEdge(this)">Edge</button>
    <div class="edgeOut"></div>
  </div>
</div>"""

NO_PITCHER_ROW = """<div class="noPitcher">Probable pitcher not yet announced</div>"""


# ----------------------------------------------------------------------
# Safest Bet Builder — same pattern as Match IQ/Blitz IQ/Blue Line/Player
# Stat Model: every prop the model prices gets a safety-margin line set
# below the projection, turned into a real probability plus an empirical
# hit-rate from the same raw last-5 values, flattened into legs the HTML
# panel round-robins through. Capped per PITCHER/TEAM (a pitcher's own
# Strikeouts and Outs Recorded legs are obviously correlated — same
# outing), not per game the way Match IQ caps per match.
# ----------------------------------------------------------------------

def safe_line(lam, factor=0.72, round_to=0.5):
    if lam is None:
        return None
    raw = lam * factor
    line = math.floor(raw / round_to) * round_to
    if line < round_to:
        line = round_to
    return line


def prob_over(lam, line):
    threshold = math.floor(line) + 1
    cum = sum(poisson_pmf(i, lam) for i in range(threshold))
    return 1 - cum


def hit_rate(values, line):
    """(hits, total) from the same raw last-5 values feeding the
    projection — a model-free empirical cross-check, same idea as the
    other predictor tools' hit-rate."""
    if not values:
        return None
    hits = sum(1 for v in values if v > line)
    return {"hits": hits, "total": len(values)}


def build_legs(slate):
    legs = []
    for g in slate:
        match_label = f"{g['away']} @ {g['home']}"
        for p in g.get("pitchers", []):
            if not p.get("name") or p.get("error") or p.get("no_stats"):
                continue
            k_line = safe_line(p.get("lambda"))
            if k_line:
                legs.append({
                    "match": match_label, "subject": p["name"],
                    "market": f"{p['name']} Over {k_line} Strikeouts",
                    "prob": round(prob_over(p["lambda"], k_line) * 100),
                    "category": "Strikeouts",
                    "hit_rate": hit_rate(p.get("k_last5"), k_line),
                    "detail": f"{p['team']} vs {p['opp']} · proj {p['lambda']} K",
                    "history": "/".join(str(v) for v in p.get("k_last5", [])) or None,
                })
            outs_line = safe_line(p.get("outs_lambda"))
            if outs_line:
                legs.append({
                    "match": match_label, "subject": p["name"],
                    "market": f"{p['name']} Over {outs_line} Outs Recorded",
                    "prob": round(prob_over(p["outs_lambda"], outs_line) * 100),
                    "category": "Outs Recorded",
                    "hit_rate": hit_rate(p.get("outs_last5"), outs_line),
                    "detail": f"{p['team']} vs {p['opp']} · proj {p['proj_ip']} IP",
                    "history": "/".join(str(v) for v in p.get("outs_last5", [])) or None,
                })

        for side in ("away", "home"):
            tr = g.get("team_runs", {}).get(side)
            if tr and "lambda" in tr:
                line = safe_line(tr["lambda"])
                if line:
                    legs.append({
                        "match": match_label, "subject": tr["team"],
                        "market": f"{tr['team']} Over {line} Runs",
                        "prob": round(prob_over(tr["lambda"], line) * 100),
                        "category": "Team Runs",
                        "hit_rate": hit_rate(tr.get("last5_runs"), line),
                        "detail": f"vs {tr['opp']} · proj {tr['lambda']} runs",
                        "history": "/".join(str(v) for v in tr.get("last5_runs", [])) or None,
                    })
            th = g.get("team_hits", {}).get(side)
            if th and "lambda" in th:
                line = safe_line(th["lambda"])
                if line:
                    legs.append({
                        "match": match_label, "subject": th["team"],
                        "market": f"{th['team']} Over {line} Hits",
                        "prob": round(prob_over(th["lambda"], line) * 100),
                        "category": "Team Hits",
                        "hit_rate": hit_rate(th.get("last5_hits"), line),
                        "detail": f"vs {th['opp']} · proj {th['lambda']} hits",
                        "history": "/".join(str(v) for v in th.get("last5_hits", [])) or None,
                    })
    return legs


def render_html(slate, target_date):
    games_html = []
    for g in slate:
        rows = []
        for p in g["pitchers"]:
            if not p.get("name"):
                rows.append(NO_PITCHER_ROW)
            elif p.get("error") or p.get("no_stats"):
                rows.append(f'<div class="noPitcher">{p["name"]}: no stats available</div>')
            else:
                delta = f'{"+" if p["l5_vs_season"]>=0 else ""}{p["l5_vs_season"]:.0f}% vs season'
                rows.append(PITCHER_ROW.format(
                    name=p["name"], team=p["team"], opp=p["opp"],
                    l5_str=p["l5_str"], l5_delta=delta,
                    bb9=p["bb9"] if p["bb9"] is not None else "—",
                    lam=p["lambda"], lo=p["lo"], hi=p["hi"], proj_ip=p["proj_ip"],
                    outs_lam=p["outs_lambda"], outs_lo=p["outs_lo"], outs_hi=p["outs_hi"],
                ))

        team_rows = []
        for side_name in ("away", "home"):
            tr = g.get("team_runs", {}).get(side_name)
            if tr and "lambda" in tr:
                team_rows.append(TEAM_RUN_ROW.format(
                    team=tr["team"], opp=tr["opp"], l5_str=tr["l5_str"],
                    season_rpg=tr["season_rpg"], run_factor=tr["run_factor"],
                    lam=tr["lambda"], lo=tr["lo"], hi=tr["hi"],
                ))
            th = g.get("team_hits", {}).get(side_name)
            if th and "lambda" in th:
                team_rows.append(TEAM_HIT_ROW.format(
                    team=th["team"], opp=th["opp"], l5_str=th["l5_str"],
                    season_hpg=th["season_hpg"], hit_factor=th["hit_factor"],
                    lam=th["lambda"], lo=th["lo"], hi=th["hi"],
                ))

        games_html.append(GAME_TEMPLATE.format(
            away=g["away"], home=g["home"], time=g["time"],
            team_run_rows="".join(team_rows), pitcher_rows="".join(rows)
        ))

    legs = build_legs(slate)
    builder_html = BUILDER_TEMPLATE if legs else ""

    return HTML_TEMPLATE.format(
        date=target_date.isoformat(), generated=datetime.now().strftime("%Y-%m-%d %H:%M"),
        games_html="".join(games_html), weight=int(RECENT_WEIGHT * 100),
        builder_html=builder_html, legs_json=json.dumps(legs),
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = date.fromisoformat(sys.argv[1])
    else:
        target = date.today()

    print(f"Fetching slate for {target.isoformat()}…")
    slate = build_slate(target)
    html = render_html(slate, target)

    out_path = "docs/strike-zone/index.html"
    import os
    os.makedirs("docs/strike-zone", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(html)
    print(f"\nDone. {len(slate)} games written to {out_path} — open it in your browser.")

    # also dump raw JSON in case you want to feed it into another tool
    with open("docs/strike-zone/slate_report.json", "w") as f:
        json.dump(slate, f, indent=2, default=str)
