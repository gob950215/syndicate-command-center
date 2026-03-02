#!/usr/bin/env python3
"""
NBA Training Pipeline V7 — THE GOD MODE
=========================================
Building on V6 (72.1 % VIP accuracy), this version targets 85 %+ via
five surgical innovations:

  1. Matchup DNA          — Ataque-vs-defensa segmentado por tipo de jugada
  2. Chronic Fatigue Index — Carga de viaje acumulada 7 d + zonas horarias
  3. Referee/Value Trap    — FT-dependencia vs rival low-foul + eFG penalty
  4. Elite VIP Threshold   — ≥ 78 % confianza AND EV > 0.10
  5. Risk-Level CSV        — Volatilidad MC → Bajo / Medio / Alto

Requisitos: pip install numpy pandas scikit-learn xgboost
"""

import os, sys, time, sqlite3, logging, argparse, warnings, pickle, csv
from datetime import datetime, timedelta
from collections import defaultdict
from math import radians, cos, sin, asin, sqrt

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
    from sklearn.metrics import accuracy_score, brier_score_loss
    from sklearn.preprocessing import StandardScaler
except ImportError:
    print("pip install xgboost scikit-learn"); sys.exit(1)

warnings.filterwarnings("ignore")

# ═══════════════════════ CONFIG ══════════════════════════════════════════════
DB_PATH        = "data/nba_historical.db"
MODEL_DIR      = "models"
PICKS_CSV      = "picks_profesionales.csv"
CHECKPOINT     = 10
DEFAULT_SIMS   = 10_000
VIP_THRESHOLD  = 0.78          # V7: más estricto que V6 (0.72)
VIP_FALLBACK   = 0.65          # V7: fallback más alto (era 0.60)
MIN_EV         = 0.10          # V7: EV mínimo positivo exigible
MKT_GAP_MIN    = 0.08
FP_PENALTY     = 5

TRAIN_SEASONS  = ["2019-2020", "2020-2021", "2021-2022", "2022-2023", "2023-2024"]
EVAL_SEASON    = "2024-2025"

ELO_INIT = 1500; ELO_K = 20; ELO_HCA = 55

# Matchup DNA thresholds (percentile ranks converted to top-5 = top 16.7%)
MATCHUP_TOP5_RANK = 5          # top 5 out of 30 teams
MATCHUP_PENALTY   = 0.07       # 7 % wp reduction

# Chronic fatigue
TIMEZONE_PENALTY_THRESHOLD = 2   # ≥ 2 time zone crossings
HEAVY_LEGS_GAMES_IN_DAYS   = (4, 6)  # 4 games in 6 days
HEAVY_LEGS_EFG_PENALTY     = 0.015   # eFG absolute reduction

# Value Trap
FT_DEPENDENCY_THRESHOLD = 0.32       # ft_rate > 0.32 → FT dependent
LOW_FOUL_RANK_THRESHOLD = 5          # top 5 lowest-fouling team

os.makedirs("data", exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(
            stream=open(sys.stdout.fileno(), mode="w", encoding="utf-8", closefd=False)
        ),
        logging.FileHandler("data/training_log_v7.txt", "a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("V7-GodMode")

# ═══════════════════════ TEAM CONSTANTS ══════════════════════════════════════
TEAM_ABBR = {
    132:"ATL",133:"BOS",134:"BKN",135:"CHA",136:"CHI",137:"CLE",138:"DAL",
    139:"DEN",140:"DET",141:"GSW",142:"HOU",143:"IND",144:"LAC",145:"LAL",
    146:"MEM",147:"MIA",148:"MIL",149:"MIN",150:"NOP",151:"NYK",152:"OKC",
    153:"ORL",154:"PHI",155:"PHX",156:"POR",157:"SAC",158:"SAS",159:"TOR",
    160:"UTA",161:"WAS",
}
NBA_IDS = {
    132:1610612737,133:1610612738,134:1610612751,135:1610612766,136:1610612741,
    137:1610612739,138:1610612742,139:1610612743,140:1610612765,141:1610612744,
    142:1610612745,143:1610612754,144:1610612746,145:1610612747,146:1610612763,
    147:1610612748,148:1610612749,149:1610612750,150:1610612740,151:1610612752,
    152:1610612760,153:1610612753,154:1610612755,155:1610612756,156:1610612757,
    157:1610612758,158:1610612759,159:1610612761,160:1610612762,161:1610612764,
}
REV_TEAM = {v: k for k, v in NBA_IDS.items()}
COORDS = {
    132:(33.757,-84.396),133:(42.366,-71.062),134:(40.683,-73.975),
    135:(35.225,-80.839),136:(41.881,-87.674),137:(41.497,-81.688),
    138:(32.791,-96.810),139:(39.749,-105.008),140:(42.341,-83.055),
    141:(37.768,-122.388),142:(29.751,-95.362),143:(39.764,-86.156),
    144:(34.043,-118.267),145:(34.043,-118.267),146:(35.138,-90.051),
    147:(25.781,-80.188),148:(43.044,-87.917),149:(44.980,-93.276),
    150:(29.949,-90.082),151:(40.751,-73.993),152:(35.463,-97.515),
    153:(28.539,-81.384),154:(39.901,-75.172),155:(33.446,-112.071),
    156:(45.532,-122.667),157:(38.580,-121.500),158:(29.427,-98.438),
    159:(43.644,-79.379),160:(40.768,-111.901),161:(38.898,-77.021),
}
EAST = {132,133,134,135,136,137,140,143,147,148,149,151,153,154,159,161}

# Approximate timezone offsets (hours from UTC) for travel-zone calculation
TEAM_TZ = {
    132:-5,133:-5,134:-5,135:-5,136:-6,137:-5,138:-6,139:-7,140:-5,
    141:-8,142:-6,143:-5,144:-8,145:-8,146:-6,147:-5,148:-6,149:-6,
    150:-6,151:-5,152:-6,153:-5,154:-5,155:-7,156:-8,157:-8,158:-6,
    159:-5,160:-7,161:-5,
}


def _haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    d = lat2 - lat1; dl = lon2 - lon1
    a = sin(d/2)**2 + cos(lat1)*cos(lat2)*sin(dl/2)**2
    return 2 * 3956 * asin(sqrt(a))


# ═══════════════════════ DATA LOADER ═════════════════════════════════════════
class DataLoader:
    def __init__(self, db=DB_PATH):
        self.conn = sqlite3.connect(db)
        self.conn.row_factory = sqlite3.Row

    def load_games(self, seasons):
        ph = ",".join("?" * len(seasons))
        df = pd.read_sql_query(
            f"""SELECT game_id, season, date_local AS game_date,
                       home_team_id, away_team_id, home_score, away_score,
                       home_q1, home_q2, home_q3, home_q4,
                       away_q1, away_q2, away_q3, away_q4, nba_game_id
                FROM games
                WHERE season IN ({ph})
                  AND status_short IN ('FT','AOT')
                  AND home_score IS NOT NULL AND away_score IS NOT NULL
                ORDER BY date_local, game_id""",
            self.conn, params=seasons,
        )
        df["game_date"] = pd.to_datetime(df["game_date"])
        df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
        df["margin"] = df["home_score"] - df["away_score"]
        for side in ("home", "away"):
            q4 = df[f"{side}_q4"].fillna(0).astype(float)
            opp_q4 = df[f"{'away' if side == 'home' else 'home'}_q4"].fillna(0).astype(float)
            df[f"{side}_q4_net"] = q4 - opp_q4
        return df

    def load_boxscores(self):
        df = pd.read_sql_query(
            """SELECT nba_game_id, team_id AS nba_tid, game_id,
                      fgm, fga, fg3m, fg3a, ftm, fta,
                      oreb, dreb, reb, ast, stl, blk, tov, pf, pts
               FROM game_team_stats WHERE pts IS NOT NULL""",
            self.conn,
        )
        df["team_id"] = df["nba_tid"].map(REV_TEAM)
        df["poss"] = 0.96 * (df["fga"] + 0.44 * df["fta"] - df["oreb"] + df["tov"])
        return df

    def load_players(self):
        df = pd.read_sql_query(
            """SELECT nba_game_id, player_id, team_id AS nba_tid, game_id,
                      player_name, minutes_decimal,
                      (CASE WHEN minutes_decimal > 25 THEN 1 ELSE 0 END) AS starter,
                      fgm, fga, fg3m, fg3a, ftm, fta,
                      oreb, dreb, reb, ast, stl, blk, tov, pf, pts, plus_minus
               FROM game_player_stats WHERE minutes_decimal > 0""",
            self.conn,
        )
        df["team_id"] = df["nba_tid"].map(REV_TEAM)
        return df

    def load_odds(self):
        return pd.read_sql_query(
            """SELECT game_id_mapped AS game_id, home_team, away_team,
                      bookmaker, market, outcome_name, outcome_price, outcome_point
               FROM odds_historical WHERE game_id_mapped IS NOT NULL""",
            self.conn,
        )

    def close(self):
        self.conn.close()


# ═══════════════════════ FEATURE VECTOR ══════════════════════════════════════
FEAT = [
    # --- FOUR FACTORS (10-game, per 100 poss) ---
    "h_efg", "a_efg",
    "h_tov_pct", "a_tov_pct",
    "h_oreb_pct", "a_oreb_pct",
    "h_ft_rate", "a_ft_rate",
    "diff_efg", "diff_tov", "diff_oreb", "diff_ft",

    # --- Hot-streak eFG (5-game + trend) ---
    "h_efg_hot5", "a_efg_hot5",
    "h_efg_trend", "a_efg_trend",

    # --- Pace & Ratings per 100 poss ---
    "h_pace", "a_pace",
    "h_ortg", "a_ortg",
    "h_drtg", "a_drtg",
    "h_net_rtg", "a_net_rtg",

    # --- V7: MATCHUP DNA ---
    "matchup_ortg_vs_drtg",       # h_ortg − a_drtg (home attack vs away defense)
    "matchup_drtg_vs_ortg",       # h_drtg − a_ortg (home defense vs away attack)
    "h_ast_rate", "a_ast_rate",   # asistencias / FGM (flow del ataque)
    "h_stl_rate", "a_stl_rate",   # robos / posesión rival (disrupción defensiva)
    "matchup_ast_vs_stl",         # señal de clash: high-ast ataque vs high-stl defensa
    "matchup_dna_penalty",        # 1 si se detecta trampa de matchup DNA, 0 si no

    # --- FATIGA: V7 CHRONIC INDEX ---
    "h_games_7d", "a_games_7d",
    "h_b2b", "a_b2b",
    "h_travel_miles_7d", "a_travel_miles_7d",  # V7: acumulado 7 días (no solo último)
    "h_rest_days", "a_rest_days",
    "h_road_trip_len", "a_road_trip_len",
    "h_3in4", "a_3in4",
    "h_4in6", "a_4in6",                        # V7: 4 partidos en 6 días
    "h_tz_crossed_7d", "a_tz_crossed_7d",      # V7: zonas horarias cruzadas
    "h_heavy_legs", "a_heavy_legs",             # V7: flag compuesto de fatiga crónica

    # --- IMPACTO DE AUSENCIAS ---
    "h_missing_net_rtg", "a_missing_net_rtg",
    "h_missing_min", "a_missing_min",
    "h_missing_stars", "a_missing_stars",

    # --- TOP-8 EFICIENCIA ---
    "h_top8_off_rtg", "a_top8_off_rtg",
    "h_top8_def_rtg", "a_top8_def_rtg",
    "h_top8_net_rtg", "a_top8_net_rtg",
    "h_top8_consistency", "a_top8_consistency",

    # --- CLUTCH Q4 ---
    "h_q4_net_avg", "a_q4_net_avg",

    # --- V7: VALUE TRAP (Referee/FT dependencia) ---
    "h_ft_dependency", "a_ft_dependency",       # ft_rate normalizado (alto = depende de FT)
    "h_opp_pf_rate", "a_opp_pf_rate",          # faltas que provoca el rival (proxy)
    "value_trap_flag",                           # 1 si favorito depende de FT y rival no faulea

    # --- SMART MONEY ---
    "mkt_prob_home", "mkt_spread",
    "rlm_signal",
    "mkt_gap",
    "odds_move_home",
    "consensus_spread",

    # --- ELO ---
    "elo_diff", "elo_exp",

    # --- CONTEXTO ---
    "is_conf", "season_progress",
]
N_FEAT = len(FEAT)


# ═══════════════════════ ENGINE V7 ═══════════════════════════════════════════
class EngineV7:
    """God Mode feature engine with Matchup DNA, Chronic Fatigue, Value Trap."""

    def __init__(self, boxscores, players, odds, games):
        # --- Fast indexes ---
        self.bs_idx = defaultdict(list)
        for _, r in boxscores.iterrows():
            g = r.get("game_id")
            if pd.notna(g):
                self.bs_idx[int(g)].append(r.to_dict())

        self.odds_idx = defaultdict(list)
        for _, r in odds.iterrows():
            g = r.get("game_id")
            if pd.notna(g):
                self.odds_idx[int(g)].append(r.to_dict())

        self.pl_idx = defaultdict(list)
        for _, r in players.iterrows():
            g = r.get("game_id")
            if pd.notna(g):
                self.pl_idx[int(g)].append(r.to_dict())

        self.q4_idx = {}
        for _, r in games.iterrows():
            gid = r["game_id"]
            self.q4_idx[gid] = {
                "home_q4_net": r.get("home_q4_net", 0) or 0,
                "away_q4_net": r.get("away_q4_net", 0) or 0,
            }

        # --- Accumulated state ---
        self.log = defaultdict(list)
        self.player_history = defaultdict(list)
        self.elo = {t: ELO_INIT for t in TEAM_ABBR}
        self.rec = defaultdict(lambda: {"w": 0, "l": 0})
        self.q4_history = defaultdict(list)

        # V7: league-wide ranking caches (recomputed periodically)
        self._league_ast_rank = {}     # tid → rank (1=highest assists)
        self._league_stl_rank = {}     # tid → rank (1=highest steals)
        self._league_pf_rank = {}      # tid → rank (1=MOST fouls; 30=least fouls)
        self._rank_cache_games = 0

        logger.info(
            f"EngineV7: {len(self.bs_idx)} bs, {len(self.odds_idx)} odds, {len(self.pl_idx)} pl"
        )

    def reset_season(self, s):
        self.log = defaultdict(list)
        self.rec = defaultdict(lambda: {"w": 0, "l": 0})
        self.q4_history = defaultdict(list)
        self._league_ast_rank = {}
        self._league_stl_rank = {}
        self._league_pf_rank = {}
        self._rank_cache_games = 0

    # ------------------------------------------------------------------
    def compute(self, g):
        gid = g["game_id"]; hid = g["home_team_id"]; aid = g["away_team_id"]
        gd = g["game_date"]
        if hid not in TEAM_ABBR or aid not in TEAM_ABBR:
            return None

        hl, al = self.log[hid], self.log[aid]
        if len(hl) < 5 or len(al) < 5:
            return None

        f = {}

        # ══════ 1. FOUR FACTORS ════════════════════════════════════════
        h_ff10 = self._four_factors(hl, 10)
        a_ff10 = self._four_factors(al, 10)
        h_ff5  = self._four_factors(hl, 5)
        a_ff5  = self._four_factors(al, 5)

        f["h_efg"]      = h_ff10["efg"];       f["a_efg"]      = a_ff10["efg"]
        f["h_tov_pct"]  = h_ff10["tov_pct"];   f["a_tov_pct"]  = a_ff10["tov_pct"]
        f["h_oreb_pct"] = h_ff10["oreb_pct"];  f["a_oreb_pct"] = a_ff10["oreb_pct"]
        f["h_ft_rate"]  = h_ff10["ft_rate"];   f["a_ft_rate"]  = a_ff10["ft_rate"]

        f["diff_efg"]  = h_ff10["efg"]     - a_ff10["efg"]
        f["diff_tov"]  = a_ff10["tov_pct"] - h_ff10["tov_pct"]
        f["diff_oreb"] = h_ff10["oreb_pct"]- a_ff10["oreb_pct"]
        f["diff_ft"]   = h_ff10["ft_rate"] - a_ff10["ft_rate"]

        f["h_efg_hot5"]  = h_ff5["efg"];       f["a_efg_hot5"]  = a_ff5["efg"]
        f["h_efg_trend"] = h_ff5["efg"] - h_ff10["efg"]
        f["a_efg_trend"] = a_ff5["efg"] - a_ff10["efg"]

        # ══════ 2. PACE & RATINGS ══════════════════════════════════════
        h_pr = self._pace_ratings(hl, 10)
        a_pr = self._pace_ratings(al, 10)
        for pfx, pr in [("h", h_pr), ("a", a_pr)]:
            f[f"{pfx}_pace"]    = pr["pace"] / 100
            f[f"{pfx}_ortg"]    = pr["ortg"] / 120
            f[f"{pfx}_drtg"]    = pr["drtg"] / 120
            f[f"{pfx}_net_rtg"] = (pr["ortg"] - pr["drtg"]) / 30

        # ══════ 3. V7: MATCHUP DNA ════════════════════════════════════
        md = self._matchup_dna(hid, aid, h_pr, a_pr, hl, al)
        f["matchup_ortg_vs_drtg"] = md["ortg_vs_drtg"]
        f["matchup_drtg_vs_ortg"] = md["drtg_vs_ortg"]
        f["h_ast_rate"]           = md["h_ast_rate"]
        f["a_ast_rate"]           = md["a_ast_rate"]
        f["h_stl_rate"]           = md["h_stl_rate"]
        f["a_stl_rate"]           = md["a_stl_rate"]
        f["matchup_ast_vs_stl"]   = md["ast_vs_stl"]
        f["matchup_dna_penalty"]  = md["dna_penalty"]

        # ══════ 4. V7: CHRONIC FATIGUE INDEX ═══════════════════════════
        for pfx, lg, tid in [("h", hl, hid), ("a", al, aid)]:
            fat = self._chronic_fatigue(lg, gd, tid)
            f[f"{pfx}_games_7d"]       = fat["games_7d"]
            f[f"{pfx}_b2b"]            = fat["b2b"]
            f[f"{pfx}_travel_miles_7d"] = fat["travel_miles_7d"]
            f[f"{pfx}_rest_days"]      = fat["rest_days"]
            f[f"{pfx}_road_trip_len"]  = fat["road_trip"]
            f[f"{pfx}_3in4"]           = fat["three_in_four"]
            f[f"{pfx}_4in6"]           = fat["four_in_six"]
            f[f"{pfx}_tz_crossed_7d"]  = fat["tz_crossed"]
            f[f"{pfx}_heavy_legs"]     = fat["heavy_legs"]

        # ══════ 5. AUSENCIAS ═══════════════════════════════════════════
        for pfx, tid, lg in [("h", hid, hl), ("a", aid, al)]:
            mi = self._missing_impact(tid, gid, lg[-1] if lg else None)
            f[f"{pfx}_missing_net_rtg"] = mi["net_rtg"]
            f[f"{pfx}_missing_min"]     = mi["minutes_lost"] / 240
            f[f"{pfx}_missing_stars"]   = mi["stars_out"]

        # ══════ 6. TOP-8 ═══════════════════════════════════════════════
        for pfx, tid in [("h", hid), ("a", aid)]:
            t8 = self._top8_efficiency(tid, 10)
            f[f"{pfx}_top8_off_rtg"]     = t8["off_rtg"] / 120
            f[f"{pfx}_top8_def_rtg"]     = t8["def_rtg"] / 120
            f[f"{pfx}_top8_net_rtg"]     = (t8["off_rtg"] - t8["def_rtg"]) / 30
            f[f"{pfx}_top8_consistency"] = t8["consistency"]

        # ══════ 7. CLUTCH Q4 ═══════════════════════════════════════════
        hq = self.q4_history[hid]; aq = self.q4_history[aid]
        f["h_q4_net_avg"] = np.mean(hq[-10:]) / 10 if len(hq) >= 3 else 0
        f["a_q4_net_avg"] = np.mean(aq[-10:]) / 10 if len(aq) >= 3 else 0

        # ══════ 8. V7: VALUE TRAP ══════════════════════════════════════
        vt = self._value_trap(hid, aid, h_ff10, a_ff10, hl, al)
        f["h_ft_dependency"]  = vt["h_ft_dep"]
        f["a_ft_dependency"]  = vt["a_ft_dep"]
        f["h_opp_pf_rate"]    = vt["h_opp_pf"]
        f["a_opp_pf_rate"]    = vt["a_opp_pf"]
        f["value_trap_flag"]  = vt["trap"]

        # ══════ 9. SMART MONEY ═════════════════════════════════════════
        sm = self._smart_money(gid)
        f["mkt_prob_home"]    = sm["mkt_prob_home"]
        f["mkt_spread"]       = sm["mkt_spread"] / 10
        f["rlm_signal"]       = sm["rlm_signal"]
        f["mkt_gap"]          = 0.0   # filled at predict time
        f["odds_move_home"]   = sm["odds_move_home"]
        f["consensus_spread"] = sm["consensus_spread"] / 10

        # ══════ 10. ELO ════════════════════════════════════════════════
        he, ae = self.elo[hid], self.elo[aid]
        f["elo_diff"] = (he - ae) / 100
        f["elo_exp"]  = 1 / (1 + 10 ** (-(he - ae + ELO_HCA) / 400))

        # ══════ 11. CONTEXTO ═══════════════════════════════════════════
        f["is_conf"] = 1 if (hid in EAST) == (aid in EAST) else 0
        total = self.rec[hid]["w"] + self.rec[hid]["l"]
        f["season_progress"] = min(total / 82, 1.0)

        vec = np.array([f.get(n, 0.0) for n in FEAT], dtype=np.float64)
        return np.nan_to_num(vec)

    # ------------------------------------------------------------------
    def update(self, g):
        gid = g["game_id"]; hid = g["home_team_id"]; aid = g["away_team_id"]
        if hid not in TEAM_ABBR or aid not in TEAM_ABBR:
            return

        hs = g["home_score"]; aws = g["away_score"]; gd = g["game_date"]
        hw = hs > aws

        bs = self.bs_idx.get(gid, [])
        hbs  = next((b for b in bs if REV_TEAM.get(b.get("nba_tid")) == hid), None)
        abs_ = next((b for b in bs if REV_TEAM.get(b.get("nba_tid")) == aid), None)
        pl = self.pl_idx.get(gid, [])

        for tid, ih, won, sc, osc, bx in [
            (hid, True,  hw,     hs,  aws, hbs),
            (aid, False, not hw, aws, hs,  abs_),
        ]:
            poss = None
            if bx:
                fga  = bx.get("fga", 0) or 0
                fta  = bx.get("fta", 0) or 0
                oreb = bx.get("oreb", 0) or 0
                tov  = bx.get("tov", 0) or 0
                poss = 0.96 * (fga + 0.44 * fta - oreb + tov)

            e = {
                "gid": gid, "date": gd, "home": ih, "won": won,
                "pts": sc, "opp": osc, "margin": sc - osc,
                "opp_id": aid if ih else hid, "poss": poss,
            }
            if bx:
                for s in ["fgm","fga","fg3m","fg3a","ftm","fta",
                          "oreb","dreb","reb","ast","stl","blk","tov","pf","pts"]:
                    v = bx.get(s)
                    e[s] = float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else None
            self.log[tid].append(e)

            r = self.rec[tid]
            if won: r["w"] += 1
            else:   r["l"] += 1

        # Q4
        q4 = self.q4_idx.get(gid, {})
        self.q4_history[hid].append(q4.get("home_q4_net", 0))
        self.q4_history[aid].append(q4.get("away_q4_net", 0))

        # Player history
        for p in pl:
            pid = p["player_id"]
            tid = REV_TEAM.get(p.get("nba_tid"))
            if tid not in (hid, aid):
                continue
            self.player_history[pid].append({
                "gid": gid, "date": gd, "team": tid,
                "mins": p.get("minutes_decimal") or 0,
                "starter": p.get("starter") == 1,
                "pts": p.get("pts") or 0, "reb": p.get("reb") or 0,
                "ast": p.get("ast") or 0,
                "plus_minus": p.get("plus_minus") or 0,
                "fgm": p.get("fgm") or 0, "fga": p.get("fga") or 0,
                "fg3m": p.get("fg3m") or 0,
                "fta": p.get("fta") or 0, "tov": p.get("tov") or 0,
                "oreb": p.get("oreb") or 0, "stl": p.get("stl") or 0,
            })

        # ELO
        he, ae = self.elo[hid], self.elo[aid]
        exp = 1 / (1 + 10 ** (-(he - ae + ELO_HCA) / 400))
        act = 1.0 if hw else 0.0
        mov = min(np.log1p(abs(hs - aws)) * 0.7, 2.5)
        ac  = 2.2 / ((abs(he - ae) * 0.001) + 2.2)
        k   = ELO_K * mov * ac
        self.elo[hid] += k * (act - exp)
        self.elo[aid] += k * ((1 - act) - (1 - exp))

        # Refresh league rankings every ~30 games
        self._rank_cache_games += 1
        if self._rank_cache_games >= 30:
            self._refresh_league_rankings()
            self._rank_cache_games = 0

    # ══════════════════════ HELPER: Four Factors ═════════════════════
    def _four_factors(self, lg, window=10):
        r = lg[-window:]
        if not r:
            return {"efg": 0.52, "tov_pct": 0.13, "oreb_pct": 0.22, "ft_rate": 0.20}

        def _sm(key, default):
            vals = [g.get(key) for g in r if g.get(key) is not None]
            return np.mean(vals) if vals else default

        fga = _sm("fga", 85); fgm = _sm("fgm", 37); fg3m = _sm("fg3m", 11)
        fta = _sm("fta", 22); oreb = _sm("oreb", 10); dreb = _sm("dreb", 33)
        tov = _sm("tov", 14)

        poss = 0.96 * (fga + 0.44 * fta - oreb + tov)
        efg     = (fgm + 0.5 * fg3m) / fga if fga > 0 else 0.52
        tov_pct = tov / poss if poss > 0 else 0.13
        oreb_pct= oreb / (oreb + dreb) if (oreb + dreb) > 0 else 0.22
        ft_rate = fta / fga if fga > 0 else 0.20

        return {"efg": efg, "tov_pct": tov_pct, "oreb_pct": oreb_pct, "ft_rate": ft_rate}

    # ══════════════════════ HELPER: Pace & Ratings ═══════════════════
    def _pace_ratings(self, lg, window=10):
        r = lg[-window:]
        if not r:
            return {"pace": 98, "ortg": 110, "drtg": 110}
        paces, ortgs, drtgs = [], [], []
        for g in r:
            poss = g.get("poss"); pts = g.get("pts"); opp = g.get("opp")
            if poss and poss > 50 and pts is not None and opp is not None:
                paces.append(poss)
                ortgs.append(pts / poss * 100)
                drtgs.append(opp / poss * 100)
        return {
            "pace": np.mean(paces) if paces else 98,
            "ortg": np.mean(ortgs) if ortgs else 110,
            "drtg": np.mean(drtgs) if drtgs else 110,
        }

    # ══════════════════════ V7: MATCHUP DNA ══════════════════════════
    def _matchup_dna(self, hid, aid, h_pr, a_pr, hl, al):
        """
        Compara el estilo ofensivo de un equipo contra el estilo defensivo
        del rival usando asistencias (flujo de balón) vs robos (disrupción).

        Si un equipo top-5 en ast_rate enfrenta un rival top-5 en stl_rate,
        el ataque del primero se verá reducido → penalty flag.
        """
        # Attack vs Defense differential (per 100 poss)
        ortg_vs_drtg = (h_pr["ortg"] - a_pr["drtg"]) / 20   # home attack vs away defense
        drtg_vs_ortg = (h_pr["drtg"] - a_pr["ortg"]) / 20   # home defense vs away attack

        # Assist rate = AST / FGM (proxy de ball movement)
        h_ast_r = self._team_rate(hl, "ast", "fgm", 10, 0.60)
        a_ast_r = self._team_rate(al, "ast", "fgm", 10, 0.60)

        # Steal rate = STL / opp_poss (proxy; we use STL per game normalized)
        h_stl_r = self._team_rate(hl, "stl", None, 10, 7.5) / 10
        a_stl_r = self._team_rate(al, "stl", None, 10, 7.5) / 10

        # Clash signal: high-ast offense vs high-stl defense
        # Positive = home's ball movement faces disruptive away defense
        ast_vs_stl = h_ast_r * a_stl_r - a_ast_r * h_stl_r

        # DNA penalty: is home top-5 assists AND away top-5 steals? (or vice versa)
        dna_penalty = 0
        h_ast_rank = self._league_ast_rank.get(hid, 15)
        a_ast_rank = self._league_ast_rank.get(aid, 15)
        h_stl_rank = self._league_stl_rank.get(hid, 15)
        a_stl_rank = self._league_stl_rank.get(aid, 15)

        if h_ast_rank <= MATCHUP_TOP5_RANK and a_stl_rank <= MATCHUP_TOP5_RANK:
            dna_penalty = 1    # home's passing game will be disrupted
        elif a_ast_rank <= MATCHUP_TOP5_RANK and h_stl_rank <= MATCHUP_TOP5_RANK:
            dna_penalty = -1   # away's passing game will be disrupted

        return {
            "ortg_vs_drtg": ortg_vs_drtg,
            "drtg_vs_ortg": drtg_vs_ortg,
            "h_ast_rate": h_ast_r,
            "a_ast_rate": a_ast_r,
            "h_stl_rate": h_stl_r,
            "a_stl_rate": a_stl_r,
            "ast_vs_stl": ast_vs_stl,
            "dna_penalty": dna_penalty,
        }

    def _team_rate(self, lg, stat, denom_stat, window, default):
        """Compute stat / denom_stat (or raw stat if denom_stat is None)."""
        r = lg[-window:]
        if not r:
            return default
        if denom_stat:
            vals = [
                g.get(stat, 0) / max(g.get(denom_stat, 1), 1)
                for g in r if g.get(stat) is not None
            ]
        else:
            vals = [g.get(stat, 0) for g in r if g.get(stat) is not None]
        return np.mean(vals) if vals else default

    def _refresh_league_rankings(self):
        """Recompute league-wide AST, STL, PF rankings across all teams."""
        ast_totals = {}; stl_totals = {}; pf_totals = {}

        for tid in TEAM_ABBR:
            lg = self.log[tid]
            if len(lg) < 5:
                ast_totals[tid] = 0; stl_totals[tid] = 0; pf_totals[tid] = 999
                continue
            recent = lg[-15:]
            ast_totals[tid] = np.mean([g.get("ast", 0) or 0 for g in recent])
            stl_totals[tid] = np.mean([g.get("stl", 0) or 0 for g in recent])
            pf_totals[tid]  = np.mean([g.get("pf", 0) or 0 for g in recent])

        # Rank: 1 = highest value
        for d, target in [
            (ast_totals, "_league_ast_rank"),
            (stl_totals, "_league_stl_rank"),
        ]:
            sorted_teams = sorted(d, key=lambda t: d[t], reverse=True)
            setattr(self, target, {t: i+1 for i, t in enumerate(sorted_teams)})

        # PF rank: 1 = MOST fouls (worst discipline), 30 = fewest fouls (best)
        sorted_pf = sorted(pf_totals, key=lambda t: pf_totals[t], reverse=True)
        self._league_pf_rank = {t: i+1 for i, t in enumerate(sorted_pf)}

    # ══════════════════════ V7: CHRONIC FATIGUE ══════════════════════
    def _chronic_fatigue(self, lg, game_date, team_id):
        """
        Goes beyond B2B: accumulates 7-day travel distance, timezone
        crossings, and flags 'Heavy Legs' (4-in-6 + ≥2 TZ crossings).
        """
        if not lg:
            return {
                "games_7d": 0, "b2b": 0, "travel_miles_7d": 0,
                "rest_days": 3, "road_trip": 0, "three_in_four": 0,
                "four_in_six": 0, "tz_crossed": 0, "heavy_legs": 0,
            }

        last = lg[-1]
        last_date = pd.Timestamp(last["date"]) if isinstance(last["date"], str) else last["date"]
        rest_days = (game_date - last_date).days

        week_ago = game_date - timedelta(days=7)
        recent_7d = [
            g for g in lg
            if (pd.Timestamp(g["date"]) if isinstance(g["date"], str) else g["date"]) >= week_ago
        ]
        games_7d = len(recent_7d)

        b2b = 1 if rest_days <= 1 else 0

        # 3-in-4
        four_nights_ago = game_date - timedelta(days=3)
        games_in_4 = sum(
            1 for g in lg
            if (pd.Timestamp(g["date"]) if isinstance(g["date"], str) else g["date"]) >= four_nights_ago
        )
        three_in_four = 1 if games_in_4 >= 2 else 0

        # V7: 4-in-6
        six_days_ago = game_date - timedelta(days=5)
        games_in_6 = sum(
            1 for g in lg
            if (pd.Timestamp(g["date"]) if isinstance(g["date"], str) else g["date"]) >= six_days_ago
        )
        four_in_six = 1 if games_in_6 >= 3 else 0  # 3 prev + today = 4

        # V7: Accumulated travel miles over 7 days
        travel_miles_7d = 0
        tz_crossed_7d = 0
        prev_loc = team_id  # start at home

        for g in recent_7d:
            curr_loc = team_id if g["home"] else g["opp_id"]
            if prev_loc in COORDS and curr_loc in COORDS:
                c1, c2 = COORDS[prev_loc], COORDS[curr_loc]
                travel_miles_7d += _haversine(c1[0], c1[1], c2[0], c2[1])
            # Timezone crossings
            tz1 = TEAM_TZ.get(prev_loc, -6)
            tz2 = TEAM_TZ.get(curr_loc, -6)
            tz_crossed_7d += abs(tz1 - tz2)
            prev_loc = curr_loc

        # Road trip
        road_trip = 0
        for g in reversed(lg):
            if not g["home"]:
                road_trip += 1
            else:
                break

        # V7: HEAVY LEGS composite flag
        heavy_legs = 0
        if tz_crossed_7d >= TIMEZONE_PENALTY_THRESHOLD and four_in_six:
            heavy_legs = 1
        elif games_7d >= 4 and travel_miles_7d > 4000:
            heavy_legs = 1  # extreme travel even without TZ

        return {
            "games_7d":        min(games_7d / 4, 1.0),
            "b2b":             b2b,
            "travel_miles_7d": min(travel_miles_7d / 8000, 1.0),
            "rest_days":       min(rest_days, 7),
            "road_trip":       min(road_trip / 5, 1.0),
            "three_in_four":   three_in_four,
            "four_in_six":     four_in_six,
            "tz_crossed":      min(tz_crossed_7d / 6, 1.0),
            "heavy_legs":      heavy_legs,
        }

    # ══════════════════════ V7: VALUE TRAP ═══════════════════════════
    def _value_trap(self, hid, aid, h_ff, a_ff, hl, al):
        """
        Detects when a favorite depends on free throws but faces a rival
        that commits very few fouls → the favorite's FT-based edge evaporates.
        """
        h_ft_dep = h_ff["ft_rate"]  # raw ft_rate; high = FT dependent
        a_ft_dep = a_ff["ft_rate"]

        # Opponent PF rate (fouls that the opponent commits when facing this team)
        # We approximate by looking at the rival's average PF from their log
        def _opp_pf(lg, window=10):
            r = lg[-window:]
            if not r:
                return 20  # NBA avg ~20 PF/game
            vals = [g.get("pf", 0) or 0 for g in r if g.get("pf") is not None]
            return np.mean(vals) if vals else 20

        h_opp_pf = _opp_pf(al, 10) / 25   # away team's foul rate (normalized)
        a_opp_pf = _opp_pf(hl, 10) / 25   # home team's foul rate

        # Trap detection
        trap = 0
        a_pf_rank = self._league_pf_rank.get(aid, 15)  # 30 = fewest fouls
        h_pf_rank = self._league_pf_rank.get(hid, 15)

        # Home team depends on FT AND away team barely fouls
        if h_ft_dep > FT_DEPENDENCY_THRESHOLD and a_pf_rank >= (30 - LOW_FOUL_RANK_THRESHOLD + 1):
            trap = 1  # home favorite's FT edge is a trap
        # Away team depends on FT AND home team barely fouls
        elif a_ft_dep > FT_DEPENDENCY_THRESHOLD and h_pf_rank >= (30 - LOW_FOUL_RANK_THRESHOLD + 1):
            trap = -1  # away team's FT edge is a trap

        return {
            "h_ft_dep": h_ft_dep,
            "a_ft_dep": a_ft_dep,
            "h_opp_pf": h_opp_pf,
            "a_opp_pf": a_opp_pf,
            "trap": trap,
        }

    # ══════════════════════ Missing Impact ════════════════════════════
    def _missing_impact(self, team_id, game_id, last_game):
        if not last_game:
            return {"net_rtg": 0, "minutes_lost": 0, "stars_out": 0}
        last_gid = last_game["gid"]
        last_players = [
            p for p in self.pl_idx.get(last_gid, [])
            if REV_TEAM.get(p.get("nba_tid")) == team_id
        ]
        today_players = self.pl_idx.get(game_id, [])
        today_ids = {
            p["player_id"] for p in today_players
            if REV_TEAM.get(p.get("nba_tid")) == team_id
        }
        missing = []
        for p in last_players:
            if p["player_id"] not in today_ids:
                mins = p.get("minutes_decimal") or 0
                pm   = p.get("plus_minus") or 0
                starter = p.get("starter") == 1
                missing.append({"mins": mins, "pm": pm, "starter": starter})
        if not missing:
            return {"net_rtg": 0, "minutes_lost": 0, "stars_out": 0}
        total_mins = sum(m["mins"] for m in missing)
        avg_pm     = np.mean([m["pm"] for m in missing])
        stars_out  = sum(1 for m in missing if m["starter"] and m["mins"] > 20)
        net_rtg    = avg_pm / 48 * 10
        return {
            "net_rtg":      np.clip(net_rtg, -5, 5) / 5,
            "minutes_lost": total_mins,
            "stars_out":    min(stars_out / 3, 1.0),
        }

    # ══════════════════════ Top-8 Efficiency ══════════════════════════
    def _top8_efficiency(self, team_id, window=10):
        team_games = self.log[team_id][-window:]
        if not team_games:
            return {"off_rtg": 110, "def_rtg": 110, "consistency": 0.5}
        ps = defaultdict(lambda: {"mins":0,"pm":0,"games":0,"fgm":0,"fga":0,
                                   "fg3m":0,"fta":0,"oreb":0,"tov":0,"pts":0})
        for g in team_games:
            gid = g["gid"]
            for p in self.pl_idx.get(gid, []):
                if REV_TEAM.get(p.get("nba_tid")) != team_id:
                    continue
                pid = p["player_id"]
                mins = p.get("minutes_decimal") or 0
                if mins < 5:
                    continue
                ps[pid]["mins"]  += mins
                ps[pid]["pm"]    += (p.get("plus_minus") or 0)
                ps[pid]["games"] += 1
                ps[pid]["fgm"]   += (p.get("fgm") or 0)
                ps[pid]["fga"]   += (p.get("fga") or 0)
                ps[pid]["fg3m"]  += (p.get("fg3m") or 0)
                ps[pid]["fta"]   += (p.get("fta") or 0)
                ps[pid]["oreb"]  += (p.get("oreb") or 0)
                ps[pid]["tov"]   += (p.get("tov") or 0)
                ps[pid]["pts"]   += (p.get("pts") or 0)
        if not ps:
            return {"off_rtg": 110, "def_rtg": 110, "consistency": 0.5}
        avg = []
        for pid, s in ps.items():
            if s["games"] > 0:
                avg.append({
                    "mins": s["mins"]/s["games"], "pm": s["pm"]/s["games"],
                    "pts": s["pts"]/s["games"],
                    "poss": (s["fga"]+0.44*s["fta"]-s["oreb"]+s["tov"])/s["games"],
                })
        avg.sort(key=lambda x: x["mins"], reverse=True)
        top8 = avg[:8]
        if not top8:
            return {"off_rtg": 110, "def_rtg": 110, "consistency": 0.5}
        ortgs, drtgs, weights = [], [], []
        for p in top8:
            poss = max(p["poss"], 1)
            ortgs.append(p["pts"] / poss * 100)
            drtgs.append(max(110 - p["pm"] / max(p["mins"], 1) * 48 / 2, 80))
            weights.append(p["mins"])
        w = np.array(weights)
        w = w / w.sum() if w.sum() > 0 else np.ones(len(w)) / len(w)
        off_rtg = np.average(ortgs, weights=w)
        def_rtg = np.average(drtgs, weights=w)
        pm_vals = [p["pm"] for p in top8]
        consistency = 1 - min(np.std(pm_vals) / 15, 1) if len(pm_vals) > 1 else 0.5
        return {"off_rtg": off_rtg, "def_rtg": def_rtg, "consistency": consistency}

    # ══════════════════════ Smart Money ═══════════════════════════════
    def _smart_money(self, gid):
        odds = self.odds_idx.get(gid, [])
        if not odds:
            return {"mkt_prob_home": 0.5, "mkt_spread": 0,
                    "rlm_signal": 0, "odds_move_home": 0, "consensus_spread": 0}
        home_name = odds[0].get("home_team", "")
        bk_home_probs = {}; all_home_probs = []; all_spreads = []
        for o in odds:
            mk = o.get("market",""); pr = o.get("outcome_price")
            pt = o.get("outcome_point"); nm = o.get("outcome_name","")
            bk = o.get("bookmaker","unknown")
            is_home = (nm == home_name or home_name in nm)
            if mk == "h2h" and pr is not None:
                prob = self._odds_to_prob(pr)
                if prob is not None:
                    if is_home:
                        bk_home_probs[bk] = prob; all_home_probs.append(prob)
                    else:
                        bk_home_probs.setdefault(bk, 1 - prob)
                        all_home_probs.append(1 - prob)
            elif mk == "spreads" and pt is not None and is_home:
                all_spreads.append(pt)
        mkt_prob = np.mean(all_home_probs) if all_home_probs else 0.5
        mkt_spread = np.mean(all_spreads) if all_spreads else 0
        consensus = np.median(all_spreads) if all_spreads else 0
        rlm_signal = 0; odds_move = 0
        if len(all_home_probs) >= 3:
            prob_median = np.median(all_home_probs)
            prob_max = max(all_home_probs); prob_min = min(all_home_probs)
            odds_move = prob_max - prob_min
            if prob_median > 0.55 and prob_min < prob_median - 0.05:
                rlm_signal = -1
            elif prob_median < 0.45 and prob_max > prob_median + 0.05:
                rlm_signal = 1
        return {"mkt_prob_home": mkt_prob, "mkt_spread": mkt_spread,
                "rlm_signal": rlm_signal, "odds_move_home": odds_move,
                "consensus_spread": consensus}

    @staticmethod
    def _odds_to_prob(price):
        if price is None: return None
        try: price = float(price)
        except (ValueError, TypeError): return None
        if price == 0: return None
        if abs(price) >= 100:
            return 100/(price+100) if price > 0 else abs(price)/(abs(price)+100)
        elif price >= 1.01:
            return 1 / price
        return None


# ═══════════════════════ MONTE CARLO V7 ══════════════════════════════════════
class MonteCarloV7:
    """V7 MC: applies Heavy Legs eFG penalty and returns volatility for risk level."""

    def __init__(self, engine, n_sims=DEFAULT_SIMS):
        self.engine = engine
        self.n_sims = n_sims

    def run(self, game, feat):
        gid = game["game_id"]; hid = game["home_team_id"]; aid = game["away_team_id"]
        h_pl = self._get_expected_players(hid, gid)
        a_pl = self._get_expected_players(aid, gid)
        if len(h_pl) < 5 or len(a_pl) < 5:
            return self._fallback(feat)

        h_pace = self._team_pace(hid); a_pace = self._team_pace(aid)
        avg_pace = (h_pace + a_pace) / 2

        h_effs = np.array([p["ortg_100"] for p in h_pl[:10]])
        a_effs = np.array([p["ortg_100"] for p in a_pl[:10]])
        h_mins = np.array([p["expected_mins"] for p in h_pl[:10]])
        a_mins = np.array([p["expected_mins"] for p in a_pl[:10]])
        h_stds = np.array([p["ortg_std"] for p in h_pl[:10]])
        a_stds = np.array([p["ortg_std"] for p in a_pl[:10]])

        # V7: Apply Heavy Legs eFG penalty to mean efficiency
        h_hl_idx = FEAT.index("h_heavy_legs")
        a_hl_idx = FEAT.index("a_heavy_legs")
        if feat[h_hl_idx] == 1:
            h_effs = h_effs * (1 - HEAVY_LEGS_EFG_PENALTY * 2)  # ~3% reduction
        if feat[a_hl_idx] == 1:
            a_effs = a_effs * (1 - HEAVY_LEGS_EFG_PENALTY * 2)

        h_wt = h_mins / h_mins.sum() if h_mins.sum() > 0 else np.ones(len(h_mins))/len(h_mins)
        a_wt = a_mins / a_mins.sum() if a_mins.sum() > 0 else np.ones(len(a_mins))/len(a_mins)

        wins = 0
        margins = np.empty(self.n_sims)
        for i in range(self.n_sims):
            h_samp = np.random.normal(h_effs, h_stds)
            a_samp = np.random.normal(a_effs, a_stds)
            h_team = np.dot(h_wt, h_samp)
            a_team = np.dot(a_wt, a_samp)
            sim_pace = avg_pace + np.random.normal(0, 2.5)
            h_pts = sim_pace * h_team / 100 + np.random.normal(0, 1.2)
            a_pts = sim_pace * a_team / 100 + np.random.normal(0, 1.2)
            h_pts += np.random.normal(1.5, 0.5)  # HCA
            if h_pts > a_pts: wins += 1
            margins[i] = h_pts - a_pts

        wp = wins / self.n_sims
        ms = np.std(margins)
        return {
            "wp": wp, "em": np.mean(margins), "ms": ms,
            "conf": max(0, 1 - ms / 18),
            "volatility": ms,  # V7: used for risk-level in CSV
            "n_players": (len(h_pl), len(a_pl)),
        }

    def _get_expected_players(self, team_id, game_id):
        team_games = self.engine.log[team_id][-5:]
        if not team_games: return []
        ps = defaultdict(lambda: {"games":0,"total_mins":0,"total_pts":0,
                                   "total_poss":0,"ortg_vals":[]})
        for g in team_games:
            gid = g["gid"]
            if gid == game_id: continue
            for p in self.engine.pl_idx.get(gid, []):
                if REV_TEAM.get(p.get("nba_tid")) != team_id: continue
                pid = p["player_id"]; mins = p.get("minutes_decimal") or 0
                if mins < 5: continue
                pts = p.get("pts") or 0; fga = p.get("fga") or 0
                fta = p.get("fta") or 0; tov = p.get("tov") or 0
                oreb = p.get("oreb") or 0
                poss_used = 0.96 * (fga + 0.44*fta - oreb + tov)
                ortg = (pts / poss_used * 100) if poss_used > 3 else 100
                ps[pid]["games"] += 1; ps[pid]["total_mins"] += mins
                ps[pid]["total_pts"] += pts; ps[pid]["total_poss"] += poss_used
                ps[pid]["ortg_vals"].append(ortg)
        result = []
        for pid, s in ps.items():
            if s["games"] >= 2:
                result.append({
                    "player_id": pid,
                    "expected_mins": s["total_mins"]/s["games"],
                    "ortg_100": np.mean(s["ortg_vals"]) if s["ortg_vals"] else 100,
                    "ortg_std": max(np.std(s["ortg_vals"]) if len(s["ortg_vals"])>1 else 8, 3),
                    "games_played": s["games"],
                })
        result.sort(key=lambda x: x["expected_mins"], reverse=True)
        return result[:10]

    def _team_pace(self, tid):
        lg = self.engine.log[tid][-10:]
        paces = [g["poss"] for g in lg if g.get("poss") and g["poss"] > 50]
        return np.mean(paces) if paces else 98

    def _fallback(self, feat):
        try: wp = feat[FEAT.index("elo_exp")]
        except: wp = 0.5
        return {"wp":wp,"em":0,"ms":12,"conf":0.5,"volatility":12,"n_players":(0,0)}


# ═══════════════════════ MODEL V7 ════════════════════════════════════════════
class ModelV7:
    """God Mode model: Elite VIP threshold + matchup/fatigue/trap adjustments."""

    def __init__(self):
        self.xgb = None
        self.scaler = StandardScaler()
        self.mc = None
        self.engine = None
        self.trained = False
        self.tX, self.ty = [], []
        self._day_picks = []
        self._current_day = None
        logger.info(
            f"ModelV7 init | VIP≥{VIP_THRESHOLD:.0%} EV≥{MIN_EV} | "
            f"Fallback≥{VIP_FALLBACK:.0%} | FP×{FP_PENALTY}"
        )

    def connect_mc(self, mc): self.mc = mc
    def connect_engine(self, eng): self.engine = eng

    def add(self, X, y, gid):
        self.tX.append(X); self.ty.append(y)

    def retrain(self):
        if len(self.tX) < 300: return False
        X = np.array(self.tX); y = np.array(self.ty)
        X_sc = self.scaler.fit_transform(X)
        n_pos = max(np.sum(y), 1); n_neg = len(y) - n_pos
        spw = (n_neg / n_pos) * 2.5
        self.xgb = xgb.XGBClassifier(
            n_estimators=500, max_depth=5, learning_rate=0.02,
            subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.3, reg_lambda=2.0,
            min_child_weight=6, gamma=0.15,
            scale_pos_weight=spw, random_state=42,
            use_label_encoder=False, eval_metric="logloss",
        )
        self.xgb.fit(X_sc, y); self.trained = True
        logger.info(f"XGB trained | {len(self.tX)} samples | spw={spw:.2f}")
        return True

    def predict(self, X, game=None):
        gid = game["game_id"] if game is not None else None
        game_date = game["game_date"] if game is not None else None

        # ── WP base ──────────────────────────────────────────────────
        mc_volatility = 12  # default
        if not self.trained:
            if self.mc and game is not None:
                mc_res = self.mc.run(game, X)
                wp = mc_res["wp"]; conf = mc_res["conf"]
                mc_volatility = mc_res.get("volatility", 12)
            else:
                try: wp = X[FEAT.index("elo_exp")]
                except: wp = 0.5
                conf = 0.5
        else:
            X_sc = self.scaler.transform(X.reshape(1, -1))
            wp_xgb = self.xgb.predict_proba(X_sc)[0][1]
            if self.mc and game is not None:
                mc_res = self.mc.run(game, X)
                wp_mc = mc_res["wp"]; conf = mc_res["conf"]
                mc_volatility = mc_res.get("volatility", 12)
                wp = 0.60 * wp_xgb + 0.40 * wp_mc  # V7: more MC weight
            else:
                wp = wp_xgb; conf = 0.6

        # ── Market gap ───────────────────────────────────────────────
        mkt_prob = X[FEAT.index("mkt_prob_home")]
        mkt_gap = wp - mkt_prob
        try: X[FEAT.index("mkt_gap")] = mkt_gap
        except: pass

        rlm = X[FEAT.index("rlm_signal")]

        # Fair odds → EV
        pick_home = wp > 0.5
        pick_prob = wp if pick_home else 1 - wp
        mkt_pick_prob = mkt_prob if pick_home else 1 - mkt_prob
        fair_odds = 1 / max(mkt_pick_prob, 0.01)
        market_odds = fair_odds * 0.95
        ev = pick_prob * market_odds - 1

        # ── V7: MATCHUP DNA PENALTY ──────────────────────────────────
        dna = X[FEAT.index("matchup_dna_penalty")]
        if pick_home and dna == 1:
            wp *= (1 - MATCHUP_PENALTY)       # home's passing game disrupted
        elif not pick_home and dna == -1:
            wp = 1 - ((1-wp) * (1 - MATCHUP_PENALTY))  # away's passing disrupted

        # ── V7: HEAVY LEGS PENALTY ───────────────────────────────────
        fatigue_trap = False
        if pick_home:
            hl = X[FEAT.index("h_heavy_legs")]
            q4 = X[FEAT.index("h_q4_net_avg")]
            b2b = X[FEAT.index("h_b2b")]
            tin4 = X[FEAT.index("h_3in4")]
            if hl == 1 or (b2b == 1 and q4 < 0) or (tin4 == 1 and q4 < 0):
                fatigue_trap = True
                wp *= 0.90  # 10% reduction for heavy legs
        else:
            hl = X[FEAT.index("a_heavy_legs")]
            q4 = X[FEAT.index("a_q4_net_avg")]
            b2b = X[FEAT.index("a_b2b")]
            tin4 = X[FEAT.index("a_3in4")]
            if hl == 1 or (b2b == 1 and q4 < 0) or (tin4 == 1 and q4 < 0):
                fatigue_trap = True
                wp = 1 - ((1-wp) * 0.90)

        # ── V7: VALUE TRAP PENALTY ───────────────────────────────────
        value_trap = X[FEAT.index("value_trap_flag")]
        value_trap_active = False
        if pick_home and value_trap == 1:
            wp *= 0.95; value_trap_active = True
        elif not pick_home and value_trap == -1:
            wp = 1 - ((1-wp) * 0.95); value_trap_active = True

        # Recompute pick_prob after adjustments
        pick_prob_adj = max(wp, 1 - wp)
        ev_adj = pick_prob_adj * market_odds - 1

        # ── VIP DECISION (V7: stricter) ──────────────────────────────
        is_vip = False; vip_reason = ""

        rlm_contradicts = (pick_home and rlm == -1) or (not pick_home and rlm == 1)
        gap_ok = abs(mkt_gap) >= MKT_GAP_MIN
        ev_ok = ev_adj >= MIN_EV

        # Tier 1: Elite pick (≥78%, EV≥0.10, gap, no RLM contradiction)
        if pick_prob_adj >= VIP_THRESHOLD and ev_ok and gap_ok and not rlm_contradicts:
            is_vip = True; vip_reason = "ELITE"
        # Tier 2: RLM-confirmed (≥74%, EV≥0.10)
        elif not is_vip and not rlm_contradicts:
            rlm_confirms = (pick_home and rlm == 1) or (not pick_home and rlm == -1)
            if rlm_confirms and pick_prob_adj >= 0.74 and ev_ok:
                is_vip = True; vip_reason = "RLM_ELITE"

        # ── V7: Risk level from MC volatility ────────────────────────
        if mc_volatility <= 8:
            risk_level = "Bajo"
        elif mc_volatility <= 13:
            risk_level = "Medio"
        else:
            risk_level = "Alto"

        result = {
            "wp": wp, "conf": conf, "is_vip": is_vip,
            "vip_reason": vip_reason, "mkt_gap": mkt_gap,
            "ev": ev_adj, "rlm": rlm,
            "fatigue_trap": fatigue_trap,
            "value_trap": value_trap_active,
            "mkt_odds": market_odds, "pick_home": pick_home,
            "risk_level": risk_level,
            "mc_volatility": mc_volatility,
        }

        # Top-3 fallback tracking
        if game_date is not None:
            day_str = str(game_date)[:10]
            if self._current_day != day_str:
                self._flush_day_picks()
                self._current_day = day_str; self._day_picks = []
            self._day_picks.append({
                "game": game, "result": result,
                "confidence": pick_prob_adj, "ev": ev_adj,
            })
        return result

    def _flush_day_picks(self):
        if not self._day_picks: return
        vip_count = sum(1 for p in self._day_picks if p["result"]["is_vip"])
        if vip_count == 0:
            candidates = [
                p for p in self._day_picks
                if p["confidence"] >= VIP_FALLBACK and p["ev"] > 0
            ]
            candidates.sort(key=lambda x: x["ev"], reverse=True)
            for c in candidates[:3]:
                c["result"]["is_vip"] = True
                c["result"]["vip_reason"] = "TOP3_FALLBACK"
                self._save_pick(c["game"], c["result"])

    def _save_pick(self, game, result):
        if game is None: return
        hid = game["home_team_id"]; aid = game["away_team_id"]
        h_abbr = TEAM_ABBR.get(hid, "???"); a_abbr = TEAM_ABBR.get(aid, "???")
        partido = f"{a_abbr} @ {h_abbr}"
        pick = h_abbr if result["pick_home"] else a_abbr
        conf = max(result["wp"], 1 - result["wp"])
        row = {
            "Fecha": str(game["game_date"])[:10],
            "Partido": partido,
            "Pick": pick,
            "Confianza_IA": f"{conf:.3f}",
            "Cuota_Mkt": f"{result['mkt_odds']:.3f}",
            "Valor_Esperado": f"{result['ev']:.4f}",
            "Nivel_de_Riesgo": result.get("risk_level", "Medio"),
            "Razon": result.get("vip_reason", ""),
            "RLM": result.get("rlm", 0),
            "Fatigue_Trap": result.get("fatigue_trap", False),
            "Value_Trap": result.get("value_trap", False),
            "MC_Volatility": f"{result.get('mc_volatility', 0):.2f}",
        }
        file_exists = os.path.exists(PICKS_CSV)
        with open(PICKS_CSV, "a", newline="", encoding="utf-8") as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=row.keys())
            if not file_exists: writer.writeheader()
            writer.writerow(row)

    def save_vip_pick(self, game, result):
        self._save_pick(game, result)

    def save(self, path=None):
        path = path or f"{MODEL_DIR}/nba_model_v7.pkl"
        with open(path, "wb") as f:
            pickle.dump({"xgb":self.xgb,"scaler":self.scaler,
                         "trained":self.trained,"n":len(self.tX)}, f)

    def load(self, path=None):
        path = path or f"{MODEL_DIR}/nba_model_v7.pkl"
        if not os.path.exists(path): return False
        with open(path, "rb") as f: d = pickle.load(f)
        if d.get("trained"):
            self.xgb=d["xgb"]; self.scaler=d["scaler"]; self.trained=True; return True
        return False


# ═══════════════════════ METRICS V7 ══════════════════════════════════════════
class MetricsV7:
    def __init__(self):
        self.all_probs=[]; self.all_acts=[]; self.all_margs=[]
        self.all_gids=[]; self.all_is_vip=[]; self.all_ev=[]

    def add(self, prob, actual, margin, gid, is_vip=False, ev=0):
        self.all_probs.append(prob); self.all_acts.append(actual)
        self.all_margs.append(margin); self.all_gids.append(gid)
        self.all_is_vip.append(is_vip); self.all_ev.append(ev)

    def report(self, last_n=None, label=""):
        if not self.all_probs: return {}
        n = last_n or len(self.all_probs)
        probs=self.all_probs[-n:]; acts=self.all_acts[-n:]
        is_vip=self.all_is_vip[-n:]; evs=self.all_ev[-n:]
        preds=[1 if p>0.5 else 0 for p in probs]
        acc=accuracy_score(acts,preds); brier=brier_score_loss(acts,probs)
        vi=[i for i,v in enumerate(is_vip) if v]; n_vip=len(vi)
        if n_vip>0:
            vp=[probs[i] for i in vi]; va=[acts[i] for i in vi]
            vpreds=[1 if p>0.5 else 0 for p in vp]
            vip_acc=accuracy_score(va,vpreds)
            vip_brier=brier_score_loss(va,vp)
            vip_ev_avg=np.mean([evs[i] for i in vi])
            vip_correct=sum(1 for j in range(n_vip) if vpreds[j]==va[j])
        else:
            vip_acc=0;vip_brier=1;vip_ev_avg=0;vip_correct=0

        print(f"\n{'='*65}")
        print(f"  CHECKPOINT {label} ({n} juegos)")
        print(f"{'='*65}")
        print(f"  📊 GLOBAL: Acc {acc:.1%} | Brier {brier:.4f}")
        print(f"  🏆 VIP: {n_vip}/{n} ({n_vip/n*100:.1f}%)")
        if n_vip>0:
            fp=n_vip-vip_correct
            emoji = "🔥" if vip_acc>=0.85 else "🎯" if vip_acc>=0.78 else "⚠️"
            print(f"    Acc:   {vip_acc:.1%}  {emoji}")
            print(f"    Brier: {vip_brier:.4f}")
            print(f"    EV avg:{vip_ev_avg:+.3f}")
            print(f"    FP:    {fp}/{n_vip} ({fp/n_vip*100:.1f}%)")
        print(f"{'='*65}")
        return {"acc":acc,"brier":brier,"vip_acc":vip_acc,"vip_brier":vip_brier,
                "vip_pct":n_vip/n if n>0 else 0,"vip_ev":vip_ev_avg}


# ═══════════════════════ PIPELINE V7 ════════════════════════════════════════
def run(train_s, eval_s, ckpt, db, n_sims=DEFAULT_SIMS):
    t0 = time.time()
    all_s = train_s + [eval_s]

    print(f"\n{'='*65}")
    print(f"  NBA PIPELINE V7 — THE GOD MODE")
    print(f"  Train: {', '.join(train_s)}")
    print(f"  Eval:  {eval_s}")
    print(f"  Features: {N_FEAT} (+Matchup DNA, Chronic Fatigue, Value Trap)")
    print(f"  Sims: {n_sims:,} | VIP≥{VIP_THRESHOLD:.0%} EV≥{MIN_EV}")
    print(f"{'='*65}\n")

    if os.path.exists(PICKS_CSV): os.remove(PICKS_CSV)

    dl = DataLoader(db)
    games = dl.load_games(all_s)
    bs = dl.load_boxscores(); pl = dl.load_players(); od = dl.load_odds()
    dl.close()
    logger.info(f"Games:{len(games)} BS:{len(bs)} PL:{len(pl)} Odds:{len(od)}")

    eng = EngineV7(bs, pl, od, games)
    mc  = MonteCarloV7(eng, n_sims=n_sims)
    mdl = ModelV7()
    mdl.connect_mc(mc); mdl.connect_engine(eng)

    train_m = MetricsV7(); eval_m = MetricsV7()
    proc = 0; skip = 0; cur_s = None

    for _, g in games.iterrows():
        s = g["season"]; is_ev = (s == eval_s)
        if s != cur_s:
            if cur_s is not None and not is_ev:
                train_m.report(label=f"FIN {cur_s}")
                mdl._flush_day_picks()
                for t in eng.elo:
                    eng.elo[t] = 0.75*eng.elo[t] + 0.25*ELO_INIT
            eng.reset_season(s); cur_s = s
            logger.info(f"Season: {s} {'[EVAL]' if is_ev else '[TRAIN]'}")

        feat = eng.compute(g)
        if feat is not None:
            pred = mdl.predict(feat, game=g)
            aw = g["home_win"]
            tk = eval_m if is_ev else train_m
            tk.add(pred["wp"], aw, g["margin"], g["game_id"], pred["is_vip"], pred["ev"])
            if not is_ev: mdl.add(feat, aw, g["game_id"])
            if pred["is_vip"]: mdl.save_vip_pick(g, pred)
            proc += 1
        else:
            skip += 1

        eng.update(g)
        if not is_ev and proc > 0 and proc % 300 == 0:
            if mdl.retrain(): logger.info(f"Retrained ({len(mdl.tX)} samples)")
        if proc > 0 and proc % ckpt == 0:
            tk = eval_m if is_ev else train_m
            tk.report(last_n=ckpt, label=f"{'EVAL' if is_ev else 'TRAIN'} #{proc}")

    mdl._flush_day_picks()
    if mdl.retrain(): logger.info(f"Final train ({len(mdl.tX)} samples)")
    mdl.save()

    print(f"\n{'='*65}")
    print(f"  RESUMEN FINAL V7 — THE GOD MODE")
    print(f"{'='*65}")
    print(f"  Procesados: {proc}  |  Omitidos: {skip}")
    print(f"  Tiempo: {(time.time()-t0)/60:.1f} min\n")

    print(f"  --- TRAIN ---")
    tf = train_m.report(label="TRAIN FINAL")
    print(f"\n  --- EVAL ({eval_s}) ---")
    ef = eval_m.report(label="EVAL FINAL")

    if mdl.trained and mdl.xgb is not None:
        print(f"\n  Top 25 Features:")
        imp = mdl.xgb.feature_importances_
        fi = sorted(zip(FEAT, imp), key=lambda x: x[1], reverse=True)
        for nm, im in fi[:25]:
            bar = "█" * int(im * 120)
            print(f"    {nm:24s} {im:.4f} {bar}")

    if os.path.exists(PICKS_CSV):
        picks_df = pd.read_csv(PICKS_CSV)
        print(f"\n  📋 VIP Picks: {len(picks_df)}")
        print(f"     Archivo: {PICKS_CSV}")
        if len(picks_df) > 0:
            print(f"     EV prom: {picks_df['Valor_Esperado'].astype(float).mean():.4f}")
            for r, c in picks_df["Razon"].value_counts().items():
                print(f"       {r}: {c}")
            for r, c in picks_df["Nivel_de_Riesgo"].value_counts().items():
                print(f"       Riesgo {r}: {c}")

    print(f"{'='*65}\n")
    return tf, ef


def main():
    p = argparse.ArgumentParser(description="NBA Pipeline V7 — The God Mode")
    p.add_argument("--seasons", nargs="+", default=TRAIN_SEASONS)
    p.add_argument("--eval", default=EVAL_SEASON)
    p.add_argument("--checkpoint", type=int, default=CHECKPOINT)
    p.add_argument("--db", default=DB_PATH)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--sims", type=int, default=DEFAULT_SIMS)
    a = p.parse_args()

    if a.eval_only:
        m = ModelV7()
        if m.load():
            print("Model V7 loaded.")
            if m.xgb: print(f"Features: {len(m.xgb.feature_importances_)}")
        else:
            print("No saved model")
        return

    run(a.seasons, a.eval, a.checkpoint, a.db, a.sims)


if __name__ == "__main__":
    main()
