#!/usr/bin/env python3
"""
NBA Training Pipeline V6 — SMART MONEY + FOUR FACTORS + FATIGUE
================================================================
Rediseño completo del motor de decisión.  Tres pilares:

  1. Smart Money / Reverse Line Movement
     - Detecta RLM cuando la línea se mueve contra el dinero público.
     - Market Probability Gap ≥ 8 % con confirmación de línea.

  2. Four Factors corregidos (por 100 posesiones)
     - Posesiones = 0.96 * (FGA + 0.44*FTA − ORB + TOV)
     - ORtg / DRtg por 100 posesiones reales
     - eFG% hot-streak (últimos 5) con peso extra

  3. Vulnerabilidad por Contexto  (Fatigue Factor)
     - Schedule Spot: 3-en-4 noches, B2B + viaje largo
     - Clutch Underperformance: Net Rating Q4 negativo

  4. Salida VIP dinámica
     - Umbral principal 72 % + EV positivo
     - Fallback Top-3 oportunidades (≥ 60 %, EV > 0)
     - Persistencia en picks_profesionales.csv

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
N_SIMS         = 10_000
VIP_THRESHOLD  = 0.72      # umbral principal
VIP_FALLBACK   = 0.60      # umbral mínimo para top-3 diario
MIN_EV         = 0.0       # expected value mínimo para emitir pick
MKT_GAP_MIN    = 0.08      # gap mínimo prob. modelo vs mercado
FP_PENALTY     = 5          # costo relativo de un falso positivo

TRAIN_SEASONS  = ["2019-2020", "2020-2021", "2021-2022", "2022-2023", "2023-2024"]
EVAL_SEASON    = "2024-2025"

ELO_INIT = 1500; ELO_K = 20; ELO_HCA = 55

os.makedirs("data", exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(
            stream=open(sys.stdout.fileno(), mode="w", encoding="utf-8", closefd=False)
        ),
        logging.FileHandler("data/training_log_v6.txt", "a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("V6-SmartMoney")

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
        # Q4 net rating (puntos propios − rival en 4to cuarto)
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
        # Posesiones CORREGIDAS (factor 0.96)
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


# ═══════════════════════ FEATURE NAMES ═══════════════════════════════════════
FEAT = [
    # --- FOUR FACTORS (10-game window, per 100 poss) ---
    "h_efg", "a_efg",
    "h_tov_pct", "a_tov_pct",
    "h_oreb_pct", "a_oreb_pct",
    "h_ft_rate", "a_ft_rate",
    "diff_efg", "diff_tov", "diff_oreb", "diff_ft",

    # --- Hot-streak eFG (últimos 5) con peso extra ---
    "h_efg_hot5", "a_efg_hot5",
    "h_efg_trend", "a_efg_trend",  # diferencia eFG 5-game vs 10-game

    # --- Pace & Ratings per 100 possessions ---
    "h_pace", "a_pace",
    "h_ortg", "a_ortg",
    "h_drtg", "a_drtg",
    "h_net_rtg", "a_net_rtg",

    # --- FATIGA / SCHEDULE SPOT ---
    "h_games_7d", "a_games_7d",
    "h_b2b", "a_b2b",
    "h_travel_miles", "a_travel_miles",
    "h_rest_days", "a_rest_days",
    "h_road_trip_len", "a_road_trip_len",
    "h_3in4", "a_3in4",  # 3 partidos en 4 noches

    # --- IMPACTO DE AUSENCIAS ---
    "h_missing_net_rtg", "a_missing_net_rtg",
    "h_missing_min", "a_missing_min",
    "h_missing_stars", "a_missing_stars",

    # --- TOP-8 JUGADORES (eficiencia + consistencia) ---
    "h_top8_off_rtg", "a_top8_off_rtg",
    "h_top8_def_rtg", "a_top8_def_rtg",
    "h_top8_net_rtg", "a_top8_net_rtg",
    "h_top8_consistency", "a_top8_consistency",

    # --- CLUTCH / Q4 ---
    "h_q4_net_avg", "a_q4_net_avg",

    # --- SMART MONEY ---
    "mkt_prob_home", "mkt_spread",
    "rlm_signal",          # +1 = RLM hacia home, -1 = RLM hacia away, 0 = sin señal
    "mkt_gap",             # prob_modelo − prob_mercado (signo relativo a home)
    "odds_move_home",      # cambio de cuota del home entre bookmakers (proxy)
    "consensus_spread",    # promedio de spreads de todos los bookmakers

    # --- ELO ---
    "elo_diff", "elo_exp",

    # --- CONTEXTO ---
    "is_conf", "season_progress",
]
N_FEAT = len(FEAT)


# ═══════════════════════ FEATURE ENGINE V6 ═══════════════════════════════════
class EngineV6:
    """Motor de features con Smart Money, Four Factors corregidos y fatiga."""

    def __init__(self, boxscores, players, odds, games):
        # --- Índices rápidos ---
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

        # Pre-index Q4 data from games table
        self.q4_idx = {}
        for _, r in games.iterrows():
            gid = r["game_id"]
            self.q4_idx[gid] = {
                "home_q4_net": r.get("home_q4_net", 0) or 0,
                "away_q4_net": r.get("away_q4_net", 0) or 0,
            }

        # --- Estado acumulado ---
        self.log = defaultdict(list)              # team_id → [game_entry]
        self.player_history = defaultdict(list)
        self.elo = {t: ELO_INIT for t in TEAM_ABBR}
        self.rec = defaultdict(lambda: {"w": 0, "l": 0})

        # --- Q4 clutch history per team ---
        self.q4_history = defaultdict(list)       # team_id → [net_q4]

        logger.info(
            f"EngineV6: {len(self.bs_idx)} bs, {len(self.odds_idx)} odds, {len(self.pl_idx)} pl"
        )

    # ------------------------------------------------------------------
    def reset_season(self, s):
        self.log = defaultdict(list)
        self.rec = defaultdict(lambda: {"w": 0, "l": 0})
        self.q4_history = defaultdict(list)

    # ------------------------------------------------------------------
    def compute(self, g):
        """Construye vector de features para un partido."""
        gid = g["game_id"]; hid = g["home_team_id"]; aid = g["away_team_id"]
        gd = g["game_date"]
        if hid not in TEAM_ABBR or aid not in TEAM_ABBR:
            return None

        hl, al = self.log[hid], self.log[aid]
        if len(hl) < 5 or len(al) < 5:
            return None

        f = {}

        # ══════ 1. FOUR FACTORS (ventana 10 y 5) ═══════════════════════
        h_ff10 = self._four_factors(hl, 10)
        a_ff10 = self._four_factors(al, 10)
        h_ff5  = self._four_factors(hl, 5)
        a_ff5  = self._four_factors(al, 5)

        f["h_efg"]     = h_ff10["efg"]
        f["a_efg"]     = a_ff10["efg"]
        f["h_tov_pct"] = h_ff10["tov_pct"]
        f["a_tov_pct"] = a_ff10["tov_pct"]
        f["h_oreb_pct"]= h_ff10["oreb_pct"]
        f["a_oreb_pct"]= a_ff10["oreb_pct"]
        f["h_ft_rate"] = h_ff10["ft_rate"]
        f["a_ft_rate"] = a_ff10["ft_rate"]

        f["diff_efg"]  = h_ff10["efg"] - a_ff10["efg"]
        f["diff_tov"]  = a_ff10["tov_pct"] - h_ff10["tov_pct"]
        f["diff_oreb"] = h_ff10["oreb_pct"] - a_ff10["oreb_pct"]
        f["diff_ft"]   = h_ff10["ft_rate"] - a_ff10["ft_rate"]

        # Hot-streak eFG% (últimos 5) con peso extra
        f["h_efg_hot5"]  = h_ff5["efg"]
        f["a_efg_hot5"]  = a_ff5["efg"]
        f["h_efg_trend"] = h_ff5["efg"] - h_ff10["efg"]
        f["a_efg_trend"] = a_ff5["efg"] - a_ff10["efg"]

        # ══════ 2. PACE & RATINGS PER 100 POSS ════════════════════════
        for pfx, lg in [("h", hl), ("a", al)]:
            pr = self._pace_ratings(lg, 10)
            f[f"{pfx}_pace"]    = pr["pace"] / 100        # normalizado
            f[f"{pfx}_ortg"]    = pr["ortg"] / 120
            f[f"{pfx}_drtg"]    = pr["drtg"] / 120
            f[f"{pfx}_net_rtg"] = (pr["ortg"] - pr["drtg"]) / 30

        # ══════ 3. FATIGA / SCHEDULE SPOT ══════════════════════════════
        for pfx, lg, tid in [("h", hl, hid), ("a", al, aid)]:
            fat = self._fatiga(lg, gd, tid)
            f[f"{pfx}_games_7d"]      = fat["games_7d"]
            f[f"{pfx}_b2b"]           = fat["b2b"]
            f[f"{pfx}_travel_miles"]  = fat["travel_miles"]
            f[f"{pfx}_rest_days"]     = fat["rest_days"]
            f[f"{pfx}_road_trip_len"] = fat["road_trip"]
            f[f"{pfx}_3in4"]          = fat["three_in_four"]

        # ══════ 4. IMPACTO DE AUSENCIAS ════════════════════════════════
        for pfx, tid, lg in [("h", hid, hl), ("a", aid, al)]:
            mi = self._missing_impact(tid, gid, lg[-1] if lg else None)
            f[f"{pfx}_missing_net_rtg"] = mi["net_rtg"]
            f[f"{pfx}_missing_min"]     = mi["minutes_lost"] / 240
            f[f"{pfx}_missing_stars"]   = mi["stars_out"]

        # ══════ 5. TOP-8 EFICIENCIA ════════════════════════════════════
        for pfx, tid in [("h", hid), ("a", aid)]:
            t8 = self._top8_efficiency(tid, 10)
            f[f"{pfx}_top8_off_rtg"]     = t8["off_rtg"] / 120
            f[f"{pfx}_top8_def_rtg"]     = t8["def_rtg"] / 120
            f[f"{pfx}_top8_net_rtg"]     = (t8["off_rtg"] - t8["def_rtg"]) / 30
            f[f"{pfx}_top8_consistency"] = t8["consistency"]

        # ══════ 6. CLUTCH Q4 ═══════════════════════════════════════════
        hq = self.q4_history[hid]
        aq = self.q4_history[aid]
        f["h_q4_net_avg"] = np.mean(hq[-10:]) / 10 if len(hq) >= 3 else 0
        f["a_q4_net_avg"] = np.mean(aq[-10:]) / 10 if len(aq) >= 3 else 0

        # ══════ 7. SMART MONEY ═════════════════════════════════════════
        sm = self._smart_money(gid)
        f["mkt_prob_home"]   = sm["mkt_prob_home"]
        f["mkt_spread"]      = sm["mkt_spread"] / 10
        f["rlm_signal"]      = sm["rlm_signal"]
        f["mkt_gap"]         = 0.0  # se rellena en predict (necesita wp del modelo)
        f["odds_move_home"]  = sm["odds_move_home"]
        f["consensus_spread"]= sm["consensus_spread"] / 10

        # ══════ 8. ELO ═════════════════════════════════════════════════
        he, ae = self.elo[hid], self.elo[aid]
        f["elo_diff"] = (he - ae) / 100
        f["elo_exp"]  = 1 / (1 + 10 ** (-(he - ae + ELO_HCA) / 400))

        # ══════ 9. CONTEXTO ════════════════════════════════════════════
        f["is_conf"] = 1 if (hid in EAST) == (aid in EAST) else 0
        total = self.rec[hid]["w"] + self.rec[hid]["l"]
        f["season_progress"] = min(total / 82, 1.0)

        vec = np.array([f.get(n, 0.0) for n in FEAT], dtype=np.float64)
        return np.nan_to_num(vec)

    # ------------------------------------------------------------------
    def update(self, g):
        """Actualiza historial tras cada partido."""
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
                "opp_id": aid if ih else hid,
                "poss": poss,
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

        # Q4 clutch
        q4 = self.q4_idx.get(gid, {})
        self.q4_history[hid].append(q4.get("home_q4_net", 0))
        self.q4_history[aid].append(q4.get("away_q4_net", 0))

        # Player history
        for p in pl:
            pid = p["player_id"]
            tid = REV_TEAM.get(p.get("nba_tid"))
            if tid not in (hid, aid):
                continue
            pm   = p.get("plus_minus") or 0
            mins = p.get("minutes_decimal") or 0
            self.player_history[pid].append({
                "gid": gid, "date": gd, "team": tid,
                "mins": mins, "starter": p.get("starter") == 1,
                "pts": p.get("pts") or 0, "reb": p.get("reb") or 0,
                "ast": p.get("ast") or 0, "plus_minus": pm,
                "fgm": p.get("fgm") or 0, "fga": p.get("fga") or 0,
                "fg3m": p.get("fg3m") or 0,
                "fta": p.get("fta") or 0, "tov": p.get("tov") or 0,
                "oreb": p.get("oreb") or 0,
            })

        # ELO (Margin-of-Victory adjusted)
        he, ae = self.elo[hid], self.elo[aid]
        exp = 1 / (1 + 10 ** (-(he - ae + ELO_HCA) / 400))
        act = 1.0 if hw else 0.0
        mov = min(np.log1p(abs(hs - aws)) * 0.7, 2.5)
        ac  = 2.2 / ((abs(he - ae) * 0.001) + 2.2)
        k   = ELO_K * mov * ac
        self.elo[hid] += k * (act - exp)
        self.elo[aid] += k * ((1 - act) - (1 - exp))

    # ══════════════════════ HELPERS ═══════════════════════════════════════

    def _four_factors(self, lg, window=10):
        """Four Factors con posesiones corregidas (0.96 factor)."""
        r = lg[-window:]
        if not r:
            return {"efg": 0.52, "tov_pct": 0.13, "oreb_pct": 0.22, "ft_rate": 0.20}

        def _safe_mean(key, default):
            vals = [g.get(key) for g in r if g.get(key) is not None]
            return np.mean(vals) if vals else default

        fga  = _safe_mean("fga", 85)
        fgm  = _safe_mean("fgm", 37)
        fg3m = _safe_mean("fg3m", 11)
        fta  = _safe_mean("fta", 22)
        oreb = _safe_mean("oreb", 10)
        dreb = _safe_mean("dreb", 33)
        tov  = _safe_mean("tov", 14)

        # Posesiones corregidas
        poss = 0.96 * (fga + 0.44 * fta - oreb + tov)

        efg     = (fgm + 0.5 * fg3m) / fga if fga > 0 else 0.52
        tov_pct = tov / poss if poss > 0 else 0.13
        oreb_pct = oreb / (oreb + dreb) if (oreb + dreb) > 0 else 0.22
        ft_rate  = fta / fga if fga > 0 else 0.20

        return {"efg": efg, "tov_pct": tov_pct, "oreb_pct": oreb_pct, "ft_rate": ft_rate}

    def _pace_ratings(self, lg, window=10):
        """Pace, ORtg, DRtg por 100 posesiones reales."""
        r = lg[-window:]
        if not r:
            return {"pace": 98, "ortg": 110, "drtg": 110}

        paces, ortgs, drtgs = [], [], []
        for g in r:
            poss = g.get("poss")
            pts  = g.get("pts")
            opp  = g.get("opp")
            if poss and poss > 50 and pts is not None and opp is not None:
                paces.append(poss)
                ortgs.append(pts / poss * 100)
                drtgs.append(opp / poss * 100)

        return {
            "pace": np.mean(paces) if paces else 98,
            "ortg": np.mean(ortgs) if ortgs else 110,
            "drtg": np.mean(drtgs) if drtgs else 110,
        }

    def _fatiga(self, lg, game_date, team_id):
        """Fatiga: B2B, 3-en-4, viaje, etc."""
        if not lg:
            return {"games_7d": 0, "b2b": 0, "travel_miles": 0,
                    "rest_days": 3, "road_trip": 0, "three_in_four": 0}

        last = lg[-1]
        last_date = pd.Timestamp(last["date"]) if isinstance(last["date"], str) else last["date"]
        rest_days = (game_date - last_date).days

        week_ago = game_date - timedelta(days=7)
        games_7d = sum(
            1 for g in lg
            if (pd.Timestamp(g["date"]) if isinstance(g["date"], str) else g["date"]) >= week_ago
        )

        b2b = 1 if rest_days <= 1 else 0

        # 3 partidos en 4 noches (incluyendo hoy)
        four_nights_ago = game_date - timedelta(days=3)
        games_in_4 = sum(
            1 for g in lg
            if (pd.Timestamp(g["date"]) if isinstance(g["date"], str) else g["date"]) >= four_nights_ago
        )
        three_in_four = 1 if games_in_4 >= 2 else 0  # 2 prev + today = 3

        # Millas viajadas
        travel_miles = 0
        if len(lg) >= 2:
            prev = lg[-2]
            prev_loc = team_id if prev["home"] else prev["opp_id"]
            curr_loc = team_id if last["home"] else last["opp_id"]
            if prev_loc in COORDS and curr_loc in COORDS:
                c1, c2 = COORDS[prev_loc], COORDS[curr_loc]
                travel_miles = _haversine(c1[0], c1[1], c2[0], c2[1])

        # Road trip length
        road_trip = 0
        for g in reversed(lg):
            if not g["home"]:
                road_trip += 1
            else:
                break

        return {
            "games_7d":      min(games_7d / 4, 1.0),
            "b2b":           b2b,
            "travel_miles":  min(travel_miles / 3000, 1.0),
            "rest_days":     min(rest_days, 7),
            "road_trip":     min(road_trip / 5, 1.0),
            "three_in_four": three_in_four,
        }

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

    def _top8_efficiency(self, team_id, window=10):
        team_games = self.log[team_id][-window:]
        if not team_games:
            return {"off_rtg": 110, "def_rtg": 110, "consistency": 0.5}

        ps = defaultdict(lambda: {"mins": 0, "pm": 0, "games": 0,
                                   "fgm": 0, "fga": 0, "fg3m": 0,
                                   "fta": 0, "oreb": 0, "tov": 0, "pts": 0})
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
                    "mins": s["mins"] / s["games"],
                    "pm":   s["pm"]   / s["games"],
                    "pts":  s["pts"]  / s["games"],
                    "poss": (s["fga"] + 0.44*s["fta"] - s["oreb"] + s["tov"]) / s["games"],
                })
        avg.sort(key=lambda x: x["mins"], reverse=True)
        top8 = avg[:8]
        if not top8:
            return {"off_rtg": 110, "def_rtg": 110, "consistency": 0.5}

        # Per-100-poss ratings for top8 weighted by minutes
        ortgs, drtgs, weights = [], [], []
        for p in top8:
            poss = max(p["poss"], 1)
            o = p["pts"] / poss * 100
            d = max(110 - p["pm"] / max(p["mins"], 1) * 48 / 2, 80)  # proxy
            ortgs.append(o)
            drtgs.append(d)
            weights.append(p["mins"])

        w = np.array(weights)
        w = w / w.sum() if w.sum() > 0 else np.ones(len(w)) / len(w)
        off_rtg = np.average(ortgs, weights=w)
        def_rtg = np.average(drtgs, weights=w)
        pm_vals = [p["pm"] for p in top8]
        consistency = 1 - min(np.std(pm_vals) / 15, 1) if len(pm_vals) > 1 else 0.5

        return {"off_rtg": off_rtg, "def_rtg": def_rtg, "consistency": consistency}

    # ──────────────────────────────────────────────────────────────────
    #  SMART MONEY ANALYSIS
    # ──────────────────────────────────────────────────────────────────
    def _smart_money(self, gid):
        """
        Analiza cuotas para detectar Reverse Line Movement y valor de mercado.

        Lógica RLM:
        - Agrupamos cuotas h2h por bookmaker.
        - Si la mayoría de bookmakers favorecen al home (cuota baja) pero
          uno o más mueven la línea CONTRA el público (cuota sube para el favorito),
          interpretamos RLM.
        - En la práctica con datos estáticos, usamos la dispersión de cuotas
          entre bookmakers como proxy del movimiento de línea.
        """
        odds = self.odds_idx.get(gid, [])
        if not odds:
            return {
                "mkt_prob_home": 0.5, "mkt_spread": 0,
                "rlm_signal": 0, "odds_move_home": 0, "consensus_spread": 0,
            }

        home_name = odds[0].get("home_team", "")

        # --- Recolectar probabilidades h2h y spreads por bookmaker ---
        bk_home_probs = {}  # bookmaker → prob home
        bk_spreads    = {}  # bookmaker → spread home
        all_home_probs = []
        all_spreads    = []

        for o in odds:
            mk  = o.get("market", "")
            pr  = o.get("outcome_price")
            pt  = o.get("outcome_point")
            nm  = o.get("outcome_name", "")
            bk  = o.get("bookmaker", "unknown")

            is_home = (nm == home_name or home_name in nm)

            if mk == "h2h" and pr is not None:
                # Conversión cuotas americanas → probabilidad implícita
                prob = self._odds_to_prob(pr)
                if prob is not None:
                    if is_home:
                        bk_home_probs[bk] = prob
                        all_home_probs.append(prob)
                    else:
                        # Probabilidad del away → complemento = home
                        bk_home_probs.setdefault(bk, 1 - prob)
                        all_home_probs.append(1 - prob)

            elif mk == "spreads" and pt is not None:
                if is_home:
                    bk_spreads[bk] = pt
                    all_spreads.append(pt)

        mkt_prob = np.mean(all_home_probs) if all_home_probs else 0.5
        mkt_spread = np.mean(all_spreads) if all_spreads else 0
        consensus_spread = np.median(all_spreads) if all_spreads else 0

        # --- RLM Detection ---
        # Proxy: si la dispersión de probabilidades entre bookmakers es alta
        # Y la mediana indica favorito pero algún bookmaker se aleja,
        # eso sugiere "sharp money" moviendo la línea.
        rlm_signal = 0
        odds_move = 0

        if len(all_home_probs) >= 3:
            prob_std = np.std(all_home_probs)
            prob_median = np.median(all_home_probs)
            prob_max = max(all_home_probs)
            prob_min = min(all_home_probs)
            odds_move = prob_max - prob_min  # rango de movimiento

            # RLM: el consenso dice home favorito (>55%) pero al menos un
            # bookmaker pone prob mucho más baja → dinero profesional en away
            if prob_median > 0.55 and prob_min < prob_median - 0.05:
                rlm_signal = -1  # RLM hacia away
            elif prob_median < 0.45 and prob_max > prob_median + 0.05:
                rlm_signal = 1   # RLM hacia home

        return {
            "mkt_prob_home":   mkt_prob,
            "mkt_spread":      mkt_spread,
            "rlm_signal":      rlm_signal,
            "odds_move_home":  odds_move,
            "consensus_spread": consensus_spread,
        }

    @staticmethod
    def _odds_to_prob(price):
        """Convierte cuota americana o decimal a probabilidad implícita."""
        if price is None:
            return None
        try:
            price = float(price)
        except (ValueError, TypeError):
            return None

        if price == 0:
            return None

        # Detectar formato: si |price| >= 100, es americano; si < 15, decimal
        if abs(price) >= 100:
            # Americano
            if price > 0:
                return 100 / (price + 100)
            else:
                return abs(price) / (abs(price) + 100)
        elif price >= 1.01:
            # Decimal
            return 1 / price
        else:
            return None


# ═══════════════════════ MONTE CARLO V6 ══════════════════════════════════════
class MonteCarloV6:
    """Simulación por jugador con pace-adjusted ratings y hot-streak."""

    def __init__(self, engine, n_sims=N_SIMS):
        self.engine = engine
        self.n_sims = n_sims

    def run(self, game, feat):
        gid = game["game_id"]
        hid = game["home_team_id"]
        aid = game["away_team_id"]

        h_players = self._get_expected_players(hid, gid)
        a_players = self._get_expected_players(aid, gid)

        if len(h_players) < 5 or len(a_players) < 5:
            return self._fallback(feat)

        # Pace promedio esperado (ambos equipos, últimos 10)
        h_pace = self._team_pace(hid)
        a_pace = self._team_pace(aid)
        avg_pace = (h_pace + a_pace) / 2

        # Distribuciones de eficiencia (ORtg per 100 poss per player)
        h_effs = np.array([p["ortg_100"] for p in h_players[:10]])
        a_effs = np.array([p["ortg_100"] for p in a_players[:10]])
        h_mins = np.array([p["expected_mins"] for p in h_players[:10]])
        a_mins = np.array([p["expected_mins"] for p in a_players[:10]])
        h_stds = np.array([p["ortg_std"] for p in h_players[:10]])
        a_stds = np.array([p["ortg_std"] for p in a_players[:10]])

        # Normalizar pesos de minutos
        h_wt = h_mins / h_mins.sum() if h_mins.sum() > 0 else np.ones(len(h_mins)) / len(h_mins)
        a_wt = a_mins / a_mins.sum() if a_mins.sum() > 0 else np.ones(len(a_mins)) / len(a_mins)

        wins = 0
        margins = np.empty(self.n_sims)

        for i in range(self.n_sims):
            # Muestrear eficiencia de cada jugador (normal con su media y std)
            h_samp = np.random.normal(h_effs, h_stds)
            a_samp = np.random.normal(a_effs, a_stds)

            h_team_ortg = np.dot(h_wt, h_samp)
            a_team_ortg = np.dot(a_wt, a_samp)

            sim_pace = avg_pace + np.random.normal(0, 2.5)
            h_pts = sim_pace * h_team_ortg / 100 + np.random.normal(0, 1.2)
            a_pts = sim_pace * a_team_ortg / 100 + np.random.normal(0, 1.2)

            # Home-court advantage (~+3 pts)
            h_pts += np.random.normal(1.5, 0.5)

            if h_pts > a_pts:
                wins += 1
            margins[i] = h_pts - a_pts

        wp = wins / self.n_sims
        return {
            "wp": wp,
            "em": np.mean(margins),
            "ms": np.std(margins),
            "conf": max(0, 1 - np.std(margins) / 18),
            "n_players": (len(h_players), len(a_players)),
        }

    def _get_expected_players(self, team_id, game_id):
        team_games = self.engine.log[team_id][-5:]
        if not team_games:
            return []

        ps = defaultdict(lambda: {
            "games": 0, "total_mins": 0, "total_pts": 0,
            "total_poss": 0, "ortg_vals": [],
        })

        for g in team_games:
            gid = g["gid"]
            if gid == game_id:
                continue
            for p in self.engine.pl_idx.get(gid, []):
                if REV_TEAM.get(p.get("nba_tid")) != team_id:
                    continue
                pid  = p["player_id"]
                mins = p.get("minutes_decimal") or 0
                if mins < 5:
                    continue
                pts  = p.get("pts") or 0
                fga  = p.get("fga") or 0
                fta  = p.get("fta") or 0
                tov  = p.get("tov") or 0
                oreb = p.get("oreb") or 0
                poss_used = 0.96 * (fga + 0.44 * fta - oreb + tov)

                # ORtg individual ≈ pts / poss * 100
                ortg = (pts / poss_used * 100) if poss_used > 3 else 100

                ps[pid]["games"] += 1
                ps[pid]["total_mins"] += mins
                ps[pid]["total_pts"]  += pts
                ps[pid]["total_poss"] += poss_used
                ps[pid]["ortg_vals"].append(ortg)

        result = []
        for pid, s in ps.items():
            if s["games"] >= 2:
                avg_mins = s["total_mins"] / s["games"]
                avg_ortg = np.mean(s["ortg_vals"]) if s["ortg_vals"] else 100
                std_ortg = np.std(s["ortg_vals"]) if len(s["ortg_vals"]) > 1 else 8
                result.append({
                    "player_id":     pid,
                    "expected_mins": avg_mins,
                    "ortg_100":      avg_ortg,
                    "ortg_std":      max(std_ortg, 3),  # piso de variabilidad
                    "games_played":  s["games"],
                })

        result.sort(key=lambda x: x["expected_mins"], reverse=True)
        return result[:10]

    def _team_pace(self, team_id):
        lg = self.engine.log[team_id][-10:]
        paces = [g["poss"] for g in lg if g.get("poss") and g["poss"] > 50]
        return np.mean(paces) if paces else 98

    def _fallback(self, feat):
        try:
            idx = FEAT.index("elo_exp")
            wp = feat[idx]
        except (ValueError, IndexError):
            wp = 0.5
        return {"wp": wp, "em": 0, "ms": 12, "conf": 0.5, "n_players": (0, 0)}


# ═══════════════════════ MODELO XGBOOST V6 ═══════════════════════════════════
class ModelV6:
    """XGBoost con penalización FP, integración Smart Money y VIP dinámico."""

    def __init__(self):
        self.xgb = None
        self.scaler = StandardScaler()
        self.mc = None
        self.engine = None  # se conecta después para acceso a smart money
        self.trained = False
        self.tX, self.ty = [], []

        # Historial de picks del día (para top-3 fallback)
        self._day_picks = []
        self._current_day = None

        logger.info(f"ModelV6 init | FP penalty {FP_PENALTY}x | VIP≥{VIP_THRESHOLD:.0%} | Fallback≥{VIP_FALLBACK:.0%}")

    def connect_mc(self, mc):
        self.mc = mc

    def connect_engine(self, eng):
        self.engine = eng

    def add(self, X, y, game_id):
        self.tX.append(X)
        self.ty.append(y)

    def retrain(self):
        if len(self.tX) < 300:
            return False

        X = np.array(self.tX)
        y = np.array(self.ty)
        X_scaled = self.scaler.fit_transform(X)

        n_pos = max(np.sum(y), 1)
        n_neg = len(y) - n_pos
        spw = (n_neg / n_pos) * 2.5  # penalización extra FP

        self.xgb = xgb.XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.025,
            subsample=0.8,
            colsample_bytree=0.75,
            reg_alpha=0.2,
            reg_lambda=1.5,
            min_child_weight=5,
            gamma=0.1,
            scale_pos_weight=spw,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
        )
        self.xgb.fit(X_scaled, y)
        self.trained = True
        logger.info(f"XGBoost trained | {len(self.tX)} samples | spw={spw:.2f}")
        return True

    def predict(self, X, game=None):
        """
        Predicción con integración Smart Money:
        - wp base del XGBoost / Monte Carlo
        - mkt_gap = wp − mkt_prob  (solo VIP si gap ≥ 8%)
        - RLM check: si RLM contradice nuestro pick, no es VIP
        - EV = wp * cuota − 1
        """
        gid = game["game_id"] if game is not None else None
        game_date = game["game_date"] if game is not None else None

        # --- WP base ---
        if not self.trained:
            if self.mc and game is not None:
                mc_res = self.mc.run(game, X)
                wp = mc_res["wp"]
                conf = mc_res["conf"]
            else:
                try:
                    wp = X[FEAT.index("elo_exp")]
                except (ValueError, IndexError):
                    wp = 0.5
                conf = 0.5
        else:
            X_scaled = self.scaler.transform(X.reshape(1, -1))
            wp_xgb = self.xgb.predict_proba(X_scaled)[0][1]

            # Monte Carlo como segundo opinión
            if self.mc and game is not None:
                mc_res = self.mc.run(game, X)
                wp_mc = mc_res["wp"]
                conf  = mc_res["conf"]
                # Ensemble: 65% XGBoost + 35% Monte Carlo
                wp = 0.65 * wp_xgb + 0.35 * wp_mc
            else:
                wp = wp_xgb
                conf = 0.6

        # --- Rellenar mkt_gap en el vector de features ---
        mkt_prob_idx = FEAT.index("mkt_prob_home")
        mkt_prob = X[mkt_prob_idx]
        mkt_gap = wp - mkt_prob
        try:
            gap_idx = FEAT.index("mkt_gap")
            X[gap_idx] = mkt_gap
        except ValueError:
            pass

        # --- Smart Money signals ---
        rlm_idx = FEAT.index("rlm_signal")
        rlm = X[rlm_idx]

        spread_idx = FEAT.index("mkt_spread")
        mkt_spread = X[spread_idx] * 10  # desnormalizar

        # Cuota implícita → para calcular EV
        # cuota decimal ≈ 1 / prob_mercado (sin vig, aprox)
        fair_odds = 1 / max(mkt_prob, 0.01) if wp > 0.5 else 1 / max(1 - mkt_prob, 0.01)
        # Ajustar por vigorish típico (~5%)
        market_odds = fair_odds * 0.95
        ev = wp * market_odds - 1 if wp > 0.5 else (1 - wp) * market_odds - 1

        # --- FATIGUE TRAP: penalizar favorito cansado ---
        pick_home = wp > 0.5
        fatigue_trap = False
        if pick_home:
            # Check home fatigue
            b2b_idx = FEAT.index("h_b2b")
            three_in_four_idx = FEAT.index("h_3in4")
            q4_idx = FEAT.index("h_q4_net_avg")
            if X[b2b_idx] == 1 or X[three_in_four_idx] == 1:
                if X[q4_idx] < 0:
                    fatigue_trap = True
                    wp = wp * 0.92  # reduce confianza un 8%
        else:
            b2b_idx = FEAT.index("a_b2b")
            three_in_four_idx = FEAT.index("a_3in4")
            q4_idx = FEAT.index("a_q4_net_avg")
            if X[b2b_idx] == 1 or X[three_in_four_idx] == 1:
                if X[q4_idx] < 0:
                    fatigue_trap = True
                    wp = 1 - ((1 - wp) * 0.92)  # adjust for away pick

        # --- VIP Decision ---
        is_vip = False
        vip_reason = ""

        # Condición principal: confianza >= 72% + gap >= 8% + RLM no contradice
        high_conf = max(wp, 1 - wp) >= VIP_THRESHOLD

        # RLM contradiction check
        rlm_contradicts = False
        if pick_home and rlm == -1:
            rlm_contradicts = True   # modelo dice home, pero smart money dice away
        elif not pick_home and rlm == 1:
            rlm_contradicts = True   # modelo dice away, pero smart money dice home

        gap_ok = abs(mkt_gap) >= MKT_GAP_MIN
        ev_positive = ev > MIN_EV

        if high_conf and gap_ok and not rlm_contradicts and ev_positive:
            is_vip = True
            vip_reason = "HIGH_CONF"
        elif high_conf and not gap_ok and not rlm_contradicts and ev_positive:
            # Confianza alta pero sin gap suficiente → no VIP pero candidato top-3
            vip_reason = "CONF_NO_GAP"

        # RLM bonus: si RLM CONFIRMA nuestro pick, baja el umbral
        if not is_vip and not rlm_contradicts:
            rlm_confirms = (pick_home and rlm == 1) or (not pick_home and rlm == -1)
            if rlm_confirms and max(wp, 1 - wp) >= 0.68 and ev_positive:
                is_vip = True
                vip_reason = "RLM_CONFIRMED"

        result = {
            "wp": wp,
            "conf": conf,
            "is_vip": is_vip,
            "vip_reason": vip_reason,
            "mkt_gap": mkt_gap,
            "ev": ev,
            "rlm": rlm,
            "fatigue_trap": fatigue_trap,
            "mkt_odds": market_odds,
            "pick_home": pick_home,
        }

        # --- Top-3 fallback tracking ---
        if game_date is not None:
            day_str = str(game_date)[:10]
            if self._current_day != day_str:
                self._flush_day_picks()
                self._current_day = day_str
                self._day_picks = []

            self._day_picks.append({
                "game": game,
                "result": result,
                "confidence": max(wp, 1 - wp),
                "ev": ev,
            })

        return result

    def _flush_day_picks(self):
        """Al cambiar de día, si no hubo VIPs, emite top-3 fallback."""
        if not self._day_picks:
            return

        vip_count = sum(1 for p in self._day_picks if p["result"]["is_vip"])
        if vip_count == 0:
            # Buscar top-3 candidatos con conf >= 60% y EV > 0
            candidates = [
                p for p in self._day_picks
                if p["confidence"] >= VIP_FALLBACK and p["ev"] > MIN_EV
            ]
            candidates.sort(key=lambda x: x["ev"], reverse=True)
            for c in candidates[:3]:
                c["result"]["is_vip"] = True
                c["result"]["vip_reason"] = "TOP3_FALLBACK"
                self._save_pick(c["game"], c["result"])

    def _save_pick(self, game, result):
        """Guarda un pick VIP en el CSV."""
        if game is None:
            return
        hid = game["home_team_id"]
        aid = game["away_team_id"]
        h_abbr = TEAM_ABBR.get(hid, "???")
        a_abbr = TEAM_ABBR.get(aid, "???")
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
            "Razon": result.get("vip_reason", ""),
            "RLM": result.get("rlm", 0),
            "Fatigue_Trap": result.get("fatigue_trap", False),
        }

        file_exists = os.path.exists(PICKS_CSV)
        with open(PICKS_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def save_vip_pick(self, game, result):
        """Interfaz pública para guardar pick VIP."""
        self._save_pick(game, result)

    def save(self, path=None):
        path = path or f"{MODEL_DIR}/nba_model_v6.pkl"
        with open(path, "wb") as f:
            pickle.dump({
                "xgb": self.xgb, "scaler": self.scaler,
                "trained": self.trained, "n": len(self.tX),
            }, f)

    def load(self, path=None):
        path = path or f"{MODEL_DIR}/nba_model_v6.pkl"
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            d = pickle.load(f)
        if d.get("trained"):
            self.xgb = d["xgb"]
            self.scaler = d["scaler"]
            self.trained = True
            return True
        return False


# ═══════════════════════ METRICS V6 ══════════════════════════════════════════
class MetricsV6:
    def __init__(self):
        self.all_probs   = []
        self.all_acts    = []
        self.all_margs   = []
        self.all_gids    = []
        self.all_is_vip  = []
        self.all_ev      = []

    def add(self, prob, actual, margin, gid, is_vip=False, ev=0):
        self.all_probs.append(prob)
        self.all_acts.append(actual)
        self.all_margs.append(margin)
        self.all_gids.append(gid)
        self.all_is_vip.append(is_vip)
        self.all_ev.append(ev)

    def report(self, last_n=None, label=""):
        if not self.all_probs:
            return {}
        n = last_n or len(self.all_probs)
        probs  = self.all_probs[-n:]
        acts   = self.all_acts[-n:]
        is_vip = self.all_is_vip[-n:]
        evs    = self.all_ev[-n:]

        preds = [1 if p > 0.5 else 0 for p in probs]
        acc   = accuracy_score(acts, preds)
        brier = brier_score_loss(acts, probs)

        vi = [i for i, v in enumerate(is_vip) if v]
        n_vip = len(vi)

        if n_vip > 0:
            vp = [probs[i] for i in vi]
            va = [acts[i] for i in vi]
            vpreds = [1 if p > 0.5 else 0 for p in vp]
            vip_acc = accuracy_score(va, vpreds)
            vip_brier = brier_score_loss(va, vp)
            # ROI aproximado basado en EV
            vip_ev_avg = np.mean([evs[i] for i in vi])
            vip_correct = sum(1 for j in range(n_vip) if vpreds[j] == va[j])
        else:
            vip_acc = 0; vip_brier = 1; vip_ev_avg = 0; vip_correct = 0

        print(f"\n{'='*65}")
        print(f"  CHECKPOINT {label} ({n} juegos)")
        print(f"{'='*65}")
        print(f"  📊 GLOBAL: Acc {acc:.1%} | Brier {brier:.4f}")
        print(f"  🏆 VIP: {n_vip}/{n} ({n_vip/n*100:.1f}%)")
        if n_vip > 0:
            fp = n_vip - vip_correct
            print(f"    Acc:   {vip_acc:.1%}  {'🎯' if vip_acc >= 0.70 else '❌'}")
            print(f"    Brier: {vip_brier:.4f}")
            print(f"    EV avg:{vip_ev_avg:+.3f}")
            print(f"    FP:    {fp}/{n_vip} ({fp/n_vip*100:.1f}%)")
        print(f"{'='*65}")

        return {
            "acc": acc, "brier": brier,
            "vip_acc": vip_acc, "vip_brier": vip_brier,
            "vip_pct": n_vip / n if n > 0 else 0,
            "vip_ev": vip_ev_avg,
        }


# ═══════════════════════ PIPELINE V6 ════════════════════════════════════════
def run(train_s, eval_s, ckpt, db, n_sims=N_SIMS):
    t0 = time.time()
    all_s = train_s + [eval_s]

    print(f"\n{'='*65}")
    print(f"  NBA PIPELINE V6 — SMART MONEY + FOUR FACTORS + FATIGUE")
    print(f"  Train: {', '.join(train_s)}")
    print(f"  Eval:  {eval_s}")
    print(f"  Features: {N_FEAT}")
    print(f"  Sims: {n_sims:,} | VIP≥{VIP_THRESHOLD:.0%} | Fallback≥{VIP_FALLBACK:.0%}")
    print(f"{'='*65}\n")

    # Limpiar CSV de picks previos
    if os.path.exists(PICKS_CSV):
        os.remove(PICKS_CSV)

    dl = DataLoader(db)
    games = dl.load_games(all_s)
    bs    = dl.load_boxscores()
    pl    = dl.load_players()
    od    = dl.load_odds()
    dl.close()
    logger.info(f"Games:{len(games)} BS:{len(bs)} PL:{len(pl)} Odds:{len(od)}")

    eng = EngineV6(bs, pl, od, games)
    mc  = MonteCarloV6(eng, n_sims=n_sims)
    mdl = ModelV6()
    mdl.connect_mc(mc)
    mdl.connect_engine(eng)

    train_m = MetricsV6()
    eval_m  = MetricsV6()

    proc = 0; skip = 0; cur_s = None

    for _, g in games.iterrows():
        s = g["season"]
        is_ev = (s == eval_s)

        if s != cur_s:
            if cur_s is not None and not is_ev:
                train_m.report(label=f"FIN {cur_s}")
                # Flush top-3 fallback del último día
                mdl._flush_day_picks()
                for t in eng.elo:
                    eng.elo[t] = 0.75 * eng.elo[t] + 0.25 * ELO_INIT
            eng.reset_season(s)
            cur_s = s
            logger.info(f"Season: {s} {'[EVAL]' if is_ev else '[TRAIN]'}")

        feat = eng.compute(g)

        if feat is not None:
            pred = mdl.predict(feat, game=g)
            aw = g["home_win"]

            tk = eval_m if is_ev else train_m
            tk.add(pred["wp"], aw, g["margin"], g["game_id"],
                   pred["is_vip"], pred["ev"])

            if not is_ev:
                mdl.add(feat, aw, g["game_id"])

            # Guardar VIP picks
            if pred["is_vip"]:
                mdl.save_vip_pick(g, pred)

            proc += 1
        else:
            skip += 1

        eng.update(g)

        if not is_ev and proc > 0 and proc % 300 == 0:
            if mdl.retrain():
                logger.info(f"Retrained ({len(mdl.tX)} samples)")

        if proc > 0 and proc % ckpt == 0:
            tk = eval_m if is_ev else train_m
            metrics = tk.report(last_n=ckpt, label=f"{'EVAL' if is_ev else 'TRAIN'} #{proc}")

            if metrics.get("vip_acc", 1) < 0.65 and metrics.get("vip_pct", 0) > 0.03:
                logger.warning("⚠️  VIP accuracy baja — revisando parámetros de fatiga y RLM")

    # Flush último día
    mdl._flush_day_picks()

    # Entrenamiento final
    if mdl.retrain():
        logger.info(f"Final train ({len(mdl.tX)} samples)")
    mdl.save()

    print(f"\n{'='*65}")
    print(f"  RESUMEN FINAL V6 — SMART MONEY + FOUR FACTORS + FATIGUE")
    print(f"{'='*65}")
    print(f"  Procesados: {proc}  |  Omitidos: {skip}")
    print(f"  Tiempo: {(time.time() - t0) / 60:.1f} min\n")

    print(f"  --- TRAIN ---")
    tf = train_m.report(label="TRAIN FINAL")

    print(f"\n  --- EVAL ({eval_s}) ---")
    ef = eval_m.report(label="EVAL FINAL")

    # Feature importance
    if mdl.trained and mdl.xgb is not None:
        print(f"\n  Top 20 Features:")
        imp = mdl.xgb.feature_importances_
        fi = sorted(zip(FEAT, imp), key=lambda x: x[1], reverse=True)
        for nm, im in fi[:20]:
            bar = "█" * int(im * 120)
            print(f"    {nm:22s} {im:.4f} {bar}")

    # Resumen de picks CSV
    if os.path.exists(PICKS_CSV):
        picks_df = pd.read_csv(PICKS_CSV)
        print(f"\n  📋 VIP Picks guardados: {len(picks_df)}")
        print(f"     Archivo: {PICKS_CSV}")
        if len(picks_df) > 0:
            print(f"     EV promedio: {picks_df['Valor_Esperado'].astype(float).mean():.4f}")
            reasons = picks_df["Razon"].value_counts()
            for r, c in reasons.items():
                print(f"       {r}: {c}")

    print(f"{'='*65}\n")
    return tf, ef


def main():
    p = argparse.ArgumentParser(description="NBA Pipeline V6 — Smart Money + Four Factors + Fatigue")
    p.add_argument("--seasons", nargs="+", default=TRAIN_SEASONS)
    p.add_argument("--eval", default=EVAL_SEASON)
    p.add_argument("--checkpoint", type=int, default=CHECKPOINT)
    p.add_argument("--db", default=DB_PATH)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--sims", type=int, default=N_SIMS)
    a = p.parse_args()

    if a.sims != N_SIMS:
        # Update sim count in MonteCarloV6 instances (applied at runtime)
        pass  # handled below via a.sims

    if a.eval_only:
        m = ModelV6()
        if m.load():
            print("Model V6 loaded.")
            if m.xgb:
                print(f"Features: {len(m.xgb.feature_importances_)}")
        else:
            print("No saved model")
        return

    run(a.seasons, a.eval, a.checkpoint, a.db, a.sims)


if __name__ == "__main__":
    main()
