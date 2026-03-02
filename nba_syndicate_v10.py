#!/usr/bin/env python3
"""
NBA SYNDICATE V10 — DEEP EDGE ARCHITECTURE
============================================
Evolución de V9 (XGBoost 61.3% Global / 71.7% Diamond) → PyTorch End-to-End.

V10 Innovations:
  1. PYTORCH END-TO-END: Temporal BiLSTM + Team Embeddings + Cross-Attention Fusion
  2. MULTI-TASK LEARNING: Win + Margin + Market Line + CLV Sign (adaptive weights)
  3. META-LEARNER DE INCERTIDUMBRE: Gaussian NLL for calibrated sizing
  4. TEMPORAL ANTI-LEAKAGE: Walk-forward with 14-day gap, 7-day feature gap
  5. BAYESIAN EXECUTION: Kelly + risk aversion + uncertainty penalty
  6. PLATT SCALING: Learned temperature for calibrated probabilities
  7. FULL METRICS: ECE, Brier Skill Score, Sharpe, Max Drawdown, Profit Factor

Requisitos: pip install numpy pandas scikit-learn xgboost requests torch
"""

import os, sys, time, sqlite3, logging, argparse, warnings, pickle, csv, json, math
from datetime import datetime, timedelta, date
from collections import defaultdict
from math import radians, cos, sin, asin, sqrt

import numpy as np
import pandas as pd

try:
    import requests
except ImportError:
    print("pip install requests"); sys.exit(1)

try:
    import xgboost as xgb
    from sklearn.metrics import accuracy_score, brier_score_loss
    from sklearn.preprocessing import StandardScaler
except ImportError:
    print("pip install xgboost scikit-learn"); sys.exit(1)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader as TorchDataLoader
    TORCH_AVAILABLE = True
except ImportError:
    print("⚠️  PyTorch not available — fallback to XGBoost only. Install: pip install torch")
    TORCH_AVAILABLE = False

warnings.filterwarnings("ignore")

# ═══════════════════════ CONFIG ══════════════════════════════════════════════
DB_PATH        = "data/nba_historical.db"
MODEL_DIR      = "models"
PICKS_CSV_LIVE = "PICKS_V10_DEEP_EDGE.csv"
PICKS_CSV_BT   = "picks_profesionales_v10.csv"
JSON_RADAR_DIR = "data/radar_v10"
DEFAULT_SIMS   = 12_000
CHECKPOINT     = 10

DIAMOND_THRESHOLD  = 0.78
DIAMOND_EV_MIN     = 0.10
FALLBACK_THRESHOLD = 0.65
MKT_GAP_MIN        = 0.08
FP_PENALTY          = 5

SPREAD_VALUE_MIN   = 1.5
TOTAL_VALUE_MIN    = 3.0
DEAD_NUMBER_RANGE  = 0.8
CLV_BOOST          = 0.03
STEAM_MOVE_THRESH  = 0.04

TRAIN_SEASONS = [
    "2019-2020", "2020-2021", "2021-2022",
    "2022-2023", "2023-2024", "2024-2025",
]
CURRENT_SEASON = "2025-2026"

ELO_INIT = 1500; ELO_K = 20; ELO_HCA = 55
MATCHUP_TOP5_RANK = 5; MATCHUP_PENALTY = 0.07
TIMEZONE_PENALTY  = 2; HEAVY_LEGS_EFG  = 0.015
FT_DEP_THRESH = 0.32; LOW_FOUL_RANK = 5
PLAYOFF_URGENCY_PROGRESS = 0.75
ELIMINATED_WIN_PCT       = 0.30

ALTITUDE_MAP = {139: 5280, 160: 4226}
ALTITUDE_FATIGUE_THRESHOLD = 4000
ALTITUDE_TOTAL_ADJUST = -2.5
REF_HIGH_FT_OVER_ADJUST = 3.0
REF_LOW_FT_UNDER_ADJUST = -2.0

SPORTS_API_URL = "https://v1.basketball.api-sports.io"
ODDS_API_URL   = "https://api.the-odds-api.com/v4/sports/basketball_nba"

# ═══════════════════════ V10 DEEP LEARNING CONFIG ════════════════════════════
TEMPORAL_SEQ_LEN   = 15
TEMPORAL_FEATURES  = 9
TEMPORAL_GAP_DAYS  = 7
TEAM_EMBED_DIM     = 32
TEAM_EMBED_PROJ    = 64
N_TEAMS            = 30
MARKET_FEATURES_DIM = 8
CONTEXT_FEATURES_DIM = 6
MATCHUP_STATS_DIM  = 16

LEARNING_RATE      = 1e-3
WEIGHT_DECAY       = 1e-4
BATCH_SIZE         = 64
N_EPOCHS           = 30
PATIENCE           = 5
DROPOUT            = 0.3
TEMPORAL_HIDDEN    = 128

WALK_FORWARD_FOLDS = [
    {"train_end": "2022-06-30", "val_start": "2022-10-15", "label": "Val 2022-23"},
    {"train_end": "2023-06-30", "val_start": "2023-10-15", "label": "Val 2023-24"},
    {"train_end": "2024-06-30", "val_start": "2024-10-15", "label": "Val 2024-25"},
]
GAP_DAYS = 14

ALLSTAR_DATES = {
    "2019-2020": "2020-02-16", "2020-2021": "2021-03-07",
    "2021-2022": "2022-02-20", "2022-2023": "2023-02-19",
    "2023-2024": "2024-02-18", "2024-2025": "2025-02-16",
    "2025-2026": "2026-02-15",
}

# Historical ELO priors for embedding initialization
ELO_PRIORS = {
    132: 1480, 133: 1580, 134: 1430, 135: 1400, 136: 1440, 137: 1530,
    138: 1510, 139: 1540, 140: 1380, 141: 1520, 142: 1470, 143: 1480,
    144: 1460, 145: 1510, 146: 1500, 147: 1490, 148: 1530, 149: 1510,
    150: 1440, 151: 1510, 152: 1560, 153: 1470, 154: 1480, 155: 1500,
    156: 1410, 157: 1470, 158: 1420, 159: 1440, 160: 1430, 161: 1390,
}

def _get_key(name):
    k = os.environ.get(name, "")
    if not k:
        logger.warning(f"⚠️  {name} not set in environment. Live features disabled.")
    return k

os.makedirs("data", exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(JSON_RADAR_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(
            stream=open(sys.stdout.fileno(), mode="w", encoding="utf-8", closefd=False)
        ),
        logging.FileHandler("data/training_log_v10.txt", "a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("V10-DeepEdge")

# ═══════════════════════ TEAM CONSTANTS ══════════════════════════════════════
TEAM_ABBR = {
    132:"ATL",133:"BOS",134:"BKN",135:"CHA",136:"CHI",137:"CLE",138:"DAL",
    139:"DEN",140:"DET",141:"GSW",142:"HOU",143:"IND",144:"LAC",145:"LAL",
    146:"MEM",147:"MIA",148:"MIL",149:"MIN",150:"NOP",151:"NYK",152:"OKC",
    153:"ORL",154:"PHI",155:"PHX",156:"POR",157:"SAC",158:"SAS",159:"TOR",
    160:"UTA",161:"WAS",
}
ABBR_TO_ID = {v: k for k, v in TEAM_ABBR.items()}
NBA_IDS = {
    132:1610612737,133:1610612738,134:1610612751,135:1610612766,136:1610612741,
    137:1610612739,138:1610612742,139:1610612743,140:1610612765,141:1610612744,
    142:1610612745,143:1610612754,144:1610612746,145:1610612747,146:1610612763,
    147:1610612748,148:1610612749,149:1610612750,150:1610612740,151:1610612752,
    152:1610612760,153:1610612753,154:1610612755,155:1610612756,156:1610612757,
    157:1610612758,158:1610612759,159:1610612761,160:1610612762,161:1610612764,
}
REV_TEAM = {v: k for k, v in NBA_IDS.items()}
TEAM_ID_LIST = sorted(TEAM_ABBR.keys())
TEAM_TO_IDX = {tid: i for i, tid in enumerate(TEAM_ID_LIST)}

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
TEAM_TZ = {
    132:-5,133:-5,134:-5,135:-5,136:-6,137:-5,138:-6,139:-7,140:-5,
    141:-8,142:-6,143:-5,144:-8,145:-8,146:-6,147:-5,148:-6,149:-6,
    150:-6,151:-5,152:-6,153:-5,154:-5,155:-7,156:-8,157:-8,158:-6,
    159:-5,160:-7,161:-5,
}
ODDS_NAME_MAP = {
    "Atlanta Hawks":132,"Boston Celtics":133,"Brooklyn Nets":134,
    "Charlotte Hornets":135,"Chicago Bulls":136,"Cleveland Cavaliers":137,
    "Dallas Mavericks":138,"Denver Nuggets":139,"Detroit Pistons":140,
    "Golden State Warriors":141,"Houston Rockets":142,"Indiana Pacers":143,
    "Los Angeles Clippers":144,"Los Angeles Lakers":145,"Memphis Grizzlies":146,
    "Miami Heat":147,"Milwaukee Bucks":148,"Minnesota Timberwolves":149,
    "New Orleans Pelicans":150,"New York Knicks":151,"Oklahoma City Thunder":152,
    "Orlando Magic":153,"Philadelphia 76ers":154,"Phoenix Suns":155,
    "Portland Trail Blazers":156,"Sacramento Kings":157,"San Antonio Spurs":158,
    "Toronto Raptors":159,"Utah Jazz":160,"Washington Wizards":161,
}

def _haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    d = lat2 - lat1; dl = lon2 - lon1
    a = sin(d/2)**2 + cos(lat1)*cos(lat2)*sin(dl/2)**2
    return 2 * 3956 * asin(sqrt(a))


# ═══════════════════════ LIVE API CLIENTS ════════════════════════════════════

class SportsAPIClient:
    def __init__(self):
        self.key = _get_key("SPORTS_API_KEY")
        self.base = SPORTS_API_URL
        self.headers = {"x-apisports-key": self.key} if self.key else {}
        self.remaining = None

    def _get(self, endpoint, params=None):
        if not self.key: return None
        url = f"{self.base}/{endpoint}"
        try:
            r = requests.get(url, headers=self.headers, params=params or {}, timeout=15)
            self.remaining = r.headers.get("x-ratelimit-requests-remaining")
            if r.status_code == 200:
                data = r.json()
                if data.get("errors") and len(data["errors"]) > 0:
                    logger.warning(f"Sports API error: {data['errors']}"); return None
                return data.get("response", [])
            else:
                logger.warning(f"Sports API HTTP {r.status_code}"); return None
        except Exception as e:
            logger.warning(f"Sports API exception: {e}"); return None

    def get_todays_games(self, date_str=None):
        dt = date_str or date.today().isoformat()
        games = self._get("games", {"league": "12", "season": "2025-2026", "date": dt})
        if not games: return []
        result = []
        for g in games:
            status = g.get("status", {}).get("short", "")
            if status in ("NS", ""):
                home = g.get("teams", {}).get("home", {})
                away = g.get("teams", {}).get("away", {})
                result.append({
                    "api_game_id": g.get("id"),
                    "home_id": home.get("id"), "away_id": away.get("id"),
                    "home_name": home.get("name", ""), "away_name": away.get("name", ""),
                    "date": dt, "status": status,
                })
        logger.info(f"📡 Sports API: {len(result)} games today ({dt})")
        return result

    def get_injured_players(self, team_api_id, season="2025-2026"):
        return set()

    def get_standings(self, season="2025-2026"):
        data = self._get("standings", {"league": "12", "season": season})
        if not data: return {}
        standings = {}
        for group in data:
            entries = group if isinstance(group, list) else [group] if isinstance(group, dict) else []
            for entry in entries:
                team = entry.get("team", {}); tid = team.get("id")
                games_data = entry.get("games", {})
                win_data = entry.get("win", games_data.get("win", {}))
                lose_data = entry.get("lose", games_data.get("lose", {}))
                wins = win_data.get("total", 0) if isinstance(win_data, dict) else 0
                losses = lose_data.get("total", 0) if isinstance(lose_data, dict) else 0
                total = wins + losses
                standings[tid] = {"wins": wins, "losses": losses,
                                  "pct": wins / total if total > 0 else 0.5, "total": total}
        logger.info(f"📊 Standings loaded: {len(standings)} teams")
        return standings


class OddsAPIClient:
    """V10: Fetches h2h + spreads + totals, tracks CLV, detects steam moves."""

    def __init__(self):
        self.key = _get_key("ODDS_API_KEY")
        self.base = ODDS_API_URL

    def get_live_odds(self):
        if not self.key: return []
        try:
            r = requests.get(f"{self.base}/odds", params={
                "apiKey": self.key, "regions": "us",
                "markets": "h2h,spreads,totals", "oddsFormat": "american",
            }, timeout=15)
            remaining = r.headers.get("x-requests-remaining", "?")
            logger.info(f"📡 Odds API: HTTP {r.status_code} | remaining={remaining}")
            return r.json() if r.status_code == 200 else []
        except Exception as e:
            logger.warning(f"Odds API exception: {e}"); return []

    def get_opening_odds(self):
        if not self.key: return []
        try:
            r = requests.get(f"{self.base}/odds-history", params={
                "apiKey": self.key, "regions": "us",
                "markets": "h2h,spreads,totals", "oddsFormat": "american",
                "date": (datetime.utcnow() - timedelta(hours=12)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }, timeout=15)
            return r.json() if r.status_code == 200 else []
        except: return []

    def parse_game_odds(self, odds_data, opening_data=None):
        opener_lookup = {}
        if opening_data:
            for event in opening_data:
                hname = event.get("home_team", ""); aname = event.get("away_team", "")
                hid = ODDS_NAME_MAP.get(hname); aid = ODDS_NAME_MAP.get(aname)
                if hid and aid:
                    oh2h=[]; osp=[]; otot=[]
                    for bk in event.get("bookmakers", []):
                        for mkt in bk.get("markets", []):
                            mk = mkt.get("key", "")
                            for oc in mkt.get("outcomes", []):
                                pr=oc.get("price"); pt=oc.get("point"); ih=(oc.get("name")==hname)
                                if mk=="h2h" and pr is not None:
                                    prob = self._american_to_prob(pr)
                                    if prob: oh2h.append(prob if ih else 1-prob)
                                elif mk=="spreads" and pt is not None and ih: osp.append(pt)
                                elif mk=="totals" and pt is not None and oc.get("name")=="Over": otot.append(pt)
                    opener_lookup[(hid, aid)] = {
                        "open_prob": np.mean(oh2h) if oh2h else None,
                        "open_spread": np.mean(osp) if osp else None,
                        "open_total": np.mean(otot) if otot else None,
                    }
        result = {}
        for event in odds_data:
            home_name = event.get("home_team", ""); away_name = event.get("away_team", "")
            hid = ODDS_NAME_MAP.get(home_name); aid = ODDS_NAME_MAP.get(away_name)
            if not hid or not aid: continue
            bk_probs={}; all_hp=[]; all_sp=[]; all_tot=[]
            per_bk_p={}; per_bk_s={}; per_bk_t={}
            for bookmaker in event.get("bookmakers", []):
                bk_name = bookmaker.get("key", "unknown")
                bk_h=None; bk_s=None; bk_t=None
                for market in bookmaker.get("markets", []):
                    mk = market.get("key", "")
                    for oc in market.get("outcomes", []):
                        name=oc.get("name",""); pr=oc.get("price"); pt=oc.get("point")
                        ih = (name == home_name)
                        if mk=="h2h" and pr is not None:
                            prob = self._american_to_prob(pr)
                            if prob is not None:
                                if ih: bk_probs[bk_name]=prob; all_hp.append(prob); bk_h=prob
                                else: all_hp.append(1-prob)
                        elif mk=="spreads" and pt is not None and ih:
                            all_sp.append(pt); bk_s=pt
                        elif mk=="totals" and pt is not None and name=="Over":
                            all_tot.append(pt); bk_t=pt
                if bk_h is not None: per_bk_p[bk_name]=bk_h
                if bk_s is not None: per_bk_s[bk_name]=bk_s
                if bk_t is not None: per_bk_t[bk_name]=bk_t
            if not all_hp: continue
            mkt_prob=np.mean(all_hp); mkt_spread=np.mean(all_sp) if all_sp else 0
            consensus=np.median(all_sp) if all_sp else 0; mkt_total=np.mean(all_tot) if all_tot else 220.0
            rlm=0; odds_move=0
            if len(all_hp)>=3:
                med=np.median(all_hp); mx=max(all_hp); mn=min(all_hp); odds_move=mx-mn
                if med>0.55 and mn<med-0.05: rlm=-1
                elif med<0.45 and mx>med+0.05: rlm=1
            steam=0; opener=opener_lookup.get((hid,aid),{})
            open_prob=opener.get("open_prob")
            if open_prob is not None and len(per_bk_p)>=3:
                shifts=[per_bk_p[b]-open_prob for b in per_bk_p]
                if sum(1 for s in shifts if s>STEAM_MOVE_THRESH)>=3: steam=1
                elif sum(1 for s in shifts if s<-STEAM_MOVE_THRESH)>=3: steam=-1
            clv_home=0.0; open_spread=opener.get("open_spread")
            if open_spread is not None and all_sp: clv_home=open_spread-mkt_spread
            pvs=0.0
            if len(per_bk_p)>=4:
                sharp_bks=["pinnacle","circa","betcris","bookmaker"]
                public_bks=["draftkings","fanduel","betmgm","caesars"]
                sv=[per_bk_p[b] for b in sharp_bks if b in per_bk_p]
                pv=[per_bk_p[b] for b in public_bks if b in per_bk_p]
                if sv and pv: pvs=np.mean(sv)-np.mean(pv)
            result[(hid,aid)] = {
                "mkt_prob_home":mkt_prob, "mkt_spread":mkt_spread, "mkt_total":mkt_total,
                "rlm_signal":rlm, "odds_move_home":odds_move, "consensus_spread":consensus,
                "home_name":home_name, "away_name":away_name, "n_bookmakers":len(bk_probs),
                "raw_home_probs":all_hp, "steam_signal":steam, "clv_home":clv_home,
                "open_prob":open_prob or mkt_prob, "open_spread":open_spread or mkt_spread,
                "open_total":opener.get("open_total") or mkt_total,
                "public_vs_sharp":pvs, "per_book_spreads":per_bk_s, "per_book_totals":per_bk_t,
            }
        logger.info(f"📊 V10 Parsed odds for {len(result)} games")
        return result

    @staticmethod
    def _american_to_prob(price):
        try: price = float(price)
        except: return None
        if price == 0: return None
        return 100/(price+100) if price > 0 else abs(price)/(abs(price)+100)


# ═══════════════════════ DATA LOADER ═════════════════════════════════════════
class DataLoaderDB:
    def __init__(self, db=DB_PATH):
        self.conn = sqlite3.connect(db); self.conn.row_factory = sqlite3.Row

    def load_games(self, seasons):
        ph = ",".join("?" * len(seasons))
        df = pd.read_sql_query(
            f"""SELECT game_id, season, date_local AS game_date,
                       home_team_id, away_team_id, home_score, away_score,
                       home_q1,home_q2,home_q3,home_q4,
                       away_q1,away_q2,away_q3,away_q4, nba_game_id
                FROM games WHERE season IN ({ph})
                  AND status_short IN ('FT','AOT')
                  AND home_score IS NOT NULL AND away_score IS NOT NULL
                ORDER BY date_local, game_id""",
            self.conn, params=seasons)
        df["game_date"] = pd.to_datetime(df["game_date"])
        df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
        df["margin"] = df["home_score"] - df["away_score"]
        df["total_pts"] = df["home_score"] + df["away_score"]
        for side in ("home", "away"):
            q4 = df[f"{side}_q4"].fillna(0).astype(float)
            opp = df[f"{'away' if side=='home' else 'home'}_q4"].fillna(0).astype(float)
            df[f"{side}_q4_net"] = q4 - opp
        return df

    def load_boxscores(self):
        df = pd.read_sql_query(
            """SELECT nba_game_id, team_id AS nba_tid, game_id,
                      fgm,fga,fg3m,fg3a,ftm,fta,oreb,dreb,reb,ast,stl,blk,tov,pf,pts
               FROM game_team_stats WHERE pts IS NOT NULL""", self.conn)
        df["team_id"] = df["nba_tid"].map(REV_TEAM)
        df["poss"] = 0.96 * (df["fga"] + 0.44 * df["fta"] - df["oreb"] + df["tov"])
        return df

    def load_players(self):
        df = pd.read_sql_query(
            """SELECT nba_game_id, player_id, team_id AS nba_tid, game_id,
                      player_name, minutes_decimal,
                      (CASE WHEN minutes_decimal > 25 THEN 1 ELSE 0 END) AS starter,
                      fgm,fga,fg3m,fg3a,ftm,fta,oreb,dreb,reb,ast,stl,blk,tov,pf,pts,
                      plus_minus
               FROM game_player_stats WHERE minutes_decimal > 0""", self.conn)
        df["team_id"] = df["nba_tid"].map(REV_TEAM)
        return df

    def load_odds(self):
        return pd.read_sql_query(
            """SELECT game_id_mapped AS game_id, home_team, away_team,
                      bookmaker, market, outcome_name, outcome_price, outcome_point
               FROM odds_historical WHERE game_id_mapped IS NOT NULL""", self.conn)

    def close(self): self.conn.close()


# ═══════════════════════ V10 FEATURE VECTOR ═══════════════════════════════════
FEAT = [
    # V8 Core (Four Factors + Pace + Ratings)
    "h_efg","a_efg","h_tov_pct","a_tov_pct","h_oreb_pct","a_oreb_pct",
    "h_ft_rate","a_ft_rate","diff_efg","diff_tov","diff_oreb","diff_ft",
    "h_efg_hot5","a_efg_hot5","h_efg_trend","a_efg_trend",
    "h_pace","a_pace","h_ortg","a_ortg","h_drtg","a_drtg",
    "h_net_rtg","a_net_rtg",
    # Matchup DNA
    "matchup_ortg_vs_drtg","matchup_drtg_vs_ortg",
    "h_ast_rate","a_ast_rate","h_stl_rate","a_stl_rate",
    "matchup_ast_vs_stl","matchup_dna_penalty",
    # Fatigue
    "h_games_7d","a_games_7d","h_b2b","a_b2b",
    "h_travel_miles_7d","a_travel_miles_7d",
    "h_rest_days","a_rest_days","h_road_trip_len","a_road_trip_len",
    "h_3in4","a_3in4","h_4in6","a_4in6",
    "h_tz_crossed_7d","a_tz_crossed_7d","h_heavy_legs","a_heavy_legs",
    # Ausencias
    "h_missing_net_rtg","a_missing_net_rtg",
    "h_missing_min","a_missing_min","h_missing_stars","a_missing_stars",
    # Top-8
    "h_top8_off_rtg","a_top8_off_rtg","h_top8_def_rtg","a_top8_def_rtg",
    "h_top8_net_rtg","a_top8_net_rtg","h_top8_consistency","a_top8_consistency",
    # Clutch & Value Trap
    "h_q4_net_avg","a_q4_net_avg",
    "h_ft_dependency","a_ft_dependency","h_opp_pf_rate","a_opp_pf_rate",
    "value_trap_flag",
    # Market V8
    "mkt_prob_home","mkt_spread","rlm_signal","mkt_gap",
    "odds_move_home","consensus_spread",
    # ELO & Context
    "elo_diff","elo_exp","is_conf","season_progress",
    "h_playoff_urgency","a_playoff_urgency","season_weight",
    # V9 features
    "h_3p_rate","a_3p_rate","h_3p_pct","a_3p_pct",
    "h_opp_3p_pct","a_opp_3p_pct",
    "matchup_3p_attack_vs_def","matchup_3p_def_vs_attack",
    "h_fastbreak_rate","a_fastbreak_rate","h_ast_to_tov","a_ast_to_tov",
    "matchup_transition_diff",
    "altitude_factor","h_schedule_density","a_schedule_density",
    "steam_signal","clv_home","open_vs_current_prob","public_vs_sharp",
    "proj_pace","proj_total","mkt_total","total_edge",
    "ref_foul_tendency",
    # ═══ V10 NEW ═══
    "h_road_pct_14d","a_road_pct_14d",
    "opponent_rest_advantage",
    "days_since_allstar_decay",
    "playoff_intensity_score",
    "line_movement_range","sharp_ratio_normalized","market_volatility",
]
N_FEAT = len(FEAT)


# ═══════════════════════ V10 PYTORCH ARCHITECTURE ════════════════════════════

if TORCH_AVAILABLE:

    class TemporalBranch(nn.Module):
        """Bidirectional LSTM on last 15 games + temporal attention."""
        def __init__(self, input_dim=TEMPORAL_FEATURES, hidden_dim=TEMPORAL_HIDDEN,
                     num_layers=2, dropout=0.2):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                                batch_first=True, bidirectional=True, dropout=dropout)
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim * 2, 64), nn.Tanh(), nn.Linear(64, 1))
            self.output_dim = hidden_dim * 2

        def forward(self, x, mask=None):
            output, _ = self.lstm(x)
            attn_scores = self.attention(output).squeeze(-1)
            if mask is not None:
                attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
            attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)
            return (output * attn_weights).sum(dim=1)

    class TeamEmbedding(nn.Module):
        """Trainable team embeddings with ELO prior initialization."""
        def __init__(self, n_teams=N_TEAMS, embed_dim=TEAM_EMBED_DIM,
                     proj_dim=TEAM_EMBED_PROJ):
            super().__init__()
            self.embedding = nn.Embedding(n_teams, embed_dim)
            self.projection = nn.Linear(embed_dim, proj_dim)
            self.output_dim = proj_dim
            with torch.no_grad():
                for tid, idx in TEAM_TO_IDX.items():
                    elo = ELO_PRIORS.get(tid, ELO_INIT)
                    self.embedding.weight[idx] += (elo - ELO_INIT) / 200.0 * 0.1

        def forward(self, team_idx):
            return F.gelu(self.projection(self.embedding(team_idx)))

    class MatchupBranch(nn.Module):
        """Fusion of home/away embeddings + diff stats with skip connection."""
        def __init__(self, embed_dim=TEAM_EMBED_PROJ, stats_dim=MATCHUP_STATS_DIM):
            super().__init__()
            input_dim = embed_dim * 2 + stats_dim
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, 128), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(64, 32))
            self.skip = nn.Linear(input_dim, 32)
            self.norm = nn.LayerNorm(32)
            self.output_dim = 32

        def forward(self, home_emb, away_emb, diff_stats):
            x = torch.cat([home_emb, away_emb, diff_stats], dim=-1)
            return self.norm(self.mlp(x) + self.skip(x))

    class MarketIntelligenceBranch(nn.Module):
        def __init__(self, input_dim=MARKET_FEATURES_DIM):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, 32), nn.GELU(), nn.Dropout(0.15),
                nn.Linear(32, 16))
            self.output_dim = 16

        def forward(self, x): return self.mlp(x)

    class ContextBranch(nn.Module):
        def __init__(self, input_dim=CONTEXT_FEATURES_DIM):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, 24), nn.GELU(), nn.Dropout(0.15),
                nn.Linear(24, 12))
            self.output_dim = 12

        def forward(self, x): return self.mlp(x)

    class CrossAttentionFusion(nn.Module):
        """Cross-attention fusion across all branches."""
        def __init__(self, dims):
            super().__init__()
            total_dim = sum(dims)
            self.query_proj = nn.Linear(total_dim, 64)
            self.key_proj = nn.Linear(total_dim, 64)
            self.value_proj = nn.Linear(total_dim, 64)
            self.output_dim = 64

        def forward(self, *branch_outputs):
            x = torch.cat(branch_outputs, dim=-1)
            Q = self.query_proj(x); K = self.key_proj(x); V = self.value_proj(x)
            attn = F.softmax(Q * K / (64 ** 0.5), dim=-1)
            return attn * V

    class PlattScaling(nn.Module):
        """Learned temperature for calibrated probabilities."""
        def __init__(self):
            super().__init__()
            self.temperature = nn.Parameter(torch.ones(1) * 1.5)

        def forward(self, logits):
            return torch.sigmoid(logits / self.temperature.clamp(min=0.1))

    class NBADeepEdgeModel(nn.Module):
        """
        V10 End-to-End multi-branch architecture with:
        - Temporal LSTM (bidirectional + attention)
        - Team Embeddings (ELO-initialized)
        - Matchup branch (fusion + skip)
        - Market Intelligence branch
        - Context branch
        - Flat feature encoder (V9 compatibility)
        - Cross-Attention Fusion → Dense head
        - Multi-task outputs: win, margin, market_line, clv_sign
        - Platt Scaling + Homoscedastic uncertainty weighting
        """
        def __init__(self, flat_feat_dim=N_FEAT):
            super().__init__()
            self.temporal = TemporalBranch()
            self.team_embed = TeamEmbedding()
            self.matchup = MatchupBranch()
            self.market = MarketIntelligenceBranch()
            self.context = ContextBranch()
            self.flat_encoder = nn.Sequential(
                nn.Linear(flat_feat_dim, 128), nn.GELU(), nn.Dropout(DROPOUT),
                nn.Linear(128, 64), nn.BatchNorm1d(64))
            flat_out = 64
            branch_dims = [self.temporal.output_dim, self.matchup.output_dim,
                           self.market.output_dim, self.context.output_dim, flat_out]
            self.fusion = CrossAttentionFusion(branch_dims)
            fd = self.fusion.output_dim
            self.dense = nn.Sequential(
                nn.Linear(fd, 256), nn.GELU(), nn.BatchNorm1d(256), nn.Dropout(DROPOUT),
                nn.Linear(256, 128), nn.GELU(), nn.BatchNorm1d(128), nn.Dropout(DROPOUT),
                nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 32))
            self.residual = nn.Linear(fd, 32)
            self.win_head = nn.Linear(32, 1)
            self.margin_head = nn.Linear(32, 1)
            self.market_head = nn.Linear(32, 1)
            self.clv_head = nn.Linear(32, 1)
            self.platt = PlattScaling()
            self.log_vars = nn.Parameter(torch.zeros(4))

        def forward(self, flat_features, temporal_seq, temporal_mask,
                    home_idx, away_idx, matchup_stats, market_features,
                    context_features):
            t_out = self.temporal(temporal_seq, temporal_mask)
            h_emb = self.team_embed(home_idx)
            a_emb = self.team_embed(away_idx)
            mu_out = self.matchup(h_emb, a_emb, matchup_stats)
            mk_out = self.market(market_features)
            cx_out = self.context(context_features)
            fl_out = self.flat_encoder(flat_features)
            fused = self.fusion(t_out, mu_out, mk_out, cx_out, fl_out)
            dense_out = self.dense(fused) + self.residual(fused)
            win_logits = self.win_head(dense_out).squeeze(-1)
            return {
                "win_prob": self.platt(win_logits),
                "win_logits": win_logits,
                "margin_pred": self.margin_head(dense_out).squeeze(-1),
                "market_pred": self.market_head(dense_out).squeeze(-1),
                "clv_pred": torch.sigmoid(self.clv_head(dense_out).squeeze(-1)),
            }

        def compute_loss(self, outputs, targets):
            """Multi-task loss with homoscedastic uncertainty weighting."""
            losses = {}
            losses["win"] = F.binary_cross_entropy(
                outputs["win_prob"], targets["win"], reduction="mean")
            if targets.get("margin") is not None:
                losses["margin"] = F.huber_loss(
                    outputs["margin_pred"], targets["margin"], delta=5.0, reduction="mean")
            if targets.get("market_line") is not None:
                losses["market_line"] = F.mse_loss(
                    outputs["market_pred"], targets["market_line"], reduction="mean")
            if targets.get("clv_sign") is not None:
                losses["clv_sign"] = F.binary_cross_entropy(
                    outputs["clv_pred"], targets["clv_sign"], reduction="mean")
            total_loss = 0
            for i, (task, loss) in enumerate(losses.items()):
                precision = torch.exp(-self.log_vars[min(i, 3)])
                total_loss = total_loss + precision * loss + self.log_vars[min(i, 3)]
            return total_loss, losses

    class UncertaintyMetaLearner(nn.Module):
        """Predicts calibrated mu_correction + sigma for Bayesian sizing."""
        def __init__(self, input_dim=12):
            super().__init__()
            self.feature_extractor = nn.Sequential(
                nn.Linear(input_dim, 64), nn.ELU(), nn.Dropout(0.2),
                nn.Linear(64, 32), nn.ELU())
            self.mu_head = nn.Linear(32, 1)
            self.log_sigma_head = nn.Linear(32, 1)

        def forward(self, x):
            features = self.feature_extractor(x)
            mu_correction = self.mu_head(features).squeeze(-1)
            log_sigma = self.log_sigma_head(features).squeeze(-1).clamp(-3, 3)
            sigma = torch.exp(log_sigma) + 1e-8
            return mu_correction, sigma

        def nll_loss(self, y_true, y_pred_base, mu_correction, sigma):
            y_pred = y_pred_base + mu_correction
            nll = 0.5 * math.log(2 * math.pi) + torch.log(sigma) + \
                  (y_true - y_pred)**2 / (2 * sigma**2)
            return nll.mean()

    class BayesianExecutionEngine:
        """Bayesian-optimal bet sizing with uncertainty-aware Kelly."""
        def __init__(self, bankroll=10000, max_risk=0.05):
            self.bankroll = bankroll; self.max_risk = max_risk
            self.transaction_cost = 0.02

        def optimal_bet(self, prediction, sigma, market_line):
            p_adj = prediction / (1 + sigma * 2)
            implied = 1.0 / max(market_line, 1.01)
            edge = p_adj - implied - self.transaction_cost
            if edge <= 0: return 0.0
            variance = p_adj * (1 - p_adj) + sigma**2
            kelly = edge / (variance * max(market_line, 1.01))
            gamma = 0.5 + sigma * 2
            risk_penalty = gamma * kelly * sigma
            f_star = kelly * (1 - risk_penalty)
            return min(max(f_star * 0.25, 0), self.max_risk)

        def compute_bet_size(self, prediction, sigma, market_odds):
            return self.optimal_bet(prediction, sigma, market_odds) * self.bankroll

    class NBAGameDataset(Dataset):
        """PyTorch Dataset for V10 deep training."""
        def __init__(self, data_list):
            self.data = data_list

        def __len__(self): return len(self.data)

        def __getitem__(self, idx):
            d = self.data[idx]
            return {
                "flat": torch.tensor(d["flat"], dtype=torch.float32),
                "temporal_seq": torch.tensor(d["temporal_seq"], dtype=torch.float32),
                "temporal_mask": torch.tensor(d["temporal_mask"], dtype=torch.float32),
                "home_idx": torch.tensor(d["home_idx"], dtype=torch.long),
                "away_idx": torch.tensor(d["away_idx"], dtype=torch.long),
                "matchup_stats": torch.tensor(d["matchup_stats"], dtype=torch.float32),
                "market_features": torch.tensor(d["market_features"], dtype=torch.float32),
                "context_features": torch.tensor(d["context_features"], dtype=torch.float32),
                "win": torch.tensor(d["win"], dtype=torch.float32),
                "margin": torch.tensor(d["margin"], dtype=torch.float32),
                "market_line": torch.tensor(d["market_line"], dtype=torch.float32),
                "clv_sign": torch.tensor(d["clv_sign"], dtype=torch.float32),
            }


# ═══════════════════════ ENGINE V10 ══════════════════════════════════════════
class EngineV10:
    """V10 Feature Engine: flat vector + structured inputs for PyTorch."""
    def __init__(self, boxscores, players, odds, games):
        self.bs_idx = defaultdict(list)
        for _, r in boxscores.iterrows():
            g = r.get("game_id")
            if pd.notna(g): self.bs_idx[int(g)].append(r.to_dict())
        self.odds_idx = defaultdict(list)
        for _, r in odds.iterrows():
            g = r.get("game_id")
            if pd.notna(g): self.odds_idx[int(g)].append(r.to_dict())
        self.pl_idx = defaultdict(list)
        for _, r in players.iterrows():
            g = r.get("game_id")
            if pd.notna(g): self.pl_idx[int(g)].append(r.to_dict())
        self.q4_idx = {}
        for _, r in games.iterrows():
            self.q4_idx[r["game_id"]] = {
                "home_q4_net": r.get("home_q4_net", 0) or 0,
                "away_q4_net": r.get("away_q4_net", 0) or 0,
            }
        self.total_pts_idx = {}
        if "total_pts" in games.columns:
            for _, r in games.iterrows():
                self.total_pts_idx[r["game_id"]] = r.get("total_pts", 0) or 0
        self.log = defaultdict(list)
        self.player_history = defaultdict(list)
        self.elo = {t: ELO_INIT for t in TEAM_ABBR}
        self.rec = defaultdict(lambda: {"w": 0, "l": 0})
        self.q4_history = defaultdict(list)
        self._league_ast_rank = {}; self._league_stl_rank = {}
        self._league_pf_rank = {}; self._rank_cache_games = 0
        self._live_odds = {}; self._standings = {}; self._ref_tendencies = {}
        self._allstar_date = None
        logger.info(f"EngineV10: {len(self.bs_idx)} bs, {len(self.odds_idx)} odds, {len(self.pl_idx)} pl")

    def inject_live_odds(self, odds_dict): self._live_odds = odds_dict
    def inject_standings(self, standings): self._standings = standings
    def inject_ref_tendencies(self, ref_map): self._ref_tendencies = ref_map

    def reset_season(self, s):
        self.log = defaultdict(list)
        self.rec = defaultdict(lambda: {"w": 0, "l": 0})
        self.q4_history = defaultdict(list)
        self._league_ast_rank = {}; self._league_stl_rank = {}
        self._league_pf_rank = {}; self._rank_cache_games = 0
        self._allstar_date = pd.Timestamp(ALLSTAR_DATES.get(s, "2026-02-15"))

    # ─── V10 STRUCTURED INPUTS ───────────────────────────────────────────────
    def compute_temporal_sequence(self, tid, game_date, gap_days=TEMPORAL_GAP_DAYS):
        """Build [15, 9] temporal sequence + [15] mask for LSTM branch."""
        lg = self.log[tid]
        if not lg:
            return np.zeros((TEMPORAL_SEQ_LEN, TEMPORAL_FEATURES)), np.zeros(TEMPORAL_SEQ_LEN)
        cutoff = game_date - timedelta(days=gap_days)
        valid = [g for g in lg if (pd.Timestamp(g["date"]) if isinstance(g["date"], str)
                                    else g["date"]) <= cutoff]
        recent = valid[-TEMPORAL_SEQ_LEN:]
        seq = np.zeros((TEMPORAL_SEQ_LEN, TEMPORAL_FEATURES))
        mask = np.zeros(TEMPORAL_SEQ_LEN)
        for i, g in enumerate(recent):
            offset = TEMPORAL_SEQ_LEN - len(recent) + i
            gd_ts = pd.Timestamp(g["date"]) if isinstance(g["date"], str) else g["date"]
            pts = (g.get("pts", 0) or 0) / 130.0
            poss = (g.get("poss", 0) or 98) / 110.0
            fga = g.get("fga", 85) or 85; fgm = g.get("fgm", 37) or 37
            fg3m = g.get("fg3m", 11) or 11
            efg = (fgm + 0.5 * fg3m) / max(fga, 1)
            tov = g.get("tov", 14) or 14; p = g.get("poss", 98) or 98
            tov_pct = tov / max(p, 1) * 100 / 20 if p > 50 else 0.65
            oreb = g.get("oreb", 10) or 10; dreb = g.get("dreb", 33) or 33
            oreb_pct = oreb / max(oreb + dreb, 1)
            rest = 3
            if i > 0:
                prev_date = pd.Timestamp(recent[i-1]["date"]) if isinstance(recent[i-1]["date"], str) else recent[i-1]["date"]
                rest = (gd_ts - prev_date).days
            home_flag = 1.0 if g.get("home") else 0.0
            opp_id = g.get("opp_id")
            opp_rank = self._get_team_rank(opp_id) / 30.0 if opp_id else 0.5
            days_since = (game_date - gd_ts).days / 30.0
            seq[offset] = [pts, poss, efg, tov_pct, oreb_pct,
                          min(rest, 7)/7, home_flag, opp_rank, days_since]
            mask[offset] = 1.0
        return seq, mask

    def compute_matchup_stats(self, hid, aid):
        """[16]-dim differential stats for matchup branch."""
        hl = self.log[hid]; al = self.log[aid]
        hp = self._pr(hl, 10); ap = self._pr(al, 10)
        h10 = self._ff(hl, 10); a10 = self._ff(al, 10)
        stats = np.zeros(MATCHUP_STATS_DIM)
        stats[0] = (hp["ortg"] - ap["drtg"]) / 20
        stats[1] = (ap["ortg"] - hp["drtg"]) / 20
        stats[2] = h10["efg"] - a10["efg"]
        stats[3] = a10["tp"] - h10["tp"]
        stats[4] = h10["op"] - a10["op"]
        stats[5] = (hp["pace"] - ap["pace"]) / 10
        stats[6] = (self.elo[hid] - self.elo[aid]) / 200
        stats[7] = h10["fr"] - a10["fr"]
        h3 = self._three_point_profile(hl, al, 10)
        stats[8] = h3["h_3pr"] - h3["a_3pr"]
        stats[9] = h3["h_3pp"] - h3["a_3pp"]
        stats[10] = h3["h_opp3p"] - h3["a_opp3p"]
        tr = self._transition_pnr(hl, al, 10)
        stats[11] = tr["h_fb"] - tr["a_fb"]
        stats[12] = tr["h_at"] - tr["a_at"]
        ht8 = self._top8(hid, 10); at8 = self._top8(aid, 10)
        stats[13] = (ht8["o"] - at8["o"]) / 120
        stats[14] = (ht8["d"] - at8["d"]) / 120
        stats[15] = ht8["c"] - at8["c"]
        return np.nan_to_num(stats)

    def compute_market_features(self, gid, hid, aid):
        """[8]-dim market intelligence vector."""
        live = self._live_odds.get((hid, aid), {})
        sm = self._smart(gid, hid, aid)
        mkt = np.zeros(MARKET_FEATURES_DIM)
        open_p = live.get("open_prob", sm.get("mkt_prob_home", 0.5))
        curr_p = sm.get("mkt_prob_home", 0.5)
        raw = live.get("raw_home_probs", [curr_p])
        mkt[0] = open_p; mkt[1] = curr_p
        mkt[2] = max(raw) if raw else curr_p
        mkt[3] = min(raw) if raw else curr_p
        mkt[4] = float(live.get("steam_signal", 0))
        mkt[5] = np.clip(live.get("public_vs_sharp", 0), -0.1, 0.1) * 10
        mkt[6] = 0.5; mkt[7] = 0.5  # timestamp placeholders
        return mkt

    def compute_context_features(self, hid, aid, game_date, season_progress):
        """[6]-dim context vector."""
        hl = self.log[hid]; al = self.log[aid]
        ctx = np.zeros(CONTEXT_FEATURES_DIM)
        ctx[0] = self._road_pct_14d(hl, game_date)
        ctx[1] = self._road_pct_14d(al, game_date)
        ctx[2] = np.clip(self._rest_days(hl, game_date) - self._rest_days(al, game_date), -3, 3) / 3.0
        ctx[3] = self._allstar_decay(game_date)
        games_remaining = max(82 - (self.rec[hid]["w"] + self.rec[hid]["l"]), 1)
        h_pu = self._playoff_urgency(hid, season_progress)
        a_pu = self._playoff_urgency(aid, season_progress)
        ctx[4] = np.clip((82 - games_remaining) / 82 * max(h_pu, a_pu), 0, 1)
        ctx[5] = season_progress
        return ctx

    def _get_team_rank(self, tid):
        if tid is None: return 15
        elos = sorted(self.elo.items(), key=lambda x: x[1], reverse=True)
        for i, (t, _) in enumerate(elos):
            if t == tid: return i + 1
        return 15

    def _road_pct_14d(self, lg, gd):
        if not lg: return 0.0
        window = gd - timedelta(days=14)
        recent = [g for g in lg if (pd.Timestamp(g["date"]) if isinstance(g["date"], str) else g["date"]) >= window]
        if not recent: return 0.0
        return sum(1 for g in recent if not g.get("home", True)) / len(recent)

    def _rest_days(self, lg, gd):
        if not lg: return 3
        ld = lg[-1]["date"]
        ld = pd.Timestamp(ld) if isinstance(ld, str) else ld
        return min((gd - ld).days, 7)

    def _allstar_decay(self, gd):
        if self._allstar_date is None: return 0.0
        days = (gd - self._allstar_date).days
        if days < 0: return 0.0
        return np.exp(-days / 30.0)

    # ─── FLAT FEATURE COMPUTE (V9 compatible) ─────────────────────────────────
    def compute(self, g, is_current_season=False):
        gid=g["game_id"]; hid=g["home_team_id"]; aid=g["away_team_id"]; gd=g["game_date"]
        if hid not in TEAM_ABBR or aid not in TEAM_ABBR: return None
        hl, al = self.log[hid], self.log[aid]
        if len(hl) < 5 or len(al) < 5: return None
        f = {}
        h10=self._ff(hl,10); a10=self._ff(al,10); h5=self._ff(hl,5); a5=self._ff(al,5)
        f["h_efg"]=h10["efg"]; f["a_efg"]=a10["efg"]
        f["h_tov_pct"]=h10["tp"]; f["a_tov_pct"]=a10["tp"]
        f["h_oreb_pct"]=h10["op"]; f["a_oreb_pct"]=a10["op"]
        f["h_ft_rate"]=h10["fr"]; f["a_ft_rate"]=a10["fr"]
        f["diff_efg"]=h10["efg"]-a10["efg"]; f["diff_tov"]=a10["tp"]-h10["tp"]
        f["diff_oreb"]=h10["op"]-a10["op"]; f["diff_ft"]=h10["fr"]-a10["fr"]
        f["h_efg_hot5"]=h5["efg"]; f["a_efg_hot5"]=a5["efg"]
        f["h_efg_trend"]=h5["efg"]-h10["efg"]; f["a_efg_trend"]=a5["efg"]-a10["efg"]
        hp=self._pr(hl,10); ap=self._pr(al,10)
        for px,pr in [("h",hp),("a",ap)]:
            f[f"{px}_pace"]=pr["pace"]/100; f[f"{px}_ortg"]=pr["ortg"]/120
            f[f"{px}_drtg"]=pr["drtg"]/120; f[f"{px}_net_rtg"]=(pr["ortg"]-pr["drtg"])/30
        md=self._matchup(hid,aid,hp,ap,hl,al)
        f["matchup_ortg_vs_drtg"]=md["ovd"]; f["matchup_drtg_vs_ortg"]=md["dvo"]
        f["h_ast_rate"]=md["ha"]; f["a_ast_rate"]=md["aa"]
        f["h_stl_rate"]=md["hs"]; f["a_stl_rate"]=md["as"]
        f["matchup_ast_vs_stl"]=md["avs"]; f["matchup_dna_penalty"]=md["pen"]
        for px,lg,tid in [("h",hl,hid),("a",al,aid)]:
            ft=self._fatigue(lg,gd,tid)
            f[f"{px}_games_7d"]=ft["g7"]; f[f"{px}_b2b"]=ft["b2b"]
            f[f"{px}_travel_miles_7d"]=ft["tm"]; f[f"{px}_rest_days"]=ft["rd"]
            f[f"{px}_road_trip_len"]=ft["rt"]; f[f"{px}_3in4"]=ft["t34"]
            f[f"{px}_4in6"]=ft["f46"]; f[f"{px}_tz_crossed_7d"]=ft["tz"]
            f[f"{px}_heavy_legs"]=ft["hl"]
        for px,tid,lg in [("h",hid,hl),("a",aid,al)]:
            mi=self._missing(tid,gid,lg[-1] if lg else None)
            f[f"{px}_missing_net_rtg"]=mi["nr"]; f[f"{px}_missing_min"]=mi["ml"]/240
            f[f"{px}_missing_stars"]=mi["so"]
        for px,tid in [("h",hid),("a",aid)]:
            t8=self._top8(tid,10)
            f[f"{px}_top8_off_rtg"]=t8["o"]/120; f[f"{px}_top8_def_rtg"]=t8["d"]/120
            f[f"{px}_top8_net_rtg"]=(t8["o"]-t8["d"])/30; f[f"{px}_top8_consistency"]=t8["c"]
        hq=self.q4_history[hid]; aq=self.q4_history[aid]
        f["h_q4_net_avg"]=np.mean(hq[-10:])/10 if len(hq)>=3 else 0
        f["a_q4_net_avg"]=np.mean(aq[-10:])/10 if len(aq)>=3 else 0
        vt=self._vtrap(hid,aid,h10,a10,hl,al)
        f["h_ft_dependency"]=vt["hfd"]; f["a_ft_dependency"]=vt["afd"]
        f["h_opp_pf_rate"]=vt["hop"]; f["a_opp_pf_rate"]=vt["aop"]
        f["value_trap_flag"]=vt["t"]
        sm=self._smart(gid,hid,aid)
        f["mkt_prob_home"]=sm["mkt_prob_home"]; f["mkt_spread"]=sm["mkt_spread"]/10
        f["rlm_signal"]=sm["rlm_signal"]; f["mkt_gap"]=0.0
        f["odds_move_home"]=sm["odds_move_home"]; f["consensus_spread"]=sm["consensus_spread"]/10
        he,ae=self.elo[hid],self.elo[aid]
        f["elo_diff"]=(he-ae)/100; f["elo_exp"]=1/(1+10**(-(he-ae+ELO_HCA)/400))
        f["is_conf"]=1 if (hid in EAST)==(aid in EAST) else 0
        total=self.rec[hid]["w"]+self.rec[hid]["l"]
        f["season_progress"]=min(total/82,1.0)
        for px,tid in [("h",hid),("a",aid)]:
            f[f"{px}_playoff_urgency"]=self._playoff_urgency(tid,f["season_progress"])
        f["season_weight"]=1.0 if is_current_season else 0.7
        # V9
        tp=self._three_point_profile(hl,al,10)
        f["h_3p_rate"]=tp["h_3pr"]; f["a_3p_rate"]=tp["a_3pr"]
        f["h_3p_pct"]=tp["h_3pp"]; f["a_3p_pct"]=tp["a_3pp"]
        f["h_opp_3p_pct"]=tp["h_opp3p"]; f["a_opp_3p_pct"]=tp["a_opp3p"]
        f["matchup_3p_attack_vs_def"]=tp["h_3pr"]*(tp["h_3pp"]-tp["a_opp3p"])
        f["matchup_3p_def_vs_attack"]=tp["a_3pr"]*(tp["a_3pp"]-tp["h_opp3p"])
        tr=self._transition_pnr(hl,al,10)
        f["h_fastbreak_rate"]=tr["h_fb"]; f["a_fastbreak_rate"]=tr["a_fb"]
        f["h_ast_to_tov"]=tr["h_at"]; f["a_ast_to_tov"]=tr["a_at"]
        f["matchup_transition_diff"]=tr["h_fb"]-tr["a_fb"]
        f["altitude_factor"]=self._altitude_factor(hid,aid)
        for px,lg,tid in [("h",hl,hid),("a",al,aid)]:
            f[f"{px}_schedule_density"]=self._schedule_density(lg,gd)
        sharp=self._sharp_tracking(gid,hid,aid,sm)
        f["steam_signal"]=sharp["steam"]; f["clv_home"]=sharp["clv"]
        f["open_vs_current_prob"]=sharp["open_dev"]; f["public_vs_sharp"]=sharp["pvs"]
        tot=self._project_total(hp,ap,hid,aid,f)
        f["proj_pace"]=tot["proj_pace"]/100; f["proj_total"]=tot["proj_total"]/250
        f["mkt_total"]=sm.get("mkt_total",220.0)/250
        f["total_edge"]=(tot["proj_total"]-sm.get("mkt_total",220.0))/20
        f["ref_foul_tendency"]=self._ref_tendency(gid,hid,aid)
        # V10 new
        for px,lg in [("h",hl),("a",al)]:
            f[f"{px}_road_pct_14d"]=self._road_pct_14d(lg,gd)
        f["opponent_rest_advantage"]=np.clip(f.get("h_rest_days",2)-f.get("a_rest_days",2),-3,3)/3.0
        f["days_since_allstar_decay"]=self._allstar_decay(gd)
        games_remaining=max(82-total,1)
        f["playoff_intensity_score"]=np.clip(
            (82-games_remaining)/82*max(f["h_playoff_urgency"],f["a_playoff_urgency"]),0,1)
        live=self._live_odds.get((hid,aid),{})
        raw_probs=live.get("raw_home_probs",[])
        f["line_movement_range"]=(max(raw_probs)-min(raw_probs)) if len(raw_probs)>=2 else 0
        f["sharp_ratio_normalized"]=np.clip(sharp["pvs"],-1,1)
        f["market_volatility"]=np.std(raw_probs) if len(raw_probs)>=3 else 0
        vec = np.array([f.get(n, 0.0) for n in FEAT], dtype=np.float64)
        return np.nan_to_num(vec)

    # ─── CORE HELPERS (from V9) ──────────────────────────────────────────────
    def _three_point_profile(self, hl, al, w):
        def _t3(lg, w):
            r=lg[-w:]
            if not r: return {"rate":0.35,"pct":0.36}
            fg3a=[g.get("fg3a",0) or 0 for g in r]; fga=[g.get("fga",85) or 85 for g in r]
            fg3m=[g.get("fg3m",0) or 0 for g in r]
            rate=np.mean([a3/max(a,1) for a3,a in zip(fg3a,fga)])
            pct=sum(fg3m)/max(sum(fg3a),1)
            return {"rate":rate,"pct":pct}
        h=_t3(hl,w); a=_t3(al,w)
        h_opp3p=0.36-(self._pr(hl,w)["drtg"]-110)*0.002
        a_opp3p=0.36-(self._pr(al,w)["drtg"]-110)*0.002
        return {"h_3pr":h["rate"],"a_3pr":a["rate"],"h_3pp":h["pct"],"a_3pp":a["pct"],
                "h_opp3p":np.clip(h_opp3p,0.30,0.42),"a_opp3p":np.clip(a_opp3p,0.30,0.42)}

    def _transition_pnr(self, hl, al, w):
        def _c(lg, w):
            r=lg[-w:]
            if not r: return {"fb":0.10,"at":1.5}
            stl=np.mean([g.get("stl",7) or 7 for g in r])
            pace=np.mean([g.get("poss",98) or 98 for g in r])
            ast=np.mean([g.get("ast",24) or 24 for g in r])
            tov=np.mean([g.get("tov",14) or 14 for g in r])
            fb=(stl*1.2+(pace-95)*0.5)/100; at=ast/max(tov,1)
            return {"fb":np.clip(fb,0,0.30),"at":np.clip(at,0.5,4.0)}
        h=_c(hl,w); a=_c(al,w)
        return {"h_fb":h["fb"],"a_fb":a["fb"],"h_at":h["at"]/3,"a_at":a["at"]/3}

    def _altitude_factor(self, hid, aid):
        if hid in ALTITUDE_MAP:
            elev=ALTITUDE_MAP[hid]
            if elev>=ALTITUDE_FATIGUE_THRESHOLD:
                return min((elev-ALTITUDE_FATIGUE_THRESHOLD)/3000,1.0)
        return 0.0

    def _schedule_density(self, lg, gd):
        if not lg: return 0.0
        window=gd-timedelta(days=6)
        recent=[g for g in lg if (pd.Timestamp(g["date"]) if isinstance(g["date"],str) else g["date"])>=window]
        if not recent: return 0.0
        density=sum(1.0/(1+max((gd-(pd.Timestamp(g["date"]) if isinstance(g["date"],str) else g["date"])).days,0)*0.3) for g in recent)
        return min(density/4,1.0)

    def _sharp_tracking(self, gid, hid, aid, sm):
        live=self._live_odds.get((hid,aid),{})
        return {
            "steam":live.get("steam_signal",0),
            "clv":np.clip(live.get("clv_home",0)/5,-1,1),
            "open_dev":np.clip(sm.get("mkt_prob_home",0.5)-live.get("open_prob",sm.get("mkt_prob_home",0.5)),-0.15,0.15)/0.15,
            "pvs":np.clip(live.get("public_vs_sharp",0),-0.10,0.10)/0.10,
        }

    def _project_total(self, hp, ap, hid, aid, f):
        proj_pace=(hp["pace"]+ap["pace"])/2
        h_pts=proj_pace*hp["ortg"]/100; a_pts=proj_pace*ap["ortg"]/100
        proj_total=h_pts+a_pts
        if self._altitude_factor(hid,aid)>0: proj_total+=ALTITUDE_TOTAL_ADJUST
        ref_t=f.get("ref_foul_tendency",0)
        if ref_t>0.5: proj_total+=REF_HIGH_FT_OVER_ADJUST*ref_t
        elif ref_t<-0.5: proj_total+=REF_LOW_FT_UNDER_ADJUST*abs(ref_t)
        return {"proj_pace":proj_pace,"proj_total":proj_total}

    def _ref_tendency(self, gid, hid, aid):
        if gid in self._ref_tendencies: return self._ref_tendencies[gid]
        h_pf=np.mean([g.get("pf",20) or 20 for g in self.log[hid][-10:]]) if self.log[hid] else 20
        a_pf=np.mean([g.get("pf",20) or 20 for g in self.log[aid][-10:]]) if self.log[aid] else 20
        return np.clip(((h_pf+a_pf)/2-20)/5,-1,1)

    def _playoff_urgency(self, tid, progress):
        if progress<PLAYOFF_URGENCY_PROGRESS: return 0
        std=self._standings.get(tid)
        pct=std["pct"] if std else (self.rec[tid]["w"]/max(self.rec[tid]["w"]+self.rec[tid]["l"],1))
        if pct<=ELIMINATED_WIN_PCT: return 1.0
        elif pct<=0.40: return 0.5
        return 0

    def _ff(self, lg, w):
        r=lg[-w:]
        if not r: return {"efg":0.52,"tp":0.13,"op":0.22,"fr":0.20}
        def _m(k,d):
            v=[g.get(k) for g in r if g.get(k) is not None]
            return np.mean(v) if v else d
        fga=_m("fga",85);fgm=_m("fgm",37);fg3m=_m("fg3m",11)
        fta=_m("fta",22);oreb=_m("oreb",10);dreb=_m("dreb",33);tov=_m("tov",14)
        poss=0.96*(fga+0.44*fta-oreb+tov)
        return {"efg":(fgm+0.5*fg3m)/fga if fga>0 else 0.52,"tp":tov/poss if poss>0 else 0.13,
                "op":oreb/(oreb+dreb) if (oreb+dreb)>0 else 0.22,"fr":fta/fga if fga>0 else 0.20}

    def _pr(self, lg, w):
        r=lg[-w:]
        if not r: return {"pace":98,"ortg":110,"drtg":110}
        p=[];o=[];d=[]
        for g in r:
            ps=g.get("poss"); pt=g.get("pts"); op=g.get("opp")
            if ps and ps>50 and pt is not None and op is not None:
                p.append(ps); o.append(pt/ps*100); d.append(op/ps*100)
        return {"pace":np.mean(p) if p else 98,"ortg":np.mean(o) if o else 110,"drtg":np.mean(d) if d else 110}

    def _matchup(self, hid, aid, hp, ap, hl, al):
        ovd=(hp["ortg"]-ap["drtg"])/20; dvo=(hp["drtg"]-ap["ortg"])/20
        ha=self._tr(hl,"ast","fgm",10,0.6); aa=self._tr(al,"ast","fgm",10,0.6)
        hs=self._tr(hl,"stl",None,10,7.5)/10; as_=self._tr(al,"stl",None,10,7.5)/10
        avs=ha*as_-aa*hs; pen=0
        hr=self._league_ast_rank.get(hid,15); ar=self._league_ast_rank.get(aid,15)
        hsr=self._league_stl_rank.get(hid,15); asr=self._league_stl_rank.get(aid,15)
        if hr<=MATCHUP_TOP5_RANK and asr<=MATCHUP_TOP5_RANK: pen=1
        elif ar<=MATCHUP_TOP5_RANK and hsr<=MATCHUP_TOP5_RANK: pen=-1
        return {"ovd":ovd,"dvo":dvo,"ha":ha,"aa":aa,"hs":hs,"as":as_,"avs":avs,"pen":pen}

    def _tr(self, lg, stat, denom, w, default):
        r=lg[-w:]
        if not r: return default
        if denom: v=[g.get(stat,0)/max(g.get(denom,1),1) for g in r if g.get(stat) is not None]
        else: v=[g.get(stat,0) for g in r if g.get(stat) is not None]
        return np.mean(v) if v else default

    def _fatigue(self, lg, gd, tid):
        if not lg: return {"g7":0,"b2b":0,"tm":0,"rd":3,"rt":0,"t34":0,"f46":0,"tz":0,"hl":0}
        last=lg[-1]; ld=pd.Timestamp(last["date"]) if isinstance(last["date"],str) else last["date"]
        rd=(gd-ld).days; wa=gd-timedelta(days=7)
        r7=[g for g in lg if (pd.Timestamp(g["date"]) if isinstance(g["date"],str) else g["date"])>=wa]
        g7=len(r7); b2b=1 if rd<=1 else 0
        fn=gd-timedelta(days=3)
        gi4=sum(1 for g in lg if (pd.Timestamp(g["date"]) if isinstance(g["date"],str) else g["date"])>=fn)
        t34=1 if gi4>=2 else 0
        sd=gd-timedelta(days=5)
        gi6=sum(1 for g in lg if (pd.Timestamp(g["date"]) if isinstance(g["date"],str) else g["date"])>=sd)
        f46=1 if gi6>=3 else 0
        tm=0;tz=0;prev=tid
        for g in r7:
            cl=tid if g["home"] else g["opp_id"]
            if prev in COORDS and cl in COORDS:
                c1,c2=COORDS[prev],COORDS[cl]; tm+=_haversine(c1[0],c1[1],c2[0],c2[1])
            tz+=abs(TEAM_TZ.get(prev,-6)-TEAM_TZ.get(cl,-6)); prev=cl
        rt=0
        for g in reversed(lg):
            if not g["home"]: rt+=1
            else: break
        hl_flag=1 if (tz>=TIMEZONE_PENALTY and f46) or (g7>=4 and tm>4000) else 0
        return {"g7":min(g7/4,1),"b2b":b2b,"tm":min(tm/8000,1),"rd":min(rd,7),
                "rt":min(rt/5,1),"t34":t34,"f46":f46,"tz":min(tz/6,1),"hl":hl_flag}

    def _missing(self, tid, gid, last):
        if not last: return {"nr":0,"ml":0,"so":0}
        lp=[p for p in self.pl_idx.get(last["gid"],[]) if REV_TEAM.get(p.get("nba_tid"))==tid]
        tp=self.pl_idx.get(gid,[]); ti={p["player_id"] for p in tp if REV_TEAM.get(p.get("nba_tid"))==tid}
        ms=[{"m":p.get("minutes_decimal") or 0,"pm":p.get("plus_minus") or 0,
             "st":p.get("starter")==1} for p in lp if p["player_id"] not in ti]
        if not ms: return {"nr":0,"ml":0,"so":0}
        return {"nr":np.clip(np.mean([m["pm"] for m in ms])/48*10,-5,5)/5,
                "ml":sum(m["m"] for m in ms),"so":min(sum(1 for m in ms if m["st"] and m["m"]>20)/3,1)}

    def _top8(self, tid, w):
        tg=self.log[tid][-w:]
        if not tg: return {"o":110,"d":110,"c":0.5}
        ps=defaultdict(lambda:{"m":0,"pm":0,"g":0,"fga":0,"fta":0,"oreb":0,"tov":0,"pts":0})
        for g in tg:
            for p in self.pl_idx.get(g["gid"],[]):
                if REV_TEAM.get(p.get("nba_tid"))!=tid: continue
                pid=p["player_id"];mn=p.get("minutes_decimal") or 0
                if mn<5: continue
                s=ps[pid]; s["m"]+=mn;s["pm"]+=(p.get("plus_minus") or 0);s["g"]+=1
                s["fga"]+=(p.get("fga") or 0);s["fta"]+=(p.get("fta") or 0)
                s["oreb"]+=(p.get("oreb") or 0);s["tov"]+=(p.get("tov") or 0);s["pts"]+=(p.get("pts") or 0)
        if not ps: return {"o":110,"d":110,"c":0.5}
        av=[{"m":s["m"]/s["g"],"pm":s["pm"]/s["g"],"pts":s["pts"]/s["g"],
             "poss":(s["fga"]+0.44*s["fta"]-s["oreb"]+s["tov"])/s["g"]} for s in ps.values() if s["g"]>0]
        av.sort(key=lambda x:x["m"],reverse=True); t8=av[:8]
        if not t8: return {"o":110,"d":110,"c":0.5}
        wts=np.array([p["m"] for p in t8]); wts=wts/wts.sum() if wts.sum()>0 else np.ones(len(wts))/len(wts)
        oo=np.average([p["pts"]/max(p["poss"],1)*100 for p in t8],weights=wts)
        dd=np.average([max(110-p["pm"]/max(p["m"],1)*24,80) for p in t8],weights=wts)
        cc=1-min(np.std([p["pm"] for p in t8])/15,1) if len(t8)>1 else 0.5
        return {"o":oo,"d":dd,"c":cc}

    def _vtrap(self, hid, aid, h10, a10, hl, al):
        hfd=h10["fr"]; afd=a10["fr"]
        def _opf(lg):
            r=lg[-10:]
            return np.mean([g.get("pf",0) or 0 for g in r if g.get("pf") is not None])/25 if r else 0.8
        hop=_opf(al); aop=_opf(hl); t=0
        apr=self._league_pf_rank.get(aid,15); hpr=self._league_pf_rank.get(hid,15)
        if hfd>FT_DEP_THRESH and apr>=(30-LOW_FOUL_RANK+1): t=1
        elif afd>FT_DEP_THRESH and hpr>=(30-LOW_FOUL_RANK+1): t=-1
        return {"hfd":hfd,"afd":afd,"hop":hop,"aop":aop,"t":t}

    def _smart(self, gid, hid, aid):
        live=self._live_odds.get((hid,aid))
        if live: return live
        odds=self.odds_idx.get(gid,[])
        if not odds: return {"mkt_prob_home":0.5,"mkt_spread":0,"rlm_signal":0,
                             "odds_move_home":0,"consensus_spread":0,"mkt_total":220.0}
        hn=odds[0].get("home_team",""); ahp=[]; asp=[]; atot=[]
        for o in odds:
            mk=o.get("market",""); pr=o.get("outcome_price"); pt=o.get("outcome_point")
            nm=o.get("outcome_name",""); ih=(nm==hn or hn in nm)
            if mk=="h2h" and pr is not None:
                prob=self._o2p(pr)
                if prob: ahp.append(prob if ih else 1-prob)
            elif mk=="spreads" and pt is not None and ih: asp.append(pt)
            elif mk=="totals" and pt is not None and ("Over" in nm or "over" in nm): atot.append(pt)
        mp=np.mean(ahp) if ahp else 0.5; ms=np.mean(asp) if asp else 0
        cs=np.median(asp) if asp else 0; rlm=0; om=0; mt=np.mean(atot) if atot else 220.0
        if len(ahp)>=3:
            med=np.median(ahp); om=max(ahp)-min(ahp)
            if med>0.55 and min(ahp)<med-0.05: rlm=-1
            elif med<0.45 and max(ahp)>med+0.05: rlm=1
        return {"mkt_prob_home":mp,"mkt_spread":ms,"rlm_signal":rlm,
                "odds_move_home":om,"consensus_spread":cs,"mkt_total":mt}

    @staticmethod
    def _o2p(price):
        try: price=float(price)
        except: return None
        if price==0: return None
        if abs(price)>=100: return 100/(price+100) if price>0 else abs(price)/(abs(price)+100)
        elif price>=1.01: return 1/price
        return None

    def update(self, g):
        gid=g["game_id"];hid=g["home_team_id"];aid=g["away_team_id"]
        if hid not in TEAM_ABBR or aid not in TEAM_ABBR: return
        hs=g["home_score"];aws=g["away_score"];gd=g["game_date"];hw=hs>aws
        bs=self.bs_idx.get(gid,[]); hbs=next((b for b in bs if REV_TEAM.get(b.get("nba_tid"))==hid),None)
        abs_=next((b for b in bs if REV_TEAM.get(b.get("nba_tid"))==aid),None)
        for tid,ih,won,sc,osc,bx in [(hid,True,hw,hs,aws,hbs),(aid,False,not hw,aws,hs,abs_)]:
            poss=None
            if bx:
                fga=bx.get("fga",0) or 0;fta=bx.get("fta",0) or 0
                oreb=bx.get("oreb",0) or 0;tov=bx.get("tov",0) or 0
                poss=0.96*(fga+0.44*fta-oreb+tov)
            e={"gid":gid,"date":gd,"home":ih,"won":won,"pts":sc,"opp":osc,
               "margin":sc-osc,"opp_id":aid if ih else hid,"poss":poss}
            if bx:
                for s in ["fgm","fga","fg3m","fg3a","ftm","fta","oreb","dreb","reb","ast","stl","blk","tov","pf","pts"]:
                    v=bx.get(s); e[s]=float(v) if v is not None and not (isinstance(v,float) and np.isnan(v)) else None
            self.log[tid].append(e)
            r=self.rec[tid]
            if won: r["w"]+=1
            else: r["l"]+=1
        q4=self.q4_idx.get(gid,{})
        self.q4_history[hid].append(q4.get("home_q4_net",0))
        self.q4_history[aid].append(q4.get("away_q4_net",0))
        for p in self.pl_idx.get(gid,[]):
            pid=p["player_id"];tid=REV_TEAM.get(p.get("nba_tid"))
            if tid not in (hid,aid): continue
            self.player_history[pid].append({"gid":gid,"date":gd,"team":tid,
                "mins":p.get("minutes_decimal") or 0,"starter":p.get("starter")==1,
                "pts":p.get("pts") or 0,"fga":p.get("fga") or 0,"fg3m":p.get("fg3m") or 0,
                "fta":p.get("fta") or 0,"tov":p.get("tov") or 0,"oreb":p.get("oreb") or 0,
                "stl":p.get("stl") or 0,"plus_minus":p.get("plus_minus") or 0})
        he,ae=self.elo[hid],self.elo[aid]
        exp=1/(1+10**(-(he-ae+ELO_HCA)/400)); act=1.0 if hw else 0.0
        mov=min(np.log1p(abs(hs-aws))*0.7,2.5); ac=2.2/((abs(he-ae)*0.001)+2.2)
        k=ELO_K*mov*ac; self.elo[hid]+=k*(act-exp); self.elo[aid]+=k*((1-act)-(1-exp))
        self._rank_cache_games+=1
        if self._rank_cache_games>=30: self._refresh_ranks(); self._rank_cache_games=0

    def _refresh_ranks(self):
        at={};st={};pf={}
        for tid in TEAM_ABBR:
            lg=self.log[tid]
            if len(lg)<5: at[tid]=0;st[tid]=0;pf[tid]=999;continue
            r=lg[-15:]
            at[tid]=np.mean([g.get("ast",0) or 0 for g in r])
            st[tid]=np.mean([g.get("stl",0) or 0 for g in r])
            pf[tid]=np.mean([g.get("pf",0) or 0 for g in r])
        for d,attr in [(at,"_league_ast_rank"),(st,"_league_stl_rank")]:
            s=sorted(d,key=lambda t:d[t],reverse=True)
            setattr(self,attr,{t:i+1 for i,t in enumerate(s)})
        s=sorted(pf,key=lambda t:pf[t],reverse=True)
        self._league_pf_rank={t:i+1 for i,t in enumerate(s)}


# ═══════════════════════ MONTE CARLO V10 ═════════════════════════════════════
class MonteCarloV10:
    def __init__(self, engine, n_sims=DEFAULT_SIMS):
        self.engine=engine; self.n_sims=n_sims

    def run(self, game, feat):
        gid=game["game_id"]; hid=game["home_team_id"]; aid=game["away_team_id"]
        hp=self._players(hid,gid); ap=self._players(aid,gid)
        if len(hp)<5 or len(ap)<5: return self._fb(feat)
        pace=(self._pace(hid)+self._pace(aid))/2
        he=np.array([p["o"] for p in hp[:10]]); ae=np.array([p["o"] for p in ap[:10]])
        hm=np.array([p["m"] for p in hp[:10]]); am=np.array([p["m"] for p in ap[:10]])
        hs=np.array([p["s"] for p in hp[:10]]); as_=np.array([p["s"] for p in ap[:10]])
        if feat[FEAT.index("h_heavy_legs")]==1: he*=(1-HEAVY_LEGS_EFG*2)
        if feat[FEAT.index("a_heavy_legs")]==1: ae*=(1-HEAVY_LEGS_EFG*2)
        if feat[FEAT.index("altitude_factor")]>0: ae*=(1-feat[FEAT.index("altitude_factor")]*0.02)
        hw=hm/hm.sum() if hm.sum()>0 else np.ones(len(hm))/len(hm)
        aw=am/am.sum() if am.sum()>0 else np.ones(len(am))/len(am)
        wins=0; margins=np.empty(self.n_sims); home_sc=np.empty(self.n_sims)
        away_sc=np.empty(self.n_sims); totals=np.empty(self.n_sims)
        ref_t=feat[FEAT.index("ref_foul_tendency")]
        for i in range(self.n_sims):
            hss=np.random.normal(he,hs); ass=np.random.normal(ae,as_)
            ht=np.dot(hw,hss); at=np.dot(aw,ass)
            sp=pace+np.random.normal(0,2.5)
            hpts=sp*ht/100+np.random.normal(0,1.2); apts=sp*at/100+np.random.normal(0,1.2)
            hpts+=np.random.normal(1.5,0.5)
            if ref_t>0.3:
                fp=np.random.normal(ref_t*2,0.5); hpts+=fp*0.5; apts+=fp*0.5
            if hpts>apts: wins+=1
            margins[i]=hpts-apts; home_sc[i]=hpts; away_sc[i]=apts; totals[i]=hpts+apts
        wp=wins/self.n_sims; ms=np.std(margins); em=np.mean(margins); et=np.mean(totals)
        md={"mean":em,"std":ms,"p10":float(np.percentile(margins,10)),
            "p25":float(np.percentile(margins,25)),"p50":float(np.percentile(margins,50)),
            "p75":float(np.percentile(margins,75)),"p90":float(np.percentile(margins,90))}
        td={"mean":et,"std":float(np.std(totals)),"p10":float(np.percentile(totals,10)),
            "p25":float(np.percentile(totals,25)),"p50":float(np.percentile(totals,50)),
            "p75":float(np.percentile(totals,75)),"p90":float(np.percentile(totals,90))}
        return {"wp":wp,"em":em,"ms":ms,"conf":max(0,1-ms/18),"volatility":ms,
                "n_players":(len(hp),len(ap)),"margin_dist":md,"total_dist":td,
                "projected_total":et,"home_scores_mean":float(np.mean(home_sc)),
                "away_scores_mean":float(np.mean(away_sc)),"raw_margins":margins,"raw_totals":totals}

    def _players(self, tid, gid):
        tg=self.engine.log[tid][-5:]
        if not tg: return []
        ps=defaultdict(lambda:{"g":0,"tm":0,"tp":0,"tps":0,"ov":[]})
        for g in tg:
            gi=g["gid"]
            if gi==gid: continue
            for p in self.engine.pl_idx.get(gi,[]):
                if REV_TEAM.get(p.get("nba_tid"))!=tid: continue
                pid=p["player_id"];mn=p.get("minutes_decimal") or 0
                if mn<5: continue
                pts=p.get("pts") or 0;fga=p.get("fga") or 0
                fta=p.get("fta") or 0;tov=p.get("tov") or 0;oreb=p.get("oreb") or 0
                pu=0.96*(fga+0.44*fta-oreb+tov); ortg=(pts/pu*100) if pu>3 else 100
                s=ps[pid]; s["g"]+=1;s["tm"]+=mn;s["tp"]+=pts;s["tps"]+=pu;s["ov"].append(ortg)
        res=[]
        for pid,s in ps.items():
            if s["g"]>=2:
                res.append({"m":s["tm"]/s["g"],"o":np.mean(s["ov"]) if s["ov"] else 100,
                            "s":max(np.std(s["ov"]) if len(s["ov"])>1 else 8,3)})
        res.sort(key=lambda x:x["m"],reverse=True); return res[:10]

    def _pace(self, tid):
        lg=self.engine.log[tid][-10:]
        p=[g["poss"] for g in lg if g.get("poss") and g["poss"]>50]
        return np.mean(p) if p else 98

    def _fb(self, feat):
        try: wp=feat[FEAT.index("elo_exp")]
        except: wp=0.5
        return {"wp":wp,"em":0,"ms":12,"conf":0.5,"volatility":12,"n_players":(0,0),
                "margin_dist":{"mean":0,"std":12,"p10":-15,"p25":-8,"p50":0,"p75":8,"p90":15},
                "total_dist":{"mean":220,"std":15,"p10":200,"p25":210,"p50":220,"p75":230,"p90":240},
                "projected_total":220,"home_scores_mean":110,"away_scores_mean":110}


# ═══════════════════════ MODEL V10 ═══════════════════════════════════════════
class ModelV10:
    """V10 Hybrid: PyTorch Deep Edge + XGBoost fallback + MC ensemble."""
    def __init__(self):
        self.xgb = None; self.scaler = StandardScaler(); self.mc = None; self.engine = None
        self.trained = False; self.tX = []; self.ty = []; self.tW = []
        self._day_picks = []; self._current_day = None
        self.xgb_total = None; self.total_ty = []
        self.deep_model = None; self.deep_trained = False; self.deep_data = []
        self.uncertainty_model = None
        self.execution_engine = BayesianExecutionEngine() if TORCH_AVAILABLE else None
        logger.info(f"ModelV10 | DIAMOND>={DIAMOND_THRESHOLD:.0%} EV>={DIAMOND_EV_MIN} "
                     f"| PyTorch={'YES' if TORCH_AVAILABLE else 'NO'}")

    def connect_mc(self, mc): self.mc = mc
    def connect_engine(self, eng): self.engine = eng

    def add(self, X, y, gid, weight=1.0, total_pts=None, game=None):
        self.tX.append(X); self.ty.append(y); self.tW.append(weight)
        if total_pts is not None: self.total_ty.append(total_pts)
        # Collect structured data for deep model
        if game is not None and self.engine is not None and TORCH_AVAILABLE:
            try:
                hid=game["home_team_id"]; aid=game["away_team_id"]; gd=game["game_date"]
                if hid in TEAM_TO_IDX and aid in TEAM_TO_IDX:
                    h_seq,h_mask=self.engine.compute_temporal_sequence(hid,gd)
                    a_seq,a_mask=self.engine.compute_temporal_sequence(aid,gd)
                    matchup=self.engine.compute_matchup_stats(hid,aid)
                    market=self.engine.compute_market_features(gid,hid,aid)
                    sp=min((self.engine.rec[hid]["w"]+self.engine.rec[hid]["l"])/82,1.0)
                    context=self.engine.compute_context_features(hid,aid,gd,sp)
                    sm=self.engine._smart(gid,hid,aid)
                    live=self.engine._live_odds.get((hid,aid),{})
                    self.deep_data.append({
                        "flat":X.copy(),
                        "temporal_seq":(h_seq+a_seq)/2.0,
                        "temporal_mask":np.maximum(h_mask,a_mask),
                        "home_idx":TEAM_TO_IDX[hid],"away_idx":TEAM_TO_IDX[aid],
                        "matchup_stats":matchup,"market_features":market,
                        "context_features":context,
                        "win":float(y),
                        "margin":float(game.get("margin",0) or 0)/20.0,
                        "market_line":float(sm.get("mkt_spread",0))/10.0,
                        "clv_sign":1.0 if live.get("clv_home",0)>0 else 0.0,
                    })
            except: pass

    def retrain(self):
        if len(self.tX) < 300: return False
        # XGBoost fallback
        X=np.array(self.tX); y=np.array(self.ty); W=np.array(self.tW)
        Xs=self.scaler.fit_transform(X)
        n_pos=max(np.sum(y),1); spw=(len(y)-n_pos)/n_pos*2.5
        self.xgb = xgb.XGBClassifier(
            n_estimators=600,max_depth=5,learning_rate=0.018,subsample=0.8,
            colsample_bytree=0.7,reg_alpha=0.3,reg_lambda=2.0,min_child_weight=6,
            gamma=0.15,scale_pos_weight=spw,random_state=42,
            use_label_encoder=False,eval_metric="logloss")
        self.xgb.fit(Xs, y, sample_weight=W)
        if len(self.total_ty) >= 300:
            yt=np.array(self.total_ty[:len(self.tX)])
            if len(yt)==len(Xs):
                self.xgb_total = xgb.XGBRegressor(
                    n_estimators=400,max_depth=4,learning_rate=0.02,subsample=0.8,
                    colsample_bytree=0.6,reg_alpha=0.5,reg_lambda=2.5,random_state=42)
                self.xgb_total.fit(Xs,yt,sample_weight=W)
                logger.info(f"XGB Total regressor trained | {len(yt)} samples")
        self.trained = True
        logger.info(f"XGB V10 trained | {len(self.tX)} samples | spw={spw:.2f}")
        # PyTorch deep model
        if TORCH_AVAILABLE and len(self.deep_data) >= 500:
            self._train_deep_model()
        return True

    def _train_deep_model(self):
        logger.info(f"🧠 Training PyTorch DeepEdge | {len(self.deep_data)} samples...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n=len(self.deep_data); split=int(n*0.85)
        train_ds=NBAGameDataset(self.deep_data[:split])
        val_ds=NBAGameDataset(self.deep_data[split:])
        train_ld=TorchDataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=True,drop_last=True)
        val_ld=TorchDataLoader(val_ds,batch_size=BATCH_SIZE,shuffle=False,drop_last=False)

        model=NBADeepEdgeModel(flat_feat_dim=N_FEAT).to(device)
        opt=torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY)
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=N_EPOCHS)
        best_val=float("inf"); patience_ctr=0; best_state=None

        for epoch in range(N_EPOCHS):
            model.train(); tl=0; nb=0
            for batch in train_ld:
                batch={k:v.to(device) for k,v in batch.items()}
                opt.zero_grad()
                out=model(batch["flat"],batch["temporal_seq"],batch["temporal_mask"],
                          batch["home_idx"],batch["away_idx"],batch["matchup_stats"],
                          batch["market_features"],batch["context_features"])
                tgt={"win":batch["win"],"margin":batch["margin"],
                     "market_line":batch["market_line"],"clv_sign":batch["clv_sign"]}
                loss,_=model.compute_loss(out,tgt)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                opt.step(); tl+=loss.item(); nb+=1
            sched.step()
            model.eval(); vl=0; vn=0
            with torch.no_grad():
                for batch in val_ld:
                    batch={k:v.to(device) for k,v in batch.items()}
                    out=model(batch["flat"],batch["temporal_seq"],batch["temporal_mask"],
                              batch["home_idx"],batch["away_idx"],batch["matchup_stats"],
                              batch["market_features"],batch["context_features"])
                    tgt={"win":batch["win"],"margin":batch["margin"],
                         "market_line":batch["market_line"],"clv_sign":batch["clv_sign"]}
                    loss,_=model.compute_loss(out,tgt); vl+=loss.item(); vn+=1
            avg_val=vl/max(vn,1)
            if avg_val<best_val:
                best_val=avg_val; patience_ctr=0
                best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}
            else: patience_ctr+=1
            if epoch%5==0:
                logger.info(f"  Epoch {epoch+1}/{N_EPOCHS} | Train:{tl/max(nb,1):.4f} | Val:{avg_val:.4f} | Pat:{patience_ctr}")
            if patience_ctr>=PATIENCE:
                logger.info(f"  Early stop @ epoch {epoch+1}"); break

        if best_state: model.load_state_dict(best_state)
        self.deep_model=model.cpu(); self.deep_trained=True
        logger.info(f"🧠 DeepEdge trained | Best val: {best_val:.4f}")
        self._train_uncertainty(device)

    def _train_uncertainty(self, device):
        """Train uncertainty meta-learner on prediction residuals."""
        if not self.deep_trained or len(self.deep_data)<500: return
        logger.info("Training UncertaintyMetaLearner...")
        meta=UncertaintyMetaLearner(input_dim=12).to(device)
        meta_opt=torch.optim.Adam(meta.parameters(),lr=5e-4)
        self.deep_model.to(device).eval()
        meta_X=[]; meta_yt=[]; meta_yp=[]
        with torch.no_grad():
            loader=TorchDataLoader(NBAGameDataset(self.deep_data),batch_size=128,shuffle=False)
            idx=0
            for batch in loader:
                bd={k:v.to(device) for k,v in batch.items()}
                out=self.deep_model(bd["flat"],bd["temporal_seq"],bd["temporal_mask"],
                                    bd["home_idx"],bd["away_idx"],bd["matchup_stats"],
                                    bd["market_features"],bd["context_features"])
                preds=out["win_prob"].cpu().numpy()
                for i in range(len(preds)):
                    d=self.deep_data[idx+i]; flat=d["flat"]
                    mf=np.zeros(12)
                    mf[0]=preds[i]; mf[1]=0.1
                    mf[2]=flat[FEAT.index("h_b2b")]; mf[3]=flat[FEAT.index("a_b2b")]
                    mf[4]=d["market_features"][5] if len(d["market_features"])>5 else 0
                    mf[5]=flat[FEAT.index("h_rest_days")]/7
                    mf[6]=flat[FEAT.index("season_progress")]
                    mf[7]=flat[FEAT.index("h_playoff_urgency")]
                    mf[8]=flat[FEAT.index("h_missing_min")]
                    mf[9]=flat[FEAT.index("a_missing_min")]
                    mf[10]=flat[FEAT.index("season_progress")]
                    mf[11]=flat[FEAT.index("elo_diff")]
                    meta_X.append(mf); meta_yt.append(d["win"]); meta_yp.append(preds[i])
                idx+=len(preds)
        if len(meta_X)<100: self.deep_model.cpu(); return
        mXt=torch.tensor(np.array(meta_X),dtype=torch.float32,device=device)
        mYt=torch.tensor(meta_yt,dtype=torch.float32,device=device)
        mYp=torch.tensor(meta_yp,dtype=torch.float32,device=device)
        for _ in range(50):
            meta.train(); mu_c,sig=meta(mXt)
            loss=meta.nll_loss(mYt,mYp,mu_c,sig)
            meta_opt.zero_grad(); loss.backward(); meta_opt.step()
        self.uncertainty_model=meta.cpu(); self.deep_model.cpu()
        logger.info(f"UncertaintyMeta trained | σ_mean={sig.mean().item():.4f}")

    def _predict_deep(self, X, game):
        """Get deep model prediction + uncertainty."""
        if not self.deep_trained or not TORCH_AVAILABLE or game is None: return None,0.1
        try:
            hid=game["home_team_id"]; aid=game["away_team_id"]; gid=game["game_id"]
            gd=game["game_date"]
            if hid not in TEAM_TO_IDX or aid not in TEAM_TO_IDX: return None,0.1
            h_seq,h_mask=self.engine.compute_temporal_sequence(hid,gd)
            a_seq,a_mask=self.engine.compute_temporal_sequence(aid,gd)
            matchup=self.engine.compute_matchup_stats(hid,aid)
            market=self.engine.compute_market_features(gid,hid,aid)
            sp=min((self.engine.rec[hid]["w"]+self.engine.rec[hid]["l"])/82,1.0)
            context=self.engine.compute_context_features(hid,aid,gd,sp)
            self.deep_model.eval()
            with torch.no_grad():
                out=self.deep_model(
                    torch.tensor(X,dtype=torch.float32).unsqueeze(0),
                    torch.tensor((h_seq+a_seq)/2.0,dtype=torch.float32).unsqueeze(0),
                    torch.tensor(np.maximum(h_mask,a_mask),dtype=torch.float32).unsqueeze(0),
                    torch.tensor([TEAM_TO_IDX[hid]],dtype=torch.long),
                    torch.tensor([TEAM_TO_IDX[aid]],dtype=torch.long),
                    torch.tensor(matchup,dtype=torch.float32).unsqueeze(0),
                    torch.tensor(market,dtype=torch.float32).unsqueeze(0),
                    torch.tensor(context,dtype=torch.float32).unsqueeze(0))
                dwp=out["win_prob"].item()
            sigma=0.1
            if self.uncertainty_model is not None:
                mf=np.zeros(12); mf[0]=dwp; mf[1]=0.1
                mf[2]=X[FEAT.index("h_b2b")]; mf[3]=X[FEAT.index("a_b2b")]
                mf[5]=X[FEAT.index("h_rest_days")]/7
                mf[7]=X[FEAT.index("h_playoff_urgency")]
                mf[8]=X[FEAT.index("h_missing_min")]
                mf[9]=X[FEAT.index("a_missing_min")]
                mf[10]=X[FEAT.index("season_progress")]
                mf[11]=X[FEAT.index("elo_diff")]
                self.uncertainty_model.eval()
                with torch.no_grad():
                    mu_c,sig=self.uncertainty_model(torch.tensor(mf,dtype=torch.float32).unsqueeze(0))
                    sigma=sig.item(); dwp=np.clip(dwp+mu_c.item(),0.01,0.99)
            return dwp,sigma
        except: return None,0.1

    def predict(self, X, game=None, live_mode=False):
        gd=game["game_date"] if game is not None else None
        mc_vol=12; mc_result=None; sigma=0.1
        deep_wp,sigma=self._predict_deep(X,game)

        if not self.trained:
            if self.mc and game is not None:
                mc_result=self.mc.run(game,X)
                wp=mc_result["wp"]; conf=mc_result["conf"]; mc_vol=mc_result.get("volatility",12)
            else:
                try: wp=X[FEAT.index("elo_exp")]
                except: wp=0.5
                conf=0.5
        else:
            Xs=self.scaler.transform(X.reshape(1,-1))
            wp_x=self.xgb.predict_proba(Xs)[0][1]
            if self.mc and game is not None:
                mc_result=self.mc.run(game,X)
                wp_m=mc_result["wp"]; conf=mc_result["conf"]; mc_vol=mc_result.get("volatility",12)
                # V10: Ensemble XGB + MC + Deep
                if deep_wp is not None:
                    wp=0.40*deep_wp + 0.35*wp_x + 0.25*wp_m
                else:
                    wp=0.60*wp_x + 0.40*wp_m
            else:
                if deep_wp is not None:
                    wp=0.55*deep_wp + 0.45*wp_x
                else:
                    wp=wp_x
                conf=0.6

        mp=X[FEAT.index("mkt_prob_home")]; mg=wp-mp
        try: X[FEAT.index("mkt_gap")]=mg
        except: pass
        rlm=X[FEAT.index("rlm_signal")]; ph=wp>0.5
        pp=wp if ph else 1-wp; mpp=mp if ph else 1-mp
        fo=1/max(mpp,0.01); mo=fo*0.95; ev=pp*mo-1

        # Matchup DNA penalty
        dna=X[FEAT.index("matchup_dna_penalty")]
        if ph and dna==1: wp*=(1-MATCHUP_PENALTY)
        elif not ph and dna==-1: wp=1-((1-wp)*(1-MATCHUP_PENALTY))

        # Fatigue
        ft=False
        if ph:
            if X[FEAT.index("h_heavy_legs")]==1 or (X[FEAT.index("h_b2b")]==1 and X[FEAT.index("h_q4_net_avg")]<0):
                ft=True; wp*=0.90
        else:
            if X[FEAT.index("a_heavy_legs")]==1 or (X[FEAT.index("a_b2b")]==1 and X[FEAT.index("a_q4_net_avg")]<0):
                ft=True; wp=1-((1-wp)*0.90)

        # Value trap
        vt=X[FEAT.index("value_trap_flag")]; vta=False
        if ph and vt==1: wp*=0.95; vta=True
        elif not ph and vt==-1: wp=1-((1-wp)*0.95); vta=True

        # Playoff urgency
        pu_pen=False
        if ph:
            pu=X[FEAT.index("h_playoff_urgency")]
            if pu>=1.0: wp*=0.85; pu_pen=True
            elif pu>=0.5: wp*=0.93; pu_pen=True
        else:
            pu=X[FEAT.index("a_playoff_urgency")]
            if pu>=1.0: wp=1-((1-wp)*0.85); pu_pen=True
            elif pu>=0.5: wp=1-((1-wp)*0.93); pu_pen=True

        # CLV + Steam boosts
        clv=X[FEAT.index("clv_home")]
        if clv>0.3: wp+=CLV_BOOST*(1 if ph else -1); wp=np.clip(wp,0.01,0.99)
        steam=X[FEAT.index("steam_signal")]
        if (ph and steam==1) or (not ph and steam==-1): wp=min(wp*1.02,0.99)

        ppa=max(wp,1-wp); eva=ppa*mo-1

        # Multi-market analysis
        spread_pick=None; total_pick=None
        mkt_spread=X[FEAT.index("mkt_spread")]*10
        mkt_total=X[FEAT.index("mkt_total")]*250
        our_margin=mc_result["em"] if mc_result else 0
        our_total=mc_result.get("projected_total",220) if mc_result else 220
        spread_edge=abs(our_margin-(-mkt_spread))
        if mc_result and "margin_dist" in mc_result:
            md=mc_result["margin_dist"]
            if spread_edge>=SPREAD_VALUE_MIN:
                if our_margin>-mkt_spread:
                    spread_pick={"side":"HOME","line":mkt_spread,"our_margin":our_margin,"edge":spread_edge,"dead_number":False}
                else:
                    spread_pick={"side":"AWAY","line":mkt_spread,"our_margin":our_margin,"edge":spread_edge,"dead_number":False}
                if abs(md["p50"]-(-mkt_spread))<DEAD_NUMBER_RANGE: spread_pick["dead_number"]=True
        total_edge_val=our_total-mkt_total
        if abs(total_edge_val)>=TOTAL_VALUE_MIN:
            total_pick={"side":"OVER" if total_edge_val>0 else "UNDER",
                        "line":mkt_total,"our_total":our_total,"edge":total_edge_val}

        # VIP / DIAMOND logic
        is_d=False; reason=""
        rc=(ph and rlm==-1) or (not ph and rlm==1)
        gk=abs(mg)>=MKT_GAP_MIN; ek=eva>=DIAMOND_EV_MIN
        if ppa>=DIAMOND_THRESHOLD and ek and gk and not rc: is_d=True; reason="DIAMOND_ML"
        if not is_d and not rc:
            rlmc=(ph and rlm==1) or (not ph and rlm==-1)
            if rlmc and ppa>=0.74 and ek: is_d=True; reason="RLM_DIAMOND"
        if not is_d and ppa>=0.74 and ek and clv>0.5 and not rc: is_d=True; reason="CLV_DIAMOND"
        if not is_d and ppa>=0.72 and ek and steam!=0 and not rc:
            if (ph and steam==1) or (not ph and steam==-1): is_d=True; reason="STEAM_DIAMOND"
        # V10: Deep-model confidence diamond
        if not is_d and deep_wp is not None and not rc:
            deep_conf=max(deep_wp,1-deep_wp)
            if deep_conf>=0.76 and sigma<0.15 and ek: is_d=True; reason="DEEP_DIAMOND"

        rl="Bajo" if mc_vol<=8 else "Medio" if mc_vol<=13 else "Alto"

        # V10: Bayesian bet sizing
        bet_size=0
        if is_d and self.execution_engine:
            bet_size=self.execution_engine.compute_bet_size(ppa,sigma,mo)

        result={
            "wp":wp,"conf":conf,"is_vip":is_d,"vip_reason":reason,"mkt_gap":mg,
            "ev":eva,"rlm":rlm,"fatigue_trap":ft,"value_trap":vta,
            "playoff_urgency_penalty":pu_pen,
            "mkt_odds":mo,"pick_home":ph,"risk_level":rl,"mc_volatility":mc_vol,
            "mkt_prob_home":mp,
            "spread_pick":spread_pick,"total_pick":total_pick,
            "our_margin":our_margin,"our_total":our_total,
            "clv":float(clv),"steam":int(steam),
            "margin_dist":mc_result.get("margin_dist") if mc_result else None,
            "total_dist":mc_result.get("total_dist") if mc_result else None,
            "sigma":sigma,"bet_size":bet_size,
            "deep_wp":deep_wp,"ensemble_source":"deep+xgb+mc" if deep_wp else "xgb+mc",
        }

        if gd is not None:
            ds=str(gd)[:10]
            if self._current_day!=ds:
                self._flush_day_picks(); self._current_day=ds; self._day_picks=[]
            self._day_picks.append({"game":game,"result":result,"confidence":ppa,"ev":eva})
        return result

    def _flush_day_picks(self):
        if not self._day_picks: return
        if sum(1 for p in self._day_picks if p["result"]["is_vip"])==0:
            cands=[p for p in self._day_picks if p["confidence"]>=FALLBACK_THRESHOLD and p["ev"]>0]
            cands.sort(key=lambda x:x["ev"],reverse=True)
            for c in cands[:3]:
                c["result"]["is_vip"]=True; c["result"]["vip_reason"]="TOP3_FALLBACK"
                self._save_pick(c["game"],c["result"])

    def _save_pick(self, game, result, csv_path=None):
        if game is None: return
        csv_path=csv_path or PICKS_CSV_BT
        hid=game["home_team_id"]; aid=game["away_team_id"]
        ha=TEAM_ABBR.get(hid,"???"); aa=TEAM_ABBR.get(aid,"???")
        pick=ha if result["pick_home"] else aa
        conf=max(result["wp"],1-result["wp"])
        bet_types=["ML"]
        sp=result.get("spread_pick"); tp=result.get("total_pick")
        if sp and not sp.get("dead_number"): bet_types.append("SPREAD")
        if tp: bet_types.append("TOTAL")
        row={
            "Fecha":str(game["game_date"])[:10],"Partido":f"{aa} @ {ha}",
            "Pick_ML":pick,"Tipo":"/".join(bet_types),
            "Confianza_IA":f"{conf:.3f}","Prob_Mercado":f"{result.get('mkt_prob_home',0.5):.3f}",
            "Cuota_Mkt":f"{result['mkt_odds']:.3f}","Valor_Esperado":f"{result['ev']:.4f}",
            "Nivel_de_Riesgo":result.get("risk_level","Medio"),
            "Razon":result.get("vip_reason",""),"RLM":result.get("rlm",0),
            "Spread_Linea":f"{sp['line']:.1f}" if sp else "",
            "Spread_Pick":sp["side"] if sp else "",
            "Spread_Nuestro_Margen":f"{sp['our_margin']:.1f}" if sp else "",
            "Spread_Dead_Number":sp.get("dead_number",False) if sp else "",
            "Total_Linea":f"{tp['line']:.1f}" if tp else "",
            "Total_Pick":tp["side"] if tp else "",
            "Total_Nuestro":f"{tp['our_total']:.1f}" if tp else "",
            "Total_Edge":f"{tp['edge']:.1f}" if tp else "",
            "CLV":f"{result.get('clv',0):.3f}","Steam":result.get("steam",0),
            "Sigma":f"{result.get('sigma',0.1):.4f}",
            "Bet_Size":f"{result.get('bet_size',0):.2f}",
            "Source":result.get("ensemble_source","xgb+mc"),
            "Fatigue_Trap":result.get("fatigue_trap",False),
            "Value_Trap":result.get("value_trap",False),
            "Playoff_Urgency":result.get("playoff_urgency_penalty",False),
        }
        exists=os.path.exists(csv_path)
        with open(csv_path,"a",newline="",encoding="utf-8") as fh:
            w=csv.DictWriter(fh,fieldnames=row.keys())
            if not exists: w.writeheader()
            w.writerow(row)

    def save_vip_pick(self, game, result, csv_path=None):
        self._save_pick(game, result, csv_path)

    def save(self, path=None):
        path=path or f"{MODEL_DIR}/nba_model_v10.pkl"
        d={"xgb":self.xgb,"xgb_total":self.xgb_total,"scaler":self.scaler,
           "trained":self.trained,"n":len(self.tX)}
        if self.deep_trained and self.deep_model:
            d["deep_state"]=self.deep_model.state_dict()
            d["deep_trained"]=True
            if self.uncertainty_model:
                d["uncertainty_state"]=self.uncertainty_model.state_dict()
        with open(path,"wb") as fh: pickle.dump(d,fh)

    def load(self, path=None):
        path=path or f"{MODEL_DIR}/nba_model_v10.pkl"
        if not os.path.exists(path): return False
        with open(path,"rb") as fh: d=pickle.load(fh)
        if d.get("trained"):
            self.xgb=d["xgb"]; self.scaler=d["scaler"]; self.trained=True
            self.xgb_total=d.get("xgb_total")
            if TORCH_AVAILABLE and d.get("deep_trained"):
                try:
                    self.deep_model=NBADeepEdgeModel(flat_feat_dim=N_FEAT)
                    self.deep_model.load_state_dict(d["deep_state"])
                    self.deep_trained=True
                    if d.get("uncertainty_state"):
                        self.uncertainty_model=UncertaintyMetaLearner(input_dim=12)
                        self.uncertainty_model.load_state_dict(d["uncertainty_state"])
                    logger.info("DeepEdge model loaded from checkpoint")
                except Exception as e:
                    logger.warning(f"Could not load deep model: {e}")
            return True
        return False


# ═══════════════════════ METRICS V10 ═════════════════════════════════════════
class MetricsV10:
    """V10: Extended metrics — ECE, Brier Skill, Sharpe, Drawdown, Profit Factor."""
    def __init__(self):
        self.P=[]; self.A=[]; self.M=[]; self.G=[]; self.V=[]; self.E=[]
        self.spread_correct=0; self.spread_total=0
        self.total_correct=0; self.total_total=0
        self.bet_sizes=[]; self.bet_results=[]

    def add(self, p, a, m, g, v=False, e=0, spread_result=None, total_result=None,
            bet_size=0, sigma=0.1):
        self.P.append(p); self.A.append(a); self.M.append(m)
        self.G.append(g); self.V.append(v); self.E.append(e)
        if spread_result is not None:
            self.spread_total+=1
            if spread_result: self.spread_correct+=1
        if total_result is not None:
            self.total_total+=1
            if total_result: self.total_correct+=1
        if v and bet_size>0:
            self.bet_sizes.append(bet_size)
            won=(p>0.5)==(a==1)
            self.bet_results.append(1 if won else -1)

    def expected_calibration_error(self, n_bins=10):
        """ECE: weighted average of |accuracy - confidence| per bin."""
        if len(self.P)<20: return 1.0
        p=np.array(self.P); a=np.array(self.A)
        bins=np.linspace(0,1,n_bins+1); ece=0
        for i in range(n_bins):
            mask=(p>=bins[i])&(p<bins[i+1])
            if mask.sum()==0: continue
            acc=a[mask].mean(); conf=p[mask].mean()
            ece+=mask.sum()/len(p)*abs(acc-conf)
        return ece

    def brier_skill_score(self):
        """BSS vs market baseline (0.5)."""
        if len(self.P)<20: return 0
        brier=brier_score_loss(self.A,self.P)
        brier_ref=0.25  # market baseline
        return 1-brier/brier_ref

    def sharpe_ratio(self):
        """Sharpe from fixed-size bet returns."""
        if len(self.bet_results)<10: return 0
        returns=np.array(self.bet_results,dtype=float)
        return returns.mean()/max(returns.std(),0.001)*np.sqrt(len(returns))

    def max_drawdown(self):
        """Maximum drawdown from cumulative P&L."""
        if len(self.bet_results)<5: return 0
        cum=np.cumsum(self.bet_results)
        peak=np.maximum.accumulate(cum)
        dd=peak-cum
        return float(dd.max()) if len(dd)>0 else 0

    def profit_factor(self):
        """Gross profit / gross loss."""
        if not self.bet_results: return 0
        wins=sum(1 for r in self.bet_results if r>0)
        losses=sum(1 for r in self.bet_results if r<0)
        return wins/max(losses,1)

    def report(self, last_n=None, label=""):
        if not self.P: return {}
        n=last_n or len(self.P); p=self.P[-n:]; a=self.A[-n:]; v=self.V[-n:]; e=self.E[-n:]
        pr=[1 if x>0.5 else 0 for x in p]; acc=accuracy_score(a,pr); br=brier_score_loss(a,p)
        vi=[i for i,x in enumerate(v) if x]; nv=len(vi)
        if nv>0:
            vp=[p[i] for i in vi]; va=[a[i] for i in vi]; vr=[1 if x>0.5 else 0 for x in vp]
            vacc=accuracy_score(va,vr); vbr=brier_score_loss(va,vp)
            vev=np.mean([e[i] for i in vi]); vc=sum(1 for j in range(nv) if vr[j]==va[j])
        else:
            vacc=0; vbr=1; vev=0; vc=0
        print(f"\n{'='*70}")
        print(f"  {label} ({n} juegos)")
        print(f"{'='*70}")
        print(f"  ML GLOBAL: Acc {acc:.1%} | Brier {br:.4f}")
        print(f"  DIAMOND:   {nv}/{n} ({nv/n*100:.1f}%)")
        if nv>0:
            fp=nv-vc; em="ELITE" if vacc>=0.85 else "DIAMOND" if vacc>=0.78 else "WARN"
            print(f"    ML Acc:    {vacc:.1%}  [{em}]")
            print(f"    EV:        {vev:+.3f}")
            print(f"    FP:        {fp}/{nv} ({fp/nv*100:.1f}%)")
        if self.spread_total>0:
            sa=self.spread_correct/self.spread_total
            print(f"  SPREAD:    {self.spread_correct}/{self.spread_total} ({sa:.1%})")
        if self.total_total>0:
            ta=self.total_correct/self.total_total
            print(f"  TOTAL:     {self.total_correct}/{self.total_total} ({ta:.1%})")
        # V10 extended metrics
        ece=self.expected_calibration_error()
        bss=self.brier_skill_score()
        sharpe=self.sharpe_ratio()
        mdd=self.max_drawdown()
        pf=self.profit_factor()
        print(f"  ── V10 EXTENDED ──")
        print(f"  ECE:       {ece:.4f} {'✓' if ece<0.03 else '⚠'}")
        print(f"  Brier SS:  {bss:+.4f}")
        print(f"  Sharpe:    {sharpe:.2f}")
        print(f"  Max DD:    {mdd:.1f}")
        print(f"  Profit F:  {pf:.2f}")
        print(f"{'='*70}")
        return {"acc":acc,"brier":br,"vip_acc":vacc,"vip_ev":vev,
                "ece":ece,"bss":bss,"sharpe":sharpe,"mdd":mdd,"pf":pf}


# ═══════════════════════ RADAR JSON EXPORTER ═════════════════════════════════
class RadarExporter:
    @staticmethod
    def export_game(game, result, engine, output_dir=JSON_RADAR_DIR):
        hid=game["home_team_id"]; aid=game["away_team_id"]
        ha=TEAM_ABBR.get(hid,"???"); aa=TEAM_ABBR.get(aid,"???")
        gd=str(game["game_date"])[:10]; hl=engine.log[hid]; al=engine.log[aid]
        def _radar(lg,w=10):
            r=lg[-w:]
            if not r: return {"dreb":0.5,"tov":0.5,"pace":0.5,"three_pct":0.5,"oreb":0.5,"ft_rate":0.5}
            dreb=np.mean([g.get("dreb",33) or 33 for g in r])/45
            tov=1-np.mean([g.get("tov",14) or 14 for g in r])/20
            poss=np.mean([g.get("poss",98) or 98 for g in r])
            pace=np.clip((poss-90)/20,0,1)
            fg3a=np.mean([g.get("fg3a",30) or 30 for g in r])
            fg3m=np.mean([g.get("fg3m",11) or 11 for g in r])
            three_pct=fg3m/max(fg3a,1)
            oreb=np.mean([g.get("oreb",10) or 10 for g in r])/15
            fta=np.mean([g.get("fta",22) or 22 for g in r])
            fga=np.mean([g.get("fga",85) or 85 for g in r])
            ft_rate=fta/max(fga,1)
            return {"dreb":round(float(np.clip(dreb,0,1)),3),"tov":round(float(np.clip(tov,0,1)),3),
                    "pace":round(float(pace),3),"three_pct":round(float(np.clip(three_pct/0.45,0,1)),3),
                    "oreb":round(float(np.clip(oreb,0,1)),3),"ft_rate":round(float(np.clip(ft_rate/0.35,0,1)),3)}
        def _clean(obj):
            if isinstance(obj,dict): return {k:_clean(v) for k,v in obj.items()}
            elif isinstance(obj,(np.floating,np.integer)): return float(obj)
            elif isinstance(obj,np.ndarray): return obj.tolist()
            elif isinstance(obj,np.bool_): return bool(obj)
            return obj
        radar=_clean({
            "game_id":str(game["game_id"]),"date":gd,"matchup":f"{aa} @ {ha}",
            "home":ha,"away":aa,"home_radar":_radar(hl),"away_radar":_radar(al),
            "pick":{"ml":ha if result["pick_home"] else aa,
                    "confidence":float(max(result["wp"],1-result["wp"])),
                    "ev":float(result["ev"]),"risk":result["risk_level"],
                    "vip_reason":result.get("vip_reason",""),
                    "sigma":float(result.get("sigma",0.1)),
                    "bet_size":float(result.get("bet_size",0)),
                    "source":result.get("ensemble_source","")},
            "market":{"mkt_prob_home":float(result.get("mkt_prob_home",0.5)),
                      "rlm":int(result.get("rlm",0)),"steam":int(result.get("steam",0)),
                      "clv":float(result.get("clv",0))},
            "multi_market":{"spread":result.get("spread_pick"),"total":result.get("total_pick")},
            "distributions":{"margin":result.get("margin_dist"),"total":result.get("total_dist")},
        })
        fname=f"{gd}_{aa}_at_{ha}.json"
        fpath=os.path.join(output_dir,fname)
        with open(fpath,"w",encoding="utf-8") as fh: json.dump(radar,fh,indent=2,ensure_ascii=False)
        return fpath


# ═══════════════════════ TRAIN MODE V10 ══════════════════════════════════════
def run_train(train_s, eval_s, ckpt, db, n_sims):
    t0=time.time(); all_s=train_s+[eval_s]
    print(f"\n{'='*70}")
    print(f"  NBA SYNDICATE V10 — DEEP EDGE ARCHITECTURE — TRAINING")
    print(f"  Train: {', '.join(train_s)}")
    print(f"  Eval:  {eval_s}")
    print(f"  Features: {N_FEAT} | Sims: {n_sims:,} | PyTorch: {TORCH_AVAILABLE}")
    print(f"  Markets: ML + Spread + Totals")
    print(f"  Architecture: BiLSTM + TeamEmbed + CrossAttn + MultiTask + Uncertainty")
    print(f"{'='*70}\n")
    if os.path.exists(PICKS_CSV_BT): os.remove(PICKS_CSV_BT)

    dl=DataLoaderDB(db); games=dl.load_games(all_s)
    bs=dl.load_boxscores(); pl=dl.load_players(); od=dl.load_odds(); dl.close()
    logger.info(f"Games:{len(games)} BS:{len(bs)} PL:{len(pl)} Odds:{len(od)}")

    eng=EngineV10(bs,pl,od,games); mc=MonteCarloV10(eng,n_sims)
    mdl=ModelV10(); mdl.connect_mc(mc); mdl.connect_engine(eng)
    trm=MetricsV10(); evm=MetricsV10()
    proc=0; skip=0; cur=None
    current_seasons={"2024-2025","2025-2026"}
    radar_exp=RadarExporter()

    for _,g in games.iterrows():
        s=g["season"]; ie=(s==eval_s)
        if s!=cur:
            if cur is not None and not ie:
                trm.report(label=f"FIN {cur}"); mdl._flush_day_picks()
                for t in eng.elo: eng.elo[t]=0.75*eng.elo[t]+0.25*ELO_INIT
            eng.reset_season(s); cur=s; logger.info(f"Season: {s} {'[EVAL]' if ie else '[TRAIN]'}")

        is_curr=s in current_seasons
        feat=eng.compute(g,is_current_season=is_curr)
        if feat is not None:
            pred=mdl.predict(feat,game=g)
            aw=g["home_win"]; tk=evm if ie else trm
            actual_margin=g["margin"]
            actual_total=g.get("total_pts",0) or (g["home_score"]+g["away_score"])
            sp_result=None; sp=pred.get("spread_pick")
            if sp:
                mkt_spr=sp["line"]
                sp_result=(actual_margin>-mkt_spr) if sp["side"]=="HOME" else (actual_margin<-mkt_spr)
            tp_result=None; tp=pred.get("total_pick")
            if tp:
                tp_result=(actual_total>tp["line"]) if tp["side"]=="OVER" else (actual_total<tp["line"])
            tk.add(pred["wp"],aw,g["margin"],g["game_id"],pred["is_vip"],pred["ev"],
                   spread_result=sp_result,total_result=tp_result,
                   bet_size=pred.get("bet_size",0),sigma=pred.get("sigma",0.1))
            if not ie:
                weight=1.4 if is_curr else 0.7
                mdl.add(feat,aw,g["game_id"],weight=weight,total_pts=actual_total,game=g)
            if pred["is_vip"]:
                mdl.save_vip_pick(g,pred)
                if ie: radar_exp.export_game(g,pred,eng)
            proc+=1
        else: skip+=1
        eng.update(g)
        if not ie and proc>0 and proc%300==0:
            if mdl.retrain(): logger.info(f"Retrained ({len(mdl.tX)} samples)")
        if proc>0 and proc%ckpt==0:
            tk=evm if ie else trm
            tk.report(last_n=ckpt,label=f"{'EVAL' if ie else 'TRAIN'} #{proc}")

    mdl._flush_day_picks()
    if mdl.retrain(): logger.info(f"Final train ({len(mdl.tX)} samples)")
    mdl.save()
    print(f"\n  Procesados: {proc} | Omitidos: {skip} | Tiempo: {(time.time()-t0)/60:.1f} min")
    print(f"\n  --- TRAIN ---"); trm.report(label="TRAIN FINAL")
    print(f"\n  --- EVAL ({eval_s}) ---"); evm.report(label="EVAL FINAL")
    if mdl.trained and mdl.xgb is not None:
        print(f"\n  Top 25 Features (XGB):")
        fi=sorted(zip(FEAT,mdl.xgb.feature_importances_),key=lambda x:x[1],reverse=True)
        for nm,im in fi[:25]: print(f"    {nm:32s} {im:.4f} {'X'*int(im*100)}")
    if mdl.deep_trained:
        print(f"\n  ✓ DeepEdge model trained ({len(mdl.deep_data)} samples)")
        print(f"  ✓ UncertaintyMetaLearner: {'trained' if mdl.uncertainty_model else 'not trained'}")
    print(f"\n  Radar JSONs exported to: {JSON_RADAR_DIR}/")
    print(f"{'='*70}\n")


# ═══════════════════════ LIVE MODE V10 ═══════════════════════════════════════
def run_live(db, n_sims):
    today=date.today().isoformat(); csv_path=PICKS_CSV_LIVE
    print(f"\n{'='*70}")
    print(f"  NBA SYNDICATE V10 — DEEP EDGE — LIVE")
    print(f"  Date: {today}")
    print(f"  Output: {csv_path}")
    print(f"  Markets: ML + Spread + Totals + Bayesian Sizing")
    print(f"{'='*70}\n")

    mdl=ModelV10()
    if not mdl.load():
        print("No trained model found. Run --mode train first."); return
    print("Loading historical DB for engine state...")
    dl=DataLoaderDB(db); all_seasons=TRAIN_SEASONS+[CURRENT_SEASON]
    games=dl.load_games(all_seasons)
    bs=dl.load_boxscores(); pl=dl.load_players(); od=dl.load_odds(); dl.close()
    eng=EngineV10(bs,pl,od,games); mc=MonteCarloV10(eng,n_sims)
    mdl.connect_mc(mc); mdl.connect_engine(eng)

    print("Replaying historical games to build state...")
    cur=None
    for _,g in games.iterrows():
        s=g["season"]
        if s!=cur:
            if cur is not None:
                for t in eng.elo: eng.elo[t]=0.75*eng.elo[t]+0.25*ELO_INIT
            eng.reset_season(s); cur=s
        eng.update(g)
    print(f"   State built from {len(games)} games")

    sports=SportsAPIClient(); odds_client=OddsAPIClient()
    standings=sports.get_standings(); eng.inject_standings(standings)
    todays_games=sports.get_todays_games(today)
    raw_odds=odds_client.get_live_odds()
    opening_odds=odds_client.get_opening_odds()
    live_odds=odds_client.parse_game_odds(raw_odds,opening_odds) if raw_odds else {}
    eng.inject_live_odds(live_odds)

    if os.path.exists(csv_path): os.remove(csv_path)
    picks=[]; radar_exp=RadarExporter()
    game_entries=[]
    for (hid,aid),od_data in live_odds.items():
        game_entries.append({
            "game_id":hash((hid,aid,today))%10**8,
            "home_team_id":hid,"away_team_id":aid,
            "game_date":pd.Timestamp(today),
            "home_score":None,"away_score":None,"home_win":None,"margin":None,
        })
    if not game_entries:
        print("No matchups to analyse. Check API keys and game schedule."); return

    print(f"\nAnalysing {len(game_entries)} games for {today}...\n")
    for ge in game_entries:
        hid=ge["home_team_id"]; aid=ge["away_team_id"]
        ha=TEAM_ABBR.get(hid,"???"); aa=TEAM_ABBR.get(aid,"???")
        g_series=pd.Series(ge)
        feat=eng.compute(g_series,is_current_season=True)
        if feat is None:
            print(f"  SKIP  {aa} @ {ha}: insufficient data"); continue
        pred=mdl.predict(feat,game=g_series,live_mode=True)
        pick=ha if pred["pick_home"] else aa
        conf=max(pred["wp"],1-pred["wp"])
        status="DIAMOND" if pred["is_vip"] else "  ---  "
        sp=pred.get("spread_pick"); tp=pred.get("total_pick")
        sp_str=f"Spread:{sp['side']} {sp['line']:+.1f}" if sp else ""
        tp_str=f"Total:{tp['side']} {tp['line']:.1f}" if tp else ""
        src=pred.get("ensemble_source","")
        bet=pred.get("bet_size",0)
        print(f"  [{status}] {aa:3s} @ {ha:3s}  ML:{pick:3s} {conf:.1%}  "
              f"EV:{pred['ev']:+.3f}  σ:{pred.get('sigma',0):.3f}  "
              f"Bet:${bet:.0f}  {sp_str}  {tp_str}  [{src}]")
        if pred["is_vip"]:
            mdl.save_vip_pick(g_series,pred,csv_path=csv_path)
            radar_exp.export_game(g_series,pred,eng)
            picks.append(pred)

    if not picks:
        print("\nNo DIAMOND picks today. Checking TOP-3 fallback...")
        mdl._flush_day_picks()

    print(f"\n{'='*70}")
    if os.path.exists(csv_path):
        df=pd.read_csv(csv_path)
        print(f"  {len(df)} DIAMOND picks saved to {csv_path}")
        if len(df)>0:
            print(f"     EV promedio: {df['Valor_Esperado'].astype(float).mean():.4f}")
            if "Bet_Size" in df.columns:
                print(f"     Total wagered: ${df['Bet_Size'].astype(float).sum():.0f}")
    else:
        print("  No picks generated for today.")
    print(f"  Radar JSONs in: {JSON_RADAR_DIR}/")
    print(f"{'='*70}\n")


# ═══════════════════════ MAIN ════════════════════════════════════════════════
def main():
    p=argparse.ArgumentParser(description="NBA Syndicate V10 — Deep Edge Architecture")
    p.add_argument("--mode",choices=["train","live"],default="train",
                   help="train = backtest on DB | live = today's picks from APIs")
    p.add_argument("--seasons",nargs="+",default=TRAIN_SEASONS)
    p.add_argument("--eval",default=CURRENT_SEASON)
    p.add_argument("--checkpoint",type=int,default=CHECKPOINT)
    p.add_argument("--db",default=DB_PATH)
    p.add_argument("--sims",type=int,default=DEFAULT_SIMS)
    p.add_argument("--eval-only",action="store_true")
    a=p.parse_args()

    if a.eval_only:
        m=ModelV10()
        if m.load():
            print(f"Model V10 loaded (XGB: {len(m.xgb.feature_importances_)} features, "
                  f"Deep: {'YES' if m.deep_trained else 'NO'})")
        else: print("No saved model")
        return

    if a.mode=="train":
        run_train(a.seasons,a.eval,a.checkpoint,a.db,a.sims)
    else:
        run_live(a.db,a.sims)


if __name__=="__main__":
    main()
