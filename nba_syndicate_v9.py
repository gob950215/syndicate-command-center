#!/usr/bin/env python3
"""
NBA SYNDICATE V9 — THE PROFESSIONAL EDGE
==========================================
Evolución de V8 (76.7% Acc) → objetivo 85%+ efectividad.

V9 Innovations:
  1. MULTI-MARKET: Moneyline + Spread + Totals (Over/Under)
  2. SHARP TRACKING: Steam Moves, CLV, Tickets vs Money
  3. DEEP TACTICAL: 3P profiles, PnR, Transition, Referee, Altitude
  4. DISTRIBUTIONS: Full margin/total from Monte Carlo, Dead Numbers
  5. DASHBOARD JSON: Radar charts per game

Requisitos: pip install numpy pandas scikit-learn xgboost requests
"""

import os, sys, time, sqlite3, logging, argparse, warnings, pickle, csv, json
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

warnings.filterwarnings("ignore")

# ═══════════════════════ CONFIG ══════════════════════════════════════════════
DB_PATH        = "data/nba_historical.db"
MODEL_DIR      = "models"
PICKS_CSV_LIVE = "PICKS_V9_PROFESSIONAL.csv"
PICKS_CSV_BT   = "picks_profesionales_v9.csv"
JSON_RADAR_DIR = "data/radar_v9"
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
        logging.FileHandler("data/training_log_v9.txt", "a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("V9-Professional")

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
    """V9: Fetches h2h + spreads + totals, tracks CLV, detects steam moves."""

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
        # Build opener lookup
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
            consensus=np.median(all_sp) if all_sp else 0
            mkt_total=np.mean(all_tot) if all_tot else 220.0

            # RLM
            rlm=0; odds_move=0
            if len(all_hp)>=3:
                med=np.median(all_hp); mx=max(all_hp); mn=min(all_hp)
                odds_move=mx-mn
                if med>0.55 and mn<med-0.05: rlm=-1
                elif med<0.45 and mx>med+0.05: rlm=1

            # Steam Move Detection
            steam=0; opener=opener_lookup.get((hid,aid),{})
            open_prob=opener.get("open_prob")
            if open_prob is not None and len(per_bk_p)>=3:
                shifts=[per_bk_p[b]-open_prob for b in per_bk_p]
                if sum(1 for s in shifts if s>STEAM_MOVE_THRESH)>=3: steam=1
                elif sum(1 for s in shifts if s<-STEAM_MOVE_THRESH)>=3: steam=-1

            # CLV
            clv_home=0.0; open_spread=opener.get("open_spread")
            if open_spread is not None and all_sp:
                clv_home=open_spread-mkt_spread

            # Tickets vs Money (sharp vs public book divergence)
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
        logger.info(f"📊 V9 Parsed odds for {len(result)} games (incl. totals/steam/CLV)")
        return result

    @staticmethod
    def _american_to_prob(price):
        try: price = float(price)
        except: return None
        if price == 0: return None
        return 100/(price+100) if price > 0 else abs(price)/(abs(price)+100)


# ═══════════════════════ DATA LOADER ═════════════════════════════════════════
class DataLoader:
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


# ═══════════════════════ V9 FEATURE VECTOR ═══════════════════════════════════
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
    # ─── V9 NEW ───
    # 3-Point Profile Matchup
    "h_3p_rate","a_3p_rate","h_3p_pct","a_3p_pct",
    "h_opp_3p_pct","a_opp_3p_pct",
    "matchup_3p_attack_vs_def","matchup_3p_def_vs_attack",
    # Transition & PnR
    "h_fastbreak_rate","a_fastbreak_rate","h_ast_to_tov","a_ast_to_tov",
    "matchup_transition_diff",
    # Altitude & Geography
    "altitude_factor","h_schedule_density","a_schedule_density",
    # Sharp Tracking
    "steam_signal","clv_home","open_vs_current_prob","public_vs_sharp",
    # Totals Inputs
    "proj_pace","proj_total","mkt_total","total_edge",
    # Referee Effect
    "ref_foul_tendency",
]
N_FEAT = len(FEAT)


# ═══════════════════════ ENGINE V9 ═══════════════════════════════════════════
class EngineV9:
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
        logger.info(f"EngineV9: {len(self.bs_idx)} bs, {len(self.odds_idx)} odds, {len(self.pl_idx)} pl")

    def inject_live_odds(self, odds_dict): self._live_odds = odds_dict
    def inject_standings(self, standings): self._standings = standings
    def inject_ref_tendencies(self, ref_map): self._ref_tendencies = ref_map

    def reset_season(self, s):
        self.log = defaultdict(list)
        self.rec = defaultdict(lambda: {"w": 0, "l": 0})
        self.q4_history = defaultdict(list)
        self._league_ast_rank = {}; self._league_stl_rank = {}
        self._league_pf_rank = {}; self._rank_cache_games = 0

    def compute(self, g, is_current_season=False):
        gid=g["game_id"]; hid=g["home_team_id"]; aid=g["away_team_id"]; gd=g["game_date"]
        if hid not in TEAM_ABBR or aid not in TEAM_ABBR: return None
        hl, al = self.log[hid], self.log[aid]
        if len(hl) < 5 or len(al) < 5: return None
        f = {}

        # 1. FOUR FACTORS
        h10=self._ff(hl,10); a10=self._ff(al,10); h5=self._ff(hl,5); a5=self._ff(al,5)
        f["h_efg"]=h10["efg"]; f["a_efg"]=a10["efg"]
        f["h_tov_pct"]=h10["tp"]; f["a_tov_pct"]=a10["tp"]
        f["h_oreb_pct"]=h10["op"]; f["a_oreb_pct"]=a10["op"]
        f["h_ft_rate"]=h10["fr"]; f["a_ft_rate"]=a10["fr"]
        f["diff_efg"]=h10["efg"]-a10["efg"]; f["diff_tov"]=a10["tp"]-h10["tp"]
        f["diff_oreb"]=h10["op"]-a10["op"]; f["diff_ft"]=h10["fr"]-a10["fr"]
        f["h_efg_hot5"]=h5["efg"]; f["a_efg_hot5"]=a5["efg"]
        f["h_efg_trend"]=h5["efg"]-h10["efg"]; f["a_efg_trend"]=a5["efg"]-a10["efg"]

        # 2. PACE & RATINGS
        hp=self._pr(hl,10); ap=self._pr(al,10)
        for px,pr in [("h",hp),("a",ap)]:
            f[f"{px}_pace"]=pr["pace"]/100; f[f"{px}_ortg"]=pr["ortg"]/120
            f[f"{px}_drtg"]=pr["drtg"]/120; f[f"{px}_net_rtg"]=(pr["ortg"]-pr["drtg"])/30

        # 3. MATCHUP DNA
        md=self._matchup(hid,aid,hp,ap,hl,al)
        f["matchup_ortg_vs_drtg"]=md["ovd"]; f["matchup_drtg_vs_ortg"]=md["dvo"]
        f["h_ast_rate"]=md["ha"]; f["a_ast_rate"]=md["aa"]
        f["h_stl_rate"]=md["hs"]; f["a_stl_rate"]=md["as"]
        f["matchup_ast_vs_stl"]=md["avs"]; f["matchup_dna_penalty"]=md["pen"]

        # 4. FATIGUE
        for px,lg,tid in [("h",hl,hid),("a",al,aid)]:
            ft=self._fatigue(lg,gd,tid)
            f[f"{px}_games_7d"]=ft["g7"]; f[f"{px}_b2b"]=ft["b2b"]
            f[f"{px}_travel_miles_7d"]=ft["tm"]; f[f"{px}_rest_days"]=ft["rd"]
            f[f"{px}_road_trip_len"]=ft["rt"]; f[f"{px}_3in4"]=ft["t34"]
            f[f"{px}_4in6"]=ft["f46"]; f[f"{px}_tz_crossed_7d"]=ft["tz"]
            f[f"{px}_heavy_legs"]=ft["hl"]

        # 5. AUSENCIAS
        for px,tid,lg in [("h",hid,hl),("a",aid,al)]:
            mi=self._missing(tid,gid,lg[-1] if lg else None)
            f[f"{px}_missing_net_rtg"]=mi["nr"]; f[f"{px}_missing_min"]=mi["ml"]/240
            f[f"{px}_missing_stars"]=mi["so"]

        # 6. TOP-8
        for px,tid in [("h",hid),("a",aid)]:
            t8=self._top8(tid,10)
            f[f"{px}_top8_off_rtg"]=t8["o"]/120; f[f"{px}_top8_def_rtg"]=t8["d"]/120
            f[f"{px}_top8_net_rtg"]=(t8["o"]-t8["d"])/30; f[f"{px}_top8_consistency"]=t8["c"]

        # 7. CLUTCH Q4
        hq=self.q4_history[hid]; aq=self.q4_history[aid]
        f["h_q4_net_avg"]=np.mean(hq[-10:])/10 if len(hq)>=3 else 0
        f["a_q4_net_avg"]=np.mean(aq[-10:])/10 if len(aq)>=3 else 0

        # 8. VALUE TRAP
        vt=self._vtrap(hid,aid,h10,a10,hl,al)
        f["h_ft_dependency"]=vt["hfd"]; f["a_ft_dependency"]=vt["afd"]
        f["h_opp_pf_rate"]=vt["hop"]; f["a_opp_pf_rate"]=vt["aop"]
        f["value_trap_flag"]=vt["t"]

        # 9. SMART MONEY
        sm=self._smart(gid,hid,aid)
        f["mkt_prob_home"]=sm["mkt_prob_home"]; f["mkt_spread"]=sm["mkt_spread"]/10
        f["rlm_signal"]=sm["rlm_signal"]; f["mkt_gap"]=0.0
        f["odds_move_home"]=sm["odds_move_home"]; f["consensus_spread"]=sm["consensus_spread"]/10

        # 10. ELO
        he,ae=self.elo[hid],self.elo[aid]
        f["elo_diff"]=(he-ae)/100; f["elo_exp"]=1/(1+10**(-(he-ae+ELO_HCA)/400))

        # 11. CONTEXT
        f["is_conf"]=1 if (hid in EAST)==(aid in EAST) else 0
        total=self.rec[hid]["w"]+self.rec[hid]["l"]
        f["season_progress"]=min(total/82,1.0)
        for px,tid in [("h",hid),("a",aid)]:
            f[f"{px}_playoff_urgency"]=self._playoff_urgency(tid,f["season_progress"])
        f["season_weight"]=1.0 if is_current_season else 0.7

        # ═══ V9 NEW FEATURES ═══

        # 12. 3-Point Profile
        tp=self._three_point_profile(hl,al,10)
        f["h_3p_rate"]=tp["h_3pr"]; f["a_3p_rate"]=tp["a_3pr"]
        f["h_3p_pct"]=tp["h_3pp"]; f["a_3p_pct"]=tp["a_3pp"]
        f["h_opp_3p_pct"]=tp["h_opp3p"]; f["a_opp_3p_pct"]=tp["a_opp3p"]
        f["matchup_3p_attack_vs_def"]=tp["h_3pr"]*(tp["h_3pp"]-tp["a_opp3p"])
        f["matchup_3p_def_vs_attack"]=tp["a_3pr"]*(tp["a_3pp"]-tp["h_opp3p"])

        # 13. Transition & PnR
        tr=self._transition_pnr(hl,al,10)
        f["h_fastbreak_rate"]=tr["h_fb"]; f["a_fastbreak_rate"]=tr["a_fb"]
        f["h_ast_to_tov"]=tr["h_at"]; f["a_ast_to_tov"]=tr["a_at"]
        f["matchup_transition_diff"]=tr["h_fb"]-tr["a_fb"]

        # 14. Altitude & Schedule
        f["altitude_factor"]=self._altitude_factor(hid,aid)
        for px,lg,tid in [("h",hl,hid),("a",al,aid)]:
            f[f"{px}_schedule_density"]=self._schedule_density(lg,gd)

        # 15. Sharp Tracking
        sharp=self._sharp_tracking(gid,hid,aid,sm)
        f["steam_signal"]=sharp["steam"]; f["clv_home"]=sharp["clv"]
        f["open_vs_current_prob"]=sharp["open_dev"]; f["public_vs_sharp"]=sharp["pvs"]

        # 16. Totals Projection
        tot=self._project_total(hp,ap,hid,aid,f)
        f["proj_pace"]=tot["proj_pace"]/100; f["proj_total"]=tot["proj_total"]/250
        f["mkt_total"]=sm.get("mkt_total",220.0)/250
        f["total_edge"]=(tot["proj_total"]-sm.get("mkt_total",220.0))/20

        # 17. Referee Effect
        f["ref_foul_tendency"]=self._ref_tendency(gid,hid,aid)

        vec = np.array([f.get(n, 0.0) for n in FEAT], dtype=np.float64)
        return np.nan_to_num(vec)

    # ═══ V9 NEW HELPERS ═══
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

    # ═══ V8 CORE HELPERS (unchanged) ═══
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
        hl=1 if (tz>=TIMEZONE_PENALTY and f46) or (g7>=4 and tm>4000) else 0
        return {"g7":min(g7/4,1),"b2b":b2b,"tm":min(tm/8000,1),"rd":min(rd,7),
                "rt":min(rt/5,1),"t34":t34,"f46":f46,"tz":min(tz/6,1),"hl":hl}

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


# ═══════════════════════ MONTE CARLO V9 ══════════════════════════════════════
class MonteCarloV9:
    """V9: Returns full margin & total distributions for Spread/Total analysis."""

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
        # V9: Altitude penalty
        if feat[FEAT.index("altitude_factor")]>0:
            ae*=(1-feat[FEAT.index("altitude_factor")]*0.02)
        hw=hm/hm.sum() if hm.sum()>0 else np.ones(len(hm))/len(hm)
        aw=am/am.sum() if am.sum()>0 else np.ones(len(am))/len(am)

        wins=0; margins=np.empty(self.n_sims)
        home_sc=np.empty(self.n_sims); away_sc=np.empty(self.n_sims)
        totals=np.empty(self.n_sims)

        ref_t=feat[FEAT.index("ref_foul_tendency")]
        for i in range(self.n_sims):
            hss=np.random.normal(he,hs); ass=np.random.normal(ae,as_)
            ht=np.dot(hw,hss); at=np.dot(aw,ass)
            sp=pace+np.random.normal(0,2.5)
            hpts=sp*ht/100+np.random.normal(0,1.2)
            apts=sp*at/100+np.random.normal(0,1.2)
            hpts+=np.random.normal(1.5,0.5)  # HCA
            if ref_t>0.3:
                fp=np.random.normal(ref_t*2,0.5)
                hpts+=fp*0.5; apts+=fp*0.5
            if hpts>apts: wins+=1
            margins[i]=hpts-apts; home_sc[i]=hpts; away_sc[i]=apts; totals[i]=hpts+apts

        wp=wins/self.n_sims; ms=np.std(margins); em=np.mean(margins); et=np.mean(totals)
        margin_dist={"mean":em,"std":ms,
            "p10":float(np.percentile(margins,10)),"p25":float(np.percentile(margins,25)),
            "p50":float(np.percentile(margins,50)),"p75":float(np.percentile(margins,75)),
            "p90":float(np.percentile(margins,90))}
        total_dist={"mean":et,"std":float(np.std(totals)),
            "p10":float(np.percentile(totals,10)),"p25":float(np.percentile(totals,25)),
            "p50":float(np.percentile(totals,50)),"p75":float(np.percentile(totals,75)),
            "p90":float(np.percentile(totals,90))}
        return {"wp":wp,"em":em,"ms":ms,"conf":max(0,1-ms/18),"volatility":ms,
                "n_players":(len(hp),len(ap)),"margin_dist":margin_dist,"total_dist":total_dist,
                "projected_total":et,"home_scores_mean":float(np.mean(home_sc)),
                "away_scores_mean":float(np.mean(away_sc)),
                "raw_margins":margins,"raw_totals":totals}

    def spread_probability(self, margins, spread_line):
        return float(np.mean(margins > -spread_line))

    def total_probability(self, totals, total_line):
        return {"over_prob":float(np.mean(totals>total_line)),
                "under_prob":float(np.mean(totals<total_line))}

    def dead_number_check(self, margins, spread_line):
        near=np.abs(margins-(-spread_line))<DEAD_NUMBER_RANGE
        return float(np.mean(near))>0.12

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

# ═══════════════════════ MODEL V9 ════════════════════════════════════════════
class ModelV9:
    def __init__(self):
        self.xgb = None; self.scaler = StandardScaler(); self.mc = None; self.engine = None
        self.trained = False; self.tX = []; self.ty = []; self.tW = []
        self._day_picks = []; self._current_day = None
        self.xgb_total = None
        self.total_ty = []
        logger.info(f"ModelV9 | DIAMOND>={DIAMOND_THRESHOLD:.0%} EV>={DIAMOND_EV_MIN} "
                     f"| SpreadVal>={SPREAD_VALUE_MIN} | TotalVal>={TOTAL_VALUE_MIN}")

    def connect_mc(self, mc): self.mc = mc
    def connect_engine(self, eng): self.engine = eng

    def add(self, X, y, gid, weight=1.0, total_pts=None):
        self.tX.append(X); self.ty.append(y); self.tW.append(weight)
        if total_pts is not None:
            self.total_ty.append(total_pts)

    def retrain(self):
        if len(self.tX) < 300: return False
        X = np.array(self.tX); y = np.array(self.ty); W = np.array(self.tW)
        Xs = self.scaler.fit_transform(X)
        n_pos = max(np.sum(y), 1); n_neg = len(y) - n_pos; spw = (n_neg / n_pos) * 2.5
        self.xgb = xgb.XGBClassifier(
            n_estimators=600, max_depth=5, learning_rate=0.018, subsample=0.8,
            colsample_bytree=0.7, reg_alpha=0.3, reg_lambda=2.0, min_child_weight=6,
            gamma=0.15, scale_pos_weight=spw, random_state=42,
            use_label_encoder=False, eval_metric="logloss")
        self.xgb.fit(Xs, y, sample_weight=W)
        if len(self.total_ty) >= 300:
            yt = np.array(self.total_ty[:len(self.tX)])
            if len(yt) == len(Xs):
                self.xgb_total = xgb.XGBRegressor(
                    n_estimators=400, max_depth=4, learning_rate=0.02, subsample=0.8,
                    colsample_bytree=0.6, reg_alpha=0.5, reg_lambda=2.5, random_state=42)
                self.xgb_total.fit(Xs, yt, sample_weight=W)
                logger.info(f"XGB Total regressor trained | {len(yt)} samples")
        self.trained = True
        logger.info(f"XGB V9 trained | {len(self.tX)} samples | spw={spw:.2f}")
        return True

    def predict(self, X, game=None, live_mode=False):
        gd = game["game_date"] if game is not None else None
        mc_vol = 12; mc_result = None
        if not self.trained:
            if self.mc and game is not None:
                mc_result = self.mc.run(game, X)
                wp = mc_result["wp"]; conf = mc_result["conf"]; mc_vol = mc_result.get("volatility", 12)
            else:
                try: wp = X[FEAT.index("elo_exp")]
                except: wp = 0.5
                conf = 0.5
        else:
            Xs = self.scaler.transform(X.reshape(1, -1))
            wp_x = self.xgb.predict_proba(Xs)[0][1]
            if self.mc and game is not None:
                mc_result = self.mc.run(game, X)
                wp_m = mc_result["wp"]; conf = mc_result["conf"]; mc_vol = mc_result.get("volatility", 12)
                wp = 0.60 * wp_x + 0.40 * wp_m
            else:
                wp = wp_x; conf = 0.6

        mp = X[FEAT.index("mkt_prob_home")]; mg = wp - mp
        try: X[FEAT.index("mkt_gap")] = mg
        except: pass
        rlm = X[FEAT.index("rlm_signal")]; ph = wp > 0.5
        pp = wp if ph else 1 - wp; mpp = mp if ph else 1 - mp
        fo = 1 / max(mpp, 0.01); mo = fo * 0.95; ev = pp * mo - 1

        # Matchup DNA penalty
        dna = X[FEAT.index("matchup_dna_penalty")]
        if ph and dna == 1: wp *= (1 - MATCHUP_PENALTY)
        elif not ph and dna == -1: wp = 1 - ((1 - wp) * (1 - MATCHUP_PENALTY))

        # Fatigue
        ft = False
        if ph:
            if X[FEAT.index("h_heavy_legs")] == 1 or (X[FEAT.index("h_b2b")] == 1 and X[FEAT.index("h_q4_net_avg")] < 0):
                ft = True; wp *= 0.90
        else:
            if X[FEAT.index("a_heavy_legs")] == 1 or (X[FEAT.index("a_b2b")] == 1 and X[FEAT.index("a_q4_net_avg")] < 0):
                ft = True; wp = 1 - ((1 - wp) * 0.90)

        # Value trap
        vt = X[FEAT.index("value_trap_flag")]; vta = False
        if ph and vt == 1: wp *= 0.95; vta = True
        elif not ph and vt == -1: wp = 1 - ((1 - wp) * 0.95); vta = True

        # Playoff urgency
        pu_pen = False
        if ph:
            pu = X[FEAT.index("h_playoff_urgency")]
            if pu >= 1.0: wp *= 0.85; pu_pen = True
            elif pu >= 0.5: wp *= 0.93; pu_pen = True
        else:
            pu = X[FEAT.index("a_playoff_urgency")]
            if pu >= 1.0: wp = 1 - ((1 - wp) * 0.85); pu_pen = True
            elif pu >= 0.5: wp = 1 - ((1 - wp) * 0.93); pu_pen = True

        # V9: CLV Boost
        clv = X[FEAT.index("clv_home")]
        if clv > 0.3:
            wp += CLV_BOOST * (1 if ph else -1)
            wp = np.clip(wp, 0.01, 0.99)

        # V9: Steam Move Boost
        steam = X[FEAT.index("steam_signal")]
        if (ph and steam == 1) or (not ph and steam == -1):
            wp = min(wp * 1.02, 0.99)

        ppa = max(wp, 1 - wp); eva = ppa * mo - 1

        # V9: MULTI-MARKET ANALYSIS
        spread_pick = None; total_pick = None
        mkt_spread = X[FEAT.index("mkt_spread")] * 10
        mkt_total = X[FEAT.index("mkt_total")] * 250
        our_margin = mc_result["em"] if mc_result else 0
        our_total = mc_result.get("projected_total", 220) if mc_result else 220

        # Spread analysis
        spread_edge = abs(our_margin - (-mkt_spread))
        if mc_result and "margin_dist" in mc_result:
            md = mc_result["margin_dist"]
            if spread_edge >= SPREAD_VALUE_MIN:
                if our_margin > -mkt_spread:
                    spread_pick = {"side": "HOME", "line": mkt_spread, "our_margin": our_margin,
                                   "edge": spread_edge, "dead_number": False}
                else:
                    spread_pick = {"side": "AWAY", "line": mkt_spread, "our_margin": our_margin,
                                   "edge": spread_edge, "dead_number": False}
                near_line = abs(md["p50"] - (-mkt_spread)) < DEAD_NUMBER_RANGE
                if near_line:
                    spread_pick["dead_number"] = True

        # Totals analysis
        total_edge = our_total - mkt_total
        if abs(total_edge) >= TOTAL_VALUE_MIN:
            total_pick = {
                "side": "OVER" if total_edge > 0 else "UNDER",
                "line": mkt_total, "our_total": our_total, "edge": total_edge,
            }

        # VIP / DIAMOND with multi-market
        is_d = False; reason = ""
        rc = (ph and rlm == -1) or (not ph and rlm == 1)
        gk = abs(mg) >= MKT_GAP_MIN; ek = eva >= DIAMOND_EV_MIN

        if ppa >= DIAMOND_THRESHOLD and ek and gk and not rc:
            is_d = True; reason = "DIAMOND_ML"
        elif not is_d and not rc:
            rlmc = (ph and rlm == 1) or (not ph and rlm == -1)
            if rlmc and ppa >= 0.74 and ek: is_d = True; reason = "RLM_DIAMOND"
        if not is_d and ppa >= 0.74 and ek and clv > 0.5 and not rc:
            is_d = True; reason = "CLV_DIAMOND"
        if not is_d and ppa >= 0.72 and ek and steam != 0 and not rc:
            steam_aligned = (ph and steam == 1) or (not ph and steam == -1)
            if steam_aligned:
                is_d = True; reason = "STEAM_DIAMOND"

        rl = "Bajo" if mc_vol <= 8 else "Medio" if mc_vol <= 13 else "Alto"

        result = {
            "wp": wp, "conf": conf, "is_vip": is_d, "vip_reason": reason, "mkt_gap": mg,
            "ev": eva, "rlm": rlm, "fatigue_trap": ft, "value_trap": vta,
            "playoff_urgency_penalty": pu_pen,
            "mkt_odds": mo, "pick_home": ph, "risk_level": rl, "mc_volatility": mc_vol,
            "mkt_prob_home": mp,
            "spread_pick": spread_pick, "total_pick": total_pick,
            "our_margin": our_margin, "our_total": our_total,
            "clv": float(clv), "steam": int(steam),
            "margin_dist": mc_result.get("margin_dist") if mc_result else None,
            "total_dist": mc_result.get("total_dist") if mc_result else None,
        }

        if gd is not None:
            ds = str(gd)[:10]
            if self._current_day != ds:
                self._flush_day_picks(); self._current_day = ds; self._day_picks = []
            self._day_picks.append({"game": game, "result": result, "confidence": ppa, "ev": eva})
        return result

    def _flush_day_picks(self):
        if not self._day_picks: return
        if sum(1 for p in self._day_picks if p["result"]["is_vip"]) == 0:
            cands = [p for p in self._day_picks if p["confidence"] >= FALLBACK_THRESHOLD and p["ev"] > 0]
            cands.sort(key=lambda x: x["ev"], reverse=True)
            for c in cands[:3]:
                c["result"]["is_vip"] = True; c["result"]["vip_reason"] = "TOP3_FALLBACK"
                self._save_pick(c["game"], c["result"])

    def _save_pick(self, game, result, csv_path=None):
        if game is None: return
        csv_path = csv_path or PICKS_CSV_BT
        hid = game["home_team_id"]; aid = game["away_team_id"]
        ha = TEAM_ABBR.get(hid, "???"); aa = TEAM_ABBR.get(aid, "???")
        pick = ha if result["pick_home"] else aa
        conf = max(result["wp"], 1 - result["wp"])
        bet_types = ["ML"]
        sp = result.get("spread_pick")
        tp = result.get("total_pick")
        if sp and not sp.get("dead_number"): bet_types.append("SPREAD")
        if tp: bet_types.append("TOTAL")
        row = {
            "Fecha": str(game["game_date"])[:10],
            "Partido": f"{aa} @ {ha}",
            "Pick_ML": pick,
            "Tipo": "/".join(bet_types),
            "Confianza_IA": f"{conf:.3f}",
            "Prob_Mercado": f"{result.get('mkt_prob_home', 0.5):.3f}",
            "Cuota_Mkt": f"{result['mkt_odds']:.3f}",
            "Valor_Esperado": f"{result['ev']:.4f}",
            "Nivel_de_Riesgo": result.get("risk_level", "Medio"),
            "Razon": result.get("vip_reason", ""),
            "RLM": result.get("rlm", 0),
            "Spread_Linea": f"{sp['line']:.1f}" if sp else "",
            "Spread_Pick": sp["side"] if sp else "",
            "Spread_Nuestro_Margen": f"{sp['our_margin']:.1f}" if sp else "",
            "Spread_Dead_Number": sp.get("dead_number", False) if sp else "",
            "Total_Linea": f"{tp['line']:.1f}" if tp else "",
            "Total_Pick": tp["side"] if tp else "",
            "Total_Nuestro": f"{tp['our_total']:.1f}" if tp else "",
            "Total_Edge": f"{tp['edge']:.1f}" if tp else "",
            "CLV": f"{result.get('clv', 0):.3f}",
            "Steam": result.get("steam", 0),
            "Fatigue_Trap": result.get("fatigue_trap", False),
            "Value_Trap": result.get("value_trap", False),
            "Playoff_Urgency": result.get("playoff_urgency_penalty", False),
        }
        exists = os.path.exists(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=row.keys())
            if not exists: w.writeheader()
            w.writerow(row)

    def save_vip_pick(self, game, result, csv_path=None):
        self._save_pick(game, result, csv_path)

    def save(self, path=None):
        path = path or f"{MODEL_DIR}/nba_model_v9.pkl"
        with open(path, "wb") as fh:
            pickle.dump({"xgb": self.xgb, "xgb_total": self.xgb_total,
                          "scaler": self.scaler, "trained": self.trained,
                          "n": len(self.tX)}, fh)

    def load(self, path=None):
        path = path or f"{MODEL_DIR}/nba_model_v9.pkl"
        if not os.path.exists(path): return False
        with open(path, "rb") as fh: d = pickle.load(fh)
        if d.get("trained"):
            self.xgb = d["xgb"]; self.scaler = d["scaler"]; self.trained = True
            self.xgb_total = d.get("xgb_total")
            return True
        return False


# ═══════════════════════ RADAR JSON EXPORTER ═════════════════════════════════
class RadarExporter:
    @staticmethod
    def export_game(game, result, engine, output_dir=JSON_RADAR_DIR):
        hid = game["home_team_id"]; aid = game["away_team_id"]
        ha = TEAM_ABBR.get(hid, "???"); aa = TEAM_ABBR.get(aid, "???")
        gd = str(game["game_date"])[:10]
        hl = engine.log[hid]; al = engine.log[aid]

        def _radar(lg, w=10):
            r = lg[-w:]
            if not r:
                return {"dreb":0.5,"tov":0.5,"pace":0.5,"three_pct":0.5,"oreb":0.5,"ft_rate":0.5}
            dreb = np.mean([g.get("dreb", 33) or 33 for g in r]) / 45
            tov = 1 - np.mean([g.get("tov", 14) or 14 for g in r]) / 20
            poss = np.mean([g.get("poss", 98) or 98 for g in r])
            pace = np.clip((poss - 90) / 20, 0, 1)
            fg3a = np.mean([g.get("fg3a", 30) or 30 for g in r])
            fg3m = np.mean([g.get("fg3m", 11) or 11 for g in r])
            three_pct = fg3m / max(fg3a, 1)
            oreb = np.mean([g.get("oreb", 10) or 10 for g in r]) / 15
            fta = np.mean([g.get("fta", 22) or 22 for g in r])
            fga = np.mean([g.get("fga", 85) or 85 for g in r])
            ft_rate = fta / max(fga, 1)
            return {
                "dreb": round(float(np.clip(dreb, 0, 1)), 3),
                "tov": round(float(np.clip(tov, 0, 1)), 3),
                "pace": round(float(pace), 3),
                "three_pct": round(float(np.clip(three_pct / 0.45, 0, 1)), 3),
                "oreb": round(float(np.clip(oreb, 0, 1)), 3),
                "ft_rate": round(float(np.clip(ft_rate / 0.35, 0, 1)), 3),
            }

        def _clean(obj):
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items()}
            elif isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            return obj

        radar = _clean({
            "game_id": str(game["game_id"]),
            "date": gd,
            "matchup": f"{aa} @ {ha}",
            "home": ha, "away": aa,
            "home_radar": _radar(hl),
            "away_radar": _radar(al),
            "pick": {
                "ml": ha if result["pick_home"] else aa,
                "confidence": float(max(result["wp"], 1 - result["wp"])),
                "ev": float(result["ev"]),
                "risk": result["risk_level"],
                "vip_reason": result.get("vip_reason", ""),
            },
            "market": {
                "mkt_prob_home": float(result.get("mkt_prob_home", 0.5)),
                "rlm": int(result.get("rlm", 0)),
                "steam": int(result.get("steam", 0)),
                "clv": float(result.get("clv", 0)),
            },
            "multi_market": {
                "spread": result.get("spread_pick"),
                "total": result.get("total_pick"),
            },
            "distributions": {
                "margin": result.get("margin_dist"),
                "total": result.get("total_dist"),
            },
        })
        fname = f"{gd}_{aa}_at_{ha}.json"
        fpath = os.path.join(output_dir, fname)
        with open(fpath, "w", encoding="utf-8") as fh:
            json.dump(radar, fh, indent=2, ensure_ascii=False)
        return fpath


# ═══════════════════════ METRICS V9 ══════════════════════════════════════════
class MetricsV9:
    def __init__(self):
        self.P=[]; self.A=[]; self.M=[]; self.G=[]; self.V=[]; self.E=[]
        self.spread_correct = 0; self.spread_total = 0
        self.total_correct = 0; self.total_total = 0

    def add(self, p, a, m, g, v=False, e=0, spread_result=None, total_result=None):
        self.P.append(p); self.A.append(a); self.M.append(m)
        self.G.append(g); self.V.append(v); self.E.append(e)
        if spread_result is not None:
            self.spread_total += 1
            if spread_result: self.spread_correct += 1
        if total_result is not None:
            self.total_total += 1
            if total_result: self.total_correct += 1

    def report(self, last_n=None, label=""):
        if not self.P: return {}
        n = last_n or len(self.P); p = self.P[-n:]; a = self.A[-n:]; v = self.V[-n:]; e = self.E[-n:]
        pr = [1 if x > 0.5 else 0 for x in p]; acc = accuracy_score(a, pr); br = brier_score_loss(a, p)
        vi = [i for i, x in enumerate(v) if x]; nv = len(vi)
        if nv > 0:
            vp = [p[i] for i in vi]; va = [a[i] for i in vi]; vr = [1 if x > 0.5 else 0 for x in vp]
            vacc = accuracy_score(va, vr); vbr = brier_score_loss(va, vp)
            vev = np.mean([e[i] for i in vi]); vc = sum(1 for j in range(nv) if vr[j] == va[j])
        else:
            vacc = 0; vbr = 1; vev = 0; vc = 0
        print(f"\n{'='*70}")
        print(f"  {label} ({n} juegos)")
        print(f"{'='*70}")
        print(f"  ML GLOBAL: Acc {acc:.1%} | Brier {br:.4f}")
        print(f"  DIAMOND:   {nv}/{n} ({nv/n*100:.1f}%)")
        if nv > 0:
            fp = nv - vc; em = "ELITE" if vacc >= 0.85 else "DIAMOND" if vacc >= 0.78 else "WARN"
            print(f"    ML Acc:    {vacc:.1%}  [{em}]")
            print(f"    EV:        {vev:+.3f}")
            print(f"    FP:        {fp}/{nv} ({fp/nv*100:.1f}%)")
        if self.spread_total > 0:
            sa = self.spread_correct / self.spread_total
            print(f"  SPREAD:    {self.spread_correct}/{self.spread_total} ({sa:.1%})")
        if self.total_total > 0:
            ta = self.total_correct / self.total_total
            print(f"  TOTAL:     {self.total_correct}/{self.total_total} ({ta:.1%})")
        print(f"{'='*70}")
        return {"acc": acc, "brier": br, "vip_acc": vacc, "vip_ev": vev}


# ═══════════════════════ TRAIN MODE V9 ═══════════════════════════════════════
def run_train(train_s, eval_s, ckpt, db, n_sims):
    t0 = time.time(); all_s = train_s + [eval_s]
    print(f"\n{'='*70}")
    print(f"  NBA SYNDICATE V9 - THE PROFESSIONAL EDGE - TRAINING")
    print(f"  Train: {', '.join(train_s)}")
    print(f"  Eval:  {eval_s}")
    print(f"  Features: {N_FEAT} | Sims: {n_sims:,}")
    print(f"  Markets: ML + Spread + Totals")
    print(f"{'='*70}\n")
    if os.path.exists(PICKS_CSV_BT): os.remove(PICKS_CSV_BT)

    dl = DataLoader(db); games = dl.load_games(all_s)
    bs = dl.load_boxscores(); pl = dl.load_players(); od = dl.load_odds(); dl.close()
    logger.info(f"Games:{len(games)} BS:{len(bs)} PL:{len(pl)} Odds:{len(od)}")

    eng = EngineV9(bs, pl, od, games); mc = MonteCarloV9(eng, n_sims)
    mdl = ModelV9(); mdl.connect_mc(mc); mdl.connect_engine(eng)
    trm = MetricsV9(); evm = MetricsV9()
    proc = 0; skip = 0; cur = None
    current_seasons = {"2024-2025", "2025-2026"}
    radar_exp = RadarExporter()

    for _, g in games.iterrows():
        s = g["season"]; ie = (s == eval_s)
        if s != cur:
            if cur is not None and not ie:
                trm.report(label=f"FIN {cur}"); mdl._flush_day_picks()
                for t in eng.elo: eng.elo[t] = 0.75 * eng.elo[t] + 0.25 * ELO_INIT
            eng.reset_season(s); cur = s; logger.info(f"Season: {s} {'[EVAL]' if ie else '[TRAIN]'}")

        is_curr = s in current_seasons
        feat = eng.compute(g, is_current_season=is_curr)
        if feat is not None:
            pred = mdl.predict(feat, game=g)
            aw = g["home_win"]; tk = evm if ie else trm
            actual_margin = g["margin"]
            actual_total = g.get("total_pts", 0) or (g["home_score"] + g["away_score"])

            sp_result = None
            sp = pred.get("spread_pick")
            if sp:
                mkt_spr = sp["line"]
                if sp["side"] == "HOME":
                    sp_result = (actual_margin > -mkt_spr)
                else:
                    sp_result = (actual_margin < -mkt_spr)

            tp_result = None
            tp = pred.get("total_pick")
            if tp:
                if tp["side"] == "OVER":
                    tp_result = (actual_total > tp["line"])
                else:
                    tp_result = (actual_total < tp["line"])

            tk.add(pred["wp"], aw, g["margin"], g["game_id"], pred["is_vip"], pred["ev"],
                   spread_result=sp_result, total_result=tp_result)

            if not ie:
                weight = 1.4 if is_curr else 0.7
                mdl.add(feat, aw, g["game_id"], weight=weight, total_pts=actual_total)
            if pred["is_vip"]:
                mdl.save_vip_pick(g, pred)
                if ie:
                    radar_exp.export_game(g, pred, eng)
            proc += 1
        else:
            skip += 1
        eng.update(g)
        if not ie and proc > 0 and proc % 300 == 0:
            if mdl.retrain(): logger.info(f"Retrained ({len(mdl.tX)} samples)")
        if proc > 0 and proc % ckpt == 0:
            tk = evm if ie else trm
            tk.report(last_n=ckpt, label=f"{'EVAL' if ie else 'TRAIN'} #{proc}")

    mdl._flush_day_picks()
    if mdl.retrain(): logger.info(f"Final train ({len(mdl.tX)} samples)")
    mdl.save()
    print(f"\n  Procesados: {proc} | Omitidos: {skip} | Tiempo: {(time.time()-t0)/60:.1f} min")
    print(f"\n  --- TRAIN ---"); trm.report(label="TRAIN FINAL")
    print(f"\n  --- EVAL ({eval_s}) ---"); evm.report(label="EVAL FINAL")
    if mdl.trained and mdl.xgb is not None:
        print(f"\n  Top 25 Features:")
        fi = sorted(zip(FEAT, mdl.xgb.feature_importances_), key=lambda x: x[1], reverse=True)
        for nm, im in fi[:25]: print(f"    {nm:32s} {im:.4f} {'X'*int(im*100)}")
    print(f"\n  Radar JSONs exported to: {JSON_RADAR_DIR}/")
    print(f"{'='*70}\n")


# ═══════════════════════ LIVE MODE V9 ═══════════════════════════════════════
def run_live(db, n_sims):
    today = date.today().isoformat()
    csv_path = PICKS_CSV_LIVE
    print(f"\n{'='*70}")
    print(f"  NBA SYNDICATE V9 - THE PROFESSIONAL EDGE - LIVE")
    print(f"  Date: {today}")
    print(f"  Output: {csv_path}")
    print(f"  Markets: ML + Spread + Totals")
    print(f"{'='*70}\n")

    mdl = ModelV9()
    if not mdl.load():
        print("No trained model found. Run --mode train first.")
        return

    print("Loading historical DB for engine state...")
    dl = DataLoader(db)
    all_seasons = TRAIN_SEASONS + [CURRENT_SEASON]
    games = dl.load_games(all_seasons)
    bs = dl.load_boxscores(); pl = dl.load_players(); od = dl.load_odds(); dl.close()

    eng = EngineV9(bs, pl, od, games)
    mc = MonteCarloV9(eng, n_sims)
    mdl.connect_mc(mc); mdl.connect_engine(eng)

    print("Replaying historical games to build state...")
    cur = None
    for _, g in games.iterrows():
        s = g["season"]
        if s != cur:
            if cur is not None:
                for t in eng.elo: eng.elo[t] = 0.75 * eng.elo[t] + 0.25 * ELO_INIT
            eng.reset_season(s); cur = s
        eng.update(g)
    print(f"   State built from {len(games)} games")

    sports = SportsAPIClient()
    odds_client = OddsAPIClient()
    standings = sports.get_standings()
    eng.inject_standings(standings)

    todays_games = sports.get_todays_games(today)
    raw_odds = odds_client.get_live_odds()
    opening_odds = odds_client.get_opening_odds()
    live_odds = odds_client.parse_game_odds(raw_odds, opening_odds) if raw_odds else {}
    eng.inject_live_odds(live_odds)

    if os.path.exists(csv_path): os.remove(csv_path)
    picks = []
    radar_exp = RadarExporter()

    game_entries = []
    for (hid, aid), od_data in live_odds.items():
        game_entries.append({
            "game_id": hash((hid, aid, today)) % 10**8,
            "home_team_id": hid, "away_team_id": aid,
            "game_date": pd.Timestamp(today),
            "home_score": None, "away_score": None,
            "home_win": None, "margin": None,
        })

    if not game_entries:
        print("No matchups to analyse. Check API keys and game schedule.")
        return

    print(f"\nAnalysing {len(game_entries)} games for {today}...\n")

    for ge in game_entries:
        hid = ge["home_team_id"]; aid = ge["away_team_id"]
        ha = TEAM_ABBR.get(hid, "???"); aa = TEAM_ABBR.get(aid, "???")
        g_series = pd.Series(ge)
        feat = eng.compute(g_series, is_current_season=True)
        if feat is None:
            print(f"  SKIP  {aa} @ {ha}: insufficient data")
            continue

        pred = mdl.predict(feat, game=g_series, live_mode=True)
        pick = ha if pred["pick_home"] else aa
        conf = max(pred["wp"], 1 - pred["wp"])
        mkt_p = pred.get("mkt_prob_home", 0.5)
        mkt_pick = mkt_p if pred["pick_home"] else 1 - mkt_p

        # Build status line
        status = "DIAMOND" if pred["is_vip"] else "  ---  "
        sp = pred.get("spread_pick")
        tp = pred.get("total_pick")
        sp_str = f"Spread:{sp['side']} {sp['line']:+.1f}" if sp else ""
        tp_str = f"Total:{tp['side']} {tp['line']:.1f}" if tp else ""

        print(f"  [{status}] {aa:3s} @ {ha:3s}  ML:{pick:3s} {conf:.1%}  "
              f"EV:{pred['ev']:+.3f}  RLM:{pred['rlm']:+.1f}  "
              f"Steam:{pred['steam']}  CLV:{pred['clv']:.2f}  "
              f"{sp_str}  {tp_str}  Risk:{pred['risk_level']}")

        if pred["is_vip"]:
            mdl.save_vip_pick(g_series, pred, csv_path=csv_path)
            radar_exp.export_game(g_series, pred, eng)
            picks.append(pred)

    if not picks:
        print("\nNo DIAMOND picks today. Checking TOP-3 fallback...")
        mdl._flush_day_picks()

    print(f"\n{'='*70}")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"  {len(df)} DIAMOND picks saved to {csv_path}")
        if len(df) > 0:
            print(f"     EV promedio: {df['Valor_Esperado'].astype(float).mean():.4f}")
        n_spread = sum(1 for _, r in df.iterrows() if r.get("Spread_Pick", ""))
        n_total = sum(1 for _, r in df.iterrows() if r.get("Total_Pick", ""))
        print(f"     Con Spread: {n_spread} | Con Total: {n_total}")
    else:
        print("  No picks generated for today.")
    print(f"  Radar JSONs in: {JSON_RADAR_DIR}/")
    print(f"{'='*70}\n")


# ═══════════════════════ MAIN ════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(description="NBA Syndicate V9 - The Professional Edge")
    p.add_argument("--mode", choices=["train", "live"], default="train",
                   help="train = backtest on DB | live = today's picks from APIs")
    p.add_argument("--seasons", nargs="+", default=TRAIN_SEASONS)
    p.add_argument("--eval", default=CURRENT_SEASON)
    p.add_argument("--checkpoint", type=int, default=CHECKPOINT)
    p.add_argument("--db", default=DB_PATH)
    p.add_argument("--sims", type=int, default=DEFAULT_SIMS)
    p.add_argument("--eval-only", action="store_true")
    a = p.parse_args()

    if a.eval_only:
        m = ModelV9()
        if m.load(): print(f"Model V9 loaded ({len(m.xgb.feature_importances_)} features)")
        else: print("No saved model")
        return

    if a.mode == "train":
        run_train(a.seasons, a.eval, a.checkpoint, a.db, a.sims)
    else:
        run_live(a.db, a.sims)


if __name__ == "__main__":
    main()
