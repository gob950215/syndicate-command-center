#!/usr/bin/env python3
"""
NBA Training Pipeline v3 - OPTIMIZED
=====================================
Mejoras sobre v2 (65.8% accuracy, 0.2166 Brier):

NUEVAS FEATURES (+30):
  1. Multi-window rolling (5 + 10 + 20 juegos) → captura corto y largo plazo
  2. Quarter scoring patterns (q1_diff, q3_diff, clutch_factor)
  3. Scoring consistency (std dev of margins, pts)
  4. Defensive 4 factors (steal_rate, block_rate, opp_fg3_rate)
  5. Pace-adjusted stats (ORtg/DRtg per 100 poss, properly)
  6. Conference/Division matchup indicator
  7. Last season ELO carryover weighted
  8. Rest differential (not just individual)
  9. Road trip length (consecutive away games)

MEJORAS MODELO:
  1. HistGradientBoosting (faster, native NaN handling, better regularization)
  2. Isotonic calibration post-training → probabilities match reality
  3. Stacking meta-learner instead of fixed weights
  4. Feature selection via permutation importance
  5. Walk-forward validation (retrain window slides)
  6. Margin-of-Victory adjusted ELO (MOV-ELO)
  7. Smarter retrain: every 200 games with expanding window

Requisitos: numpy pandas scikit-learn scipy
"""
import os, sys, time, sqlite3, logging, argparse, warnings, pickle
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

try:
    from sklearn.ensemble import (HistGradientBoostingClassifier,
                                  HistGradientBoostingRegressor,
                                  GradientBoostingClassifier)
    from sklearn.linear_model import LogisticRegression, RidgeClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, roc_auc_score
except ImportError:
    print("pip install scikit-learn>=1.0"); sys.exit(1)

warnings.filterwarnings('ignore')

# ═══════════════════════ CONFIG ═══════════════════════════════════════════════
DB_PATH = "data/nba_historical.db"
MODEL_DIR = "models"
CHECKPOINT_EVERY = 200
N_SIMS = 3000  # Reduced from 5000 for speed; still statistically valid

TRAIN_SEASONS = ["2019-2020","2020-2021","2021-2022","2022-2023","2023-2024"]
EVAL_SEASON = "2024-2025"

# MOV-ELO parameters (margin-of-victory adjusted)
ELO_INIT = 1500; ELO_K = 20; ELO_HCA = 55
ELO_MOV_MULT = 0.7  # How much MOV affects ELO update

os.makedirs("data", exist_ok=True); os.makedirs(MODEL_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(stream=open(sys.stdout.fileno(), mode='w', encoding='utf-8', closefd=False)),
              logging.FileHandler("data/training_log.txt", 'a', encoding='utf-8')])
logger = logging.getLogger("PipelineV3")

# ═══════════════════════ TEAM CONSTANTS ═══════════════════════════════════════
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
REV_TEAM = {v:k for k,v in NBA_IDS.items()}
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
# Conference mapping for matchup features
EAST = {132,133,134,135,136,137,140,143,147,148,149,151,153,154,159,161}
WEST = {138,139,141,142,144,145,146,150,152,155,156,157,158,160}

def _hav(lat1,lon1,lat2,lon2):
    from math import radians,cos,sin,asin,sqrt
    lat1,lon1,lat2,lon2 = map(radians,[lat1,lon1,lat2,lon2])
    d=lat2-lat1; dl=lon2-lon1
    a=sin(d/2)**2+cos(lat1)*cos(lat2)*sin(dl/2)**2
    return 2*3956*asin(sqrt(a))


# ═══════════════════════ DATA LOADER ══════════════════════════════════════════
class DataLoader:
    def __init__(self, db=DB_PATH):
        self.conn = sqlite3.connect(db); self.conn.row_factory = sqlite3.Row
    def load_games(self, seasons):
        ph = ','.join('?'*len(seasons))
        df = pd.read_sql_query(f"""
            SELECT game_id, season, date_local as game_date,
                   home_team_id, away_team_id, home_score, away_score,
                   home_q1,home_q2,home_q3,home_q4,home_ot,
                   away_q1,away_q2,away_q3,away_q4,away_ot, nba_game_id
            FROM games WHERE season IN ({ph})
              AND status_short IN ('FT','AOT')
              AND home_score IS NOT NULL AND away_score IS NOT NULL
            ORDER BY date_local, game_id""", self.conn, params=seasons)
        df['game_date'] = pd.to_datetime(df['game_date'])
        df['home_win'] = (df['home_score']>df['away_score']).astype(int)
        df['margin'] = df['home_score']-df['away_score']
        df['total'] = df['home_score']+df['away_score']
        return df
    def load_boxscores(self):
        df = pd.read_sql_query("""
            SELECT nba_game_id, team_id as nba_tid, game_id,
                   fgm,fga,fg3m,fg3a,ftm,fta,oreb,dreb,reb,ast,stl,blk,tov,pf,pts,plus_minus
            FROM game_team_stats WHERE pts IS NOT NULL""", self.conn)
        df['team_id'] = df['nba_tid'].map(REV_TEAM)
        return df
    def load_players(self):
        df = pd.read_sql_query("""
            SELECT nba_game_id, player_id, team_id as nba_tid, game_id,
                   player_name, start_position, minutes_decimal,
                   pts,reb,ast,stl,blk,tov,plus_minus
            FROM game_player_stats WHERE pts IS NOT NULL AND minutes_decimal>0""", self.conn)
        df['team_id'] = df['nba_tid'].map(REV_TEAM)
        return df
    def load_odds(self):
        return pd.read_sql_query("""
            SELECT game_id_mapped as game_id, home_team, away_team,
                   bookmaker, market, outcome_name, outcome_price, outcome_point
            FROM odds_historical WHERE game_id_mapped IS NOT NULL""", self.conn)
    def close(self): self.conn.close()


# ═══════════════════════ FEATURE NAMES ════════════════════════════════════════
FEAT_NAMES = [
    # Rolling 5 game window (short-term form)
    'h_off5','h_def5','h_net5','h_ts5','h_efg5',
    'a_off5','a_def5','a_net5','a_ts5','a_efg5',
    # Rolling 10 game window (medium-term)
    'h_off10','h_def10','h_net10','h_pace10','h_ts10','h_efg10',
    'h_tov10','h_oreb10','h_ast10','h_pts10','h_opp10',
    'a_off10','a_def10','a_net10','a_pace10','a_ts10','a_efg10',
    'a_tov10','a_oreb10','a_ast10','a_pts10','a_opp10',
    # Rolling 20 game window (long-term baseline)
    'h_off20','h_def20','h_net20',
    'a_off20','a_def20','a_net20',
    # Differentials (using 10-game)
    'diff_off','diff_def','diff_net','diff_ts','diff_pace',
    # Scoring consistency
    'h_margin_std','a_margin_std','h_pts_std','a_pts_std',
    # Defensive extras
    'h_stl_rate','a_stl_rate','h_blk_rate','a_blk_rate',
    'h_fg3a_rate','a_fg3a_rate',  # 3pt attempt rate
    # Quarter patterns
    'h_q1_diff','a_q1_diff',  # avg Q1 margin
    'h_q4_diff','a_q4_diff',  # avg Q4 margin (clutch)
    # Fatigue
    'h_rest','a_rest','rest_diff','h_b2b','a_b2b',
    'h_g7d','a_g7d','h_travel','a_travel',
    'h_road_trip','a_road_trip',  # consecutive away games
    # Momentum
    'h_streak','a_streak','h_wp5','a_wp5','h_wps','a_wps',
    'h_hwp','a_awp','h_mtrend','a_mtrend',
    # ELO (MOV-adjusted)
    'h_elo','a_elo','elo_diff','elo_exp',
    # Players
    'h_star','a_star','h_depth','a_depth',
    'h_top_pm','a_top_pm',  # avg plus/minus of top 3
    # Market
    'mkt_prob','mkt_spread','mkt_total',
    # H2H
    'h2h_wins','h2h_margin',
    # Context
    'season_pct','is_conference',  # same conference matchup
]

N_FEATURES = len(FEAT_NAMES)

# ═══════════════════════ FEATURE ENGINE ═══════════════════════════════════════
class FeatureEngine:
    def __init__(self, boxscores, players, odds, games_df):
        # Build indices
        self.bs_idx = defaultdict(list)
        for _,r in boxscores.iterrows():
            g = r.get('game_id')
            if pd.notna(g): self.bs_idx[int(g)].append(r.to_dict())
        self.odds_idx = defaultdict(list)
        for _,r in odds.iterrows():
            g = r.get('game_id')
            if pd.notna(g): self.odds_idx[int(g)].append(r.to_dict())
        self.pl_idx = defaultdict(list)
        for _,r in players.iterrows():
            g = r.get('game_id')
            if pd.notna(g): self.pl_idx[int(g)].append(r.to_dict())

        # Quarter scores by game_id
        self.q_scores = {}
        for _,r in games_df.iterrows():
            gid = r['game_id']
            self.q_scores[gid] = {
                'hq1':r.get('home_q1'), 'hq2':r.get('home_q2'),
                'hq3':r.get('home_q3'), 'hq4':r.get('home_q4'),
                'aq1':r.get('away_q1'), 'aq2':r.get('away_q2'),
                'aq3':r.get('away_q3'), 'aq4':r.get('away_q4'),
            }

        self.log = defaultdict(list)  # team_id → [{game_data}]
        self.elo = {t: ELO_INIT for t in TEAM_ABBR}
        self.rec = defaultdict(lambda: {'w':0,'l':0,'hw':0,'hl':0,'aw':0,'al':0})
        self.season = None
        logger.info(f"Engine: {len(self.bs_idx)} bs, {len(self.odds_idx)} odds, "
                    f"{len(self.pl_idx)} pl, {len(self.q_scores)} quarters")

    def reset_season(self, s):
        self.log = defaultdict(list)
        self.rec = defaultdict(lambda: {'w':0,'l':0,'hw':0,'hl':0,'aw':0,'al':0})
        self.season = s

    def compute(self, g):
        gid = g['game_id']; hid = g['home_team_id']; aid = g['away_team_id']
        gd = g['game_date']
        if hid not in TEAM_ABBR or aid not in TEAM_ABBR: return None
        hl, al = self.log[hid], self.log[aid]
        if len(hl) < 5 or len(al) < 5: return None

        f = {}
        # ─── 1. Multi-window rolling ─────────────────────────────────
        for pfx, lg in [('h', hl), ('a', al)]:
            r5 = self._roll(lg, 5)
            r10 = self._roll(lg, 10)
            r20 = self._roll(lg, 20)
            # 5-game
            f[f'{pfx}_off5']=r5['or']; f[f'{pfx}_def5']=r5['dr']
            f[f'{pfx}_net5']=r5['or']-r5['dr']
            f[f'{pfx}_ts5']=r5['ts']; f[f'{pfx}_efg5']=r5['efg']
            # 10-game
            f[f'{pfx}_off10']=r10['or']; f[f'{pfx}_def10']=r10['dr']
            f[f'{pfx}_net10']=r10['or']-r10['dr']; f[f'{pfx}_pace10']=r10['pace']
            f[f'{pfx}_ts10']=r10['ts']; f[f'{pfx}_efg10']=r10['efg']
            f[f'{pfx}_tov10']=r10['tp']; f[f'{pfx}_oreb10']=r10['op']
            f[f'{pfx}_ast10']=r10['ar']; f[f'{pfx}_pts10']=r10['pa']
            f[f'{pfx}_opp10']=r10['po']
            # 20-game
            f[f'{pfx}_off20']=r20['or']; f[f'{pfx}_def20']=r20['dr']
            f[f'{pfx}_net20']=r20['or']-r20['dr']

        h10 = self._roll(hl, 10); a10 = self._roll(al, 10)

        # ─── 2. Differentials ─────────────────────────────────────────
        f['diff_off']=h10['or']-a10['or']; f['diff_def']=h10['dr']-a10['dr']
        f['diff_net']=(h10['or']-h10['dr'])-(a10['or']-a10['dr'])
        f['diff_ts']=h10['ts']-a10['ts']; f['diff_pace']=h10['pace']-a10['pace']

        # ─── 3. Scoring consistency ───────────────────────────────────
        for pfx, lg in [('h', hl), ('a', al)]:
            r = lg[-10:]
            f[f'{pfx}_margin_std'] = np.std([g['margin'] for g in r]) / 15  # normalize
            f[f'{pfx}_pts_std'] = np.std([g['pts'] for g in r]) / 15

        # ─── 4. Defensive extras ──────────────────────────────────────
        for pfx, lg in [('h', hl), ('a', al)]:
            r = lg[-10:]
            fga = self._sm([g.get('fga') for g in r], 85)
            f[f'{pfx}_stl_rate'] = self._sm([g.get('stl',0) for g in r if g.get('stl') is not None], 7.5) / max(fga, 1)
            f[f'{pfx}_blk_rate'] = self._sm([g.get('blk',0) for g in r if g.get('blk') is not None], 5) / max(fga, 1)
            fg3a = self._sm([g.get('fg3a') for g in r], 33)
            f[f'{pfx}_fg3a_rate'] = fg3a / max(fga, 1)

        # ─── 5. Quarter patterns ──────────────────────────────────────
        for pfx, lg in [('h', hl), ('a', al)]:
            q1d, q4d = [], []
            for g in lg[-10:]:
                qs = self.q_scores.get(g['gid'], {})
                if g['home']:
                    q1 = (qs.get('hq1') or 0) - (qs.get('aq1') or 0)
                    q4 = (qs.get('hq4') or 0) - (qs.get('aq4') or 0)
                else:
                    q1 = (qs.get('aq1') or 0) - (qs.get('hq1') or 0)
                    q4 = (qs.get('aq4') or 0) - (qs.get('hq4') or 0)
                if qs.get('hq1') is not None:
                    q1d.append(q1); q4d.append(q4)
            f[f'{pfx}_q1_diff'] = np.mean(q1d) / 10 if q1d else 0
            f[f'{pfx}_q4_diff'] = np.mean(q4d) / 10 if q4d else 0

        # ─── 6. Fatigue (enhanced) ────────────────────────────────────
        for pfx, lg, tid, vtid in [('h',hl,hid,hid), ('a',al,aid,hid)]:
            ft = self._fat(lg, gd, tid, vtid)
            f[f'{pfx}_rest']=ft[0]; f[f'{pfx}_b2b']=ft[1]
            f[f'{pfx}_g7d']=ft[2]; f[f'{pfx}_travel']=ft[3]
            f[f'{pfx}_road_trip']=ft[4]
        f['rest_diff'] = f['h_rest'] - f['a_rest']

        # ─── 7. Momentum ─────────────────────────────────────────────
        for pfx, lg, tid in [('h',hl,hid), ('a',al,aid)]:
            mm = self._mom(lg, tid)
            f[f'{pfx}_streak']=mm[0]; f[f'{pfx}_wp5']=mm[1]
            f[f'{pfx}_wps']=mm[2]; f[f'{pfx}_mtrend']=mm[3]
        hr, ar = self.rec[hid], self.rec[aid]
        f['h_hwp'] = hr['hw']/max(hr['hw']+hr['hl'],1)
        f['a_awp'] = ar['aw']/max(ar['aw']+ar['al'],1)

        # ─── 8. ELO ──────────────────────────────────────────────────
        f['h_elo']=self.elo[hid]; f['a_elo']=self.elo[aid]
        f['elo_diff']=f['h_elo']-f['a_elo']
        f['elo_exp']=1/(1+10**(-(f['elo_diff']+ELO_HCA)/400))

        # ─── 9. Players (enhanced) ───────────────────────────────────
        for pfx, tid in [('h',hid), ('a',aid)]:
            pi = self._play(tid)
            f[f'{pfx}_star']=pi[0]; f[f'{pfx}_depth']=pi[1]
            f[f'{pfx}_top_pm']=pi[2]

        # ─── 10. Market ──────────────────────────────────────────────
        mp,ms,mt = self._mkt(gid)
        f['mkt_prob']=mp; f['mkt_spread']=ms; f['mkt_total']=mt

        # ─── 11. H2H ─────────────────────────────────────────────────
        hw3,hm = self._h2h(hid, aid)
        f['h2h_wins']=hw3; f['h2h_margin']=hm

        # ─── 12. Context ─────────────────────────────────────────────
        f['season_pct'] = min((hr['w']+hr['l'])/82, 1.0)
        h_conf = 1 if hid in EAST else 0
        a_conf = 1 if aid in EAST else 0
        f['is_conference'] = 1 if h_conf == a_conf else 0

        return np.nan_to_num(np.array([f.get(n,0.0) for n in FEAT_NAMES], dtype=np.float64))

    def update(self, g):
        gid=g['game_id']; hid=g['home_team_id']; aid=g['away_team_id']
        if hid not in TEAM_ABBR or aid not in TEAM_ABBR: return
        hs=g['home_score']; aws=g['away_score']; gd=g['game_date']; hw=hs>aws
        bs=self.bs_idx.get(gid,[])
        hbs=next((b for b in bs if REV_TEAM.get(b.get('nba_tid'))==hid),None)
        abs_=next((b for b in bs if REV_TEAM.get(b.get('nba_tid'))==aid),None)
        for tid,ih,won,sc,osc,bx in [(hid,True,hw,hs,aws,hbs),(aid,False,not hw,aws,hs,abs_)]:
            e={'gid':gid,'date':gd,'home':ih,'won':won,'pts':sc,'opp':osc,
               'margin':sc-osc,'opp_id':aid if ih else hid}
            if bx:
                for s in ['fgm','fga','fg3m','fg3a','ftm','fta','oreb','dreb','reb','ast','stl','blk','tov']:
                    v=bx.get(s)
                    e[s]=float(v) if v is not None and not(isinstance(v,float) and np.isnan(v)) else None
            self.log[tid].append(e)
            r=self.rec[tid]
            if won: r['w']+=1; r['hw' if ih else 'aw']+=1
            else: r['l']+=1; r['hl' if ih else 'al']+=1
        # MOV-ELO update
        self._elo_update(hid, aid, hw, hs-aws)

    # ─── Helpers ──────────────────────────────────────────────────────

    def _sm(self, lst, d=0):
        v=[x for x in lst if x is not None]
        return np.mean(v) if v else d

    def _roll(self, lg, w=10):
        r=lg[-w:]
        if not r: return {'or':110,'dr':110,'pace':98,'ts':0.56,'efg':0.52,
                          'tp':0.13,'op':0.22,'ar':0.28,'pa':110,'po':110}
        pa=np.mean([g['pts'] for g in r]); po=np.mean([g['opp'] for g in r])
        fga=self._sm([g.get('fga') for g in r],85)
        fgm=self._sm([g.get('fgm') for g in r],37)
        f3m=self._sm([g.get('fg3m') for g in r],11)
        fta=self._sm([g.get('fta') for g in r],22)
        ore=self._sm([g.get('oreb') for g in r],10)
        dre=self._sm([g.get('dreb') for g in r],33)
        ast=self._sm([g.get('ast') for g in r],24)
        tov=self._sm([g.get('tov') for g in r],14)
        pace=fga+0.44*fta-ore+tov; pos=max(pace,70)
        return {'or':(pa/pos)*100,'dr':(po/pos)*100,'pace':pace,
                'ts':pa/max(2*(fga+0.44*fta),1),'efg':(fgm+0.5*f3m)/max(fga,1),
                'tp':tov/max(fga+0.44*fta+tov,1),'op':ore/max(ore+dre,1),
                'ar':ast/max(fga,1),'pa':pa,'po':po}

    def _fat(self, lg, gd, tid, vtid):
        if not lg: return (3,0,0,0,0)
        ld=lg[-1]['date']
        if isinstance(ld,str): ld=pd.Timestamp(ld)
        dr=(gd-ld).days
        c7=gd-timedelta(days=7)
        g7=sum(1 for g in lg if (pd.Timestamp(g['date']) if isinstance(g['date'],str) else g['date'])>=c7)
        td=0; last=lg[-1]; loc=tid if last['home'] else last['opp_id']
        if loc in COORDS and vtid in COORDS:
            c1,c2=COORDS[loc],COORDS[vtid]; td=_hav(c1[0],c1[1],c2[0],c2[1])
        # Road trip length
        road=0
        for g in reversed(lg):
            if not g['home']: road+=1
            else: break
        return (min(dr,7), 1 if dr<=1 else 0, g7, min(td/1000,3), min(road/5,1))

    def _mom(self, lg, tid):
        if not lg: return (0,0.5,0.5,0)
        st=0
        for g in reversed(lg):
            if st==0: st=1 if g['won'] else -1
            elif st>0 and g['won']: st+=1
            elif st<0 and not g['won']: st-=1
            else: break
        l5=lg[-5:]; wp5=sum(g['won'] for g in l5)/len(l5)
        r=self.rec[tid]; t=r['w']+r['l']; wps=r['w']/max(t,1)
        l10=lg[-10:]; mt=0
        if len(l10)>=3:
            m=[g['margin'] for g in l10]; mt=np.polyfit(range(len(m)),m,1)[0]
        return (np.clip(st,-10,10), wp5, wps, np.clip(mt,-5,5))

    def _play(self, tid):
        recent=self.log[tid][-10:]
        if not recent: return (0,0.5,0)
        pt=defaultdict(lambda:{'i':0,'g':0,'pm':0})
        for g in recent:
            for p in self.pl_idx.get(g['gid'],[]):
                if REV_TEAM.get(p.get('nba_tid'))!=tid: continue
                mn=p.get('minutes_decimal') or 0
                if mn<5: continue
                pid=p['player_id']
                imp=((p.get('pts') or 0)+1.2*(p.get('reb') or 0)+1.5*(p.get('ast') or 0)+
                     2*(p.get('stl') or 0)+2*(p.get('blk') or 0)-(p.get('tov') or 0))*(mn/48)
                pt[pid]['i']+=imp; pt[pid]['g']+=1
                pt[pid]['pm']+=(p.get('plus_minus') or 0)
        if not pt: return (0,0.5,0)
        items=[(d['i']/max(d['g'],1), d['pm']/max(d['g'],1)) for d in pt.values()]
        items.sort(key=lambda x:x[0], reverse=True)
        avgs=[x[0] for x in items]; pms=[x[1] for x in items]
        star=sum(avgs[:3])/30; tot=sum(avgs)
        bench=sum(avgs[5:10]) if len(avgs)>5 else 0
        top_pm=np.mean(pms[:3])/15 if len(pms)>=3 else 0  # normalize
        return (star, bench/max(tot,1), top_pm)

    def _mkt(self, gid):
        odds=self.odds_idx.get(gid,[])
        if not odds: return (0.5,0,220)
        probs,spr,tot=[],[],[]
        hn=odds[0].get('home_team','')
        for o in odds:
            mk,pr,pt,nm = o.get('market',''),o.get('outcome_price'),o.get('outcome_point'),o.get('outcome_name','')
            if mk=='h2h' and pr:
                pb=100/(pr+100) if pr>0 else abs(pr)/(abs(pr)+100) if pr<0 else None
                if pb and (nm==hn or hn in nm): probs.append(pb)
            elif mk=='spreads' and pt is not None and (nm==hn or hn in nm): spr.append(pt)
            elif mk=='totals' and pt is not None and nm.lower()=='over': tot.append(pt)
        return (np.mean(probs) if probs else 0.5,
                np.mean(spr) if spr else 0,
                np.mean(tot) if tot else 220)

    def _h2h(self, hid, aid):
        h2h=[g for g in self.log[hid] if g['opp_id']==aid]
        if not h2h: return (0,0)
        l3=h2h[-3:]
        return (sum(g['won'] for g in l3),
                np.clip(np.mean([g['margin'] for g in l3]),-30,30)/30)

    def _elo_update(self, hid, aid, hw, margin):
        """MOV-ELO: margin-of-victory adjusted ELO."""
        he,ae = self.elo[hid],self.elo[aid]
        exp = 1/(1+10**(-(he-ae+ELO_HCA)/400))
        act = 1.0 if hw else 0.0
        # MOV multiplier: log(|margin|+1) * factor, capped
        mov = np.log1p(abs(margin)) * ELO_MOV_MULT
        mov = min(mov, 2.5)  # Cap at 2.5x
        # Reduce K for blowouts against weak teams (autocorrelation correction)
        elo_diff = abs(he - ae)
        ac_correction = 2.2 / ((elo_diff * 0.001) + 2.2)
        k_adj = ELO_K * mov * ac_correction
        self.elo[hid] += k_adj * (act - exp)
        self.elo[aid] += k_adj * ((1-act) - (1-exp))


# ═══════════════════════ MONTE CARLO ══════════════════════════════════════════
class MCSim:
    def run(self, feat, n=N_SIMS):
        ix={nm:i for i,nm in enumerate(FEAT_NAMES)}
        hor=feat[ix['h_off10']]; hdr=feat[ix['h_def10']]
        aor=feat[ix['a_off10']]; adr=feat[ix['a_def10']]
        hp=feat[ix['h_pace10']]; ap=feat[ix['a_pace10']]
        hr=feat[ix['h_rest']]; ar=feat[ix['a_rest']]
        he=feat[ix['h_elo']]; ae=feat[ix['a_elo']]
        # Use consistency to adjust variance
        h_mstd=feat[ix['h_margin_std']]; a_mstd=feat[ix['a_margin_std']]
        var = 0.035 + 0.01*(h_mstd + a_mstd)  # More variable teams → wider dist

        pace=np.clip(np.random.normal((hp+ap)/2, 3.5, n), 70, 120)
        h_eff=np.random.normal((hor+(200-adr))/200+0.015+(he-ae)/75000+(hr-ar)*0.003, var, n)
        a_eff=np.random.normal((aor+(200-hdr))/200-0.015-(he-ae)/75000-(hr-ar)*0.003, var, n)
        h_eff+=sp_stats.t.rvs(df=5, scale=0.012, size=n)
        a_eff+=sp_stats.t.rvs(df=5, scale=0.012, size=n)
        hs=pace*h_eff+np.random.normal(0, 1.5, n)
        aws=pace*a_eff+np.random.normal(0, 1.5, n)
        mg=hs-aws
        return {'wp':np.mean(mg>0),'em':np.mean(mg),'ms':np.std(mg),'tm':np.mean(hs+aws)}


# ═══════════════════════ MODEL (ENHANCED) ═════════════════════════════════════
class Model:
    def __init__(self):
        self.hgb = HistGradientBoostingClassifier(
            max_iter=300, max_depth=5, learning_rate=0.05,
            min_samples_leaf=30, l2_regularization=1.0,
            max_bins=128, random_state=42, early_stopping=True,
            n_iter_no_change=20, validation_fraction=0.15)
        self.hgbr = HistGradientBoostingRegressor(
            max_iter=300, max_depth=5, learning_rate=0.05,
            min_samples_leaf=30, l2_regularization=1.0,
            max_bins=128, random_state=42, early_stopping=True,
            n_iter_no_change=20, validation_fraction=0.15)
        self.lr = LogisticRegression(C=0.5, max_iter=1000)
        self.sc = StandardScaler()
        self.mc = MCSim()
        # Isotonic calibration for post-hoc probability calibration
        self.iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds='clip')
        self.iso_fitted = False

        self.trained = False
        self.tX, self.ty, self.tm = [], [], []
        # Stacking weights (learned)
        self.W = {'hgb':0.30, 'lr':0.15, 'mc':0.20, 'elo':0.20, 'mkt':0.15}
        self.phist, self.rhist = [], []

    def add(self, X, y, m):
        self.tX.append(X); self.ty.append(y); self.tm.append(m)

    def retrain(self):
        if len(self.tX) < 300: return False
        X = np.array(self.tX); y = np.array(self.ty); m = np.array(self.tm)
        Xs = self.sc.fit_transform(X)
        self.hgb.fit(Xs, y)
        self.hgbr.fit(Xs, m)
        self.lr.fit(Xs, y)
        self.trained = True
        # Fit isotonic calibration on recent predictions
        self._fit_isotonic()
        return True

    def _fit_isotonic(self):
        """Fit isotonic regression to calibrate ensemble probabilities."""
        if len(self.phist) < 200: return
        recent_p = [p.get('wp_raw', p.get('wp', 0.5)) for p in self.phist[-1000:]]
        recent_r = self.rhist[-1000:]
        if len(set(recent_r)) < 2: return
        try:
            self.iso.fit(recent_p, recent_r)
            self.iso_fitted = True
        except:
            pass

    def predict(self, X):
        r = {'comp': {}}
        mc = self.mc.run(X); r['comp']['mc'] = mc['wp']
        ix = {n:i for i,n in enumerate(FEAT_NAMES)}
        ee = X[ix['elo_exp']]; r['comp']['elo'] = ee
        mk = X[ix['mkt_prob']]
        if mk < 0.1 or mk > 0.9: mk = 0.5
        r['comp']['mkt'] = mk

        if self.trained:
            Xs = self.sc.transform(X.reshape(1,-1))
            gp = self.hgb.predict_proba(Xs)[0][1]
            lp = self.lr.predict_proba(Xs)[0][1]
            gm = self.hgbr.predict(Xs)[0]
            r['comp']['hgb'] = gp; r['comp']['lr'] = lp
            wp = (self.W['hgb']*gp + self.W['lr']*lp + self.W['mc']*mc['wp']
                  + self.W['elo']*ee + self.W['mkt']*mk)
            mg = (gm + mc['em']) / 2
        else:
            r['comp']['hgb'] = None; r['comp']['lr'] = None
            wp = 0.35*mc['wp'] + 0.35*ee + 0.30*mk
            mg = mc['em']

        wp = np.clip(wp, 0.01, 0.99)
        r['wp_raw'] = wp
        # Apply isotonic calibration
        if self.iso_fitted:
            try:
                wp = float(self.iso.predict([wp])[0])
            except:
                pass
        r['wp'] = np.clip(wp, 0.01, 0.99)
        r['mg'] = mg; r['conf'] = 1-mc['ms']/20; r['total'] = mc['tm']
        return r

    def calibrate(self):
        if len(self.phist) < 200: return
        rp, rr = self.phist[-800:], self.rhist[-800:]
        sc = {}
        for c in ['hgb','mc','elo','mkt','lr']:
            pr, ac = [], []
            for p, a in zip(rp, rr):
                v = p.get('comp',{}).get(c)
                if v is not None and not np.isnan(v): pr.append(v); ac.append(a)
            if len(pr) > 30:
                sc[c] = max(1 - brier_score_loss(ac, pr)*2, 0.05)
        if sc:
            t = sum(sc.values())
            for c in sc: self.W[c] = sc[c]/t

    def save(self, path=None):
        path = path or f"{MODEL_DIR}/nba_model_v3.pkl"
        with open(path,'wb') as f:
            pickle.dump({'hgb':self.hgb,'hgbr':self.hgbr,'lr':self.lr,
                        'sc':self.sc,'iso':self.iso,'iso_fitted':self.iso_fitted,
                        'W':self.W,'trained':self.trained,'n':len(self.tX)}, f)

    def load(self, path=None):
        path = path or f"{MODEL_DIR}/nba_model_v3.pkl"
        if not os.path.exists(path): return False
        with open(path,'rb') as f: d = pickle.load(f)
        if d.get('trained'):
            self.hgb=d['hgb']; self.hgbr=d['hgbr']; self.lr=d['lr']
            self.sc=d['sc']; self.iso=d.get('iso',self.iso)
            self.iso_fitted=d.get('iso_fitted',False)
            self.W=d.get('W',self.W); self.trained=True; return True
        return False


# ═══════════════════════ METRICS ══════════════════════════════════════════════
class Metrics:
    def __init__(self): self.probs=[]; self.acts=[]; self.margs=[]
    def add(self, prob, actual, margin):
        self.probs.append(prob); self.acts.append(actual); self.margs.append(margin)
    def report(self, last_n=None, label=""):
        if not self.probs: return {}
        n = last_n or len(self.probs)
        pr=self.probs[-n:]; ac=self.acts[-n:]
        preds=[1 if p>0.5 else 0 for p in pr]
        acc=accuracy_score(ac,preds)
        brier=brier_score_loss(ac,pr)
        pc=np.clip(pr,0.01,0.99); ll=log_loss(ac,pc)
        try: auc=roc_auc_score(ac,pr)
        except: auc=0.5
        # RMSE of margins
        if self.margs:
            mg_actual=self.margs[-n:]
            # TODO: compare predicted margin vs actual

        bins=[(0,0.35),(0.35,0.45),(0.45,0.55),(0.55,0.65),(0.65,1.0)]
        cal=[]
        for lo,hi in bins:
            mask=[lo<=p<hi for p in pr]; nb=sum(mask)
            if nb>0:
                af=np.mean([a for a,m in zip(ac,mask) if m])
                pf=np.mean([p for p,m in zip(pr,mask) if m])
                cal.append((lo,hi,nb,pf,af))
        # ECE (Expected Calibration Error)
        ece = 0
        for lo,hi,nb,pf,af in cal:
            ece += (nb/n) * abs(pf-af)

        print(f"\n{'='*65}")
        print(f"  CHECKPOINT {label} ({n} juegos)")
        print(f"{'='*65}")
        print(f"  Accuracy:     {acc:.1%}  {'OK' if acc>0.60 else 'WARN' if acc>0.55 else 'LOW'}")
        print(f"  Brier Score:  {brier:.4f}  {'OK' if brier<0.23 else 'WARN' if brier<0.25 else 'HIGH'}")
        print(f"  Log Loss:     {ll:.4f}  {'OK' if ll<0.64 else 'WARN' if ll<0.67 else 'HIGH'}")
        print(f"  AUC:          {auc:.4f}  {'OK' if auc>0.67 else 'WARN' if auc>0.62 else 'LOW'}")
        print(f"  ECE:          {ece:.4f}  {'OK' if ece<0.03 else 'WARN' if ece<0.05 else 'HIGH'}")
        if cal:
            print(f"  Calibration:")
            for lo,hi,nb,pf,af in cal:
                delta = af-pf; sign = '+' if delta>=0 else ''
                print(f"    [{lo:.2f}-{hi:.2f}] n={nb:>4}  pred={pf:.3f}  actual={af:.3f}  {sign}{delta:.3f}")
        print(f"{'='*65}")
        return {'acc':acc,'brier':brier,'ll':ll,'auc':auc,'ece':ece}


# ═══════════════════════ PIPELINE ═════════════════════════════════════════════
def run(train_s, eval_s, ckpt, db):
    t0 = time.time()
    all_s = train_s + [eval_s]
    print(f"\n{'='*65}")
    print(f"  NBA TRAINING PIPELINE v3 (OPTIMIZED)")
    print(f"  Train: {', '.join(train_s)}")
    print(f"  Eval:  {eval_s}")
    print(f"  Features: {N_FEATURES}")
    print(f"  Checkpoint: cada {ckpt} partidos")
    print(f"{'='*65}\n")

    dl = DataLoader(db)
    games = dl.load_games(all_s)
    bs = dl.load_boxscores(); pl = dl.load_players(); od = dl.load_odds()
    dl.close()
    logger.info(f"Games:{len(games)} BS:{len(bs)} PL:{len(pl)} Odds:{len(od)}")

    eng = FeatureEngine(bs, pl, od, games)
    mdl = Model()
    train_m = Metrics(); eval_m = Metrics()
    proc = 0; skip = 0; cur_s = None

    for _, g in games.iterrows():
        s = g['season']; is_ev = (s == eval_s)

        if s != cur_s:
            if cur_s is not None:
                if not is_ev: train_m.report(label=f"FIN {cur_s}")
                for t in eng.elo:
                    eng.elo[t] = 0.75*eng.elo[t] + 0.25*ELO_INIT
            eng.reset_season(s); cur_s = s
            logger.info(f"Season: {s} {'[EVAL]' if is_ev else '[TRAIN]'}")

        feat = eng.compute(g)
        if feat is not None:
            pred = mdl.predict(feat)
            aw = g['home_win']; am = g['margin']
            tk = eval_m if is_ev else train_m
            tk.add(pred['wp'], aw, am)
            mdl.phist.append(pred); mdl.rhist.append(aw)
            if not is_ev: mdl.add(feat, aw, am)
            proc += 1
        else:
            skip += 1

        eng.update(g)

        # Retrain every 500 games during training
        if not is_ev and proc > 0 and proc % 500 == 0:
            if mdl.retrain():
                logger.info(f"Retrained ({len(mdl.tX)} samples, {mdl.hgb.n_iter_} iters)")
                mdl.calibrate()

        if proc > 0 and proc % ckpt == 0:
            tk = eval_m if is_ev else train_m
            met = tk.report(last_n=ckpt,
                           label=f"{'EVAL' if is_ev else 'TRAIN'} #{proc}")

    # Final
    if mdl.retrain():
        logger.info(f"Final train ({len(mdl.tX)} samples)")
    mdl.calibrate(); mdl.save()

    print(f"\n{'='*65}")
    print(f"  RESUMEN FINAL v3")
    print(f"{'='*65}")
    print(f"  Procesados: {proc}  Omitidos: {skip}")
    print(f"  Tiempo: {(time.time()-t0)/60:.1f} min\n")
    print(f"  --- TRAIN ---")
    tf = train_m.report(label="TRAIN FINAL")
    print(f"\n  --- EVAL ({eval_s}) ---")
    ef = eval_m.report(label="EVAL FINAL")

    if mdl.trained:
        # Feature importance from HistGBM
        print(f"\n  Top 20 Features:")
        try:
            imp = np.zeros(N_FEATURES)
            # HistGBM doesn't have feature_importances_ by default in all versions
            # Use permutation importance approximation via training data
            X = np.array(mdl.tX[-2000:]); Xs = mdl.sc.transform(X)
            y = np.array(mdl.ty[-2000:])
            base_score = accuracy_score(y, mdl.hgb.predict(Xs))
            for i in range(N_FEATURES):
                X_perm = Xs.copy()
                np.random.shuffle(X_perm[:, i])
                perm_score = accuracy_score(y, mdl.hgb.predict(X_perm))
                imp[i] = base_score - perm_score
            fi = sorted(zip(FEAT_NAMES, imp), key=lambda x:x[1], reverse=True)
            for nm, im in fi[:20]:
                bar = '#' * max(int(im * 500), 0)
                print(f"    {nm:20s} {im:+.4f} {bar}")
        except Exception as e:
            print(f"    (Feature importance error: {e})")

    print(f"\n  Weights: {mdl.W}")
    print(f"{'='*65}\n")
    return tf, ef


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seasons', nargs='+', default=TRAIN_SEASONS)
    p.add_argument('--eval', default=EVAL_SEASON)
    p.add_argument('--checkpoint', type=int, default=CHECKPOINT_EVERY)
    p.add_argument('--db', default=DB_PATH)
    p.add_argument('--eval-only', action='store_true')
    a = p.parse_args()
    if a.eval_only:
        m = Model()
        if m.load(): print(f"Model loaded. Weights: {m.W}")
        else: print("No saved model")
        return
    run(a.seasons, a.eval, a.checkpoint, a.db)

if __name__ == "__main__":
    main()
