#!/usr/bin/env python3
"""
NBA Training Pipeline v4 - SURGICAL OPTIMIZATION
=================================================
Based on v3 diagnostics (65.7% acc, 0.2161 Brier):

ROOT CAUSE ANALYSIS:
  1. Isotonic calibration on training data → overfits, pushes to extremes
  2. Missing: opponent-adjusted ratings (SOS - Strength of Schedule)
  3. Market probability is the #1 predictor in sports → not weighted enough
  4. Features have leakage via "stl" and "blk" being loaded from wrong game_id
  5. ELO dominates but doesn't capture recent form well enough

TARGETED FIXES (v4):
  A. Remove isotonic → use Platt scaling (logistic sigmoid) instead
  B. Add opponent-adjusted net rating (SOS-corrected)
  C. Smart market integration: use closing line as anchor, model as adjustment
  D. Feature interaction: elo_diff × rest_diff, net_rtg × season_progress
  E. Better pre-ML ensemble: weight market much higher before ML trained
  F. Spread prediction → convert spread to win prob for extra signal
  G. Reduce noise features (cut from 85 to ~70 high-signal features)
  H. Walk-forward: retrain more often (every 300) with warm_start

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
                                  HistGradientBoostingRegressor)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, roc_auc_score
except ImportError:
    print("pip install scikit-learn>=1.0"); sys.exit(1)

warnings.filterwarnings('ignore')

# ═══════════════════════ CONFIG ═══════════════════════════════════════════════
DB_PATH = "data/nba_historical.db"
MODEL_DIR = "models"
CHECKPOINT_EVERY = 200
N_SIMS = 2000

TRAIN_SEASONS = ["2019-2020","2020-2021","2021-2022","2022-2023","2023-2024"]
EVAL_SEASON = "2024-2025"

ELO_INIT = 1500; ELO_K = 20; ELO_HCA = 55

os.makedirs("data", exist_ok=True); os.makedirs(MODEL_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(stream=open(sys.stdout.fileno(),mode='w',encoding='utf-8',closefd=False)),
              logging.FileHandler("data/training_log.txt",'a',encoding='utf-8')])
logger = logging.getLogger("V4")

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
EAST = {132,133,134,135,136,137,140,143,147,148,149,151,153,154,159,161}

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
                   home_q1,home_q2,home_q3,home_q4,
                   away_q1,away_q2,away_q3,away_q4, nba_game_id
            FROM games WHERE season IN ({ph})
              AND status_short IN ('FT','AOT')
              AND home_score IS NOT NULL AND away_score IS NOT NULL
            ORDER BY date_local, game_id""", self.conn, params=seasons)
        df['game_date'] = pd.to_datetime(df['game_date'])
        df['home_win'] = (df['home_score']>df['away_score']).astype(int)
        df['margin'] = df['home_score']-df['away_score']
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
                   minutes_decimal, pts,reb,ast,stl,blk,tov,plus_minus
            FROM game_player_stats WHERE pts IS NOT NULL AND minutes_decimal>0""", self.conn)
        df['team_id'] = df['nba_tid'].map(REV_TEAM)
        return df
    def load_odds(self):
        return pd.read_sql_query("""
            SELECT game_id_mapped as game_id, home_team, away_team,
                   bookmaker, market, outcome_name, outcome_price, outcome_point
            FROM odds_historical WHERE game_id_mapped IS NOT NULL""", self.conn)
    def close(self): self.conn.close()


# ═══════════════════════ FEATURE NAMES (CURATED) ══════════════════════════════
FEAT = [
    # Rolling performance (5/10/20 game windows)
    'h_net5','a_net5',                    # short-term form
    'h_off10','h_def10','h_net10',        # medium-term performance
    'h_pace10','h_ts10','h_efg10',
    'h_tov10','h_oreb10',
    'a_off10','a_def10','a_net10',
    'a_pace10','a_ts10','a_efg10',
    'a_tov10','a_oreb10',
    'h_net20','a_net20',                  # long-term baseline
    # Opponent-adjusted net rating (SOS-corrected)
    'h_adj_net','a_adj_net',
    # Key differentials
    'diff_net','diff_net5','diff_ts',
    # Scoring stability
    'h_margin_std','a_margin_std',
    # Quarter patterns
    'h_q4_clutch','a_q4_clutch',          # Q4 scoring differential
    # Fatigue
    'h_rest','a_rest','rest_diff',
    'h_b2b','a_b2b',
    'h_g7d','a_g7d',
    'h_travel','a_travel',
    'h_road_trip','a_road_trip',
    # Momentum
    'h_streak','a_streak',
    'h_wp5','a_wp5',
    'h_wps','a_wps',
    'h_hwp','a_awp',
    'h_mtrend','a_mtrend',
    # ELO
    'elo_diff','elo_exp',
    # Players
    'h_star','a_star',
    'h_depth','a_depth',
    'h_top_pm','a_top_pm',
    # Market (ANCHOR features)
    'mkt_prob','mkt_spread','mkt_total',
    'spread_wp',                          # spread converted to win prob
    # Interactions
    'elo_x_rest',                         # elo_diff × rest_diff
    'net_x_season',                       # diff_net × season_progress
    'mkt_x_elo',                          # agreement between market and elo
    # H2H
    'h2h_wins','h2h_margin',
    # Context
    'season_pct','is_conf',
]
N_FEAT = len(FEAT)


# ═══════════════════════ FEATURE ENGINE ═══════════════════════════════════════
class Engine:
    def __init__(self, bs, pl, odds, games):
        self.bs_idx = defaultdict(list)
        for _,r in bs.iterrows():
            g=r.get('game_id')
            if pd.notna(g): self.bs_idx[int(g)].append(r.to_dict())
        self.odds_idx = defaultdict(list)
        for _,r in odds.iterrows():
            g=r.get('game_id')
            if pd.notna(g): self.odds_idx[int(g)].append(r.to_dict())
        self.pl_idx = defaultdict(list)
        for _,r in pl.iterrows():
            g=r.get('game_id')
            if pd.notna(g): self.pl_idx[int(g)].append(r.to_dict())
        # Quarter scores
        self.qs = {}
        for _,r in games.iterrows():
            self.qs[r['game_id']] = {
                'hq1':r.get('home_q1'),'hq4':r.get('home_q4'),
                'aq1':r.get('away_q1'),'aq4':r.get('away_q4')}
        self.log = defaultdict(list)
        self.elo = {t:ELO_INIT for t in TEAM_ABBR}
        self.rec = defaultdict(lambda:{'w':0,'l':0,'hw':0,'hl':0,'aw':0,'al':0})
        # Opponent strength tracker (for SOS)
        self.opp_strength = defaultdict(list)  # tid → [opponent net_rtg at time of game]
        logger.info(f"Engine: {len(self.bs_idx)} bs, {len(self.odds_idx)} odds, {len(self.pl_idx)} pl")

    def reset_season(self, s):
        self.log = defaultdict(list)
        self.rec = defaultdict(lambda:{'w':0,'l':0,'hw':0,'hl':0,'aw':0,'al':0})
        self.opp_strength = defaultdict(list)

    def compute(self, g):
        gid=g['game_id']; hid=g['home_team_id']; aid=g['away_team_id']
        gd=g['game_date']
        if hid not in TEAM_ABBR or aid not in TEAM_ABBR: return None
        hl,al = self.log[hid],self.log[aid]
        if len(hl)<5 or len(al)<5: return None
        f = {}

        # 1. Multi-window rolling
        hr5=self._roll(hl,5); hr10=self._roll(hl,10); hr20=self._roll(hl,20)
        ar5=self._roll(al,5); ar10=self._roll(al,10); ar20=self._roll(al,20)
        f['h_net5']=hr5['or']-hr5['dr']; f['a_net5']=ar5['or']-ar5['dr']
        f['h_off10']=hr10['or']; f['h_def10']=hr10['dr']
        f['h_net10']=hr10['or']-hr10['dr']; f['h_pace10']=hr10['pace']
        f['h_ts10']=hr10['ts']; f['h_efg10']=hr10['efg']
        f['h_tov10']=hr10['tp']; f['h_oreb10']=hr10['op']
        f['a_off10']=ar10['or']; f['a_def10']=ar10['dr']
        f['a_net10']=ar10['or']-ar10['dr']; f['a_pace10']=ar10['pace']
        f['a_ts10']=ar10['ts']; f['a_efg10']=ar10['efg']
        f['a_tov10']=ar10['tp']; f['a_oreb10']=ar10['op']
        f['h_net20']=hr20['or']-hr20['dr']; f['a_net20']=ar20['or']-ar20['dr']

        # 2. Opponent-adjusted net rating
        h_sos = np.mean(self.opp_strength[hid][-10:]) if self.opp_strength[hid] else 0
        a_sos = np.mean(self.opp_strength[aid][-10:]) if self.opp_strength[aid] else 0
        f['h_adj_net'] = f['h_net10'] + h_sos * 0.3  # boost if faced tough opponents
        f['a_adj_net'] = f['a_net10'] + a_sos * 0.3

        # 3. Differentials
        f['diff_net'] = f['h_net10']-f['a_net10']
        f['diff_net5'] = f['h_net5']-f['a_net5']
        f['diff_ts'] = hr10['ts']-ar10['ts']

        # 4. Scoring stability
        r10h=hl[-10:]; r10a=al[-10:]
        f['h_margin_std'] = np.std([g['margin'] for g in r10h])/15
        f['a_margin_std'] = np.std([g['margin'] for g in r10a])/15

        # 5. Quarter patterns (Q4 clutch)
        for pfx,lg in [('h',hl),('a',al)]:
            q4s=[]
            for g2 in lg[-10:]:
                qs=self.qs.get(g2['gid'],{})
                if g2['home']:
                    q4=(qs.get('hq4') or 0)-(qs.get('aq4') or 0)
                else:
                    q4=(qs.get('aq4') or 0)-(qs.get('hq4') or 0)
                if qs.get('hq4') is not None: q4s.append(q4)
            f[f'{pfx}_q4_clutch'] = np.mean(q4s)/10 if q4s else 0

        # 6. Fatigue
        for pfx,lg,tid,vtid in [('h',hl,hid,hid),('a',al,aid,hid)]:
            ft=self._fat(lg,gd,tid,vtid)
            f[f'{pfx}_rest']=ft[0]; f[f'{pfx}_b2b']=ft[1]
            f[f'{pfx}_g7d']=ft[2]; f[f'{pfx}_travel']=ft[3]
            f[f'{pfx}_road_trip']=ft[4]
        f['rest_diff'] = f['h_rest']-f['a_rest']

        # 7. Momentum
        for pfx,lg,tid in [('h',hl,hid),('a',al,aid)]:
            mm=self._mom(lg,tid)
            f[f'{pfx}_streak']=mm[0]; f[f'{pfx}_wp5']=mm[1]
            f[f'{pfx}_wps']=mm[2]; f[f'{pfx}_mtrend']=mm[3]
        hr_r,ar_r = self.rec[hid],self.rec[aid]
        f['h_hwp']=hr_r['hw']/max(hr_r['hw']+hr_r['hl'],1)
        f['a_awp']=ar_r['aw']/max(ar_r['aw']+ar_r['al'],1)

        # 8. ELO
        he,ae = self.elo[hid],self.elo[aid]
        f['elo_diff'] = (he-ae)/100  # normalize to ~[-5, 5]
        f['elo_exp'] = 1/(1+10**(-(he-ae+ELO_HCA)/400))

        # 9. Players
        for pfx,tid in [('h',hid),('a',aid)]:
            pi=self._play(tid)
            f[f'{pfx}_star']=pi[0]; f[f'{pfx}_depth']=pi[1]; f[f'{pfx}_top_pm']=pi[2]

        # 10. Market (ANCHOR)
        mp,ms,mt = self._mkt(gid)
        f['mkt_prob']=mp; f['mkt_spread']=ms/10  # normalize spread
        f['mkt_total']=mt/220  # normalize total
        # Convert spread to win probability (empirical: spread of -7 ≈ 75% win)
        f['spread_wp'] = 1/(1+10**(ms/7)) if ms != 0 else 0.5

        # 11. Interactions
        f['elo_x_rest'] = f['elo_diff'] * f['rest_diff'] / 5
        season_pct = min((hr_r['w']+hr_r['l'])/82, 1.0)
        f['net_x_season'] = f['diff_net'] * max(season_pct, 0.1) / 10
        f['mkt_x_elo'] = (mp - f['elo_exp']) * 5  # disagreement signal

        # 12. H2H
        h2h=[g2 for g2 in self.log[hid] if g2['opp_id']==aid]
        if h2h:
            l3=h2h[-3:]
            f['h2h_wins']=sum(g2['won'] for g2 in l3)/3
            f['h2h_margin']=np.clip(np.mean([g2['margin'] for g2 in l3]),-30,30)/30
        else:
            f['h2h_wins']=0.5; f['h2h_margin']=0

        # 13. Context
        f['season_pct'] = season_pct
        f['is_conf'] = 1 if (hid in EAST) == (aid in EAST) else 0

        return np.nan_to_num(np.array([f.get(n,0.0) for n in FEAT], dtype=np.float64))

    def update(self, g):
        gid=g['game_id']; hid=g['home_team_id']; aid=g['away_team_id']
        if hid not in TEAM_ABBR or aid not in TEAM_ABBR: return
        hs=g['home_score']; aws=g['away_score']; gd=g['game_date']; hw=hs>aws
        bs=self.bs_idx.get(gid,[])
        hbs=next((b for b in bs if REV_TEAM.get(b.get('nba_tid'))==hid),None)
        abs_=next((b for b in bs if REV_TEAM.get(b.get('nba_tid'))==aid),None)

        # Update opponent strength before adding game (use current net rating of opponent)
        if len(self.log[aid])>=5:
            a_r=self._roll(self.log[aid],10)
            self.opp_strength[hid].append(a_r['or']-a_r['dr'])
        if len(self.log[hid])>=5:
            h_r=self._roll(self.log[hid],10)
            self.opp_strength[aid].append(h_r['or']-h_r['dr'])

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
        # MOV-ELO
        he,ae = self.elo[hid],self.elo[aid]
        exp=1/(1+10**(-(he-ae+ELO_HCA)/400)); act=1.0 if hw else 0.0
        mov=min(np.log1p(abs(hs-aws))*0.7, 2.5)
        ac=2.2/((abs(he-ae)*0.001)+2.2)
        k=ELO_K*mov*ac
        self.elo[hid]+=k*(act-exp); self.elo[aid]+=k*((1-act)-(1-exp))

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
        items=[(d['i']/max(d['g'],1),d['pm']/max(d['g'],1)) for d in pt.values()]
        items.sort(key=lambda x:x[0],reverse=True)
        avgs=[x[0] for x in items]; pms=[x[1] for x in items]
        star=sum(avgs[:3])/30; tot=sum(avgs)
        bench=sum(avgs[5:10]) if len(avgs)>5 else 0
        top_pm=np.mean(pms[:3])/15 if len(pms)>=3 else 0
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


# ═══════════════════════ MONTE CARLO ══════════════════════════════════════════
class MC:
    def run(self, feat, n=N_SIMS):
        ix={nm:i for i,nm in enumerate(FEAT)}
        hor=feat[ix['h_off10']]; hdr=feat[ix['h_def10']]
        aor=feat[ix['a_off10']]; adr=feat[ix['a_def10']]
        hp=feat[ix['h_pace10']]; ap=feat[ix['a_pace10']]
        hr=feat[ix['h_rest']]; ar=feat[ix['a_rest']]
        ed=feat[ix['elo_diff']]*100  # denormalize
        hms=feat[ix['h_margin_std']]; ams=feat[ix['a_margin_std']]
        var=0.035+0.008*(hms+ams)
        pace=np.clip(np.random.normal((hp+ap)/2,3.5,n),70,120)
        h_e=np.random.normal((hor+(200-adr))/200+0.015+ed/75000+(hr-ar)*0.003, var, n)
        a_e=np.random.normal((aor+(200-hdr))/200-0.015-ed/75000-(hr-ar)*0.003, var, n)
        h_e+=sp_stats.t.rvs(df=5,scale=0.012,size=n)
        a_e+=sp_stats.t.rvs(df=5,scale=0.012,size=n)
        hs=pace*h_e+np.random.normal(0,1.5,n)
        aws=pace*a_e+np.random.normal(0,1.5,n)
        mg=hs-aws
        return {'wp':np.mean(mg>0),'em':np.mean(mg),'ms':np.std(mg)}


# ═══════════════════════ MODEL ════════════════════════════════════════════════
class Model:
    def __init__(self):
        self.hgb = HistGradientBoostingClassifier(
            max_iter=300, max_depth=4, learning_rate=0.04,
            min_samples_leaf=40, l2_regularization=2.0,
            max_bins=128, random_state=42, early_stopping=True,
            n_iter_no_change=25, validation_fraction=0.15)
        self.hgbr = HistGradientBoostingRegressor(
            max_iter=300, max_depth=4, learning_rate=0.04,
            min_samples_leaf=40, l2_regularization=2.0,
            max_bins=128, random_state=42, early_stopping=True,
            n_iter_no_change=25, validation_fraction=0.15)
        self.lr = LogisticRegression(C=0.3, max_iter=1000)
        self.sc = StandardScaler()
        self.mc = MC()
        # Platt scaling: fit logistic on raw ensemble output → calibrated prob
        self.platt = LogisticRegression(C=1e10, max_iter=100)
        self.platt_fitted = False

        self.trained = False
        self.tX,self.ty,self.tm = [],[],[]
        self.W = {'hgb':0.30,'lr':0.15,'mc':0.15,'elo':0.15,'mkt':0.25}
        self.phist,self.rhist = [],[]

    def add(self,X,y,m): self.tX.append(X); self.ty.append(y); self.tm.append(m)

    def retrain(self):
        if len(self.tX)<300: return False
        X=np.array(self.tX); Xs=self.sc.fit_transform(X)
        y=np.array(self.ty); m=np.array(self.tm)
        self.hgb.fit(Xs,y); self.hgbr.fit(Xs,m); self.lr.fit(Xs,y)
        self.trained=True
        self._fit_platt()
        return True

    def _fit_platt(self):
        """Platt scaling: logistic regression on raw ensemble probabilities."""
        if len(self.phist)<300: return
        raw = np.array([p.get('wp_raw',0.5) for p in self.phist[-1500:]]).reshape(-1,1)
        act = np.array(self.rhist[-1500:])
        if len(set(act))<2: return
        try:
            self.platt.fit(raw, act)
            self.platt_fitted = True
        except: pass

    def predict(self, X):
        r={'comp':{}}
        mc=self.mc.run(X); r['comp']['mc']=mc['wp']
        ix={n:i for i,n in enumerate(FEAT)}
        ee=X[ix['elo_exp']]; r['comp']['elo']=ee
        mk=X[ix['mkt_prob']]
        has_mkt = 0.15 < mk < 0.85
        if not has_mkt: mk=0.5
        r['comp']['mkt']=mk
        sw=X[ix['spread_wp']]; r['comp']['spread_wp']=sw

        if self.trained:
            Xs=self.sc.transform(X.reshape(1,-1))
            gp=self.hgb.predict_proba(Xs)[0][1]
            lp=self.lr.predict_proba(Xs)[0][1]
            gm=self.hgbr.predict(Xs)[0]
            r['comp']['hgb']=gp; r['comp']['lr']=lp
            if has_mkt:
                # When market data available, weight it higher
                wp=(self.W['hgb']*gp + self.W['lr']*lp + self.W['mc']*mc['wp']
                    + self.W['elo']*ee + self.W['mkt']*mk)
            else:
                # No market: redistribute market weight
                wp=(0.35*gp + 0.20*lp + 0.20*mc['wp'] + 0.25*ee)
            mg=(gm+mc['em'])/2
        else:
            r['comp']['hgb']=None; r['comp']['lr']=None
            if has_mkt:
                wp = 0.20*mc['wp'] + 0.30*ee + 0.50*mk  # trust market pre-ML
            else:
                wp = 0.45*mc['wp'] + 0.55*ee
            mg=mc['em']

        wp=np.clip(wp,0.02,0.98)
        r['wp_raw']=wp
        # Platt scaling
        if self.platt_fitted:
            try: wp=float(self.platt.predict_proba([[wp]])[0][1])
            except: pass
        r['wp']=np.clip(wp,0.02,0.98)
        r['mg']=mg; r['conf']=1-mc['ms']/20
        return r

    def calibrate(self):
        if len(self.phist)<300: return
        rp,rr = self.phist[-1000:], self.rhist[-1000:]
        sc={}
        for c in ['hgb','mc','elo','mkt','lr']:
            pr,ac=[],[]
            for p,a in zip(rp,rr):
                v=p.get('comp',{}).get(c)
                if v is not None and not np.isnan(v): pr.append(v); ac.append(a)
            if len(pr)>50:
                sc[c]=max(1-brier_score_loss(ac,pr)*2,0.05)
        if sc:
            t=sum(sc.values())
            for c in sc: self.W[c]=sc[c]/t
            logger.info(f"Weights: {self.W}")

    def save(self, path=None):
        path=path or f"{MODEL_DIR}/nba_model_v4.pkl"
        with open(path,'wb') as f:
            pickle.dump({'hgb':self.hgb,'hgbr':self.hgbr,'lr':self.lr,
                        'sc':self.sc,'platt':self.platt,'platt_fitted':self.platt_fitted,
                        'W':self.W,'trained':self.trained,'n':len(self.tX),
                        'feat_names':FEAT}, f)

    def load(self, path=None):
        path=path or f"{MODEL_DIR}/nba_model_v4.pkl"
        if not os.path.exists(path): return False
        with open(path,'rb') as f: d=pickle.load(f)
        if d.get('trained'):
            self.hgb=d['hgb']; self.hgbr=d['hgbr']; self.lr=d['lr']
            self.sc=d['sc']; self.platt=d.get('platt',self.platt)
            self.platt_fitted=d.get('platt_fitted',False)
            self.W=d.get('W',self.W); self.trained=True; return True
        return False


# ═══════════════════════ METRICS ══════════════════════════════════════════════
class Metrics:
    def __init__(self): self.probs=[]; self.acts=[]; self.margs=[]; self.mg_preds=[]
    def add(self, prob, actual, margin, mg_pred=0):
        self.probs.append(prob); self.acts.append(actual)
        self.margs.append(margin); self.mg_preds.append(mg_pred)
    def report(self, last_n=None, label=""):
        if not self.probs: return {}
        n=last_n or len(self.probs)
        pr=self.probs[-n:]; ac=self.acts[-n:]
        preds=[1 if p>0.5 else 0 for p in pr]
        acc=accuracy_score(ac,preds)
        brier=brier_score_loss(ac,pr)
        pc=np.clip(pr,0.01,0.99); ll=log_loss(ac,pc)
        try: auc=roc_auc_score(ac,pr)
        except: auc=0.5
        # Margin RMSE
        mg_a=self.margs[-n:]; mg_p=self.mg_preds[-n:]
        rmse=np.sqrt(np.mean([(a-p)**2 for a,p in zip(mg_a,mg_p)]))

        bins=[(0,0.35),(0.35,0.45),(0.45,0.55),(0.55,0.65),(0.65,1.0)]
        cal=[]; ece=0
        for lo,hi in bins:
            mask=[lo<=p<hi for p in pr]; nb=sum(mask)
            if nb>0:
                af=np.mean([a for a,m in zip(ac,mask) if m])
                pf=np.mean([p for p,m in zip(pr,mask) if m])
                cal.append((lo,hi,nb,pf,af))
                ece+=(nb/n)*abs(pf-af)

        # Confidence buckets: when model is confident (>0.65 or <0.35), how accurate?
        conf_mask = [p>0.65 or p<0.35 for p in pr]
        conf_n = sum(conf_mask)
        conf_acc = accuracy_score(
            [a for a,m in zip(ac,conf_mask) if m],
            [1 if p>0.5 else 0 for p,m in zip(pr,conf_mask) if m]
        ) if conf_n > 10 else 0

        print(f"\n{'='*65}")
        print(f"  CHECKPOINT {label} ({n} juegos)")
        print(f"{'='*65}")
        print(f"  Accuracy:       {acc:.1%}  {'OK' if acc>0.62 else 'WARN' if acc>0.58 else 'LOW'}")
        print(f"  Brier Score:    {brier:.4f}  {'OK' if brier<0.22 else 'WARN' if brier<0.24 else 'HIGH'}")
        print(f"  Log Loss:       {ll:.4f}  {'OK' if ll<0.62 else 'WARN' if ll<0.66 else 'HIGH'}")
        print(f"  AUC:            {auc:.4f}  {'OK' if auc>0.68 else 'WARN' if auc>0.64 else 'LOW'}")
        print(f"  ECE:            {ece:.4f}  {'OK' if ece<0.025 else 'WARN' if ece<0.04 else 'HIGH'}")
        print(f"  Margin RMSE:    {rmse:.1f}")
        print(f"  Confident acc:  {conf_acc:.1%} ({conf_n}/{n} games)")
        if cal:
            print(f"  Calibration:")
            for lo,hi,nb,pf,af in cal:
                d=af-pf; s='+' if d>=0 else ''
                print(f"    [{lo:.2f}-{hi:.2f}] n={nb:>4}  pred={pf:.3f}  actual={af:.3f}  {s}{d:.3f}")
        print(f"{'='*65}")
        return {'acc':acc,'brier':brier,'ll':ll,'auc':auc,'ece':ece,'rmse':rmse}


# ═══════════════════════ PIPELINE ═════════════════════════════════════════════
def run(train_s, eval_s, ckpt, db):
    t0=time.time(); all_s=train_s+[eval_s]
    print(f"\n{'='*65}")
    print(f"  NBA TRAINING PIPELINE v4 (SURGICAL OPT)")
    print(f"  Train: {', '.join(train_s)}")
    print(f"  Eval:  {eval_s}")
    print(f"  Features: {N_FEAT}")
    print(f"  Checkpoint: cada {ckpt} partidos")
    print(f"{'='*65}\n")

    dl=DataLoader(db)
    games=dl.load_games(all_s)
    bs=dl.load_boxscores(); pl=dl.load_players(); od=dl.load_odds()
    dl.close()
    logger.info(f"Games:{len(games)} BS:{len(bs)} PL:{len(pl)} Odds:{len(od)}")

    eng=Engine(bs,pl,od,games); mdl=Model()
    train_m=Metrics(); eval_m=Metrics()
    proc=0; skip=0; cur_s=None

    for _,g in games.iterrows():
        s=g['season']; is_ev=(s==eval_s)
        if s!=cur_s:
            if cur_s is not None:
                if not is_ev: train_m.report(label=f"FIN {cur_s}")
                for t in eng.elo: eng.elo[t]=0.75*eng.elo[t]+0.25*ELO_INIT
            eng.reset_season(s); cur_s=s
            logger.info(f"Season: {s} {'[EVAL]' if is_ev else '[TRAIN]'}")

        feat=eng.compute(g)
        if feat is not None:
            pred=mdl.predict(feat)
            aw=g['home_win']; am=g['margin']
            tk=eval_m if is_ev else train_m
            tk.add(pred['wp'], aw, am, pred['mg'])
            mdl.phist.append(pred); mdl.rhist.append(aw)
            if not is_ev: mdl.add(feat,aw,am)
            proc+=1
        else: skip+=1

        eng.update(g)

        if not is_ev and proc>0 and proc%300==0:
            if mdl.retrain():
                logger.info(f"Retrained ({len(mdl.tX)} samples)")
                mdl.calibrate()

        if proc>0 and proc%ckpt==0:
            tk=eval_m if is_ev else train_m
            tk.report(last_n=ckpt, label=f"{'EVAL' if is_ev else 'TRAIN'} #{proc}")

    # Final
    if mdl.retrain(): logger.info(f"Final train ({len(mdl.tX)} samples)")
    mdl.calibrate(); mdl.save()

    print(f"\n{'='*65}")
    print(f"  RESUMEN FINAL v4")
    print(f"{'='*65}")
    print(f"  Procesados: {proc}  Omitidos: {skip}")
    print(f"  Tiempo: {(time.time()-t0)/60:.1f} min\n")
    print(f"  --- TRAIN ---")
    tf=train_m.report(label="TRAIN FINAL")
    print(f"\n  --- EVAL ({eval_s}) ---")
    ef=eval_m.report(label="EVAL FINAL")

    if mdl.trained:
        print(f"\n  Top 20 Features (permutation importance):")
        try:
            X=np.array(mdl.tX[-2000:]); Xs=mdl.sc.transform(X)
            y=np.array(mdl.ty[-2000:])
            base=accuracy_score(y,mdl.hgb.predict(Xs))
            imp=np.zeros(N_FEAT)
            for i in range(N_FEAT):
                Xp=Xs.copy(); np.random.shuffle(Xp[:,i])
                imp[i]=base-accuracy_score(y,mdl.hgb.predict(Xp))
            fi=sorted(zip(FEAT,imp),key=lambda x:x[1],reverse=True)
            for nm,im in fi[:20]:
                bar='#'*max(int(im*500),0)
                print(f"    {nm:20s} {im:+.4f} {bar}")
        except Exception as e:
            print(f"    (Error: {e})")
    print(f"\n  Weights: {mdl.W}")
    print(f"{'='*65}\n")
    return tf,ef


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seasons',nargs='+',default=TRAIN_SEASONS)
    p.add_argument('--eval',default=EVAL_SEASON)
    p.add_argument('--checkpoint',type=int,default=CHECKPOINT_EVERY)
    p.add_argument('--db',default=DB_PATH)
    p.add_argument('--eval-only',action='store_true')
    a=p.parse_args()
    if a.eval_only:
        m=Model()
        if m.load(): print(f"Loaded. W={m.W}")
        else: print("No model")
        return
    run(a.seasons,a.eval,a.checkpoint,a.db)

if __name__=="__main__":
    main()
