#!/usr/bin/env python3
"""
NBA Training Pipeline v2
========================
Lee de nba_historical.db. Para cada partido calcula features SOLO con
datos previos (anti-leakage). Entrena ensemble (GBM + LR + MC + ELO + Market).
Checkpoints cada N partidos. Train/Eval split por temporada.

Features (55):
  Performance rolling 10 (home/away): off_rtg, def_rtg, net_rtg, pace,
    ts_pct, efg_pct, tov_pct, oreb_pct, ast_ratio, pts_avg, pts_allowed
  Differentials: off_rtg, def_rtg, net_rtg, ts_pct, pace
  Fatigue: days_rest, is_b2b, games_7d, travel_dist
  Momentum: streak, win_pct_5, win_pct_season, home/away_win_pct, margin_trend
  ELO: h_elo, a_elo, diff, expected
  Players: star_power, depth_score
  Market: implied_prob, spread, total_line
  H2H: wins_last3, avg_margin
  Context: season_progress

Uso:
  python nba_training_pipeline.py                 # Full train + eval
  python nba_training_pipeline.py --checkpoint 50
  python nba_training_pipeline.py --eval-only

Requisitos: numpy pandas scikit-learn scipy
"""
import os, sys, time, sqlite3, logging, argparse, warnings, pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

try:
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, roc_auc_score
except ImportError:
    print("pip install scikit-learn"); sys.exit(1)

warnings.filterwarnings('ignore')

# ═══════════════════════ CONFIG ═══════════════════════════════════════════════
DB_PATH = "data/nba_historical.db"
MODEL_DIR = "models"
CHECKPOINT_EVERY = 100
N_SIMS = 5000
TRAIN_SEASONS = ["2019-2020","2020-2021","2021-2022","2022-2023","2023-2024"]
EVAL_SEASON = "2024-2025"
ELO_INIT = 1500; ELO_K = 20; ELO_HCA = 60

os.makedirs("data", exist_ok=True); os.makedirs(MODEL_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(stream=open(sys.stdout.fileno(), mode='w', encoding='utf-8', closefd=False)),
              logging.FileHandler("data/training_log.txt", 'a', encoding='utf-8')])
logger = logging.getLogger("Pipeline")

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

def _hav(lat1,lon1,lat2,lon2):
    from math import radians,cos,sin,asin,sqrt
    lat1,lon1,lat2,lon2 = map(radians,[lat1,lon1,lat2,lon2])
    d = lat2-lat1; dl = lon2-lon1
    a = sin(d/2)**2 + cos(lat1)*cos(lat2)*sin(dl/2)**2
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
                   pts,reb,ast,stl,blk,tov, plus_minus
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
    'h_off_rtg','h_def_rtg','h_net_rtg','h_pace','h_ts','h_efg',
    'h_tov_pct','h_oreb_pct','h_ast_r','h_pts_avg','h_pts_opp',
    'a_off_rtg','a_def_rtg','a_net_rtg','a_pace','a_ts','a_efg',
    'a_tov_pct','a_oreb_pct','a_ast_r','a_pts_avg','a_pts_opp',
    'diff_off','diff_def','diff_net','diff_ts','diff_pace',
    'h_rest','a_rest','h_b2b','a_b2b','h_g7d','a_g7d','h_travel','a_travel',
    'h_streak','a_streak','h_wp5','a_wp5','h_wps','a_wps',
    'h_hwp','a_awp','h_mtrend','a_mtrend',
    'h_elo','a_elo','elo_diff','elo_exp',
    'h_star','a_star','h_depth','a_depth',
    'mkt_prob','mkt_spread','mkt_total',
    'h2h_wins','h2h_margin','season_pct',
]

# ═══════════════════════ FEATURE ENGINE ═══════════════════════════════════════
class FeatureEngine:
    def __init__(self, boxscores, players, odds):
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
        self.log = defaultdict(list)
        self.elo = {t: ELO_INIT for t in TEAM_ABBR}
        self.rec = defaultdict(lambda: {'w':0,'l':0,'hw':0,'hl':0,'aw':0,'al':0})
        self.season = None
        logger.info(f"FeatureEngine: {len(self.bs_idx)} bs, {len(self.odds_idx)} odds, {len(self.pl_idx)} pl")

    def reset_season(self, s):
        self.log = defaultdict(list)
        self.rec = defaultdict(lambda: {'w':0,'l':0,'hw':0,'hl':0,'aw':0,'al':0})
        self.season = s

    def compute(self, g):
        gid,hid,aid,gd = g['game_id'],g['home_team_id'],g['away_team_id'],g['game_date']
        if hid not in TEAM_ABBR or aid not in TEAM_ABBR: return None
        hl,al = self.log[hid], self.log[aid]
        if len(hl)<5 or len(al)<5: return None
        f = {}
        # 1. Rolling
        for pfx,lg in [('h',hl),('a',al)]:
            p = self._roll(lg)
            f[f'{pfx}_off_rtg']=p['or']; f[f'{pfx}_def_rtg']=p['dr']
            f[f'{pfx}_net_rtg']=p['or']-p['dr']; f[f'{pfx}_pace']=p['pace']
            f[f'{pfx}_ts']=p['ts']; f[f'{pfx}_efg']=p['efg']
            f[f'{pfx}_tov_pct']=p['tp']; f[f'{pfx}_oreb_pct']=p['op']
            f[f'{pfx}_ast_r']=p['ar']; f[f'{pfx}_pts_avg']=p['pa']
            f[f'{pfx}_pts_opp']=p['po']
        hp = self._roll(hl); ap = self._roll(al)
        f['diff_off']=hp['or']-ap['or']; f['diff_def']=hp['dr']-ap['dr']
        f['diff_net']=(hp['or']-hp['dr'])-(ap['or']-ap['dr'])
        f['diff_ts']=hp['ts']-ap['ts']; f['diff_pace']=hp['pace']-ap['pace']
        # 2. Fatigue
        for pfx,lg,tid,vtid in [('h',hl,hid,hid),('a',al,aid,hid)]:
            ft = self._fat(lg,gd,tid,vtid)
            f[f'{pfx}_rest']=ft[0]; f[f'{pfx}_b2b']=ft[1]
            f[f'{pfx}_g7d']=ft[2]; f[f'{pfx}_travel']=ft[3]
        # 3. Momentum
        for pfx,lg,tid in [('h',hl,hid),('a',al,aid)]:
            mm = self._mom(lg,tid)
            f[f'{pfx}_streak']=mm[0]; f[f'{pfx}_wp5']=mm[1]
            f[f'{pfx}_wps']=mm[2]; f[f'{pfx}_mtrend']=mm[3]
        hr,ar = self.rec[hid], self.rec[aid]
        f['h_hwp'] = hr['hw']/max(hr['hw']+hr['hl'],1)
        f['a_awp'] = ar['aw']/max(ar['aw']+ar['al'],1)
        # 4. ELO
        f['h_elo']=self.elo[hid]; f['a_elo']=self.elo[aid]
        f['elo_diff']=f['h_elo']-f['a_elo']
        f['elo_exp'] = 1/(1+10**(-(f['elo_diff']+ELO_HCA)/400))
        # 5. Players
        for pfx,tid in [('h',hid),('a',aid)]:
            s,d = self._play(tid)
            f[f'{pfx}_star']=s; f[f'{pfx}_depth']=d
        # 6. Market
        mp,ms,mt = self._mkt(gid)
        f['mkt_prob']=mp; f['mkt_spread']=ms; f['mkt_total']=mt
        # 7. H2H
        hw3,hm = self._h2h(hid,aid)
        f['h2h_wins']=hw3; f['h2h_margin']=hm
        # 8. Season pct
        f['season_pct'] = min((hr['w']+hr['l'])/82, 1.0)
        return np.nan_to_num(np.array([f.get(n,0.0) for n in FEAT_NAMES], dtype=np.float64))

    def update(self, g):
        gid=g['game_id']; hid=g['home_team_id']; aid=g['away_team_id']
        # Skip games with unknown teams (All-Star, etc.)
        if hid not in TEAM_ABBR or aid not in TEAM_ABBR: return
        hs=g['home_score']; aws=g['away_score']; gd=g['game_date']
        hw = hs>aws
        bs = self.bs_idx.get(gid,[])
        hbs = next((b for b in bs if REV_TEAM.get(b.get('nba_tid'))==hid), None)
        abs_ = next((b for b in bs if REV_TEAM.get(b.get('nba_tid'))==aid), None)
        for tid,ih,won,sc,osc,bx in [(hid,True,hw,hs,aws,hbs),(aid,False,not hw,aws,hs,abs_)]:
            e = {'gid':gid,'date':gd,'home':ih,'won':won,'pts':sc,'opp':osc,
                 'margin':sc-osc,'opp_id':aid if ih else hid}
            if bx:
                for s in ['fgm','fga','fg3m','fg3a','ftm','fta','oreb','dreb','ast','tov']:
                    v=bx.get(s); e[s]=float(v) if v is not None and not(isinstance(v,float) and np.isnan(v)) else None
            self.log[tid].append(e)
            r = self.rec[tid]
            if won: r['w']+=1; r['hw' if ih else 'aw']+=1
            else: r['l']+=1; r['hl' if ih else 'al']+=1
        # ELO
        he,ae = self.elo[hid],self.elo[aid]
        exp = 1/(1+10**(-(he-ae+ELO_HCA)/400))
        a = 1.0 if hw else 0.0
        self.elo[hid] += ELO_K*(a-exp); self.elo[aid] += ELO_K*((1-a)-(1-exp))

    def _sm(self, lst, d=0):
        v=[x for x in lst if x is not None]
        return np.mean(v) if v else d

    def _roll(self, lg, w=10):
        r = lg[-w:]
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
        if not lg: return (3,0,0,0)
        ld=lg[-1]['date']
        if isinstance(ld,str): ld=pd.Timestamp(ld)
        dr=(gd-ld).days; c7=gd-timedelta(days=7)
        g7=sum(1 for g in lg if (pd.Timestamp(g['date']) if isinstance(g['date'],str) else g['date'])>=c7)
        td=0; last=lg[-1]; loc=tid if last['home'] else last['opp_id']
        if loc in COORDS and vtid in COORDS:
            c1,c2=COORDS[loc],COORDS[vtid]; td=_hav(c1[0],c1[1],c2[0],c2[1])
        return (min(dr,7), 1 if dr<=1 else 0, g7, min(td/1000,3))

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
        if not recent: return (0,0.5)
        pt=defaultdict(lambda:{'i':0,'g':0})
        for g in recent:
            for p in self.pl_idx.get(g['gid'],[]):
                if REV_TEAM.get(p.get('nba_tid'))!=tid: continue
                mn=p.get('minutes_decimal') or 0
                if mn<5: continue
                imp=((p.get('pts') or 0)+1.2*(p.get('reb') or 0)+1.5*(p.get('ast') or 0)+
                     2*(p.get('stl') or 0)+2*(p.get('blk') or 0)-(p.get('tov') or 0))*(mn/48)
                pt[p['player_id']]['i']+=imp; pt[p['player_id']]['g']+=1
        if not pt: return (0,0.5)
        avgs=sorted([(d['i']/max(d['g'],1)) for d in pt.values()], reverse=True)
        star=sum(avgs[:3])/30; tot=sum(avgs)
        bench=sum(avgs[5:10]) if len(avgs)>5 else 0
        return (star, bench/max(tot,1))

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


# ═══════════════════════ MONTE CARLO ══════════════════════════════════════════
class MCSim:
    def run(self, feat, n=N_SIMS):
        ix = {nm:i for i,nm in enumerate(FEAT_NAMES)}
        hor=feat[ix['h_off_rtg']]; hdr=feat[ix['h_def_rtg']]
        aor=feat[ix['a_off_rtg']]; adr=feat[ix['a_def_rtg']]
        hp=feat[ix['h_pace']]; ap=feat[ix['a_pace']]
        hr=feat[ix['h_rest']]; ar=feat[ix['a_rest']]
        he=feat[ix['h_elo']]; ae=feat[ix['a_elo']]
        pace=np.clip(np.random.normal((hp+ap)/2,3.5,n),70,120)
        h_eff=np.random.normal((hor+(200-adr))/200+0.015+(he-ae)/75000+(hr-ar)*0.003, 0.04, n)
        a_eff=np.random.normal((aor+(200-hdr))/200-0.015-(he-ae)/75000-(hr-ar)*0.003, 0.04, n)
        h_eff+=sp_stats.t.rvs(df=4,scale=0.015,size=n)
        a_eff+=sp_stats.t.rvs(df=4,scale=0.015,size=n)
        hs=pace*h_eff+np.random.normal(0,2,n)
        aws=pace*a_eff+np.random.normal(0,2,n)
        mg=hs-aws
        return {'wp':np.mean(mg>0),'em':np.mean(mg),'ms':np.std(mg),'tm':np.mean(hs+aws)}


# ═══════════════════════ MODEL ════════════════════════════════════════════════
class Model:
    def __init__(self):
        self.gb=GradientBoostingClassifier(n_estimators=200,max_depth=4,learning_rate=0.05,
            subsample=0.8,min_samples_leaf=20,random_state=42)
        self.gbr=GradientBoostingRegressor(n_estimators=200,max_depth=4,learning_rate=0.05,
            subsample=0.8,min_samples_leaf=20,random_state=42)
        self.lr=LogisticRegression(C=1.0,max_iter=1000)
        self.sc=StandardScaler(); self.mc=MCSim()
        self.trained=False; self.tX=[]; self.ty=[]; self.tm=[]
        self.W={'gb':0.35,'lr':0.15,'mc':0.25,'elo':0.15,'mkt':0.10}
        self.phist=[]; self.rhist=[]

    def add(self,X,y,m): self.tX.append(X); self.ty.append(y); self.tm.append(m)

    def retrain(self):
        if len(self.tX)<200: return False
        X=np.array(self.tX); Xs=self.sc.fit_transform(X)
        y=np.array(self.ty); m=np.array(self.tm)
        # Adaptive complexity: fewer estimators early, more later
        n_est = min(100 + len(self.tX)//10, 300)
        self.gb.set_params(n_estimators=n_est)
        self.gbr.set_params(n_estimators=n_est)
        self.gb.fit(Xs, y)
        self.gbr.fit(Xs, m)
        self.lr.fit(Xs, y)
        self.trained=True; return True

    def predict(self, X):
        r={'comp':{}}
        mc=self.mc.run(X); r['comp']['mc']=mc['wp']
        ix={n:i for i,n in enumerate(FEAT_NAMES)}
        ee=X[ix['elo_exp']]; r['comp']['elo']=ee
        mk=X[ix['mkt_prob']]
        if mk<0.1 or mk>0.9: mk=0.5
        r['comp']['mkt']=mk
        if self.trained:
            Xs=self.sc.transform(X.reshape(1,-1))
            gp=self.gb.predict_proba(Xs)[0][1]
            lp=self.lr.predict_proba(Xs)[0][1]
            gm=self.gbr.predict(Xs)[0]
            r['comp']['gb']=gp; r['comp']['lr']=lp
            wp=self.W['gb']*gp+self.W['lr']*lp+self.W['mc']*mc['wp']+self.W['elo']*ee+self.W['mkt']*mk
            mg=(gm+mc['em'])/2
        else:
            r['comp']['gb']=None; r['comp']['lr']=None
            wp=0.4*mc['wp']+0.3*ee+0.3*mk; mg=mc['em']
        r['wp']=np.clip(wp,0.01,0.99); r['mg']=mg
        r['conf']=1-mc['ms']/20; r['total']=mc['tm']
        return r

    def calibrate(self):
        if len(self.phist)<100: return
        rp,rr = self.phist[-500:], self.rhist[-500:]
        sc={}
        for c in ['gb','mc','elo','mkt','lr']:
            pr,ac=[],[]
            for p,a in zip(rp,rr):
                v=p.get('comp',{}).get(c)
                if v is not None and not np.isnan(v): pr.append(v); ac.append(a)
            if len(pr)>20: sc[c]=max(1-brier_score_loss(ac,pr)*2,0.05)
        if sc:
            t=sum(sc.values())
            for c in sc: self.W[c]=sc[c]/t

    def save(self, path=None):
        path=path or f"{MODEL_DIR}/nba_model.pkl"
        with open(path,'wb') as f:
            pickle.dump({'gb':self.gb,'gbr':self.gbr,'lr':self.lr,'sc':self.sc,
                        'W':self.W,'trained':self.trained,'n':len(self.tX)},f)

    def load(self, path=None):
        path=path or f"{MODEL_DIR}/nba_model.pkl"
        if not os.path.exists(path): return False
        with open(path,'rb') as f: d=pickle.load(f)
        if d.get('trained'):
            self.gb=d['gb']; self.gbr=d['gbr']; self.lr=d['lr']
            self.sc=d['sc']; self.W=d.get('W',self.W); self.trained=True
            return True
        return False


# ═══════════════════════ METRICS ══════════════════════════════════════════════
class Metrics:
    def __init__(self): self.probs=[]; self.acts=[]; self.margs=[]
    def add(self, prob, actual, margin):
        self.probs.append(prob); self.acts.append(actual); self.margs.append(margin)
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
        # Calibration bins
        bins=[(0,0.35),(0.35,0.45),(0.45,0.55),(0.55,0.65),(0.65,1.0)]
        cal=[]
        for lo,hi in bins:
            mask=[lo<=p<hi for p in pr]
            nb=sum(mask)
            if nb>0:
                af=np.mean([a for a,m in zip(ac,mask) if m])
                pf=np.mean([p for p,m in zip(pr,mask) if m])
                cal.append((lo,hi,nb,pf,af))
        print(f"\n{'='*65}")
        print(f"  📊 CHECKPOINT {label} ({n} juegos)")
        print(f"{'='*65}")
        print(f"  Accuracy:     {acc:.1%}  {'✅' if acc>0.60 else '⚠️' if acc>0.55 else '❌'}")
        print(f"  Brier Score:  {brier:.4f}  {'✅' if brier<0.24 else '⚠️' if brier<0.26 else '❌'}")
        print(f"  Log Loss:     {ll:.4f}  {'✅' if ll<0.65 else '⚠️' if ll<0.70 else '❌'}")
        print(f"  AUC:          {auc:.4f}  {'✅' if auc>0.65 else '⚠️' if auc>0.60 else '❌'}")
        if cal:
            print(f"\n  Calibración:")
            for lo,hi,nb,pf,af in cal:
                print(f"    [{lo:.2f}-{hi:.2f}] n={nb:>4}  pred={pf:.3f}  actual={af:.3f}")
        print(f"{'='*65}")
        return {'acc':acc,'brier':brier,'ll':ll,'auc':auc}


# ═══════════════════════ PIPELINE ═════════════════════════════════════════════
def run(train_s, eval_s, ckpt=CHECKPOINT_EVERY, db=DB_PATH):
    t0=time.time()
    all_s = train_s + [eval_s]
    print(f"\n{'='*65}")
    print(f"  NBA TRAINING PIPELINE v2")
    print(f"  Train: {', '.join(train_s)}")
    print(f"  Eval:  {eval_s}")
    print(f"  Checkpoint: cada {ckpt} partidos")
    print(f"{'='*65}\n")

    dl=DataLoader(db)
    games=dl.load_games(all_s)
    bs=dl.load_boxscores(); pl=dl.load_players(); od=dl.load_odds()
    dl.close()
    logger.info(f"Games:{len(games)} BS:{len(bs)} PL:{len(pl)} Odds:{len(od)}")

    eng=FeatureEngine(bs,pl,od); mdl=Model()
    train_m=Metrics(); eval_m=Metrics()
    proc=0; skip=0; cur_s=None; retrain_every=500

    for _,g in games.iterrows():
        s=g['season']; is_ev=(s==eval_s)
        # Season transition
        if s!=cur_s:
            if cur_s is not None:
                if not is_ev: train_m.report(label=f"FIN {cur_s}")
                for t in eng.elo: eng.elo[t]=0.75*eng.elo[t]+0.25*ELO_INIT
            eng.reset_season(s); cur_s=s
            logger.info(f"\n🏀 {s} {'[EVAL]' if is_ev else '[TRAIN]'}")

        feat=eng.compute(g)
        if feat is not None:
            pred=mdl.predict(feat)
            aw=g['home_win']; am=g['margin']
            tk = eval_m if is_ev else train_m
            tk.add(pred['wp'], aw, am)
            mdl.phist.append(pred); mdl.rhist.append(aw)
            if not is_ev: mdl.add(feat, aw, am)
            proc+=1
        else:
            skip+=1

        eng.update(g)

        if not is_ev and proc>0 and proc%retrain_every==0:
            if mdl.retrain():
                logger.info(f"🔄 Retrained ({len(mdl.tX)} samples)")
                mdl.calibrate()

        if proc>0 and proc%ckpt==0:
            tk = eval_m if is_ev else train_m
            met=tk.report(last_n=ckpt, label=f"{'EVAL' if is_ev else 'TRAIN'} #{proc}")
            if met.get('acc',1)<0.52 and proc>500:
                logger.warning("⚠️  Low accuracy, recalibrating...")
                mdl.calibrate()
                if not mdl.trained and len(mdl.tX)>=200: mdl.retrain()

    # Final
    if mdl.retrain(): logger.info(f"🔄 Final train ({len(mdl.tX)} samples)")
    mdl.calibrate(); mdl.save()

    print(f"\n{'='*65}")
    print(f"  📊 RESUMEN FINAL")
    print(f"{'='*65}")
    print(f"  Procesados: {proc}  Omitidos: {skip}")
    print(f"  Tiempo: {(time.time()-t0)/60:.1f} min\n")
    print(f"  --- TRAIN ---")
    tf=train_m.report(label="TRAIN FINAL")
    print(f"\n  --- EVAL ({eval_s}) ---")
    ef=eval_m.report(label="EVAL FINAL")

    if mdl.trained:
        print(f"\n  Top 15 Features:")
        imp=mdl.gb.feature_importances_
        fi=sorted(zip(FEAT_NAMES,imp),key=lambda x:x[1],reverse=True)
        for nm,im in fi[:15]:
            print(f"    {nm:20s} {im:.4f} {'█'*int(im*200)}")
    print(f"\n  Weights: {mdl.W}")
    print(f"{'='*65}\n")
    return tf, ef


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
        if m.load(): print(f"Model loaded. Weights: {m.W}")
        else: print("No saved model")
        return
    run(a.seasons, a.eval, a.checkpoint, a.db)

if __name__=="__main__":
    main()
