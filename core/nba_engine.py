"""
SYNDICATE — NBA Engine Adapter
===============================
Wraps nba_syndicate_v8.py classes for the GUI.
Translates raw V8 output into our typed data models.
"""
from __future__ import annotations
import sys
import os
import importlib
import logging
from typing import List, Dict, Optional
from datetime import date
from pathlib import Path

from core.base_sport import BaseSportEngine
from core.data_models import (
    PickResult, PickTier, RiskLevel, GameSchedule,
    TeamAnalysis, FourFactors, SmartMoneyData,
)

logger = logging.getLogger("NBA-Adapter")


class NBAEngine(BaseSportEngine):
    """
    Concrete NBA adapter. Dynamically loads nba_syndicate_v8.py
    and wraps its classes for the GUI.
    """

    sport_name = "NBA"
    sport_icon = "🏀"
    sport_color = "#E8590C"

    def __init__(self):
        super().__init__()
        self._v8_module = None
        self._engine = None        # EngineV8 instance
        self._model = None         # ModelV8 instance
        self._mc = None            # MonteCarloV8 instance
        self._sports_client = None
        self._odds_client = None
        self._last_picks: List[PickResult] = []
        self._feature_cache: Dict[str, list] = {}

    # ── Interface implementation ────────────────────────────────────────

    def initialize(self, db_path: str, model_dir: str) -> bool:
        """Load the V8 module and replay historical games."""
        try:
            self._v8_module = self._import_v8()
            if self._v8_module is None:
                self._last_error = "Could not import nba_syndicate_v8.py"
                return False

            v8 = self._v8_module

            # Adjust paths in V8
            v8.DB_PATH = db_path
            v8.MODEL_DIR = model_dir

            if not os.path.exists(db_path):
                self._last_error = f"Database not found: {db_path}"
                return False

            # Load model
            self._model = v8.ModelV8()
            model_loaded = self._model.load(os.path.join(model_dir, "nba_model_v8.pkl"))
            self._model_loaded = model_loaded

            if not model_loaded:
                logger.warning("No trained model found — predictions will use MC/Elo only")

            # Load historical data for engine state
            logger.info("Loading historical data...")
            dl = v8.DataLoader(db_path)
            all_seasons = v8.TRAIN_SEASONS + [v8.CURRENT_SEASON]
            games = dl.load_games(all_seasons)
            bs = dl.load_boxscores()
            pl = dl.load_players()
            od = dl.load_odds()
            dl.close()

            # Build engine state
            self._engine = v8.EngineV8(bs, pl, od, games)
            self._mc = v8.MonteCarloV8(self._engine, 5000)
            self._model.connect_mc(self._mc)
            self._model.connect_engine(self._engine)

            # Replay history
            logger.info(f"Replaying {len(games)} games to build state...")
            cur = None
            for _, g in games.iterrows():
                s = g["season"]
                if s != cur:
                    if cur is not None:
                        for t in self._engine.elo:
                            self._engine.elo[t] = 0.75 * self._engine.elo[t] + 0.25 * v8.ELO_INIT
                    self._engine.reset_season(s)
                    cur = s
                self._engine.update(g)

            # Initialize live API clients
            self._sports_client = v8.SportsAPIClient()
            self._odds_client = v8.OddsAPIClient()

            self._is_ready = True
            logger.info(f"NBA Engine ready | Model: {'✓' if model_loaded else '✗'} | Games: {len(games)}")
            return True

        except Exception as e:
            self._last_error = str(e)
            logger.error(f"NBA init failed: {e}", exc_info=True)
            return False

    def get_todays_games(self) -> List[GameSchedule]:
        """Fetch today's NBA schedule."""
        if not self._sports_client:
            return []
        try:
            today = date.today().isoformat()
            raw = self._sports_client.get_todays_games(today)
            games = []
            for g in raw:
                games.append(GameSchedule(
                    game_id=str(g.get("api_game_id", "")),
                    sport="NBA",
                    date=today,
                    home_team=g.get("home_name", ""),
                    away_team=g.get("away_name", ""),
                    status=g.get("status", "scheduled"),
                ))
            return games
        except Exception as e:
            logger.error(f"get_todays_games: {e}")
            return []

    def generate_picks(self) -> List[PickResult]:
        """Run the V8 live pipeline and return typed PickResults."""
        if not self._is_ready:
            return []

        v8 = self._v8_module
        today = date.today().isoformat()
        picks = []

        try:
            # Inject standings
            standings = self._sports_client.get_standings() if self._sports_client else {}
            self._engine.inject_standings(standings)

            # Fetch live odds
            raw_odds = self._odds_client.get_live_odds() if self._odds_client else []
            live_odds = self._odds_client.parse_game_odds(raw_odds) if raw_odds else {}
            self._engine.inject_live_odds(live_odds)

            # Build game entries from live odds
            import pandas as pd
            game_entries = []
            for (hid, aid), od_data in live_odds.items():
                game_entries.append({
                    "game_id": hash((hid, aid, today)) % 10 ** 8,
                    "home_team_id": hid,
                    "away_team_id": aid,
                    "game_date": pd.Timestamp(today),
                    "home_score": None,
                    "away_score": None,
                    "home_win": None,
                    "margin": None,
                })

            if not game_entries:
                logger.info("No matchups to analyze")
                return []

            for ge in game_entries:
                hid = ge["home_team_id"]
                aid = ge["away_team_id"]
                ha = v8.TEAM_ABBR.get(hid, "???")
                aa = v8.TEAM_ABBR.get(aid, "???")

                g_series = pd.Series(ge)
                feat = self._engine.compute(g_series, is_current_season=True)
                if feat is None:
                    continue

                pred = self._model.predict(feat, game=g_series, live_mode=True)

                # Map to our data model
                pick_team = ha if pred["pick_home"] else aa
                conf = max(pred["wp"], 1 - pred["wp"])
                mkt_p = pred.get("mkt_prob_home", 0.5)

                tier = PickTier.STANDARD
                if pred.get("vip_reason") == "DIAMOND":
                    tier = PickTier.DIAMOND
                elif pred.get("vip_reason") == "RLM_DIAMOND":
                    tier = PickTier.RLM_DIAMOND
                elif pred.get("vip_reason") == "TOP3_FALLBACK":
                    tier = PickTier.TOP3_FALLBACK

                risk_map = {"Bajo": RiskLevel.LOW, "Medio": RiskLevel.MEDIUM, "Alto": RiskLevel.HIGH}
                risk = risk_map.get(pred.get("risk_level", "Medio"), RiskLevel.MEDIUM)

                pick_id = f"NBA-{today}-{aa}@{ha}"

                result = PickResult(
                    pick_id=pick_id,
                    sport="NBA",
                    date=today,
                    matchup=f"{aa} @ {ha}",
                    home_team=ha,
                    away_team=aa,
                    pick=pick_team,
                    pick_home=pred["pick_home"],
                    confidence=conf,
                    mkt_prob=mkt_p if pred["pick_home"] else 1 - mkt_p,
                    mkt_gap=pred.get("mkt_gap", 0),
                    ev=pred.get("ev", 0),
                    mkt_odds=pred.get("mkt_odds", 0),
                    tier=tier,
                    risk_level=risk,
                    vip_reason=pred.get("vip_reason", ""),
                    rlm=int(pred.get("rlm", 0)),
                    fatigue_trap=pred.get("fatigue_trap", False),
                    value_trap=pred.get("value_trap", False),
                    playoff_urgency=pred.get("playoff_urgency_penalty", False),
                    mc_volatility=pred.get("mc_volatility", 12),
                    feature_vector=feat.tolist() if feat is not None else None,
                )

                # Build smart money data
                od = live_odds.get((hid, aid), {})
                result.smart_money = SmartMoneyData(
                    mkt_prob_home=od.get("mkt_prob_home", 0.5),
                    mkt_spread=od.get("mkt_spread", 0),
                    rlm_signal=od.get("rlm_signal", 0),
                    odds_move=od.get("odds_move_home", 0),
                    consensus_spread=od.get("consensus_spread", 0),
                    n_bookmakers=od.get("n_bookmakers", 0),
                )

                # Store feature vector for analysis
                self._feature_cache[pick_id] = feat.tolist()

                picks.append(result)

            self._last_picks = picks
            return picks

        except Exception as e:
            logger.error(f"generate_picks: {e}", exc_info=True)
            self._last_error = str(e)
            return []

    def get_detailed_analysis(self, pick: PickResult) -> PickResult:
        """Enrich a pick with full team breakdowns."""
        if not self._is_ready or not self._v8_module:
            return pick

        v8 = self._v8_module
        feat_vec = self._feature_cache.get(pick.pick_id)
        if feat_vec is None:
            return pick

        feat_names = v8.FEAT

        def _get(name):
            try:
                idx = feat_names.index(name)
                return feat_vec[idx]
            except (ValueError, IndexError):
                return 0.0

        # Build home team analysis
        pick.home_analysis = TeamAnalysis(
            team_abbr=pick.home_team,
            four_factors=FourFactors(
                efg=_get("h_efg"),
                tov_pct=_get("h_tov_pct"),
                oreb_pct=_get("h_oreb_pct"),
                ft_rate=_get("h_ft_rate"),
            ),
            pace=_get("h_pace") * 100,
            off_rtg=_get("h_ortg") * 120,
            def_rtg=_get("h_drtg") * 120,
            net_rtg=_get("h_net_rtg") * 30,
            fatigue_b2b=bool(_get("h_b2b")),
            fatigue_heavy_legs=bool(_get("h_heavy_legs")),
            games_7d=int(_get("h_games_7d") * 4),
            travel_miles_7d=_get("h_travel_miles_7d") * 8000,
            missing_stars=_get("h_missing_stars"),
            q4_clutch=_get("h_q4_net_avg"),
            playoff_urgency=_get("h_playoff_urgency"),
        )

        # Build away team analysis
        pick.away_analysis = TeamAnalysis(
            team_abbr=pick.away_team,
            four_factors=FourFactors(
                efg=_get("a_efg"),
                tov_pct=_get("a_tov_pct"),
                oreb_pct=_get("a_oreb_pct"),
                ft_rate=_get("a_ft_rate"),
            ),
            pace=_get("a_pace") * 100,
            off_rtg=_get("a_ortg") * 120,
            def_rtg=_get("a_drtg") * 120,
            net_rtg=_get("a_net_rtg") * 30,
            fatigue_b2b=bool(_get("a_b2b")),
            fatigue_heavy_legs=bool(_get("a_heavy_legs")),
            games_7d=int(_get("a_games_7d") * 4),
            travel_miles_7d=_get("a_travel_miles_7d") * 8000,
            missing_stars=_get("a_missing_stars"),
            q4_clutch=_get("a_q4_net_avg"),
            playoff_urgency=_get("a_playoff_urgency"),
        )

        return pick

    def get_feature_names(self) -> List[str]:
        if self._v8_module:
            return list(self._v8_module.FEAT)
        return []

    # ── Internal helpers ────────────────────────────────────────────────

    def _import_v8(self):
        """Dynamically import nba_syndicate_v8.py."""
        # Search paths: current dir, parent dir, data dir
        search = [
            Path("."),
            Path(__file__).parent.parent,
            Path(__file__).parent.parent / "data",
        ]

        for base in search:
            v8_path = base / "nba_syndicate_v8.py"
            if v8_path.exists():
                spec = importlib.util.spec_from_file_location("nba_syndicate_v8", str(v8_path))
                mod = importlib.util.module_from_spec(spec)
                sys.modules["nba_syndicate_v8"] = mod
                spec.loader.exec_module(mod)
                logger.info(f"Loaded V8 from {v8_path}")
                return mod

        logger.error("nba_syndicate_v8.py not found in search paths")
        return None

    @property
    def team_abbr_map(self) -> Dict:
        if self._v8_module:
            return dict(self._v8_module.TEAM_ABBR)
        return {}

    @property
    def last_picks(self) -> List[PickResult]:
        return self._last_picks
