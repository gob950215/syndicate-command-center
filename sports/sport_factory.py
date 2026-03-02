"""
SYNDICATE — Sport Factory
==========================
Factory pattern for registering and instantiating sport engines.
Adding a new sport = create engine class + register here.
"""
from __future__ import annotations
from typing import Dict, Type
from core.base_sport import BaseSportEngine, NullSportEngine
from core.nba_engine import NBAEngine


class SportFactory:
    """
    Registry of sport engines. Instantiates the correct engine per sport.

    Usage:
        factory = SportFactory()
        nba = factory.get("NBA")       # → NBAEngine instance
        mlb = factory.get("MLB")       # → NullSportEngine (stub)
    """

    _registry: Dict[str, Type[BaseSportEngine]] = {}

    @classmethod
    def register(cls, sport_name: str, engine_class: Type[BaseSportEngine]):
        cls._registry[sport_name.upper()] = engine_class

    @classmethod
    def get(cls, sport_name: str) -> BaseSportEngine:
        engine_class = cls._registry.get(sport_name.upper())
        if engine_class:
            return engine_class()
        return NullSportEngine(sport_name, "🏟️", "#888888")

    @classmethod
    def available_sports(cls) -> Dict[str, bool]:
        """Return dict of sport_name → is_real_engine."""
        from config import SUPPORTED_SPORTS
        result = {}
        for sport in SUPPORTED_SPORTS:
            result[sport] = sport.upper() in cls._registry
        return result


# ── Register known engines ──────────────────────────────────────────────
SportFactory.register("NBA", NBAEngine)

# Future:
# SportFactory.register("MLB", MLBEngine)
# SportFactory.register("NFL", NFLEngine)
