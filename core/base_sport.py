"""
SYNDICATE — Base Sport Engine (Abstract)
========================================
Factory pattern base class. All sports inherit from this.
To add MLB or NFL: subclass BaseSportEngine, implement the 5 abstract methods,
register in sport_factory.py → the GUI auto-detects it.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from core.data_models import PickResult, GameSchedule, TeamAnalysis, FourFactors


class BaseSportEngine(ABC):
    """Abstract base for any sport prediction engine."""

    sport_name: str = "UNKNOWN"
    sport_icon: str = "🏟️"
    sport_color: str = "#FFFFFF"

    def __init__(self):
        self._is_ready = False
        self._model_loaded = False
        self._last_error: Optional[str] = None

    # ── Required overrides ──────────────────────────────────────────────

    @abstractmethod
    def initialize(self, db_path: str, model_dir: str) -> bool:
        """Load historical data & trained model. Return True if ready."""
        ...

    @abstractmethod
    def get_todays_games(self) -> List[GameSchedule]:
        """Fetch today's game schedule from live API."""
        ...

    @abstractmethod
    def generate_picks(self) -> List[PickResult]:
        """Run the full pipeline: fetch data → compute → predict → return picks."""
        ...

    @abstractmethod
    def get_detailed_analysis(self, pick: PickResult) -> PickResult:
        """Enrich a pick with full team analysis (Four Factors, fatigue, etc.)."""
        ...

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Return ordered list of feature names for the model."""
        ...

    # ── Common interface ────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    def get_config(self) -> Dict:
        """Return sport-specific configuration for the settings panel."""
        return {
            "sport": self.sport_name,
            "icon": self.sport_icon,
            "color": self.sport_color,
            "ready": self._is_ready,
            "model_loaded": self._model_loaded,
        }


class NullSportEngine(BaseSportEngine):
    """Placeholder for sports not yet implemented (MLB, NFL stubs)."""

    def __init__(self, name: str, icon: str, color: str):
        super().__init__()
        self.sport_name = name
        self.sport_icon = icon
        self.sport_color = color

    def initialize(self, db_path, model_dir):
        self._last_error = f"{self.sport_name} module not yet implemented"
        return False

    def get_todays_games(self):
        return []

    def generate_picks(self):
        return []

    def get_detailed_analysis(self, pick):
        return pick

    def get_feature_names(self):
        return []
