"""
SYNDICATE — Data Models
Typed dataclasses for picks, games, analysis results.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from enum import Enum


class PickTier(Enum):
    DIAMOND = "DIAMOND"
    RLM_DIAMOND = "RLM_DIAMOND"
    TOP3_FALLBACK = "TOP3_FALLBACK"
    STANDARD = "STANDARD"


class RiskLevel(Enum):
    LOW = "Bajo"
    MEDIUM = "Medio"
    HIGH = "Alto"


class PickStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    SETTLED_WIN = "win"
    SETTLED_LOSS = "loss"


@dataclass
class FourFactors:
    """The Four Factors of basketball analytics."""
    efg: float = 0.0           # Effective Field Goal %
    tov_pct: float = 0.0       # Turnover %
    oreb_pct: float = 0.0      # Offensive Rebound %
    ft_rate: float = 0.0       # Free Throw Rate

    @property
    def as_dict(self) -> dict:
        return {
            "eFG%": self.efg,
            "TOV%": self.tov_pct,
            "OREB%": self.oreb_pct,
            "FT Rate": self.ft_rate,
        }


@dataclass
class TeamAnalysis:
    """Detailed team analysis for the War Room."""
    team_abbr: str = ""
    team_name: str = ""
    four_factors: FourFactors = field(default_factory=FourFactors)
    pace: float = 0.0
    off_rtg: float = 0.0
    def_rtg: float = 0.0
    net_rtg: float = 0.0
    elo: float = 1500.0
    fatigue_b2b: bool = False
    fatigue_heavy_legs: bool = False
    games_7d: int = 0
    travel_miles_7d: float = 0.0
    missing_stars: float = 0.0
    q4_clutch: float = 0.0
    playoff_urgency: float = 0.0


@dataclass
class SmartMoneyData:
    """Market/odds intelligence."""
    mkt_prob_home: float = 0.5
    mkt_spread: float = 0.0
    rlm_signal: int = 0           # -1, 0, +1
    odds_move: float = 0.0
    consensus_spread: float = 0.0
    n_bookmakers: int = 0

    @property
    def rlm_label(self) -> str:
        if self.rlm_signal == 1:
            return "🟢 CONFIRMED"
        elif self.rlm_signal == -1:
            return "🔴 REVERSE"
        return "⚪ NEUTRAL"


@dataclass
class PickResult:
    """A single prediction result from the model."""
    pick_id: str = ""
    sport: str = "NBA"
    date: str = ""
    matchup: str = ""               # e.g., "LAL @ BOS"
    home_team: str = ""
    away_team: str = ""
    pick: str = ""                   # team abbreviation
    pick_home: bool = True

    # Core metrics
    confidence: float = 0.0          # model probability
    mkt_prob: float = 0.5            # market implied probability
    mkt_gap: float = 0.0             # confidence - mkt_prob
    ev: float = 0.0                  # expected value
    mkt_odds: float = 0.0            # decimal market odds
    bet_type: str = "Moneyline"      # "Moneyline", "Spread", "Total"

    # Classification
    tier: PickTier = PickTier.STANDARD
    risk_level: RiskLevel = RiskLevel.MEDIUM
    vip_reason: str = ""

    # Signals
    rlm: int = 0
    fatigue_trap: bool = False
    value_trap: bool = False
    playoff_urgency: bool = False
    mc_volatility: float = 12.0

    # Detailed analysis (populated on demand)
    home_analysis: Optional[TeamAnalysis] = None
    away_analysis: Optional[TeamAnalysis] = None
    smart_money: Optional[SmartMoneyData] = None

    # User interaction
    status: PickStatus = PickStatus.PENDING
    expert_notes: str = ""

    # Feature vector (for debugging)
    feature_vector: Optional[list] = None

    @property
    def is_diamond(self) -> bool:
        return self.tier in (PickTier.DIAMOND, PickTier.RLM_DIAMOND)

    @property
    def confidence_pct(self) -> str:
        return f"{self.confidence:.1%}"

    @property
    def ev_display(self) -> str:
        return f"{self.ev:+.3f}"

    @property
    def edge_display(self) -> str:
        return f"{self.mkt_gap:+.1%}"


@dataclass
class GameSchedule:
    """A scheduled game from the API."""
    game_id: str = ""
    sport: str = "NBA"
    date: str = ""
    time: str = ""
    home_team: str = ""
    away_team: str = ""
    status: str = "scheduled"        # scheduled, live, final
    lines_opened: bool = False


@dataclass
class SchedulerRule:
    """User-defined scheduling rule."""
    rule_id: str = ""
    name: str = ""
    sport: str = "NBA"
    trigger_type: str = ""           # "after_open", "before_lock", "fixed_time"
    offset_minutes: int = 120        # minutes relative to trigger
    enabled: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None


@dataclass
class AccessLog:
    """Admin access log entry."""
    timestamp: str = ""
    username: str = ""
    action: str = ""                 # login, logout, run_analysis, etc.
    ip_address: str = ""
    hostname: str = ""
    success: bool = True
    details: str = ""
