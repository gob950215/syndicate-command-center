"""
SYNDICATE — Pick Manager
Handles pick lifecycle: generate → validate → track → settle.
"""
from __future__ import annotations
import json
import sqlite3
import logging
from datetime import datetime
from typing import List, Optional, Dict
from pathlib import Path

from core.data_models import PickResult, PickStatus, PickTier

logger = logging.getLogger("PickManager")


class PickManager:
    """Manages picks persistence and lifecycle."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS picks (
                pick_id TEXT PRIMARY KEY,
                sport TEXT,
                date TEXT,
                matchup TEXT,
                pick TEXT,
                confidence REAL,
                mkt_prob REAL,
                ev REAL,
                tier TEXT,
                risk_level TEXT,
                rlm INTEGER,
                status TEXT DEFAULT 'pending',
                expert_notes TEXT DEFAULT '',
                raw_json TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)
        conn.commit()
        conn.close()

    def save_picks(self, picks: List[PickResult]):
        conn = sqlite3.connect(self.db_path)
        now = datetime.now().isoformat()
        for p in picks:
            raw = json.dumps({
                "confidence": p.confidence,
                "mkt_prob": p.mkt_prob,
                "mkt_gap": p.mkt_gap,
                "ev": p.ev,
                "mkt_odds": p.mkt_odds,
                "tier": p.tier.value,
                "vip_reason": p.vip_reason,
                "rlm": p.rlm,
                "fatigue_trap": p.fatigue_trap,
                "value_trap": p.value_trap,
                "playoff_urgency": p.playoff_urgency,
                "mc_volatility": p.mc_volatility,
            })
            conn.execute("""
                INSERT OR REPLACE INTO picks
                (pick_id, sport, date, matchup, pick, confidence, mkt_prob, ev,
                 tier, risk_level, rlm, status, expert_notes, raw_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                p.pick_id, p.sport, p.date, p.matchup, p.pick,
                p.confidence, p.mkt_prob, p.ev,
                p.tier.value, p.risk_level.value, p.rlm,
                p.status.value, p.expert_notes, raw, now, now,
            ))
        conn.commit()
        conn.close()
        logger.info(f"Saved {len(picks)} picks")

    def update_status(self, pick_id: str, status: PickStatus, notes: str = ""):
        conn = sqlite3.connect(self.db_path)
        now = datetime.now().isoformat()
        conn.execute(
            "UPDATE picks SET status=?, expert_notes=?, updated_at=? WHERE pick_id=?",
            (status.value, notes, now, pick_id)
        )
        conn.commit()
        conn.close()

    def get_picks_by_date(self, date_str: str) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM picks WHERE date=? ORDER BY confidence DESC", (date_str,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_pick_stats(self) -> Dict:
        conn = sqlite3.connect(self.db_path)
        total = conn.execute("SELECT COUNT(*) FROM picks").fetchone()[0]
        diamonds = conn.execute(
            "SELECT COUNT(*) FROM picks WHERE tier IN ('DIAMOND','RLM_DIAMOND')"
        ).fetchone()[0]
        wins = conn.execute("SELECT COUNT(*) FROM picks WHERE status='win'").fetchone()[0]
        losses = conn.execute("SELECT COUNT(*) FROM picks WHERE status='loss'").fetchone()[0]
        settled = wins + losses
        conn.close()
        return {
            "total": total,
            "diamonds": diamonds,
            "wins": wins,
            "losses": losses,
            "win_rate": wins / settled if settled > 0 else 0,
            "settled": settled,
        }
