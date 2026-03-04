"""
SYNDICATE — Pick Manager
Handles pick lifecycle: generate → validate → track → settle.
"""
from __future__ import annotations
import json
import sqlite3
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path

from core.data_models import PickResult, PickStatus, PickTier

logger = logging.getLogger("PickManager")

def _to_jsonable(x: Any):
        """Convierte numpy/torch/otros tipos a tipos serializables por JSON."""
        if x is None:
            return None

        # numpy scalar -> python scalar
        try:
            import numpy as np
            if isinstance(x, (np.floating, np.integer)):
                return x.item()
            if isinstance(x, np.ndarray):
                return x.tolist()
        except Exception:
            pass

        # torch tensor -> python scalar / list
        try:
            import torch
            if isinstance(x, torch.Tensor):
                return x.item() if x.ndim == 0 else x.detach().cpu().tolist()
        except Exception:
            pass

        # objetos con .item() (muchos scalars lo tienen)
        try:
            return x.item()
        except Exception:
            pass

        # ints/floats python
        if isinstance(x, (int, float, str, bool)):
            return x

        # fallback: stringify (para no romper el guardado)
        return str(x)

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
            raw_payload = {
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
            }

            raw = json.dumps(raw_payload, default=_to_jsonable, ensure_ascii=False)

            # Normaliza también los campos numéricos que van a columnas REAL/INTEGER
            confidence = float(_to_jsonable(p.confidence)) if p.confidence is not None else None
            mkt_prob = float(_to_jsonable(p.mkt_prob)) if p.mkt_prob is not None else None
            ev = float(_to_jsonable(p.ev)) if p.ev is not None else None
            rlm = int(_to_jsonable(p.rlm)) if p.rlm is not None else None

            conn.execute("""
                INSERT OR REPLACE INTO picks
                (pick_id, sport, date, matchup, pick, confidence, mkt_prob, ev,
                tier, risk_level, rlm, status, expert_notes, raw_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                p.pick_id, p.sport, p.date, p.matchup, p.pick,
                confidence, mkt_prob, ev,
                p.tier.value, p.risk_level.value, rlm,
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

    