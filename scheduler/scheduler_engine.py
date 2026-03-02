"""
SYNDICATE — Smart Scheduler Engine
====================================
User-defined rules that trigger model runs based on:
  - Lines opening time (offset)
  - Lock-in time (offset before tipoff)
  - Fixed daily schedules
  - Calendar-aware (reads game times from API)
"""
from __future__ import annotations
import json
import sqlite3
import logging
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Callable
from pathlib import Path

from core.data_models import SchedulerRule

logger = logging.getLogger("Scheduler")


class SchedulerEngine:
    """
    Rule-based scheduler that triggers analysis runs.
    Designed for integration with APScheduler or QTimer.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._rules: List[SchedulerRule] = []
        self._callbacks: Dict[str, Callable] = {}
        self._init_db()
        self._load_rules()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scheduler_rules (
                rule_id TEXT PRIMARY KEY,
                name TEXT,
                sport TEXT,
                trigger_type TEXT,
                offset_minutes INTEGER,
                enabled INTEGER DEFAULT 1,
                last_run TEXT,
                next_run TEXT,
                created_at TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scheduler_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rule_id TEXT,
                timestamp TEXT,
                status TEXT,
                details TEXT
            )
        """)
        conn.commit()
        conn.close()

    def _load_rules(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM scheduler_rules ORDER BY name").fetchall()
        conn.close()
        self._rules = []
        for r in rows:
            rule = SchedulerRule(
                rule_id=r["rule_id"],
                name=r["name"],
                sport=r["sport"],
                trigger_type=r["trigger_type"],
                offset_minutes=r["offset_minutes"],
                enabled=bool(r["enabled"]),
                last_run=datetime.fromisoformat(r["last_run"]) if r["last_run"] else None,
                next_run=datetime.fromisoformat(r["next_run"]) if r["next_run"] else None,
            )
            self._rules.append(rule)

    def add_rule(self, name: str, sport: str, trigger_type: str, offset_minutes: int) -> SchedulerRule:
        """Add a new scheduling rule."""
        rule = SchedulerRule(
            rule_id=str(uuid.uuid4())[:8],
            name=name,
            sport=sport,
            trigger_type=trigger_type,
            offset_minutes=offset_minutes,
            enabled=True,
        )
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO scheduler_rules
            (rule_id, name, sport, trigger_type, offset_minutes, enabled, created_at)
            VALUES (?, ?, ?, ?, ?, 1, ?)
        """, (rule.rule_id, name, sport, trigger_type, offset_minutes, datetime.now().isoformat()))
        conn.commit()
        conn.close()
        self._rules.append(rule)
        logger.info(f"Rule added: {name} ({trigger_type}, {offset_minutes}min)")
        return rule

    def remove_rule(self, rule_id: str) -> bool:
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM scheduler_rules WHERE rule_id=?", (rule_id,))
        conn.commit()
        conn.close()
        self._rules = [r for r in self._rules if r.rule_id != rule_id]
        return True

    def toggle_rule(self, rule_id: str) -> bool:
        for r in self._rules:
            if r.rule_id == rule_id:
                r.enabled = not r.enabled
                conn = sqlite3.connect(self.db_path)
                conn.execute(
                    "UPDATE scheduler_rules SET enabled=? WHERE rule_id=?",
                    (int(r.enabled), rule_id)
                )
                conn.commit()
                conn.close()
                return True
        return False

    def get_rules(self) -> List[SchedulerRule]:
        return self._rules

    def compute_next_runs(self, game_times: List[datetime]):
        """
        Given a list of game start times (from API calendar),
        compute the next run time for each rule.
        """
        if not game_times:
            return

        earliest_game = min(game_times)
        latest_game = max(game_times)

        for rule in self._rules:
            if not rule.enabled:
                continue

            if rule.trigger_type == "after_open":
                # Lines typically open ~12 hours before tipoff
                lines_open = earliest_game - timedelta(hours=12)
                rule.next_run = lines_open + timedelta(minutes=rule.offset_minutes)

            elif rule.trigger_type == "before_lock":
                # Lock-in = before earliest game start
                rule.next_run = earliest_game - timedelta(minutes=rule.offset_minutes)

            elif rule.trigger_type == "fixed_time":
                # offset_minutes = minutes from midnight
                today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                rule.next_run = today + timedelta(minutes=rule.offset_minutes)
                if rule.next_run < datetime.now():
                    rule.next_run += timedelta(days=1)

        # Persist
        conn = sqlite3.connect(self.db_path)
        for r in self._rules:
            if r.next_run:
                conn.execute(
                    "UPDATE scheduler_rules SET next_run=? WHERE rule_id=?",
                    (r.next_run.isoformat(), r.rule_id)
                )
        conn.commit()
        conn.close()

    def check_pending(self) -> List[SchedulerRule]:
        """Check which rules are due to fire now."""
        now = datetime.now()
        due = []
        for r in self._rules:
            if r.enabled and r.next_run and r.next_run <= now:
                if r.last_run is None or r.last_run < r.next_run:
                    due.append(r)
        return due

    def mark_run(self, rule_id: str, status: str = "success", details: str = ""):
        """Mark a rule as executed."""
        now = datetime.now()
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "UPDATE scheduler_rules SET last_run=? WHERE rule_id=?",
            (now.isoformat(), rule_id)
        )
        conn.execute(
            "INSERT INTO scheduler_log (rule_id, timestamp, status, details) VALUES (?,?,?,?)",
            (rule_id, now.isoformat(), status, details)
        )
        conn.commit()
        conn.close()
        for r in self._rules:
            if r.rule_id == rule_id:
                r.last_run = now
                break

    def get_log(self, limit: int = 50) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM scheduler_log ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def register_callback(self, sport: str, callback: Callable):
        """Register a callback to execute when a rule fires."""
        self._callbacks[sport] = callback

    def fire_rule(self, rule: SchedulerRule):
        """Execute the callback for a due rule."""
        cb = self._callbacks.get(rule.sport)
        if cb:
            try:
                cb()
                self.mark_run(rule.rule_id, "success")
                logger.info(f"Rule fired: {rule.name}")
            except Exception as e:
                self.mark_run(rule.rule_id, "error", str(e))
                logger.error(f"Rule failed: {rule.name}: {e}")
        else:
            logger.warning(f"No callback for sport: {rule.sport}")
