"""
SYNDICATE — Dashboard Panel
=============================
Home screen showing today's picks, stats, and quick actions.
"""
from __future__ import annotations
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QFrame, QProgressBar, QSizePolicy,
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont
from datetime import datetime

from gui.theme import (
    BG_PRIMARY, BG_SECONDARY, BG_TERTIARY, BORDER_MEDIUM,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED,
    ORANGE_PRIMARY, GREEN_PROFIT, RED_LOSS, YELLOW_WARN,
    GOLD_DIAMOND, FONT_FAMILY_DISPLAY, FONT_FAMILY_UI,
)
from gui.widgets.pick_card import PickCard
from gui.widgets.gauge import ConfidenceGauge
from core.data_models import PickResult, PickTier


class StatCard(QFrame):
    """Small stat display card."""

    def __init__(self, title: str, value: str, color: str = ORANGE_PRIMARY, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            StatCard {{
                background-color: {BG_SECONDARY};
                border: 1px solid {BORDER_MEDIUM};
                border-radius: 8px;
                border-top: 2px solid {color};
            }}
        """)
        self.setFixedHeight(90)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(4)

        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(f"""
            color: {TEXT_SECONDARY};
            font-size: 11px;
            font-family: "{FONT_FAMILY_DISPLAY}";
            letter-spacing: 1px;
        """)
        layout.addWidget(title_lbl)

        self._value_lbl = QLabel(value)
        self._value_lbl.setStyleSheet(f"""
            color: {color};
            font-size: 24px;
            font-weight: bold;
            font-family: "{FONT_FAMILY_DISPLAY}";
        """)
        layout.addWidget(self._value_lbl)
        layout.addStretch()

    def set_value(self, value: str, color: str = None):
        self._value_lbl.setText(value)
        if color:
            self._value_lbl.setStyleSheet(f"""
                color: {color};
                font-size: 24px;
                font-weight: bold;
                font-family: "{FONT_FAMILY_DISPLAY}";
            """)


class DashboardPanel(QWidget):
    """Main dashboard showing today's picks and portfolio stats."""

    pick_selected = pyqtSignal(str)   # emits pick_id -> navigates to War Room
    run_analysis = pyqtSignal()       # request model run

    def __init__(self, parent=None):
        super().__init__(parent)
        self._picks: list[PickResult] = []
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(24, 20, 24, 20)
        main_layout.setSpacing(20)

        # ── Header ──────────────────────────────────────
        header = QHBoxLayout()

        title_area = QVBoxLayout()
        title = QLabel("DASHBOARD")
        title.setStyleSheet(f"""
            color: {TEXT_PRIMARY};
            font-size: 20px;
            font-weight: bold;
            font-family: "{FONT_FAMILY_DISPLAY}";
            letter-spacing: 2px;
        """)
        title_area.addWidget(title)

        date_str = datetime.now().strftime("%A, %B %d, %Y")
        date_lbl = QLabel(date_str)
        date_lbl.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px; font-family: '{FONT_FAMILY_DISPLAY}';")
        title_area.addWidget(date_lbl)

        header.addLayout(title_area)
        header.addStretch()

        # Run Analysis button
        self._run_btn = QPushButton("  RUN ANALYSIS")
        self._run_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {ORANGE_PRIMARY};
                color: #0D1117;
                border: none;
                border-radius: 6px;
                font-size: 13px;
                font-weight: bold;
                font-family: "{FONT_FAMILY_DISPLAY}";
                letter-spacing: 1px;
                padding: 10px 24px;
            }}
            QPushButton:hover {{ background-color: #FF6B2B; }}
            QPushButton:pressed {{ background-color: #C44500; }}
        """)
        self._run_btn.setFixedSize(200, 44)
        self._run_btn.clicked.connect(self.run_analysis.emit)
        header.addWidget(self._run_btn)

        main_layout.addLayout(header)

        # ── Status bar ──────────────────────────────────
        self._status_bar = QLabel("Ready — No analysis run yet")
        self._status_bar.setStyleSheet(f"""
            color: {TEXT_MUTED};
            font-size: 12px;
            font-family: "{FONT_FAMILY_DISPLAY}";
            background: {BG_TERTIARY};
            border-radius: 4px;
            padding: 8px 14px;
        """)
        main_layout.addWidget(self._status_bar)

        # ── Stats row ───────────────────────────────────
        stats_row = QHBoxLayout()
        stats_row.setSpacing(12)

        self._stat_total = StatCard("TODAY'S PICKS", "0", ORANGE_PRIMARY)
        self._stat_diamonds = StatCard("DIAMONDS", "0", GOLD_DIAMOND)
        self._stat_avg_ev = StatCard("AVG EV", "+0.000", GREEN_PROFIT)
        self._stat_avg_conf = StatCard("AVG CONF", "0%", ORANGE_PRIMARY)

        stats_row.addWidget(self._stat_total)
        stats_row.addWidget(self._stat_diamonds)
        stats_row.addWidget(self._stat_avg_ev)
        stats_row.addWidget(self._stat_avg_conf)

        main_layout.addLayout(stats_row)

        # ── Section header ──────────────────────────────
        picks_header = QHBoxLayout()
        picks_title = QLabel("TODAY'S PICKS")
        picks_title.setStyleSheet(f"""
            color: {TEXT_SECONDARY};
            font-size: 11px;
            font-weight: bold;
            font-family: "{FONT_FAMILY_DISPLAY}";
            letter-spacing: 2px;
        """)
        picks_header.addWidget(picks_title)
        picks_header.addStretch()

        self._pick_count = QLabel("0 picks")
        self._pick_count.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 10px; font-family: '{FONT_FAMILY_DISPLAY}';")
        picks_header.addWidget(self._pick_count)

        main_layout.addLayout(picks_header)

        # ── Picks scroll area ───────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"""
            QScrollArea {{
                background: transparent;
                border: none;
            }}
        """)

        self._picks_container = QWidget()
        self._picks_layout = QVBoxLayout(self._picks_container)
        self._picks_layout.setContentsMargins(0, 0, 0, 0)
        self._picks_layout.setSpacing(8)
        self._picks_layout.addStretch()

        scroll.setWidget(self._picks_container)
        main_layout.addWidget(scroll, 1)

        # ── Empty state ─────────────────────────────────
        self._empty_state = QLabel(
            "No picks yet.\nClick RUN ANALYSIS to generate today's predictions."
        )
        self._empty_state.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_state.setStyleSheet(f"""
            color: {TEXT_MUTED};
            font-size: 13px;
            font-family: "{FONT_FAMILY_DISPLAY}";
            padding: 60px;
        """)
        self._picks_layout.insertWidget(0, self._empty_state)

    # ── Public API ──────────────────────────────────────────

    def set_status(self, text: str, color: str = TEXT_MUTED):
        self._status_bar.setText(text)
        self._status_bar.setStyleSheet(f"""
            color: {color};
            font-size: 10px;
            font-family: "{FONT_FAMILY_DISPLAY}";
            background: {BG_TERTIARY};
            border-radius: 4px;
            padding: 6px 12px;
        """)

    def set_running(self, running: bool):
        self._run_btn.setEnabled(not running)
        if running:
            self._run_btn.setText("  RUNNING...")
            self.set_status("Analysis in progress...", YELLOW_WARN)
        else:
            self._run_btn.setText("  RUN ANALYSIS")

    def load_picks(self, picks: list[PickResult]):
        """Display a list of picks on the dashboard."""
        self._picks = picks

        # Clear existing cards
        while self._picks_layout.count() > 1:
            item = self._picks_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self._empty_state.setVisible(len(picks) == 0)

        if not picks:
            self._update_stats([])
            return

        # Sort: diamonds first, then by confidence desc
        sorted_picks = sorted(picks, key=lambda p: (
            0 if p.tier == PickTier.DIAMOND else
            1 if p.tier == PickTier.RLM_DIAMOND else
            2 if p.tier == PickTier.TOP3_FALLBACK else 3,
            -p.confidence,
        ))

        # Insert cards
        for i, pick in enumerate(sorted_picks):
            card = PickCard(pick)
            card.clicked.connect(self.pick_selected.emit)
            self._picks_layout.insertWidget(i, card)

        self._update_stats(picks)
        self._pick_count.setText(f"{len(picks)} picks")

        diamond_count = sum(1 for p in picks if p.is_diamond)
        if diamond_count > 0:
            self.set_status(
                f"Analysis complete — {diamond_count} DIAMOND picks found",
                GOLD_DIAMOND,
            )
        else:
            self.set_status(f"Analysis complete — {len(picks)} picks generated", GREEN_PROFIT)

    def _update_stats(self, picks: list[PickResult]):
        self._stat_total.set_value(str(len(picks)))
        diamonds = [p for p in picks if p.is_diamond]
        self._stat_diamonds.set_value(str(len(diamonds)))

        if picks:
            avg_ev = sum(p.ev for p in picks) / len(picks)
            avg_conf = sum(p.confidence for p in picks) / len(picks)
            ev_color = GREEN_PROFIT if avg_ev > 0 else RED_LOSS
            self._stat_avg_ev.set_value(f"{avg_ev:+.3f}", ev_color)
            self._stat_avg_conf.set_value(f"{avg_conf:.0%}")
        else:
            self._stat_avg_ev.set_value("+0.000")
            self._stat_avg_conf.set_value("0%")
