"""
SYNDICATE — War Room Panel (v2 — Multi-Tab Analysis)
======================================================
4 analysis tabs:
  1. Market Analysis — IA vs Market, Smart Money, gauges
  2. Statistical Breakdown — Top 20 feature importances per team
  3. Lineups & Injuries — Rosters with injury tooltips
  4. Breaking News — Headlines with source links
Plus: Expert notes and approve/reject actions.
"""
from __future__ import annotations
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QScrollArea, QSizePolicy, QGridLayout,
    QTabWidget, QToolTip, QTableWidget, QTableWidgetItem,
    QHeaderView, QGroupBox,
)
from PyQt6.QtCore import Qt, pyqtSignal, QUrl
from PyQt6.QtGui import QFont, QDesktopServices, QColor, QCursor

from gui.theme import (
    BG_PRIMARY, BG_SECONDARY, BG_TERTIARY, BG_HOVER, BORDER_MEDIUM,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED,
    ORANGE_PRIMARY, GREEN_PROFIT, RED_LOSS, YELLOW_WARN,
    GOLD_DIAMOND, PURPLE_RLM, BLUE_INFO,
    GREEN_BG, RED_BG, BLUE_BG,
    FONT_FAMILY_DISPLAY, FONT_FAMILY_UI,
)
from gui.widgets.gauge import ConfidenceGauge
from core.data_models import PickResult, PickTier, PickStatus, RiskLevel


# ─────────────────────────────────────────────────────────────────────────────
# Reusable team data card
# ─────────────────────────────────────────────────────────────────────────────
class TeamDataCard(QFrame):
    """A bordered card showing key-value data for one team."""

    def __init__(self, title: str, color: str = ORANGE_PRIMARY, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            TeamDataCard {{
                background-color: {BG_SECONDARY};
                border: 1px solid {BORDER_MEDIUM};
                border-radius: 8px;
                border-top: 2px solid {color};
            }}
        """)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(14, 10, 14, 12)
        self._layout.setSpacing(6)

        header = QLabel(title)
        header.setStyleSheet(f"""
            color: {color};
            font-size: 13px;
            font-weight: bold;
            font-family: '{FONT_FAMILY_DISPLAY}';
            letter-spacing: 1px;
        """)
        self._layout.addWidget(header)

        self._rows_layout = QVBoxLayout()
        self._rows_layout.setSpacing(3)
        self._layout.addLayout(self._rows_layout)

    def add_row(self, label: str, value: str, value_color: str = TEXT_PRIMARY):
        row = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px; font-family: '{FONT_FAMILY_DISPLAY}';")
        row.addWidget(lbl)
        row.addStretch()
        val = QLabel(value)
        val.setStyleSheet(f"color: {value_color}; font-size: 12px; font-weight: bold; font-family: '{FONT_FAMILY_DISPLAY}';")
        row.addWidget(val)
        self._rows_layout.addLayout(row)

    def add_separator(self):
        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background-color: {BORDER_MEDIUM};")
        self._rows_layout.addWidget(sep)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1: Market Analysis
# ─────────────────────────────────────────────────────────────────────────────
class MarketAnalysisTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(8, 8, 8, 8)
        self._layout.setSpacing(12)

        # Gauges row
        gauges = QHBoxLayout()
        gauges.setSpacing(20)
        self._gauge_conf = ConfidenceGauge(size=140, label="CONFIDENCE")
        self._gauge_ev = ConfidenceGauge(size=140, label="EDGE STRENGTH")
        gauges.addStretch()
        gauges.addWidget(self._gauge_conf)
        gauges.addWidget(self._gauge_ev)
        gauges.addStretch()
        self._layout.addLayout(gauges)

        # Team comparison cards
        cards_row = QHBoxLayout()
        cards_row.setSpacing(12)
        self._home_card = TeamDataCard("HOME", ORANGE_PRIMARY)
        self._away_card = TeamDataCard("AWAY", BLUE_INFO)
        cards_row.addWidget(self._home_card)
        cards_row.addWidget(self._away_card)
        self._layout.addLayout(cards_row)

        self._layout.addStretch()

    def load(self, pick: PickResult):
        self._gauge_conf.set_value(pick.confidence)
        ev_norm = min(max((pick.ev + 0.1) / 0.5, 0), 1.0)
        self._gauge_ev.set_value(ev_norm)

        # Rebuild home/away cards
        self._home_card.deleteLater()
        self._away_card.deleteLater()
        self._home_card = TeamDataCard(f"HOME — {pick.home_team}", ORANGE_PRIMARY)
        self._away_card = TeamDataCard(f"AWAY — {pick.away_team}", BLUE_INFO)

        ha = pick.home_analysis
        aa = pick.away_analysis
        sm = pick.smart_money

        if ha:
            self._home_card.add_row("Off Rating", f"{ha.off_rtg:.1f}", GREEN_PROFIT if ha.off_rtg > 110 else TEXT_PRIMARY)
            self._home_card.add_row("Def Rating", f"{ha.def_rtg:.1f}", GREEN_PROFIT if ha.def_rtg < 110 else RED_LOSS)
            self._home_card.add_row("Net Rating", f"{ha.net_rtg:+.1f}", GREEN_PROFIT if ha.net_rtg > 0 else RED_LOSS)
            self._home_card.add_row("Pace", f"{ha.pace:.1f}")
            self._home_card.add_separator()
            self._home_card.add_row("B2B", "YES" if ha.fatigue_b2b else "No", RED_LOSS if ha.fatigue_b2b else GREEN_PROFIT)
            self._home_card.add_row("Heavy Legs", "YES" if ha.fatigue_heavy_legs else "No", RED_LOSS if ha.fatigue_heavy_legs else GREEN_PROFIT)
            self._home_card.add_row("Missing Stars", f"{ha.missing_stars:.0%}", RED_LOSS if ha.missing_stars > 0.3 else TEXT_PRIMARY)
            self._home_card.add_row("Clutch Q4", f"{ha.q4_clutch:+.2f}")

        if aa:
            self._away_card.add_row("Off Rating", f"{aa.off_rtg:.1f}", GREEN_PROFIT if aa.off_rtg > 110 else TEXT_PRIMARY)
            self._away_card.add_row("Def Rating", f"{aa.def_rtg:.1f}", GREEN_PROFIT if aa.def_rtg < 110 else RED_LOSS)
            self._away_card.add_row("Net Rating", f"{aa.net_rtg:+.1f}", GREEN_PROFIT if aa.net_rtg > 0 else RED_LOSS)
            self._away_card.add_row("Pace", f"{aa.pace:.1f}")
            self._away_card.add_separator()
            self._away_card.add_row("B2B", "YES" if aa.fatigue_b2b else "No", RED_LOSS if aa.fatigue_b2b else GREEN_PROFIT)
            self._away_card.add_row("Heavy Legs", "YES" if aa.fatigue_heavy_legs else "No", RED_LOSS if aa.fatigue_heavy_legs else GREEN_PROFIT)
            self._away_card.add_row("Missing Stars", f"{aa.missing_stars:.0%}", RED_LOSS if aa.missing_stars > 0.3 else TEXT_PRIMARY)
            self._away_card.add_row("Clutch Q4", f"{aa.q4_clutch:+.2f}")

        if sm:
            self._home_card.add_separator()
            self._home_card.add_row("Mkt Prob Home", f"{sm.mkt_prob_home:.1%}")
            self._home_card.add_row("Spread", f"{sm.mkt_spread:+.1f}")
            self._home_card.add_row("RLM", sm.rlm_label)
            self._home_card.add_row("Bookmakers", str(sm.n_bookmakers))

        cards = self._layout.itemAt(1)
        if cards and cards.layout():
            cards.layout().addWidget(self._home_card)
            cards.layout().addWidget(self._away_card)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2: Statistical Breakdown — Top 20 features
# ─────────────────────────────────────────────────────────────────────────────
class StatsBreakdownTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        desc = QLabel(
            "Top 20 feature values from the model's analysis. "
            "These are the variables that most influenced the prediction."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px; font-family: '{FONT_FAMILY_DISPLAY}';")
        layout.addWidget(desc)

        # Side by side tables
        tables_row = QHBoxLayout()
        tables_row.setSpacing(12)

        self._home_table = self._make_table()
        self._away_table = self._make_table()

        home_group = QGroupBox("HOME TEAM FACTORS")
        home_layout = QVBoxLayout(home_group)
        home_layout.addWidget(self._home_table)
        tables_row.addWidget(home_group)

        away_group = QGroupBox("AWAY TEAM FACTORS")
        away_layout = QVBoxLayout(away_group)
        away_layout.addWidget(self._away_table)
        tables_row.addWidget(away_group)

        layout.addLayout(tables_row, 1)

    def _make_table(self) -> QTableWidget:
        t = QTableWidget()
        t.setColumnCount(3)
        t.setHorizontalHeaderLabels(["Feature", "Value", "Impact"])
        t.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        t.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        t.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        t.setAlternatingRowColors(True)
        t.verticalHeader().setVisible(False)
        t.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        return t

    def load(self, pick: PickResult, feature_names: list):
        vec = pick.feature_vector
        if not vec or not feature_names:
            return

        # Split into home and away features
        home_feats = [(n, vec[i]) for i, n in enumerate(feature_names) if n.startswith("h_")]
        away_feats = [(n, vec[i]) for i, n in enumerate(feature_names) if n.startswith("a_")]
        shared_feats = [(n, vec[i]) for i, n in enumerate(feature_names) if not n.startswith("h_") and not n.startswith("a_")]

        # Sort by absolute value (most impactful first)
        home_feats.sort(key=lambda x: abs(x[1]), reverse=True)
        away_feats.sort(key=lambda x: abs(x[1]), reverse=True)

        self._fill_table(self._home_table, home_feats[:20] + shared_feats[:5])
        self._fill_table(self._away_table, away_feats[:20] + shared_feats[:5])

    def _fill_table(self, table: QTableWidget, features: list):
        table.setRowCount(len(features))
        for i, (name, value) in enumerate(features):
            # Clean name
            clean = name.replace("h_", "").replace("a_", "").replace("_", " ").title()
            table.setItem(i, 0, QTableWidgetItem(clean))
            table.setItem(i, 1, QTableWidgetItem(f"{value:.4f}"))

            # Impact bar
            impact = min(abs(value) * 100, 100)
            bar = "█" * max(1, int(impact / 5))
            color = GREEN_PROFIT if value > 0 else RED_LOSS if value < 0 else TEXT_MUTED
            item = QTableWidgetItem(bar)
            item.setForeground(QColor(color))
            table.setItem(i, 2, item)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3: Lineups & Injuries
# ─────────────────────────────────────────────────────────────────────────────
class PlayerLabel(QLabel):
    """Player name label with injury tooltip."""
    def __init__(self, name: str, status: str = "available", detail: str = "", parent=None):
        super().__init__(parent)
        self.setText(name)
        self.setCursor(QCursor(Qt.CursorShape.WhatsThisCursor))

        if status == "out":
            self.setStyleSheet(f"""
                color: {RED_LOSS};
                font-size: 12px;
                font-family: '{FONT_FAMILY_DISPLAY}';
                text-decoration: line-through;
                padding: 4px 8px;
                background: {RED_LOSS}12;
                border-radius: 4px;
            """)
            self.setToolTip(f"OUT — {detail or 'Unavailable'}")
        elif status == "questionable":
            self.setStyleSheet(f"""
                color: {YELLOW_WARN};
                font-size: 12px;
                font-family: '{FONT_FAMILY_DISPLAY}';
                padding: 4px 8px;
                background: {YELLOW_WARN}12;
                border-radius: 4px;
            """)
            self.setToolTip(f"QUESTIONABLE — {detail or 'Game-time decision'}")
        else:
            self.setStyleSheet(f"""
                color: {TEXT_PRIMARY};
                font-size: 12px;
                font-family: '{FONT_FAMILY_DISPLAY}';
                padding: 4px 8px;
            """)
            self.setToolTip("Available")


class LineupsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        desc = QLabel("Hover over players to see availability status and injury details.")
        desc.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px; font-family: '{FONT_FAMILY_DISPLAY}';")
        layout.addWidget(desc)

        # Legend
        legend = QHBoxLayout()
        for text, color in [("Available", TEXT_PRIMARY), ("Questionable", YELLOW_WARN), ("Out", RED_LOSS)]:
            dot = QLabel(f"● {text}")
            dot.setStyleSheet(f"color: {color}; font-size: 10px; font-family: '{FONT_FAMILY_DISPLAY}';")
            legend.addWidget(dot)
        legend.addStretch()
        layout.addLayout(legend)

        # Two columns
        cols = QHBoxLayout()
        cols.setSpacing(16)

        self._home_col = QVBoxLayout()
        self._away_col = QVBoxLayout()

        self._home_header = QLabel("HOME")
        self._home_header.setStyleSheet(f"color: {ORANGE_PRIMARY}; font-size: 14px; font-weight: bold; font-family: '{FONT_FAMILY_DISPLAY}';")
        self._home_col.addWidget(self._home_header)

        self._away_header = QLabel("AWAY")
        self._away_header.setStyleSheet(f"color: {BLUE_INFO}; font-size: 14px; font-weight: bold; font-family: '{FONT_FAMILY_DISPLAY}';")
        self._away_col.addWidget(self._away_header)

        self._home_scroll = QScrollArea()
        self._home_scroll.setWidgetResizable(True)
        self._home_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        self._home_container = QWidget()
        self._home_players = QVBoxLayout(self._home_container)
        self._home_players.setSpacing(2)
        self._home_scroll.setWidget(self._home_container)
        self._home_col.addWidget(self._home_scroll)

        self._away_scroll = QScrollArea()
        self._away_scroll.setWidgetResizable(True)
        self._away_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        self._away_container = QWidget()
        self._away_players = QVBoxLayout(self._away_container)
        self._away_players.setSpacing(2)
        self._away_scroll.setWidget(self._away_container)
        self._away_col.addWidget(self._away_scroll)

        cols.addLayout(self._home_col)
        cols.addLayout(self._away_col)
        layout.addLayout(cols, 1)

    def load(self, pick: PickResult, home_roster: list = None, away_roster: list = None):
        self._home_header.setText(f"HOME — {pick.home_team}")
        self._away_header.setText(f"AWAY — {pick.away_team}")

        # Clear old
        self._clear_layout(self._home_players)
        self._clear_layout(self._away_players)

        if not home_roster:
            home_roster = [{"name": "Roster data not available", "status": "available", "detail": ""}]
        if not away_roster:
            away_roster = [{"name": "Roster data not available", "status": "available", "detail": ""}]

        for p in home_roster:
            lbl = PlayerLabel(p["name"], p.get("status", "available"), p.get("detail", ""))
            self._home_players.addWidget(lbl)
        self._home_players.addStretch()

        for p in away_roster:
            lbl = PlayerLabel(p["name"], p.get("status", "available"), p.get("detail", ""))
            self._away_players.addWidget(lbl)
        self._away_players.addStretch()

    def _clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()


# ─────────────────────────────────────────────────────────────────────────────
# Tab 4: Breaking News
# ─────────────────────────────────────────────────────────────────────────────
class NewsItem(QFrame):
    """A single news headline with source link."""
    def __init__(self, headline: str, source: str, url: str, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            NewsItem {{
                background: {BG_SECONDARY};
                border: 1px solid {BORDER_MEDIUM};
                border-radius: 6px;
            }}
            NewsItem:hover {{
                border-color: {BLUE_INFO}44;
            }}
        """)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._url = url

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)

        title = QLabel(headline)
        title.setWordWrap(True)
        title.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 12px; font-weight: bold; font-family: '{FONT_FAMILY_UI}';")
        layout.addWidget(title)

        src = QLabel(f"Source: {source}")
        src.setStyleSheet(f"color: {BLUE_INFO}; font-size: 10px; font-family: '{FONT_FAMILY_DISPLAY}';")
        layout.addWidget(src)

    def mousePressEvent(self, event):
        if self._url and event.button() == Qt.MouseButton.LeftButton:
            QDesktopServices.openUrl(QUrl(self._url))
        super().mousePressEvent(event)


class NewsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        desc = QLabel("Latest news about the teams in this matchup. Click to open in browser.")
        desc.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px; font-family: '{FONT_FAMILY_DISPLAY}';")
        layout.addWidget(desc)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        self._container = QWidget()
        self._news_layout = QVBoxLayout(self._container)
        self._news_layout.setSpacing(6)
        self._news_layout.addStretch()
        scroll.setWidget(self._container)
        layout.addWidget(scroll, 1)

    def load(self, news_items: list = None):
        # Clear
        while self._news_layout.count() > 1:
            item = self._news_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not news_items:
            news_items = [{
                "headline": "No news available for this matchup yet.",
                "source": "System",
                "url": "",
            }]

        for i, n in enumerate(news_items):
            item = NewsItem(n["headline"], n.get("source", ""), n.get("url", ""))
            self._news_layout.insertWidget(i, item)


# ─────────────────────────────────────────────────────────────────────────────
# Main War Room Panel
# ─────────────────────────────────────────────────────────────────────────────
class WarRoomPanel(QWidget):
    """Multi-tab analysis panel (The War Room)."""

    pick_approved = pyqtSignal(str, str)
    pick_rejected = pyqtSignal(str, str)
    back_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_pick: PickResult | None = None
        self._feature_names: list = []
        self._setup_ui()

    def _setup_ui(self):
        main = QVBoxLayout(self)
        main.setContentsMargins(20, 16, 20, 16)
        main.setSpacing(12)

        # ── Header ──────────────────────────────────────
        header = QHBoxLayout()

        back_btn = QPushButton("  Back to Dashboard")
        back_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent; color: {TEXT_SECONDARY}; border: none;
                font-size: 12px; font-family: '{FONT_FAMILY_DISPLAY}'; padding: 4px 8px;
            }}
            QPushButton:hover {{ color: {ORANGE_PRIMARY}; }}
        """)
        back_btn.clicked.connect(self.back_requested.emit)
        header.addWidget(back_btn)
        header.addStretch()

        self._title = QLabel("WAR ROOM")
        self._title.setStyleSheet(f"""
            color: {TEXT_PRIMARY}; font-size: 20px; font-weight: bold;
            font-family: '{FONT_FAMILY_DISPLAY}'; letter-spacing: 2px;
        """)
        header.addWidget(self._title)
        header.addStretch()

        self._tier_badge = QLabel("")
        header.addWidget(self._tier_badge)

        main.addLayout(header)

        # ── Matchup summary bar ─────────────────────────
        self._summary = QFrame()
        self._summary.setStyleSheet(f"""
            QFrame {{
                background: {BG_SECONDARY}; border: 1px solid {BORDER_MEDIUM};
                border-radius: 8px;
            }}
        """)
        summary_layout = QHBoxLayout(self._summary)
        summary_layout.setContentsMargins(20, 12, 20, 12)
        summary_layout.setSpacing(20)

        self._matchup_lbl = QLabel("--- @ ---")
        self._matchup_lbl.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 20px; font-weight: bold; font-family: '{FONT_FAMILY_DISPLAY}';")
        summary_layout.addWidget(self._matchup_lbl)

        self._pick_lbl = QLabel("PICK: ---")
        self._pick_lbl.setStyleSheet(f"color: {ORANGE_PRIMARY}; font-size: 16px; font-weight: bold; font-family: '{FONT_FAMILY_DISPLAY}';")
        summary_layout.addWidget(self._pick_lbl)

        summary_layout.addStretch()

        for key, label in [("conf", "Conf"), ("ev", "EV"), ("risk", "Risk")]:
            col = QVBoxLayout()
            col.setSpacing(0)
            name = QLabel(label)
            name.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 9px; font-family: '{FONT_FAMILY_DISPLAY}';")
            val = QLabel("---")
            val.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 14px; font-weight: bold; font-family: '{FONT_FAMILY_DISPLAY}';")
            col.addWidget(name)
            col.addWidget(val)
            summary_layout.addLayout(col)
            if not hasattr(self, '_summary_vals'):
                self._summary_vals = {}
            self._summary_vals[key] = val

        main.addWidget(self._summary)

        # ── Analysis Tabs ───────────────────────────────
        self._tabs = QTabWidget()
        self._tabs.setStyleSheet(f"""
            QTabBar::tab {{ padding: 10px 24px; font-size: 12px; font-family: '{FONT_FAMILY_DISPLAY}'; }}
        """)

        self._market_tab = MarketAnalysisTab()
        self._stats_tab = StatsBreakdownTab()
        self._lineups_tab = LineupsTab()
        self._news_tab = NewsTab()

        self._tabs.addTab(self._market_tab, "Market Analysis")
        self._tabs.addTab(self._stats_tab, "Stats Top-20")
        self._tabs.addTab(self._lineups_tab, "Lineups & Injuries")
        self._tabs.addTab(self._news_tab, "Breaking News")

        main.addWidget(self._tabs, 1)

        # ── Action Buttons ────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(8, 6, 8, 6)
        btn_row.addStretch()

        self._reject_btn = QPushButton("  REJECT PICK")
        self._reject_btn.setMinimumHeight(38)
        self._reject_btn.setStyleSheet(f"""
            QPushButton {{
                background: {RED_BG}; color: {RED_LOSS}; border: 1px solid {RED_LOSS}44;
                border-radius: 6px; padding: 8px 24px; font-weight: bold; font-family: '{FONT_FAMILY_DISPLAY}'; font-size: 12px;
            }}
            QPushButton:hover {{ background: #5A1A1A; border-color: {RED_LOSS}; }}
        """)
        self._reject_btn.clicked.connect(self._on_reject)
        btn_row.addWidget(self._reject_btn)

        self._approve_btn = QPushButton("  APPROVE PICK")
        self._approve_btn.setMinimumHeight(38)
        self._approve_btn.setStyleSheet(f"""
            QPushButton {{
                background: {GREEN_BG}; color: {GREEN_PROFIT}; border: 1px solid {GREEN_PROFIT}44;
                border-radius: 6px; padding: 8px 24px; font-weight: bold; font-family: '{FONT_FAMILY_DISPLAY}'; font-size: 12px;
            }}
            QPushButton:hover {{ background: #1A4A27; border-color: {GREEN_PROFIT}; }}
        """)
        self._approve_btn.clicked.connect(self._on_approve)
        btn_row.addWidget(self._approve_btn)

        main.addLayout(btn_row)

    # ── Public API ──────────────────────────────────────────

    def set_feature_names(self, names: list):
        self._feature_names = names

    def load_pick(self, pick: PickResult, home_roster=None, away_roster=None, news=None):
        self._current_pick = pick

        # Header
        tier_map = {
            PickTier.DIAMOND: ("DIAMOND", GOLD_DIAMOND),
            PickTier.RLM_DIAMOND: ("RLM DIAMOND", PURPLE_RLM),
            PickTier.TOP3_FALLBACK: ("TOP-3", BLUE_INFO),
            PickTier.STANDARD: ("STANDARD", TEXT_MUTED),
        }
        tt, tc = tier_map.get(pick.tier, ("", TEXT_MUTED))
        self._tier_badge.setText(tt)
        self._tier_badge.setStyleSheet(f"""
            background: {tc}22; color: {tc}; border: 1px solid {tc}44;
            border-radius: 4px; padding: 4px 12px; font-size: 11px;
            font-weight: bold; font-family: '{FONT_FAMILY_DISPLAY}';
        """)

        # Summary
        self._matchup_lbl.setText(pick.matchup)
        self._pick_lbl.setText(f"PICK: {pick.pick}")

        conf_color = GREEN_PROFIT if pick.confidence >= 0.78 else YELLOW_WARN
        self._summary_vals["conf"].setText(f"{pick.confidence:.1%}")
        self._summary_vals["conf"].setStyleSheet(f"color: {conf_color}; font-size: 14px; font-weight: bold; font-family: '{FONT_FAMILY_DISPLAY}';")

        ev_color = GREEN_PROFIT if pick.ev > 0 else RED_LOSS
        self._summary_vals["ev"].setText(f"{pick.ev:+.3f}")
        self._summary_vals["ev"].setStyleSheet(f"color: {ev_color}; font-size: 14px; font-weight: bold; font-family: '{FONT_FAMILY_DISPLAY}';")

        risk_c = {RiskLevel.LOW: GREEN_PROFIT, RiskLevel.MEDIUM: YELLOW_WARN, RiskLevel.HIGH: RED_LOSS}
        self._summary_vals["risk"].setText(pick.risk_level.value)
        self._summary_vals["risk"].setStyleSheet(f"color: {risk_c.get(pick.risk_level, TEXT_PRIMARY)}; font-size: 14px; font-weight: bold; font-family: '{FONT_FAMILY_DISPLAY}';")

        # Load tabs
        self._market_tab.load(pick)
        self._stats_tab.load(pick, self._feature_names)
        self._lineups_tab.load(pick, home_roster, away_roster)
        self._news_tab.load(news)

        self._tabs.setCurrentIndex(0)

    def _on_approve(self):
        if self._current_pick:
            self.pick_approved.emit(self._current_pick.pick_id, "")

    def _on_reject(self):
        if self._current_pick:
            self.pick_rejected.emit(self._current_pick.pick_id, "")
