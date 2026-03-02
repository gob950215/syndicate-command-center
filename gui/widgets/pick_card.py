"""
SYNDICATE — Pick Card Widget
==============================
Clickable card displaying a single prediction pick.
Shows bet type (Moneyline/Spread/Total), tier, and signals.
"""
from __future__ import annotations
from PyQt6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QLabel, QSizePolicy,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QCursor

from core.data_models import PickResult, PickTier, RiskLevel
from gui.theme import (
    BG_SECONDARY, BG_HOVER, BORDER_MEDIUM,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED,
    ORANGE_PRIMARY, GREEN_PROFIT, RED_LOSS, YELLOW_WARN,
    GOLD_DIAMOND, PURPLE_RLM, BLUE_INFO,
    FONT_FAMILY_DISPLAY,
    RISK_LOW, RISK_MEDIUM, RISK_HIGH,
)

BET_TYPE_COLORS = {
    "Moneyline": "#58A6FF",
    "Spread": "#BC8CFF",
    "Total": "#3FB950",
}


class PickCard(QFrame):
    """Interactive card for a single pick. Emits clicked(pick_id)."""

    clicked = pyqtSignal(str)

    def __init__(self, pick: PickResult, parent=None):
        super().__init__(parent)
        self.pick = pick
        self._setup_ui()
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setFixedHeight(120)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def _setup_ui(self):
        p = self.pick
        is_diamond = p.is_diamond

        border_color = (
            GOLD_DIAMOND if p.tier == PickTier.DIAMOND else
            PURPLE_RLM if p.tier == PickTier.RLM_DIAMOND else
            BLUE_INFO if p.tier == PickTier.TOP3_FALLBACK else BORDER_MEDIUM
        )
        border_alpha = "88" if is_diamond else "44"

        self.setStyleSheet(f"""
            PickCard {{
                background-color: {BG_SECONDARY};
                border: 1px solid {border_color}{border_alpha};
                border-radius: 8px;
                border-left: 3px solid {border_color};
            }}
            PickCard:hover {{
                background-color: {BG_HOVER};
                border-color: {border_color};
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(7)

        # ── Row 1: Tier badge + Bet Type + Matchup + Date ──
        row1 = QHBoxLayout()
        row1.setSpacing(8)

        tier_map = {
            PickTier.DIAMOND: ("DIAMOND", GOLD_DIAMOND),
            PickTier.RLM_DIAMOND: ("RLM", PURPLE_RLM),
            PickTier.TOP3_FALLBACK: ("TOP-3", BLUE_INFO),
            PickTier.STANDARD: ("", TEXT_MUTED),
        }
        tier_text, tier_color = tier_map.get(p.tier, ("", TEXT_MUTED))

        if tier_text:
            badge = QLabel(tier_text)
            badge.setStyleSheet(f"""
                background-color: {tier_color}22;
                color: {tier_color};
                border: 1px solid {tier_color}44;
                border-radius: 4px;
                padding: 2px 10px;
                font-size: 10px;
                font-weight: bold;
                font-family: '{FONT_FAMILY_DISPLAY}';
            """)
            badge.setFixedHeight(22)
            row1.addWidget(badge)

        # Bet type badge
        bt_color = BET_TYPE_COLORS.get(p.bet_type, BLUE_INFO)
        bt_badge = QLabel(p.bet_type.upper())
        bt_badge.setStyleSheet(f"""
            background-color: {bt_color}18;
            color: {bt_color};
            border: 1px solid {bt_color}40;
            border-radius: 4px;
            padding: 2px 8px;
            font-size: 9px;
            font-weight: bold;
            font-family: '{FONT_FAMILY_DISPLAY}';
        """)
        bt_badge.setFixedHeight(22)
        row1.addWidget(bt_badge)

        matchup = QLabel(p.matchup)
        matchup.setStyleSheet(f"""
            color: {TEXT_PRIMARY};
            font-size: 15px;
            font-weight: bold;
            font-family: '{FONT_FAMILY_DISPLAY}';
        """)
        row1.addWidget(matchup)
        row1.addStretch()

        date_lbl = QLabel(p.date)
        date_lbl.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 10px; font-family: '{FONT_FAMILY_DISPLAY}';")
        row1.addWidget(date_lbl)

        layout.addLayout(row1)

        # ── Row 2: Pick + Confidence + EV + Risk ──
        row2 = QHBoxLayout()
        row2.setSpacing(18)

        pick_lbl = QLabel(f"PICK: {p.pick}")
        pick_lbl.setStyleSheet(f"""
            color: {ORANGE_PRIMARY};
            font-size: 14px;
            font-weight: bold;
            font-family: '{FONT_FAMILY_DISPLAY}';
        """)
        row2.addWidget(pick_lbl)

        conf_color = GREEN_PROFIT if p.confidence >= 0.78 else YELLOW_WARN if p.confidence >= 0.65 else TEXT_SECONDARY
        conf_lbl = QLabel(f"Conf: {p.confidence:.1%}")
        conf_lbl.setStyleSheet(f"color: {conf_color}; font-family: '{FONT_FAMILY_DISPLAY}'; font-size: 12px;")
        row2.addWidget(conf_lbl)

        ev_color = GREEN_PROFIT if p.ev > 0 else RED_LOSS
        ev_lbl = QLabel(f"EV: {p.ev:+.3f}")
        ev_lbl.setStyleSheet(f"color: {ev_color}; font-family: '{FONT_FAMILY_DISPLAY}'; font-size: 12px;")
        row2.addWidget(ev_lbl)

        row2.addStretch()

        risk_map = {
            RiskLevel.LOW: ("LOW", RISK_LOW),
            RiskLevel.MEDIUM: ("MED", RISK_MEDIUM),
            RiskLevel.HIGH: ("HIGH", RISK_HIGH),
        }
        risk_text, risk_color = risk_map.get(p.risk_level, ("MED", RISK_MEDIUM))
        risk_badge = QLabel(f"Risk: {risk_text}")
        risk_badge.setStyleSheet(f"""
            color: {risk_color};
            font-size: 10px;
            font-family: '{FONT_FAMILY_DISPLAY}';
            font-weight: bold;
        """)
        row2.addWidget(risk_badge)

        layout.addLayout(row2)

        # ── Row 3: Signals strip ──
        row3 = QHBoxLayout()
        row3.setSpacing(6)

        signals = []
        if p.rlm == 1:
            signals.append(("RLM OK", GREEN_PROFIT))
        elif p.rlm == -1:
            signals.append(("RLM REV", RED_LOSS))
        if p.fatigue_trap:
            signals.append(("FATIGUE", YELLOW_WARN))
        if p.value_trap:
            signals.append(("V-TRAP", YELLOW_WARN))
        if p.playoff_urgency:
            signals.append(("URGENCY", PURPLE_RLM))

        edge = p.mkt_gap
        edge_color = GREEN_PROFIT if edge > 0.05 else YELLOW_WARN if edge > 0 else RED_LOSS
        signals.append((f"Edge: {edge:+.1%}", edge_color))

        for text, color in signals:
            sig = QLabel(text)
            sig.setStyleSheet(f"""
                color: {color};
                font-size: 9px;
                font-family: '{FONT_FAMILY_DISPLAY}';
                background: {color}15;
                border-radius: 3px;
                padding: 2px 6px;
            """)
            row3.addWidget(sig)

        row3.addStretch()
        layout.addLayout(row3)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.pick.pick_id)
        super().mousePressEvent(event)
