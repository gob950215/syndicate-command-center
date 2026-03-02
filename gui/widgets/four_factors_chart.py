"""
SYNDICATE — Four Factors Comparative Bar Chart
================================================
Horizontal bar chart comparing home vs away Four Factors.
"""
from __future__ import annotations
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QPainter, QPen, QColor, QFont, QLinearGradient

from gui.theme import (
    BG_SECONDARY, BG_TERTIARY, BORDER_MEDIUM,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED,
    ORANGE_PRIMARY, BLUE_INFO, FONT_FAMILY_DISPLAY,
)


class FourFactorsChart(QWidget):
    """
    Horizontal grouped bar chart: Home (orange) vs Away (blue).
    Factors: eFG%, TOV%, OREB%, FT Rate
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._home_data = {}
        self._away_data = {}
        self._home_label = "HOME"
        self._away_label = "AWAY"
        self.setMinimumSize(320, 200)

    def set_data(
        self,
        home: dict,
        away: dict,
        home_label: str = "HOME",
        away_label: str = "AWAY"
    ):
        """
        Set chart data.
        Expects: {"eFG%": 0.54, "TOV%": 0.12, "OREB%": 0.25, "FT Rate": 0.22}
        """
        self._home_data = home
        self._away_data = away
        self._home_label = home_label
        self._away_label = away_label
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()

        # Background
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(BG_SECONDARY))
        painter.drawRoundedRect(0, 0, w, h, 8, 8)

        if not self._home_data:
            painter.setPen(QColor(TEXT_MUTED))
            painter.setFont(QFont(FONT_FAMILY_DISPLAY, 10))
            painter.drawText(QRectF(0, 0, w, h), Qt.AlignmentFlag.AlignCenter, "No data")
            painter.end()
            return

        factors = list(self._home_data.keys())
        n = len(factors)
        if n == 0:
            painter.end()
            return

        # Layout
        margin_top = 36
        margin_bottom = 16
        margin_left = 70
        margin_right = 50
        chart_w = w - margin_left - margin_right
        chart_h = h - margin_top - margin_bottom
        bar_group_h = chart_h / n
        bar_h = bar_group_h * 0.30
        gap = bar_group_h * 0.10

        # Title
        painter.setPen(QColor(TEXT_SECONDARY))
        painter.setFont(QFont(FONT_FAMILY_DISPLAY, 9, QFont.Weight.Bold))
        painter.drawText(QRectF(margin_left, 4, chart_w, 24), Qt.AlignmentFlag.AlignCenter, "FOUR FACTORS COMPARISON")

        # Legend
        legend_y = 10
        # Home
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(ORANGE_PRIMARY))
        painter.drawRoundedRect(int(w - margin_right - 120), legend_y, 10, 10, 2, 2)
        painter.setPen(QColor(TEXT_SECONDARY))
        painter.setFont(QFont(FONT_FAMILY_DISPLAY, 7))
        painter.drawText(w - margin_right - 106, legend_y + 9, self._home_label)
        # Away
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(BLUE_INFO))
        painter.drawRoundedRect(int(w - margin_right - 56), legend_y, 10, 10, 2, 2)
        painter.setPen(QColor(TEXT_SECONDARY))
        painter.drawText(w - margin_right - 42, legend_y + 9, self._away_label)

        # Draw bars for each factor
        # Determine max for normalization
        all_vals = list(self._home_data.values()) + list(self._away_data.values())
        max_val = max(all_vals) if all_vals else 1.0
        if max_val <= 0:
            max_val = 1.0

        for i, factor in enumerate(factors):
            hv = self._home_data.get(factor, 0)
            av = self._away_data.get(factor, 0)

            y_base = margin_top + i * bar_group_h
            y_home = y_base + gap
            y_away = y_home + bar_h + gap * 0.5

            # Factor label
            painter.setPen(QColor(TEXT_SECONDARY))
            painter.setFont(QFont(FONT_FAMILY_DISPLAY, 8))
            painter.drawText(
                QRectF(4, y_base, margin_left - 8, bar_group_h),
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
                factor
            )

            # Home bar
            bw_home = max(2, (hv / max_val) * chart_w) if max_val > 0 else 2
            grad_home = QLinearGradient(margin_left, 0, margin_left + bw_home, 0)
            grad_home.setColorAt(0, QColor(ORANGE_PRIMARY))
            grad_home.setColorAt(1, QColor("#FF8C42"))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(grad_home)
            painter.drawRoundedRect(QRectF(margin_left, y_home, bw_home, bar_h), 3, 3)

            # Home value
            painter.setPen(QColor(TEXT_PRIMARY))
            painter.setFont(QFont(FONT_FAMILY_DISPLAY, 7, QFont.Weight.Bold))
            painter.drawText(int(margin_left + bw_home + 4), int(y_home + bar_h - 2), f"{hv:.3f}")

            # Away bar
            bw_away = max(2, (av / max_val) * chart_w) if max_val > 0 else 2
            grad_away = QLinearGradient(margin_left, 0, margin_left + bw_away, 0)
            grad_away.setColorAt(0, QColor(BLUE_INFO))
            grad_away.setColorAt(1, QColor("#7EC8FF"))
            painter.setBrush(grad_away)
            painter.drawRoundedRect(QRectF(margin_left, y_away, bw_away, bar_h), 3, 3)

            # Away value
            painter.setPen(QColor(TEXT_PRIMARY))
            painter.drawText(int(margin_left + bw_away + 4), int(y_away + bar_h - 2), f"{av:.3f}")

            # Separator line
            if i < n - 1:
                painter.setPen(QPen(QColor(BORDER_MEDIUM), 1, Qt.PenStyle.DotLine))
                sep_y = y_base + bar_group_h
                painter.drawLine(int(margin_left), int(sep_y), int(w - margin_right), int(sep_y))

        painter.end()
