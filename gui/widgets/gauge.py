"""
SYNDICATE — Confidence Gauge Widget
=====================================
Radial gauge that displays confidence level (0–100%).
Inspired by trading dashboard speedometers.
"""
from __future__ import annotations
import math
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt, QRectF, QPointF
from PyQt6.QtGui import (
    QPainter, QPen, QColor, QConicalGradient,
    QRadialGradient, QFont, QPainterPath,
)
from gui.theme import (
    BG_SECONDARY, BORDER_MEDIUM, TEXT_PRIMARY, TEXT_SECONDARY,
    ORANGE_PRIMARY, GREEN_PROFIT, RED_LOSS, YELLOW_WARN,
    GOLD_DIAMOND, FONT_FAMILY_DISPLAY, GAUGE_SIZE,
)


class ConfidenceGauge(QWidget):
    """
    Radial confidence gauge with color-coded arc.
    Red (0-50%) → Yellow (50-70%) → Green (70-85%) → Gold (85-100%)
    """

    def __init__(self, parent=None, size: int = GAUGE_SIZE, label: str = "Confidence"):
        super().__init__(parent)
        self._value = 0.0           # 0.0 – 1.0
        self._label = label
        self._size = size
        self.setFixedSize(size, size + 28)

    def set_value(self, value: float, label: str = None):
        """Set gauge value (0.0 to 1.0)."""
        self._value = max(0.0, min(1.0, value))
        if label:
            self._label = label
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        s = self._size
        cx, cy = s // 2, s // 2
        radius = s // 2 - 12
        arc_width = 10

        # ── Background circle ─────────────────────────────
        bg_color = QColor(BG_SECONDARY)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(bg_color)
        painter.drawEllipse(QPointF(cx, cy), radius + 4, radius + 4)

        # ── Track arc (background) ────────────────────────
        track_pen = QPen(QColor(BORDER_MEDIUM), arc_width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        painter.setPen(track_pen)
        arc_rect = QRectF(cx - radius, cy - radius, radius * 2, radius * 2)
        # Arc from 225° to -45° (270° sweep, bottom-left to bottom-right)
        painter.drawArc(arc_rect, 225 * 16, -270 * 16)

        # ── Value arc (colored) ───────────────────────────
        if self._value > 0.005:
            color = self._get_color(self._value)
            value_pen = QPen(QColor(color), arc_width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
            painter.setPen(value_pen)
            sweep = int(-270 * self._value)
            painter.drawArc(arc_rect, 225 * 16, sweep * 16)

        # ── Needle dot ────────────────────────────────────
        angle_deg = 225 - 270 * self._value
        angle_rad = math.radians(angle_deg)
        nx = cx + (radius - 2) * math.cos(angle_rad)
        ny = cy - (radius - 2) * math.sin(angle_rad)
        needle_color = QColor(self._get_color(self._value))
        painter.setPen(Qt.PenStyle.NoPen)

        # Glow
        glow = QRadialGradient(QPointF(nx, ny), 8)
        glow.setColorAt(0, needle_color)
        glow.setColorAt(1, QColor(0, 0, 0, 0))
        painter.setBrush(glow)
        painter.drawEllipse(QPointF(nx, ny), 8, 8)

        # Dot
        painter.setBrush(needle_color)
        painter.drawEllipse(QPointF(nx, ny), 4, 4)

        # ── Center text ───────────────────────────────────
        pct_text = f"{self._value:.0%}"
        font_value = QFont(FONT_FAMILY_DISPLAY, 20, QFont.Weight.Bold)
        painter.setFont(font_value)
        painter.setPen(QColor(TEXT_PRIMARY))
        painter.drawText(QRectF(0, cy - 20, s, 36), Qt.AlignmentFlag.AlignCenter, pct_text)

        # ── Bottom label ──────────────────────────────────
        font_label = QFont(FONT_FAMILY_DISPLAY, 8)
        painter.setFont(font_label)
        painter.setPen(QColor(TEXT_SECONDARY))
        painter.drawText(QRectF(0, s + 2, s, 20), Qt.AlignmentFlag.AlignCenter, self._label)

        painter.end()

    @staticmethod
    def _get_color(value: float) -> str:
        if value >= 0.85:
            return GOLD_DIAMOND
        elif value >= 0.70:
            return GREEN_PROFIT
        elif value >= 0.50:
            return YELLOW_WARN
        else:
            return RED_LOSS
