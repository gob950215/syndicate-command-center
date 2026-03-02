"""
SYNDICATE — Navigation Sidebar
================================
Bloomberg-style left sidebar with icon navigation.
"""
from __future__ import annotations
from PyQt6.QtWidgets import (
    QFrame, QVBoxLayout, QLabel, QPushButton, QWidget, QSizePolicy,
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QFont

from gui.theme import (
    BG_PRIMARY, BG_SELECTED, BORDER_MEDIUM,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED,
    ORANGE_PRIMARY, ORANGE_GLOW,
    FONT_FAMILY_DISPLAY, FONT_FAMILY_UI, SIDEBAR_WIDTH,
)


class SidebarButton(QPushButton):
    """Single navigation button with icon + label."""

    def __init__(self, icon: str, label: str, page_key: str, parent=None):
        super().__init__(parent)
        self.page_key = page_key
        self._active = False
        self.setText(f"  {icon}  {label}")
        self.setFixedHeight(44)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._update_style()

    def set_active(self, active: bool):
        self._active = active
        self._update_style()

    def _update_style(self):
        if self._active:
            self.setStyleSheet(f"""
                QPushButton {{
                    background-color: {BG_SELECTED};
                    color: {ORANGE_PRIMARY};
                    border: none;
                    border-left: 3px solid {ORANGE_PRIMARY};
                    border-radius: 0px;
                    text-align: left;
                    padding-left: 16px;
                    font-size: 12px;
                    font-weight: bold;
                    font-family: "{FONT_FAMILY_UI}";
                }}
            """)
        else:
            self.setStyleSheet(f"""
                QPushButton {{
                    background-color: transparent;
                    color: {TEXT_SECONDARY};
                    border: none;
                    border-left: 3px solid transparent;
                    border-radius: 0px;
                    text-align: left;
                    padding-left: 16px;
                    font-size: 12px;
                    font-family: "{FONT_FAMILY_UI}";
                }}
                QPushButton:hover {{
                    background-color: {ORANGE_GLOW};
                    color: {TEXT_PRIMARY};
                    border-left: 3px solid {ORANGE_PRIMARY}66;
                }}
            """)


class Sidebar(QFrame):
    """
    Left-side navigation panel.
    Emits `page_changed(key)` when a nav button is clicked.
    """

    page_changed = pyqtSignal(str)

    PAGES = [
        ("📊", "Dashboard", "dashboard"),
        ("⚔️", "War Room", "war_room"),
        ("⏰", "Scheduler", "scheduler"),
        ("⚙️", "Settings", "settings"),
    ]

    ADMIN_PAGES = [
        ("🔒", "Admin Panel", "admin"),
    ]

    def __init__(self, is_admin: bool = False, parent=None):
        super().__init__(parent)
        self.setFixedWidth(SIDEBAR_WIDTH)
        self.setStyleSheet(f"""
            Sidebar {{
                background-color: {BG_PRIMARY};
                border-right: 1px solid {BORDER_MEDIUM};
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Logo area ─────────────────────────────────
        logo_frame = QWidget()
        logo_layout = QVBoxLayout(logo_frame)
        logo_layout.setContentsMargins(16, 20, 16, 20)
        logo_layout.setSpacing(2)

        logo = QLabel("SYNDICATE")
        logo.setStyleSheet(f"""
            color: {ORANGE_PRIMARY};
            font-size: 16px;
            font-weight: bold;
            font-family: "{FONT_FAMILY_DISPLAY}";
            letter-spacing: 3px;
        """)
        logo_layout.addWidget(logo)

        subtitle = QLabel("COMMAND CENTER")
        subtitle.setStyleSheet(f"""
            color: {TEXT_MUTED};
            font-size: 8px;
            font-family: "{FONT_FAMILY_DISPLAY}";
            letter-spacing: 2px;
        """)
        logo_layout.addWidget(subtitle)

        layout.addWidget(logo_frame)

        # ── Separator ─────────────────────────────────
        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background-color: {BORDER_MEDIUM};")
        layout.addWidget(sep)

        # ── Nav section label ─────────────────────────
        nav_label = QLabel("  NAVIGATION")
        nav_label.setStyleSheet(f"""
            color: {TEXT_MUTED};
            font-size: 8px;
            font-family: "{FONT_FAMILY_DISPLAY}";
            letter-spacing: 2px;
            padding: 12px 0 6px 16px;
        """)
        layout.addWidget(nav_label)

        # ── Nav buttons ───────────────────────────────
        self._buttons: list[SidebarButton] = []
        pages = self.PAGES + (self.ADMIN_PAGES if is_admin else [])

        for icon, label, key in pages:
            btn = SidebarButton(icon, label, key)
            btn.clicked.connect(lambda checked, k=key: self._on_click(k))
            layout.addWidget(btn)
            self._buttons.append(btn)

        layout.addStretch()

        # ── Bottom: sport indicators ──────────────────
        sport_label = QLabel("  SPORTS")
        sport_label.setStyleSheet(f"""
            color: {TEXT_MUTED};
            font-size: 8px;
            font-family: "{FONT_FAMILY_DISPLAY}";
            letter-spacing: 2px;
            padding: 0 0 6px 16px;
        """)
        layout.addWidget(sport_label)

        from config import SUPPORTED_SPORTS
        for sport, info in SUPPORTED_SPORTS.items():
            status_color = ORANGE_PRIMARY if info["enabled"] else TEXT_MUTED
            status_dot = "●" if info["enabled"] else "○"
            sport_item = QLabel(f"  {info['icon']}  {sport}  {status_dot}")
            sport_item.setStyleSheet(f"""
                color: {status_color};
                font-size: 10px;
                font-family: "{FONT_FAMILY_DISPLAY}";
                padding: 3px 16px;
            """)
            layout.addWidget(sport_item)

        layout.addSpacing(16)

        # Set first button active
        if self._buttons:
            self._buttons[0].set_active(True)

    def _on_click(self, key: str):
        for btn in self._buttons:
            btn.set_active(btn.page_key == key)
        self.page_changed.emit(key)
