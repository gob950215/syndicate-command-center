"""
SYNDICATE — Scheduler Panel
=============================
UI for managing automated analysis scheduling rules.
"""
from __future__ import annotations
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSpinBox, QLineEdit, QFrame, QTableWidget,
    QTableWidgetItem, QHeaderView, QCheckBox, QGroupBox,
    QScrollArea, QSizePolicy,
)
from PyQt6.QtCore import Qt, pyqtSignal, QTime
from PyQt6.QtGui import QFont

from gui.theme import (
    BG_PRIMARY, BG_SECONDARY, BG_TERTIARY, BORDER_MEDIUM,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED,
    ORANGE_PRIMARY, GREEN_PROFIT, RED_LOSS, YELLOW_WARN,
    FONT_FAMILY_DISPLAY, FONT_FAMILY_UI,
)
from scheduler.scheduler_engine import SchedulerEngine
from core.data_models import SchedulerRule


class SchedulerPanel(QWidget):
    """Scheduler / automation configuration panel."""

    rule_added = pyqtSignal()
    manual_trigger = pyqtSignal(str)  # sport name

    def __init__(self, scheduler: SchedulerEngine, parent=None):
        super().__init__(parent)
        self._scheduler = scheduler
        self._setup_ui()
        self._refresh_rules()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(24, 20, 24, 20)
        main_layout.setSpacing(20)

        # ── Header ──────────────────────────────────────
        header = QHBoxLayout()
        title = QLabel("THE SCHEDULER")
        title.setStyleSheet(f"""
            color: {TEXT_PRIMARY};
            font-size: 20px;
            font-weight: bold;
            font-family: "{FONT_FAMILY_DISPLAY}";
            letter-spacing: 2px;
        """)
        header.addWidget(title)
        header.addStretch()

        subtitle = QLabel("Automated Analysis Scheduling")
        subtitle.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px; font-family: '{FONT_FAMILY_DISPLAY}';")
        header.addWidget(subtitle)
        main_layout.addLayout(header)

        # ── Add Rule Section ────────────────────────────
        add_group = QGroupBox("Add New Rule")
        add_layout = QVBoxLayout(add_group)
        add_layout.setSpacing(12)

        # Rule name
        name_row = QHBoxLayout()
        name_row.addWidget(self._make_label("Rule Name:"))
        self._name_input = QLineEdit()
        self._name_input.setPlaceholderText("e.g., Pre-game analysis 2h before tip")
        name_row.addWidget(self._name_input, 1)
        add_layout.addLayout(name_row)

        # Sport + Trigger type
        config_row = QHBoxLayout()
        config_row.setSpacing(16)

        config_row.addWidget(self._make_label("Sport:"))
        self._sport_combo = QComboBox()
        self._sport_combo.addItems(["NBA", "MLB", "NFL"])
        self._sport_combo.setFixedWidth(100)
        config_row.addWidget(self._sport_combo)

        config_row.addWidget(self._make_label("Trigger:"))
        self._trigger_combo = QComboBox()
        self._trigger_combo.addItems([
            "After Lines Open",
            "Before Lock-in",
            "Fixed Daily Time",
        ])
        self._trigger_combo.setFixedWidth(180)
        self._trigger_combo.currentTextChanged.connect(self._on_trigger_changed)
        config_row.addWidget(self._trigger_combo)

        # Offset spinner (for After Lines Open / Before Lock-in)
        self._offset_label = self._make_label("Offset (min):")
        config_row.addWidget(self._offset_label)
        self._offset_spin = QSpinBox()
        self._offset_spin.setRange(5, 1440)
        self._offset_spin.setValue(120)
        self._offset_spin.setSuffix(" min")
        self._offset_spin.setFixedWidth(100)
        config_row.addWidget(self._offset_spin)

        # Time picker (for Fixed Daily Time)
        self._time_label = self._make_label("Time:")
        self._time_label.hide()
        config_row.addWidget(self._time_label)

        from PyQt6.QtWidgets import QTimeEdit
        self._time_edit = QTimeEdit()
        self._time_edit.setDisplayFormat("hh:mm AP")
        self._time_edit.setTime(QTime(14, 0))
        self._time_edit.setFixedWidth(120)
        self._time_edit.hide()
        config_row.addWidget(self._time_edit)

        config_row.addStretch()

        add_btn = QPushButton("  ADD RULE")
        add_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {ORANGE_PRIMARY};
                color: #0D1117;
                border: none;
                border-radius: 6px;
                font-size: 11px;
                font-weight: bold;
                font-family: "{FONT_FAMILY_DISPLAY}";
                padding: 8px 20px;
            }}
            QPushButton:hover {{ background-color: #FF6B2B; }}
        """)
        add_btn.clicked.connect(self._add_rule)
        config_row.addWidget(add_btn)

        add_layout.addLayout(config_row)

        # Explanation
        explain = QLabel(
            "After Lines Open: Runs X minutes after lines open (typically 12h before tipoff)  |  "
            "Before Lock-in: Runs X minutes before the first game starts  |  "
            "Fixed Daily Time: Offset = minutes from midnight (e.g., 840 = 2:00 PM)"
        )
        explain.setWordWrap(True)
        explain.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 9px; font-family: '{FONT_FAMILY_DISPLAY}';")
        add_layout.addWidget(explain)

        main_layout.addWidget(add_group)

        # ── Active Rules Table ──────────────────────────
        rules_header = QHBoxLayout()
        rules_title = QLabel("ACTIVE RULES")
        rules_title.setStyleSheet(f"""
            color: {TEXT_SECONDARY};
            font-size: 11px;
            font-weight: bold;
            font-family: "{FONT_FAMILY_DISPLAY}";
            letter-spacing: 2px;
        """)
        rules_header.addWidget(rules_title)
        rules_header.addStretch()

        refresh_btn = QPushButton("Refresh")
        refresh_btn.setFixedWidth(80)
        refresh_btn.clicked.connect(self._refresh_rules)
        rules_header.addWidget(refresh_btn)
        main_layout.addLayout(rules_header)

        self._rules_table = QTableWidget()
        self._rules_table.setColumnCount(7)
        self._rules_table.setHorizontalHeaderLabels([
            "Enabled", "Name", "Sport", "Trigger", "Offset", "Next Run", "Actions"
        ])
        self._rules_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._rules_table.setAlternatingRowColors(True)
        self._rules_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._rules_table.verticalHeader().setVisible(False)
        main_layout.addWidget(self._rules_table, 1)

        # ── Scheduler Log ───────────────────────────────
        log_header = QLabel("RECENT LOG")
        log_header.setStyleSheet(f"""
            color: {TEXT_SECONDARY};
            font-size: 11px;
            font-weight: bold;
            font-family: "{FONT_FAMILY_DISPLAY}";
            letter-spacing: 2px;
        """)
        main_layout.addWidget(log_header)

        self._log_table = QTableWidget()
        self._log_table.setColumnCount(4)
        self._log_table.setHorizontalHeaderLabels(["Timestamp", "Rule", "Status", "Details"])
        self._log_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self._log_table.setAlternatingRowColors(True)
        self._log_table.verticalHeader().setVisible(False)
        self._log_table.setMaximumHeight(180)
        main_layout.addWidget(self._log_table)

    def _make_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px; font-family: '{FONT_FAMILY_DISPLAY}';")
        return lbl

    def _on_trigger_changed(self, trigger_text: str):
        """Show/hide time picker vs offset spinner based on trigger type."""
        is_fixed = (trigger_text == "Fixed Daily Time")
        self._offset_label.setVisible(not is_fixed)
        self._offset_spin.setVisible(not is_fixed)
        self._time_label.setVisible(is_fixed)
        self._time_edit.setVisible(is_fixed)

    def _add_rule(self):
        name = self._name_input.text().strip()
        if not name:
            return

        trigger_map = {
            "After Lines Open": "after_open",
            "Before Lock-in": "before_lock",
            "Fixed Daily Time": "fixed_time",
        }
        trigger = trigger_map.get(self._trigger_combo.currentText(), "after_open")
        sport = self._sport_combo.currentText()

        if trigger == "fixed_time":
            # Convert QTime to minutes from midnight
            t = self._time_edit.time()
            offset = t.hour() * 60 + t.minute()
        else:
            offset = self._offset_spin.value()

        self._scheduler.add_rule(name, sport, trigger, offset)
        self._name_input.clear()
        self._refresh_rules()
        self.rule_added.emit()

    def _refresh_rules(self):
        rules = self._scheduler.get_rules()
        self._rules_table.setRowCount(len(rules))

        for i, rule in enumerate(rules):
            # Enabled checkbox
            cb = QCheckBox()
            cb.setChecked(rule.enabled)
            cb.stateChanged.connect(lambda state, rid=rule.rule_id: self._toggle_rule(rid))
            self._rules_table.setCellWidget(i, 0, cb)

            self._rules_table.setItem(i, 1, QTableWidgetItem(rule.name))
            self._rules_table.setItem(i, 2, QTableWidgetItem(rule.sport))

            trigger_labels = {
                "after_open": "After Lines Open",
                "before_lock": "Before Lock-in",
                "fixed_time": "Fixed Daily Time",
            }
            self._rules_table.setItem(i, 3, QTableWidgetItem(trigger_labels.get(rule.trigger_type, rule.trigger_type)))

            # Show human-readable offset
            if rule.trigger_type == "fixed_time":
                hours = rule.offset_minutes // 60
                mins = rule.offset_minutes % 60
                period = "AM" if hours < 12 else "PM"
                display_h = hours % 12 or 12
                offset_text = f"{display_h}:{mins:02d} {period}"
            else:
                offset_text = f"{rule.offset_minutes} min"
            self._rules_table.setItem(i, 4, QTableWidgetItem(offset_text))

            next_run = rule.next_run.strftime("%H:%M:%S") if rule.next_run else "Not scheduled"
            self._rules_table.setItem(i, 5, QTableWidgetItem(next_run))

            # Delete button — fit to row height
            del_btn = QPushButton("Delete")
            del_btn.setStyleSheet(f"""
                QPushButton {{
                    background: {BG_TERTIARY};
                    color: {RED_LOSS};
                    border: 1px solid {RED_LOSS}44;
                    border-radius: 4px;
                    font-size: 10px;
                    padding: 2px 8px;
                    margin: 2px;
                }}
                QPushButton:hover {{ background: #3D1117; }}
            """)
            del_btn.clicked.connect(lambda _, rid=rule.rule_id: self._delete_rule(rid))
            self._rules_table.setCellWidget(i, 6, del_btn)

        # Auto-fit row heights
        self._rules_table.resizeRowsToContents()
        self._rules_table.setColumnWidth(6, 80)

        # Refresh log
        logs = self._scheduler.get_log(20)
        self._log_table.setRowCount(len(logs))
        for i, log in enumerate(logs):
            self._log_table.setItem(i, 0, QTableWidgetItem(log.get("timestamp", "")))
            self._log_table.setItem(i, 1, QTableWidgetItem(log.get("rule_id", "")))
            status = log.get("status", "")
            item = QTableWidgetItem(status)
            if status == "success":
                item.setForeground(Qt.GlobalColor.green)
            elif status == "error":
                item.setForeground(Qt.GlobalColor.red)
            self._log_table.setItem(i, 2, item)
            self._log_table.setItem(i, 3, QTableWidgetItem(log.get("details", "")))

    def _toggle_rule(self, rule_id: str):
        self._scheduler.toggle_rule(rule_id)

    def _delete_rule(self, rule_id: str):
        self._scheduler.remove_rule(rule_id)
        self._refresh_rules()
