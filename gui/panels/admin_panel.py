"""
SYNDICATE — Admin Panel (Master Admin Dashboard)
==================================================
Hidden section for administrators:
  - Access logs (who, when, from where)
  - User management (create, delete, toggle)
  - Login stats
"""
from __future__ import annotations
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QFrame, QTableWidget, QTableWidgetItem,
    QHeaderView, QComboBox, QGroupBox, QMessageBox,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QColor

from gui.theme import (
    BG_PRIMARY, BG_SECONDARY, BG_TERTIARY, BORDER_MEDIUM,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED,
    ORANGE_PRIMARY, GREEN_PROFIT, RED_LOSS, YELLOW_WARN,
    FONT_FAMILY_DISPLAY,
)
from security.auth import UserDB
from security.session import SessionManager


class AdminPanel(QWidget):
    """Master admin dashboard — access logs and user management."""

    def __init__(self, user_db: UserDB, session_mgr: SessionManager, parent=None):
        super().__init__(parent)
        self._user_db = user_db
        self._session = session_mgr
        self._setup_ui()
        self._refresh_all()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(24, 20, 24, 20)
        main_layout.setSpacing(20)

        # ── Header ──────────────────────────────────────
        header = QHBoxLayout()
        title = QLabel("ADMIN PANEL")
        title.setStyleSheet(f"""
            color: {RED_LOSS};
            font-size: 20px;
            font-weight: bold;
            font-family: "{FONT_FAMILY_DISPLAY}";
            letter-spacing: 2px;
        """)
        header.addWidget(title)
        header.addStretch()

        classify = QLabel("CLASSIFIED — ADMIN ACCESS ONLY")
        classify.setStyleSheet(f"""
            color: {RED_LOSS};
            font-size: 9px;
            font-family: "{FONT_FAMILY_DISPLAY}";
            background: {RED_LOSS}15;
            border: 1px solid {RED_LOSS}33;
            border-radius: 4px;
            padding: 4px 10px;
            letter-spacing: 1px;
        """)
        header.addWidget(classify)
        main_layout.addLayout(header)

        # ── Stats Cards ─────────────────────────────────
        stats_row = QHBoxLayout()
        stats_row.setSpacing(12)

        self._stat_labels = {}
        stat_defs = [
            ("Total Logins", "total_logins", ORANGE_PRIMARY),
            ("Failed Logins", "failed_logins", RED_LOSS),
            ("Unique Users", "unique_users", GREEN_PROFIT),
            ("Unique IPs", "unique_ips", YELLOW_WARN),
        ]

        for title_text, key, color in stat_defs:
            card = QFrame()
            card.setStyleSheet(f"""
                QFrame {{
                    background-color: {BG_SECONDARY};
                    border: 1px solid {BORDER_MEDIUM};
                    border-radius: 8px;
                    border-top: 2px solid {color};
                }}
            """)
            card.setFixedHeight(70)
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(12, 8, 12, 8)
            card_layout.setSpacing(2)

            t = QLabel(title_text.upper())
            t.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 8px; font-family: '{FONT_FAMILY_DISPLAY}'; letter-spacing: 1px;")
            card_layout.addWidget(t)

            v = QLabel("0")
            v.setStyleSheet(f"color: {color}; font-size: 20px; font-weight: bold; font-family: '{FONT_FAMILY_DISPLAY}';")
            self._stat_labels[key] = v
            card_layout.addWidget(v)

            stats_row.addWidget(card)

        main_layout.addLayout(stats_row)

        # ── User Management ─────────────────────────────
        user_group = QGroupBox("User Management")
        user_layout = QVBoxLayout(user_group)
        user_layout.setSpacing(12)

        # Create user row
        create_row = QHBoxLayout()
        create_row.setSpacing(8)

        self._new_user = QLineEdit()
        self._new_user.setPlaceholderText("Username")
        self._new_user.setFixedWidth(150)
        create_row.addWidget(self._new_user)

        self._new_pass = QLineEdit()
        self._new_pass.setPlaceholderText("Password")
        self._new_pass.setEchoMode(QLineEdit.EchoMode.Password)
        self._new_pass.setFixedWidth(150)
        create_row.addWidget(self._new_pass)

        self._role_combo = QComboBox()
        self._role_combo.addItems(["analyst", "admin"])
        self._role_combo.setFixedWidth(100)
        create_row.addWidget(self._role_combo)

        create_btn = QPushButton("Create User")
        create_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {ORANGE_PRIMARY};
                color: #0D1117;
                border: none;
                border-radius: 6px;
                font-weight: bold;
                font-family: "{FONT_FAMILY_DISPLAY}";
                padding: 6px 16px;
                font-size: 10px;
            }}
            QPushButton:hover {{ background-color: #FF6B2B; }}
        """)
        create_btn.clicked.connect(self._create_user)
        create_row.addWidget(create_btn)
        create_row.addStretch()
        user_layout.addLayout(create_row)

        # Users table
        self._users_table = QTableWidget()
        self._users_table.setColumnCount(6)
        self._users_table.setHorizontalHeaderLabels([
            "Username", "Role", "Created", "Last Login", "2FA", "Actions"
        ])
        self._users_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._users_table.setAlternatingRowColors(True)
        self._users_table.verticalHeader().setVisible(False)
        self._users_table.setMaximumHeight(200)
        user_layout.addWidget(self._users_table)

        main_layout.addWidget(user_group)

        # ── Access Logs ─────────────────────────────────
        log_header = QHBoxLayout()
        log_title = QLabel("ACCESS LOG")
        log_title.setStyleSheet(f"""
            color: {TEXT_SECONDARY};
            font-size: 11px;
            font-weight: bold;
            font-family: "{FONT_FAMILY_DISPLAY}";
            letter-spacing: 2px;
        """)
        log_header.addWidget(log_title)
        log_header.addStretch()

        refresh_btn = QPushButton("Refresh")
        refresh_btn.setFixedWidth(80)
        refresh_btn.clicked.connect(self._refresh_all)
        log_header.addWidget(refresh_btn)
        main_layout.addLayout(log_header)

        self._logs_table = QTableWidget()
        self._logs_table.setColumnCount(6)
        self._logs_table.setHorizontalHeaderLabels([
            "Timestamp", "User", "Action", "Status", "IP Address", "Hostname"
        ])
        self._logs_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self._logs_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        self._logs_table.setAlternatingRowColors(True)
        self._logs_table.verticalHeader().setVisible(False)
        main_layout.addWidget(self._logs_table, 1)

    def _refresh_all(self):
        self._refresh_stats()
        self._refresh_users()
        self._refresh_logs()

    def _refresh_stats(self):
        stats = self._session.get_login_stats()
        for key, label in self._stat_labels.items():
            label.setText(str(stats.get(key, 0)))

    def _refresh_users(self):
        users = self._user_db.list_users()
        self._users_table.setRowCount(len(users))

        for i, u in enumerate(users):
            self._users_table.setItem(i, 0, QTableWidgetItem(u["username"]))
            self._users_table.setItem(i, 1, QTableWidgetItem(u["role"]))
            self._users_table.setItem(i, 2, QTableWidgetItem(u.get("created", "")[:10]))
            self._users_table.setItem(i, 3, QTableWidgetItem(u.get("last_login", "Never") or "Never"))
            self._users_table.setItem(i, 4, QTableWidgetItem("Yes" if u.get("totp_enabled") else "No"))

            if u["username"] != "admin":
                del_btn = QPushButton("Delete")
                del_btn.setFixedSize(60, 24)
                del_btn.setStyleSheet(f"""
                    QPushButton {{
                        background: {BG_TERTIARY};
                        color: {RED_LOSS};
                        border: 1px solid {RED_LOSS}44;
                        border-radius: 4px;
                        font-size: 9px;
                    }}
                    QPushButton:hover {{ background: #3D1117; }}
                """)
                del_btn.clicked.connect(lambda _, uname=u["username"]: self._delete_user(uname))
                self._users_table.setCellWidget(i, 5, del_btn)

    def _refresh_logs(self):
        logs = self._session.get_logs(100)
        self._logs_table.setRowCount(len(logs))

        for i, log in enumerate(logs):
            self._logs_table.setItem(i, 0, QTableWidgetItem(log.get("timestamp", "")[:19]))
            self._logs_table.setItem(i, 1, QTableWidgetItem(log.get("username", "")))
            self._logs_table.setItem(i, 2, QTableWidgetItem(log.get("action", "")))

            status_text = "OK" if log.get("success") else "FAIL"
            status_item = QTableWidgetItem(status_text)
            color = QColor(GREEN_PROFIT) if log.get("success") else QColor(RED_LOSS)
            status_item.setForeground(color)
            self._logs_table.setItem(i, 3, status_item)

            self._logs_table.setItem(i, 4, QTableWidgetItem(log.get("ip", "")))
            self._logs_table.setItem(i, 5, QTableWidgetItem(log.get("hostname", "")))

    def _create_user(self):
        username = self._new_user.text().strip()
        password = self._new_pass.text().strip()
        role = self._role_combo.currentText()

        if not username or not password:
            QMessageBox.warning(self, "Error", "Username and password are required.")
            return

        if len(password) < 8:
            QMessageBox.warning(self, "Error", "Password must be at least 8 characters.")
            return

        if self._user_db.create_user(username, password, role):
            self._session.log_event(self._session.current_user or "admin", "create_user", True, f"Created: {username}")
            self._new_user.clear()
            self._new_pass.clear()
            self._refresh_users()
            QMessageBox.information(self, "Success", f"User '{username}' created successfully.")
        else:
            QMessageBox.warning(self, "Error", f"User '{username}' already exists.")

    def _delete_user(self, username: str):
        reply = QMessageBox.question(
            self, "Confirm",
            f"Delete user '{username}'? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            if self._user_db.delete_user(username):
                self._session.log_event(
                    self._session.current_user or "admin", "delete_user", True, f"Deleted: {username}"
                )
                self._refresh_users()
