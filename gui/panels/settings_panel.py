"""
SYNDICATE — Settings Panel
============================
API key management and application configuration.
"""
from __future__ import annotations
import os
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QFrame, QGroupBox, QComboBox, QCheckBox,
    QMessageBox, QFileDialog,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

from gui.theme import (
    BG_PRIMARY, BG_SECONDARY, BG_TERTIARY, BORDER_MEDIUM,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED,
    ORANGE_PRIMARY, GREEN_PROFIT, RED_LOSS, YELLOW_WARN,
    FONT_FAMILY_DISPLAY, FONT_FAMILY_UI,
)


class SettingsPanel(QWidget):
    """Settings and configuration panel."""

    settings_saved = pyqtSignal()
    train_requested = pyqtSignal()  # emitted when user clicks TRAIN MODEL

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._load_current()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(24, 20, 24, 20)
        main_layout.setSpacing(20)

        # ── Header ──────────────────────────────────────
        title = QLabel("SETTINGS")
        title.setStyleSheet(f"""
            color: {TEXT_PRIMARY};
            font-size: 20px;
            font-weight: bold;
            font-family: "{FONT_FAMILY_DISPLAY}";
            letter-spacing: 2px;
        """)
        main_layout.addWidget(title)

        # ── API Keys Section ────────────────────────────
        api_group = QGroupBox("API Configuration")
        api_layout = QVBoxLayout(api_group)
        api_layout.setSpacing(16)

        # Sports API Key
        api_layout.addWidget(self._make_section_label("Sports API Key (api-sports.io)"))
        sports_row = QHBoxLayout()
        self._sports_key = QLineEdit()
        self._sports_key.setPlaceholderText("Enter your Sports API key...")
        self._sports_key.setEchoMode(QLineEdit.EchoMode.Password)
        sports_row.addWidget(self._sports_key, 1)

        self._sports_toggle = QPushButton("Show")
        self._sports_toggle.setFixedWidth(60)
        self._sports_toggle.clicked.connect(lambda: self._toggle_visibility(self._sports_key, self._sports_toggle))
        sports_row.addWidget(self._sports_toggle)

        self._sports_status = QLabel("")
        self._sports_status.setFixedWidth(24)
        sports_row.addWidget(self._sports_status)
        api_layout.addLayout(sports_row)

        # Odds API Key
        api_layout.addWidget(self._make_section_label("Odds API Key (the-odds-api.com)"))
        odds_row = QHBoxLayout()
        self._odds_key = QLineEdit()
        self._odds_key.setPlaceholderText("Enter your Odds API key...")
        self._odds_key.setEchoMode(QLineEdit.EchoMode.Password)
        odds_row.addWidget(self._odds_key, 1)

        self._odds_toggle = QPushButton("Show")
        self._odds_toggle.setFixedWidth(60)
        self._odds_toggle.clicked.connect(lambda: self._toggle_visibility(self._odds_key, self._odds_toggle))
        odds_row.addWidget(self._odds_toggle)

        self._odds_status = QLabel("")
        self._odds_status.setFixedWidth(24)
        odds_row.addWidget(self._odds_status)
        api_layout.addLayout(odds_row)

        # Save keys button
        save_row = QHBoxLayout()
        save_row.addStretch()
        note = QLabel("Keys are saved to environment variables for the current session.")
        note.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 9px; font-family: '{FONT_FAMILY_DISPLAY}';")
        save_row.addWidget(note)

        save_btn = QPushButton("  SAVE API KEYS")
        save_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {ORANGE_PRIMARY};
                color: #0D1117;
                border: none;
                border-radius: 6px;
                font-weight: bold;
                font-family: "{FONT_FAMILY_DISPLAY}";
                padding: 8px 20px;
            }}
            QPushButton:hover {{ background-color: #FF6B2B; }}
        """)
        save_btn.clicked.connect(self._save_keys)
        save_row.addWidget(save_btn)
        api_layout.addLayout(save_row)

        main_layout.addWidget(api_group)

        # ── Model Configuration ─────────────────────────
        model_group = QGroupBox("Model Configuration")
        model_layout = QVBoxLayout(model_group)
        model_layout.setSpacing(12)

        # DB path
        db_row = QHBoxLayout()
        db_row.addWidget(self._make_section_label("Database Path:"))
        self._db_path = QLineEdit()
        self._db_path.setText("data/nba_historical.db")
        db_row.addWidget(self._db_path, 1)

        browse_btn = QPushButton("Browse")
        browse_btn.setFixedWidth(70)
        browse_btn.clicked.connect(self._browse_db)
        db_row.addWidget(browse_btn)
        model_layout.addLayout(db_row)

        # Model path
        model_row = QHBoxLayout()
        model_row.addWidget(self._make_section_label("Model File:"))
        self._model_path = QLineEdit()
        self._model_path.setText("models/nba_model_v8.pkl")
        self._model_path.setReadOnly(True)
        model_row.addWidget(self._model_path, 1)

        self._model_status = QLabel("")
        self._model_status.setFixedWidth(100)
        model_row.addWidget(self._model_status)
        model_layout.addLayout(model_row)

        # Train model button
        train_row = QHBoxLayout()
        train_explain = QLabel(
            "If no trained model exists, the system uses Monte Carlo + Elo only. "
            "Training processes all historical seasons and takes several minutes."
        )
        train_explain.setWordWrap(True)
        train_explain.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 9px; font-family: '{FONT_FAMILY_DISPLAY}';")
        train_row.addWidget(train_explain, 1)

        self._train_btn = QPushButton("  TRAIN MODEL")
        self._train_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {YELLOW_WARN};
                color: #0D1117;
                border: none;
                border-radius: 6px;
                font-weight: bold;
                font-family: "{FONT_FAMILY_DISPLAY}";
                padding: 8px 20px;
                font-size: 11px;
            }}
            QPushButton:hover {{ background-color: #E5A200; }}
        """)
        self._train_btn.clicked.connect(self._on_train_requested)
        train_row.addWidget(self._train_btn)
        model_layout.addLayout(train_row)

        # Simulations
        sim_row = QHBoxLayout()
        sim_row.addWidget(self._make_section_label("Monte Carlo Simulations:"))
        self._sim_input = QComboBox()
        self._sim_input.addItems(["1,000", "5,000", "10,000", "25,000", "50,000"])
        self._sim_input.setCurrentIndex(1)
        self._sim_input.setFixedWidth(120)
        sim_row.addWidget(self._sim_input)
        sim_row.addStretch()
        model_layout.addLayout(sim_row)

        main_layout.addWidget(model_group)

        # ── User Account ────────────────────────────────
        account_group = QGroupBox("Account Security")
        account_layout = QVBoxLayout(account_group)
        account_layout.setSpacing(12)

        # Change password
        account_layout.addWidget(self._make_section_label("Change Password"))
        pw_row = QHBoxLayout()
        self._old_pw = QLineEdit()
        self._old_pw.setPlaceholderText("Current password")
        self._old_pw.setEchoMode(QLineEdit.EchoMode.Password)
        pw_row.addWidget(self._old_pw)

        self._new_pw = QLineEdit()
        self._new_pw.setPlaceholderText("New password")
        self._new_pw.setEchoMode(QLineEdit.EchoMode.Password)
        pw_row.addWidget(self._new_pw)

        change_btn = QPushButton("Change")
        change_btn.setFixedWidth(80)
        change_btn.clicked.connect(self._change_password)
        pw_row.addWidget(change_btn)

        account_layout.addLayout(pw_row)

        # 2FA prep
        totp_row = QHBoxLayout()
        totp_row.addWidget(self._make_section_label("Two-Factor Authentication (2FA):"))
        self._totp_status = QLabel("Not configured")
        self._totp_status.setStyleSheet(f"color: {YELLOW_WARN}; font-size: 10px; font-family: '{FONT_FAMILY_DISPLAY}';")
        totp_row.addWidget(self._totp_status)
        totp_row.addStretch()

        setup_2fa_btn = QPushButton("Setup 2FA")
        setup_2fa_btn.setFixedWidth(90)
        setup_2fa_btn.setToolTip("Generates a TOTP secret for Google Authenticator")
        setup_2fa_btn.clicked.connect(self._setup_2fa)
        totp_row.addWidget(setup_2fa_btn)
        account_layout.addLayout(totp_row)

        main_layout.addWidget(account_group)

        # ── Supabase Cloud ──────────────────────────────
        cloud_group = QGroupBox("Cloud Integration (Supabase)")
        cloud_layout = QVBoxLayout(cloud_group)
        cloud_layout.setSpacing(12)

        # URL
        cloud_layout.addWidget(self._make_section_label("Supabase Project URL"))
        self._supa_url = QLineEdit()
        self._supa_url.setPlaceholderText("https://your-project.supabase.co")
        self._supa_url.setText(os.environ.get("SUPABASE_URL", ""))
        cloud_layout.addWidget(self._supa_url)

        # Key
        cloud_layout.addWidget(self._make_section_label("Supabase Anon Key"))
        key_row = QHBoxLayout()
        self._supa_key = QLineEdit()
        self._supa_key.setPlaceholderText("eyJ...")
        self._supa_key.setEchoMode(QLineEdit.EchoMode.Password)
        self._supa_key.setText(os.environ.get("SUPABASE_KEY", ""))
        key_row.addWidget(self._supa_key, 1)
        supa_toggle = QPushButton("Show")
        supa_toggle.setFixedWidth(60)
        supa_toggle.clicked.connect(lambda: self._toggle_visibility(self._supa_key, supa_toggle))
        key_row.addWidget(supa_toggle)
        cloud_layout.addLayout(key_row)

        # Status + Actions
        actions_row = QHBoxLayout()

        self._cloud_status = QLabel("Not connected")
        self._cloud_status.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 10px; font-family: '{FONT_FAMILY_DISPLAY}';")
        actions_row.addWidget(self._cloud_status)
        actions_row.addStretch()

        save_cloud_btn = QPushButton("  SAVE & CONNECT")
        save_cloud_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {ORANGE_PRIMARY};
                color: #0D1117;
                border: none;
                border-radius: 6px;
                font-weight: bold;
                font-family: "{FONT_FAMILY_DISPLAY}";
                padding: 8px 20px;
            }}
            QPushButton:hover {{ background-color: #FF6B2B; }}
        """)
        save_cloud_btn.clicked.connect(self._save_supabase)
        actions_row.addWidget(save_cloud_btn)

        sync_btn = QPushButton("Sync DB to Cloud")
        sync_btn.setFixedWidth(140)
        sync_btn.clicked.connect(self._sync_to_cloud)
        actions_row.addWidget(sync_btn)

        cloud_layout.addLayout(actions_row)

        cloud_note = QLabel(
            "Supabase manages: user access control (remote kill-switch), "
            "API key storage, and DB/model backup to the cloud."
        )
        cloud_note.setWordWrap(True)
        cloud_note.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 9px; font-family: '{FONT_FAMILY_DISPLAY}';")
        cloud_layout.addWidget(cloud_note)

        main_layout.addWidget(cloud_group)

        main_layout.addStretch()

    def _make_section_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 10px; font-family: '{FONT_FAMILY_DISPLAY}';")
        return lbl

    def _toggle_visibility(self, line_edit: QLineEdit, btn: QPushButton):
        if line_edit.echoMode() == QLineEdit.EchoMode.Password:
            line_edit.setEchoMode(QLineEdit.EchoMode.Normal)
            btn.setText("Hide")
        else:
            line_edit.setEchoMode(QLineEdit.EchoMode.Password)
            btn.setText("Show")

    def _load_current(self):
        sports_key = os.environ.get("SPORTS_API_KEY", "")
        odds_key = os.environ.get("ODDS_API_KEY", "")

        if sports_key:
            self._sports_key.setText(sports_key)
            self._sports_status.setText("OK")
            self._sports_status.setStyleSheet(f"color: {GREEN_PROFIT};")
        else:
            self._sports_status.setText("--")
            self._sports_status.setStyleSheet(f"color: {RED_LOSS};")

        if odds_key:
            self._odds_key.setText(odds_key)
            self._odds_status.setText("OK")
            self._odds_status.setStyleSheet(f"color: {GREEN_PROFIT};")
        else:
            self._odds_status.setText("--")
            self._odds_status.setStyleSheet(f"color: {RED_LOSS};")

        # Check model file
        self.check_model_status()

    def check_model_status(self):
        """Check if trained model file exists and update the UI."""
        model_path = self._model_path.text().strip()
        if os.path.exists(model_path):
            size_kb = os.path.getsize(model_path) / 1024
            self._model_status.setText(f"Found ({size_kb:.0f} KB)")
            self._model_status.setStyleSheet(f"color: {GREEN_PROFIT}; font-size: 9px; font-family: '{FONT_FAMILY_DISPLAY}';")
            self._train_btn.setText("  RETRAIN MODEL")
        else:
            self._model_status.setText("NOT FOUND")
            self._model_status.setStyleSheet(f"color: {RED_LOSS}; font-size: 9px; font-weight: bold; font-family: '{FONT_FAMILY_DISPLAY}';")
            self._train_btn.setText("  TRAIN MODEL")

    def _browse_db(self):
        """Open file dialog to select the database file."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select NBA Historical Database",
            os.path.dirname(self._db_path.text()) or ".",
            "SQLite Database (*.db *.sqlite *.sqlite3);;All Files (*)",
        )
        if path:
            self._db_path.setText(path)

    def _on_train_requested(self):
        """Validate and emit the train signal."""
        db_path = self._db_path.text().strip()
        if not db_path or not os.path.exists(db_path):
            QMessageBox.warning(
                self, "Database Not Found",
                f"Cannot find database at:\n{db_path}\n\n"
                "Use the Browse button to select the correct file."
            )
            return

        reply = QMessageBox.question(
            self, "Train Model",
            "This will train the XGBoost model on all historical seasons.\n"
            "It may take several minutes depending on your hardware.\n\n"
            "Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.train_requested.emit()

    def _save_keys(self):
        sports = self._sports_key.text().strip()
        odds = self._odds_key.text().strip()

        if sports:
            os.environ["SPORTS_API_KEY"] = sports
            self._sports_status.setText("OK")
            self._sports_status.setStyleSheet(f"color: {GREEN_PROFIT};")

        if odds:
            os.environ["ODDS_API_KEY"] = odds
            self._odds_status.setText("OK")
            self._odds_status.setStyleSheet(f"color: {GREEN_PROFIT};")

        self.settings_saved.emit()

    def _change_password(self):
        # Handled by main_window connecting to UserDB
        pass

    def _setup_2fa(self):
        try:
            from security.totp_prep import TOTPManager
            if not TOTPManager.is_available():
                QMessageBox.information(
                    self, "2FA",
                    "pyotp not installed. Install with:\npip install pyotp\n\n"
                    "The system is prepared for 2FA integration."
                )
                return

            secret = TOTPManager.generate_secret()
            uri = TOTPManager.get_provisioning_uri("user", secret)
            QMessageBox.information(
                self, "2FA Setup",
                f"TOTP Secret (save this):\n{secret}\n\n"
                f"Provisioning URI:\n{uri}\n\n"
                "Scan the URI as a QR code in Google Authenticator."
            )
            self._totp_status.setText("Configured")
            self._totp_status.setStyleSheet(f"color: {GREEN_PROFIT}; font-size: 10px; font-family: '{FONT_FAMILY_DISPLAY}';")
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))

    def get_db_path(self) -> str:
        return self._db_path.text().strip()

    def get_sims(self) -> int:
        text = self._sim_input.currentText().replace(",", "")
        return int(text)

    def _save_supabase(self):
        """Save Supabase credentials and try to connect."""
        url = self._supa_url.text().strip()
        key = self._supa_key.text().strip()
        if not url or not key:
            QMessageBox.warning(self, "Missing", "Both URL and Key are required.")
            return
        os.environ["SUPABASE_URL"] = url
        os.environ["SUPABASE_KEY"] = key

        try:
            from cloud import SupabaseManager
            mgr = SupabaseManager(url, key)
            if mgr.connect():
                self._cloud_status.setText("Connected")
                self._cloud_status.setStyleSheet(f"color: {GREEN_PROFIT}; font-size: 10px; font-family: '{FONT_FAMILY_DISPLAY}';")
                # Also save API keys to cloud
                mgr.save_env_to_cloud()
                QMessageBox.information(self, "Cloud", "Connected to Supabase.\nAPI keys synced to cloud.")
            else:
                self._cloud_status.setText("Connection failed")
                self._cloud_status.setStyleSheet(f"color: {RED_LOSS}; font-size: 10px; font-family: '{FONT_FAMILY_DISPLAY}';")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Supabase error: {e}")

    def _sync_to_cloud(self):
        """Upload DB and model files to Supabase Storage."""
        try:
            from cloud import SupabaseManager
            url = os.environ.get("SUPABASE_URL", "")
            key = os.environ.get("SUPABASE_KEY", "")
            if not url or not key:
                QMessageBox.warning(self, "Not Connected", "Configure Supabase first.")
                return

            mgr = SupabaseManager(url, key)
            if not mgr.connect():
                QMessageBox.warning(self, "Error", "Could not connect to Supabase.")
                return

            db_path = self._db_path.text().strip()
            results = []
            if os.path.exists(db_path):
                if mgr.sync_db_to_cloud(db_path):
                    results.append("Database uploaded")
                else:
                    results.append("Database upload FAILED")

            model_path = self._model_path.text().strip()
            if os.path.exists(model_path):
                if mgr.sync_model_to_cloud(model_path):
                    results.append("Model uploaded")
                else:
                    results.append("Model upload FAILED")

            QMessageBox.information(self, "Sync", "\n".join(results) if results else "No files to sync.")
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))
