"""
SYNDICATE COMMAND CENTER — Main Window
========================================
Central window with sidebar navigation, stacked panels,
and connection to the model engine, scheduler, and security.
"""
from __future__ import annotations
import json
import logging
import os
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QStackedWidget, QMessageBox, QDialog, QLabel,
    QLineEdit, QPushButton, QFrame, QApplication,
    QCheckBox,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QIcon

from gui.theme import (
    STYLESHEET, BG_VOID, BG_PRIMARY, BG_SECONDARY, BG_TERTIARY,
    BORDER_MEDIUM, TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED,
    ORANGE_PRIMARY, GREEN_PROFIT, RED_LOSS, YELLOW_WARN,
    FONT_FAMILY_DISPLAY, FONT_FAMILY_UI,
)
from gui.widgets.sidebar import Sidebar
from gui.panels.dashboard import DashboardPanel
from gui.panels.war_room import WarRoomPanel
from gui.panels.scheduler_panel import SchedulerPanel
from gui.panels.settings_panel import SettingsPanel
from gui.panels.admin_panel import AdminPanel

from core.nba_engine import NBAEngine
from core.pick_manager import PickManager
from core.data_models import PickResult, PickStatus

from security.auth import UserDB
from security.session import SessionManager
from scheduler.scheduler_engine import SchedulerEngine

from config import (
    DATA_DIR, DB_PATH, USER_DB_PATH, SESSION_LOG_PATH,
    PICKS_DB_PATH, SCHEDULER_DB_PATH, FERNET_KEY_PATH,
    MODEL_DIR, APP_NAME, APP_VERSION, REMEMBER_ME_PATH,
    SUPABASE_URL, SUPABASE_KEY, SUPABASE_HEARTBEAT_SECONDS,
)
from cloud import SupabaseManager

logger = logging.getLogger("MainWindow")


# ─────────────────────────────────────────────────────────────────────────────
# Worker thread for running analysis without blocking the GUI
# ─────────────────────────────────────────────────────────────────────────────
class EngineInitWorker(QThread):
    """Initializes the NBA engine in a background thread."""
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, engine: NBAEngine, db_path: str, model_dir: str):
        super().__init__()
        self._engine = engine
        self._db_path = db_path
        self._model_dir = model_dir

    def run(self):
        try:
            success = self._engine.initialize(self._db_path, self._model_dir)
            if success:
                self.finished.emit()
            else:
                self.error.emit(self._engine.last_error or "Unknown error")
        except Exception as e:
            self.error.emit(str(e))


class AnalysisWorker(QThread):
    """Runs model analysis in a background thread."""
    finished = pyqtSignal(list)   # List[PickResult]
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, engine: NBAEngine):
        super().__init__()
        self._engine = engine

    def run(self):
        try:
            self.progress.emit("Fetching live data from APIs...")
            picks = self._engine.generate_picks()
            self.finished.emit(picks)
        except Exception as e:
            self.error.emit(str(e))


class TrainWorker(QThread):
    """Runs nba_syndicate_v8.py training in a background thread."""
    finished = pyqtSignal()
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, db_path: str, model_dir: str):
        super().__init__()
        self._db_path = db_path
        self._model_dir = model_dir

    def run(self):
        try:
            self.progress.emit("Importing V8 module...")
            import importlib
            from pathlib import Path

            # Find and import nba_syndicate_v8.py
            search = [Path("."), Path(__file__).parent.parent]
            v8 = None
            for base in search:
                v8_path = base / "nba_syndicate_v8.py"
                if v8_path.exists():
                    spec = importlib.util.spec_from_file_location("nba_syndicate_v8_train", str(v8_path))
                    v8 = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(v8)
                    break

            if v8 is None:
                self.error.emit("nba_syndicate_v8.py not found")
                return

            # Override paths
            v8.DB_PATH = self._db_path
            v8.MODEL_DIR = self._model_dir

            self.progress.emit("Training model on historical data (this may take several minutes)...")
            v8.run_train(
                train_s=v8.TRAIN_SEASONS,
                eval_s=v8.CURRENT_SEASON,
                ckpt=v8.CHECKPOINT,
                db=self._db_path,
                n_sims=5000,
            )
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Login Dialog
# ─────────────────────────────────────────────────────────────────────────────
class LoginDialog(QDialog):
    """Dark-themed login dialog with Supabase Auth + Remember Me."""

    def __init__(self, user_db: UserDB, supabase_mgr: SupabaseManager = None, parent=None):
        super().__init__(parent)
        self._user_db = user_db
        self._supabase = supabase_mgr
        self._result_data = None
        self.setWindowTitle("SYNDICATE — Login")
        self.setFixedSize(460, 450)

        # Set window icon
        icon_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "icon.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {BG_VOID};
            }}
        """)
        self._setup_ui()
        self._load_remembered_user()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(50, 36, 50, 30)
        layout.setSpacing(6)

        # Logo
        logo = QLabel("SYNDICATE")
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo.setStyleSheet(f"""
            color: {ORANGE_PRIMARY};
            font-size: 28px;
            font-weight: bold;
            font-family: "{FONT_FAMILY_DISPLAY}";
            letter-spacing: 6px;
        """)
        layout.addWidget(logo)

        subtitle = QLabel("COMMAND CENTER")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet(f"""
            color: {TEXT_MUTED};
            font-size: 10px;
            font-family: "{FONT_FAMILY_DISPLAY}";
            letter-spacing: 4px;
        """)
        layout.addWidget(subtitle)
        layout.addSpacing(28)

        # Separator
        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background-color: {BORDER_MEDIUM};")
        layout.addWidget(sep)
        layout.addSpacing(18)

        # Username label + input
        user_label = QLabel("USERNAME")
        user_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 10px; font-family: '{FONT_FAMILY_DISPLAY}'; letter-spacing: 1px; margin-bottom: 2px;")
        layout.addWidget(user_label)

        self._username = QLineEdit()
        self._username.setPlaceholderText("Enter your username")
        self._username.setMinimumHeight(44)
        self._username.setStyleSheet(f"""
            QLineEdit {{
                background-color: {BG_TERTIARY};
                color: {TEXT_PRIMARY};
                border: 1px solid {BORDER_MEDIUM};
                border-radius: 6px;
                padding: 10px 16px;
                font-family: "{FONT_FAMILY_DISPLAY}";
                font-size: 14px;
            }}
            QLineEdit:focus {{ border-color: {ORANGE_PRIMARY}; }}
            QLineEdit::placeholder {{ color: {TEXT_MUTED}; font-size: 12px; }}
        """)
        layout.addWidget(self._username)
        layout.addSpacing(10)

        # Password label + input
        pw_label = QLabel("PASSWORD")
        pw_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 10px; font-family: '{FONT_FAMILY_DISPLAY}'; letter-spacing: 1px; margin-bottom: 2px;")
        layout.addWidget(pw_label)

        self._password = QLineEdit()
        self._password.setPlaceholderText("Enter your password")
        self._password.setEchoMode(QLineEdit.EchoMode.Password)
        self._password.setMinimumHeight(44)
        self._password.setStyleSheet(self._username.styleSheet())
        self._password.returnPressed.connect(self._on_login)
        layout.addWidget(self._password)

        layout.addSpacing(8)

        # Remember me checkbox
        self._remember_me = QCheckBox("Recordarme")
        self._remember_me.setStyleSheet(f"""
            QCheckBox {{
                color: {TEXT_SECONDARY};
                font-size: 11px;
                font-family: "{FONT_FAMILY_DISPLAY}";
                spacing: 8px;
            }}
            QCheckBox::indicator {{
                width: 16px;
                height: 16px;
                border: 1px solid {BORDER_MEDIUM};
                border-radius: 3px;
                background-color: {BG_TERTIARY};
            }}
            QCheckBox::indicator:checked {{
                background-color: {ORANGE_PRIMARY};
                border-color: {ORANGE_PRIMARY};
            }}
            QCheckBox::indicator:hover {{
                border-color: {ORANGE_PRIMARY};
            }}
        """)
        layout.addWidget(self._remember_me)

        layout.addSpacing(6)

        # Error label
        self._error_lbl = QLabel("")
        self._error_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._error_lbl.setStyleSheet(f"color: {RED_LOSS}; font-size: 11px; font-family: '{FONT_FAMILY_DISPLAY}';")
        self._error_lbl.setMinimumHeight(20)
        layout.addWidget(self._error_lbl)

        # Login button
        login_btn = QPushButton("LOGIN")
        login_btn.setMinimumHeight(46)
        login_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {ORANGE_PRIMARY};
                color: #0D1117;
                border: none;
                border-radius: 6px;
                font-size: 14px;
                font-weight: bold;
                font-family: "{FONT_FAMILY_DISPLAY}";
                letter-spacing: 3px;
            }}
            QPushButton:hover {{ background-color: #FF6B2B; }}
            QPushButton:pressed {{ background-color: #C44500; }}
        """)
        login_btn.clicked.connect(self._on_login)
        layout.addWidget(login_btn)

        layout.addStretch()

        # Version
        ver = QLabel(f"v{APP_VERSION}")
        ver.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ver.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 9px; font-family: '{FONT_FAMILY_DISPLAY}';")
        layout.addWidget(ver)

    def _on_login(self):
        username = self._username.text().strip()
        password = self._password.text()

        if not username or not password:
            self._error_lbl.setText("Please enter username and password")
            return

        result = None

        # 1) Try Supabase Auth first (sign_in_with_password)
        if self._supabase and self._supabase.is_available and self._supabase._client:
            supa_result = self._supabase.sign_in_with_password(username, password)
            if supa_result:
                result = {
                    "username": supa_result["username"],
                    "role": supa_result.get("role", "analyst"),
                    "totp_enabled": False,
                    "auth_source": "supabase",
                }
                logger.info(f"Supabase auth success for: {username}")

        # 2) Fallback to local encrypted DB if Supabase didn't work
        if result is None:
            local_result = self._user_db.authenticate(username, password)
            if local_result:
                result = local_result
                result["auth_source"] = "local"

        if result:
            # Save or clear remember-me preference
            self._save_remembered_user(username if self._remember_me.isChecked() else None)
            self._result_data = result
            self.accept()
        else:
            self._error_lbl.setText("Invalid credentials or account locked")
            self._password.clear()
            self._password.setFocus()

    def _load_remembered_user(self):
        """Load saved username from remember-me file."""
        try:
            path = str(REMEMBER_ME_PATH)
            if os.path.exists(path):
                with open(path, "r") as f:
                    data = json.load(f)
                saved_user = data.get("username", "")
                if saved_user:
                    self._username.setText(saved_user)
                    self._remember_me.setChecked(True)
                    self._password.setFocus()
        except Exception:
            pass

    @staticmethod
    def _save_remembered_user(username: str | None):
        """Persist or clear the remembered username."""
        path = str(REMEMBER_ME_PATH)
        try:
            if username:
                with open(path, "w") as f:
                    json.dump({"username": username}, f)
            else:
                if os.path.exists(path):
                    os.remove(path)
        except Exception:
            pass

    @property
    def auth_result(self):
        return self._result_data


# ─────────────────────────────────────────────────────────────────────────────
# Main Application Window
# ─────────────────────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    """The central application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.setMinimumSize(1280, 800)
        self.resize(1440, 900)

        # Set window icon
        icon_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "icon.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        # Prevent premature window destruction
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

        # Apply global stylesheet
        self.setStyleSheet(STYLESHEET)

        # ── Initialize services ─────────────────────────
        self._user_db = UserDB(str(USER_DB_PATH), str(FERNET_KEY_PATH))
        self._session = SessionManager(str(SESSION_LOG_PATH), str(FERNET_KEY_PATH))
        self._pick_mgr = PickManager(str(PICKS_DB_PATH))
        self._scheduler = SchedulerEngine(str(SCHEDULER_DB_PATH))

        # Supabase cloud integration
        self._supabase = SupabaseManager(SUPABASE_URL, SUPABASE_KEY)
        if self._supabase.is_available:
            if self._supabase.connect():
                self._supabase.load_env_from_cloud()
                logger.info("Supabase connected — cloud config loaded")
            else:
                logger.warning("Supabase configured but connection failed — running offline")

        # NBA engine (initialized after login)
        self._nba_engine = NBAEngine()
        self._analysis_worker: AnalysisWorker | None = None

        # Picks cache
        self._current_picks: dict[str, PickResult] = {}

        # ── Login ───────────────────────────────────────
        self._current_user = None
        self._is_admin = False

    def show_login(self) -> bool:
        """Show login dialog. Returns True if authenticated."""
        dialog = LoginDialog(self._user_db, supabase_mgr=self._supabase, parent=None)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            result = dialog.auth_result
            self._current_user = result["username"]
            self._is_admin = result["role"] == "admin"

            # Validate against Supabase app_users table if connected
            # (skip if auth already came from Supabase Auth)
            if self._supabase.is_available and self._supabase._client:
                if result.get("auth_source") != "supabase":
                    cloud_user = self._supabase.validate_user(self._current_user)
                    if cloud_user is None:
                        QMessageBox.critical(
                            None, "Access Denied",
                            "Your account is not authorized in the cloud system.\n"
                            "Contact the administrator."
                        )
                        return False

                # Start heartbeat kill-switch
                self._supabase.start_heartbeat(
                    self._current_user,
                    SUPABASE_HEARTBEAT_SECONDS,
                    on_kill=self._remote_kill,
                )

            self._session.start_session(result["username"], result["role"])
            self._build_ui()
            return True
        return False

    def _remote_kill(self):
        """Called by Supabase heartbeat when user is disabled remotely."""
        logger.warning("Remote kill triggered — user disabled in Supabase")
        # Use QTimer to safely close from the main thread
        QTimer.singleShot(0, self._force_close_remote)

    def _force_close_remote(self):
        QMessageBox.critical(
            self, "Session Terminated",
            "Your account has been disabled by the administrator.\n"
            "The application will now close."
        )
        self._session.end_session()
        QApplication.instance().quit()

    def _build_ui(self):
        """Build the main UI after successful login."""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ── Sidebar ─────────────────────────────────────
        self._sidebar = Sidebar(is_admin=self._is_admin)
        self._sidebar.page_changed.connect(self._navigate)
        main_layout.addWidget(self._sidebar)

        # ── Stacked pages ───────────────────────────────
        self._stack = QStackedWidget()
        self._stack.setStyleSheet(f"background-color: {BG_VOID};")

        # Dashboard
        self._dashboard = DashboardPanel()
        self._dashboard.run_analysis.connect(self._run_analysis)
        self._dashboard.pick_selected.connect(self._open_war_room)
        self._stack.addWidget(self._dashboard)

        # War Room
        self._war_room = WarRoomPanel()
        self._war_room.back_requested.connect(lambda: self._navigate("dashboard"))
        self._war_room.pick_approved.connect(self._approve_pick)
        self._war_room.pick_rejected.connect(self._reject_pick)
        self._stack.addWidget(self._war_room)

        # Scheduler
        self._scheduler_panel = SchedulerPanel(self._scheduler)
        self._stack.addWidget(self._scheduler_panel)

        # Settings
        self._settings = SettingsPanel()
        self._settings.settings_saved.connect(self._on_settings_saved)
        self._settings.train_requested.connect(self._run_training)
        self._stack.addWidget(self._settings)

        # Admin (if applicable)
        if self._is_admin:
            self._admin = AdminPanel(self._user_db, self._session)
            self._stack.addWidget(self._admin)

        main_layout.addWidget(self._stack, 1)

        # ── Page index map ──────────────────────────────
        self._page_map = {
            "dashboard": 0,
            "war_room": 1,
            "scheduler": 2,
            "settings": 3,
        }
        if self._is_admin:
            self._page_map["admin"] = 4

        # ── Scheduler timer (checks every 60 seconds) ──
        self._scheduler_timer = QTimer(self)
        self._scheduler_timer.timeout.connect(self._check_scheduler)
        self._scheduler_timer.start(60_000)

        # Register NBA callback
        self._scheduler.register_callback("NBA", self._run_analysis)

        # ── Defer engine init until after window is visible ──
        # Use singleShot so the event loop has started and UI is painted
        QTimer.singleShot(200, self._init_engine)

    # ── Navigation ──────────────────────────────────────────

    def _navigate(self, page_key: str):
        idx = self._page_map.get(page_key, 0)
        self._stack.setCurrentIndex(idx)
        if page_key == "admin" and self._is_admin:
            self._admin._refresh_all()

    def _open_war_room(self, pick_id: str):
        pick = self._current_picks.get(pick_id)
        if pick:
            # Enrich with detailed analysis
            pick = self._nba_engine.get_detailed_analysis(pick)
            self._war_room.load_pick(pick)
            self._navigate("war_room")

    # ── Engine initialization ───────────────────────────────

    def _init_engine(self):
        """Initialize the NBA engine in a background thread."""
        self._dashboard.set_status("Initializing NBA engine...", ORANGE_PRIMARY)
        db_path = str(DB_PATH)
        model_dir = str(MODEL_DIR)

        # Check if DB exists
        import os
        if not os.path.exists(db_path):
            self._dashboard.set_status(
                f"Database not found at {db_path} — configure in Settings",
                RED_LOSS,
            )
            return

        # Run init in a background thread to avoid freezing the GUI
        self._init_worker = EngineInitWorker(self._nba_engine, db_path, model_dir)
        self._init_worker.finished.connect(self._on_engine_ready)
        self._init_worker.error.connect(self._on_engine_error)
        self._init_worker.start()

    def _on_engine_ready(self):
        self._dashboard.set_status(
            "NBA engine ready — Click RUN ANALYSIS to generate picks",
            GREEN_PROFIT,
        )
        logger.info("NBA engine initialized successfully")

    def _on_engine_error(self, msg: str):
        self._dashboard.set_status(
            f"Engine init failed: {msg}",
            RED_LOSS,
        )
        logger.error(f"NBA engine init failed: {msg}")

    # ── Analysis ────────────────────────────────────────────

    def _run_analysis(self):
        if not self._nba_engine.is_ready:
            QMessageBox.warning(
                self, "Not Ready",
                "NBA engine is not initialized. Check database path in Settings."
            )
            return

        if self._analysis_worker and self._analysis_worker.isRunning():
            return

        self._dashboard.set_running(True)
        self._session.log_event(self._current_user, "run_analysis")

        self._analysis_worker = AnalysisWorker(self._nba_engine)
        self._analysis_worker.finished.connect(self._on_analysis_done)
        self._analysis_worker.error.connect(self._on_analysis_error)
        self._analysis_worker.progress.connect(
            lambda msg: self._dashboard.set_status(msg, ORANGE_PRIMARY)
        )
        self._analysis_worker.start()

    def _on_analysis_done(self, picks: list[PickResult]):
        self._dashboard.set_running(False)

        # Cache picks
        self._current_picks = {p.pick_id: p for p in picks}

        # Save to DB
        self._pick_mgr.save_picks(picks)

        # Update dashboard
        self._dashboard.load_picks(picks)

        self._session.log_event(
            self._current_user, "analysis_complete", True,
            f"{len(picks)} picks generated"
        )

    def _on_analysis_error(self, error_msg: str):
        self._dashboard.set_running(False)
        self._dashboard.set_status(f"Analysis failed: {error_msg}", RED_LOSS)
        self._session.log_event(self._current_user, "analysis_error", False, error_msg)

    # ── Pick actions ────────────────────────────────────────

    def _approve_pick(self, pick_id: str, notes: str):
        self._pick_mgr.update_status(pick_id, PickStatus.APPROVED, notes)
        if pick_id in self._current_picks:
            self._current_picks[pick_id].status = PickStatus.APPROVED
            self._current_picks[pick_id].expert_notes = notes
        self._session.log_event(self._current_user, "approve_pick", True, pick_id)
        QMessageBox.information(self, "Pick Approved", f"Pick {pick_id} has been APPROVED.")

    def _reject_pick(self, pick_id: str, notes: str):
        self._pick_mgr.update_status(pick_id, PickStatus.REJECTED, notes)
        if pick_id in self._current_picks:
            self._current_picks[pick_id].status = PickStatus.REJECTED
            self._current_picks[pick_id].expert_notes = notes
        self._session.log_event(self._current_user, "reject_pick", True, pick_id)
        QMessageBox.information(self, "Pick Rejected", f"Pick {pick_id} has been REJECTED.")

    # ── Settings ────────────────────────────────────────────

    def _on_settings_saved(self):
        self._session.log_event(self._current_user, "settings_saved")

    # ── Model Training ──────────────────────────────────────

    def _run_training(self):
        """Launch model training in a background thread."""
        if hasattr(self, '_train_worker') and self._train_worker and self._train_worker.isRunning():
            QMessageBox.information(self, "Training", "Training is already in progress.")
            return

        db_path = self._settings.get_db_path()
        model_dir = str(MODEL_DIR)

        self._dashboard.set_status("Training model — this may take several minutes...", YELLOW_WARN)
        self._session.log_event(self._current_user, "train_model_start")

        self._train_worker = TrainWorker(db_path, model_dir)
        self._train_worker.finished.connect(self._on_train_done)
        self._train_worker.error.connect(self._on_train_error)
        self._train_worker.progress.connect(
            lambda msg: self._dashboard.set_status(msg, YELLOW_WARN)
        )
        self._train_worker.start()

    def _on_train_done(self):
        self._dashboard.set_status("Model training complete! Reloading engine...", GREEN_PROFIT)
        self._session.log_event(self._current_user, "train_model_complete")
        self._settings.check_model_status()

        # Reinitialize engine to use the newly trained model
        QMessageBox.information(
            self, "Training Complete",
            "The XGBoost model has been trained successfully.\n"
            "The engine will now reload with the trained model."
        )
        self._init_engine()

    def _on_train_error(self, msg: str):
        self._dashboard.set_status(f"Training failed: {msg}", RED_LOSS)
        self._session.log_event(self._current_user, "train_model_error", False, msg)
        QMessageBox.warning(self, "Training Failed", f"Error during training:\n{msg}")

    # ── Scheduler ───────────────────────────────────────────

    def _check_scheduler(self):
        """Called every 60 seconds to check for due rules."""
        due = self._scheduler.check_pending()
        for rule in due:
            logger.info(f"Scheduler firing rule: {rule.name}")
            self._scheduler.fire_rule(rule)

    # ── Cleanup ─────────────────────────────────────────────

    def closeEvent(self, event):
        if self._session.current_user:
            self._session.end_session()
        # Stop Supabase heartbeat
        if self._supabase:
            self._supabase.stop_heartbeat()
        # Stop any running workers
        for worker_attr in ('_analysis_worker', '_init_worker', '_train_worker'):
            worker = getattr(self, worker_attr, None)
            if worker and worker.isRunning():
                worker.quit()
                worker.wait(3000)
        # Stop scheduler timer
        if hasattr(self, '_scheduler_timer'):
            self._scheduler_timer.stop()
        super().closeEvent(event)
