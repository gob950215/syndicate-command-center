"""
SYNDICATE COMMAND CENTER — Global Configuration
"""
import os
from pathlib import Path

# ── Load .env file ─────────────────────────────────────────────────────────
APP_DIR = Path(__file__).parent.resolve()
_env_path = APP_DIR / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _val = _line.split("=", 1)
                os.environ.setdefault(_key.strip(), _val.strip())

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR = APP_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

DB_PATH = DATA_DIR / "nba_historical.db"
USER_DB_PATH = DATA_DIR / "users.enc"
SESSION_LOG_PATH = DATA_DIR / "access_log.enc"
PICKS_DB_PATH = DATA_DIR / "picks.db"
SCHEDULER_DB_PATH = DATA_DIR / "scheduler.db"
REMEMBER_ME_PATH = DATA_DIR / ".remember_me.json"

MODEL_DIR = APP_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# ── API Keys (from environment) ────────────────────────────────────────────
SPORTS_API_KEY = os.environ.get("SPORTS_API_KEY", "")
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")

# ── App Meta ───────────────────────────────────────────────────────────────
APP_NAME = "SYNDICATE COMMAND CENTER"
APP_VERSION = "1.0.0"
APP_SUBTITLE = "Elite Sports Prediction Platform"

# ── Security ───────────────────────────────────────────────────────────────
FERNET_KEY_PATH = DATA_DIR / ".master.key"
SESSION_TIMEOUT_MINUTES = 120
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_MINUTES = 15

# ── Diamond Thresholds (mirror V8) ────────────────────────────────────────
DIAMOND_THRESHOLD = 0.78
DIAMOND_EV_MIN = 0.10
FALLBACK_THRESHOLD = 0.65

# ── Supabase Cloud Integration ────────────────────────────────────────────
SUPABASE_URL = os.environ.get("SUPABASE_URL", "") or os.environ.get("NEXT_PUBLIC_SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
SUPABASE_HEARTBEAT_SECONDS = 120  # Check every 2 min if user still active

# ── Supported Sports ──────────────────────────────────────────────────────
SUPPORTED_SPORTS = {
    "NBA": {
        "enabled": True,
        "icon": "🏀",
        "color": "#E8590C",
        "module": "nba_syndicate_v8",
    },
    "MLB": {
        "enabled": False,
        "icon": "⚾",
        "color": "#1971C2",
        "module": None,
    },
    "NFL": {
        "enabled": False,
        "icon": "🏈",
        "color": "#2F9E44",
        "module": None,
    },
}
