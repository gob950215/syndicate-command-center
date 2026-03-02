"""
SYNDICATE — Supabase Cloud Integration
========================================
Handles:
  1. User authentication & session validation via Supabase Auth
  2. Remote kill-switch: if admin disables a user in Supabase, app exits
  3. Cloud storage for the DB, trained model, and API keys (env vars)
  4. Heartbeat: periodically checks user is still authorized

Setup in Supabase:
  - Create project at supabase.com
  - Create table `app_users`: id (uuid), username (text), role (text), enabled (bool), created_at (timestamptz)
  - Create table `app_config`: key (text PK), value (text), updated_at (timestamptz)
  - Create bucket `syndicate-files` in Storage for DB/model files
  - Set your SUPABASE_URL and SUPABASE_KEY in environment or config
"""
from __future__ import annotations
import os
import json
import logging
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger("Supabase")

# Try to import supabase client
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    logger.warning("supabase-py not installed. Run: pip install supabase")


class SupabaseManager:
    """
    Manages all Supabase interactions.

    Environment variables required:
        SUPABASE_URL   - Your Supabase project URL
        SUPABASE_KEY   - Your Supabase anon/service key

    Tables expected:
        app_users  (id, username, role, enabled, created_at)
        app_config (key, value, updated_at)

    Storage bucket:
        syndicate-files  (for .db and .pkl files)
    """

    def __init__(self, url: str = None, key: str = None):
        self._url = url or os.environ.get("SUPABASE_URL", "")
        self._key = key or os.environ.get("SUPABASE_KEY", "")
        self._client: Optional[Client] = None
        self._current_user: Optional[str] = None
        self._heartbeat_timer: Optional[threading.Timer] = None
        self._on_kill_callback = None

    @property
    def is_available(self) -> bool:
        return SUPABASE_AVAILABLE and bool(self._url) and bool(self._key)

    def connect(self) -> bool:
        """Initialize Supabase client connection."""
        if not SUPABASE_AVAILABLE:
            logger.warning("supabase-py not installed")
            return False
        if not self._url or not self._key:
            logger.warning("SUPABASE_URL or SUPABASE_KEY not set")
            return False
        try:
            self._client = create_client(self._url, self._key)
            logger.info("Supabase connected")
            return True
        except Exception as e:
            logger.error(f"Supabase connection failed: {e}")
            return False

    # ── User Authentication (Supabase Auth) ────────────────────────

    def sign_in_with_password(self, email: str, password: str) -> dict | None:
        """
        Authenticate user via Supabase Auth (supabase.auth.sign_in_with_password).
        Returns dict with user info on success, None on failure.
        The `email` parameter can be an email address or a username —
        if it doesn't contain '@', we append a default domain.
        """
        if not self._client:
            return None

        # Normalize: if user typed a plain username, convert to email format
        login_email = email if "@" in email else f"{email}@syndicate.app"

        try:
            response = self._client.auth.sign_in_with_password({
                "email": login_email,
                "password": password,
            })

            if response and response.user:
                user_meta = response.user.user_metadata or {}
                return {
                    "id": response.user.id,
                    "email": response.user.email,
                    "username": user_meta.get("username", email),
                    "role": user_meta.get("role", "analyst"),
                    "access_token": response.session.access_token if response.session else None,
                }
            return None
        except Exception as e:
            logger.error(f"Supabase sign_in_with_password failed: {e}")
            return None

    # ── User Validation ─────────────────────────────────────────

    def validate_user(self, username: str) -> dict | None:
        """
        Check if user exists and is enabled in Supabase app_users table.
        Returns user dict or None if not found/disabled.
        """
        if not self._client:
            return None
        try:
            result = (
                self._client.table("app_users")
                .select("*")
                .eq("username", username)
                .eq("enabled", True)
                .execute()
            )
            if result.data and len(result.data) > 0:
                self._current_user = username
                return result.data[0]
            return None
        except Exception as e:
            logger.error(f"User validation error: {e}")
            return None

    def is_user_still_active(self, username: str = None) -> bool:
        """Check if user is still enabled — used by heartbeat."""
        username = username or self._current_user
        if not username or not self._client:
            return True  # If no Supabase, don't block
        try:
            result = (
                self._client.table("app_users")
                .select("enabled")
                .eq("username", username)
                .execute()
            )
            if result.data and len(result.data) > 0:
                return result.data[0].get("enabled", False)
            return False  # User deleted from table
        except Exception as e:
            logger.error(f"Heartbeat check error: {e}")
            return True  # Don't kill on network errors

    # ── Heartbeat / Kill Switch ─────────────────────────────────

    def start_heartbeat(self, username: str, interval: int = 120, on_kill=None):
        """
        Start periodic checks (every `interval` seconds).
        If user is disabled in Supabase, calls `on_kill` callback.
        """
        self._current_user = username
        self._on_kill_callback = on_kill
        self._heartbeat_check()

    def _heartbeat_check(self):
        if not self._current_user:
            return
        if not self.is_user_still_active():
            logger.warning(f"User {self._current_user} has been disabled remotely!")
            if self._on_kill_callback:
                self._on_kill_callback()
            return
        # Schedule next check
        self._heartbeat_timer = threading.Timer(120, self._heartbeat_check)
        self._heartbeat_timer.daemon = True
        self._heartbeat_timer.start()

    def stop_heartbeat(self):
        if self._heartbeat_timer:
            self._heartbeat_timer.cancel()
            self._heartbeat_timer = None

    # ── Cloud Config (API Keys / Env Vars) ──────────────────────

    def get_config(self, key: str) -> str | None:
        """Get a config value from Supabase app_config table."""
        if not self._client:
            return None
        try:
            result = (
                self._client.table("app_config")
                .select("value")
                .eq("key", key)
                .execute()
            )
            if result.data and len(result.data) > 0:
                return result.data[0]["value"]
            return None
        except Exception as e:
            logger.error(f"Get config error: {e}")
            return None

    def set_config(self, key: str, value: str) -> bool:
        """Save a config value to Supabase app_config table."""
        if not self._client:
            return False
        try:
            self._client.table("app_config").upsert({
                "key": key,
                "value": value,
            }).execute()
            return True
        except Exception as e:
            logger.error(f"Set config error: {e}")
            return False

    def load_env_from_cloud(self):
        """Load API keys from Supabase into environment variables."""
        keys_to_load = ["SPORTS_API_KEY", "ODDS_API_KEY"]
        loaded = 0
        for k in keys_to_load:
            val = self.get_config(k)
            if val:
                os.environ[k] = val
                loaded += 1
                logger.info(f"Loaded {k} from cloud")
        return loaded

    def save_env_to_cloud(self):
        """Save current API keys from environment to Supabase."""
        for k in ["SPORTS_API_KEY", "ODDS_API_KEY"]:
            val = os.environ.get(k, "")
            if val:
                self.set_config(k, val)

    # ── File Storage (DB, Model) ────────────────────────────────

    def upload_file(self, local_path: str, remote_name: str = None) -> bool:
        """Upload a file to Supabase Storage bucket 'syndicate-files'."""
        if not self._client:
            return False
        try:
            remote_name = remote_name or Path(local_path).name
            with open(local_path, "rb") as f:
                self._client.storage.from_("syndicate-files").upload(
                    remote_name, f.read(),
                    file_options={"upsert": "true"}
                )
            logger.info(f"Uploaded {remote_name} to cloud")
            return True
        except Exception as e:
            logger.error(f"Upload error: {e}")
            return False

    def download_file(self, remote_name: str, local_path: str) -> bool:
        """Download a file from Supabase Storage to local path."""
        if not self._client:
            return False
        try:
            data = self._client.storage.from_("syndicate-files").download(remote_name)
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            with open(local_path, "wb") as f:
                f.write(data)
            logger.info(f"Downloaded {remote_name} → {local_path}")
            return True
        except Exception as e:
            logger.error(f"Download error: {e}")
            return False

    def sync_db_to_cloud(self, db_path: str) -> bool:
        """Upload the historical database to cloud storage."""
        return self.upload_file(db_path, "nba_historical.db")

    def sync_model_to_cloud(self, model_path: str) -> bool:
        """Upload the trained model to cloud storage."""
        return self.upload_file(model_path, "nba_model_v8.pkl")

    def download_db(self, local_path: str) -> bool:
        """Download database from cloud."""
        return self.download_file("nba_historical.db", local_path)

    def download_model(self, local_path: str) -> bool:
        """Download model from cloud."""
        return self.download_file("nba_model_v8.pkl", local_path)

    # ── Cleanup ─────────────────────────────────────────────────

    def disconnect(self):
        self.stop_heartbeat()
        self._client = None
        self._current_user = None
