"""
SYNDICATE — Session Management & Access Logging
Encrypted access logs with IP/hostname tracking.
"""
from __future__ import annotations
import json
import socket
import platform
import logging
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path

from cryptography.fernet import Fernet

logger = logging.getLogger("Session")


class SessionManager:
    """Manages active sessions and encrypted access logs."""

    def __init__(self, log_path: str, key_path: str):
        self.log_path = Path(log_path)
        self.key_path = Path(key_path)
        self._fernet: Optional[Fernet] = None
        self._logs: List[Dict] = []
        self._active_session: Optional[Dict] = None
        self._init_crypto()
        self._load_logs()

    def _init_crypto(self):
        if self.key_path.exists():
            key = self.key_path.read_bytes()
        else:
            key = Fernet.generate_key()
            self.key_path.parent.mkdir(parents=True, exist_ok=True)
            self.key_path.write_bytes(key)
        self._fernet = Fernet(key)

    def _load_logs(self):
        if not self.log_path.exists():
            self._logs = []
            return
        try:
            encrypted = self.log_path.read_bytes()
            decrypted = self._fernet.decrypt(encrypted)
            self._logs = json.loads(decrypted.decode("utf-8"))
        except Exception:
            self._logs = []

    def _save_logs(self):
        try:
            data = json.dumps(self._logs).encode("utf-8")
            encrypted = self._fernet.encrypt(data)
            self.log_path.write_bytes(encrypted)
        except Exception as e:
            logger.error(f"Failed to save logs: {e}")

    @staticmethod
    def _get_system_info() -> Dict:
        """Gather IP address and hostname."""
        try:
            hostname = socket.gethostname()
            ip = socket.gethostbyname(hostname)
        except Exception:
            hostname = platform.node() or "unknown"
            ip = "127.0.0.1"
        return {
            "ip": ip,
            "hostname": hostname,
            "os": f"{platform.system()} {platform.release()}",
            "machine": platform.machine(),
        }

    def log_event(self, username: str, action: str, success: bool = True, details: str = ""):
        """Record an access event."""
        info = self._get_system_info()
        entry = {
            "timestamp": datetime.now().isoformat(),
            "username": username,
            "action": action,
            "success": success,
            "ip": info["ip"],
            "hostname": info["hostname"],
            "os": info["os"],
            "details": details,
        }
        self._logs.append(entry)
        self._save_logs()
        logger.info(f"Access log: {username} | {action} | {'OK' if success else 'FAIL'} | {info['ip']}")

    def start_session(self, username: str, role: str):
        """Begin an active session."""
        self._active_session = {
            "username": username,
            "role": role,
            "started": datetime.now().isoformat(),
            "system": self._get_system_info(),
        }
        self.log_event(username, "login", True)

    def end_session(self):
        """End the active session."""
        if self._active_session:
            self.log_event(self._active_session["username"], "logout", True)
            self._active_session = None

    @property
    def current_user(self) -> Optional[str]:
        return self._active_session["username"] if self._active_session else None

    @property
    def current_role(self) -> Optional[str]:
        return self._active_session["role"] if self._active_session else None

    @property
    def is_admin(self) -> bool:
        return self._active_session is not None and self._active_session.get("role") == "admin"

    def get_logs(self, limit: int = 200) -> List[Dict]:
        """Return last N log entries (for admin dashboard)."""
        return list(reversed(self._logs[-limit:]))

    def get_login_stats(self) -> Dict:
        """Statistics for admin dashboard."""
        logins = [l for l in self._logs if l["action"] == "login"]
        failed = [l for l in logins if not l["success"]]
        unique_users = set(l["username"] for l in logins if l["success"])
        unique_ips = set(l["ip"] for l in self._logs)
        return {
            "total_logins": len(logins),
            "failed_logins": len(failed),
            "unique_users": len(unique_users),
            "unique_ips": len(unique_ips),
            "total_events": len(self._logs),
        }
