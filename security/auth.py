"""
SYNDICATE — Authentication & Encrypted User DB
================================================
Uses cryptography.fernet for AES-128-CBC encryption of the user database.
Passwords are hashed with PBKDF2-SHA256 before encryption.
"""
from __future__ import annotations
import json
import hashlib
import secrets
import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List

from cryptography.fernet import Fernet

logger = logging.getLogger("Auth")


class UserDB:
    """Fernet-encrypted local user database."""

    def __init__(self, db_path: str, key_path: str):
        self.db_path = Path(db_path)
        self.key_path = Path(key_path)
        self._fernet: Optional[Fernet] = None
        self._users: Dict[str, Dict] = {}
        self._failed_attempts: Dict[str, List[datetime]] = {}
        self._init_crypto()
        self._load()

    def _init_crypto(self):
        """Load or generate master encryption key."""
        if self.key_path.exists():
            key = self.key_path.read_bytes()
        else:
            key = Fernet.generate_key()
            self.key_path.parent.mkdir(parents=True, exist_ok=True)
            self.key_path.write_bytes(key)
            # Restrict permissions
            try:
                os.chmod(str(self.key_path), 0o600)
            except OSError:
                pass
        self._fernet = Fernet(key)

    def _load(self):
        """Load and decrypt user database."""
        if not self.db_path.exists():
            # Create default admin
            self._users = {}
            self.create_user("admin", "Syndicate2026!", role="admin")
            logger.info("Created default admin user")
            return

        try:
            encrypted = self.db_path.read_bytes()
            decrypted = self._fernet.decrypt(encrypted)
            self._users = json.loads(decrypted.decode("utf-8"))
            logger.info(f"Loaded {len(self._users)} users")
        except Exception as e:
            logger.error(f"Failed to load user DB: {e}")
            self._users = {}

    def _save(self):
        """Encrypt and persist user database."""
        try:
            data = json.dumps(self._users).encode("utf-8")
            encrypted = self._fernet.encrypt(data)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.db_path.write_bytes(encrypted)
        except Exception as e:
            logger.error(f"Failed to save user DB: {e}")

    @staticmethod
    def _hash_password(password: str, salt: str = "") -> tuple:
        """PBKDF2-SHA256 password hashing."""
        if not salt:
            salt = secrets.token_hex(16)
        hashed = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), salt.encode(), 310_000
        ).hex()
        return hashed, salt

    def create_user(self, username: str, password: str, role: str = "analyst") -> bool:
        """Create a new user."""
        if username in self._users:
            return False
        hashed, salt = self._hash_password(password)
        self._users[username] = {
            "hash": hashed,
            "salt": salt,
            "role": role,
            "created": datetime.now().isoformat(),
            "last_login": None,
            "totp_secret": None,  # 2FA prep
            "active": True,
        }
        self._save()
        logger.info(f"User created: {username} (role={role})")
        return True

    def authenticate(self, username: str, password: str) -> Optional[Dict]:
        """
        Authenticate user. Returns user dict on success, None on failure.
        Implements brute-force protection.
        """
        # Check lockout
        if self._is_locked_out(username):
            logger.warning(f"Account locked: {username}")
            return None

        user = self._users.get(username)
        if not user or not user.get("active", True):
            self._record_failure(username)
            return None

        hashed, _ = self._hash_password(password, user["salt"])
        if hashed != user["hash"]:
            self._record_failure(username)
            logger.warning(f"Failed login: {username}")
            return None

        # Success
        self._failed_attempts.pop(username, None)
        user["last_login"] = datetime.now().isoformat()
        self._save()
        logger.info(f"Login success: {username}")
        return {
            "username": username,
            "role": user["role"],
            "totp_enabled": user.get("totp_secret") is not None,
        }

    def _record_failure(self, username: str):
        if username not in self._failed_attempts:
            self._failed_attempts[username] = []
        self._failed_attempts[username].append(datetime.now())

    def _is_locked_out(self, username: str, max_attempts: int = 5, window_min: int = 15) -> bool:
        attempts = self._failed_attempts.get(username, [])
        cutoff = datetime.now() - timedelta(minutes=window_min)
        recent = [a for a in attempts if a > cutoff]
        self._failed_attempts[username] = recent
        return len(recent) >= max_attempts

    def change_password(self, username: str, old_pw: str, new_pw: str) -> bool:
        user = self._users.get(username)
        if not user:
            return False
        hashed, _ = self._hash_password(old_pw, user["salt"])
        if hashed != user["hash"]:
            return False
        new_hash, new_salt = self._hash_password(new_pw)
        user["hash"] = new_hash
        user["salt"] = new_salt
        self._save()
        return True

    def list_users(self) -> List[Dict]:
        return [
            {
                "username": u,
                "role": d["role"],
                "created": d["created"],
                "last_login": d.get("last_login"),
                "active": d.get("active", True),
                "totp_enabled": d.get("totp_secret") is not None,
            }
            for u, d in self._users.items()
        ]

    def delete_user(self, username: str) -> bool:
        if username == "admin":
            return False  # Can't delete master admin
        if username in self._users:
            del self._users[username]
            self._save()
            return True
        return False

    def set_totp_secret(self, username: str, secret: str) -> bool:
        """Store TOTP secret for 2FA (prep for Google Authenticator)."""
        if username in self._users:
            self._users[username]["totp_secret"] = secret
            self._save()
            return True
        return False

    def get_totp_secret(self, username: str) -> Optional[str]:
        user = self._users.get(username)
        return user.get("totp_secret") if user else None
