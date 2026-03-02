"""
SYNDICATE — 2FA Preparation (TOTP / Google Authenticator)
==========================================================
Prepares the system for TOTP-based 2FA integration.
When pyotp is available, generates secrets and verifies tokens.
"""
from __future__ import annotations
import logging
from typing import Optional, Tuple

logger = logging.getLogger("TOTP")

try:
    import pyotp
    TOTP_AVAILABLE = True
except ImportError:
    TOTP_AVAILABLE = False
    logger.info("pyotp not installed — 2FA prep mode only")


class TOTPManager:
    """
    TOTP (Time-based One-Time Password) manager.
    Compatible with Google Authenticator, Authy, etc.
    """

    APP_NAME = "SyndicateCC"

    @staticmethod
    def is_available() -> bool:
        return TOTP_AVAILABLE

    @staticmethod
    def generate_secret() -> str:
        """Generate a new TOTP secret for a user."""
        if not TOTP_AVAILABLE:
            return ""
        return pyotp.random_base32()

    @staticmethod
    def get_provisioning_uri(username: str, secret: str) -> str:
        """
        Generate the otpauth:// URI for QR code generation.
        User scans this with Google Authenticator.
        """
        if not TOTP_AVAILABLE or not secret:
            return ""
        totp = pyotp.TOTP(secret)
        return totp.provisioning_uri(name=username, issuer_name=TOTPManager.APP_NAME)

    @staticmethod
    def verify_token(secret: str, token: str) -> bool:
        """Verify a TOTP token against the secret."""
        if not TOTP_AVAILABLE or not secret:
            return False
        try:
            totp = pyotp.TOTP(secret)
            return totp.verify(token, valid_window=1)  # ±30 seconds tolerance
        except Exception:
            return False

    @staticmethod
    def get_current_token(secret: str) -> str:
        """Get current token (for testing only)."""
        if not TOTP_AVAILABLE or not secret:
            return ""
        return pyotp.TOTP(secret).now()
