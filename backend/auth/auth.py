import secrets
import logging
from typing import Dict, Optional

from backend.interfaces import UserRole

logger = logging.getLogger(__name__)

# Argon2id password hashing implementation
try:
    from argon2 import PasswordHasher
    ph = PasswordHasher()
    USE_ARGON2 = True
    logger.info("[Auth] Using Argon2id password hasher.")
except Exception as e:
    import bcrypt
    USE_ARGON2 = False
    logger.info(f"[Auth] Argon2id unavailable ({e}). Using bcrypt fallback.")

def hash_password(password: str) -> str:
    """Hashes plain password using Argon2id (or bcrypt fallback)."""
    if USE_ARGON2:
        return ph.hash(password)
    else:
        import bcrypt
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies plain password against stored hashed password."""
    try:
        if USE_ARGON2:
            return ph.verify(hashed_password, plain_password)
        else:
            import bcrypt
            return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
    except Exception:
        return False


class AuthManager:
    """Manages active user session tokens and role-based permission validation."""

    def __init__(self):
        # Structure: token_str -> {"username": str, "role": UserRole}
        self.active_sessions: Dict[str, Dict] = {}

    def create_session(self, username: str, role: UserRole) -> str:
        """Generates a secure random session token."""
        token = secrets.token_hex(24)
        self.active_sessions[token] = {
            "username": username,
            "role": role
        }
        return token

    def get_session(self, token: str) -> Optional[Dict]:
        return self.active_sessions.get(token)

    def validate_role_access(self, token: str, required_role: UserRole) -> bool:
        """Validates if session user has sufficient role permissions."""
        session = self.get_session(token)
        if not session:
            return False

        user_role = session["role"]

        # Admin has access to all pages
        if user_role == UserRole.ADMIN:
            return True

        # Command Operator has access to Command Operator and BOP pages
        if required_role == UserRole.BOP_OPERATOR and user_role in [UserRole.BOP_OPERATOR, UserRole.COMMAND_OPERATOR]:
            return True

        if required_role == UserRole.COMMAND_OPERATOR and user_role == UserRole.COMMAND_OPERATOR:
            return True

        return user_role == required_role
