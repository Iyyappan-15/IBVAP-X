import pytest
from backend.auth.auth import hash_password, verify_password, AuthManager
from backend.interfaces import UserRole

def test_password_hashing_and_verification():
    raw_pwd = "SecretDemoPassword123!"
    hashed = hash_password(raw_pwd)

    assert hashed != raw_pwd
    assert verify_password(raw_pwd, hashed) is True
    assert verify_password("WrongPassword", hashed) is False

def test_auth_manager_session_and_roles():
    auth_mgr = AuthManager()
    token = auth_mgr.create_session("bop_operator", UserRole.BOP_OPERATOR)
    
    assert token is not None
    session = auth_mgr.get_session(token)
    assert session["username"] == "bop_operator"
    assert session["role"] == UserRole.BOP_OPERATOR

    # Role validation checks
    assert auth_mgr.validate_role_access(token, UserRole.BOP_OPERATOR) is True
    assert auth_mgr.validate_role_access(token, UserRole.ADMIN) is False

    # Admin access check
    admin_token = auth_mgr.create_session("admin", UserRole.ADMIN)
    assert auth_mgr.validate_role_access(admin_token, UserRole.BOP_OPERATOR) is True
    assert auth_mgr.validate_role_access(admin_token, UserRole.COMMAND_OPERATOR) is True
    assert auth_mgr.validate_role_access(admin_token, UserRole.ADMIN) is True
