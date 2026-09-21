"""Role-based authentication (Admin / Researcher / Moderator) with JWT bearer tokens."""

from __future__ import annotations

from datetime import datetime, timedelta

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.orm import Session

from sentinai.config import get_settings
from sentinai.schemas import Role
from sentinai.storage.db import get_session
from sentinai.storage.models import UserRow

_bearer = HTTPBearer(auto_error=False)

# Capability matrix — what each role may do.
PERMISSIONS: dict[str, set[str]] = {
    Role.ADMIN.value: {"read", "review", "annotate", "export", "ingest", "manage_users", "retrain", "audit"},
    Role.RESEARCHER.value: {"read", "export", "audit"},
    Role.MODERATOR.value: {"read", "review", "annotate", "export"},
}


def hash_password(pw: str) -> str:
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt(rounds=10)).decode()


def verify_password(pw: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(pw.encode(), hashed.encode())
    except ValueError:
        return False


def create_token(username: str, role: str) -> str:
    s = get_settings()
    payload = {"sub": username, "role": role, "exp": datetime.utcnow() + timedelta(minutes=s.jwt_expire_minutes), "iat": datetime.utcnow()}
    return jwt.encode(payload, s.jwt_secret, algorithm=s.jwt_algorithm)


def decode_token(token: str) -> dict:
    s = get_settings()
    return jwt.decode(token, s.jwt_secret, algorithms=[s.jwt_algorithm])


class CurrentUser:
    def __init__(self, username: str, role: str):
        self.username = username
        self.role = role

    def can(self, perm: str) -> bool:
        return perm in PERMISSIONS.get(self.role, set())


def get_current_user(creds: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> CurrentUser:
    if creds is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing bearer token", headers={"WWW-Authenticate": "Bearer"})
    try:
        data = decode_token(creds.credentials)
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token expired") from exc
    except jwt.PyJWTError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid token") from exc
    return CurrentUser(data["sub"], data.get("role", Role.RESEARCHER.value))


def require(perm: str):
    def _dep(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        if not user.can(perm):
            raise HTTPException(status.HTTP_403_FORBIDDEN, f"Role '{user.role}' lacks permission '{perm}'")
        return user

    return _dep


def authenticate(session: Session, username: str, password: str) -> UserRow | None:
    user = session.scalar(select(UserRow).where(UserRow.username == username, UserRow.active.is_(True)))
    if user and verify_password(password, user.password_hash):
        return user
    return None


def bootstrap_users(session: Session) -> int:
    """Create the default dev accounts if the users table is empty."""
    s = get_settings()
    if session.scalar(select(UserRow).limit(1)) is not None:
        return 0
    defaults = [("admin", s.bootstrap_admin_password, Role.ADMIN), ("researcher", s.bootstrap_researcher_password, Role.RESEARCHER), ("moderator", s.bootstrap_moderator_password, Role.MODERATOR)]
    for name, pw, role in defaults:
        session.add(UserRow(username=name, password_hash=hash_password(pw), role=role.value))
    return len(defaults)


__all__ = ["CurrentUser", "PERMISSIONS", "authenticate", "bootstrap_users", "create_token", "get_current_user", "get_session", "require"]
