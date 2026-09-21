"""Role-based authentication (Admin / Researcher / Moderator) with JWT bearer tokens."""

from __future__ import annotations

from datetime import datetime, timedelta

import bcrypt
import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.orm import Session

from sentinai.config import get_settings
from sentinai.schemas import Role
from sentinai.storage.db import get_session
from sentinai.storage.models import UserRow

_bearer = HTTPBearer(auto_error=False)

# Some reverse proxies / preview tunnels consume or rewrite the ``Authorization`` header before
# the request reaches the API.  The dashboard therefore also sends the token in a custom header,
# and the API additionally accepts a cookie set at login.  All three carry the same JWT.
TOKEN_HEADER = "X-SentinAI-Token"
TOKEN_COOKIE = "sentinai_token"

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


def candidate_tokens(request: Request, creds: HTTPAuthorizationCredentials | None) -> list[str]:
    """Tokens presented by the client, in order of preference (bearer header, custom header, cookie)."""
    out: list[str] = []
    if creds is not None and creds.credentials:
        out.append(creds.credentials.strip())
    alt = request.headers.get(TOKEN_HEADER)
    if alt and alt.strip() not in out:
        out.append(alt.strip())
    cookie = request.cookies.get(TOKEN_COOKIE)
    if cookie and cookie not in out:
        out.append(cookie)
    return out


def get_current_user(request: Request, creds: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> CurrentUser:
    tokens = candidate_tokens(request, creds)
    if not tokens:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing bearer token", headers={"WWW-Authenticate": "Bearer"})
    error = "Invalid token"
    for token in tokens:  # a proxy may have replaced the bearer header — fall through to the alternatives
        try:
            data = decode_token(token)
        except jwt.ExpiredSignatureError:
            error = "Token expired"
            continue
        except jwt.PyJWTError:
            continue
        return CurrentUser(data["sub"], data.get("role", Role.RESEARCHER.value))
    raise HTTPException(status.HTTP_401_UNAUTHORIZED, error, headers={"WWW-Authenticate": "Bearer"})


def session_cookie_params(request: Request) -> dict:
    """Cookie attributes that work both on plain http (dev) and behind an https proxy / in an iframe."""
    secure = request.url.scheme == "https" or request.headers.get("x-forwarded-proto", "").split(",")[0].strip() == "https"
    return {"key": TOKEN_COOKIE, "httponly": True, "secure": secure, "samesite": "none" if secure else "lax", "path": "/", "max_age": get_settings().jwt_expire_minutes * 60}


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


__all__ = ["CurrentUser", "PERMISSIONS", "TOKEN_COOKIE", "TOKEN_HEADER", "authenticate", "bootstrap_users", "candidate_tokens", "create_token", "get_current_user", "get_session", "require", "session_cookie_params"]
