"""Bearer-token resolution for the X API.

Order of precedence: token passed explicitly → ``SENTINAI_X_BEARER_TOKEN`` (env / .env) →
token set at runtime through the dashboard (kept in memory and, optionally, in
``data/secrets/x_bearer_token`` with mode 0600 so it survives restarts).  The token is never
returned by the API; :func:`token_status` only exposes a masked hint.
"""

from __future__ import annotations

import logging
import os
import stat
import threading
from pathlib import Path

from sentinai.config import get_settings

log = logging.getLogger(__name__)

_lock = threading.Lock()
_runtime_token: str | None = None
_runtime_source: str | None = None


def _token_file() -> Path:
    return get_settings().data_dir / "secrets" / "x_bearer_token"


def _load_file_token() -> str | None:
    p = _token_file()
    try:
        if p.exists():
            tok = p.read_text(encoding="utf-8").strip()
            return tok or None
    except OSError as exc:  # pragma: no cover
        log.warning("could not read %s: %s", p, exc)
    return None


def _resolve() -> tuple[str | None, str | None]:
    """Return ``(token, source)`` — source is ``runtime`` / ``env`` / ``file`` / ``None``."""
    global _runtime_token, _runtime_source
    with _lock:
        if _runtime_token:
            return _runtime_token, _runtime_source or "runtime"
    env = get_settings().x_bearer_token
    if env:
        return env, "env"
    tok = _load_file_token()
    if tok:
        with _lock:
            _runtime_token, _runtime_source = tok, "file"
        return tok, "file"
    return None, None


def get_bearer_token() -> str | None:
    return _resolve()[0]


def set_bearer_token(token: str, persist: bool = False) -> dict:
    global _runtime_token, _runtime_source
    token = token.strip()
    if len(token) < 20:
        raise ValueError("That does not look like an X API bearer token (too short)")
    with _lock:
        _runtime_token, _runtime_source = token, "runtime"
    if persist:
        p = _token_file()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(token, encoding="utf-8")
        try:
            os.chmod(p, stat.S_IRUSR | stat.S_IWUSR)
        except OSError:  # pragma: no cover - windows
            pass
        with _lock:
            _runtime_source = "file"
    return token_status()


def clear_bearer_token() -> dict:
    global _runtime_token, _runtime_source
    with _lock:
        _runtime_token, _runtime_source = None, None
    p = _token_file()
    if p.exists():
        p.unlink()
    return token_status()


def token_status() -> dict:
    """Where the active token comes from (never the token itself)."""
    tok, source = _resolve()
    return {"configured": bool(tok), "source": source, "hint": f"…{tok[-4:]}" if tok else None}
