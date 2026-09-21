"""Shared fixtures: isolated SQLite DB per test session + seeded API client."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

_tmp = tempfile.mkdtemp(prefix="sentinai-test-")
os.environ["SENTINAI_DATABASE_URL"] = f"sqlite:///{Path(_tmp) / 'test.db'}"
os.environ["SENTINAI_DATA_DIR"] = _tmp
os.environ["SENTINAI_SEED_DEMO_DATA"] = "false"
os.environ["SENTINAI_JWT_SECRET"] = "test-secret-key-that-is-long-enough-for-hs256"


@pytest.fixture(scope="session")
def engine():
    from sentinai.classification.engine import ClassificationEngine

    return ClassificationEngine(dedup=False)


@pytest.fixture(scope="session")
def seeded_db():
    from sentinai.demo import seed
    from sentinai.storage.db import init_db

    init_db()
    n = seed(n_posts=250, seed_value=7)
    assert n > 250
    return n


@pytest.fixture(scope="session")
def client(seeded_db):
    from fastapi.testclient import TestClient

    from sentinai.api.app import app

    with TestClient(app) as c:
        yield c


def _token(client, username, password):
    r = client.post("/api/auth/login", json={"username": username, "password": password})
    assert r.status_code == 200, r.text
    client.cookies.clear()  # tests exercise RBAC via explicit headers; keep the shared jar anonymous
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


@pytest.fixture(scope="session")
def admin(client):
    return _token(client, "admin", "admin")


@pytest.fixture(scope="session")
def researcher(client):
    return _token(client, "researcher", "researcher")


@pytest.fixture(scope="session")
def moderator(client):
    return _token(client, "moderator", "moderator")
