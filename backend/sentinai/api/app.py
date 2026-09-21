"""FastAPI application factory."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from sentinai import __version__
from sentinai.api import routes_analytics, routes_core, routes_review
from sentinai.auth import bootstrap_users
from sentinai.config import REPO_DIR, get_settings
from sentinai.storage.db import init_db, session_scope

log = logging.getLogger("sentinai")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    init_db()
    with session_scope() as session:
        created = bootstrap_users(session)
        if created:
            log.info("bootstrapped %d default users (admin / researcher / moderator)", created)
    if settings.seed_demo_data:
        from sentinai.demo import seed_if_empty

        seeded = seed_if_empty()
        if seeded:
            log.info("seeded %d synthetic demo posts", seeded)
    # warm the classifier so the first request is fast
    from sentinai.classification.engine import get_engine

    get_engine()
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="SentinAI API", version=__version__, description="Social Media Bias & Hate Speech Intelligence Platform", lifespan=lifespan, docs_url="/api/docs", openapi_url="/api/openapi.json", redoc_url=None)
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
    app.include_router(routes_core.router, prefix="/api")
    app.include_router(routes_analytics.router, prefix="/api")
    app.include_router(routes_review.router, prefix="/api")

    # Serve the built dashboard when present (single-container deployment).
    dist = REPO_DIR / "frontend" / "dist"
    if dist.exists():
        app.mount("/", _SPAStaticFiles(directory=dist, html=True), name="dashboard")
    return app


class _SPAStaticFiles(StaticFiles):
    """Static files with SPA fallback to index.html for client-side routes."""

    async def get_response(self, path: str, scope):  # type: ignore[override]
        from starlette.exceptions import HTTPException as StarletteHTTPException

        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            # Unknown API routes must stay 404 (never fall back to the HTML shell).
            if exc.status_code == 404 and not Path(path).suffix and not path.startswith("api/"):
                return await super().get_response("index.html", scope)
            raise


app = create_app()
