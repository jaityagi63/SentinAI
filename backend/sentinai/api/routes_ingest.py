"""Live X ingestion, bulk imports and job monitoring (``/api/ingest/*``).

All endpoints need the ``ingest`` permission (admin).  Interactive calls use a bounded
rate-limit wait so the dashboard never hangs; longer pulls run as background jobs.
"""

from __future__ import annotations

import logging
from datetime import datetime

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from sentinai.api.schemas import XSearchRequest, XStreamRequest, XTokenRequest, XTweetRequest, XUserRequest
from sentinai.auth import CurrentUser, require
from sentinai.config import get_settings
from sentinai.ingestion import credentials
from sentinai.ingestion.jobs import Job, registry
from sentinai.ingestion.worker import IngestReport, IngestionWorker
from sentinai.ingestion.x_client import XAccessError, XClient, XNotFoundError, XRateLimitError, parse_tweet_ref, parse_username
from sentinai.storage.db import get_session
from sentinai.storage.models import IngestionCursorRow, PostRow

log = logging.getLogger(__name__)
router = APIRouter(prefix="/ingest", tags=["ingestion"])


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------


def _iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _client(*, full_archive: bool | None = None) -> XClient:
    token = credentials.get_bearer_token()
    if not token:
        raise HTTPException(409, "No X API bearer token configured. Set SENTINAI_X_BEARER_TOKEN or save one under Ingest → Connect X.")
    return XClient(bearer_token=token, max_wait=get_settings().x_api_max_wait_seconds, full_archive=full_archive)


def _translate(exc: Exception) -> HTTPException:
    if isinstance(exc, XRateLimitError):
        return HTTPException(429, str(exc))
    if isinstance(exc, XAccessError):
        return HTTPException(502 if exc.status_code == 402 else 403, str(exc))
    if isinstance(exc, XNotFoundError):
        return HTTPException(404, str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(422, str(exc))
    if isinstance(exc, httpx.HTTPError):
        return HTTPException(502, f"X API request failed: {exc}"[:400])
    if isinstance(exc, RuntimeError):
        return HTTPException(409, str(exc))
    log.exception("ingestion failed")
    return HTTPException(500, f"ingestion failed: {exc}"[:400])


def _run_inline(fn) -> dict:
    try:
        report = fn()
    except Exception as exc:  # noqa: BLE001
        raise _translate(exc) from exc
    return {"job": None, "report": report.to_dict() if isinstance(report, IngestReport) else report}


def _submit(kind: str, label: str, user: CurrentUser, params: dict, run) -> dict:
    job = registry.submit(kind, label, user.username, params, run)
    return {"job": job.to_dict(), "report": None}


# --------------------------------------------------------------------------------------------
# status / credentials
# --------------------------------------------------------------------------------------------


@router.get("/status")
def ingest_status(user: CurrentUser = Depends(require("ingest")), session: Session = Depends(get_session)):
    s = get_settings()
    by_source = dict(session.execute(select(PostRow.source, func.count()).where(PostRow.deleted_upstream.is_(False)).group_by(PostRow.source)).all())
    cursors = session.scalars(select(IngestionCursorRow).order_by(IngestionCursorRow.last_run.desc().nullslast())).all()
    active = registry.active()
    return {
        "token": credentials.token_status(),
        "api_base": s.x_api_base,
        "full_archive_default": s.x_full_archive,
        "download_media_default": s.x_download_media,
        "max_wait_seconds": s.x_api_max_wait_seconds,
        "upload_max_mb": s.ingest_upload_max_mb,
        "posts_by_source": {k or "x": v for k, v in by_source.items()},
        "cursors": [{"query": c.query, "since_id": c.since_id, "last_run": c.last_run.isoformat() if c.last_run else None, "total_ingested": c.total_ingested or 0} for c in cursors[:50]],
        "active_jobs": [j.to_dict() for j in active],
        "stream_running": any(j.kind == "stream" for j in active),
        "endpoints": {
            "search": "/tweets/search/all" if s.x_full_archive else "/tweets/search/recent",
            "tweet": "/tweets/:id",
            "user": "/users/:id/tweets",
            "stream": "/tweets/search/stream",
        },
    }


@router.post("/x/token")
def set_token(body: XTokenRequest, user: CurrentUser = Depends(require("ingest"))):
    try:
        status = credentials.set_bearer_token(body.bearer_token, persist=body.persist)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return status


@router.delete("/x/token")
def clear_token(user: CurrentUser = Depends(require("ingest"))):
    return credentials.clear_bearer_token()


@router.post("/x/test")
def test_connection(user: CurrentUser = Depends(require("ingest"))):
    """Cheap connectivity / access-level probe: one recent-search request with max_results=10."""
    client = _client()
    try:
        started = datetime.utcnow()
        posts = list(client.search("hello lang:en -is:retweet", max_results=10, max_pages=1, full_archive=False))
        recent_ok = True
        full_archive = None
        if get_settings().x_full_archive:
            try:
                list(client.search("hello lang:en", max_results=10, max_pages=1, full_archive=True))
                full_archive = client.full_archive_available
            except (XAccessError, XRateLimitError):
                full_archive = False
        return {
            "ok": True,
            "recent_search": recent_ok,
            "full_archive": full_archive,
            "sample_posts": len(posts),
            "latency_ms": int((datetime.utcnow() - started).total_seconds() * 1000),
            "rate_limits": client.rate_limits(),
        }
    except Exception as exc:  # noqa: BLE001
        err = _translate(exc)
        return {"ok": False, "status": err.status_code, "error": err.detail, "rate_limits": client.rate_limits()}
    finally:
        client.close()


# --------------------------------------------------------------------------------------------
# live pulls
# --------------------------------------------------------------------------------------------


@router.post("/x/search")
def x_search(body: XSearchRequest, user: CurrentUser = Depends(require("ingest")), session: Session = Depends(get_session)):
    """Search X (recent / full-archive) and run every hit through the pipeline."""
    params = {"query": body.query, "max_pages": body.max_pages, "full_archive": body.full_archive, "start_time": _iso(body.start_time), "end_time": _iso(body.end_time), "use_cursor": body.use_cursor, "hydrate_parents": body.hydrate_parents, "download_media": body.download_media, "max_wait": get_settings().x_api_max_wait_seconds}
    if not credentials.get_bearer_token():
        raise HTTPException(409, "No X API bearer token configured. Set SENTINAI_X_BEARER_TOKEN or save one under Ingest → Connect X.")

    def run(job: Job, worker: IngestionWorker) -> IngestReport:
        if worker.client is None:
            raise RuntimeError("No X API bearer token configured")
        report = IngestReport(kind="search", query=body.query)
        worker.run_query(body.query, full_archive=body.full_archive, max_pages=body.max_pages, start_time=params["start_time"], end_time=params["end_time"], hydrate_parents=body.hydrate_parents, use_cursor=body.use_cursor, report=report)
        return report

    if body.background:
        return _submit("search", body.query, user, params, run)
    client = _client(full_archive=body.full_archive)
    try:
        worker = IngestionWorker(session, client, download_media=body.download_media)
        return _run_inline(lambda: run(None, worker))  # type: ignore[arg-type]
    finally:
        client.close()


@router.post("/x/tweet")
def x_tweet(body: XTweetRequest, user: CurrentUser = Depends(require("ingest")), session: Session = Depends(get_session)):
    """Ingest one tweet by URL or id (plus its parent chain and, optionally, its replies)."""
    tid = parse_tweet_ref(body.ref)
    if not tid:
        raise HTTPException(422, "Paste a tweet URL like https://x.com/<user>/status/<id> or a numeric tweet id")
    params = {"ref": tid, "include_conversation": body.include_conversation, "max_pages": body.max_pages, "download_media": body.download_media, "max_wait": get_settings().x_api_max_wait_seconds}
    if not credentials.get_bearer_token():
        raise HTTPException(409, "No X API bearer token configured. Set SENTINAI_X_BEARER_TOKEN or save one under Ingest → Connect X.")

    def run(job: Job, worker: IngestionWorker) -> IngestReport:
        if worker.client is None:
            raise RuntimeError("No X API bearer token configured")
        return worker.run_tweet(tid, include_conversation=body.include_conversation, max_pages=body.max_pages)

    if body.background:
        return _submit("tweet", tid, user, params, run)
    client = _client()
    try:
        worker = IngestionWorker(session, client, download_media=body.download_media)
        return _run_inline(lambda: run(None, worker))  # type: ignore[arg-type]
    finally:
        client.close()


@router.post("/x/user")
def x_user(body: XUserRequest, user: CurrentUser = Depends(require("ingest")), session: Session = Depends(get_session)):
    """Ingest a user's timeline."""
    name = parse_username(body.username)
    if not name:
        raise HTTPException(422, "Enter an X handle like @username or a profile URL")
    params = {"username": name, "max_pages": body.max_pages, "exclude_replies": body.exclude_replies, "exclude_retweets": body.exclude_retweets, "start_time": _iso(body.start_time), "hydrate_parents": body.hydrate_parents, "download_media": body.download_media, "max_wait": get_settings().x_api_max_wait_seconds}
    if not credentials.get_bearer_token():
        raise HTTPException(409, "No X API bearer token configured. Set SENTINAI_X_BEARER_TOKEN or save one under Ingest → Connect X.")

    def run(job: Job, worker: IngestionWorker) -> IngestReport:
        if worker.client is None:
            raise RuntimeError("No X API bearer token configured")
        return worker.run_user(name, max_pages=body.max_pages, exclude_replies=body.exclude_replies, exclude_retweets=body.exclude_retweets, start_time=params["start_time"], hydrate_parents=body.hydrate_parents)

    if body.background:
        return _submit("user", f"@{name}", user, params, run)
    client = _client()
    try:
        worker = IngestionWorker(session, client, download_media=body.download_media)
        return _run_inline(lambda: run(None, worker))  # type: ignore[arg-type]
    finally:
        client.close()


# --------------------------------------------------------------------------------------------
# filtered stream
# --------------------------------------------------------------------------------------------


@router.get("/x/stream/rules")
def stream_rules(user: CurrentUser = Depends(require("ingest"))):
    client = _client()
    try:
        return {"rules": client.get_stream_rules()}
    except Exception as exc:  # noqa: BLE001
        raise _translate(exc) from exc
    finally:
        client.close()


@router.post("/x/stream/start")
def stream_start(body: XStreamRequest, user: CurrentUser = Depends(require("ingest"))):
    if any(j.kind == "stream" for j in registry.active()):
        raise HTTPException(409, "A stream job is already running — stop it first")
    if not credentials.get_bearer_token():
        raise HTTPException(409, "No X API bearer token configured. Set SENTINAI_X_BEARER_TOKEN or save one under Ingest → Connect X.")
    if body.rules is not None and not body.sample:
        client = _client()
        try:
            client.set_stream_rules([r.model_dump() for r in body.rules], replace=True)
        except Exception as exc:  # noqa: BLE001
            raise _translate(exc) from exc
        finally:
            client.close()
    params = {"sample": body.sample, "max_posts": body.max_posts, "max_minutes": body.max_minutes, "download_media": body.download_media, "max_wait": None, "rules": [r.value for r in body.rules or []]}

    def run(job: Job, worker: IngestionWorker) -> IngestReport:
        if worker.client is None:
            raise RuntimeError("No X API bearer token configured")
        return worker.run_stream(max_posts=body.max_posts, max_seconds=body.max_minutes * 60 if body.max_minutes else None, stop=job.stop, sample=body.sample)

    return _submit("stream", "sample stream" if body.sample else "filtered stream", user, params, run)


@router.post("/x/stream/stop")
def stream_stop(user: CurrentUser = Depends(require("ingest"))):
    stopped = [j.id for j in registry.active() if j.kind == "stream" and registry.cancel(j.id)]
    return {"stopped": stopped}


# --------------------------------------------------------------------------------------------
# bulk file upload
# --------------------------------------------------------------------------------------------


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    hydrate_parents: bool = Form(False),
    background: bool = Form(False),
    user: CurrentUser = Depends(require("ingest")),
    session: Session = Depends(get_session),
):
    """Import a file of posts: X API v2 JSON/JSONL (twarc), X data archive (tweets.js / .zip),
    CSV/TSV with a text column, or plain text (one post per line)."""
    limit = get_settings().ingest_upload_max_mb * 1024 * 1024
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"File exceeds {get_settings().ingest_upload_max_mb} MB")
    if not data:
        raise HTTPException(422, "Empty file")
    filename = file.filename or "upload"
    params = {"filename": filename, "size": len(data), "hydrate_parents": hydrate_parents, "needs_client": hydrate_parents, "max_wait": get_settings().x_api_max_wait_seconds}

    def run(job: Job, worker: IngestionWorker) -> IngestReport:
        return worker.run_import(data, filename, hydrate_parents=hydrate_parents and worker.client is not None)

    if background or len(data) > 2 * 1024 * 1024:
        return _submit("import", filename, user, params, run)
    client = None
    if hydrate_parents and credentials.get_bearer_token():
        client = _client()
    try:
        worker = IngestionWorker(session, client)
        return _run_inline(lambda: run(None, worker))  # type: ignore[arg-type]
    finally:
        if client is not None:
            client.close()


# --------------------------------------------------------------------------------------------
# jobs
# --------------------------------------------------------------------------------------------


@router.get("/jobs")
def list_jobs(limit: int = Query(20, ge=1, le=50), user: CurrentUser = Depends(require("ingest"))):
    return {"items": registry.list(limit)}


@router.get("/jobs/{job_id}")
def get_job(job_id: str, user: CurrentUser = Depends(require("ingest"))):
    job = registry.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return job.to_dict()


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str, user: CurrentUser = Depends(require("ingest"))):
    if not registry.cancel(job_id):
        raise HTTPException(409, "job is not running")
    return registry.get(job_id).to_dict()  # type: ignore[union-attr]


@router.delete("/cursors/{query:path}")
def reset_cursor(query: str, user: CurrentUser = Depends(require("ingest")), session: Session = Depends(get_session)):
    """Forget the since_id for a query so the next run re-fetches the full window."""
    row = session.get(IngestionCursorRow, query)
    if row is None:
        raise HTTPException(404, "no cursor for that query")
    session.delete(row)
    session.commit()
    return {"deleted": query}
