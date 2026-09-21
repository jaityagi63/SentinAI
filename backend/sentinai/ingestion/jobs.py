"""In-process background jobs for ingestion (search / user / tweet / import / stream).

Dashboard requests must return quickly, but a 5-page search with classification can take a
minute and a stream runs until stopped.  Jobs therefore execute on a daemon thread with their
own DB session; the registry keeps the last ``MAX_JOBS`` job states in memory for polling via
``GET /api/ingest/jobs``.  For a multi-process deployment swap this for Celery / RQ — the
:class:`IngestionWorker` API is identical.
"""

from __future__ import annotations

import logging
import threading
import traceback
import uuid
from collections import OrderedDict
from collections.abc import Callable
from datetime import datetime
from typing import Any

from sentinai.ingestion.worker import IngestReport, IngestionWorker

log = logging.getLogger(__name__)

MAX_JOBS = 50


class Job:
    def __init__(self, kind: str, label: str, owner: str, params: dict[str, Any]):
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.label = label
        self.owner = owner
        self.params = params
        self.status = "queued"  # queued | running | done | failed | cancelled
        self.created_at = datetime.utcnow()
        self.started_at: datetime | None = None
        self.finished_at: datetime | None = None
        self.progress: dict[str, int] = {}
        self.report: IngestReport | None = None
        self.error: str | None = None
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None

    def on_progress(self, stage: str, n: int) -> None:
        self.progress[stage] = n

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "owner": self.owner,
            "params": {k: v for k, v in self.params.items() if k not in ("data",)},
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "progress": self.progress,
            "report": self.report.to_dict() if self.report else None,
            "error": self.error,
        }


class JobRegistry:
    def __init__(self) -> None:
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------------------------
    def list(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            items = list(self._jobs.values())[-limit:]
        return [j.to_dict() for j in reversed(items)]

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def active(self) -> list[Job]:
        with self._lock:
            return [j for j in self._jobs.values() if j.status in ("queued", "running")]

    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if job is None or job.status not in ("queued", "running"):
            return False
        job.stop.set()
        if job.status == "queued":
            job.status = "cancelled"
            job.finished_at = datetime.utcnow()
        return True

    def cancel_all(self) -> int:
        return sum(1 for j in self.active() if self.cancel(j.id))

    # ------------------------------------------------------------------------------------
    def submit(self, kind: str, label: str, owner: str, params: dict[str, Any], run: Callable[[Job, IngestionWorker], IngestReport]) -> Job:
        """``run(job, worker)`` executes on a background thread with a fresh session + worker."""
        job = Job(kind, label, owner, params)
        with self._lock:
            self._jobs[job.id] = job
            while len(self._jobs) > MAX_JOBS:
                oldest_id, oldest = next(iter(self._jobs.items()))
                if oldest.status in ("queued", "running"):
                    break
                self._jobs.pop(oldest_id)

        def _target() -> None:
            from sentinai.storage.db import session_scope

            if job.stop.is_set():
                return
            job.status = "running"
            job.started_at = datetime.utcnow()
            client = None
            try:
                with session_scope() as session:
                    client = _make_client(params)
                    worker = IngestionWorker(session, client, progress=job.on_progress, download_media=params.get("download_media"))
                    job.report = run(job, worker)
                job.status = "cancelled" if job.stop.is_set() and kind == "stream" else "done"
            except Exception as exc:  # noqa: BLE001 - surfaced to the UI
                log.warning("ingestion job %s (%s) failed: %s\n%s", job.id, kind, exc, traceback.format_exc())
                job.error = str(exc)[:600]
                job.status = "failed"
            finally:
                job.finished_at = datetime.utcnow()
                if client is not None:
                    try:
                        client.close()
                    except Exception:  # pragma: no cover
                        pass

        job.thread = threading.Thread(target=_target, name=f"ingest-{kind}-{job.id}", daemon=True)
        job.thread.start()
        return job


def _make_client(params: dict[str, Any]):
    if not params.get("needs_client", True):
        return None
    from sentinai.ingestion.credentials import get_bearer_token
    from sentinai.ingestion.x_client import XClient

    token = get_bearer_token()
    if not token:
        return None
    max_wait = params.get("max_wait")
    return XClient(bearer_token=token, max_wait=max_wait, full_archive=params.get("full_archive"))


registry = JobRegistry()
