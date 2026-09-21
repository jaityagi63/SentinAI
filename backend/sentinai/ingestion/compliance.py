"""Compliance utilities (Module 1).

* **Deleted posts** — X's Developer Agreement requires removing content that has been deleted
  upstream.  :func:`sync_deletions` re-hydrates stored IDs in batches of 100 and soft-deletes
  those no longer returned (``errors[].title == "Not Found Error"`` / ``Authorization Error``
  for protected accounts) and purges their raw payloads.  The batch compliance endpoint
  (``POST /2/compliance/jobs``) is the preferred path at scale; :func:`apply_compliance_job`
  ingests its JSONL output.
* **Retention** — :func:`enforce_retention` purges raw payloads older than
  ``settings.retention_days`` and anonymises author metadata for posts past the window.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from sentinai.config import get_settings
from sentinai.storage.models import AuthorRow, PostRow
from sentinai.storage.raw_store import get_raw_store
from sentinai.storage.repository import mark_deleted_upstream

log = logging.getLogger(__name__)


def sync_deletions(session: Session, client, post_ids: list[str]) -> int:
    """Return the number of posts newly marked as deleted upstream."""
    removed: list[str] = []
    for i in range(0, len(post_ids), 100):
        chunk = post_ids[i : i + 100]
        data = client.request("/tweets", {"ids": ",".join(chunk), "tweet.fields": "id"})
        present = {str(t["id"]) for t in data.get("data", []) or []}
        for err in data.get("errors", []) or []:
            if err.get("title") in ("Not Found Error", "Authorization Error", "Forbidden") and err.get("resource_id"):
                removed.append(str(err["resource_id"]))
        removed.extend(pid for pid in chunk if pid not in present and pid not in removed)
    return mark_deleted_upstream(session, removed)


def apply_compliance_job(session: Session, jsonl_path: Path) -> int:
    """Apply a downloaded compliance job result (one JSON object per line)."""
    ids: list[str] = []
    with open(jsonl_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("action") in ("delete", "withhold") or obj.get("reason") in ("deleted", "suspended", "protected"):
                ids.append(str(obj.get("id")))
    return mark_deleted_upstream(session, ids)


def enforce_retention(session: Session, days: int | None = None) -> dict[str, int]:
    days = days or get_settings().retention_days
    purged_raw = get_raw_store(session).purge_older_than(days)
    cutoff = datetime.utcnow() - timedelta(days=days)
    # Data minimisation: anonymise author profile fields for accounts with no recent posts.
    stale_authors = session.scalars(
        select(AuthorRow).where(~AuthorRow.id.in_(select(PostRow.author_id).where(PostRow.created_at >= cutoff)))
    ).all()
    anonymised = 0
    for a in stale_authors:
        if a.description or a.location or a.display_name:
            a.description = None
            a.location = None
            a.display_name = None
            anonymised += 1
    return {"purged_raw_payloads": purged_raw, "anonymised_authors": anonymised}
