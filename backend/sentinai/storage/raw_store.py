"""Raw JSON payload store (Module 1): MongoDB or Elasticsearch, with a relational fallback.

Retention (GDPR/CCPA) is enforced by :meth:`RawStore.purge_older_than`, and deleted-upstream
posts are removed with :meth:`RawStore.delete`.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Protocol

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from sentinai.config import get_settings
from sentinai.storage.models import RawPayload

log = logging.getLogger(__name__)


class RawStore(Protocol):
    def put(self, post_id: str, payload: dict[str, Any]) -> None: ...
    def get(self, post_id: str) -> dict[str, Any] | None: ...
    def delete(self, post_id: str) -> None: ...
    def purge_older_than(self, days: int) -> int: ...


class SQLRawStore:
    def __init__(self, session: Session):
        self.session = session

    def put(self, post_id: str, payload: dict[str, Any]) -> None:
        row = self.session.get(RawPayload, post_id)
        if row is None:
            self.session.add(RawPayload(post_id=post_id, payload=payload))
        else:
            row.payload = payload

    def get(self, post_id: str) -> dict[str, Any] | None:
        row = self.session.get(RawPayload, post_id)
        return row.payload if row else None

    def delete(self, post_id: str) -> None:
        self.session.execute(delete(RawPayload).where(RawPayload.post_id == post_id))

    def purge_older_than(self, days: int) -> int:
        cutoff = datetime.utcnow() - timedelta(days=days)
        ids = self.session.scalars(select(RawPayload.post_id).where(RawPayload.stored_at < cutoff)).all()
        if ids:
            self.session.execute(delete(RawPayload).where(RawPayload.post_id.in_(ids)))
        return len(ids)


class MongoRawStore:  # pragma: no cover - requires a live MongoDB
    def __init__(self, url: str, db: str = "sentinai", collection: str = "raw_posts"):
        from pymongo import MongoClient  # type: ignore

        self.col = MongoClient(url)[db][collection]
        self.col.create_index("stored_at")

    def put(self, post_id: str, payload: dict[str, Any]) -> None:
        self.col.replace_one({"_id": post_id}, {"_id": post_id, "payload": payload, "stored_at": datetime.utcnow()}, upsert=True)

    def get(self, post_id: str) -> dict[str, Any] | None:
        doc = self.col.find_one({"_id": post_id})
        return doc["payload"] if doc else None

    def delete(self, post_id: str) -> None:
        self.col.delete_one({"_id": post_id})

    def purge_older_than(self, days: int) -> int:
        res = self.col.delete_many({"stored_at": {"$lt": datetime.utcnow() - timedelta(days=days)}})
        return int(res.deleted_count)


class ElasticRawStore:  # pragma: no cover - requires a live Elasticsearch
    def __init__(self, url: str, index: str = "sentinai-raw"):
        import httpx

        self.client = httpx.Client(base_url=url, timeout=10)
        self.index = index

    def put(self, post_id: str, payload: dict[str, Any]) -> None:
        self.client.put(f"/{self.index}/_doc/{post_id}", json={"payload": payload, "stored_at": datetime.utcnow().isoformat()}).raise_for_status()

    def get(self, post_id: str) -> dict[str, Any] | None:
        r = self.client.get(f"/{self.index}/_doc/{post_id}")
        return r.json()["_source"]["payload"] if r.status_code == 200 else None

    def delete(self, post_id: str) -> None:
        self.client.delete(f"/{self.index}/_doc/{post_id}")

    def purge_older_than(self, days: int) -> int:
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        r = self.client.post(f"/{self.index}/_delete_by_query", json={"query": {"range": {"stored_at": {"lt": cutoff}}}})
        return int(r.json().get("deleted", 0)) if r.status_code == 200 else 0


def get_raw_store(session: Session) -> RawStore:
    url = get_settings().raw_store_url
    if url:
        if url.startswith("mongodb"):
            return MongoRawStore(url)
        if url.startswith("http"):
            return ElasticRawStore(url)
        log.warning("Unknown raw store URL scheme %r; falling back to SQL raw store", url)
    return SQLRawStore(session)
