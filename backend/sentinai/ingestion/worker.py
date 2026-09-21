"""Ingestion worker: search → persist → classify (with parent context) → analytics refresh.

The worker is used by the CLI (``sentinai ingest --query "..."``) and can be scheduled with
cron / a Celery beat.  It stores per-query ``since_id`` cursors so re-runs only fetch new posts.
"""

from __future__ import annotations

import logging
from datetime import datetime

from sqlalchemy.orm import Session

from sentinai.classification.context import ParentInfo
from sentinai.classification.engine import ClassificationEngine, get_engine
from sentinai.schemas import Post
from sentinai.storage.models import ClassificationRow, IngestionCursorRow, PostRow
from sentinai.storage.repository import save_classification, upsert_post

log = logging.getLogger(__name__)


class IngestionWorker:
    def __init__(self, session: Session, client=None, engine: ClassificationEngine | None = None):
        self.session = session
        self.client = client
        self.engine = engine or get_engine()

    # ------------------------------------------------------------------------------------
    def parent_info(self, post: Post) -> ParentInfo | None:
        pid = post.parent_id or post.quoted_id or post.retweeted_id
        if not pid:
            return None
        row = self.session.get(PostRow, pid)
        if row is None:
            return ParentInfo(id=pid)
        cls = self.session.get(ClassificationRow, pid)
        return ParentInfo(id=pid, text=row.text, toxicity=cls.final_toxicity if cls else None, author_id=row.author_id)

    def process(self, posts: list[Post], explain: bool = True) -> int:
        """Persist + classify a batch.  Parents are processed before children so that
        reply-chain context is available."""
        by_id = {p.id: p for p in posts}
        ordered = _topological(posts)
        n = 0
        for post in ordered:
            upsert_post(self.session, post)
            self.session.flush()
            parent = self.parent_info(post)
            if parent is not None and parent.text is None and parent.id in by_id:
                parent = ParentInfo(id=parent.id, text=by_id[parent.id].text)
            c = self.engine.classify_post(post, parent=parent, explain=explain)
            save_classification(self.session, post, c)
            n += 1
        self.session.commit()
        return n

    def run_query(self, query: str, *, full_archive: bool = True, max_pages: int | None = 5, start_time: str | None = None, hydrate_parents: bool = True) -> int:
        if self.client is None:
            raise RuntimeError("An XClient is required to run live queries")
        cursor = self.session.get(IngestionCursorRow, query)
        if cursor is None:
            cursor = IngestionCursorRow(query=query)
            self.session.add(cursor)
        posts = list(self.client.search(query, since_id=cursor.since_id, full_archive=full_archive, max_pages=max_pages, start_time=start_time))
        if hydrate_parents and posts:
            have = {p.id for p in posts}
            missing = sorted({p.parent_id or p.quoted_id or p.retweeted_id for p in posts if (p.parent_id or p.quoted_id or p.retweeted_id)} - have)
            missing = [m for m in missing if self.session.get(PostRow, m) is None]
            if missing:
                posts.extend(self.client.hydrate(missing))
        n = self.process(posts)
        if posts:
            cursor.since_id = max((p.id for p in posts), key=lambda s: (len(s), s))
        cursor.last_run = datetime.utcnow()
        cursor.total_ingested = (cursor.total_ingested or 0) + n
        self.session.commit()
        log.info("query %r: ingested %d posts", query, n)
        return n


def _topological(posts: list[Post]) -> list[Post]:
    by_id = {p.id: p for p in posts}
    seen: set[str] = set()
    out: list[Post] = []

    def visit(p: Post, depth: int = 0) -> None:
        if p.id in seen or depth > 50:
            return
        seen.add(p.id)
        for ref in (p.parent_id, p.quoted_id, p.retweeted_id):
            if ref and ref in by_id:
                visit(by_id[ref], depth + 1)
        out.append(p)

    for p in sorted(posts, key=lambda x: x.created_at):
        visit(p)
    return out
