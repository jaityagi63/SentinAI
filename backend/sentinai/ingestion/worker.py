"""Ingestion worker: fetch → persist → classify (with parent context) → analytics refresh.

The worker is used by the CLI (``sentinai ingest ...``), the ``/api/ingest/x/*`` endpoints and
the background :mod:`sentinai.ingestion.jobs` runner.  It stores per-query ``since_id`` cursors
so re-runs only fetch new posts, and every entry point ends in :meth:`IngestionWorker.finish`
which refreshes bot / account scores for the touched authors and tops up the review queue.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy.orm import Session

from sentinai.classification.context import ParentInfo
from sentinai.classification.engine import ClassificationEngine, get_engine
from sentinai.config import get_settings
from sentinai.schemas import Post
from sentinai.storage.models import ClassificationRow, IngestionCursorRow, PostRow
from sentinai.storage.repository import save_classification, upsert_post

log = logging.getLogger(__name__)

ProgressFn = Callable[[str, int], None]


@dataclass
class IngestReport:
    """What one ingestion run did — returned by the API / CLI."""

    kind: str
    query: str | None = None
    fetched: int = 0
    ingested: int = 0
    hydrated_parents: int = 0
    media_downloaded: int = 0
    endpoint: str | None = None
    since_id: str | None = None
    newest_id: str | None = None
    toxic: int = 0
    by_label: dict[str, int] = field(default_factory=dict)
    post_ids: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    rate_limits: dict[str, dict] = field(default_factory=dict)
    duration_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "query": self.query,
            "fetched": self.fetched,
            "ingested": self.ingested,
            "hydrated_parents": self.hydrated_parents,
            "media_downloaded": self.media_downloaded,
            "endpoint": self.endpoint,
            "since_id": self.since_id,
            "newest_id": self.newest_id,
            "toxic": self.toxic,
            "by_label": self.by_label,
            "post_ids": self.post_ids[:200],
            "warnings": self.warnings,
            "rate_limits": self.rate_limits,
            "duration_seconds": round(self.duration_seconds, 2),
        }


class IngestionWorker:
    def __init__(self, session: Session, client=None, engine: ClassificationEngine | None = None, *, download_media: bool | None = None, progress: ProgressFn | None = None):
        self.session = session
        self.client = client
        self.engine = engine or get_engine()
        settings = get_settings()
        self.download_media = settings.x_download_media if download_media is None else download_media
        self.progress = progress or (lambda _stage, _n: None)
        self.touched_authors: set[str] = set()

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

    def process(self, posts: list[Post], explain: bool = True, report: IngestReport | None = None) -> int:
        """Persist + classify a batch.  Parents are processed before children so that
        reply-chain context is available."""
        if not posts:
            return 0
        if self.download_media:
            try:
                from sentinai.ingestion.media import download_post_media

                n_media = download_post_media(posts)
                if report is not None:
                    report.media_downloaded += n_media
            except Exception as exc:  # pragma: no cover - network dependent
                log.warning("media download failed: %s", exc)
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
            self.touched_authors.add(post.author_id)
            n += 1
            if report is not None:
                report.post_ids.append(post.id)
                label = c.toxicity.label.value
                report.by_label[label] = report.by_label.get(label, 0) + 1
                if c.final_toxicity >= 0.5:
                    report.toxic += 1
            if n % 25 == 0:
                self.progress("classify", n)
        self.session.commit()
        self.progress("classify", n)
        return n

    def process_stream(self, posts: Iterable[Post], *, batch_size: int = 100, report: IngestReport | None = None, stop: Callable[[], bool] | None = None) -> int:
        """Process a (possibly endless) iterator of posts in batches."""
        total = 0
        batch: list[Post] = []
        for p in posts:
            batch.append(p)
            if report is not None:
                report.fetched += 1
            if len(batch) >= batch_size:
                total += self.process(batch, report=report)
                batch = []
                if stop and stop():
                    break
        if batch:
            total += self.process(batch, report=report)
        if report is not None:
            report.ingested = total
        return total

    # ------------------------------------------------------------------------------------
    def hydrate_missing_parents(self, posts: list[Post], report: IngestReport | None = None) -> list[Post]:
        """Fetch parents / quoted / retweeted posts that are neither in the batch nor the DB."""
        if self.client is None or not posts:
            return []
        have = {p.id for p in posts}
        missing = sorted({p.parent_id or p.quoted_id or p.retweeted_id for p in posts if (p.parent_id or p.quoted_id or p.retweeted_id)} - have)
        missing = [m for m in missing if self.session.get(PostRow, m) is None]
        if not missing:
            return []
        try:
            extra = self.client.hydrate(missing[:300])
        except Exception as exc:  # rate limit / access — keep going without context
            log.warning("parent hydration skipped: %s", exc)
            if report is not None:
                report.warnings.append(f"parent hydration skipped: {exc}"[:300])
            return []
        if report is not None:
            report.hydrated_parents += len(extra)
        return extra

    def finish(self, report: IngestReport | None = None, *, rescore: bool = True) -> None:
        """Post-run housekeeping: bot / account scores for touched authors, review queue top-up."""
        if not rescore or not self.touched_authors:
            return
        from sentinai.analytics.accounts import score_account
        from sentinai.analytics.bots import score_author
        from sentinai.hitl import enqueue_candidates

        for aid in list(self.touched_authors):
            try:
                score_author(self.session, aid)
                score_account(self.session, aid)
            except Exception as exc:  # pragma: no cover - defensive
                log.warning("rescoring %s failed: %s", aid, exc)
        enqueue_candidates(self.session, limit=50)
        self.session.commit()
        self.progress("rescore", len(self.touched_authors))
        if report is not None and self.client is not None and hasattr(self.client, "rate_limits"):
            report.rate_limits = self.client.rate_limits()

    # ------------------------------------------------------------------------------------
    # live X API entry points
    # ------------------------------------------------------------------------------------
    def _require_client(self):
        if self.client is None:
            raise RuntimeError("An XClient is required to run live queries")
        return self.client

    def run_query(self, query: str, *, full_archive: bool | None = None, max_pages: int | None = 5, start_time: str | None = None, end_time: str | None = None, hydrate_parents: bool = True, use_cursor: bool = True, rescore: bool = True, report: IngestReport | None = None) -> int:
        """Search X for ``query``, persist + classify the results.  Returns the number ingested."""
        client = self._require_client()
        started = datetime.utcnow()
        report = report or IngestReport(kind="search", query=query)
        report.query = query
        cursor = self.session.get(IngestionCursorRow, query)
        if cursor is None:
            cursor = IngestionCursorRow(query=query)
            self.session.add(cursor)
        since_id = cursor.since_id if (use_cursor and not start_time) else None
        report.since_id = since_id
        posts: list[Post] = []
        rate_limited: Exception | None = None
        try:
            for path, page, _meta in client.search_pages(query, since_id=since_id, full_archive=full_archive, max_pages=max_pages, start_time=start_time, end_time=end_time):
                report.endpoint = path
                posts.extend(page)
                report.fetched = len(posts)
                self.progress("fetch", len(posts))
        except Exception as exc:
            from sentinai.ingestion.x_client import XRateLimitError

            if isinstance(exc, XRateLimitError) and posts:
                rate_limited = exc
                report.warnings.append(str(exc))
            else:
                raise
        if hydrate_parents and posts:
            posts.extend(self.hydrate_missing_parents(posts, report))
        n = self.process(posts, report=report)
        report.ingested = n
        if posts:
            newest = max((p.id for p in posts if p.id.isdigit()), key=lambda s: (len(s), s), default=None)
            if newest and (cursor.since_id is None or (len(newest), newest) > (len(cursor.since_id), cursor.since_id)):
                cursor.since_id = newest
            report.newest_id = newest
        cursor.last_run = datetime.utcnow()
        cursor.total_ingested = (cursor.total_ingested or 0) + n
        self.session.commit()
        self.finish(report, rescore=rescore)
        report.duration_seconds = (datetime.utcnow() - started).total_seconds()
        log.info("query %r: fetched %d, ingested %d (%s)%s", query, report.fetched, n, report.endpoint, " [rate limited]" if rate_limited else "")
        return n

    def run_tweet(self, ref: str, *, include_conversation: bool = False, max_pages: int | None = 2, rescore: bool = True, report: IngestReport | None = None) -> IngestReport:
        """Ingest a single tweet by URL / id, its parent chain and (optionally) its replies."""
        client = self._require_client()
        started = datetime.utcnow()
        report = report or IngestReport(kind="tweet", query=ref)
        post = client.get_tweet(ref)
        posts = [post]
        # walk up the reply chain (bounded) so the context module sees the originator
        seen = {post.id}
        cur = post
        for _ in range(5):
            pid = cur.parent_id or cur.quoted_id or cur.retweeted_id
            if not pid or pid in seen or self.session.get(PostRow, pid) is not None:
                break
            try:
                parent = client.get_tweet(pid)
            except Exception as exc:
                report.warnings.append(f"parent {pid}: {exc}"[:300])
                break
            posts.append(parent)
            seen.add(parent.id)
            cur = parent
        if include_conversation:
            conv_id = post.conversation_id or post.id
            try:
                for reply in client.conversation(conv_id, max_pages=max_pages):
                    if reply.id not in seen:
                        seen.add(reply.id)
                        posts.append(reply)
            except Exception as exc:
                report.warnings.append(f"conversation {conv_id}: {exc}"[:300])
        report.fetched = len(posts)
        report.endpoint = "/tweets"
        report.ingested = self.process(posts, report=report)
        report.newest_id = post.id
        self.finish(report, rescore=rescore)
        report.duration_seconds = (datetime.utcnow() - started).total_seconds()
        return report

    def run_user(self, username: str, *, max_pages: int | None = 3, exclude_replies: bool = False, exclude_retweets: bool = False, start_time: str | None = None, hydrate_parents: bool = True, rescore: bool = True, report: IngestReport | None = None) -> IngestReport:
        """Ingest a user's recent timeline (up to ``max_pages`` × 100 tweets)."""
        client = self._require_client()
        started = datetime.utcnow()
        report = report or IngestReport(kind="user", query=username)
        key = f"user:{username.lstrip('@').lower()}"
        cursor = self.session.get(IngestionCursorRow, key)
        if cursor is None:
            cursor = IngestionCursorRow(query=key)
            self.session.add(cursor)
        report.since_id = cursor.since_id if not start_time else None
        posts: list[Post] = []
        try:
            for p in client.user_timeline(username, max_pages=max_pages, exclude_replies=exclude_replies, exclude_retweets=exclude_retweets, start_time=start_time, since_id=report.since_id):
                posts.append(p)
                if len(posts) % 100 == 0:
                    self.progress("fetch", len(posts))
        except Exception as exc:
            from sentinai.ingestion.x_client import XRateLimitError

            if isinstance(exc, XRateLimitError) and posts:
                report.warnings.append(str(exc))
            else:
                raise
        report.fetched = len(posts)
        report.endpoint = "/users/:id/tweets"
        if hydrate_parents and posts:
            posts.extend(self.hydrate_missing_parents(posts, report))
        report.ingested = self.process(posts, report=report)
        if posts:
            newest = max((p.id for p in posts if p.id.isdigit()), key=lambda s: (len(s), s), default=None)
            if newest:
                cursor.since_id = newest
                report.newest_id = newest
        cursor.last_run = datetime.utcnow()
        cursor.total_ingested = (cursor.total_ingested or 0) + report.ingested
        self.session.commit()
        self.finish(report, rescore=rescore)
        report.duration_seconds = (datetime.utcnow() - started).total_seconds()
        return report

    def run_stream(self, *, max_posts: int | None = None, max_seconds: float | None = None, stop=None, sample: bool = False, batch_size: int = 25, report: IngestReport | None = None) -> IngestReport:
        """Consume the filtered stream until stopped; batches are committed as they arrive."""
        client = self._require_client()
        started = datetime.utcnow()
        report = report or IngestReport(kind="stream", query="filtered-stream")
        report.endpoint = "/tweets/sample/stream" if sample else "/tweets/search/stream"
        stop_fn = (lambda: stop.is_set()) if stop is not None else None
        self.process_stream(client.stream(sample=sample, stop=stop, max_posts=max_posts, max_seconds=max_seconds), batch_size=batch_size, report=report, stop=stop_fn)
        self.finish(report)
        report.duration_seconds = (datetime.utcnow() - started).total_seconds()
        return report

    # ------------------------------------------------------------------------------------
    # offline imports
    # ------------------------------------------------------------------------------------
    def run_import(self, data: bytes, filename: str = "upload", *, hydrate_parents: bool = False, rescore: bool = True, report: IngestReport | None = None) -> IngestReport:
        from sentinai.ingestion.importers import chunked, import_bytes

        started = datetime.utcnow()
        report = report or IngestReport(kind="import", query=filename)
        result = import_bytes(data, filename)
        report.endpoint = result.format
        report.warnings.extend(result.warnings)
        report.fetched = len(result.posts)
        if result.skipped:
            report.warnings.append(f"{result.skipped} record(s) skipped")
        posts = result.posts
        if hydrate_parents and self.client is not None and posts:
            posts = posts + self.hydrate_missing_parents(posts, report)
        total = 0
        for batch in chunked(posts, 200):
            total += self.process(batch, report=report)
            self.progress("classify", total)
        report.ingested = total
        self.finish(report, rescore=rescore)
        report.duration_seconds = (datetime.utcnow() - started).total_seconds()
        return report


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
