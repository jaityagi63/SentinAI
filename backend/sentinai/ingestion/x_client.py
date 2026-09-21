"""X (Twitter) API v2 client with exponential backoff and rate-limit aware request queue.

Endpoints used (Academic Research / Enterprise tiers):
  * ``GET /2/tweets/search/all``      full-archive search (``/search/recent`` on lower tiers)
  * ``GET /2/tweets``                  hydrate parent / quoted tweets for reply-chain analysis
  * ``GET /2/tweets/compliance``      (batch compliance jobs — see :mod:`sentinai.ingestion.compliance`)

The client is synchronous (httpx) and intentionally simple: one worker thread per bearer token
is the recommended deployment, with the queue absorbing bursts.  Every request records the
``x-rate-limit-remaining`` / ``x-rate-limit-reset`` headers and sleeps until the reset when
the budget is exhausted; 429/5xx responses trigger jittered exponential backoff.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import httpx

from sentinai.config import get_settings
from sentinai.schemas import Author, Engagement, MediaAttachment, Post, ReferencedPost

log = logging.getLogger(__name__)

TWEET_FIELDS = "id,text,created_at,author_id,lang,conversation_id,in_reply_to_user_id,referenced_tweets,public_metrics,attachments,entities,possibly_sensitive"
USER_FIELDS = "id,username,name,created_at,public_metrics,verified,description,location,profile_image_url"
MEDIA_FIELDS = "media_key,type,url,preview_image_url,alt_text,width,height"
EXPANSIONS = "author_id,attachments.media_keys,referenced_tweets.id,referenced_tweets.id.author_id,in_reply_to_user_id"


class XRateLimitError(RuntimeError):
    def __init__(self, reset_at: float):
        super().__init__(f"rate limited until {datetime.utcfromtimestamp(reset_at).isoformat()}Z")
        self.reset_at = reset_at


@dataclass
class RateLimitState:
    remaining: int | None = None
    reset_at: float | None = None
    limit: int | None = None

    def update(self, headers: httpx.Headers) -> None:
        try:
            if "x-rate-limit-remaining" in headers:
                self.remaining = int(headers["x-rate-limit-remaining"])
            if "x-rate-limit-reset" in headers:
                self.reset_at = float(headers["x-rate-limit-reset"])
            if "x-rate-limit-limit" in headers:
                self.limit = int(headers["x-rate-limit-limit"])
        except ValueError:  # pragma: no cover
            pass

    def seconds_until_reset(self) -> float:
        if self.reset_at is None:
            return 0.0
        return max(0.0, self.reset_at - time.time())


@dataclass
class QueuedRequest:
    path: str
    params: dict[str, Any]
    callback: Callable[[dict[str, Any]], None] | None = None
    attempts: int = 0
    enqueued_at: float = field(default_factory=time.time)


class BackoffPolicy:
    """Exponential backoff with full jitter (AWS style)."""

    def __init__(self, base: float = 2.0, cap: float = 900.0, max_retries: int = 6):
        self.base, self.cap, self.max_retries = base, cap, max_retries

    def delay(self, attempt: int) -> float:
        return random.uniform(0, min(self.cap, self.base * (2**attempt)))


class XClient:
    def __init__(self, bearer_token: str | None = None, base_url: str | None = None, transport: httpx.BaseTransport | None = None, sleep: Callable[[float], None] = time.sleep):
        settings = get_settings()
        self.token = bearer_token or settings.x_bearer_token
        if not self.token:
            raise RuntimeError("SENTINAI_X_BEARER_TOKEN is not configured")
        self.base_url = (base_url or settings.x_api_base).rstrip("/")
        self.backoff = BackoffPolicy(settings.x_backoff_base_seconds, settings.x_backoff_max_seconds, settings.x_max_retries)
        self.rate: dict[str, RateLimitState] = {}
        self._sleep = sleep
        self._client = httpx.Client(base_url=self.base_url, headers={"Authorization": f"Bearer {self.token}", "User-Agent": "SentinAI/0.1 (research)"}, timeout=30, transport=transport)
        self._queue: deque[QueuedRequest] = deque()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------------------------
    # low-level request with backoff
    # ------------------------------------------------------------------------------------
    def request(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        state = self.rate.setdefault(path, RateLimitState())
        attempt = 0
        while True:
            if state.remaining is not None and state.remaining <= 0:
                wait = state.seconds_until_reset() + 1
                log.info("rate budget exhausted for %s — sleeping %.0fs", path, wait)
                self._sleep(wait)
                state.remaining = None
            try:
                resp = self._client.get(path, params=params)
            except httpx.TransportError as exc:
                attempt += 1
                if attempt > self.backoff.max_retries:
                    raise
                delay = self.backoff.delay(attempt)
                log.warning("transport error on %s (%s) — retry %d in %.1fs", path, exc, attempt, delay)
                self._sleep(delay)
                continue

            state.update(resp.headers)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                attempt += 1
                wait = max(state.seconds_until_reset() + 1, self.backoff.delay(attempt))
                if attempt > self.backoff.max_retries:
                    raise XRateLimitError(time.time() + wait)
                log.warning("429 on %s — backing off %.0fs (attempt %d)", path, wait, attempt)
                self._sleep(wait)
                state.remaining = None  # we already waited for the reset; don't sleep twice
                continue
            if resp.status_code >= 500:
                attempt += 1
                if attempt > self.backoff.max_retries:
                    resp.raise_for_status()
                delay = self.backoff.delay(attempt)
                log.warning("%d on %s — retry %d in %.1fs", resp.status_code, path, attempt, delay)
                self._sleep(delay)
                continue
            resp.raise_for_status()

    # ------------------------------------------------------------------------------------
    # queue management
    # ------------------------------------------------------------------------------------
    def enqueue(self, path: str, params: dict[str, Any], callback: Callable[[dict[str, Any]], None] | None = None) -> None:
        with self._lock:
            self._queue.append(QueuedRequest(path, params, callback))

    def drain(self, max_requests: int | None = None) -> int:
        """Process queued requests sequentially, honouring rate limits.  Returns count done."""
        done = 0
        while True:
            with self._lock:
                if not self._queue or (max_requests is not None and done >= max_requests):
                    return done
                item = self._queue.popleft()
            try:
                data = self.request(item.path, item.params)
            except XRateLimitError as exc:
                with self._lock:
                    self._queue.appendleft(item)
                self._sleep(max(1.0, exc.reset_at - time.time()))
                continue
            if item.callback:
                item.callback(data)
            done += 1

    @property
    def queue_size(self) -> int:
        return len(self._queue)

    # ------------------------------------------------------------------------------------
    # high-level helpers
    # ------------------------------------------------------------------------------------
    def search(self, query: str, *, start_time: str | None = None, end_time: str | None = None, since_id: str | None = None, max_results: int = 100, full_archive: bool = True, max_pages: int | None = None) -> Iterator[Post]:
        """Iterate posts matching ``query`` (paginates with ``next_token``)."""
        path = "/tweets/search/all" if full_archive else "/tweets/search/recent"
        params: dict[str, Any] = {
            "query": query,
            "max_results": max(10, min(max_results, 500 if full_archive else 100)),
            "tweet.fields": TWEET_FIELDS,
            "user.fields": USER_FIELDS,
            "media.fields": MEDIA_FIELDS,
            "expansions": EXPANSIONS,
        }
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time
        if since_id:
            params["since_id"] = since_id
        pages = 0
        while True:
            data = self.request(path, params)
            yield from parse_search_response(data)
            pages += 1
            token = data.get("meta", {}).get("next_token")
            if not token or (max_pages is not None and pages >= max_pages):
                return
            params["next_token"] = token

    def hydrate(self, ids: list[str]) -> list[Post]:
        out: list[Post] = []
        for i in range(0, len(ids), 100):
            chunk = ids[i : i + 100]
            data = self.request("/tweets", {"ids": ",".join(chunk), "tweet.fields": TWEET_FIELDS, "user.fields": USER_FIELDS, "media.fields": MEDIA_FIELDS, "expansions": EXPANSIONS})
            out.extend(parse_search_response(data))
        return out

    def close(self) -> None:
        self._client.close()


# --------------------------------------------------------------------------------------------
# response parsing
# --------------------------------------------------------------------------------------------


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)


def parse_user(u: dict[str, Any]) -> Author:
    pm = u.get("public_metrics", {})
    img = u.get("profile_image_url") or ""
    return Author(
        id=str(u["id"]),
        username=u.get("username", ""),
        display_name=u.get("name"),
        created_at=_parse_dt(u.get("created_at")),
        followers_count=pm.get("followers_count", 0),
        following_count=pm.get("following_count", 0),
        tweet_count=pm.get("tweet_count", 0),
        listed_count=pm.get("listed_count", 0),
        verified=bool(u.get("verified", False)),
        description=u.get("description"),
        location=u.get("location"),
        profile_image_url=img or None,
        has_default_profile_image="default_profile" in img,
    )


def parse_search_response(data: dict[str, Any]) -> list[Post]:
    includes = data.get("includes", {})
    users = {str(u["id"]): parse_user(u) for u in includes.get("users", [])}
    media = {m["media_key"]: m for m in includes.get("media", [])}
    posts: list[Post] = []
    for t in data.get("data", []) or []:
        pm = t.get("public_metrics", {})
        keys = t.get("attachments", {}).get("media_keys", [])
        atts = []
        for k in keys:
            m = media.get(k, {"media_key": k})
            atts.append(MediaAttachment(media_key=k, type=m.get("type", "photo"), url=m.get("url") or m.get("preview_image_url"), alt_text=m.get("alt_text"), width=m.get("width"), height=m.get("height")))
        posts.append(
            Post(
                id=str(t["id"]),
                text=t.get("text", ""),
                created_at=_parse_dt(t.get("created_at")) or datetime.utcnow(),
                author_id=str(t.get("author_id", "")),
                author=users.get(str(t.get("author_id", ""))),
                lang=t.get("lang"),
                conversation_id=t.get("conversation_id"),
                in_reply_to_user_id=t.get("in_reply_to_user_id"),
                referenced=[ReferencedPost(type=r["type"], id=str(r["id"])) for r in t.get("referenced_tweets", [])],
                engagement=Engagement(
                    like_count=pm.get("like_count", 0),
                    retweet_count=pm.get("retweet_count", 0),
                    reply_count=pm.get("reply_count", 0),
                    quote_count=pm.get("quote_count", 0),
                    impression_count=pm.get("impression_count", 0),
                ),
                media=atts,
                raw=t,
            )
        )
    return posts
