"""X (Twitter) API v2 client with exponential backoff and rate-limit aware request queue.

Endpoints used:
  * ``GET /2/tweets/search/recent``    7-day search (available on every self-serve / pay-per-use plan)
  * ``GET /2/tweets/search/all``       full-archive search (needs full-archive access; the client
                                        falls back to recent search when the plan rejects it)
  * ``GET /2/tweets`` / ``/2/tweets/{id}``   hydrate tweets by id — single tweets, URLs, parents
  * ``GET /2/users/by/username/{u}``   resolve a handle
  * ``GET /2/users/{id}/tweets``       a user's timeline
  * ``GET /2/tweets/search/stream``    filtered stream (+ ``/rules`` management)
  * ``GET /2/tweets/compliance``       (batch compliance jobs — see :mod:`sentinai.ingestion.compliance`)

The client is synchronous (httpx) and intentionally simple: one worker thread per bearer token
is the recommended deployment, with the queue absorbing bursts.  Every request records the
``x-rate-limit-remaining`` / ``x-rate-limit-reset`` headers and sleeps until the reset when
the budget is exhausted; 429/5xx responses trigger jittered exponential backoff.  Interactive
callers (the dashboard) set ``max_wait`` so a rate limit surfaces as :class:`XRateLimitError`
instead of blocking a request for fifteen minutes.
"""

from __future__ import annotations

import json
import logging
import random
import re
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

TWEET_URL_RE = re.compile(r"(?:https?://)?(?:www\.|mobile\.)?(?:twitter|x)\.com/(?:#!/)?(?:[A-Za-z0-9_]{1,20}|i/web)/status(?:es)?/(\d{1,25})", re.I)
TWEET_ID_RE = re.compile(r"^\d{1,25}$")
USERNAME_RE = re.compile(r"^[A-Za-z0-9_]{1,15}$")


def parse_tweet_ref(value: str) -> str | None:
    """Accept a tweet URL (x.com / twitter.com, with or without scheme, mobile, ``i/web``) or a
    bare numeric id and return the id, or ``None`` when the input is neither."""
    value = (value or "").strip().strip("<>").split("?")[0].rstrip("/")
    if not value:
        return None
    if TWEET_ID_RE.match(value):
        return value
    m = TWEET_URL_RE.search(value)
    return m.group(1) if m else None


def parse_username(value: str) -> str | None:
    value = (value or "").strip()
    m = re.search(r"(?:twitter|x)\.com/(?:#!/)?@?([A-Za-z0-9_]{1,15})", value, re.I)
    if m:
        return m.group(1)
    value = value.lstrip("@")
    return value if USERNAME_RE.match(value) else None


class XRateLimitError(RuntimeError):
    def __init__(self, reset_at: float, path: str | None = None):
        wait = max(0.0, reset_at - time.time())
        super().__init__(f"X API rate limit hit{f' on {path}' if path else ''} — resets in {wait / 60:.1f} min ({datetime.utcfromtimestamp(reset_at).isoformat()}Z)")
        self.reset_at = reset_at
        self.path = path


class XAccessError(RuntimeError):
    """401/402/403 — token or plan problem.  The message is meant for humans."""

    def __init__(self, status_code: int, path: str, detail: str):
        self.status_code, self.path, self.detail = status_code, path, detail
        if status_code == 401:
            hint = "X rejected the bearer token — check SENTINAI_X_BEARER_TOKEN (Developer Portal → Project → Keys and tokens)."
        elif status_code == 402:
            hint = "X returned 402 Payment Required — the project has no credits / active plan for this endpoint."
        else:
            hint = f"X returned 403 for {path} — this endpoint is not included in the app's access level (full-archive search and the filtered stream need an upgraded plan)."
        super().__init__(f"{hint} Details: {detail}"[:600])


class XNotFoundError(RuntimeError):
    pass


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

    def as_dict(self) -> dict[str, Any]:
        return {"remaining": self.remaining, "limit": self.limit, "reset_at": datetime.utcfromtimestamp(self.reset_at).isoformat() + "Z" if self.reset_at else None}


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


def _error_detail(resp: httpx.Response) -> str:
    try:
        j = resp.json()
    except ValueError:
        return (resp.text or resp.reason_phrase or "")[:300]
    if isinstance(j, dict):
        parts = [str(j.get(k)) for k in ("title", "detail", "reason") if j.get(k)]
        for e in j.get("errors", []) or []:
            if isinstance(e, dict):
                parts.append(str(e.get("message") or e.get("detail") or e.get("title") or e))
        if parts:
            return " | ".join(parts)[:300]
    return str(j)[:300]


class XClient:
    def __init__(
        self,
        bearer_token: str | None = None,
        base_url: str | None = None,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        max_wait: float | None = None,
        full_archive: bool | None = None,
    ):
        settings = get_settings()
        self.token = bearer_token or settings.x_bearer_token
        if not self.token:
            raise RuntimeError("SENTINAI_X_BEARER_TOKEN is not configured")
        self.base_url = (base_url or settings.x_api_base).rstrip("/")
        self.backoff = BackoffPolicy(settings.x_backoff_base_seconds, settings.x_backoff_max_seconds, settings.x_max_retries)
        self.rate: dict[str, RateLimitState] = {}
        self.max_wait = max_wait  # None = wait as long as the API tells us to
        self.full_archive = settings.x_full_archive if full_archive is None else full_archive
        self.full_archive_available: bool | None = None  # learned from the first 402/403
        self.last_error: str | None = None
        self._sleep = sleep
        self._client = httpx.Client(base_url=self.base_url, headers={"Authorization": f"Bearer {self.token}", "User-Agent": "SentinAI/0.1 (research)"}, timeout=30, transport=transport)
        self._queue: deque[QueuedRequest] = deque()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------------------------
    # low-level request with backoff
    # ------------------------------------------------------------------------------------
    def _wait_or_raise(self, path: str, wait: float, reason: str) -> None:
        if self.max_wait is not None and wait > self.max_wait:
            raise XRateLimitError(time.time() + wait, path)
        log.info("%s for %s — sleeping %.0fs", reason, path, wait)
        self._sleep(wait)

    def request(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        state = self.rate.setdefault(path, RateLimitState())
        attempt = 0
        while True:
            if state.remaining is not None and state.remaining <= 0:
                self._wait_or_raise(path, state.seconds_until_reset() + 1, "rate budget exhausted")
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
                self.last_error = None
                return resp.json()
            if resp.status_code == 429:
                attempt += 1
                wait = max(state.seconds_until_reset() + 1, self.backoff.delay(attempt))
                if attempt > self.backoff.max_retries:
                    raise XRateLimitError(time.time() + wait, path)
                self._wait_or_raise(path, wait, f"429 (attempt {attempt})")
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
            if resp.status_code in (401, 402, 403):
                err = XAccessError(resp.status_code, path, _error_detail(resp))
                self.last_error = str(err)
                raise err
            if resp.status_code == 404:
                raise XNotFoundError(f"{path}: {_error_detail(resp)}")
            detail = _error_detail(resp)
            self.last_error = f"HTTP {resp.status_code} on {path}: {detail}"
            raise httpx.HTTPStatusError(f"HTTP {resp.status_code} on {path}: {detail}", request=resp.request, response=resp)

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

    def rate_limits(self) -> dict[str, dict[str, Any]]:
        return {path: st.as_dict() for path, st in self.rate.items()}

    # ------------------------------------------------------------------------------------
    # high-level helpers
    # ------------------------------------------------------------------------------------
    @staticmethod
    def _expansion_params() -> dict[str, Any]:
        return {"tweet.fields": TWEET_FIELDS, "user.fields": USER_FIELDS, "media.fields": MEDIA_FIELDS, "expansions": EXPANSIONS}

    def search_pages(self, query: str, *, start_time: str | None = None, end_time: str | None = None, since_id: str | None = None, max_results: int = 100, full_archive: bool | None = None, max_pages: int | None = None) -> Iterator[tuple[str, list[Post], dict[str, Any]]]:
        """Iterate ``(endpoint, posts, meta)`` per page.  Falls back from full-archive to recent
        search when the plan does not allow ``/search/all``."""
        use_all = self.full_archive if full_archive is None else full_archive
        if use_all and self.full_archive_available is False:
            use_all = False
        path = "/tweets/search/all" if use_all else "/tweets/search/recent"
        params: dict[str, Any] = {"query": query, "max_results": max(10, min(max_results, 500 if use_all else 100)), **self._expansion_params()}
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time
        if since_id:
            params["since_id"] = since_id
        pages = 0
        while True:
            try:
                data = self.request(path, params)
            except XAccessError as exc:
                if use_all and exc.status_code in (402, 403):
                    log.warning("full-archive search not available (%s) — falling back to /search/recent", exc.detail)
                    self.full_archive_available = False
                    use_all = False
                    path = "/tweets/search/recent"
                    params["max_results"] = min(params["max_results"], 100)
                    continue
                raise
            if use_all:
                self.full_archive_available = True
            yield path, parse_search_response(data), data.get("meta", {}) or {}
            pages += 1
            token = data.get("meta", {}).get("next_token")
            if not token or (max_pages is not None and pages >= max_pages):
                return
            params["next_token"] = token

    def search(self, query: str, *, start_time: str | None = None, end_time: str | None = None, since_id: str | None = None, max_results: int = 100, full_archive: bool | None = None, max_pages: int | None = None) -> Iterator[Post]:
        """Iterate posts matching ``query`` (paginates with ``next_token``)."""
        for _path, posts, _meta in self.search_pages(query, start_time=start_time, end_time=end_time, since_id=since_id, max_results=max_results, full_archive=full_archive, max_pages=max_pages):
            yield from posts

    def hydrate(self, ids: list[str]) -> list[Post]:
        out: list[Post] = []
        ids = [i for i in dict.fromkeys(ids) if i]
        for i in range(0, len(ids), 100):
            chunk = ids[i : i + 100]
            data = self.request("/tweets", {"ids": ",".join(chunk), **self._expansion_params()})
            out.extend(parse_search_response(data))
        return out

    def get_tweet(self, ref: str) -> Post:
        """Fetch one tweet by id or URL (``https://x.com/<user>/status/<id>``)."""
        tid = parse_tweet_ref(ref)
        if not tid:
            raise ValueError(f"{ref!r} is not a tweet URL or id")
        posts = self.hydrate([tid])
        for p in posts:
            if p.id == tid:
                return p
        raise XNotFoundError(f"tweet {tid} was not returned by X (deleted, protected or the id is wrong)")

    def conversation(self, conversation_id: str, *, max_pages: int | None = 3, full_archive: bool | None = None) -> list[Post]:
        """Replies belonging to a conversation (thread).  Recent search covers the last 7 days;
        older threads need full-archive access."""
        return list(self.search(f"conversation_id:{conversation_id}", max_pages=max_pages, full_archive=full_archive))

    def get_user(self, username: str) -> Author:
        name = parse_username(username)
        if not name:
            raise ValueError(f"{username!r} is not a valid X username")
        data = self.request(f"/users/by/username/{name}", {"user.fields": USER_FIELDS})
        if not data.get("data"):
            raise XNotFoundError(f"user @{name}: {_error_detail_dict(data)}")
        return parse_user(data["data"])

    def user_timeline(self, username: str, *, max_pages: int | None = 3, max_results: int = 100, exclude_replies: bool = False, exclude_retweets: bool = False, start_time: str | None = None, since_id: str | None = None) -> Iterator[Post]:
        """Iterate a user's recent tweets (``GET /2/users/{id}/tweets``, up to 3200 most recent)."""
        author = self.get_user(username)
        params: dict[str, Any] = {"max_results": max(5, min(max_results, 100)), **self._expansion_params()}
        excludes = [x for x, on in (("replies", exclude_replies), ("retweets", exclude_retweets)) if on]
        if excludes:
            params["exclude"] = ",".join(excludes)
        if start_time:
            params["start_time"] = start_time
        if since_id:
            params["since_id"] = since_id
        path = f"/users/{author.id}/tweets"
        pages = 0
        while True:
            data = self.request(path, params)
            for post in parse_search_response(data):
                if post.author is None and post.author_id == author.id:
                    post.author = author
                yield post
            pages += 1
            token = data.get("meta", {}).get("next_token")
            if not token or (max_pages is not None and pages >= max_pages):
                return
            params["pagination_token"] = token

    # ------------------------------------------------------------------------------------
    # filtered stream
    # ------------------------------------------------------------------------------------
    def get_stream_rules(self) -> list[dict[str, Any]]:
        data = self.request("/tweets/search/stream/rules", {})
        return list(data.get("data", []) or [])

    def set_stream_rules(self, rules: list[dict[str, str]], *, replace: bool = True) -> list[dict[str, Any]]:
        """Replace (default) or add rules: ``[{"value": "muslims lang:en -is:retweet", "tag": "..."}]``."""
        path = "/tweets/search/stream/rules"
        if replace:
            existing = self.get_stream_rules()
            if existing:
                self._post(path, {"delete": {"ids": [r["id"] for r in existing]}})
        if rules:
            data = self._post(path, {"add": [{"value": r["value"], **({"tag": r["tag"]} if r.get("tag") else {})} for r in rules]})
            errors = data.get("errors") or []
            if errors and not data.get("data"):
                raise ValueError("X rejected the stream rules: " + "; ".join(str(e.get("details") or e.get("title") or e) for e in errors)[:400])
        return self.get_stream_rules()

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        resp = self._client.post(path, json=body)
        self.rate.setdefault(path, RateLimitState()).update(resp.headers)
        if resp.status_code in (401, 402, 403):
            raise XAccessError(resp.status_code, path, _error_detail(resp))
        if resp.status_code == 429:
            raise XRateLimitError(time.time() + self.rate[path].seconds_until_reset() + 1, path)
        if resp.status_code >= 400:
            raise httpx.HTTPStatusError(f"HTTP {resp.status_code} on {path}: {_error_detail(resp)}", request=resp.request, response=resp)
        return resp.json() if resp.content else {}

    def stream(self, *, sample: bool = False, stop: threading.Event | None = None, max_posts: int | None = None, max_seconds: float | None = None) -> Iterator[Post]:
        """Yield posts from the filtered stream (or the 1% sample stream) until ``stop`` is set,
        ``max_posts`` are received or ``max_seconds`` elapse.  Disconnects are retried with
        backoff, as X's connection guidelines require."""
        path = "/tweets/sample/stream" if sample else "/tweets/search/stream"
        params = self._expansion_params()
        started = time.time()
        received = 0
        attempt = 0
        while not (stop and stop.is_set()):
            if max_seconds is not None and time.time() - started >= max_seconds:
                return
            try:
                with self._client.stream("GET", path, params=params, timeout=httpx.Timeout(90.0, connect=30.0)) as resp:
                    self.rate.setdefault(path, RateLimitState()).update(resp.headers)
                    if resp.status_code in (401, 402, 403):
                        resp.read()
                        raise XAccessError(resp.status_code, path, _error_detail(resp))
                    if resp.status_code == 429:
                        resp.read()
                        wait = self.rate[path].seconds_until_reset() + 1
                        raise XRateLimitError(time.time() + max(wait, 60), path)
                    if resp.status_code >= 400:
                        resp.read()
                        raise httpx.HTTPStatusError(f"HTTP {resp.status_code} on {path}: {_error_detail(resp)}", request=resp.request, response=resp)
                    attempt = 0
                    for line in resp.iter_lines():
                        if stop and stop.is_set():
                            return
                        if max_seconds is not None and time.time() - started >= max_seconds:
                            return
                        line = line.strip()
                        if not line:  # keep-alive newline
                            continue
                        try:
                            obj = json.loads(line)
                        except ValueError:
                            continue
                        if "data" not in obj:
                            if obj.get("errors"):
                                log.warning("stream error payload: %s", obj["errors"])
                            continue
                        main = obj["data"] if isinstance(obj["data"], dict) else (obj["data"][0] if obj["data"] else None)
                        if not main:
                            continue
                        rules = obj.get("matching_rules", []) or []
                        for post in parse_search_response({"data": [main], "includes": obj.get("includes", {})}):
                            post.source = "stream"
                            if post.id == str(main.get("id")) and rules:
                                post.raw = {**(post.raw or {}), "matching_rules": rules}
                            yield post
                            received += 1
                            if max_posts is not None and received >= max_posts:
                                return
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                attempt += 1
                if attempt > self.backoff.max_retries:
                    raise
                delay = self.backoff.delay(attempt)
                log.warning("stream disconnected (%s) — reconnect %d in %.1fs", exc, attempt, delay)
                self._sleep(delay)
            except XRateLimitError as exc:
                self._wait_or_raise(path, max(1.0, exc.reset_at - time.time()), "stream rate limited")

    def close(self) -> None:
        self._client.close()


def _error_detail_dict(data: dict[str, Any]) -> str:
    errs = data.get("errors") or []
    return "; ".join(str(e.get("detail") or e.get("title") or e) for e in errs)[:300] or "not found"


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


def parse_tweet(t: dict[str, Any], users: dict[str, Author] | None = None, media: dict[str, dict[str, Any]] | None = None) -> Post:
    """Convert one X API v2 tweet object (plus resolved expansions) into a :class:`Post`."""
    users = users or {}
    media = media or {}
    pm = t.get("public_metrics", {}) or {}
    keys = (t.get("attachments") or {}).get("media_keys", []) or []
    atts = []
    for k in keys:
        m = media.get(k, {"media_key": k})
        atts.append(MediaAttachment(media_key=k, type=m.get("type", "photo"), url=m.get("url") or m.get("preview_image_url"), alt_text=m.get("alt_text"), width=m.get("width"), height=m.get("height")))
    author_id = str(t.get("author_id", "") or "")
    return Post(
        id=str(t["id"]),
        text=t.get("text", "") or "",
        created_at=_parse_dt(t.get("created_at")) or datetime.utcnow(),
        author_id=author_id or "unknown",
        author=users.get(author_id),
        lang=t.get("lang"),
        conversation_id=str(t["conversation_id"]) if t.get("conversation_id") else None,
        in_reply_to_user_id=str(t["in_reply_to_user_id"]) if t.get("in_reply_to_user_id") else None,
        referenced=[ReferencedPost(type=r["type"], id=str(r["id"])) for r in t.get("referenced_tweets", []) or [] if r.get("type") and r.get("id")],
        engagement=Engagement(
            like_count=pm.get("like_count", 0) or 0,
            retweet_count=pm.get("retweet_count", 0) or 0,
            reply_count=pm.get("reply_count", 0) or 0,
            quote_count=pm.get("quote_count", 0) or 0,
            impression_count=pm.get("impression_count", 0) or 0,
        ),
        media=atts,
        raw=t,
    )


def parse_search_response(data: dict[str, Any], include_referenced: bool = True) -> list[Post]:
    """Parse a search / lookup / timeline page.  Referenced tweets returned through the
    ``referenced_tweets.id`` expansion (``includes.tweets``) are appended after the main
    results so reply-chain context is available without extra hydration requests."""
    includes = data.get("includes", {}) or {}
    users = {str(u["id"]): parse_user(u) for u in includes.get("users", []) or []}
    media = {m["media_key"]: m for m in includes.get("media", []) or []}
    posts: list[Post] = []
    seen: set[str] = set()
    main = data.get("data", []) or []
    if isinstance(main, dict):
        main = [main]
    for t in main:
        p = parse_tweet(t, users, media)
        if p.id not in seen:
            seen.add(p.id)
            posts.append(p)
    if include_referenced:
        for t in includes.get("tweets", []) or []:
            if str(t.get("id")) in seen:
                continue
            p = parse_tweet(t, users, media)
            seen.add(p.id)
            posts.append(p)
    return posts
