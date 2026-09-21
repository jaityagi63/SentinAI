"""Offline importers — bring X posts into the pipeline without (or in addition to) the live API.

Supported inputs (detected from the file name and a peek at the content):

* **X API v2 JSON** — a search / lookup / timeline response (``{"data": [...], "includes": {...}}``),
  a bare list of tweet objects, or a file with one such object per line (JSONL, e.g. the output of
  ``twarc2 search``).  Streams saved line-by-line (``{"data": {...}, "includes": ...}``) work too.
* **X data archive** — the ``tweets.js`` (or ``tweet.js``) file from *Settings → Download an
  archive of your data*, either alone or inside the ``.zip``.  Legacy v1.1 tweet objects are
  converted (``id_str``, ``full_text``, ``created_at`` in ``%a %b %d %H:%M:%S %z %Y`` format,
  ``in_reply_to_status_id_str``, ``retweeted_status`` / ``quoted_status_id_str``, entities.media).
* **CSV** — any spreadsheet export with at least a text column.  Column names are matched
  case-insensitively against common aliases (``id``/``tweet_id``/``status_id``, ``text``/``tweet``/
  ``content``/``full_text``, ``created_at``/``date``/``timestamp``, ``author_id``/``user_id``,
  ``username``/``screen_name``/``author``, ``lang``, ``in_reply_to_status_id``/``parent_id``,
  ``quoted_status_id``, ``retweeted_status_id``, ``like_count``/``favorite_count``, ``retweet_count``,
  ``reply_count``, ``quote_count``, ``url``/``permalink`` — a tweet URL is used to recover the id).
  Rows without an id get a deterministic ``csv-<sha1>`` id so re-imports stay idempotent.
* **Plain text** — one post per line, useful for quick experiments (ids are ``txt-<sha1>``).

Every importer returns :class:`sentinai.schemas.Post` objects with ``source`` set to
``upload`` (or ``archive``) so provenance stays visible in the dashboard.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import re
import zipfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from sentinai.ingestion.x_client import parse_search_response, parse_tweet, parse_tweet_ref, parse_user
from sentinai.schemas import Author, Engagement, MediaAttachment, Post, ReferencedPost

log = logging.getLogger(__name__)

MAX_TEXT_LEN = 10_000


@dataclass
class ImportResult:
    posts: list[Post] = field(default_factory=list)
    format: str = "unknown"
    skipped: int = 0
    warnings: list[str] = field(default_factory=list)

    def add_warning(self, msg: str) -> None:
        if len(self.warnings) < 20:
            self.warnings.append(msg)


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------

_LEGACY_DT = "%a %b %d %H:%M:%S %z %Y"


def parse_datetime(value: Any) -> datetime | None:
    """Best-effort timestamp parsing (ISO-8601, legacy Twitter format, epoch seconds/millis)."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).replace(tzinfo=None) if value.tzinfo else value
    if isinstance(value, (int, float)):
        ts = float(value)
        if ts > 1e12:
            ts /= 1000.0
        try:
            return datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None)
        except (OverflowError, OSError, ValueError):
            return None
    s = str(value).strip()
    if not s:
        return None
    if re.fullmatch(r"\d{9,13}", s):
        return parse_datetime(int(s))
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc).replace(tzinfo=None) if dt.tzinfo else dt
    except ValueError:
        pass
    for fmt in (_LEGACY_DT, "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d", "%d/%m/%Y %H:%M", "%m/%d/%Y %H:%M", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d %H:%M:%S", "%b %d, %Y", "%d %b %Y %H:%M:%S"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.astimezone(timezone.utc).replace(tzinfo=None) if dt.tzinfo else dt
        except ValueError:
            continue
    try:  # pragma: no cover - dateutil is an indirect dependency (pandas)
        from dateutil import parser as du

        dt = du.parse(s)
        return dt.astimezone(timezone.utc).replace(tzinfo=None) if dt.tzinfo else dt
    except Exception:
        return None


def _int(value: Any) -> int:
    try:
        return int(float(str(value).replace(",", ""))) if value not in (None, "") else 0
    except ValueError:
        return 0


def _stable_id(prefix: str, *parts: Any) -> str:
    h = hashlib.sha1("\u241f".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:20]
    return f"{prefix}-{h}"


def _author_from(author_id: str | None, username: str | None, display_name: str | None = None) -> tuple[str, Author | None]:
    if author_id:
        aid = str(author_id)
    elif username:
        aid = _stable_id("user", username.lower())
    else:
        return "unknown", None
    if username or display_name:
        return aid, Author(id=aid, username=(username or f"user_{aid}").lstrip("@"), display_name=display_name)
    return aid, None


# --------------------------------------------------------------------------------------------
# format detection / dispatch
# --------------------------------------------------------------------------------------------


def import_bytes(data: bytes, filename: str = "upload", *, source: str = "upload") -> ImportResult:
    """Detect the format of ``data`` and convert it into posts."""
    name = (filename or "").lower()
    if data[:2] == b"PK" or name.endswith(".zip"):
        return import_archive_zip(data)
    text = data.decode("utf-8-sig", errors="replace")
    stripped = text.lstrip()
    if name.endswith((".js",)) or stripped.startswith("window.YTD."):
        return import_archive_js(text)
    if name.endswith((".jsonl", ".ndjson")):
        return import_jsonl(text, source=source)
    if name.endswith(".json") or stripped[:1] in ("{", "["):
        try:
            obj = json.loads(text)
        except ValueError:
            return import_jsonl(text, source=source)
        return import_json_object(obj, source=source)
    if name.endswith((".csv", ".tsv")) or _looks_like_csv(text):
        return import_csv(text, source=source, delimiter="\t" if name.endswith(".tsv") else None)
    return import_plain_text(text, source=source)


def _looks_like_csv(text: str) -> bool:
    head = text[:4096].splitlines()
    if len(head) < 2:
        return False
    try:
        dialect = csv.Sniffer().sniff("\n".join(head[:10]), delimiters=",;\t|")
    except csv.Error:
        return False
    first = head[0].lower()
    return dialect.delimiter in first and any(k in first for k in ("text", "tweet", "content", "message", "body"))


# --------------------------------------------------------------------------------------------
# X API v2 JSON / JSONL
# --------------------------------------------------------------------------------------------


def _is_v2_tweet(obj: dict[str, Any]) -> bool:
    return "id" in obj and "text" in obj and "id_str" not in obj and "full_text" not in obj


def _is_legacy_tweet(obj: dict[str, Any]) -> bool:
    return ("id_str" in obj or "full_text" in obj) and ("full_text" in obj or "text" in obj)


def import_json_object(obj: Any, *, source: str = "upload") -> ImportResult:
    res = ImportResult(format="x-api-v2-json")
    if isinstance(obj, dict):
        if "posts" in obj and isinstance(obj["posts"], list):  # our own /ingest/posts payload
            return import_json_object(obj["posts"], source=source)
        if "data" in obj:
            posts = parse_search_response(obj)
            for p in posts:
                p.source = source
            res.posts.extend(posts)
            return res
        if "tweet" in obj and isinstance(obj["tweet"], dict):  # archive entry
            res.format = "x-archive"
            p = legacy_to_post(obj["tweet"])
            if p:
                res.posts.append(p)
            return res
        if _is_v2_tweet(obj):
            res.posts.append(_v2_single(obj, source))
            return res
        if _is_legacy_tweet(obj):
            res.format = "x-api-v1.1-json"
            p = legacy_to_post(obj, source=source)
            if p:
                res.posts.append(p)
            return res
        p = generic_record_to_post(obj, source=source)
        if p:
            res.format = "generic-json"
            res.posts.append(p)
        else:
            res.skipped += 1
            res.add_warning("JSON object did not look like a tweet")
        return res
    if isinstance(obj, list):
        seen: set[str] = set()
        for i, item in enumerate(obj):
            if not isinstance(item, dict):
                res.skipped += 1
                continue
            sub = import_json_object(item, source=source)
            if sub.format != "x-api-v2-json":
                res.format = sub.format
            res.skipped += sub.skipped
            for p in sub.posts:
                if p.id in seen:
                    continue
                seen.add(p.id)
                res.posts.append(p)
            if i == 0 and not sub.posts:
                res.add_warning("first element of the list was not recognised as a tweet")
        return res
    res.skipped += 1
    res.add_warning("unsupported JSON top-level type")
    return res


def _v2_single(t: dict[str, Any], source: str) -> Post:
    users = {}
    media = {}
    inc = t.get("includes") or {}
    for u in inc.get("users", []) or []:
        users[str(u["id"])] = parse_user(u)
    for m in inc.get("media", []) or []:
        media[m["media_key"]] = m
    # twarc-style flattened objects carry the author inline
    if isinstance(t.get("author"), dict) and "id" in t["author"]:
        users[str(t["author"]["id"])] = parse_user(t["author"])
        t = {**t, "author_id": t["author"]["id"]}
    p = parse_tweet(t, users, media)
    p.source = source
    return p


def import_jsonl(text: str, *, source: str = "upload") -> ImportResult:
    res = ImportResult(format="jsonl")
    seen: set[str] = set()
    for ln, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            res.skipped += 1
            res.add_warning(f"line {ln}: invalid JSON")
            continue
        sub = import_json_object(obj, source=source)
        res.skipped += sub.skipped
        for p in sub.posts:
            if p.id not in seen:
                seen.add(p.id)
                res.posts.append(p)
    return res


# --------------------------------------------------------------------------------------------
# generic dict / CSV rows
# --------------------------------------------------------------------------------------------

COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "id": ("id", "tweet_id", "tweetid", "status_id", "statusid", "post_id", "id_str", "tweet id"),
    "text": ("text", "full_text", "tweet", "tweet_text", "content", "message", "body", "post", "rawcontent", "renderedcontent"),
    "created_at": ("created_at", "createdat", "date", "datetime", "timestamp", "time", "created", "tweet_created_at", "posted_at"),
    "author_id": ("author_id", "authorid", "user_id", "userid", "user_id_str"),
    "username": ("username", "user_name", "screen_name", "screenname", "author", "user", "handle", "author_username", "user.username", "user_screen_name"),
    "display_name": ("name", "display_name", "displayname", "user.name", "author_name"),
    "lang": ("lang", "language"),
    "parent_id": ("in_reply_to_status_id", "in_reply_to_status_id_str", "inreplytotweetid", "parent_id", "reply_to_id", "in_reply_to_tweet_id"),
    "quoted_id": ("quoted_status_id", "quoted_status_id_str", "quotedtweetid", "quoted_id", "quote_id"),
    "retweeted_id": ("retweeted_status_id", "retweeted_status_id_str", "retweetedtweetid", "retweeted_id"),
    "conversation_id": ("conversation_id", "conversationid"),
    "like_count": ("like_count", "likes", "favorite_count", "favourites_count", "favoritecount", "likecount"),
    "retweet_count": ("retweet_count", "retweets", "retweetcount"),
    "reply_count": ("reply_count", "replies", "replycount"),
    "quote_count": ("quote_count", "quotes", "quotecount"),
    "url": ("url", "permalink", "tweet_url", "link", "status_url"),
    "media_url": ("media_url", "media", "image_url", "photo_url"),
    "alt_text": ("alt_text", "media_alt_text", "image_alt"),
}


def _norm_key(k: str) -> str:
    return re.sub(r"[^a-z0-9_.]", "", str(k).strip().lower().replace(" ", "_"))


def _pick(record: dict[str, Any], key: str) -> Any:
    aliases = COLUMN_ALIASES[key]
    for a in aliases:
        if a in record and record[a] not in (None, ""):
            return record[a]
    return None


def generic_record_to_post(rec: dict[str, Any], *, source: str = "upload", prefix: str = "row") -> Post | None:
    record = {_norm_key(k): v for k, v in rec.items()}
    # flattened nested objects (twarc / pandas json_normalize)
    for k, v in list(rec.items()):
        if isinstance(v, dict):
            for kk, vv in v.items():
                record.setdefault(f"{_norm_key(k)}.{_norm_key(kk)}", vv)
    text = _pick(record, "text")
    if text is None or not str(text).strip():
        return None
    text = str(text)[:MAX_TEXT_LEN]
    tid = _pick(record, "id")
    url = _pick(record, "url")
    if tid is None and url:
        tid = parse_tweet_ref(str(url))
    tid = str(tid).strip() if tid is not None else None
    if tid and re.fullmatch(r"\d+\.0", tid):  # pandas float ids
        tid = tid[:-2]
    created = parse_datetime(_pick(record, "created_at"))
    if not tid:
        tid = _stable_id(prefix, text, created.isoformat() if created else "", _pick(record, "username") or "")
    author_id, author = _author_from(_pick(record, "author_id"), _pick(record, "username"), _pick(record, "display_name"))
    if author_id != "unknown" and _pick(record, "author_id") and str(_pick(record, "author_id")).endswith(".0"):
        author_id = author_id[:-2]
        if author:
            author.id = author_id
    refs = []
    for key, typ in (("parent_id", "replied_to"), ("quoted_id", "quoted"), ("retweeted_id", "retweeted")):
        v = _pick(record, key)
        if v not in (None, "", "0", 0):
            refs.append(ReferencedPost(type=typ, id=str(v).removesuffix(".0")))
    media = []
    murl = _pick(record, "media_url")
    alt = _pick(record, "alt_text")
    if murl or alt:
        media.append(MediaAttachment(media_key=f"{tid}-m1", type="photo", url=str(murl) if murl else None, alt_text=str(alt) if alt else None))
    conv = _pick(record, "conversation_id")
    return Post(
        id=tid,
        text=text,
        created_at=created or datetime.utcnow(),
        author_id=author_id,
        author=author,
        lang=(str(_pick(record, "lang"))[:8].lower() if _pick(record, "lang") else None),
        conversation_id=str(conv).removesuffix(".0") if conv else None,
        referenced=refs,
        engagement=Engagement(like_count=_int(_pick(record, "like_count")), retweet_count=_int(_pick(record, "retweet_count")), reply_count=_int(_pick(record, "reply_count")), quote_count=_int(_pick(record, "quote_count"))),
        media=media,
        source=source,
        raw={k: v for k, v in rec.items() if isinstance(v, (str, int, float, bool)) or v is None},
    )


def import_csv(text: str, *, source: str = "upload", delimiter: str | None = None) -> ImportResult:
    res = ImportResult(format="csv")
    sample = text[:8192]
    if delimiter is None:
        try:
            delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
        except csv.Error:
            delimiter = ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    if not reader.fieldnames:
        res.add_warning("CSV has no header row")
        return res
    norm = {_norm_key(f) for f in reader.fieldnames}
    if not (norm & set(COLUMN_ALIASES["text"])):
        res.add_warning(f"no text column found (columns: {', '.join(reader.fieldnames[:12])}); expected one of {', '.join(COLUMN_ALIASES['text'][:6])}")
        return res
    seen: set[str] = set()
    for i, row in enumerate(reader, 1):
        try:
            p = generic_record_to_post(row, source=source, prefix="csv")
        except Exception as exc:  # pragma: no cover - defensive
            res.skipped += 1
            res.add_warning(f"row {i}: {exc}")
            continue
        if p is None or p.id in seen:
            res.skipped += 1
            continue
        seen.add(p.id)
        res.posts.append(p)
    return res


def import_plain_text(text: str, *, source: str = "upload") -> ImportResult:
    res = ImportResult(format="text")
    now = datetime.utcnow()
    for i, line in enumerate(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        pid = _stable_id("txt", line)
        res.posts.append(Post(id=pid, text=line[:MAX_TEXT_LEN], created_at=now, author_id="unknown", source=source, raw={"line": i + 1}))
    return res


# --------------------------------------------------------------------------------------------
# X data archive (legacy v1.1 tweet objects)
# --------------------------------------------------------------------------------------------


def legacy_to_post(t: dict[str, Any], *, source: str = "archive", account: Author | None = None) -> Post | None:
    text = t.get("full_text") or t.get("text") or ""
    tid = str(t.get("id_str") or t.get("id") or "")
    if not tid or not text:
        return None
    user = t.get("user") if isinstance(t.get("user"), dict) else None
    if user:
        author_id = str(user.get("id_str") or user.get("id"))
        author = Author(id=author_id, username=user.get("screen_name", f"user_{author_id}"), display_name=user.get("name"), created_at=parse_datetime(user.get("created_at")), followers_count=_int(user.get("followers_count")), following_count=_int(user.get("friends_count")), tweet_count=_int(user.get("statuses_count")), listed_count=_int(user.get("listed_count")), verified=bool(user.get("verified")), description=user.get("description"), location=user.get("location"), has_default_profile_image=bool(user.get("default_profile_image")))
    elif account:
        author_id, author = account.id, account
    else:
        author_id, author = "archive-owner", Author(id="archive-owner", username="archive_owner")
    refs = []
    if t.get("in_reply_to_status_id_str") or t.get("in_reply_to_status_id"):
        refs.append(ReferencedPost(type="replied_to", id=str(t.get("in_reply_to_status_id_str") or t.get("in_reply_to_status_id"))))
    rs = t.get("retweeted_status")
    if isinstance(rs, dict) and (rs.get("id_str") or rs.get("id")):
        refs.append(ReferencedPost(type="retweeted", id=str(rs.get("id_str") or rs.get("id"))))
    elif text.startswith("RT @") and not rs:
        pass  # archive retweets have no upstream id; keep as plain text
    if t.get("quoted_status_id_str") or t.get("quoted_status_id"):
        refs.append(ReferencedPost(type="quoted", id=str(t.get("quoted_status_id_str") or t.get("quoted_status_id"))))
    media = []
    ents = t.get("extended_entities") or t.get("entities") or {}
    for i, m in enumerate(ents.get("media", []) or []):
        media.append(MediaAttachment(media_key=str(m.get("id_str") or f"{tid}-m{i}"), type=m.get("type", "photo"), url=m.get("media_url_https") or m.get("media_url"), alt_text=m.get("ext_alt_text")))
    return Post(
        id=tid,
        text=text[:MAX_TEXT_LEN],
        created_at=parse_datetime(t.get("created_at")) or datetime.utcnow(),
        author_id=author_id,
        author=author,
        lang=t.get("lang"),
        in_reply_to_user_id=str(t["in_reply_to_user_id_str"]) if t.get("in_reply_to_user_id_str") else None,
        referenced=refs,
        engagement=Engagement(like_count=_int(t.get("favorite_count")), retweet_count=_int(t.get("retweet_count")), reply_count=_int(t.get("reply_count")), quote_count=_int(t.get("quote_count"))),
        media=media,
        source=source,
        raw={k: v for k, v in t.items() if k not in ("entities", "extended_entities", "user")},
    )


_YTD_RE = re.compile(r"^\s*window\.YTD\.[A-Za-z0-9_]+\.part\d+\s*=\s*", re.S)


def import_archive_js(text: str, account: Author | None = None) -> ImportResult:
    res = ImportResult(format="x-archive")
    body = _YTD_RE.sub("", text.strip(), count=1)
    try:
        items = json.loads(body)
    except ValueError as exc:
        res.add_warning(f"could not parse archive JS: {exc}")
        return res
    for item in items if isinstance(items, list) else []:
        t = item.get("tweet") if isinstance(item, dict) and "tweet" in item else item
        if not isinstance(t, dict):
            res.skipped += 1
            continue
        p = legacy_to_post(t, account=account)
        if p:
            res.posts.append(p)
        else:
            res.skipped += 1
    return res


def _archive_account(zf: zipfile.ZipFile) -> Author | None:
    for name in zf.namelist():
        if name.endswith("data/account.js"):
            try:
                items = json.loads(_YTD_RE.sub("", zf.read(name).decode("utf-8-sig"), count=1))
                acc = items[0]["account"]
                return Author(id=str(acc["accountId"]), username=acc.get("username", "archive_owner"), display_name=acc.get("accountDisplayName"), created_at=parse_datetime(acc.get("createdAt")))
            except (ValueError, KeyError, IndexError, TypeError):
                return None
    return None


def import_archive_zip(data: bytes) -> ImportResult:
    res = ImportResult(format="x-archive-zip")
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        res.add_warning("not a valid zip file")
        return res
    with zf:
        account = _archive_account(zf)
        names = [n for n in zf.namelist() if re.search(r"(^|/)tweets?(-part\d+)?\.js$", n)]
        if not names:
            names = [n for n in zf.namelist() if n.lower().endswith((".json", ".jsonl", ".csv"))]
            if not names:
                res.add_warning("zip contains no tweets.js / .json / .csv files")
                return res
        seen: set[str] = set()
        for name in sorted(names):
            raw = zf.read(name)
            sub = import_archive_js(raw.decode("utf-8-sig", errors="replace"), account) if name.endswith(".js") else import_bytes(raw, name, source="archive")
            res.skipped += sub.skipped
            res.warnings.extend(sub.warnings)
            for p in sub.posts:
                if p.id not in seen:
                    seen.add(p.id)
                    res.posts.append(p)
    return res


def chunked(items: Iterable[Post], size: int = 200) -> Iterator[list[Post]]:
    batch: list[Post] = []
    for it in items:
        batch.append(it)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch
