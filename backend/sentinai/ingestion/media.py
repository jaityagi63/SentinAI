"""Optional media download so Module 5 (OCR / CLIP / hate-symbol detector) can run on real
attachments.  Files land in ``data/media/<post_id>/<media_key>.<ext>``; ``MediaAttachment.local_path``
is filled in so :class:`sentinai.classification.multimodal.MultimodalAnalyzer` picks them up.

Only photos (and video/GIF preview frames) are fetched — never the video streams themselves.
Failures are logged and skipped; ingestion must not fail because an image is gone.
"""

from __future__ import annotations

import logging
import mimetypes
from pathlib import Path

import httpx

from sentinai.config import get_settings
from sentinai.schemas import Post

log = logging.getLogger(__name__)

MAX_BYTES = 8 * 1024 * 1024
_ALLOWED = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp", "image/gif": ".gif"}


def media_dir() -> Path:
    d = get_settings().data_dir / "media"
    d.mkdir(parents=True, exist_ok=True)
    return d


def download_post_media(posts: list[Post], client: httpx.Client | None = None, *, limit_per_post: int = 4) -> int:
    """Download photo attachments for ``posts`` in place.  Returns the number of files fetched."""
    todo = [(p, m) for p in posts for m in p.media[:limit_per_post] if m.url and not m.local_path]
    if not todo:
        return 0
    own = client is None
    client = client or httpx.Client(timeout=20, follow_redirects=True, headers={"User-Agent": "SentinAI/0.1 (research)"})
    n = 0
    try:
        for post, m in todo:
            target_dir = media_dir() / post.id
            existing = next(iter(target_dir.glob(f"{_safe(m.media_key)}.*")), None) if target_dir.exists() else None
            if existing:
                m.local_path = str(existing)
                continue
            try:
                with client.stream("GET", m.url) as resp:
                    if resp.status_code != 200:
                        log.info("media %s for post %s: HTTP %d", m.url, post.id, resp.status_code)
                        continue
                    ctype = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
                    ext = _ALLOWED.get(ctype) or mimetypes.guess_extension(ctype) or ""
                    if ctype and not ctype.startswith("image/"):
                        continue
                    size = int(resp.headers.get("content-length") or 0)
                    if size > MAX_BYTES:
                        continue
                    target_dir.mkdir(parents=True, exist_ok=True)
                    path = target_dir / f"{_safe(m.media_key)}{ext or '.img'}"
                    total = 0
                    with open(path, "wb") as fh:
                        for chunk in resp.iter_bytes():
                            total += len(chunk)
                            if total > MAX_BYTES:
                                break
                            fh.write(chunk)
                    if total > MAX_BYTES:
                        path.unlink(missing_ok=True)
                        continue
                m.local_path = str(path)
                n += 1
            except httpx.HTTPError as exc:
                log.info("media download failed for %s: %s", m.url, exc)
    finally:
        if own:
            client.close()
    return n


def _safe(key: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in key)[:80] or "media"
