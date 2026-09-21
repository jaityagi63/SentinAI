"""Persistence helpers used by the ingestion worker, the API and the demo seeder."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from sentinai.schemas import Author, Classification, Post
from sentinai.storage.models import AuthorRow, ClassificationRow, PostRow, ReviewQueueRow, TargetHitRow
from sentinai.storage.raw_store import get_raw_store


def upsert_author(session: Session, author: Author) -> AuthorRow:
    row = session.get(AuthorRow, author.id)
    if row is None:
        row = AuthorRow(id=author.id, username=author.username)
        session.add(row)
    row.username = author.username
    row.display_name = author.display_name
    row.created_at = author.created_at
    row.followers_count = author.followers_count
    row.following_count = author.following_count
    row.tweet_count = author.tweet_count
    row.listed_count = author.listed_count
    row.verified = author.verified
    row.description = author.description
    row.location = author.location
    row.has_default_profile_image = author.has_default_profile_image
    return row


def upsert_post(session: Session, post: Post, store_raw: bool = True) -> PostRow:
    if post.author is not None:
        upsert_author(session, post.author)
    elif session.get(AuthorRow, post.author_id) is None:
        session.add(AuthorRow(id=post.author_id, username=f"user_{post.author_id}"))

    row = session.get(PostRow, post.id)
    if row is None:
        row = PostRow(id=post.id)
        session.add(row)
    row.text = post.text
    row.created_at = post.created_at
    row.author_id = post.author_id
    row.lang = post.lang
    row.conversation_id = post.conversation_id
    row.in_reply_to_user_id = post.in_reply_to_user_id
    row.parent_id = post.parent_id
    row.quoted_id = post.quoted_id
    row.retweeted_id = post.retweeted_id
    row.like_count = post.engagement.like_count
    row.retweet_count = post.engagement.retweet_count
    row.reply_count = post.engagement.reply_count
    row.quote_count = post.engagement.quote_count
    row.impression_count = post.engagement.impression_count
    row.has_media = bool(post.media)
    row.media = [m.model_dump(mode="json") for m in post.media] if post.media else None
    row.source = post.source
    if store_raw and post.raw is not None:
        get_raw_store(session).put(post.id, post.raw)
    return row


def save_classification(session: Session, post: Post, c: Classification, queue_for_review: bool = True) -> ClassificationRow:
    row = session.get(ClassificationRow, c.post_id)
    if row is None:
        row = ClassificationRow(post_id=c.post_id)
        session.add(row)
    row.model_version = c.model_version
    row.language = c.language
    row.toxicity_label = c.toxicity.label.value
    row.toxicity_confidence = c.toxicity.confidence
    row.toxicity_score = c.toxicity.toxicity_score
    row.final_toxicity = c.final_toxicity
    row.severity_level = int(c.severity.level)
    row.severity_expected = c.severity.expected_level
    row.target_labels = [f"{t.category.value}:{t.label}" for t in c.targets]
    row.stance = c.context.stance.value if c.context else None
    row.discount_factor = c.context.discount_factor if c.context else 1.0
    row.visual_toxicity = c.visual.visual_toxicity if c.visual else None
    row.obfuscation_score = c.obfuscation_score
    row.duplicate_of = c.duplicate_of
    row.needs_review = c.needs_review
    row.payload = c.model_dump(mode="json")
    row.classified_at = c.classified_at

    # denormalised target hits
    existing = {(h.category, h.label): h for h in session.scalars(select(TargetHitRow).where(TargetHitRow.post_id == c.post_id)).all()}
    wanted = {(t.category.value, t.label): t for t in c.targets}
    for key, hit in existing.items():
        if key not in wanted:
            session.delete(hit)
    for key, t in wanted.items():
        hit = existing.get(key)
        if hit is None:
            hit = TargetHitRow(post_id=c.post_id, category=key[0], label=key[1])
            session.add(hit)
        hit.confidence = t.confidence
        hit.created_at = post.created_at
        hit.severity_level = int(c.severity.level)
        hit.final_toxicity = c.final_toxicity
        hit.author_id = post.author_id

    if queue_for_review and c.needs_review:
        q = session.scalar(select(ReviewQueueRow).where(ReviewQueueRow.post_id == c.post_id))
        if q is None:
            session.add(
                ReviewQueueRow(
                    post_id=c.post_id,
                    model_label=c.toxicity.label.value,
                    model_confidence=c.toxicity.confidence,
                    reason="active_learning",
                    priority=round(1.0 - abs(c.final_toxicity - 0.5) * 2, 4),
                )
            )
    return row


def mark_deleted_upstream(session: Session, post_ids: list[str]) -> int:
    """Compliance: posts deleted on X must not be retained (Module 1)."""
    n = 0
    raw = get_raw_store(session)
    for pid in post_ids:
        row = session.get(PostRow, pid)
        if row is None:
            continue
        row.deleted_upstream = True
        row.text = "[deleted upstream]"
        row.media = None
        raw.delete(pid)
        n += 1
    return n


def row_to_post(row: PostRow) -> Post:
    from sentinai.schemas import Engagement, MediaAttachment, ReferencedPost

    refs = []
    if row.parent_id:
        refs.append(ReferencedPost(type="replied_to", id=row.parent_id))
    if row.quoted_id:
        refs.append(ReferencedPost(type="quoted", id=row.quoted_id))
    if row.retweeted_id:
        refs.append(ReferencedPost(type="retweeted", id=row.retweeted_id))
    return Post(
        id=row.id,
        text=row.text,
        created_at=row.created_at or datetime.utcnow(),
        author_id=row.author_id,
        lang=row.lang,
        conversation_id=row.conversation_id,
        in_reply_to_user_id=row.in_reply_to_user_id,
        referenced=refs,
        engagement=Engagement(
            like_count=row.like_count, retweet_count=row.retweet_count, reply_count=row.reply_count, quote_count=row.quote_count, impression_count=row.impression_count
        ),
        media=[MediaAttachment(**m) for m in (row.media or [])],
        source=row.source,
    )
