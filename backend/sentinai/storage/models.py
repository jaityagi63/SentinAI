"""Relational schema (PostgreSQL in production, SQLite for dev/tests).

Structured metadata lives here (Module 1).  Raw JSON payloads go to the raw store
(MongoDB / Elasticsearch) or, when none is configured, to :class:`RawPayload`.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class AuthorRow(Base):
    __tablename__ = "authors"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    username: Mapped[str] = mapped_column(String(128), index=True)
    display_name: Mapped[str | None] = mapped_column(String(256))
    created_at: Mapped[datetime | None] = mapped_column(DateTime)
    followers_count: Mapped[int] = mapped_column(Integer, default=0)
    following_count: Mapped[int] = mapped_column(Integer, default=0)
    tweet_count: Mapped[int] = mapped_column(Integer, default=0)
    listed_count: Mapped[int] = mapped_column(Integer, default=0)
    verified: Mapped[bool] = mapped_column(Boolean, default=False)
    description: Mapped[str | None] = mapped_column(Text)
    location: Mapped[str | None] = mapped_column(String(256))
    has_default_profile_image: Mapped[bool] = mapped_column(Boolean, default=False)
    bot_probability: Mapped[float | None] = mapped_column(Float)
    bot_features: Mapped[dict | None] = mapped_column(JSON)
    account_score: Mapped[float | None] = mapped_column(Float)
    account_score_json: Mapped[dict | None] = mapped_column(JSON)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    posts: Mapped[list[PostRow]] = relationship(back_populates="author")


class PostRow(Base):
    __tablename__ = "posts"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    text: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    author_id: Mapped[str] = mapped_column(ForeignKey("authors.id"), index=True)
    lang: Mapped[str | None] = mapped_column(String(8), index=True)
    conversation_id: Mapped[str | None] = mapped_column(String(64), index=True)
    in_reply_to_user_id: Mapped[str | None] = mapped_column(String(64))
    parent_id: Mapped[str | None] = mapped_column(String(64), index=True)
    quoted_id: Mapped[str | None] = mapped_column(String(64), index=True)
    retweeted_id: Mapped[str | None] = mapped_column(String(64), index=True)
    like_count: Mapped[int] = mapped_column(Integer, default=0)
    retweet_count: Mapped[int] = mapped_column(Integer, default=0)
    reply_count: Mapped[int] = mapped_column(Integer, default=0)
    quote_count: Mapped[int] = mapped_column(Integer, default=0)
    impression_count: Mapped[int] = mapped_column(Integer, default=0)
    has_media: Mapped[bool] = mapped_column(Boolean, default=False)
    media: Mapped[list | None] = mapped_column(JSON)
    source: Mapped[str] = mapped_column(String(16), default="x")
    ingested_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    # compliance: soft-delete marker set by the compliance sweep (post deleted upstream)
    deleted_upstream: Mapped[bool] = mapped_column(Boolean, default=False, index=True)

    author: Mapped[AuthorRow] = relationship(back_populates="posts")
    classification: Mapped[ClassificationRow | None] = relationship(back_populates="post", uselist=False, cascade="all, delete-orphan")


class ClassificationRow(Base):
    __tablename__ = "classifications"

    post_id: Mapped[str] = mapped_column(ForeignKey("posts.id", ondelete="CASCADE"), primary_key=True)
    model_version: Mapped[str] = mapped_column(String(64))
    language: Mapped[str] = mapped_column(String(8), index=True)
    toxicity_label: Mapped[str] = mapped_column(String(32), index=True)
    toxicity_confidence: Mapped[float] = mapped_column(Float)
    toxicity_score: Mapped[float] = mapped_column(Float, index=True)
    final_toxicity: Mapped[float] = mapped_column(Float, index=True)
    severity_level: Mapped[int] = mapped_column(Integer, index=True)
    severity_expected: Mapped[float] = mapped_column(Float)
    target_labels: Mapped[list | None] = mapped_column(JSON)  # ["religion:islam", ...]
    stance: Mapped[str | None] = mapped_column(String(16))
    discount_factor: Mapped[float] = mapped_column(Float, default=1.0)
    visual_toxicity: Mapped[float | None] = mapped_column(Float)
    obfuscation_score: Mapped[float] = mapped_column(Float, default=0.0)
    duplicate_of: Mapped[str | None] = mapped_column(String(64), index=True)
    needs_review: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    payload: Mapped[dict] = mapped_column(JSON)  # full Classification JSON (explanation etc.)
    classified_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    post: Mapped[PostRow] = relationship(back_populates="classification")


class TargetHitRow(Base):
    """Denormalised (post, target) rows for fast aggregation by demographic."""

    __tablename__ = "target_hits"
    __table_args__ = (UniqueConstraint("post_id", "category", "label", name="uq_target_hit"), Index("ix_target_hits_cat_label", "category", "label"))

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    post_id: Mapped[str] = mapped_column(ForeignKey("posts.id", ondelete="CASCADE"), index=True)
    category: Mapped[str] = mapped_column(String(16))
    label: Mapped[str] = mapped_column(String(64))
    confidence: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    severity_level: Mapped[int] = mapped_column(Integer)
    final_toxicity: Mapped[float] = mapped_column(Float)
    author_id: Mapped[str] = mapped_column(String(64), index=True)


class RawPayload(Base):
    """Fallback raw store when no MongoDB / Elasticsearch URL is configured."""

    __tablename__ = "raw_payloads"

    post_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    payload: Mapped[dict] = mapped_column(JSON)
    stored_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class ReviewQueueRow(Base):
    __tablename__ = "review_queue"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    post_id: Mapped[str] = mapped_column(ForeignKey("posts.id", ondelete="CASCADE"), unique=True, index=True)
    model_label: Mapped[str] = mapped_column(String(32))
    model_confidence: Mapped[float] = mapped_column(Float)
    reason: Mapped[str] = mapped_column(String(64), default="active_learning")
    priority: Mapped[float] = mapped_column(Float, default=0.0, index=True)
    status: Mapped[str] = mapped_column(String(16), default="pending", index=True)
    assigned_to: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime)


class AnnotationRow(Base):
    __tablename__ = "annotations"
    __table_args__ = (UniqueConstraint("post_id", "annotator", name="uq_annotation"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    post_id: Mapped[str] = mapped_column(ForeignKey("posts.id", ondelete="CASCADE"), index=True)
    annotator: Mapped[str] = mapped_column(String(64), index=True)
    toxicity_label: Mapped[str] = mapped_column(String(32))
    severity: Mapped[int | None] = mapped_column(Integer)
    targets: Mapped[list | None] = mapped_column(JSON)
    notes: Mapped[str | None] = mapped_column(Text)
    agrees_with_model: Mapped[bool | None] = mapped_column(Boolean)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    used_in_training: Mapped[bool] = mapped_column(Boolean, default=False, index=True)


class RetrainBatchRow(Base):
    """Biweekly fine-tuning batches assembled from human corrections (Module 14)."""

    __tablename__ = "retrain_batches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    n_examples: Mapped[int] = mapped_column(Integer)
    n_corrections: Mapped[int] = mapped_column(Integer)
    export_path: Mapped[str] = mapped_column(String(512))
    status: Mapped[str] = mapped_column(String(16), default="exported")
    notes: Mapped[str | None] = mapped_column(Text)


class EventRow(Base):
    """Real-world events for trend correlation (Module 8: NewsAPI / GDELT / manual)."""

    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[datetime] = mapped_column(DateTime, index=True)
    title: Mapped[str] = mapped_column(String(512))
    category: Mapped[str] = mapped_column(String(32), default="news")
    source: Mapped[str] = mapped_column(String(32), default="manual")
    url: Mapped[str | None] = mapped_column(String(1024))
    related_targets: Mapped[list | None] = mapped_column(JSON)
    magnitude: Mapped[float] = mapped_column(Float, default=1.0)


class UserRow(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(256))
    role: Mapped[str] = mapped_column(String(16), default="researcher")
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class FairnessAuditRow(Base):
    """Weekly false-positive audit snapshots across dialect groups (Module 11)."""

    __tablename__ = "fairness_audits"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    model_version: Mapped[str] = mapped_column(String(64))
    report: Mapped[dict] = mapped_column(JSON)


class IngestionCursorRow(Base):
    """Pagination / since_id cursors per query, so ingestion resumes after restarts."""

    __tablename__ = "ingestion_cursors"

    query: Mapped[str] = mapped_column(String(512), primary_key=True)
    since_id: Mapped[str | None] = mapped_column(String(64))
    next_token: Mapped[str | None] = mapped_column(String(128))
    last_run: Mapped[datetime | None] = mapped_column(DateTime)
    total_ingested: Mapped[int] = mapped_column(Integer, default=0)
