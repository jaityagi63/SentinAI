"""Request / response models specific to the HTTP API."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from sentinai.schemas import Classification, MediaAttachment, SeverityLevel, ToxicityLabel


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    username: str
    permissions: list[str]


class ClassifyRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000)
    post_id: str | None = None
    parent_text: str | None = None
    parent_toxicity: float | None = None
    is_reply: bool = False
    is_quote: bool = False
    media_alt_text: list[str] = Field(default_factory=list)
    explain: bool = True


class ClassifyResponse(BaseModel):
    classification: Classification
    preprocessed: dict


class IngestPostRequest(BaseModel):
    id: str
    text: str
    created_at: datetime
    author_id: str
    author_username: str | None = None
    lang: str | None = None
    parent_id: str | None = None
    quoted_id: str | None = None
    retweeted_id: str | None = None
    like_count: int = 0
    retweet_count: int = 0
    reply_count: int = 0
    quote_count: int = 0
    media: list[MediaAttachment] = Field(default_factory=list)


class IngestBatchRequest(BaseModel):
    posts: list[IngestPostRequest]


class AnnotationRequest(BaseModel):
    post_id: str
    toxicity_label: ToxicityLabel
    severity: SeverityLevel | None = None
    targets: list[str] = Field(default_factory=list)
    notes: str | None = None


class EventCreate(BaseModel):
    date: datetime
    title: str
    category: str = "news"
    source: str = "manual"
    url: str | None = None
    related_targets: list[str] = Field(default_factory=list)
    magnitude: float = 1.0
