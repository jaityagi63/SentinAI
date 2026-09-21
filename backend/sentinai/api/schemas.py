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


# --- live X ingestion -------------------------------------------------------------------------


class XSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1024, description="X search operators, e.g. `(muslims OR islam) lang:en -is:retweet`")
    max_pages: int = Field(2, ge=1, le=50, description="pages of up to 100 (recent) / 500 (full-archive) posts")
    full_archive: bool | None = Field(None, description="use /search/all (needs full-archive access); default from settings")
    start_time: datetime | None = None
    end_time: datetime | None = None
    use_cursor: bool = Field(True, description="continue from the stored since_id for this query")
    hydrate_parents: bool = True
    download_media: bool | None = None
    background: bool = Field(True, description="run as a job and return immediately")


class XTweetRequest(BaseModel):
    ref: str = Field(min_length=5, max_length=512, description="tweet URL (x.com/twitter.com) or numeric id")
    include_conversation: bool = Field(False, description="also pull the replies (conversation_id search)")
    max_pages: int = Field(2, ge=1, le=20)
    download_media: bool | None = None
    background: bool = False


class XUserRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64, description="@handle or profile URL")
    max_pages: int = Field(3, ge=1, le=32)
    exclude_replies: bool = False
    exclude_retweets: bool = False
    start_time: datetime | None = None
    hydrate_parents: bool = True
    download_media: bool | None = None
    background: bool = True


class XStreamRule(BaseModel):
    value: str = Field(min_length=1, max_length=1024)
    tag: str | None = Field(None, max_length=128)


class XStreamRequest(BaseModel):
    rules: list[XStreamRule] | None = Field(None, description="replace the stream rules before connecting; omit to keep existing rules")
    sample: bool = Field(False, description="use the 1% sampled stream instead of the filtered stream")
    max_posts: int | None = Field(None, ge=1, le=1_000_000)
    max_minutes: float | None = Field(None, gt=0, le=24 * 60)
    download_media: bool | None = None


class XTokenRequest(BaseModel):
    bearer_token: str = Field(min_length=20, max_length=2048)
    persist: bool = Field(False, description="store under data/secrets so it survives restarts")


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
