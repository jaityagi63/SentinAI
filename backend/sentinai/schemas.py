"""Shared domain schemas (Pydantic v2) used across the pipeline, storage layer and API."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------------------------
# Enumerations
# --------------------------------------------------------------------------------------------


class ToxicityLabel(str, Enum):
    """Task A labels."""

    NON_TOXIC = "non_toxic"
    OFFENSIVE = "offensive"
    HATE_SPEECH = "hate_speech"
    VIOLENT_EXTREMISM = "violent_extremism"


class TargetCategory(str, Enum):
    """Task B top-level target demographic categories."""

    ETHNICITY = "ethnicity"
    RELIGION = "religion"
    NATIONALITY = "nationality"


class SeverityLevel(int, Enum):
    """Task C ordinal harm taxonomy (0 = none)."""

    NONE = 0
    MICROAGGRESSION = 1  # stereotyping
    DEHUMANIZATION = 2  # exclusion
    SLUR = 3  # targeted harassment
    INCITEMENT = 4  # incitement to violence


SEVERITY_NAMES = {
    SeverityLevel.NONE: "None",
    SeverityLevel.MICROAGGRESSION: "Microaggression / Stereotyping",
    SeverityLevel.DEHUMANIZATION: "Dehumanization / Exclusion",
    SeverityLevel.SLUR: "Slurs / Targeted Harassment",
    SeverityLevel.INCITEMENT: "Incitement to Violence",
}


class Stance(str, Enum):
    """Module 6 stance labels of a post relative to the content it responds to / quotes."""

    SUPPORT = "support"
    DENY = "deny"
    CONDEMN = "condemn"
    NEUTRAL = "neutral"
    QUERY = "query"


class ReviewStatus(str, Enum):
    PENDING = "pending"
    IN_REVIEW = "in_review"
    RESOLVED = "resolved"
    SKIPPED = "skipped"


class Role(str, Enum):
    ADMIN = "admin"
    RESEARCHER = "researcher"
    MODERATOR = "moderator"


# --------------------------------------------------------------------------------------------
# Ingestion objects
# --------------------------------------------------------------------------------------------


class Author(BaseModel):
    id: str
    username: str
    display_name: str | None = None
    created_at: datetime | None = None
    followers_count: int = 0
    following_count: int = 0
    tweet_count: int = 0
    listed_count: int = 0
    verified: bool = False
    description: str | None = None
    location: str | None = None
    profile_image_url: str | None = None
    has_default_profile_image: bool = False


class MediaAttachment(BaseModel):
    media_key: str
    type: str = "photo"  # photo | animated_gif | video
    url: str | None = None
    alt_text: str | None = None
    local_path: str | None = None
    width: int | None = None
    height: int | None = None


class ReferencedPost(BaseModel):
    type: str  # replied_to | quoted | retweeted
    id: str


class Engagement(BaseModel):
    like_count: int = 0
    retweet_count: int = 0
    reply_count: int = 0
    quote_count: int = 0
    impression_count: int = 0


class Post(BaseModel):
    """A normalised public post (X API v2 tweet object flattened)."""

    id: str
    text: str
    created_at: datetime
    author_id: str
    author: Author | None = None
    lang: str | None = None
    conversation_id: str | None = None
    in_reply_to_user_id: str | None = None
    referenced: list[ReferencedPost] = Field(default_factory=list)
    engagement: Engagement = Field(default_factory=Engagement)
    media: list[MediaAttachment] = Field(default_factory=list)
    source: str = "x"
    raw: dict[str, Any] | None = None

    @property
    def parent_id(self) -> str | None:
        for ref in self.referenced:
            if ref.type == "replied_to":
                return ref.id
        return None

    @property
    def quoted_id(self) -> str | None:
        for ref in self.referenced:
            if ref.type == "quoted":
                return ref.id
        return None

    @property
    def retweeted_id(self) -> str | None:
        for ref in self.referenced:
            if ref.type == "retweeted":
                return ref.id
        return None


# --------------------------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------------------------


class ObfuscationReport(BaseModel):
    leetspeak: list[str] = Field(default_factory=list)
    homoglyphs: list[str] = Field(default_factory=list)
    zero_width_chars: int = 0
    spaced_out_words: list[str] = Field(default_factory=list)
    repeated_chars: list[str] = Field(default_factory=list)
    score: float = 0.0  # 0..1 – how heavily the text appears to be evading filters

    @property
    def detected(self) -> bool:
        return self.score > 0


class PreprocessedText(BaseModel):
    original: str
    normalized: str  # cleaned text used by classifiers
    tokens: list[str]
    emojis: list[str] = Field(default_factory=list)
    hashtags: list[str] = Field(default_factory=list)
    mentions_count: int = 0
    urls_count: int = 0
    language: str = "und"
    language_confidence: float = 0.0
    script: str | None = None
    transliterated: bool = False
    obfuscation: ObfuscationReport = Field(default_factory=ObfuscationReport)
    minhash_signature: list[int] | None = None
    duplicate_of: str | None = None


# --------------------------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------------------------


class ToxicityResult(BaseModel):
    label: ToxicityLabel
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[str, float]
    toxicity_score: float = Field(ge=0.0, le=1.0, description="P(not non_toxic)")


class TargetResult(BaseModel):
    category: TargetCategory
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)


class SeverityResult(BaseModel):
    level: SeverityLevel
    level_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[int, float]
    expected_level: float = Field(ge=0.0, le=4.0)


class TokenAttribution(BaseModel):
    token: str
    start: int
    end: int
    weight: float  # >0 pushes toward toxic, <0 toward neutral


class Explanation(BaseModel):
    method: str  # shap | lime | lexicon-attribution
    text: str  # the text the attribution offsets refer to (normalized text)
    attributions: list[TokenAttribution]
    base_value: float = 0.0
    top_positive: list[str] = Field(default_factory=list)
    top_negative: list[str] = Field(default_factory=list)
    matched_patterns: list[dict[str, Any]] = Field(default_factory=list)


class VisualResult(BaseModel):
    ocr_text: str = ""
    ocr_engine: str | None = None
    visual_toxicity: float = 0.0
    symbols: list[dict[str, Any]] = Field(default_factory=list)
    vlm_labels: dict[str, float] = Field(default_factory=dict)
    juxtaposition_flag: bool = False


class ContextResult(BaseModel):
    is_reply: bool = False
    is_quote: bool = False
    is_retweet: bool = False
    is_originator: bool = True
    parent_id: str | None = None
    stance: Stance = Stance.NEUTRAL
    stance_confidence: float = 0.0
    discount_factor: float = 1.0
    reasoning: str | None = None


class Classification(BaseModel):
    """Full multi-task result for a single post."""

    post_id: str
    model_version: str
    language: str
    toxicity: ToxicityResult
    targets: list[TargetResult]
    severity: SeverityResult
    explanation: Explanation | None = None
    visual: VisualResult | None = None
    context: ContextResult | None = None
    obfuscation_score: float = 0.0
    duplicate_of: str | None = None
    final_toxicity: float = Field(ge=0.0, le=1.0, description="text ⊕ visual ⊕ context-adjusted")
    needs_review: bool = False
    classified_at: datetime = Field(default_factory=datetime.utcnow)


# --------------------------------------------------------------------------------------------
# Analytics
# --------------------------------------------------------------------------------------------


class BotScore(BaseModel):
    author_id: str
    probability: float = Field(ge=0.0, le=1.0)
    features: dict[str, float]
    top_signals: list[str] = Field(default_factory=list)
    model: str = "heuristic-rf"


class AccountScore(BaseModel):
    author_id: str
    username: str | None = None
    n_posts: int
    reliable: bool
    score: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    avg_severity: float = 0.0
    toxic_ratio: float = 0.0
    target_diversity: float = 0.0
    recency_weight: float = 0.0
    bot_probability: float | None = None
    disclaimer: str = "Model-Estimated Probability — not a definitive human judgment."


class ReviewItem(BaseModel):
    id: int
    post_id: str
    text: str
    model_label: str
    model_confidence: float
    status: ReviewStatus
    created_at: datetime
    assigned_to: str | None = None


class Annotation(BaseModel):
    post_id: str
    annotator: str
    toxicity_label: ToxicityLabel
    severity: SeverityLevel | None = None
    targets: list[str] = Field(default_factory=list)
    notes: str | None = None
