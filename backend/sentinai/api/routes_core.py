"""Auth, health, classification and post browsing endpoints."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from sentinai import __version__
from sentinai.api.schemas import ClassifyRequest, ClassifyResponse, IngestBatchRequest, LoginRequest, TokenResponse
from sentinai.auth import PERMISSIONS, TOKEN_COOKIE, CurrentUser, authenticate, create_token, get_current_user, require, session_cookie_params
from sentinai.classification.context import ParentInfo
from sentinai.classification.engine import get_engine
from sentinai.config import get_settings
from sentinai.ingestion.worker import IngestionWorker
from sentinai.schemas import Author, Engagement, MediaAttachment, Post, ReferencedPost
from sentinai.storage.db import get_session
from sentinai.storage.models import AuthorRow, ClassificationRow, PostRow

router = APIRouter()


# --------------------------------------------------------------------------------------------
# health / meta
# --------------------------------------------------------------------------------------------


@router.get("/health", tags=["meta"])
def health(session: Session = Depends(get_session)):
    s = get_settings()
    engine = get_engine()
    return {
        "status": "ok",
        "version": __version__,
        "environment": s.environment,
        "classifier_backend": engine.backend_name,
        "model_version": engine.model_version,
        "posts": session.scalar(select(func.count()).select_from(PostRow)) or 0,
        "time": datetime.utcnow().isoformat(),
    }


@router.get("/meta/taxonomy", tags=["meta"])
def taxonomy():
    from sentinai.schemas import SEVERITY_NAMES, Stance, ToxicityLabel

    lex = get_engine().lexicon
    targets: dict[str, list[str]] = {}
    for g in lex.groups:
        targets.setdefault(g.category, [])
        if g.label not in targets[g.category]:
            targets[g.category].append(g.label)
    return {
        "toxicity_labels": [l.value for l in ToxicityLabel],  # noqa: E741
        "severity_levels": [{"level": int(k), "name": v} for k, v in SEVERITY_NAMES.items()],
        "stances": [s.value for s in Stance],
        "targets": {k: sorted(v) for k, v in targets.items()},
        "languages": ["en", "es", "ar", "hi", "fr", "pt", "de"],
        "roles": {k: sorted(v) for k, v in PERMISSIONS.items()},
    }


# --------------------------------------------------------------------------------------------
# auth
# --------------------------------------------------------------------------------------------


@router.post("/auth/login", response_model=TokenResponse, tags=["auth"])
def login(body: LoginRequest, request: Request, response: Response, session: Session = Depends(get_session)):
    user = authenticate(session, body.username, body.password)
    if user is None:
        raise HTTPException(401, "Invalid username or password")
    token = create_token(user.username, user.role)
    # Belt and braces: the SPA sends the token as a header; the cookie covers clients/proxies that drop it.
    response.set_cookie(value=token, **session_cookie_params(request))
    return TokenResponse(access_token=token, role=user.role, username=user.username, permissions=sorted(PERMISSIONS[user.role]))


@router.post("/auth/logout", tags=["auth"])
def logout(response: Response):
    response.delete_cookie(TOKEN_COOKIE, path="/")
    return {"ok": True}


@router.get("/auth/me", tags=["auth"])
def me(user: CurrentUser = Depends(get_current_user)):
    return {"username": user.username, "role": user.role, "permissions": sorted(PERMISSIONS[user.role])}


# --------------------------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------------------------


@router.post("/classify", response_model=ClassifyResponse, tags=["classification"])
def classify(body: ClassifyRequest, user: CurrentUser = Depends(require("read"))):
    engine = get_engine()
    refs = []
    if body.is_reply:
        refs.append(ReferencedPost(type="replied_to", id="parent"))
    if body.is_quote:
        refs.append(ReferencedPost(type="quoted", id="parent"))
    media = [MediaAttachment(media_key=f"alt{i}", alt_text=t) for i, t in enumerate(body.media_alt_text)]
    post = Post(id=body.post_id or "adhoc", text=body.text, created_at=datetime.utcnow(), author_id="adhoc", referenced=refs, media=media)
    parent = ParentInfo(id="parent", text=body.parent_text, toxicity=body.parent_toxicity) if (body.parent_text or body.parent_toxicity is not None) else None
    # ad-hoc classification must not pollute the dedup index
    pre = engine.preprocessor.run(body.text)
    c = engine.classify_post(post, parent=parent, explain=body.explain)
    c.duplicate_of = None
    return ClassifyResponse(classification=c, preprocessed=pre.model_dump(mode="json", exclude={"minhash_signature"}))


@router.post("/ingest/posts", tags=["ingestion"])
def ingest_posts(body: IngestBatchRequest, user: CurrentUser = Depends(require("ingest")), session: Session = Depends(get_session)):
    """Push posts directly (webhook / offline dump).  Live X API ingestion runs via the CLI worker."""
    posts = []
    for p in body.posts:
        refs = []
        if p.parent_id:
            refs.append(ReferencedPost(type="replied_to", id=p.parent_id))
        if p.quoted_id:
            refs.append(ReferencedPost(type="quoted", id=p.quoted_id))
        if p.retweeted_id:
            refs.append(ReferencedPost(type="retweeted", id=p.retweeted_id))
        posts.append(
            Post(
                id=p.id,
                text=p.text,
                created_at=p.created_at,
                author_id=p.author_id,
                author=Author(id=p.author_id, username=p.author_username or f"user_{p.author_id}") if p.author_username else None,
                lang=p.lang,
                referenced=refs,
                engagement=Engagement(like_count=p.like_count, retweet_count=p.retweet_count, reply_count=p.reply_count, quote_count=p.quote_count),
                media=p.media,
                raw=p.model_dump(mode="json"),
            )
        )
    n = IngestionWorker(session).process(posts)
    return {"ingested": n}


# --------------------------------------------------------------------------------------------
# posts browsing
# --------------------------------------------------------------------------------------------


def _post_dict(post: PostRow, cls: ClassificationRow | None, author: AuthorRow | None, full: bool = False) -> dict:
    d = {
        "id": post.id,
        "text": post.text,
        "created_at": post.created_at.isoformat(),
        "author_id": post.author_id,
        "author_username": author.username if author else None,
        "bot_probability": author.bot_probability if author else None,
        "lang": post.lang,
        "parent_id": post.parent_id,
        "quoted_id": post.quoted_id,
        "retweeted_id": post.retweeted_id,
        "engagement": {"likes": post.like_count, "retweets": post.retweet_count, "replies": post.reply_count, "quotes": post.quote_count},
        "has_media": post.has_media,
        "media": post.media,
        "source": post.source,
        "deleted_upstream": post.deleted_upstream,
    }
    if cls:
        d.update(
            {
                "toxicity_label": cls.toxicity_label,
                "toxicity_confidence": cls.toxicity_confidence,
                "toxicity_score": cls.toxicity_score,
                "final_toxicity": cls.final_toxicity,
                "severity_level": cls.severity_level,
                "targets": cls.target_labels or [],
                "stance": cls.stance,
                "discount_factor": cls.discount_factor,
                "visual_toxicity": cls.visual_toxicity,
                "needs_review": cls.needs_review,
                "duplicate_of": cls.duplicate_of,
                "language": cls.language,
                "model_version": cls.model_version,
            }
        )
        if full:
            d["classification"] = cls.payload
    return d


@router.get("/posts", tags=["posts"])
def list_posts(
    user: CurrentUser = Depends(require("read")),
    session: Session = Depends(get_session),
    label: str | None = None,
    min_toxicity: float = Query(0.0, ge=0, le=1),
    max_toxicity: float = Query(1.0, ge=0, le=1),
    severity: int | None = Query(None, ge=0, le=4),
    target: str | None = None,
    lang: str | None = None,
    q: str | None = None,
    exclude_bots: bool = False,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    order: str = Query("recent", pattern="^(recent|toxicity|severity)$"),
):
    stmt = select(PostRow, ClassificationRow, AuthorRow).join(ClassificationRow, ClassificationRow.post_id == PostRow.id).join(AuthorRow, AuthorRow.id == PostRow.author_id).where(PostRow.deleted_upstream.is_(False))
    stmt = stmt.where(ClassificationRow.final_toxicity >= min_toxicity, ClassificationRow.final_toxicity <= max_toxicity)
    if label:
        stmt = stmt.where(ClassificationRow.toxicity_label == label)
    if severity is not None:
        stmt = stmt.where(ClassificationRow.severity_level == severity)
    if lang:
        stmt = stmt.where(ClassificationRow.language == lang)
    if q:
        stmt = stmt.where(PostRow.text.ilike(f"%{q}%"))
    if target:
        from sentinai.storage.models import TargetHitRow

        cat, _, lab = target.partition(":")
        sub = select(TargetHitRow.post_id).where(TargetHitRow.category == cat)
        if lab:
            sub = sub.where(TargetHitRow.label == lab)
        stmt = stmt.where(PostRow.id.in_(sub))
    if exclude_bots:
        stmt = stmt.where((AuthorRow.bot_probability.is_(None)) | (AuthorRow.bot_probability < 0.8))
    count = session.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    if order == "toxicity":
        stmt = stmt.order_by(ClassificationRow.final_toxicity.desc(), PostRow.created_at.desc())
    elif order == "severity":
        stmt = stmt.order_by(ClassificationRow.severity_level.desc(), ClassificationRow.final_toxicity.desc())
    else:
        stmt = stmt.order_by(PostRow.created_at.desc())
    rows = session.execute(stmt.offset(offset).limit(limit)).all()
    return {"total": count, "items": [_post_dict(p, c, a) for p, c, a in rows]}


@router.get("/posts/{post_id}", tags=["posts"])
def get_post(post_id: str, user: CurrentUser = Depends(require("read")), session: Session = Depends(get_session)):
    post = session.get(PostRow, post_id)
    if post is None:
        raise HTTPException(404, "post not found")
    cls = session.get(ClassificationRow, post_id)
    author = session.get(AuthorRow, post.author_id)
    d = _post_dict(post, cls, author, full=True)
    # thread context
    parent = session.get(PostRow, post.parent_id or post.quoted_id or post.retweeted_id or "") if (post.parent_id or post.quoted_id or post.retweeted_id) else None
    if parent:
        d["parent"] = _post_dict(parent, session.get(ClassificationRow, parent.id), session.get(AuthorRow, parent.author_id))
    children = session.execute(
        select(PostRow, ClassificationRow, AuthorRow).join(ClassificationRow, ClassificationRow.post_id == PostRow.id).join(AuthorRow, AuthorRow.id == PostRow.author_id).where((PostRow.parent_id == post_id) | (PostRow.quoted_id == post_id)).order_by(PostRow.created_at).limit(50)
    ).all()
    d["replies"] = [_post_dict(p, c, a) for p, c, a in children]
    return d


@router.post("/posts/{post_id}/reclassify", tags=["posts"])
def reclassify(post_id: str, user: CurrentUser = Depends(require("review")), session: Session = Depends(get_session)):
    from sentinai.storage.repository import row_to_post, save_classification

    post = session.get(PostRow, post_id)
    if post is None:
        raise HTTPException(404, "post not found")
    worker = IngestionWorker(session)
    p = row_to_post(post)
    c = get_engine().classify_post(p, parent=worker.parent_info(p))
    save_classification(session, p, c)
    session.commit()
    return c
