"""Analytics endpoints powering the dashboard views (Modules 8–13)."""

from __future__ import annotations

from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from sentinai.analytics import accounts as acc
from sentinai.analytics import bots, fairness, network, trends
from sentinai.api.schemas import EventCreate
from sentinai.auth import CurrentUser, require
from sentinai.storage.db import get_session
from sentinai.storage.models import AuthorRow, ClassificationRow, EventRow, PostRow, TargetHitRow

router = APIRouter(prefix="/analytics", tags=["analytics"])


def _since(days: int) -> datetime:
    return datetime.utcnow() - timedelta(days=days)


def _bot_filter(stmt, exclude_bots: bool, threshold: float = 0.8):
    if exclude_bots:
        stmt = stmt.join(AuthorRow, AuthorRow.id == PostRow.author_id).where((AuthorRow.bot_probability.is_(None)) | (AuthorRow.bot_probability < threshold))
    return stmt


@router.get("/overview")
def overview(days: int = Query(90, ge=1, le=3650), exclude_bots: bool = False, user: CurrentUser = Depends(require("read")), session: Session = Depends(get_session)):
    since = _since(days)
    base = select(ClassificationRow).join(PostRow, PostRow.id == ClassificationRow.post_id).where(PostRow.created_at >= since, PostRow.deleted_upstream.is_(False))
    base = _bot_filter(base, exclude_bots)
    sub = base.subquery()
    total = session.scalar(select(func.count()).select_from(sub)) or 0
    toxic = session.scalar(select(func.count()).select_from(sub).where(sub.c.final_toxicity >= 0.5)) or 0
    by_label = dict(session.execute(select(sub.c.toxicity_label, func.count()).group_by(sub.c.toxicity_label)).all())
    by_sev = dict(session.execute(select(sub.c.severity_level, func.count()).group_by(sub.c.severity_level)).all())
    by_lang = dict(session.execute(select(sub.c.language, func.count()).group_by(sub.c.language).order_by(func.count().desc()).limit(10)).all())
    review = session.scalar(select(func.count()).select_from(sub).where(sub.c.needs_review.is_(True))) or 0
    authors = session.scalar(select(func.count(func.distinct(PostRow.author_id))).where(PostRow.created_at >= since)) or 0
    bot_authors = session.scalar(select(func.count()).select_from(AuthorRow).where(AuthorRow.bot_probability >= 0.8)) or 0
    avg_sev = session.scalar(select(func.avg(sub.c.severity_level)).where(sub.c.final_toxicity >= 0.5)) or 0.0
    by_source = dict(session.execute(select(PostRow.source, func.count()).where(PostRow.created_at >= since, PostRow.deleted_upstream.is_(False)).group_by(PostRow.source)).all())
    return {
        "days": days,
        "posts": total,
        "by_source": {(k or "x"): int(v) for k, v in by_source.items()},
        "toxic_posts": toxic,
        "toxic_ratio": round(toxic / total, 4) if total else 0.0,
        "avg_severity_toxic": round(float(avg_sev), 3),
        "by_label": {k: int(v) for k, v in by_label.items()},
        "by_severity": {int(k): int(v) for k, v in by_sev.items()},
        "by_language": {k: int(v) for k, v in by_lang.items()},
        "needs_review": review,
        "authors": authors,
        "likely_bots": bot_authors,
    }


@router.get("/heatmap")
def heatmap(days: int = Query(90, ge=1, le=3650), exclude_bots: bool = False, min_toxicity: float = 0.5, user: CurrentUser = Depends(require("read")), session: Session = Depends(get_session)):
    """Global toxicity heatmap: target demographic × severity level (+ label breakdown)."""
    since = _since(days)
    stmt = select(TargetHitRow.category, TargetHitRow.label, TargetHitRow.severity_level, func.count(), func.avg(TargetHitRow.final_toxicity)).join(PostRow, PostRow.id == TargetHitRow.post_id).where(PostRow.created_at >= since, TargetHitRow.final_toxicity >= min_toxicity, PostRow.deleted_upstream.is_(False))
    stmt = _bot_filter(stmt, exclude_bots).group_by(TargetHitRow.category, TargetHitRow.label, TargetHitRow.severity_level)
    cells: dict[tuple[str, str], dict] = {}
    for cat, label, sev, n, avg_tox in session.execute(stmt).all():
        cell = cells.setdefault((cat, label), {"category": cat, "label": label, "total": 0, "by_severity": {i: 0 for i in range(5)}, "avg_toxicity": 0.0})
        cell["total"] += int(n)
        cell["by_severity"][int(sev)] += int(n)
        cell["avg_toxicity"] += float(avg_tox or 0) * int(n)
    rows = []
    for cell in cells.values():
        cell["avg_toxicity"] = round(cell["avg_toxicity"] / cell["total"], 3) if cell["total"] else 0.0
        cell["avg_severity"] = round(sum(k * v for k, v in cell["by_severity"].items()) / cell["total"], 3) if cell["total"] else 0.0
        rows.append(cell)
    rows.sort(key=lambda r: r["total"], reverse=True)
    by_category: dict[str, int] = {}
    for r in rows:
        by_category[r["category"]] = by_category.get(r["category"], 0) + r["total"]
    return {"days": days, "cells": rows, "by_category": by_category}


@router.get("/trends")
def trend(days: int = Query(90, ge=7, le=3650), target: str | None = None, freq: str = Query("D", pattern="^(D|H|W)$"), exclude_bots: bool = False, horizon: int = Query(14, ge=0, le=90), user: CurrentUser = Depends(require("read")), session: Session = Depends(get_session)):
    df = trends.toxic_series(session, days=days, target=target, freq=freq, exclude_bots=exclude_bots)
    pts = trends.trend_points(df)
    fc, method = trends.forecast(df, horizon) if horizon and freq == "D" else ([], "none")
    impacts = trends.event_impacts(session, df)
    return {
        "days": days,
        "target": target,
        "freq": freq,
        "series": [p.__dict__ for p in pts],
        "spikes": [p.__dict__ for p in pts if p.spike],
        "forecast": [f.__dict__ for f in fc],
        "forecast_method": method,
        "events": [e.__dict__ for e in impacts],
    }


@router.get("/trends/targets")
def trend_by_target(days: int = Query(90, ge=7, le=3650), top: int = Query(6, ge=1, le=20), user: CurrentUser = Depends(require("read")), session: Session = Depends(get_session)):
    """Daily toxic counts for the top-N targets (small multiples / stacked area)."""
    since = _since(days)
    top_rows = session.execute(select(TargetHitRow.category, TargetHitRow.label, func.count()).join(PostRow, PostRow.id == TargetHitRow.post_id).where(PostRow.created_at >= since, TargetHitRow.final_toxicity >= 0.5).group_by(TargetHitRow.category, TargetHitRow.label).order_by(func.count().desc()).limit(top)).all()
    out = []
    for cat, label, _n in top_rows:
        df = trends.toxic_series(session, days=days, target=f"{cat}:{label}")
        out.append({"target": f"{cat}:{label}", "series": [p.__dict__ for p in trends.trend_points(df)]})
    return {"days": days, "targets": out}


@router.get("/events")
def list_events(days: int = 365, user: CurrentUser = Depends(require("read")), session: Session = Depends(get_session)):
    rows = session.scalars(select(EventRow).where(EventRow.date >= _since(days)).order_by(EventRow.date.desc())).all()
    return [{"id": e.id, "date": e.date.isoformat(), "title": e.title, "category": e.category, "source": e.source, "url": e.url, "related_targets": e.related_targets or [], "magnitude": e.magnitude} for e in rows]


@router.post("/events")
def add_event(body: EventCreate, user: CurrentUser = Depends(require("ingest")), session: Session = Depends(get_session)):
    n = trends.store_events(session, [body.model_dump(mode="json") | {"date": body.date.strftime("%Y-%m-%d")}])
    session.commit()
    return {"created": n}


@router.post("/events/sync")
def sync_events(query: str, days: int = 30, source: str = Query("gdelt", pattern="^(gdelt|newsapi)$"), user: CurrentUser = Depends(require("ingest")), session: Session = Depends(get_session)):
    try:
        evs = trends.fetch_gdelt_events(query, days) if source == "gdelt" else trends.fetch_newsapi_events(query, days)
    except Exception as exc:
        raise HTTPException(502, f"{source} fetch failed: {exc}") from exc
    n = trends.store_events(session, evs)
    session.commit()
    return {"fetched": len(evs), "created": n}


@router.get("/network")
def network_graph(days: int = Query(90, ge=1, le=3650), min_toxicity: float = Query(0.5, ge=0, le=1), max_nodes: int = Query(200, ge=10, le=1000), target: str | None = None, user: CurrentUser = Depends(require("read")), session: Session = Depends(get_session)):
    payload = network.analyze_network(session, days=days, min_toxicity=min_toxicity, max_nodes=max_nodes, target=target)
    return payload.__dict__


@router.get("/severity")
def severity_distribution(days: int = Query(90, ge=1, le=3650), exclude_bots: bool = False, user: CurrentUser = Depends(require("read")), session: Session = Depends(get_session)):
    since = _since(days)
    stmt = select(ClassificationRow.severity_level, ClassificationRow.toxicity_label, func.count()).join(PostRow, PostRow.id == ClassificationRow.post_id).where(PostRow.created_at >= since, PostRow.deleted_upstream.is_(False))
    stmt = _bot_filter(stmt, exclude_bots).group_by(ClassificationRow.severity_level, ClassificationRow.toxicity_label)
    matrix: dict[int, dict[str, int]] = {i: {} for i in range(5)}
    for sev, lab, n in session.execute(stmt).all():
        matrix[int(sev)][lab] = int(n)
    # per-target average severity
    per_target = session.execute(select(TargetHitRow.category, TargetHitRow.label, func.avg(TargetHitRow.severity_level), func.count()).join(PostRow, PostRow.id == TargetHitRow.post_id).where(PostRow.created_at >= since, TargetHitRow.final_toxicity >= 0.5).group_by(TargetHitRow.category, TargetHitRow.label).having(func.count() >= 3).order_by(func.avg(TargetHitRow.severity_level).desc()).limit(15)).all()
    return {"days": days, "matrix": matrix, "per_target": [{"target": f"{c}:{l}", "avg_severity": round(float(s), 3), "n": int(n)} for c, l, s, n in per_target]}


@router.get("/bots")
def bot_breakdown(days: int = Query(90, ge=1, le=3650), threshold: float = Query(0.8, ge=0, le=1), limit: int = Query(25, ge=1, le=200), user: CurrentUser = Depends(require("read")), session: Session = Depends(get_session)):
    since = _since(days)
    scored = session.scalar(select(func.count()).select_from(AuthorRow).where(AuthorRow.bot_probability.is_not(None))) or 0
    likely = session.scalar(select(func.count()).select_from(AuthorRow).where(AuthorRow.bot_probability >= threshold)) or 0
    # toxic share by bot vs human
    is_bot = case((AuthorRow.bot_probability >= threshold, "bot"), else_="human")
    rows = session.execute(select(is_bot, func.count(), func.sum(case((ClassificationRow.final_toxicity >= 0.5, 1), else_=0))).join(PostRow, PostRow.author_id == AuthorRow.id).join(ClassificationRow, ClassificationRow.post_id == PostRow.id).where(PostRow.created_at >= since).group_by(is_bot)).all()
    breakdown = {k: {"posts": int(n), "toxic_posts": int(t or 0), "toxic_ratio": round((t or 0) / n, 4) if n else 0.0} for k, n, t in rows}
    hist_bins = [0] * 10
    for p in session.scalars(select(AuthorRow.bot_probability).where(AuthorRow.bot_probability.is_not(None))).all():
        hist_bins[min(9, int(p * 10))] += 1
    top = session.scalars(select(AuthorRow).where(AuthorRow.bot_probability.is_not(None)).order_by(AuthorRow.bot_probability.desc()).limit(limit)).all()
    return {
        "days": days,
        "threshold": threshold,
        "scored_accounts": scored,
        "likely_bots": likely,
        "breakdown": breakdown,
        "histogram": hist_bins,
        "top_accounts": [{"author_id": a.id, "username": a.username, "bot_probability": a.bot_probability, "features": a.bot_features, "followers": a.followers_count, "following": a.following_count, "account_score": a.account_score} for a in top],
        "model": bots.score_features(top[0].bot_features)[1] if top and top[0].bot_features else "heuristic-logistic",
    }


@router.post("/bots/rescore")
def rescore_bots(user: CurrentUser = Depends(require("ingest")), session: Session = Depends(get_session)):
    n = bots.score_all_authors(session)
    session.commit()
    return {"scored": n}


@router.get("/accounts")
def account_scores(limit: int = Query(50, ge=1, le=500), reliable_only: bool = False, user: CurrentUser = Depends(require("read")), session: Session = Depends(get_session)):
    stmt = select(AuthorRow).where(AuthorRow.account_score_json.is_not(None))
    if reliable_only:
        stmt = stmt.where(AuthorRow.account_score.is_not(None))
    rows = session.scalars(stmt).all()
    items = [a.account_score_json for a in rows]
    items.sort(key=lambda s: (s.get("score") or 0.0), reverse=True)
    return {"disclaimer": acc.DISCLAIMER, "weights": acc.WEIGHTS, "items": items[:limit]}


@router.get("/accounts/{author_id}")
def account_score(author_id: str, user: CurrentUser = Depends(require("read")), session: Session = Depends(get_session)):
    r = acc.score_account(session, author_id)
    if r is None:
        raise HTTPException(404, "author not found")
    bot = bots.score_author(session, author_id)
    session.commit()
    recent = session.execute(select(PostRow, ClassificationRow).join(ClassificationRow, ClassificationRow.post_id == PostRow.id).where(PostRow.author_id == author_id).order_by(PostRow.created_at.desc()).limit(20)).all()
    return {
        "score": r.model_dump(mode="json"),
        "confidence": acc.confidence_label(r),
        "bot": bot.model_dump(mode="json") if bot else None,
        "recent_posts": [{"id": p.id, "text": p.text, "created_at": p.created_at.isoformat(), "final_toxicity": c.final_toxicity, "severity_level": c.severity_level, "targets": c.target_labels or []} for p, c in recent],
    }


@router.post("/accounts/rescore")
def rescore_accounts(user: CurrentUser = Depends(require("ingest")), session: Session = Depends(get_session)):
    n = acc.score_all_accounts(session)
    session.commit()
    return {"scored": n}


@router.get("/fairness")
def fairness_report(user: CurrentUser = Depends(require("audit")), session: Session = Depends(get_session)):
    from sentinai.classification.engine import get_engine
    from sentinai.storage.models import FairnessAuditRow

    engine = get_engine()
    reports = []
    for path in fairness.bundled_benchmarks():
        rows = fairness.load_benchmark(path)
        reports.append(fairness.evaluate(engine.score_text, rows, benchmark=path.stem).to_dict())
    history = session.scalars(select(FairnessAuditRow).order_by(FairnessAuditRow.created_at.desc()).limit(12)).all()
    return {"benchmarks": reports, "audits": [{"id": h.id, "created_at": h.created_at.isoformat(), "model_version": h.model_version, "report": h.report} for h in history]}


@router.post("/fairness/audit")
def run_fairness_audit(days: int = 7, user: CurrentUser = Depends(require("audit")), session: Session = Depends(get_session)):
    from sentinai.classification.engine import get_engine

    rep = fairness.weekly_audit(session, get_engine().model_version, days=days)
    session.commit()
    return rep.to_dict()
