"""HITL review queue, annotations, agreement metrics, retraining batches (Module 14) + exports."""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response, StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from sentinai import hitl
from sentinai.api.schemas import AnnotationRequest
from sentinai.auth import CurrentUser, require
from sentinai.schemas import Annotation
from sentinai.storage.db import get_session
from sentinai.storage.models import AuthorRow, ClassificationRow, PostRow, RetrainBatchRow

router = APIRouter(tags=["review"])


@router.get("/review/queue")
def review_queue(limit: int = Query(20, ge=1, le=200), status: str = "pending", user: CurrentUser = Depends(require("review")), session: Session = Depends(get_session)):
    return {"items": hitl.next_items(session, reviewer=user.username, limit=limit, status=status), "stats": hitl.queue_stats(session)}


@router.get("/review/stats")
def review_stats(user: CurrentUser = Depends(require("read")), session: Session = Depends(get_session)):
    return hitl.queue_stats(session)


@router.post("/review/enqueue")
def enqueue(limit: int = 200, user: CurrentUser = Depends(require("review")), session: Session = Depends(get_session)):
    n = hitl.enqueue_candidates(session, limit=limit)
    session.commit()
    return {"queued": n}


@router.post("/review/annotate")
def annotate(body: AnnotationRequest, user: CurrentUser = Depends(require("annotate")), session: Session = Depends(get_session)):
    if session.get(PostRow, body.post_id) is None:
        raise HTTPException(404, "post not found")
    row = hitl.submit_annotation(session, Annotation(post_id=body.post_id, annotator=user.username, toxicity_label=body.toxicity_label, severity=body.severity, targets=body.targets, notes=body.notes))
    session.commit()
    return {"id": row.id, "agrees_with_model": row.agrees_with_model}


@router.post("/review/{item_id}/skip")
def skip(item_id: int, user: CurrentUser = Depends(require("review")), session: Session = Depends(get_session)):
    ok = hitl.skip_item(session, item_id, user.username)
    session.commit()
    if not ok:
        raise HTTPException(404, "queue item not found")
    return {"skipped": item_id}


@router.get("/review/agreement")
def agreement(days: int = 90, user: CurrentUser = Depends(require("read")), session: Session = Depends(get_session)):
    return hitl.agreement_report(session, days)


@router.get("/review/export/label-studio")
def export_label_studio(limit: int = 500, user: CurrentUser = Depends(require("review")), session: Session = Depends(get_session)):
    return {"label_config": hitl.LABEL_STUDIO_CONFIG, "tasks": hitl.export_label_studio(session, limit)}


@router.post("/review/import/label-studio")
def import_label_studio(export: list[dict], user: CurrentUser = Depends(require("annotate")), session: Session = Depends(get_session)):
    n = hitl.import_label_studio(session, export)
    session.commit()
    return {"imported": n}


@router.get("/review/export/prodigy")
def export_prodigy(limit: int = 500, user: CurrentUser = Depends(require("review")), session: Session = Depends(get_session)):
    lines = "\n".join(json.dumps(x, ensure_ascii=False) for x in hitl.export_prodigy(session, limit))
    return Response(lines, media_type="application/x-ndjson")


@router.post("/review/retrain-batch")
def retrain_batch(min_examples: int = 20, user: CurrentUser = Depends(require("retrain")), session: Session = Depends(get_session)):
    batch = hitl.build_retrain_batch(session, min_examples=min_examples)
    session.commit()
    if batch is None:
        return {"created": False, "reason": f"fewer than {min_examples} unused annotations"}
    return {"created": True, "id": batch.id, "n_examples": batch.n_examples, "n_corrections": batch.n_corrections, "export_path": batch.export_path}


@router.get("/review/retrain-batches")
def retrain_batches(user: CurrentUser = Depends(require("read")), session: Session = Depends(get_session)):
    rows = session.scalars(select(RetrainBatchRow).order_by(RetrainBatchRow.created_at.desc()).limit(20)).all()
    return [{"id": b.id, "created_at": b.created_at.isoformat(), "n_examples": b.n_examples, "n_corrections": b.n_corrections, "status": b.status, "export_path": b.export_path} for b in rows]


# --------------------------------------------------------------------------------------------
# exports (CSV / JSON / PDF)
# --------------------------------------------------------------------------------------------


def _export_rows(session: Session, days: int, min_toxicity: float, limit: int):
    since = datetime.utcnow() - timedelta(days=days)
    rows = session.execute(
        select(PostRow, ClassificationRow, AuthorRow).join(ClassificationRow, ClassificationRow.post_id == PostRow.id).join(AuthorRow, AuthorRow.id == PostRow.author_id).where(PostRow.created_at >= since, ClassificationRow.final_toxicity >= min_toxicity, PostRow.deleted_upstream.is_(False)).order_by(PostRow.created_at.desc()).limit(limit)
    ).all()
    for p, c, a in rows:
        yield {
            "post_id": p.id,
            "created_at": p.created_at.isoformat(),
            "author_id": p.author_id,
            "author_username": a.username,
            "bot_probability": a.bot_probability,
            "language": c.language,
            "text": p.text,
            "toxicity_label": c.toxicity_label,
            "toxicity_confidence": c.toxicity_confidence,
            "final_toxicity": c.final_toxicity,
            "severity_level": c.severity_level,
            "targets": ";".join(c.target_labels or []),
            "stance": c.stance,
            "discount_factor": c.discount_factor,
            "needs_review": c.needs_review,
            "model_version": c.model_version,
        }


@router.get("/export/posts.csv")
def export_csv(days: int = 90, min_toxicity: float = 0.0, limit: int = Query(5000, le=50000), user: CurrentUser = Depends(require("export")), session: Session = Depends(get_session)):
    buf = io.StringIO()
    writer = None
    for row in _export_rows(session, days, min_toxicity, limit):
        if writer is None:
            writer = csv.DictWriter(buf, fieldnames=list(row.keys()))
            writer.writeheader()
        writer.writerow(row)
    if writer is None:
        buf.write("post_id\n")
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=sentinai_posts.csv"})


@router.get("/export/posts.json")
def export_json(days: int = 90, min_toxicity: float = 0.0, limit: int = Query(5000, le=50000), user: CurrentUser = Depends(require("export")), session: Session = Depends(get_session)):
    return {"disclaimer": "Model-Estimated Probabilities — not definitive human judgments.", "items": list(_export_rows(session, days, min_toxicity, limit))}


@router.get("/export/report.pdf")
def export_pdf(days: int = 90, user: CurrentUser = Depends(require("export")), session: Session = Depends(get_session)):
    from sentinai.reporting import build_pdf_report

    pdf = build_pdf_report(session, days=days)
    return Response(pdf, media_type="application/pdf", headers={"Content-Disposition": f"attachment; filename=sentinai_report_{datetime.utcnow():%Y%m%d}.pdf"})
