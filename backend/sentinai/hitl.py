"""Module 14 — Human-in-the-loop feedback pipeline.

* **Active learning queue** — posts whose final toxicity lies in the uncertainty band
  (0.4–0.6 by default) are queued automatically by :func:`save_classification`; the queue can
  also be topped up with disagreement / high-severity samples via :func:`enqueue_candidates`.
* **Annotation** — reviewers submit labels (toxicity, severity, targets) through the API or
  via Label Studio / Prodigy using the exporters/importers below (Label Studio JSON task format
  and Prodigy JSONL).
* **Inter-annotator agreement** — Cohen's κ per annotator pair and Fleiss' κ across all
  annotators for multiply-annotated posts.
* **Feedback loop** — :func:`build_retrain_batch` exports the accumulated corrections (biweekly
  schedule via the CLI / cron) as a JSONL fine-tuning set and marks them as used.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from itertools import combinations
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from sentinai.config import get_settings
from sentinai.schemas import Annotation
from sentinai.storage.models import AnnotationRow, ClassificationRow, PostRow, RetrainBatchRow, ReviewQueueRow

# --------------------------------------------------------------------------------------------
# queue
# --------------------------------------------------------------------------------------------


def enqueue_candidates(session: Session, *, limit: int = 200, include_high_severity: bool = True) -> int:
    settings = get_settings()
    lo, hi = settings.active_learning_low, settings.active_learning_high
    queued = set(session.scalars(select(ReviewQueueRow.post_id)).all())
    q = select(ClassificationRow).where(ClassificationRow.final_toxicity.between(lo, hi))
    n = 0
    for c in session.scalars(q.limit(limit)).all():
        if c.post_id in queued:
            continue
        session.add(ReviewQueueRow(post_id=c.post_id, model_label=c.toxicity_label, model_confidence=c.toxicity_confidence, reason="active_learning", priority=round(1 - abs(c.final_toxicity - 0.5) * 2, 4)))
        queued.add(c.post_id)
        n += 1
    if include_high_severity:
        # Level-4 (incitement) predictions are always double-checked by a human.
        for c in session.scalars(select(ClassificationRow).where(ClassificationRow.severity_level >= 4).limit(limit)).all():
            if c.post_id in queued:
                continue
            session.add(ReviewQueueRow(post_id=c.post_id, model_label=c.toxicity_label, model_confidence=c.toxicity_confidence, reason="high_severity", priority=1.0))
            queued.add(c.post_id)
            n += 1
    return n


def next_items(session: Session, *, reviewer: str | None = None, limit: int = 20, status: str = "pending") -> list[dict]:
    q = select(ReviewQueueRow, PostRow, ClassificationRow).join(PostRow, PostRow.id == ReviewQueueRow.post_id).join(ClassificationRow, ClassificationRow.post_id == ReviewQueueRow.post_id)
    if status:
        q = q.where(ReviewQueueRow.status == status)
    if reviewer:
        q = q.where((ReviewQueueRow.assigned_to.is_(None)) | (ReviewQueueRow.assigned_to == reviewer))
    q = q.order_by(ReviewQueueRow.priority.desc(), ReviewQueueRow.created_at.asc()).limit(limit)
    out = []
    for item, post, cls in session.execute(q).all():
        out.append(
            {
                "id": item.id,
                "post_id": post.id,
                "text": post.text,
                "created_at": post.created_at.isoformat(),
                "model_label": item.model_label,
                "model_confidence": item.model_confidence,
                "final_toxicity": cls.final_toxicity,
                "severity_level": cls.severity_level,
                "targets": cls.target_labels or [],
                "reason": item.reason,
                "priority": item.priority,
                "status": item.status,
                "assigned_to": item.assigned_to,
                "explanation": (cls.payload or {}).get("explanation"),
            }
        )
    return out


def submit_annotation(session: Session, ann: Annotation) -> AnnotationRow:
    cls = session.get(ClassificationRow, ann.post_id)
    row = session.scalar(select(AnnotationRow).where(AnnotationRow.post_id == ann.post_id, AnnotationRow.annotator == ann.annotator))
    if row is None:
        row = AnnotationRow(post_id=ann.post_id, annotator=ann.annotator)
        session.add(row)
    row.toxicity_label = ann.toxicity_label.value
    row.severity = int(ann.severity) if ann.severity is not None else None
    row.targets = ann.targets
    row.notes = ann.notes
    row.agrees_with_model = (cls.toxicity_label == ann.toxicity_label.value) if cls else None
    row.created_at = datetime.utcnow()
    q = session.scalar(select(ReviewQueueRow).where(ReviewQueueRow.post_id == ann.post_id))
    if q is not None:
        q.status = "resolved"
        q.resolved_at = datetime.utcnow()
        q.assigned_to = q.assigned_to or ann.annotator
    return row


def skip_item(session: Session, item_id: int, reviewer: str) -> bool:
    q = session.get(ReviewQueueRow, item_id)
    if q is None:
        return False
    q.status = "skipped"
    q.assigned_to = reviewer
    q.resolved_at = datetime.utcnow()
    return True


def queue_stats(session: Session) -> dict:
    rows = session.execute(select(ReviewQueueRow.status, func.count()).group_by(ReviewQueueRow.status)).all()
    by_status = {s: int(n) for s, n in rows}
    total_ann = session.scalar(select(func.count()).select_from(AnnotationRow)) or 0
    agree = session.scalar(select(func.count()).select_from(AnnotationRow).where(AnnotationRow.agrees_with_model.is_(True))) or 0
    unused = session.scalar(select(func.count()).select_from(AnnotationRow).where(AnnotationRow.used_in_training.is_(False))) or 0
    return {
        "pending": by_status.get("pending", 0),
        "in_review": by_status.get("in_review", 0),
        "resolved": by_status.get("resolved", 0),
        "skipped": by_status.get("skipped", 0),
        "annotations": int(total_ann),
        "model_agreement_rate": round(agree / total_ann, 4) if total_ann else None,
        "pending_for_retraining": int(unused),
        "band": [get_settings().active_learning_low, get_settings().active_learning_high],
    }


# --------------------------------------------------------------------------------------------
# inter-annotator agreement
# --------------------------------------------------------------------------------------------


def cohen_kappa(a: list[str], b: list[str]) -> float | None:
    if len(a) != len(b) or not a:
        return None
    n = len(a)
    po = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb.get(k, 0) for k in ca) / (n * n)
    if pe == 1.0:
        return 1.0
    return round((po - pe) / (1 - pe), 4)


def fleiss_kappa(ratings: list[list[str]]) -> float | None:
    """``ratings``: per item, list of category labels from each rater (equal count per item)."""
    items = [r for r in ratings if len(r) >= 2]
    if not items:
        return None
    n = min(len(r) for r in items)
    items = [r[:n] for r in items]
    cats = sorted({c for r in items for c in r})
    N = len(items)
    p_j = {c: sum(r.count(c) for r in items) / (N * n) for c in cats}
    P_i = [(sum(r.count(c) ** 2 for c in cats) - n) / (n * (n - 1)) for r in items]
    P_bar = sum(P_i) / N
    P_e = sum(p ** 2 for p in p_j.values())
    if P_e == 1.0:
        return 1.0
    return round((P_bar - P_e) / (1 - P_e), 4)


def agreement_report(session: Session, days: int = 90) -> dict:
    since = datetime.utcnow() - timedelta(days=days)
    rows = session.execute(select(AnnotationRow.post_id, AnnotationRow.annotator, AnnotationRow.toxicity_label).where(AnnotationRow.created_at >= since)).all()
    by_post: dict[str, dict[str, str]] = defaultdict(dict)
    for pid, ann, lab in rows:
        by_post[pid][ann] = lab
    annotators = sorted({ann for _, ann, _ in rows})
    pairwise = []
    for a, b in combinations(annotators, 2):
        xs, ys = [], []
        for labels in by_post.values():
            if a in labels and b in labels:
                xs.append(labels[a])
                ys.append(labels[b])
        if len(xs) >= 5:
            pairwise.append({"a": a, "b": b, "n": len(xs), "kappa": cohen_kappa(xs, ys)})
    multi = [list(v.values()) for v in by_post.values() if len(v) >= 2]
    return {
        "annotators": annotators,
        "posts_with_multiple_annotations": len(multi),
        "pairwise_cohen_kappa": pairwise,
        "fleiss_kappa": fleiss_kappa(multi),
        "interpretation": _interpret(fleiss_kappa(multi)),
    }


def _interpret(k: float | None) -> str:
    if k is None:
        return "insufficient overlap"
    if k < 0.2:
        return "slight"
    if k < 0.4:
        return "fair"
    if k < 0.6:
        return "moderate"
    if k < 0.8:
        return "substantial"
    return "almost perfect"


# --------------------------------------------------------------------------------------------
# Label Studio / Prodigy interop
# --------------------------------------------------------------------------------------------

LABEL_STUDIO_CONFIG = """<View>
  <Header value="SentinAI review"/>
  <Text name="text" value="$text"/>
  <Choices name="toxicity" toName="text" choice="single" required="true">
    <Choice value="non_toxic"/><Choice value="offensive"/><Choice value="hate_speech"/><Choice value="violent_extremism"/>
  </Choices>
  <Choices name="severity" toName="text" choice="single">
    <Choice value="0"/><Choice value="1"/><Choice value="2"/><Choice value="3"/><Choice value="4"/>
  </Choices>
  <TextArea name="targets" toName="text" placeholder="religion:islam, ethnicity:black" />
</View>"""


def export_label_studio(session: Session, limit: int = 500) -> list[dict]:
    tasks = []
    for item in next_items(session, limit=limit):
        tasks.append({"id": item["id"], "data": {"text": item["text"], "post_id": item["post_id"], "model_label": item["model_label"], "model_confidence": item["model_confidence"], "targets": item["targets"]}, "predictions": [{"model_version": "sentinai", "result": [{"from_name": "toxicity", "to_name": "text", "type": "choices", "value": {"choices": [item["model_label"]]}}]}]})
    return tasks


def import_label_studio(session: Session, export: list[dict]) -> int:
    from sentinai.schemas import SeverityLevel, ToxicityLabel

    n = 0
    for task in export:
        pid = task.get("data", {}).get("post_id")
        for ann in task.get("annotations", []):
            annotator = str(ann.get("completed_by", "label-studio"))
            tox = sev = None
            targets: list[str] = []
            for r in ann.get("result", []):
                if r.get("from_name") == "toxicity":
                    tox = r["value"]["choices"][0]
                elif r.get("from_name") == "severity":
                    sev = int(r["value"]["choices"][0])
                elif r.get("from_name") == "targets":
                    targets = [t.strip() for t in ",".join(r["value"].get("text", [])).split(",") if t.strip()]
            if pid and tox:
                submit_annotation(session, Annotation(post_id=pid, annotator=annotator, toxicity_label=ToxicityLabel(tox), severity=SeverityLevel(sev) if sev is not None else None, targets=targets))
                n += 1
    return n


def export_prodigy(session: Session, limit: int = 500) -> list[dict]:
    return [{"text": it["text"], "meta": {"post_id": it["post_id"], "model_label": it["model_label"], "score": it["final_toxicity"]}, "options": [{"id": l, "text": l} for l in ("non_toxic", "offensive", "hate_speech", "violent_extremism")]} for it in next_items(session, limit=limit)]  # noqa: E741


# --------------------------------------------------------------------------------------------
# retraining batches
# --------------------------------------------------------------------------------------------


def build_retrain_batch(session: Session, out_dir: Path | None = None, min_examples: int = 20) -> RetrainBatchRow | None:
    out_dir = out_dir or (get_settings().data_dir / "retrain")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = session.execute(
        select(AnnotationRow, PostRow.text, ClassificationRow.toxicity_label, ClassificationRow.severity_level, ClassificationRow.target_labels)
        .join(PostRow, PostRow.id == AnnotationRow.post_id)
        .join(ClassificationRow, ClassificationRow.post_id == AnnotationRow.post_id)
        .where(AnnotationRow.used_in_training.is_(False))
    ).all()
    if len(rows) < min_examples:
        return None
    # majority vote per post
    by_post: dict[str, list] = defaultdict(list)
    for ann, text, mlabel, msev, mtargets in rows:
        by_post[ann.post_id].append((ann, text, mlabel, msev, mtargets))
    path = out_dir / f"batch_{datetime.utcnow():%Y%m%d_%H%M%S}.jsonl"
    n_corr = 0
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for pid, anns in by_post.items():
            labels = Counter(a.toxicity_label for a, *_ in anns)
            label, _ = labels.most_common(1)[0]
            sevs = [a.severity for a, *_ in anns if a.severity is not None]
            targets = sorted({t for a, *_ in anns for t in (a.targets or [])})
            _, text, mlabel, _, _ = anns[0]
            corrected = label != mlabel
            n_corr += int(corrected)
            fh.write(json.dumps({"post_id": pid, "text": text, "label": label, "severity": round(sum(sevs) / len(sevs)) if sevs else None, "targets": targets, "model_label": mlabel, "corrected": corrected, "n_annotators": len(anns)}, ensure_ascii=False) + "\n")
            n += 1
            for a, *_ in anns:
                a.used_in_training = True
    batch = RetrainBatchRow(n_examples=n, n_corrections=n_corr, export_path=str(path), status="exported", notes="fine-tune with: python scripts/train_toxicity.py --data " + str(path))
    session.add(batch)
    session.flush()
    return batch
