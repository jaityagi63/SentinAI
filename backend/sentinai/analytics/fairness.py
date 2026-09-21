"""Module 11 — Dialect & vernacular bias mitigation.

Components implemented here:

* **Dialect tagging** — lightweight detector for speech-pattern groups whose benign usage is
  known to trigger false positives in toxicity models: AAVE, Chicano/Spanglish, Hinglish,
  Arabizi, British/Irish slang and Standard American English as reference group.  In the
  transformer backend this is replaced by a fine-tuned dialect classifier; the tags are only
  used for *auditing and calibration*, never as model inputs.
* **Fairness benchmarking** — runner that evaluates any ``score_fn(text) -> toxicity`` against
  BOLD / ToxiGen / dialect benchmark JSONL files (``benchmarks/*.jsonl`` with
  ``{"text", "label", "group"}``) and reports per-group false-positive rate (FPR), false
  negative rate, and equalised-odds gaps.  A small bundled benchmark
  (``resources/benchmarks/dialect_smoke.jsonl``) keeps CI meaningful without the large corpora.
* **Group-specific threshold calibration** — for each group, find the decision threshold that
  equalises FPR to the reference group (or to a target FPR) and persist as a calibration table
  applied by :func:`calibrated_decision`.
* **Adversarial debiasing** — training recipe (gradient-reversal adversary predicting the
  dialect tag from the toxicity head's pooled representation) lives in
  ``scripts/train_toxicity.py --adversarial``; this module supplies the dialect labels it needs.
* **Weekly FP audit** — :func:`weekly_audit` recomputes the per-group FPR on human-annotated
  posts (Module 14 corrections) and stores a :class:`FairnessAuditRow` snapshot.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from sentinai.config import RESOURCES_DIR
from sentinai.storage.models import AnnotationRow, ClassificationRow, FairnessAuditRow, PostRow

BENCHMARK_DIR = RESOURCES_DIR / "benchmarks"

DIALECT_GROUPS = ("sae", "aave", "chicano", "hinglish", "arabizi", "british", "other")

_AAVE_RE = re.compile(
    r"\b(finna|fina|gon|gonna be|tryna|ain'?t|y'?all|yall|bruh|fam|lit|deadass|no cap|on god|frfr|fr fr|ion|iont|"
    r"he be|she be|they be|we be|it be|be like|been (done|had|knew)|done (did|told|said)|ain'?t nobody|ain'?t no|"
    r"finna be|whole time|periodt|say less|it'?s giving|lowkey|highkey|bet|sis|chile|whew|talm bout|talmbout|"
    r"nah|naw|yo|aight|ight|wit|dat|dis|dem|dey|nuthin|nothin|somethin|sumn|cuz|cause|ova|fo|tho|doe|"
    r"hella|mad (funny|good|cool)|type shit|real one|my nigga|nigga|niggas)\b",
    re.I,
)
_CHICANO_RE = re.compile(
    r"\b(órale|orale|ese|vato|homes|carnal|carnala|mijo|mija|güey|guey|wey|no manches|simón|simon|chale|"
    r"ándale|andale|pues|entonces|pero|que onda|qué onda|la neta|neta|chido|firme|ranfla|jefita|jefito|"
    r"nombre|hijole|híjole|ay dios|pinche|a huevo|ahuevo|con safos|c/s|la raza|paisa|troca|parkear|lonche|"
    r"washateria|marketa|yonke|birria|carnitas|abuelita|tía|tio|primo|prima|comadre|compadre)\b",
    re.I,
)
_HINGLISH_RE = re.compile(
    r"\b(yaar|yar|bhai|bhaiya|didi|beta|arre|arrey|accha|acha|theek|thik|hai|hain|nahi|nahin|kya|kyun|kaise|"
    r"matlab|bilkul|bahut|bohot|ekdum|mast|jugaad|timepass|chalo|chal|dekho|suno|batao|bolo|karo|karna|"
    r"hona|hoga|hogi|wala|wali|wale|abhi|kal|aaj|sab|log|logon|paisa|paise|khana|chai|ji|haan|na|toh|to)\b",
    re.I,
)
_ARABIZI_RE = re.compile(r"\b(\w*[2357]\w+|yalla|habibi|7abibi|wallah|walla|inshallah|inshalla|mashallah|khalas|5alas|yani|ya3ni|akhi|a5i|shu|kifak|keefak|ezayak|3amel eh|tamam|mesh|mish|3ala|3ashan|bas|kteer|ktir)\b", re.I)
_BRITISH_RE = re.compile(r"\b(mate|innit|bloody|bloke|cheers|knackered|gutted|chuffed|dodgy|quid|mum|lads|lass|proper|well (good|bad|fit)|peng|bare|wagwan|fam|bruv|blud|mandem|gyaldem|allow it|wasteman|bait|long|calm|safe|ting|ends|yard|peak|bants|banter|cuppa|loo|bog|arse|bollocks|bugger|wanker|tosser|git|muppet|numpty|pillock|plonker|twat|prat)\b", re.I)


def tag_dialect(text: str) -> str:
    t = text.lower()
    scores = {
        "aave": len(_AAVE_RE.findall(t)),
        "chicano": len(_CHICANO_RE.findall(t)),
        "hinglish": len(_HINGLISH_RE.findall(t)),
        "arabizi": len(_ARABIZI_RE.findall(t)),
        "british": len(_BRITISH_RE.findall(t)),
    }
    n_tokens = max(1, len(t.split()))
    best = max(scores, key=scores.get)  # type: ignore[arg-type]
    if scores[best] == 0 or scores[best] / n_tokens < 0.08:
        return "sae" if re.search(r"[a-z]", t) else "other"
    return best


# --------------------------------------------------------------------------------------------
# benchmark evaluation
# --------------------------------------------------------------------------------------------


@dataclass
class GroupMetrics:
    group: str
    n: int
    n_positive: int
    n_negative: int
    fpr: float
    fnr: float
    tpr: float
    precision: float
    mean_score_negative: float


@dataclass
class FairnessReport:
    benchmark: str
    threshold: float
    groups: list[GroupMetrics]
    fpr_gap: float  # max - min FPR across groups
    fnr_gap: float
    reference_group: str
    calibrated_thresholds: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "benchmark": self.benchmark,
            "threshold": self.threshold,
            "reference_group": self.reference_group,
            "fpr_gap": self.fpr_gap,
            "fnr_gap": self.fnr_gap,
            "groups": [g.__dict__ for g in self.groups],
            "calibrated_thresholds": self.calibrated_thresholds,
            "notes": self.notes,
        }


def load_benchmark(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _metrics(group: str, items: list[tuple[float, int]], threshold: float) -> GroupMetrics:
    pos = [s for s, y in items if y == 1]
    neg = [s for s, y in items if y == 0]
    fp = sum(1 for s in neg if s >= threshold)
    tp = sum(1 for s in pos if s >= threshold)
    fn = len(pos) - tp
    fpr = fp / len(neg) if neg else 0.0
    tpr = tp / len(pos) if pos else 0.0
    fnr = fn / len(pos) if pos else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    return GroupMetrics(group, len(items), len(pos), len(neg), round(fpr, 4), round(fnr, 4), round(tpr, 4), round(prec, 4), round(sum(neg) / len(neg), 4) if neg else 0.0)


def evaluate(score_fn: Callable[[str], float], rows: list[dict], *, threshold: float = 0.5, reference_group: str = "sae", benchmark: str = "custom", target_fpr: float | None = None) -> FairnessReport:
    by_group: dict[str, list[tuple[float, int]]] = {}
    for r in rows:
        g = r.get("group") or tag_dialect(r["text"])
        by_group.setdefault(g, []).append((float(score_fn(r["text"])), int(r["label"])))
    groups = [_metrics(g, items, threshold) for g, items in sorted(by_group.items())]
    fprs = [g.fpr for g in groups if g.n_negative >= 5]
    fnrs = [g.fnr for g in groups if g.n_positive >= 5]
    ref = next((g for g in groups if g.group == reference_group), None)
    tgt = target_fpr if target_fpr is not None else (ref.fpr if ref else min(fprs) if fprs else 0.05)
    calibrated = {g.group: calibrate_threshold(by_group[g.group], tgt, default=threshold) for g in groups}
    notes = []
    for g in groups:
        if ref and g.group != ref.group and g.n_negative >= 5 and g.fpr > ref.fpr + 0.05:
            notes.append(f"{g.group}: FPR {g.fpr:.2%} exceeds reference {ref.fpr:.2%} — group threshold raised to {calibrated[g.group]:.2f}")
    return FairnessReport(
        benchmark=benchmark,
        threshold=threshold,
        groups=groups,
        fpr_gap=round(max(fprs) - min(fprs), 4) if fprs else 0.0,
        fnr_gap=round(max(fnrs) - min(fnrs), 4) if fnrs else 0.0,
        reference_group=reference_group,
        calibrated_thresholds=calibrated,
        notes=notes,
    )


def calibrate_threshold(items: list[tuple[float, int]], target_fpr: float, default: float = 0.5) -> float:
    """Group-specific decision threshold that equalises the false-positive rate.

    The default threshold is kept whenever it already satisfies the FPR target (thresholds only
    move when a group is measurably over-flagged); otherwise the threshold is raised to the
    smallest value whose FPR on the group's benign examples is ≤ target (capped at 0.95).
    """
    neg = sorted((s for s, y in items if y == 0), reverse=True)
    if len(neg) < 5:
        return default

    def fpr_at(th: float) -> float:
        return sum(1 for s in neg if s >= th) / len(neg)

    if fpr_at(default) <= target_fpr + 1e-9:
        return default
    for th in [round(x / 100, 2) for x in range(int(default * 100) + 1, 96)]:
        if fpr_at(th) <= target_fpr + 1e-9:
            return th
    return 0.95


class ThresholdTable:
    """Group-specific decision thresholds (Module 11 calibration output)."""

    def __init__(self, table: dict[str, float] | None = None, default: float = 0.5):
        self.table = table or {}
        self.default = default

    def threshold_for(self, text: str) -> tuple[str, float]:
        g = tag_dialect(text)
        return g, self.table.get(g, self.default)

    def decision(self, text: str, score: float) -> bool:
        return score >= self.threshold_for(text)[1]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"default": self.default, "table": self.table}, indent=2))

    @classmethod
    def load(cls, path: Path) -> ThresholdTable:
        if not path.exists():
            return cls()
        data = json.loads(path.read_text())
        return cls(data.get("table", {}), data.get("default", 0.5))


def calibrated_decision(text: str, score: float, table: ThresholdTable) -> bool:
    return table.decision(text, score)


# --------------------------------------------------------------------------------------------
# weekly audit on human-labelled data
# --------------------------------------------------------------------------------------------


def weekly_audit(session: Session, model_version: str, days: int = 7, threshold: float = 0.5) -> FairnessReport:
    since = datetime.utcnow() - timedelta(days=days)
    rows = session.execute(
        select(PostRow.text, AnnotationRow.toxicity_label, ClassificationRow.final_toxicity)
        .join(AnnotationRow, AnnotationRow.post_id == PostRow.id)
        .join(ClassificationRow, ClassificationRow.post_id == PostRow.id)
        .where(AnnotationRow.created_at >= since)
    ).all()
    bench = [{"text": t, "label": 0 if lab == "non_toxic" else 1, "group": tag_dialect(t), "_score": s} for t, lab, s in rows]
    scores = {r["text"]: r["_score"] for r in bench}
    report = evaluate(lambda t: scores[t], bench, threshold=threshold, benchmark=f"weekly-audit-{days}d")
    session.add(FairnessAuditRow(model_version=model_version, report=report.to_dict()))
    return report


def bundled_benchmarks() -> list[Path]:
    return sorted(BENCHMARK_DIR.glob("*.jsonl")) if BENCHMARK_DIR.exists() else []
