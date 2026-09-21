"""Transformer classification backend (requires the ``[ml]`` extra).

Architecture (Module 3 / Module 7):

* **Task A** — sequence classifier on ``microsoft/deberta-v3-large`` (or ``GroNLP/hateBERT``)
  with 4 labels: non_toxic / offensive / hate_speech / violent_extremism.
* **Task B** — multi-label sigmoid head on ``xlm-roberta-large`` over the target taxonomy in
  ``targets.yaml``; multilingual by construction (Phase-1: en es ar hi fr pt de) and fine-tuned
  on code-switched corpora (LinCE Spanglish / Hinglish) in ``scripts/train_targets.py``.
* **Task C** — ordinal regression head (CORAL-style cumulative logits, K-1 = 4 thresholds)
  on the shared backbone.
* Languages other than English are routed to the ``mdeberta-v3`` multilingual toxicity head.

Fine-tuned weights are expected under ``settings.model_cache_dir/<task>/`` in the standard
Hugging Face layout (``config.json`` + ``model.safetensors`` + tokenizer files), produced by
the training scripts under ``backend/scripts/``.  When a checkpoint is missing the backend
falls back to the heuristic classifier for that task and reports it in ``model_version`` so
the dashboard can show which heads are live.

Everything heavy is imported lazily so the module can be imported (and unit-tested) without
torch installed.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from sentinai.classification.heuristic import HeuristicClassifier, HeuristicOutput
from sentinai.config import get_settings
from sentinai.schemas import (
    SEVERITY_NAMES,
    PreprocessedText,
    SeverityLevel,
    SeverityResult,
    TargetCategory,
    TargetResult,
    ToxicityLabel,
    ToxicityResult,
)

log = logging.getLogger(__name__)

TOXICITY_LABELS = [l.value for l in ToxicityLabel]  # noqa: E741


def ml_available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception:
        return False
    return True


@dataclass
class HeadStatus:
    name: str
    path: Path
    loaded: bool
    model_name: str
    error: str | None = None


class _SequenceHead:
    """Thin wrapper around a HF sequence classification model."""

    def __init__(self, path: Path, device: str | None = None, max_len: int = 256):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.max_len = max_len
        cfg_labels = getattr(self.model.config, "id2label", None) or {}
        self.labels = [cfg_labels[i] for i in range(len(cfg_labels))] if cfg_labels else None
        self.problem_type = getattr(self.model.config, "problem_type", None)

    def logits(self, texts: list[str]):
        import torch

        enc = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt").to(self.device)
        with torch.no_grad():
            return self.model(**enc).logits.float().cpu()

    def softmax(self, texts: list[str]) -> list[list[float]]:
        import torch

        return torch.softmax(self.logits(texts), dim=-1).tolist()

    def sigmoid(self, texts: list[str]) -> list[list[float]]:
        import torch

        return torch.sigmoid(self.logits(texts)).tolist()


class TransformerClassifier:
    """Task A/B/C transformer heads with heuristic fallback per task."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.heuristic = HeuristicClassifier()
        self.status: dict[str, HeadStatus] = {}
        self._tox_en: _SequenceHead | None = None
        self._tox_multi: _SequenceHead | None = None
        self._targets: _SequenceHead | None = None
        self._severity: _SequenceHead | None = None
        self._target_labels: list[tuple[str, str]] = []
        if ml_available():
            self._load_heads()
        else:
            log.warning("torch/transformers not installed — transformer backend degrades to heuristic heads")

    # ------------------------------------------------------------------------------------
    def _load_heads(self) -> None:
        base = Path(self.settings.model_cache_dir)
        specs = {
            "toxicity_en": (base / "toxicity_en", self.settings.toxicity_model_name),
            "toxicity_multi": (base / "toxicity_multi", self.settings.multilingual_model_name),
            "targets": (base / "targets", self.settings.target_model_name),
            "severity": (base / "severity", self.settings.toxicity_model_name),
        }
        for name, (path, model_name) in specs.items():
            if not (path / "config.json").exists():
                self.status[name] = HeadStatus(name, path, False, model_name, "checkpoint not found")
                continue
            try:
                head = _SequenceHead(path)
                setattr(self, {"toxicity_en": "_tox_en", "toxicity_multi": "_tox_multi", "targets": "_targets", "severity": "_severity"}[name], head)
                if name == "targets":
                    labels_file = path / "target_labels.json"
                    if labels_file.exists():
                        self._target_labels = [tuple(x) for x in json.loads(labels_file.read_text())]
                    elif head.labels:
                        self._target_labels = [tuple(l.split("/", 1)) for l in head.labels]  # noqa: E741
                self.status[name] = HeadStatus(name, path, True, model_name)
            except Exception as exc:  # pragma: no cover - depends on local checkpoints
                log.exception("failed to load %s", name)
                self.status[name] = HeadStatus(name, path, False, model_name, str(exc))

    @property
    def model_version(self) -> str:
        live = [k for k, v in self.status.items() if v.loaded]
        return "transformer[" + ",".join(live) + "]" if live else "transformer[none]+heuristic"

    # ------------------------------------------------------------------------------------
    def classify(self, pre: PreprocessedText) -> HeuristicOutput:
        out = self.heuristic.classify(pre)
        text = pre.normalized

        tox_head = self._tox_en if pre.language == "en" else (self._tox_multi or self._tox_en)
        if tox_head is not None:
            probs = tox_head.softmax([text])[0]
            labels = tox_head.labels or TOXICITY_LABELS
            p = {lab: float(v) for lab, v in zip(labels, probs, strict=False)}
            # Ensemble with the lexicon prior (rules are precise on explicit slurs / threats).
            for k in TOXICITY_LABELS:
                p[k] = 0.8 * p.get(k, 0.0) + 0.2 * out.toxicity.probabilities.get(k, 0.0)
            tot = sum(p.values()) or 1.0
            p = {k: v / tot for k, v in p.items()}
            label = max(p, key=p.get)  # type: ignore[arg-type]
            out.toxicity = ToxicityResult(
                label=ToxicityLabel(label),
                confidence=round(p[label], 4),
                probabilities={k: round(v, 4) for k, v in p.items()},
                toxicity_score=round(1.0 - p.get(ToxicityLabel.NON_TOXIC.value, 0.0), 4),
            )

        if self._targets is not None and self._target_labels and out.toxicity.toxicity_score >= 0.3:
            sig = self._targets.sigmoid([text])[0]
            merged: dict[tuple[str, str], TargetResult] = {(t.category.value, t.label): t for t in out.targets}
            for (cat, label), s in zip(self._target_labels, sig, strict=False):
                if s < 0.5:
                    continue
                key = (cat, label)
                prev = merged.get(key)
                conf = float(s) if prev is None else 1 - (1 - float(s)) * (1 - prev.confidence)
                merged[key] = TargetResult(category=TargetCategory(cat), label=label, confidence=round(conf, 4), evidence=prev.evidence if prev else [])
            out.targets = sorted(merged.values(), key=lambda t: t.confidence, reverse=True)

        if self._severity is not None:
            import torch

            logits = self._severity.logits([text])[0]
            # CORAL: P(level > k) = sigmoid(logit_k), k = 0..3
            cum = torch.sigmoid(logits).tolist()
            probs_list = []
            prev = 1.0
            for c in cum:
                probs_list.append(max(0.0, prev - c))
                prev = c
            probs_list.append(max(0.0, prev))
            tot = sum(probs_list) or 1.0
            probs = {k: v / tot for k, v in enumerate(probs_list)}
            level = 0
            for k in (4, 3, 2, 1):
                if sum(p for j, p in probs.items() if j >= k) > 0.5:
                    level = k
                    break
            lvl = SeverityLevel(level)
            out.severity = SeverityResult(
                level=lvl,
                level_name=SEVERITY_NAMES[lvl],
                confidence=round(probs[level], 4),
                probabilities={k: round(v, 4) for k, v in probs.items()},
                expected_level=round(sum(k * p for k, p in probs.items()), 4),
            )
        return out


@lru_cache
def get_transformer_classifier() -> TransformerClassifier:
    return TransformerClassifier()
