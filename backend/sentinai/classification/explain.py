"""Module 4 — Explainability & token attribution.

Three strategies, selected automatically:

* ``shap``  — ``shap.Explainer`` over the transformer pipeline (partition explainer on text),
              used when the transformer backend + ``shap`` are available.
* ``lime``  — ``lime.lime_text.LimeTextExplainer`` against any callable scorer (works with the
              heuristic backend too, since it only needs a ``predict_proba`` function).
* ``lexicon-attribution`` — exact span attribution from the matched lexicon patterns (always
              available; produced by :class:`HeuristicClassifier`).

The output is a list of per-token weights in ``[-1, 1]`` (red = pushes toward toxic, green =
pushes toward neutral) consumed by the dashboard's heatmap overlay.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable

from sentinai.schemas import Explanation, TokenAttribution

log = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[\w'’]+|[^\w\s]", re.UNICODE)


def _tokens_with_spans(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(0), m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]


def _normalise(weights: list[float]) -> list[float]:
    mx = max((abs(w) for w in weights), default=0.0) or 1.0
    return [round(w / mx, 4) for w in weights]


def _build(text: str, method: str, tok_weights: dict[str, float], base_value: float) -> Explanation:
    toks = _tokens_with_spans(text)
    weights = [tok_weights.get(t.lower(), 0.0) for t, _, _ in toks]
    weights = _normalise(weights)
    attributions = [TokenAttribution(token=t, start=s, end=e, weight=w) for (t, s, e), w in zip(toks, weights, strict=True)]
    pos = sorted({a.token for a in attributions if a.weight > 0.1}, key=lambda t: -tok_weights.get(t.lower(), 0))[:8]
    neg = sorted({a.token for a in attributions if a.weight < -0.1}, key=lambda t: tok_weights.get(t.lower(), 0))[:8]
    return Explanation(method=method, text=text, attributions=attributions, base_value=round(base_value, 4), top_positive=pos, top_negative=neg)


def lime_explain(text: str, predict_proba: Callable[[list[str]], list[list[float]]], toxic_index: int = 1, num_samples: int = 300) -> Explanation | None:
    """LIME local explanation.  ``predict_proba`` maps texts → [[p_non_toxic, p_toxic], ...]."""
    try:
        from lime.lime_text import LimeTextExplainer  # type: ignore
    except Exception:
        return None
    try:
        import numpy as np

        explainer = LimeTextExplainer(class_names=["non_toxic", "toxic"], bow=True, split_expression=r"\W+")
        exp = explainer.explain_instance(text, lambda xs: np.asarray(predict_proba(list(xs))), labels=(toxic_index,), num_features=20, num_samples=num_samples)
        weights = {w.lower(): float(v) for w, v in exp.as_list(label=toxic_index)}
        base = float(exp.predict_proba[toxic_index]) if getattr(exp, "predict_proba", None) is not None else 0.0
        return _build(text, "lime", weights, base)
    except Exception as exc:  # pragma: no cover - depends on optional dependency
        log.debug("LIME failed: %s", exc)
        return None


def shap_explain(text: str, hf_pipeline, toxic_labels: tuple[str, ...] = ("hate_speech", "violent_extremism", "offensive")) -> Explanation | None:
    """SHAP partition explainer on a Hugging Face text-classification pipeline."""
    try:
        import shap  # type: ignore
    except Exception:
        return None
    try:
        explainer = shap.Explainer(hf_pipeline)
        sv = explainer([text])
        # sum the toxic-class attributions
        names = list(sv.output_names) if getattr(sv, "output_names", None) is not None else []
        idx = [i for i, n in enumerate(names) if n in toxic_labels] or [1]
        values = sv.values[0]  # (tokens, classes)
        tokens = list(sv.data[0])
        tok_weights: dict[str, float] = {}
        for tok, row in zip(tokens, values, strict=False):
            w = float(sum(row[i] for i in idx))
            key = tok.strip().lower()
            if key:
                tok_weights[key] = tok_weights.get(key, 0.0) + w
        base = float(sum(sv.base_values[0][i] for i in idx)) if hasattr(sv, "base_values") else 0.0
        return _build(text, "shap", tok_weights, base)
    except Exception as exc:  # pragma: no cover - depends on optional dependency
        log.debug("SHAP failed: %s", exc)
        return None


def heuristic_predict_proba(classifier, preprocessor) -> Callable[[list[str]], list[list[float]]]:
    """Adapter that turns the heuristic classifier into a LIME-compatible scorer."""

    def _predict(texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            pre = preprocessor.run(t)
            tox = classifier.classify(pre).toxicity.toxicity_score
            out.append([1 - tox, tox])
        return out

    return _predict


def merge_explanations(primary: Explanation, secondary: Explanation | None, alpha: float = 0.6) -> Explanation:
    """Blend two attributions defined over the same text (e.g. SHAP + lexicon spans)."""
    if secondary is None or secondary.text != primary.text:
        return primary
    sec = {(a.start, a.end): a.weight for a in secondary.attributions}
    merged = []
    for a in primary.attributions:
        w = alpha * a.weight + (1 - alpha) * sec.get((a.start, a.end), 0.0)
        merged.append(TokenAttribution(token=a.token, start=a.start, end=a.end, weight=round(w, 4)))
    ws = _normalise([m.weight for m in merged])
    merged = [TokenAttribution(token=m.token, start=m.start, end=m.end, weight=w) for m, w in zip(merged, ws, strict=True)]
    return Explanation(
        method=f"{primary.method}+{secondary.method}",
        text=primary.text,
        attributions=merged,
        base_value=primary.base_value,
        top_positive=[a.token for a in sorted(merged, key=lambda a: -a.weight) if a.weight > 0.1][:8],
        top_negative=[a.token for a in sorted(merged, key=lambda a: a.weight) if a.weight < -0.1][:8],
        matched_patterns=primary.matched_patterns or secondary.matched_patterns,
    )
