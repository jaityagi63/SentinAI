"""The per-post pipeline: preprocessing → Task A/B/C → XAI → multimodal → context → final score.

``ClassificationEngine.classify_post`` is the single entry point used by the ingestion
worker, the API (``POST /api/classify``) and the demo seeder.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from sentinai.classification.context import ContextAnalyzer, ParentInfo
from sentinai.classification.heuristic import MODEL_VERSION as HEURISTIC_VERSION
from sentinai.classification.heuristic import HeuristicClassifier
from sentinai.classification.lexicon import get_lexicon
from sentinai.classification.multimodal import MultimodalAnalyzer
from sentinai.config import get_settings
from sentinai.preprocessing.pipeline import Preprocessor
from sentinai.schemas import Classification, Post, PreprocessedText, ToxicityLabel

log = logging.getLogger(__name__)


class ClassificationEngine:
    def __init__(self, backend: str | None = None, dedup: bool = True):
        self.settings = get_settings()
        self.backend_name = backend or self.settings.classifier_backend
        self.lexicon = get_lexicon()
        self.preprocessor = Preprocessor(known_words=self.lexicon.vocabulary, dedup=dedup)
        self.heuristic = HeuristicClassifier(self.lexicon)
        self.classifier = self.heuristic
        self.model_version = HEURISTIC_VERSION
        if self.backend_name == "transformer":
            from sentinai.classification.transformer import get_transformer_classifier

            self.classifier = get_transformer_classifier()
            self.model_version = self.classifier.model_version
        self.context = ContextAnalyzer()
        self.multimodal = MultimodalAnalyzer(self.score_text)

    # ------------------------------------------------------------------------------------
    def score_text(self, text: str) -> float:
        """Toxicity score for an arbitrary snippet (OCR text, alt-text, LIME perturbations)."""
        pre = self.preprocessor.run(text)
        return self.classifier.classify(pre).toxicity.toxicity_score

    def preprocess(self, text: str, key: str | None = None) -> PreprocessedText:
        return self.preprocessor.run(text, key)

    # ------------------------------------------------------------------------------------
    def classify_text(self, text: str, key: str | None = None, explain: bool = True) -> Classification:
        post = Post(id=key or "adhoc", text=text, created_at=__import__("datetime").datetime.utcnow(), author_id="adhoc")
        return self.classify_post(post, explain=explain)

    def classify_post(self, post: Post, parent: ParentInfo | None = None, explain: bool = True) -> Classification:
        pre = self.preprocessor.run(post.text, post.id)
        out = self.classifier.classify(pre)

        # --- Module 5: media ------------------------------------------------------------
        visual = self.multimodal.analyze(post.media, out.toxicity.toxicity_score) if post.media else None
        tox_after_visual = self.multimodal.ensemble(out.toxicity.toxicity_score, visual)

        # --- Module 6: context ----------------------------------------------------------
        ctx = self.context.analyze(post, pre.normalized, tox_after_visual, out.counterspeech, parent)
        final = round(min(1.0, tox_after_visual * ctx.discount_factor), 4)

        # If the visual channel raised the score, escalate the label accordingly.
        toxicity = out.toxicity
        if visual is not None and visual.visual_toxicity >= 0.6 and toxicity.label == ToxicityLabel.NON_TOXIC:
            probs = dict(toxicity.probabilities)
            probs[ToxicityLabel.HATE_SPEECH.value] = max(probs.get(ToxicityLabel.HATE_SPEECH.value, 0), final)
            probs[ToxicityLabel.NON_TOXIC.value] = 1 - final
            tot = sum(probs.values()) or 1
            probs = {k: round(v / tot, 4) for k, v in probs.items()}
            label = max(probs, key=probs.get)  # type: ignore[arg-type]
            toxicity = toxicity.model_copy(update={"label": ToxicityLabel(label), "confidence": probs[label], "probabilities": probs})

        # --- Module 14: active learning band --------------------------------------------
        lo, hi = self.settings.active_learning_low, self.settings.active_learning_high
        needs_review = lo <= final <= hi or (lo <= toxicity.confidence <= hi and final >= lo)

        explanation = out.explanation if explain else None
        if explain and self.backend_name == "transformer":
            explanation = self._transformer_explanation(pre, out.explanation)

        return Classification(
            post_id=post.id,
            model_version=self.model_version,
            language=pre.language,
            toxicity=toxicity,
            targets=out.targets,
            severity=out.severity,
            explanation=explanation,
            visual=visual,
            context=ctx,
            obfuscation_score=pre.obfuscation.score,
            duplicate_of=pre.duplicate_of,
            final_toxicity=final,
            needs_review=needs_review,
        )

    # ------------------------------------------------------------------------------------
    def _transformer_explanation(self, pre: PreprocessedText, lexicon_expl):
        """SHAP over the live toxicity head when possible, else LIME over the ensemble scorer;
        always blended with the exact lexicon spans."""
        from sentinai.classification.explain import lime_explain, merge_explanations, shap_explain

        clf = self.classifier
        head = getattr(clf, "_tox_en", None) if pre.language == "en" else getattr(clf, "_tox_multi", None)
        expl = None
        if head is not None:
            try:
                from transformers import pipeline  # type: ignore

                pipe = pipeline("text-classification", model=head.model, tokenizer=head.tokenizer, top_k=None, device=-1)
                expl = shap_explain(pre.normalized, pipe)
            except Exception as exc:  # pragma: no cover
                log.debug("shap pipeline failed: %s", exc)
        if expl is None:
            expl = lime_explain(pre.normalized, lambda xs: [[1 - self.score_text(x), self.score_text(x)] for x in xs])
        if expl is None:
            return lexicon_expl
        return merge_explanations(expl, lexicon_expl)


@lru_cache
def get_engine() -> ClassificationEngine:
    return ClassificationEngine()
