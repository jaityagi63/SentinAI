"""Dependency-free lexicon / rule classification backend.

It implements the three parallel tasks of Module 3 on top of the YAML lexicons:

* **Task A** toxicity label + probabilities  (non_toxic / offensive / hate_speech / violent_extremism)
* **Task B** target demographics (ethnicity / religion / nationality) with evidence spans
* **Task C** ordinal severity (1 microaggression … 4 incitement) as a probability distribution

and produces *token attributions* from the matched spans so that the same explanation UI works
for both this backend and the transformer backend (Module 4).

The backend is deterministic, fast (<1 ms / post) and serves three purposes:
  1. a fully functional default so the platform runs end-to-end without GPUs / checkpoints;
  2. a rule-based *prior* that the transformer backend ensembles with (evidence extraction);
  3. the reference implementation exercised by the test-suite.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from sentinai.classification.lexicon import GroupTerm, HostilityPattern, Lexicon, get_lexicon
from sentinai.schemas import (
    SEVERITY_NAMES,
    Explanation,
    PreprocessedText,
    SeverityLevel,
    SeverityResult,
    TargetCategory,
    TargetResult,
    TokenAttribution,
    ToxicityLabel,
    ToxicityResult,
)

MODEL_VERSION = "heuristic-lexicon-v1"

_TOKEN_RE = re.compile(r"[\w'’]+|\(\(\(", re.UNICODE)


@dataclass
class Match:
    category: str
    weight: float
    start: int
    end: int
    source: str
    targeted: bool = False


@dataclass
class HeuristicOutput:
    toxicity: ToxicityResult
    targets: list[TargetResult]
    severity: SeverityResult
    explanation: Explanation
    scores: dict[str, float] = field(default_factory=dict)
    counterspeech: float = 0.0
    has_target: bool = False


def _soft_or(weights: list[float]) -> float:
    """Noisy-OR combination: several moderate cues add up but never exceed 1."""
    p = 1.0
    for w in weights:
        p *= 1.0 - max(0.0, min(1.0, w))
    return 1.0 - p


class HeuristicClassifier:
    def __init__(self, lexicon: Lexicon | None = None):
        self.lex = lexicon or get_lexicon()

    # ------------------------------------------------------------------------------------
    def classify(self, pre: PreprocessedText) -> HeuristicOutput:
        text = pre.normalized
        matches: list[Match] = []

        # --- hostility patterns ---------------------------------------------------------
        for pat in self.lex.patterns:
            for m in pat.regex.finditer(text):
                if m.end() <= m.start():
                    continue
                matches.append(Match(pat.category, pat.weight, m.start(), m.end(), pat.source, pat.targeted))

        # --- group mentions and slurs ---------------------------------------------------
        mentions = self.lex.find_groups(text)
        slur_hits = [(t, s, e) for (t, s, e) in mentions if t.is_slur]
        neutral_hits = [(t, s, e) for (t, s, e) in mentions if not t.is_slur]

        hostile_context = any(m.category in ("stereotype", "dehumanization", "harassment", "incitement", "offensive") for m in matches)
        slur_weights: list[float] = []
        for term, s, e in slur_hits:
            w = term.weight
            if term.ambiguous and not hostile_context:
                w *= 0.25
            slur_weights.append(w)
            matches.append(Match("harassment", w, s, e, f"slur:{term.term}", True))

        by_cat: dict[str, list[float]] = {}
        for m in matches:
            by_cat.setdefault(m.category, []).append(m.weight)

        s_ste = _soft_or(by_cat.get("stereotype", []))
        s_deh = _soft_or(by_cat.get("dehumanization", []))
        s_har = _soft_or(by_cat.get("harassment", []))
        s_inc = _soft_or(by_cat.get("incitement", []))
        s_off = _soft_or(by_cat.get("offensive", []))
        s_cs = _soft_or(by_cat.get("counterspeech", []))

        targeted_pattern = any(m.targeted and m.category != "counterspeech" for m in matches)
        has_target = bool(neutral_hits) or bool(slur_hits) or targeted_pattern
        explicit_group = bool(neutral_hits) or any(w >= 0.5 for w in slur_weights)

        # --- Task A: toxicity -----------------------------------------------------------
        hostility = max(s_har, s_deh, s_ste * 0.9)
        if explicit_group:
            hate_raw = hostility
        elif has_target:
            hate_raw = hostility * 0.75
        else:
            hate_raw = hostility * 0.55
        ve_raw = s_inc if (explicit_group or s_inc >= 0.9) else s_inc * 0.6
        off_raw = s_off if has_target else max(s_off, 0.5 * hostility)

        # Counter-speech / reporting cues: the post is *about* hate rather than expressing it.
        cs_discount = 1.0 - 0.75 * s_cs
        hate_raw *= cs_discount
        ve_raw *= cs_discount
        off_raw *= 1.0 - 0.4 * s_cs

        # Obfuscation is itself a weak signal of evasive intent.
        if pre.obfuscation.score > 0 and (hate_raw > 0.2 or ve_raw > 0.2):
            boost = 1.0 + 0.15 * pre.obfuscation.score
            hate_raw = min(1.0, hate_raw * boost)
            ve_raw = min(1.0, ve_raw * boost)

        m_ve = ve_raw
        m_hs = hate_raw * (1.0 - 0.5 * ve_raw)
        m_off = off_raw * (1.0 - max(hate_raw, ve_raw))
        m_nt = 1.0 - max(ve_raw, hate_raw, off_raw)
        total = m_ve + m_hs + m_off + m_nt or 1.0
        probs = {
            ToxicityLabel.NON_TOXIC.value: m_nt / total,
            ToxicityLabel.OFFENSIVE.value: m_off / total,
            ToxicityLabel.HATE_SPEECH.value: m_hs / total,
            ToxicityLabel.VIOLENT_EXTREMISM.value: m_ve / total,
        }
        label = max(probs, key=probs.get)  # type: ignore[arg-type]
        toxicity = ToxicityResult(
            label=ToxicityLabel(label),
            confidence=round(probs[label], 4),
            probabilities={k: round(v, 4) for k, v in probs.items()},
            toxicity_score=round(1.0 - probs[ToxicityLabel.NON_TOXIC.value], 4),
        )

        # --- Task B: targets ------------------------------------------------------------
        targets = self._targets(neutral_hits, slur_hits, slur_weights, toxicity.toxicity_score, s_cs, text)

        # --- Task C: severity -----------------------------------------------------------
        severity = self._severity(s_ste, s_deh, s_har, ve_raw, toxicity.toxicity_score, has_target)

        # --- explanation ----------------------------------------------------------------
        explanation = self._explain(text, matches, neutral_hits, toxicity.toxicity_score)

        return HeuristicOutput(
            toxicity=toxicity,
            targets=targets,
            severity=severity,
            explanation=explanation,
            scores={
                "stereotype": round(s_ste, 4),
                "dehumanization": round(s_deh, 4),
                "harassment": round(s_har, 4),
                "incitement": round(s_inc, 4),
                "offensive": round(s_off, 4),
                "counterspeech": round(s_cs, 4),
            },
            counterspeech=round(s_cs, 4),
            has_target=has_target,
        )

    # ------------------------------------------------------------------------------------
    def _targets(
        self,
        neutral_hits: list[tuple[GroupTerm, int, int]],
        slur_hits: list[tuple[GroupTerm, int, int]],
        slur_weights: list[float],
        toxicity: float,
        counterspeech: float,
        text: str,
    ) -> list[TargetResult]:
        if toxicity < 0.3:
            return []
        agg: dict[tuple[str, str], tuple[float, list[str]]] = {}

        def add(term: GroupTerm, conf: float) -> None:
            key = (term.category, term.label)
            prev = agg.get(key)
            if prev is None:
                agg[key] = (conf, [term.term])
            else:
                agg[key] = (1 - (1 - prev[0]) * (1 - conf), prev[1] + [term.term] if term.term not in prev[1] else prev[1])

        for term, _s, _e in neutral_hits:
            # mention confidence scales with toxicity: a group named in a hateful post is its target
            add(term, min(0.97, 0.45 + 0.5 * toxicity))
        for (term, _s, _e), w in zip(slur_hits, slur_weights, strict=True):
            add(term, min(0.99, 0.5 + 0.5 * w))

        out = [
            TargetResult(category=TargetCategory(cat), label=label, confidence=round(conf * (1 - 0.5 * counterspeech), 4), evidence=ev)
            for (cat, label), (conf, ev) in agg.items()
        ]
        out.sort(key=lambda t: t.confidence, reverse=True)
        return [t for t in out if t.confidence >= 0.3]

    def _severity(self, s_ste: float, s_deh: float, s_har: float, s_inc: float, toxicity: float, has_target: bool) -> SeverityResult:
        """Ordinal (CORAL-style) severity: cumulative P(level ≥ k) from the category scores.

        The harm taxonomy is about the *most severe* behaviour present, so P(level ≥ k) is the
        strongest evidence at level k or above, damped by the overall toxicity so that
        counter-speech / non-toxic posts collapse to level 0.
        """
        scale = min(1.0, toxicity / 0.4)
        ge4 = s_inc * scale
        ge3 = max(ge4, s_har * scale)
        ge2 = max(ge3, s_deh * scale)
        ge1 = max(ge2, s_ste * scale, (0.5 * toxicity) if has_target else 0.0)
        cumulative = {4: ge4, 3: ge3, 2: ge2, 1: ge1}
        probs = {
            0: 1 - ge1,
            1: ge1 - ge2,
            2: ge2 - ge3,
            3: ge3 - ge4,
            4: ge4,
        }
        probs = {k: max(0.0, v) for k, v in probs.items()}
        tot = sum(probs.values()) or 1.0
        probs = {k: v / tot for k, v in probs.items()}
        level = 0
        for k in (4, 3, 2, 1):
            if cumulative[k] >= 0.5:
                level = k
                break
        expected = sum(k * p for k, p in probs.items())
        lvl = SeverityLevel(level)
        return SeverityResult(
            level=lvl,
            level_name=SEVERITY_NAMES[lvl],
            confidence=round(probs[level], 4),
            probabilities={k: round(v, 4) for k, v in sorted(probs.items())},
            expected_level=round(expected, 4),
        )

    def _explain(self, text: str, matches: list[Match], neutral_hits: list[tuple[GroupTerm, int, int]], toxicity: float) -> Explanation:
        tokens = [(m.group(0), m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]
        weights = [0.0] * len(tokens)
        for m in matches:
            sign = -1.0 if m.category == "counterspeech" else 1.0
            for i, (_tok, s, e) in enumerate(tokens):
                if s < m.end and e > m.start:
                    w = sign * m.weight
                    if abs(w) > abs(weights[i]):
                        weights[i] = w
        # Group mentions in a hostile post: small positive weight (the "who") so the reviewer
        # sees the target highlighted, but never dominate the hostile phrase (the "what").
        if toxicity >= 0.3:
            for _term, s, e in neutral_hits:
                for i, (_tok, ts, te) in enumerate(tokens):
                    if ts < e and te > s and weights[i] == 0.0:
                        weights[i] = 0.2 * toxicity
        max_abs = max((abs(w) for w in weights), default=0.0) or 1.0
        attributions = [
            TokenAttribution(token=tok, start=s, end=e, weight=round(w / max_abs, 4)) for (tok, s, e), w in zip(tokens, weights, strict=True)
        ]
        pos = sorted([a for a in attributions if a.weight > 0], key=lambda a: a.weight, reverse=True)
        neg = sorted([a for a in attributions if a.weight < 0], key=lambda a: a.weight)
        seen: set[str] = set()
        top_pos = [a.token for a in pos if not (a.token in seen or seen.add(a.token))][:8]  # type: ignore[func-returns-value]
        seen = set()
        top_neg = [a.token for a in neg if not (a.token in seen or seen.add(a.token))][:8]  # type: ignore[func-returns-value]
        matched = [
            {"category": m.category, "weight": m.weight, "span": text[m.start : m.end], "pattern": m.source[:80]}
            for m in sorted(matches, key=lambda m: -m.weight)[:12]
        ]
        return Explanation(
            method="lexicon-attribution",
            text=text,
            attributions=attributions,
            base_value=round(1.0 / (1.0 + math.exp(3.0)), 4),
            top_positive=top_pos,
            top_negative=top_neg,
            matched_patterns=matched,
        )


__all__ = ["HeuristicClassifier", "HeuristicOutput", "HostilityPattern", "MODEL_VERSION"]
