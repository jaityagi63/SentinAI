"""Module 6 — Context & counter-speech disambiguation.

* Reply-chain analysis: originator vs responder, parent lookup.
* Stance detection (Support / Deny / Condemn / Neutral / Query) relative to the parent /
  quoted post, using counter-speech cues, negation & quotation heuristics plus (optionally)
  the parent's own toxicity.
* Quotation detection: quote-tweets are classified as *amplification* (supportive framing)
  or *criticism* (condemning framing).
* Score adjustment: stance ∈ {Deny, Condemn} → toxicity × ``counterspeech_discount`` (0.2).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from sentinai.config import get_settings
from sentinai.schemas import ContextResult, Post, Stance

_CONDEMN_RE = re.compile(
    r"\b(this is|that is|this was|that was|thats|that's|it's|its|it is|sounds|what a) (so |just |pure |blatantly |openly |really |incredibly )?"
    r"(racist|racism|bigot(ed|ry)|xenophobi[ac]|islamophobi[ac]|antisemiti(c|sm)|hateful|hate speech|disgusting|vile|unacceptable|wrong|"
    r"dehumani[sz]ing|not ok(ay)?|shameful|appalling|sickening|dangerous rhetoric|hate)\b"
    r"|\b(condemn|denounc|call(ing|ed)? out|reported|reporting|shame on you|delete this|do better|be better|"
    r"you should be ashamed|ashamed of you|stop spreading hate|stop the hate|there is no place for|no place for this|"
    r"how is this (allowed|ok|okay)|why is this (allowed|still up|ok|okay)|disgusted|appalled|horrified|"
    r"this account (should|needs to) be (banned|suspended|reported)|ban this account|suspend this account)\w*",
    re.I,
)
_DENY_RE = re.compile(
    r"\b(that'?s not true|thats not true|this is not true|not true|false|fake news|misinformation|disinformation|lie|lies|lying|"
    r"nonsense|bullshit|baseless|no evidence|source\?|citation needed|debunked|fact[- ]check|actually[, ]|wrong[, .!]|"
    r"you'?re wrong|this is wrong|incorrect|myth|stereotype|generali[sz]ation|not all|that'?s a stereotype|"
    r"most .{1,30} (are not|aren'?t)|the data (says|shows) otherwise|studies show otherwise|statistically)\b",
    re.I,
)
_SUPPORT_RE = re.compile(
    r"^\s*(this|facts|truth|based|same|word|real)\W*$"
    r"|\b(exactly|so true|100%|agreed|i agree|well said|preach|finally someone|couldn'?t agree more|"
    r"say it louder|louder for the people in the back|thank you for saying this|been saying this|amen|"
    r"yep|yup|spot on|nailed it|real talk|no lies detected|periodt?|and i said what i said|retweet if you agree|"
    r"this (right here|so much|is the truth|is facts|is it|is the way)|he'?s right|she'?s right|they'?re right|you'?re right)\b",
    re.I,
)
_QUERY_RE = re.compile(r"\?\s*$|\b(why|what do you mean|source|can you explain|is this real|genuinely asking|serious question)\b", re.I)
_SARCASM_RE = re.compile(r"(/s\b|\bsarcasm\b|🙄|\bsure jan\b|\byeah right\b|\bimagine thinking\b)", re.I)
_QUOTE_MARKERS_RE = re.compile(r"[\"“”«»]{1}.{8,}[\"“”«»]{1}|^\s*(rt|quote|quoting|qt)\b|\bhe said\b|\bshe said\b|\bthey said\b|\bsomeone said\b|\bthis person said\b", re.I)


@dataclass
class ParentInfo:
    id: str
    text: str | None = None
    toxicity: float | None = None
    author_id: str | None = None


class ContextAnalyzer:
    def __init__(self, discount: float | None = None):
        self.discount = discount if discount is not None else get_settings().counterspeech_discount

    # ------------------------------------------------------------------------------------
    def analyze(self, post: Post, normalized_text: str, own_toxicity: float, counterspeech_score: float = 0.0, parent: ParentInfo | None = None) -> ContextResult:
        is_reply = post.parent_id is not None
        is_quote = post.quoted_id is not None
        is_retweet = post.retweeted_id is not None
        is_originator = not (is_reply or is_quote or is_retweet)
        parent_id = post.parent_id or post.quoted_id or post.retweeted_id

        stance, conf, reason = self._stance(normalized_text, own_toxicity, counterspeech_score, parent, is_quote, is_reply)

        factor = 1.0
        if stance in (Stance.DENY, Stance.CONDEMN):
            # Only discount when the post itself is not *independently* hateful (it might
            # condemn one group while attacking another).  We scale the discount by how much
            # of the hostile content can be attributed to the quoted material.
            attributable = min(1.0, counterspeech_score + (0.5 if (is_quote or is_reply) else 0.0) + (0.3 if _QUOTE_MARKERS_RE.search(normalized_text) else 0.0))
            factor = 1.0 - (1.0 - self.discount) * attributable * conf
            factor = max(self.discount, factor)
        elif stance == Stance.SUPPORT and parent is not None and parent.toxicity is not None and parent.toxicity >= 0.6:
            # Amplifying a hateful post: endorse the parent's toxicity (never below own).
            factor = 1.0
        elif is_retweet and parent is not None and parent.toxicity is not None:
            factor = 1.0

        return ContextResult(
            is_reply=is_reply,
            is_quote=is_quote,
            is_retweet=is_retweet,
            is_originator=is_originator,
            parent_id=parent_id,
            stance=stance,
            stance_confidence=round(conf, 4),
            discount_factor=round(factor, 4),
            reasoning=reason,
        )

    # ------------------------------------------------------------------------------------
    def _stance(self, text: str, own_tox: float, cs: float, parent: ParentInfo | None, is_quote: bool, is_reply: bool) -> tuple[Stance, float, str]:
        condemn = len(_CONDEMN_RE.findall(text))
        deny = len(_DENY_RE.findall(text))
        support = len(_SUPPORT_RE.findall(text))
        query = 1 if _QUERY_RE.search(text) else 0
        sarcasm = 1 if _SARCASM_RE.search(text) else 0
        quoted = 1 if _QUOTE_MARKERS_RE.search(text) else 0

        s_condemn = min(1.0, 0.45 * condemn + 0.6 * cs + 0.15 * quoted)
        s_deny = min(1.0, 0.45 * deny + 0.2 * cs)
        s_support = min(1.0, 0.4 * support) * (0.3 if sarcasm else 1.0)
        s_query = 0.5 * query

        # A short supportive reply ("exactly", "facts") to a toxic parent is amplification.
        if parent is not None and parent.toxicity is not None and parent.toxicity >= 0.6:
            if len(text.split()) <= 6 and support and not condemn:
                s_support = max(s_support, 0.85)
            # A non-toxic reply that names the behaviour is condemnation.
            if own_tox < 0.4 and (condemn or cs > 0.3):
                s_condemn = max(s_condemn, 0.7)

        # Standalone posts with no counter-speech cues: neutral (no adjustment).
        scores = {Stance.CONDEMN: s_condemn, Stance.DENY: s_deny, Stance.SUPPORT: s_support, Stance.QUERY: s_query}
        best = max(scores, key=scores.get)  # type: ignore[arg-type]
        conf = scores[best]
        if conf < 0.3:
            return Stance.NEUTRAL, round(1.0 - conf, 4), "no stance cue detected"

        reasons = []
        if condemn:
            reasons.append(f"{condemn} condemnation cue(s)")
        if deny:
            reasons.append(f"{deny} denial cue(s)")
        if support:
            reasons.append(f"{support} agreement cue(s)")
        if cs:
            reasons.append(f"counter-speech lexicon score {cs:.2f}")
        if quoted:
            reasons.append("quoted material detected")
        if sarcasm:
            reasons.append("sarcasm marker")
        if is_quote:
            reasons.append("quote-tweet")
        elif is_reply:
            reasons.append("reply")
        return best, round(conf, 4), "; ".join(reasons)


def quote_framing(stance: Stance) -> str:
    """Module 6: quote-tweet framing label for the dashboard."""
    if stance in (Stance.CONDEMN, Stance.DENY):
        return "criticism"
    if stance == Stance.SUPPORT:
        return "amplification"
    return "neutral"
