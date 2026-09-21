"""Module 2 orchestration: normalise → demojize → de-obfuscate → detect language →
transliterate → near-duplicate check."""

from __future__ import annotations

from sentinai.preprocessing.dedup import Deduplicator
from sentinai.preprocessing.language import detect_language, transliterate_if_needed
from sentinai.preprocessing.normalize import normalize, tokenize
from sentinai.preprocessing.obfuscation import deobfuscate, fuzzy_lexicon_hits
from sentinai.schemas import ObfuscationReport, PreprocessedText


class Preprocessor:
    def __init__(self, known_words: set[str] | None = None, dedup_threshold: float = 0.8, dedup: bool = True):
        self.known_words = known_words
        self.dedup = Deduplicator(threshold=dedup_threshold) if dedup else None

    def run(self, text: str, key: str | None = None) -> PreprocessedText:
        norm = normalize(text)
        obf = deobfuscate(norm.text, self.known_words)
        cleaned = obf.text

        # Deliberate misspellings of lexicon words (n1gg3r → already leet-decoded; muzlim → muslim)
        tokens = tokenize(cleaned)
        if self.known_words:
            hits = fuzzy_lexicon_hits(tokens, self.known_words)
            if hits:
                for bad, good in hits.items():
                    cleaned = _replace_word(cleaned, bad, good)
                obf.leetspeak.extend(f"{b}->{g}" for b, g in hits.items())
                obf.score = min(1.0, obf.score + 0.15 * len(hits))
                tokens = tokenize(cleaned)

        lang = detect_language(cleaned if cleaned else text)
        translit_text, changed = transliterate_if_needed(cleaned, lang)
        if changed:
            # Keep the Latin form *and* the transliterated form so that both Latin-script and
            # native-script lexicon patterns can match (code-switched text is common).
            cleaned = f"{cleaned} {translit_text}"
            tokens = tokenize(cleaned)

        dup: str | None = None
        signature: list[int] | None = None
        if self.dedup is not None and key is not None:
            dup, signature = self.dedup.check_with_signature(key, norm.text)

        return PreprocessedText(
            original=text,
            normalized=cleaned,
            tokens=tokens,
            emojis=norm.emojis,
            hashtags=norm.hashtags,
            mentions_count=norm.mentions_count,
            urls_count=norm.urls_count,
            language=lang.lang,
            language_confidence=round(lang.confidence, 4),
            script=lang.script,
            transliterated=changed,
            obfuscation=ObfuscationReport(
                leetspeak=obf.leetspeak,
                homoglyphs=obf.homoglyphs,
                zero_width_chars=obf.zero_width_chars,
                spaced_out_words=obf.spaced_out_words,
                repeated_chars=obf.repeated_chars,
                score=obf.score,
            ),
            minhash_signature=signature,
            duplicate_of=dup,
        )


def _replace_word(text: str, bad: str, good: str) -> str:
    import re

    return re.sub(rf"\b{re.escape(bad)}\b", good, text, flags=re.I)


_default: Preprocessor | None = None


def preprocess(text: str, key: str | None = None) -> PreprocessedText:
    """Convenience wrapper using a process-wide preprocessor seeded with the lexicon vocabulary."""
    global _default
    if _default is None:
        from sentinai.classification.lexicon import get_lexicon

        _default = Preprocessor(known_words=get_lexicon().vocabulary)
    return _default.run(text, key)
