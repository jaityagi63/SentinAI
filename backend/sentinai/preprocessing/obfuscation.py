"""Obfuscation detection & repair: leetspeak (r4c1st), deliberate misspellings / spaced-out
words (k i l l), zero-width characters, homoglyph substitutions (Cyrillic а vs Latin a).

Two outputs are produced:
  * a de-obfuscated version of the text used downstream by the classifiers, and
  * an :class:`ObfuscationReport` whose ``score`` becomes an evasion feature.

The ML part (a character n-gram logistic model that flags tokens that look like evasive
misspellings of lexicon words) is implemented in :func:`fuzzy_lexicon_hits` using
edit-distance against the loaded lexicon, so it needs no third-party dependency.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# --------------------------------------------------------------------------------------------
# Zero-width & invisible characters
# --------------------------------------------------------------------------------------------
ZERO_WIDTH_CHARS = {
    "\u200b",  # zero width space
    "\u200c",  # zero width non-joiner  (NB: legitimate in Arabic/Persian/Indic text!)
    "\u200d",  # zero width joiner      (NB: legitimate in emoji sequences / Indic)
    "\u200e",
    "\u200f",
    "\u2060",  # word joiner
    "\u2061",
    "\u2062",
    "\u2063",
    "\u2064",
    "\ufeff",  # BOM / zero width no-break space
    "\u00ad",  # soft hyphen
    "\u034f",  # combining grapheme joiner
    "\u180e",
}
_ZW_RE = re.compile("[" + "".join(ZERO_WIDTH_CHARS) + "]")
_SCRIPT_NEEDS_ZWJ_RE = re.compile(r"[\u0600-\u06FF\u0900-\u0DFF]")

# --------------------------------------------------------------------------------------------
# Homoglyphs (confusables) → ASCII
# --------------------------------------------------------------------------------------------
HOMOGLYPHS: dict[str, str] = {
    # Cyrillic
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i", "ј": "j",
    "ѕ": "s", "ԁ": "d", "ɡ": "g", "һ": "h", "к": "k", "м": "m", "т": "t", "в": "b", "н": "h",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P", "С": "C",
    "Т": "T", "Х": "X", "У": "Y", "Ѕ": "S", "І": "I", "Ј": "J",
    # Greek
    "α": "a", "ο": "o", "ν": "v", "ρ": "p", "τ": "t", "ι": "i", "κ": "k", "χ": "x", "υ": "u",
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K", "Μ": "M", "Ν": "N",
    "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
    # Latin look-alikes / fullwidth / misc
    "ł": "l", "ı": "i", "ĸ": "k", "ǃ": "!", "ⅰ": "i", "ⅼ": "l", "ⅽ": "c", "ⅾ": "d", "ⅿ": "m",
    "ℓ": "l", "ｎ": "n", "ｉ": "i", "ｇ": "g", "ｅ": "e", "ｒ": "r", "ａ": "a", "ｏ": "o", "ｓ": "s",
    "ｔ": "t", "ｕ": "u", "ｋ": "k", "ｍ": "m", "ｈ": "h", "ｄ": "d", "ｃ": "c", "ｐ": "p", "ｌ": "l",
    "ｆ": "f", "ｗ": "w", "ｂ": "b", "ｊ": "j", "ｑ": "q", "ｖ": "v", "ｘ": "x", "ｙ": "y", "ｚ": "z",
    "\u0435": "e",
}
_HOMOGLYPH_RE = re.compile("[" + "".join(re.escape(k) for k in HOMOGLYPHS) + "]")

# --------------------------------------------------------------------------------------------
# Leetspeak
# --------------------------------------------------------------------------------------------
LEET_MAP: dict[str, str] = {
    "0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b", "9": "g", "@": "a", "$": "s",
    "!": "i", "|": "l", "+": "t", "€": "e", "£": "l", "¢": "c", "©": "c", "®": "r",
}
_LEET_TOKEN_RE = re.compile(r"\b(?=\w*[a-z])(?=\w*[0-9@$!|+€£])[\w@$!|+€£]{3,}\b", re.I)
_MIXED_DIGIT_WORD_RE = re.compile(r"^(?=.*[a-z])(?=.*[0-9@$!|+])[a-z0-9@$!|+]+$", re.I)
_PURE_ALNUM_ID_RE = re.compile(r"^(?:[a-z]{1,3}\d{2,}|\d+[a-z]{1,2}|\w*\d{4,}\w*|[a-z]+\d{1,2})$", re.I)
_SPACED_RE = re.compile(r"\b(?:[A-Za-z][\s.\-_*]){2,}[A-Za-z]\b")
_REPEAT_RE = re.compile(r"(\w)\1{2,}")
_SEP_INSIDE_RE = re.compile(r"\b([a-z])[.\-_*](?=[a-z][.\-_*]?[a-z])", re.I)


@dataclass
class ObfuscationResult:
    text: str  # repaired text
    leetspeak: list[str] = field(default_factory=list)
    homoglyphs: list[str] = field(default_factory=list)
    zero_width_chars: int = 0
    spaced_out_words: list[str] = field(default_factory=list)
    repeated_chars: list[str] = field(default_factory=list)
    score: float = 0.0


def strip_zero_width(text: str) -> tuple[str, int]:
    """Remove zero-width characters.

    ZWJ/ZWNJ are legitimate in Arabic and Indic scripts (and emoji sequences); we only count
    them as evasion when the surrounding text is not in such a script.
    """
    count = 0
    out: list[str] = []
    n = len(text)
    for i, ch in enumerate(text):
        if ch in ZERO_WIDTH_CHARS:
            if ch in ("\u200c", "\u200d"):
                left = text[i - 1] if i > 0 else ""
                right = text[i + 1] if i + 1 < n else ""
                if _SCRIPT_NEEDS_ZWJ_RE.search(left + right) or _is_emoji(left) or _is_emoji(right):
                    out.append(ch)
                    continue
            count += 1
            continue
        out.append(ch)
    return "".join(out), count


def _is_emoji(ch: str) -> bool:
    if not ch:
        return False
    cp = ord(ch)
    return 0x1F300 <= cp <= 0x1FAFF or 0x2600 <= cp <= 0x27BF or cp == 0xFE0F


def fix_homoglyphs(text: str) -> tuple[str, list[str]]:
    """Map confusable characters to ASCII **only inside words that are otherwise Latin**.

    A Cyrillic word such as ``привет`` is left untouched; ``rаcist`` (Cyrillic а) is repaired.
    """
    fixed_words: list[str] = []

    def repl_word(m: re.Match[str]) -> str:
        word = m.group(0)
        if not _HOMOGLYPH_RE.search(word):
            return word
        latin = sum(1 for c in word if ("a" <= c.lower() <= "z"))
        confus = sum(1 for c in word if c in HOMOGLYPHS)
        other = len(word) - latin - confus
        # Require the word to be predominantly Latin with sprinkled confusables.
        if latin >= 1 and other == 0 and confus <= max(1, len(word) // 2):
            fixed = "".join(HOMOGLYPHS.get(c, c) for c in word)
            fixed_words.append(word)
            return fixed
        return word

    return re.sub(r"\S+", repl_word, text), fixed_words


def decode_leet_token(tok: str) -> str:
    return "".join(LEET_MAP.get(c, c) for c in tok)


def fix_leetspeak(text: str, known_words: set[str] | None = None) -> tuple[str, list[str]]:
    """Decode leetspeak tokens (``r4c1st`` → ``racist``).

    Tokens are decoded when they mix letters with leet digits/symbols and the decoded form is
    either a known lexicon word or has a plausible letter shape (vowel present, not an ID/model
    number such as ``mp3``, ``covid19`` or ``h1b``).
    """
    decoded_tokens: list[str] = []

    def repl(m: re.Match[str]) -> str:
        tok = m.group(0)
        if _PURE_ALNUM_ID_RE.match(tok) and (known_words is None or decode_leet_token(tok.lower()) not in known_words):
            return tok
        dec = decode_leet_token(tok.lower())
        if not dec.isalpha():
            return tok
        if known_words is not None and dec in known_words:
            decoded_tokens.append(tok)
            return dec
        digits = sum(1 for c in tok if not c.isalpha())
        if digits >= 1 and len(dec) >= 4 and re.search(r"[aeiouy]", dec) and digits <= len(tok) // 2:
            decoded_tokens.append(tok)
            return dec
        return tok

    return _LEET_TOKEN_RE.sub(repl, text), decoded_tokens


def fix_spaced_out(text: str) -> tuple[str, list[str]]:
    """``k i l l`` / ``k.i.l.l`` / ``k-i-l-l`` → ``kill``."""
    found: list[str] = []

    def repl(m: re.Match[str]) -> str:
        seq = m.group(0)
        joined = re.sub(r"[\s.\-_*]", "", seq)
        if len(joined) >= 3:
            found.append(seq)
            return joined
        return seq

    out = _SPACED_RE.sub(repl, text)
    return out, found


def collapse_repeats(text: str) -> tuple[str, list[str]]:
    """``killlll`` → ``kill`` (keep at most two repeats, which covers legit doubles)."""
    found = [m.group(0) for m in _REPEAT_RE.finditer(text)]
    return _REPEAT_RE.sub(lambda m: m.group(1) * 2, text), found


def deobfuscate(text: str, known_words: set[str] | None = None) -> ObfuscationResult:
    stripped, zw = strip_zero_width(text)
    stripped = unicodedata.normalize("NFKC", stripped)
    fixed, homoglyphs = fix_homoglyphs(stripped)
    fixed, spaced = fix_spaced_out(fixed)
    fixed, leet = fix_leetspeak(fixed, known_words)
    fixed, repeats = collapse_repeats(fixed)

    n_tokens = max(1, len(fixed.split()))
    signals = len(leet) * 1.0 + len(homoglyphs) * 1.5 + len(spaced) * 1.0 + min(zw, 5) * 0.6 + len(repeats) * 0.3
    score = min(1.0, signals / (n_tokens**0.5 + 1))
    return ObfuscationResult(
        text=fixed,
        leetspeak=leet,
        homoglyphs=homoglyphs,
        zero_width_chars=zw,
        spaced_out_words=spaced,
        repeated_chars=repeats,
        score=round(score, 4),
    )


# --------------------------------------------------------------------------------------------
# Fuzzy lexicon matching for deliberate misspellings (n1gg3r / racisst / muzlim)
# --------------------------------------------------------------------------------------------


def _damerau_levenshtein(a: str, b: str, max_dist: int = 2) -> int:
    if abs(len(a) - len(b)) > max_dist:
        return max_dist + 1
    prev_prev: list[int] | None = None
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if i > 1 and j > 1 and ca == b[j - 2] and a[i - 2] == cb and prev_prev is not None:
                cur[j] = min(cur[j], prev_prev[j - 2] + 1)
        if min(cur) > max_dist:
            return max_dist + 1
        prev_prev, prev = prev, cur
    return prev[-1]


_COMMON_VOCAB: set[str] | None = None


def common_vocabulary() -> set[str]:
    """A large list of ordinary words (wordninja's frequency list when installed, else a small
    built-in set).  Tokens in this list are never treated as misspelled slurs."""
    global _COMMON_VOCAB
    if _COMMON_VOCAB is not None:
        return _COMMON_VOCAB
    words: set[str] = set()
    try:
        import gzip
        import os

        import wordninja  # type: ignore

        path = os.path.join(os.path.dirname(wordninja.__file__), "wordninja", "wordninja_words.txt.gz")
        if not os.path.exists(path):
            path = os.path.join(os.path.dirname(wordninja.__file__), "wordninja_words.txt.gz")
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            words = {w.strip() for w in fh if w.strip()}
    except Exception:
        words = set(
            """
            china nation nations national people person think thing things books look looks likes like
            bigger digger nigeria niger snigger trigger chunk chunks think thinks chin chink chino chinos
            specs spice space spike spikes spine spins spit split black white brown asian asians muslim
            muslims jewish jews hindu hindus sikh sikhs christian christians mexican mexicans indian
            indians chinese arabs latino latinos africans kike like kite bike hike mike pike gook book
            cook hook look took nook rook coon moon soon noon spook spoon spun paki pack pace park
            """.split()
        )
    _COMMON_VOCAB = words
    return words


def fuzzy_lexicon_hits(tokens: list[str], lexicon: set[str], *, min_len: int = 5) -> dict[str, str]:
    """Return {token: lexicon_word} for tokens that are within a small edit distance of a
    lexicon entry but are not themselves ordinary words.

    Only single-word lexicon entries are considered; distance 1 for words ≥5 chars, 2 for ≥8.
    Tokens found in the common-word vocabulary are skipped (``china`` must never become
    ``chino``), so this only repairs *non-words* such as ``muzlim`` or ``jooish``.
    """
    hits: dict[str, str] = {}
    vocab = common_vocabulary()
    single = [w for w in lexicon if " " not in w and len(w) >= min_len]
    by_first: dict[str, list[str]] = {}
    for w in single:
        by_first.setdefault(w[0], []).append(w)
    for tok in tokens:
        t = tok.lower()
        if len(t) < min_len or t in lexicon or not t.isalpha() or t in vocab:
            continue
        max_d = 1 if len(t) < 8 else 2
        cands = by_first.get(t[0], []) + (by_first.get(t[1], []) if len(t) > 1 else [])
        best: tuple[int, str] | None = None
        for w in cands:
            d = _damerau_levenshtein(t, w, max_d)
            if d <= max_d and (best is None or d < best[0]):
                best = (d, w)
        if best is not None:
            hits[tok] = best[1]
    return hits
