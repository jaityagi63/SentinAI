"""Text normalisation: lowercasing, URL removal, @mention anonymisation, hashtag splitting,
emoji → text description.

Only the standard library is *required*; `emoji` and `wordninja` are used when installed and
gracefully replaced by built-in fallbacks otherwise.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

try:  # optional, preferred
    import emoji as _emoji_lib
except Exception:  # pragma: no cover - optional dependency
    _emoji_lib = None

try:  # optional, preferred for hashtag splitting
    import wordninja as _wordninja
except Exception:  # pragma: no cover - optional dependency
    _wordninja = None


URL_RE = re.compile(r"(?:https?://|www\.)\S+|\b[a-z0-9.-]+\.(?:com|org|net|io|co|ly|gl|me|tv)\b/?\S*", re.I)
MENTION_RE = re.compile(r"(?<![\w@])@([A-Za-z0-9_]{1,15})\b")
HASHTAG_RE = re.compile(r"(?<!\w)#([\w\u0900-\u097F\u0600-\u06FF]+)")
RT_PREFIX_RE = re.compile(r"^\s*rt\s+@\w+:?\s*", re.I)
MULTISPACE_RE = re.compile(r"\s+")
HTML_ENTITY_RE = re.compile(r"&(amp|lt|gt|quot|#39|nbsp);")
CAMEL_RE = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])")

_HTML_ENTITIES = {"amp": "&", "lt": "<", "gt": ">", "quot": '"', "#39": "'", "nbsp": " "}

# Small built-in emoji dictionary used when the `emoji` package is unavailable.  The
# production configuration installs `emoji`, which covers the full Unicode set.
_FALLBACK_EMOJI = {
    "😂": "face with tears of joy",
    "🤣": "rolling on the floor laughing",
    "😡": "enraged face",
    "🤬": "face with symbols on mouth",
    "🐀": "rat",
    "🐒": "monkey",
    "🐵": "monkey face",
    "🦍": "gorilla",
    "🐷": "pig face",
    "🐖": "pig",
    "🐶": "dog face",
    "🐕": "dog",
    "🪳": "cockroach",
    "🔫": "pistol",
    "🔪": "kitchen knife",
    "💣": "bomb",
    "💀": "skull",
    "☠️": "skull and crossbones",
    "☠": "skull and crossbones",
    "🔥": "fire",
    "🚫": "prohibited",
    "⛔": "no entry",
    "🗑️": "wastebasket",
    "🗑": "wastebasket",
    "💩": "pile of poo",
    "🤮": "face vomiting",
    "🤢": "nauseated face",
    "👊": "oncoming fist",
    "✊": "raised fist",
    "🖕": "middle finger",
    "❤️": "red heart",
    "❤": "red heart",
    "🙏": "folded hands",
    "🤝": "handshake",
    "✌️": "victory hand",
    "🕊️": "dove",
    "🕊": "dove",
    "🌍": "globe showing europe-africa",
    "🇺🇸": "flag united states",
    "🇮🇳": "flag india",
    "🇵🇰": "flag pakistan",
    "🇲🇽": "flag mexico",
    "🇨🇳": "flag china",
    "🇮🇱": "flag israel",
    "🇵🇸": "flag palestine",
    "🇸🇾": "flag syria",
    "🇳🇬": "flag nigeria",
    "🇫🇷": "flag france",
    "🇩🇪": "flag germany",
    "🇬🇧": "flag united kingdom",
    "🇧🇷": "flag brazil",
    "🇪🇸": "flag spain",
    "🇸🇦": "flag saudi arabia",
    "🇪🇬": "flag egypt",
    "🇺🇦": "flag ukraine",
    "🇷🇺": "flag russia",
    "🇹🇷": "flag turkey",
    "☪️": "star and crescent",
    "☪": "star and crescent",
    "✡️": "star of david",
    "✡": "star of david",
    "✝️": "latin cross",
    "✝": "latin cross",
    "🕉️": "om",
    "🕉": "om",
    "☸️": "wheel of dharma",
    "🪯": "khanda",
    "🧕": "woman with headscarf",
    "👳": "person wearing turban",
    "👳‍♂️": "man wearing turban",
    "🐐": "goat",
    "🐪": "camel",
    "🐫": "two-hump camel",
    "🍉": "watermelon",
    "🍗": "poultry leg",
    "🐔": "chicken",
    "🌮": "taco",
    "🍛": "curry rice",
    "🍚": "cooked rice",
    "🥷": "ninja",
    "🧿": "nazar amulet",
    "⚡": "high voltage",
    "⚡⚡": "high voltage high voltage",
    "🔯": "dotted six-pointed star",
    "🪖": "military helmet",
    "🪓": "axe",
    "🗡️": "dagger",
    "🗡": "dagger",
    "⚔️": "crossed swords",
    "⚔": "crossed swords",
    "🩸": "drop of blood",
    "🥛": "glass of milk",
    "👌": "ok hand",
    "🐸": "frog",
    "🚿": "shower",
    "🚂": "locomotive",
    "🔢": "input numbers",
    "1️⃣4️⃣8️⃣8️⃣": "keycap 1 keycap 4 keycap 8 keycap 8",
    "🤡": "clown face",
    "🙄": "face with rolling eyes",
    "😷": "face with medical mask",
    "🦠": "microbe",
    "🐍": "snake",
    "🦟": "mosquito",
    "🪰": "fly",
    "🐛": "bug",
    "🐜": "ant",
    "🦂": "scorpion",
    "🕷️": "spider",
    "🕷": "spider",
}

# Unicode ranges that identify emoji when the `emoji` package is unavailable.
_EMOJI_RANGE_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"  # symbols & pictographs, transport, supplemental, extended-A
    "\U00002600-\U000027BF"  # misc symbols, dingbats
    "\U0001F1E6-\U0001F1FF"  # regional indicators (flags)
    "\U0001F900-\U0001F9FF"
    "\U00002B00-\U00002BFF"
    "\U0000FE0F\U0000200D\U000020E3"
    "]+",
    re.UNICODE,
)


@dataclass
class NormalizationResult:
    text: str
    emojis: list[str] = field(default_factory=list)
    emoji_descriptions: list[str] = field(default_factory=list)
    hashtags: list[str] = field(default_factory=list)
    mentions_count: int = 0
    urls_count: int = 0
    is_retweet_text: bool = False


def strip_control_chars(text: str) -> str:
    """Remove control characters (keeping whitespace)."""
    return "".join(ch for ch in text if ch in "\n\t " or unicodedata.category(ch)[0] != "C")


def unescape_html(text: str) -> str:
    return HTML_ENTITY_RE.sub(lambda m: _HTML_ENTITIES[m.group(1)], text)


def extract_emojis(text: str) -> list[str]:
    if _emoji_lib is not None:
        return [e["emoji"] for e in _emoji_lib.emoji_list(text)]
    found: list[str] = []
    for m in _EMOJI_RANGE_RE.finditer(text):
        seq = m.group(0)
        # split flag pairs / keycap sequences roughly per grapheme
        i = 0
        while i < len(seq):
            ch = seq[i]
            # regional indicator pair → flag
            if "\U0001F1E6" <= ch <= "\U0001F1FF" and i + 1 < len(seq):
                found.append(seq[i : i + 2])
                i += 2
                continue
            cluster = ch
            j = i + 1
            while j < len(seq) and seq[j] in "\ufe0f\u200d\u20e3":
                cluster += seq[j]
                if seq[j] == "\u200d" and j + 1 < len(seq):
                    cluster += seq[j + 1]
                    j += 1
                j += 1
            found.append(cluster)
            i = j
    return found


def emoji_to_text(em: str) -> str:
    if _emoji_lib is not None:
        desc = _emoji_lib.demojize(em, delimiters=(" ", " ")).strip()
        return desc.replace("_", " ").replace(":", "")
    base = em.replace("\ufe0f", "")
    if em in _FALLBACK_EMOJI:
        return _FALLBACK_EMOJI[em]
    if base in _FALLBACK_EMOJI:
        return _FALLBACK_EMOJI[base]
    try:
        return " ".join(unicodedata.name(ch).lower() for ch in base if ch not in "\u200d\u20e3")
    except ValueError:
        return "emoji"


def demojize(text: str) -> tuple[str, list[str], list[str]]:
    """Replace every emoji by ` <description> `; return (text, emojis, descriptions)."""
    emojis = extract_emojis(text)
    descriptions: list[str] = []
    out = text
    # Replace longest sequences first to avoid partial overlaps.
    for em in sorted(set(emojis), key=len, reverse=True):
        desc = emoji_to_text(em)
        out = out.replace(em, f" {desc} ")
    for em in emojis:
        descriptions.append(emoji_to_text(em))
    return out, emojis, descriptions


_COMMON_WORDS: set[str] | None = None


def _common_words() -> set[str]:
    """Tiny built-in English vocabulary for hashtag splitting fallback (wordninja preferred)."""
    global _COMMON_WORDS
    if _COMMON_WORDS is None:
        _COMMON_WORDS = set(
            """
            a an the and or not no yes stop end ban kill hate love all our we us them they you me my your
            go back home out in on off up down now never always again first last new old good bad big small
            white black brown asian arab jew jews muslim muslims hindu hindus sikh christian islam judaism
            immigrant immigrants migrant migrants refugee refugees border wall build deport send them all
            america american americans mexico mexican mexicans china chinese india indian indians pakistan
            free speech peace war fight fighting truth lies lie fake news real great make again save protect
            defend rights right left matter lives live life stand with against for of to from by at is are
            be been being was were do does did done have has had not just only every any some more most
            people person man men woman women child children kids family nation country world city state
            justice equality freedom unity together solidarity power pride proud strong weak vote election
            terror terrorist terrorism attack invasion invaders crime criminal criminals police law order
            """.split()
        )
    return _COMMON_WORDS


def split_hashtag(tag: str) -> str:
    """#StopHate → 'stop hate'; #stophate → 'stop hate' (wordninja or greedy fallback)."""
    tag = tag.lstrip("#")
    if not tag:
        return ""
    # Non-Latin scripts: leave intact.
    if any(ord(ch) > 0x24F for ch in tag):
        return tag
    if "_" in tag:
        return " ".join(split_hashtag(part) for part in tag.split("_") if part)
    # CamelCase / digits boundaries first
    parts = [p for p in CAMEL_RE.split(tag) if p]
    if len(parts) > 1 and not tag.isupper():
        return " ".join(p.lower() for p in parts)
    lower = tag.lower()
    if _wordninja is not None:
        words = _wordninja.split(lower)
        if words and all(len(w) > 1 or w in {"a", "i"} for w in words):
            return " ".join(words)
        return lower
    return _greedy_split(lower)


def _greedy_split(word: str) -> str:
    vocab = _common_words()
    if word in vocab or len(word) <= 3:
        return word
    n = len(word)
    # dynamic programming: minimise number of non-vocab characters
    best: list[tuple[int, list[str]] | None] = [None] * (n + 1)
    best[0] = (0, [])
    for i in range(1, n + 1):
        cand: tuple[int, list[str]] | None = None
        for j in range(max(0, i - 15), i):
            prev = best[j]
            if prev is None:
                continue
            piece = word[j:i]
            cost = prev[0] + (0 if piece in vocab else len(piece))
            if cand is None or cost < cand[0] or (cost == cand[0] and len(prev[1]) + 1 < len(cand[1])):
                cand = (cost, prev[1] + [piece])
        best[i] = cand
    result = best[n]
    if result is None or result[0] > n * 0.5:
        return word
    # merge consecutive unknown fragments
    merged: list[str] = []
    for piece in result[1]:
        if merged and merged[-1] not in vocab and piece not in vocab:
            merged[-1] += piece
        else:
            merged.append(piece)
    return " ".join(merged)


def normalize(text: str, *, anonymize_mentions: bool = True) -> NormalizationResult:
    """Full normalisation pass.

    Returns the cleaned lower-cased text (URLs removed, mentions replaced by ``@user``, hashtags
    split, emojis described) together with the extracted artefacts.
    """
    raw = unescape_html(strip_control_chars(text or ""))
    raw = unicodedata.normalize("NFKC", raw)

    is_rt = bool(RT_PREFIX_RE.match(raw))
    raw = RT_PREFIX_RE.sub("", raw)

    urls = URL_RE.findall(raw)
    raw = URL_RE.sub(" ", raw)

    mentions = MENTION_RE.findall(raw)
    if anonymize_mentions:
        raw = MENTION_RE.sub("@user", raw)

    hashtags = HASHTAG_RE.findall(raw)
    raw = HASHTAG_RE.sub(lambda m: " " + split_hashtag(m.group(1)) + " ", raw)

    raw, emojis, descriptions = demojize(raw)

    cleaned = MULTISPACE_RE.sub(" ", raw).strip().lower()
    return NormalizationResult(
        text=cleaned,
        emojis=emojis,
        emoji_descriptions=descriptions,
        hashtags=[h.lower() for h in hashtags],
        mentions_count=len(mentions),
        urls_count=len(urls),
        is_retweet_text=is_rt,
    )


def tokenize(text: str) -> list[str]:
    """Simple unicode-aware word tokenizer (keeps apostrophes inside words)."""
    return re.findall(r"[\w'’]+|[^\w\s]", text, re.UNICODE)
