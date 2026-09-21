"""Language identification & transliteration (Modules 2 and 7).

* Primary: FastText ``lid.176`` (loaded when ``SENTINAI_FASTTEXT_LID_PATH`` points at the
  ``.bin``/``.ftz`` file and the ``fasttext`` package is installed).
* Fallback: script analysis + function-word profiles for the Phase-1 languages
  (en, es, ar, hi, fr, pt, de) and a Romanized Hindi / Arabizi detector for code-switching.
* Transliteration: Romanized Hindi → Devanagari and Arabizi → Arabic script for the
  vocabulary needed by the classifiers (``indic_transliteration`` is used when present; a
  built-in lookup covers the frequent forms so tests do not need the dependency).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from sentinai.config import get_settings

SUPPORTED_LANGUAGES = ("en", "es", "ar", "hi", "fr", "pt", "de")

_FUNCTION_WORDS: dict[str, set[str]] = {
    "en": set("the and is are of to in that it you for with this not on they we be have was all but".split()),
    "es": set("el la los las de que y en es un una por con no para se su al lo como más pero son del".split()),
    "fr": set("le la les de des et est un une que pas pour dans ce il elle je nous vous sont au aux du en".split()),
    "pt": set("o a os as de que e em um uma não para com se por mais do da dos das são ao é você".split()),
    "de": set("der die das und ist nicht ein eine zu den von mit sich auf für es sie ich wir ihr sind dem".split()),
    # Romanized Hindi (Hinglish) function words
    "hi-latn": set(
        "hai hain ka ki ke ko se me mein aur nahi nahin kya yeh ye woh wo par bhi hi to tha thi the ho hum tum "
        "unko inko unke inke sab kuch koi kyun kyon kaise kab kahan jo jab tab agar lekin magar phir fir sirf "
        "bahut bohot bhai yaar yar abhi kabhi sabhi apna apni apne mera meri mere tera teri tere hamara".split()
    ),
    # Arabizi function words
    "ar-latn": set(
        "ana enta enti howa heya e7na ento homa fi fe mesh mish msh la2 la 3ala 3la 3an men min ma3 wala wla "
        "leh lesh keef kif shu eh ezay ezzay ya walla wallah yalla 5alas khalas kteer ktir bas bs kol kul "
        "7abibi habibi el al ال illi elli lly 3shan 3ashan 3lshan dh da di dah".split()
    ),
}

_ARABIZI_DIGIT_RE = re.compile(r"[a-z][2357689][a-z]|\b[2357689][a-z]{2,}|[a-z]{2,}[2357689]\b", re.I)


@dataclass
class LanguageResult:
    lang: str
    confidence: float
    script: str
    romanized: bool = False  # Latin script text of a non-Latin language (Hinglish / Arabizi)
    code_switched: bool = False
    candidates: dict[str, float] | None = None


def detect_script(text: str) -> str:
    counts = {"latin": 0, "arabic": 0, "devanagari": 0, "cyrillic": 0, "cjk": 0, "other": 0}
    for ch in text:
        cp = ord(ch)
        if ch.isspace() or not ch.isalpha():
            continue
        if cp < 0x250:
            counts["latin"] += 1
        elif 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F or 0xFB50 <= cp <= 0xFEFF:
            counts["arabic"] += 1
        elif 0x0900 <= cp <= 0x097F:
            counts["devanagari"] += 1
        elif 0x0400 <= cp <= 0x04FF:
            counts["cyrillic"] += 1
        elif 0x4E00 <= cp <= 0x9FFF or 0x3040 <= cp <= 0x30FF or 0xAC00 <= cp <= 0xD7AF:
            counts["cjk"] += 1
        else:
            counts["other"] += 1
    if sum(counts.values()) == 0:
        return "unknown"
    return max(counts, key=counts.get)  # type: ignore[arg-type]


def _script_mix(text: str) -> dict[str, float]:
    total = 0
    counts: dict[str, int] = {}
    for ch in text:
        if not ch.isalpha():
            continue
        total += 1
        s = detect_script(ch)
        counts[s] = counts.get(s, 0) + 1
    return {k: v / total for k, v in counts.items()} if total else {}


class _FastTextLID:
    def __init__(self, path: Path):
        import fasttext  # type: ignore

        self.model = fasttext.load_model(str(path))

    def predict(self, text: str, k: int = 3) -> dict[str, float]:
        labels, probs = self.model.predict(text.replace("\n", " "), k=k)
        return {lab.replace("__label__", ""): float(p) for lab, p in zip(labels, probs, strict=False)}


@lru_cache
def _load_fasttext() -> _FastTextLID | None:
    settings = get_settings()
    path = settings.fasttext_lid_path
    if path and Path(path).exists():
        try:
            return _FastTextLID(Path(path))
        except Exception:  # pragma: no cover - environment specific
            return None
    return None


def _profile_scores(tokens: list[str]) -> dict[str, float]:
    scores: dict[str, float] = {}
    n = max(1, len(tokens))
    for lang, words in _FUNCTION_WORDS.items():
        hits = sum(1 for t in tokens if t in words)
        scores[lang] = hits / n
    return scores


def detect_language(text: str) -> LanguageResult:
    """Detect language with FastText when available, otherwise script + function-word profile."""
    script = detect_script(text)
    mix = _script_mix(text)
    code_switched = len([k for k, v in mix.items() if v > 0.15]) > 1

    ft = _load_fasttext()
    if ft is not None:
        cands = ft.predict(text)
        if cands:
            lang, conf = next(iter(cands.items()))
            romanized = script == "latin" and lang in ("hi", "ar", "ur")
            return LanguageResult(lang, conf, script, romanized, code_switched, cands)

    if script == "arabic":
        return LanguageResult("ar", 0.95, script, False, code_switched)
    if script == "devanagari":
        return LanguageResult("hi", 0.95, script, False, code_switched)
    if script == "cyrillic":
        return LanguageResult("ru", 0.8, script, False, code_switched)
    if script == "cjk":
        return LanguageResult("zh", 0.7, script, False, code_switched)
    if script == "unknown":
        return LanguageResult("und", 0.0, script, False, False)

    tokens = re.findall(r"[a-zà-ÿ0-9']+", text.lower())
    scores = _profile_scores(tokens)
    arabizi_digits = len(_ARABIZI_DIGIT_RE.findall(text))
    if arabizi_digits:
        scores["ar-latn"] += 0.15 * arabizi_digits / max(1, len(tokens))

    # Diacritics hints
    joined = " ".join(tokens)
    if re.search(r"[ñ¿¡]", joined):
        scores["es"] += 0.1
    if re.search(r"[ãõç]", joined):
        scores["pt"] += 0.1
    if re.search(r"[éèêàçùœ]", joined) and "'" in text:
        scores["fr"] += 0.05
    if re.search(r"[äöüß]", joined):
        scores["de"] += 0.1

    best = max(scores, key=scores.get)  # type: ignore[arg-type]
    best_score = scores[best]
    if best_score == 0:
        return LanguageResult("en", 0.3, script, False, code_switched, scores)

    sorted_scores = sorted(scores.values(), reverse=True)
    margin = sorted_scores[0] - (sorted_scores[1] if len(sorted_scores) > 1 else 0)
    conf = min(0.99, 0.5 + best_score + margin)
    # A Latin-script post with both English and Hinglish/Arabizi function words is code-switched.
    if best in ("hi-latn", "ar-latn"):
        if scores["en"] > 0.08:
            code_switched = True
        return LanguageResult(best.split("-")[0], conf, script, True, code_switched, scores)
    if best == "en" and (scores["hi-latn"] > 0.08 or scores["ar-latn"] > 0.08):
        code_switched = True
    return LanguageResult(best, conf, script, False, code_switched, scores)


# --------------------------------------------------------------------------------------------
# Transliteration (Romanized Hindi → Devanagari, Arabizi → Arabic)
# --------------------------------------------------------------------------------------------

# Frequent Hinglish tokens relevant to the domain → Devanagari.  The generic transliterator
# (indic_transliteration ITRANS) handles the long tail when installed.
HINGLISH_LEXICON: dict[str, str] = {
    "bhai": "भाई", "yaar": "यार", "musalman": "मुसलमान", "mussalman": "मुसलमान", "musalmaan": "मुसलमान",
    "hindu": "हिंदू", "hinduon": "हिंदुओं", "sikh": "सिख", "isai": "ईसाई", "yahudi": "यहूदी",
    "nafrat": "नफरत", "maro": "मारो", "maaro": "मारो", "maar": "मार", "mar": "मर", "do": "दो", "dalo": "डालो",
    "jalao": "जलाओ", "jala": "जला", "kaato": "काटो", "kaat": "काट", "bhagao": "भगाओ", "nikalo": "निकालो",
    "desh": "देश", "deshdrohi": "देशद्रोही", "gaddar": "गद्दार", "ghuspaithiye": "घुसपैठिये",
    "kutte": "कुत्ते", "kutta": "कुत्ता", "kamine": "कमीने", "harami": "हरामी", "chor": "चोर",
    "gande": "गंदे", "ganda": "गंदा", "log": "लोग", "logon": "लोगों", "sab": "सब", "saare": "सारे",
    "ye": "ये", "yeh": "यह", "wo": "वो", "woh": "वह", "hai": "है", "hain": "हैं", "nahi": "नहीं",
    "kyun": "क्यों", "kya": "क्या", "aur": "और", "ko": "को", "ka": "का", "ki": "की", "ke": "के",
    "se": "से", "mein": "में", "me": "में", "par": "पर", "bhi": "भी", "hi": "ही", "to": "तो",
    "dharm": "धर्म", "mandir": "मंदिर", "masjid": "मस्जिद", "gurudwara": "गुरुद्वारा", "girja": "गिरजा",
    "katua": "कटुआ", "katue": "कटुए", "mulle": "मुल्ले", "sulle": "सुल्ले", "bhakt": "भक्त", "sanghi": "संघी",
    "insaan": "इंसान", "insaniyat": "इंसानियत", "sharam": "शर्म", "galat": "गलत", "sahi": "सही",
    "goli": "गोली", "bomb": "बम", "hamla": "हमला", "jung": "जंग", "ladai": "लड़ाई", "khoon": "खून",
    "chuha": "चूहा", "chuhe": "चूहे", "deemak": "दीमक", "keede": "कीड़े", "janwar": "जानवर", "kachra": "कचरा",
    "pakistani": "पाकिस्तानी", "bangladeshi": "बांग्लादेशी", "chini": "चीनी", "angrez": "अंग्रेज",
}

# Arabizi digits → Arabic letters, plus frequent domain tokens.
ARABIZI_DIGITS: dict[str, str] = {"2": "ء", "3": "ع", "5": "خ", "6": "ط", "7": "ح", "8": "غ", "9": "ق"}
ARABIZI_LEXICON: dict[str, str] = {
    "ana": "أنا", "enta": "أنت", "howa": "هو", "heya": "هي", "e7na": "نحن", "homa": "هم",
    "kol": "كل", "kul": "كل", "kull": "كل", "el": "ال", "al": "ال", "fi": "في", "fe": "في", "mesh": "مش",
    "mish": "مش", "la2": "لأ", "3ala": "على", "3la": "على", "men": "من", "min": "من", "ma3": "مع",
    "yahood": "يهود", "yahud": "يahود", "el yahood": "اليهود", "masee7yeen": "مسيحيين", "masi7iyin": "مسيحيين",
    "muslimeen": "مسلمين", "muslimin": "مسلمين", "3arab": "عرب", "el 3arab": "العرب", "sood": "سود",
    "el sood": "السود", "sorieen": "سوريين", "soriyin": "سوريين", "laj2een": "لاجئين", "laje2in": "لاجئين",
    "o2tolo": "اقتلوا", "oqtolo": "اقتلوا", "uqtulu": "اقتلوا", "edb7o": "اذبحوا", "edba7o": "اذبحوا",
    "7er2o": "احرقوا", "e3dmo": "اعدموا", "abido": "أبيدوا", "atrodo": "اطردوا", "otrodo": "اطردوا",
    "el mot": "الموت", "el moot": "الموت", "almawt": "الموت", "yes2ot": "يسقط", "yasqut": "يسقط",
    "kilab": "كلاب", "klab": "كلاب", "5anazir": "خنازير", "khanazir": "خنازير", "hasharat": "حشرات",
    "sarasir": "صراصير", "7ayawanat": "حيوانات", "zbala": "زبالة", "wse5": "وسخ", "wsa5a": "وساخة",
    "bakrah": "بكره", "akrah": "أكره", "nekrah": "نكره", "3onsoreya": "عنصرية", "3onsori": "عنصري",
    "kraheya": "كراهية", "karahiya": "كراهية", "tamyeez": "تمييز", "3eeb": "عيب", "7aram": "حرام",
    "wallah": "والله", "yalla": "يلا", "5alas": "خلاص", "khalas": "خلاص", "kteer": "كتير", "bas": "بس",
}


@lru_cache
def _indic_transliterator():
    try:
        from indic_transliteration import sanscript  # type: ignore

        return sanscript
    except Exception:
        return None


def transliterate_hinglish(text: str) -> str:
    """Romanized Hindi → Devanagari (lexicon first, ITRANS engine for the rest when available)."""
    sanscript = _indic_transliterator()
    out: list[str] = []
    for tok in text.split():
        low = tok.lower()
        if low in HINGLISH_LEXICON:
            out.append(HINGLISH_LEXICON[low])
        elif sanscript is not None and low.isalpha() and len(low) > 2:
            try:
                out.append(sanscript.transliterate(low, sanscript.ITRANS, sanscript.DEVANAGARI))
            except Exception:  # pragma: no cover
                out.append(tok)
        else:
            out.append(tok)
    return " ".join(out)


_ARTICLE_TOKENS = {"el", "al", "il", "l"}
# Small Arabizi→Arabic consonant/vowel map used for tokens not covered by the lexicon.
_ARABIZI_CHARS: dict[str, str] = {
    "2": "ء", "3": "ع", "5": "خ", "6": "ط", "7": "ح", "8": "غ", "9": "ق",
    "a": "ا", "b": "ب", "d": "د", "e": "", "f": "ف", "g": "ج", "h": "ه", "i": "ي", "j": "ج", "k": "ك", "l": "ل",
    "m": "م", "n": "ن", "o": "و", "q": "ق", "r": "ر", "s": "س", "t": "ت", "u": "و", "w": "و", "y": "ي", "z": "ز",
}
_ARABIZI_DIGRAPHS: dict[str, str] = {"sh": "ش", "ch": "ش", "th": "ث", "kh": "خ", "gh": "غ", "dh": "ذ", "oo": "و", "ee": "ي", "aa": "ا"}


def _arabizi_token(low: str) -> str:
    if low in ARABIZI_LEXICON:
        return ARABIZI_LEXICON[low]
    i = 0
    out = []
    while i < len(low):
        pair = low[i : i + 2]
        if pair in _ARABIZI_DIGRAPHS:
            out.append(_ARABIZI_DIGRAPHS[pair])
            i += 2
            continue
        ch = low[i]
        # drop doubled letters (yahood -> yahod)
        if i > 0 and ch == low[i - 1] and ch not in "aeiou":
            i += 1
            continue
        out.append(_ARABIZI_CHARS.get(ch, ch))
        i += 1
    return "".join(out)


def transliterate_arabizi(text: str) -> str:
    """Arabizi (Latin + digits) → Arabic script.

    Lexicon first; other tokens that contain Arabizi digits or follow the definite article are
    transliterated character-wise.  The article (``el`` / ``al``) is attached to the following
    word so that ``el yahood`` → ``اليهود`` matches the script lexicon.
    """
    toks = text.split()
    out: list[str] = []
    i = 0
    while i < len(toks):
        low = toks[i].lower()
        if low in _ARTICLE_TOKENS and i + 1 < len(toks):
            nxt = toks[i + 1].lower()
            out.append("ال" + _arabizi_token(nxt).lstrip("ا") if nxt not in ARABIZI_LEXICON else "ال" + ARABIZI_LEXICON[nxt].lstrip("ا"))
            i += 2
            continue
        if low in ARABIZI_LEXICON:
            out.append(ARABIZI_LEXICON[low])
        elif any(d in low for d in ARABIZI_DIGITS) and re.search(r"[a-z]", low):
            out.append(_arabizi_token(low))
        else:
            out.append(toks[i])
        i += 1
    return " ".join(out)


def transliterate_if_needed(text: str, lang_result: LanguageResult) -> tuple[str, bool]:
    """Return (possibly transliterated text, changed?).  Only for Romanized hi / ar."""
    if not lang_result.romanized:
        return text, False
    if lang_result.lang == "hi":
        return transliterate_hinglish(text), True
    if lang_result.lang == "ar":
        return transliterate_arabizi(text), True
    return text, False
