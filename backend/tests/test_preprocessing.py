"""Module 2 — preprocessing tests."""

from sentinai.preprocessing.dedup import Deduplicator
from sentinai.preprocessing.language import detect_language, transliterate_arabizi, transliterate_hinglish
from sentinai.preprocessing.normalize import normalize, split_hashtag
from sentinai.preprocessing.obfuscation import deobfuscate, fuzzy_lexicon_hits
from sentinai.preprocessing.pipeline import preprocess


def test_normalize_urls_mentions_hashtags_emojis():
    r = normalize("RT @SomeUser: #StopHate now! see https://t.co/abc123 @other 🐀")
    assert "https" not in r.text and "t.co" not in r.text
    assert "@user" in r.text and r.mentions_count == 1
    assert "stop hate" in r.text
    assert r.hashtags == ["stophate"]
    assert "rat" in r.text
    assert r.emojis == ["🐀"] and r.is_retweet_text


def test_split_hashtag_variants():
    assert split_hashtag("StopHate") == "stop hate"
    assert split_hashtag("stophate") == "stop hate"
    assert split_hashtag("Build_The_Wall") == "build the wall"
    assert split_hashtag("नफरत") == "नफरत"


def test_leetspeak_and_spaced_out_and_zero_width():
    r = deobfuscate("r4c1st k i l l k\u200bill n1gg3r")
    assert "racist" in r.text and "kill" in r.text
    assert r.zero_width_chars == 1
    assert r.leetspeak and r.spaced_out_words
    assert r.score > 0


def test_leet_leaves_identifiers_alone():
    r = deobfuscate("covid19 mp3 h1b visa 2024")
    assert "covid19" in r.text and "mp3" in r.text and "h1b" in r.text


def test_homoglyphs_repaired_only_in_latin_words():
    r = deobfuscate("rаcist привет")  # Cyrillic а in racist
    assert "racist" in r.text and "привет" in r.text
    assert r.homoglyphs == ["rаcist"]


def test_fuzzy_lexicon_hits_skips_common_words():
    lex = {"muslim", "chino", "nigger"}
    hits = fuzzy_lexicon_hits(["muzlim", "china", "nations", "muslim"], lex)
    assert hits == {"muzlim": "muslim"}


def test_language_detection_scripts_and_profiles():
    assert detect_language("اقتلوا كل اليهود").lang == "ar"
    assert detect_language("यह सब गद्दार हैं").lang == "hi"
    assert detect_language("Todos los mexicanos son ladrones y no se puede confiar en ellos").lang == "es"
    assert detect_language("Les arabes sont tous des voleurs, je ne les aime pas").lang == "fr"
    assert detect_language("Ich hasse die Türken und sie gehören nicht zu uns").lang == "de"
    assert detect_language("Os haitianos são todos uns animais e não pertencem aqui").lang == "pt"
    r = detect_language("sare musalman deshdrohi hain, inko maar do bhai")
    assert r.lang == "hi" and r.romanized
    r = detect_language("ana bakrah kol el yahood wallah, 5alas")
    assert r.lang == "ar" and r.romanized


def test_code_switching_flag():
    r = detect_language("bro this movie was bahut accha yaar, kya scene tha")
    assert r.code_switched


def test_transliteration():
    assert "मुसलमान" in transliterate_hinglish("sab musalman")
    assert "اقتلوا" in transliterate_arabizi("o2tolo kol el yahood")
    assert "ع" in transliterate_arabizi("3ala")


def test_preprocess_pipeline_keeps_both_scripts_for_romanized():
    p = preprocess("sare musalman deshdrohi hain")
    assert p.language == "hi" and p.transliterated
    assert "musalman" in p.normalized and "मुसलमान" in p.normalized


def test_dedup_near_duplicates():
    d = Deduplicator(threshold=0.7)
    assert d.check("1", "these people don't belong here send them all back to where they came from") is None
    assert d.check("2", "these people don't belong here send them all back to where they came from!!!") == "1"
    assert d.check("3", "had a lovely weekend hiking with my family, the weather was perfect") is None
