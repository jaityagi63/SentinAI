"""Modules 3–7 — classification, explainability, multimodal, context, multilingual."""

from datetime import datetime

import pytest

from sentinai.classification.context import ParentInfo
from sentinai.schemas import MediaAttachment, Post, ReferencedPost, SeverityLevel, Stance, ToxicityLabel


def labels(engine, text):
    c = engine.classify_text(text)
    return c.toxicity.label, int(c.severity.level), c


@pytest.mark.parametrize(
    "text",
    [
        "Had a lovely weekend hiking with my family, the weather was perfect!",
        "China announced new tariffs today, markets reacted calmly.",
        "The Ramadan iftar at the community center was packed, amazing food from everyone.",
        "I'd kill for a coffee right now",
        "We need to get rid of the old carpets before the party",
        "Research shows hate speech against Asian people rose sharply in 2020 — new paper out today.",
    ],
)
def test_benign(engine, text):
    label, sev, c = labels(engine, text)
    assert label == ToxicityLabel.NON_TOXIC and c.final_toxicity < 0.4, (text, c.explanation.matched_patterns)


@pytest.mark.parametrize(
    "text,expected_sev",
    [
        ("Muslims are so good at running corner shops lol", 1),
        ("jews control everything and need to be dealt with once and for all", 2),
        ("These people don't belong here. Send them all back to where they came from", 2),
        ("immigrants are rats infesting our cities", 2),
        ("filthy muslims are a cancer on this country", 3),
        ("shut up you stupid paki", 3),
        ("k1ll all the j3ws", 4),
        ("We should attack every mosque in the city, time to take up arms", 4),
        ("death to arabs, burn their homes down", 4),
    ],
)
def test_severity_taxonomy(engine, text, expected_sev):
    label, sev, c = labels(engine, text)
    assert sev == expected_sev, (text, c.severity, c.explanation.matched_patterns)
    assert label != ToxicityLabel.NON_TOXIC or expected_sev == 1


def test_task_a_labels(engine):
    assert labels(engine, "you absolute idiot shut up")[0] == ToxicityLabel.OFFENSIVE
    assert labels(engine, "filthy muslims are a cancer on this country")[0] == ToxicityLabel.HATE_SPEECH
    assert labels(engine, "kill all the jews")[0] == ToxicityLabel.VIOLENT_EXTREMISM


def test_task_b_targets(engine):
    _, _, c = labels(engine, "filthy muslims and dirty mexicans are ruining this country")
    keys = {(t.category.value, t.label) for t in c.targets}
    assert ("religion", "islam") in keys and ("nationality", "mexican") in keys
    _, _, c2 = labels(engine, "The Ramadan iftar at the community center was packed")
    assert c2.targets == []  # no target when non-toxic


def test_task_c_probabilities_sum_to_one(engine):
    _, _, c = labels(engine, "immigrants are rats infesting our cities")
    assert abs(sum(c.severity.probabilities.values()) - 1) < 1e-6
    assert abs(sum(c.toxicity.probabilities.values()) - 1) < 1e-6
    assert 0 <= c.severity.expected_level <= 4


@pytest.mark.parametrize(
    "text,lang,target",
    [
        ("Todos los mexicanos son ladrones, que se vayan de aquí", "es", ("nationality", "mexican")),
        ("Les arabes sont tous des voleurs, rentrez chez vous", "fr", ("ethnicity", "middle_eastern")),
        ("Odeio os haitianos, são todos uns animais", "pt", ("nationality", "haitian")),
        ("Ich hasse die Türken, raus aus Deutschland", "de", ("nationality", "turkish")),
        ("यह सब मुसलमान गद्दार हैं, इनको भगाओ", "hi", ("religion", "islam")),
        ("sare musalman deshdrohi hain, inko maar do", "hi", ("religion", "islam")),
        ("اقتلوا كل اليهود", "ar", ("religion", "judaism")),
        ("o2tolo kol el yahood", "ar", ("religion", "judaism")),
    ],
)
def test_multilingual_and_code_switching(engine, text, lang, target):
    _, _, c = labels(engine, text)
    assert c.language == lang
    assert c.final_toxicity >= 0.6, (text, c.explanation.matched_patterns)
    assert target in {(t.category.value, t.label) for t in c.targets}, c.targets


def test_obfuscation_still_detected(engine):
    _, sev, c = labels(engine, "k1ll all the j3ws")
    assert c.obfuscation_score > 0 and sev == 4


def test_explanation_attributions(engine):
    _, _, c = labels(engine, "filthy muslims are a cancer on this country")
    e = c.explanation
    assert e is not None and e.method == "lexicon-attribution"
    hot = {a.token for a in e.attributions if a.weight > 0.5}
    assert {"filthy", "cancer"} <= hot
    assert e.text == c.explanation.text
    assert all(-1 <= a.weight <= 1 for a in e.attributions)
    assert e.matched_patterns and e.matched_patterns[0]["category"] in ("harassment", "dehumanization")


def test_counterspeech_negative_attribution(engine):
    _, _, c = labels(engine, "This is blatantly racist. How is this account still up? Reported.")
    assert c.toxicity.label == ToxicityLabel.NON_TOXIC
    assert any(a.weight < 0 for a in c.explanation.attributions)


def test_context_quote_condemn_discount(engine):
    text = "This is blatantly racist, how is this still up? 'kill all the muslims' is incitement."
    post = Post(id="q1", text=text, created_at=datetime.utcnow(), author_id="a", referenced=[ReferencedPost(type="quoted", id="p1")])
    c = engine.classify_post(post, parent=ParentInfo(id="p1", text="kill all the muslims", toxicity=0.95))
    assert c.context.is_quote and not c.context.is_originator
    assert c.context.stance == Stance.CONDEMN
    assert c.context.discount_factor < 0.5
    assert c.final_toxicity < 0.2


def test_context_support_reply_no_discount(engine):
    post = Post(id="r1", text="exactly", created_at=datetime.utcnow(), author_id="a", referenced=[ReferencedPost(type="replied_to", id="p1")])
    c = engine.classify_post(post, parent=ParentInfo(id="p1", text="kill all the muslims", toxicity=0.95))
    assert c.context.stance == Stance.SUPPORT and c.context.discount_factor == 1.0


def test_context_originator(engine):
    c = engine.classify_text("filthy muslims are a cancer")
    assert c.context.is_originator and c.context.stance == Stance.NEUTRAL and c.context.discount_factor == 1.0


def test_multimodal_alt_text_ensemble(engine):
    post = Post(id="m1", text="lovely day", created_at=datetime.utcnow(), author_id="a", media=[MediaAttachment(media_key="k", alt_text="cartoon of muslims depicted as rats swarming a city, caption: INFESTATION")])
    c = engine.classify_post(post)
    assert c.visual is not None and c.visual.visual_toxicity >= 0.6
    assert c.visual.juxtaposition_flag
    assert c.final_toxicity >= 0.6 and c.toxicity.label == ToxicityLabel.HATE_SPEECH


def test_multimodal_hate_code_in_ocr(engine):
    post = Post(id="m2", text="new profile pic", created_at=datetime.utcnow(), author_id="a", media=[MediaAttachment(media_key="k", alt_text="image with text 1488 and a black sun symbol")])
    c = engine.classify_post(post)
    assert c.visual.visual_toxicity >= 0.6


def test_active_learning_band(engine):
    c = engine.classify_text("They're all good at math, of course he got the job")
    assert 0.4 <= c.final_toxicity <= 0.6 and c.needs_review


def test_severity_enum_names():
    assert SeverityLevel(4).name == "INCITEMENT"
