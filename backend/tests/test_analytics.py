"""Modules 8–12 & 14 — analytics, bots, fairness, account scores, HITL (unit level)."""

from datetime import datetime, timedelta

import pandas as pd

from sentinai.analytics import accounts, bots, fairness, trends
from sentinai.analytics.network import analyze_network
from sentinai.hitl import agreement_report, build_retrain_batch, cohen_kappa, fleiss_kappa, next_items, queue_stats, submit_annotation
from sentinai.schemas import Annotation, ToxicityLabel
from sentinai.storage.db import session_scope
from sentinai.storage.models import AuthorRow


def test_zscore_spike_detection():
    values = [5.0] * 20 + [40.0]
    z = trends.zscores(values, window=14)
    assert z[-1] > 5 and max(z[:-1]) < 1


def test_forecast_methods():
    idx = pd.date_range(end=datetime.utcnow().date(), periods=60, freq="D")
    df = pd.DataFrame({"total": 50, "toxic": [10 + (i % 7) for i in range(60)], "avg_severity": 2.0}, index=idx)
    fc, method = trends.forecast(df, horizon=7)
    assert len(fc) == 7 and method in ("prophet", "holt-winters", "naive")
    assert all(f.yhat_lower <= f.yhat <= f.yhat_upper + 1e-9 for f in fc)


def test_trend_series_and_events(seeded_db):
    with session_scope() as s:
        df = trends.toxic_series(s, days=90)
        assert df["total"].sum() > 0
        pts = trends.trend_points(df)
        assert len(pts) >= 60
        assert any(p.spike for p in pts)  # demo corpus has event-driven spikes
        impacts = trends.event_impacts(s, df)
        assert len(impacts) >= 3 and any(e.lift > 0.5 for e in impacts)
        per_target = trends.toxic_series(s, days=90, target="religion:islam")
        assert per_target["toxic"].sum() > 0


def test_network_analysis(seeded_db):
    with session_scope() as s:
        payload = analyze_network(s, days=90, min_toxicity=0.5, max_nodes=100)
        assert payload.stats["nodes"] > 10 and payload.stats["edges"] > 10
        assert payload.stats["communities"] >= 2
        assert payload.nodes[0]["pagerank"] >= payload.nodes[-1]["pagerank"]
        assert payload.stats["top_amplifiers"]
        ids = {n["id"] for n in payload.nodes}
        assert all(l["source"] in ids and l["target"] in ids for l in payload.links)


def test_bot_scoring_separates_bots(seeded_db):
    with session_scope() as s:
        n = bots.score_all_authors(s)
        assert n > 0
        rows = s.query(AuthorRow).all()
        botlike = [a.bot_probability for a in rows if a.username[-5:].isdigit() and a.has_default_profile_image]
        humanlike = [a.bot_probability for a in rows if "_" in a.username]
        assert botlike and humanlike
        assert sum(botlike) / len(botlike) > sum(humanlike) / len(humanlike) + 0.3
        assert all(0 <= p <= 1 for p in botlike + humanlike)


def test_bot_features_shape():
    a = AuthorRow(id="x", username="bot12345", created_at=datetime.utcnow() - timedelta(days=5), followers_count=2, following_count=2000, tweet_count=20000, has_default_profile_image=True)
    now = datetime.utcnow()
    act = bots.AuthorActivity(author=a, timestamps=[now - timedelta(hours=4 * i) for i in range(30)], texts=["same text again"] * 30, retweets=25, replies=0)
    f = bots.compute_features(act)
    assert set(f) == set(bots.FEATURE_NAMES)
    p, model, top = bots.score_features(f)
    assert p > 0.8 and top


def test_account_score_formula_and_ci(seeded_db):
    with session_scope() as s:
        n = accounts.score_all_accounts(s)
        assert n > 0
        top = s.query(AuthorRow).filter(AuthorRow.account_score_json.is_not(None)).all()
        scored = [a.account_score_json for a in top if a.account_score_json.get("score") is not None]
        assert scored
        for sc in scored:
            comp = 0.4 * sc["avg_severity"] / 4 + 0.3 * sc["toxic_ratio"] + 0.2 * sc["target_diversity"] + 0.1 * sc["recency_weight"]
            # components are decay-weighted so allow tolerance on the recomputed composite
            assert abs(comp - sc["score"]) < 0.15
            assert sc["disclaimer"].startswith("Model-Estimated Probability")
            if sc["n_posts"] >= 2:
                assert sc["ci_low"] <= sc["score"] + 1e-6 and sc["ci_high"] >= sc["score"] - 1e-6
            assert sc["reliable"] == (sc["n_posts"] >= 50)


def test_fairness_evaluation_and_calibration(engine):
    rows = fairness.load_benchmark(fairness.bundled_benchmarks()[0])
    rep = fairness.evaluate(engine.score_text, rows, benchmark="smoke")
    groups = {g.group: g for g in rep.groups}
    assert {"sae", "aave", "hinglish", "arabizi", "chicano", "british"} <= set(groups)
    # benign dialect text must not be flagged more than the reference group
    assert groups["aave"].fpr <= groups["sae"].fpr + 0.1
    assert groups["hinglish"].fpr <= 0.2 and groups["arabizi"].fpr <= 0.2
    assert rep.fpr_gap <= 0.2
    assert all(0.25 <= t <= 0.95 for t in rep.calibrated_thresholds.values())


def test_dialect_tagger():
    assert fairness.tag_dialect("we finna be at the cookout all day, y'all pull up fr") == "aave"
    assert fairness.tag_dialect("yaar the traffic today was ekdum crazy bhai") == "hinglish"
    assert fairness.tag_dialect("yalla habibi 5alas wallah kteer") == "arabizi"
    assert fairness.tag_dialect("The quarterly report is due on Friday.") == "sae"


def test_kappa():
    assert cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == 1.0
    assert cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == 0.0
    k = fleiss_kappa([["a", "a", "b"], ["b", "b", "b"], ["a", "a", "a"], ["b", "a", "b"]])
    assert k is not None and -1 <= k <= 1


def test_hitl_queue_annotate_agreement_retrain(seeded_db):
    with session_scope() as s:
        stats = queue_stats(s)
        assert stats["pending"] > 0
        items = next_items(s, limit=30)
        assert items and items[0]["priority"] >= items[-1]["priority"]
        # two annotators label the same 25 posts
        for it in items[:25]:
            submit_annotation(s, Annotation(post_id=it["post_id"], annotator="alice", toxicity_label=ToxicityLabel.HATE_SPEECH))
            submit_annotation(s, Annotation(post_id=it["post_id"], annotator="bob", toxicity_label=ToxicityLabel.HATE_SPEECH if it["final_toxicity"] > 0.5 else ToxicityLabel.NON_TOXIC))
        s.flush()
        rep = agreement_report(s)
        assert rep["posts_with_multiple_annotations"] >= 25 and rep["pairwise_cohen_kappa"]
        assert queue_stats(s)["resolved"] >= 25
        batch = build_retrain_batch(s, min_examples=10)
        assert batch is not None and batch.n_examples >= 25
        assert queue_stats(s)["pending_for_retraining"] == 0
