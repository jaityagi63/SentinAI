"""Module 12 — Aggregated account-level scoring.

    Score = 0.4·AvgSeverity + 0.3·ToxicRatio + 0.2·TargetDiversity + 0.1·RecencyWeight

* every component is normalised to [0, 1] (severity /4; diversity = distinct targets / 6 capped)
* posts are weighted by an exponential time decay (half-life ``account_decay_half_life_days``)
* minimum sample size (default 50 posts) before a score is considered *reliable*
* bootstrap 95 % confidence interval around the point estimate
* all outputs carry the "Model-Estimated Probability" disclaimer
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from sentinai.config import get_settings
from sentinai.schemas import AccountScore
from sentinai.storage.models import AuthorRow, ClassificationRow, PostRow

WEIGHTS = {"severity": 0.4, "toxic_ratio": 0.3, "target_diversity": 0.2, "recency": 0.1}
DISCLAIMER = "Model-Estimated Probability — not a definitive human judgment."


def _decay_weight(age_days: float, half_life: float) -> float:
    return 0.5 ** (age_days / half_life) if half_life > 0 else 1.0


def _components(posts: list[tuple[float, float, int, list[str]]], now: datetime, half_life: float, toxic_threshold: float = 0.5) -> dict[str, float]:
    """posts: list of (age_days, final_toxicity, severity_level, target_labels)."""
    if not posts:
        return {"avg_severity": 0.0, "toxic_ratio": 0.0, "target_diversity": 0.0, "recency_weight": 0.0, "score": 0.0}
    weights = [_decay_weight(age, half_life) for age, *_ in posts]
    wsum = sum(weights) or 1.0
    avg_sev = sum(w * sev for w, (_, _, sev, _) in zip(weights, posts, strict=True)) / wsum / 4.0
    toxic_ratio = sum(w * (1.0 if tox >= toxic_threshold else 0.0) for w, (_, tox, _, _) in zip(weights, posts, strict=True)) / wsum
    targets: set[str] = set()
    for (_, tox, _, labels) in posts:
        if tox >= toxic_threshold:
            targets.update(labels)
    diversity = min(1.0, len(targets) / 6.0)
    # recency weight: share of *toxic* activity that falls in the most recent 30 days
    recent_tox = sum(1 for age, tox, _, _ in posts if tox >= toxic_threshold and age <= 30)
    total_tox = sum(1 for _, tox, _, _ in posts if tox >= toxic_threshold)
    recency = recent_tox / total_tox if total_tox else 0.0
    score = WEIGHTS["severity"] * avg_sev + WEIGHTS["toxic_ratio"] * toxic_ratio + WEIGHTS["target_diversity"] * diversity + WEIGHTS["recency"] * recency
    return {"avg_severity": avg_sev, "toxic_ratio": toxic_ratio, "target_diversity": diversity, "recency_weight": recency, "score": score}


def bootstrap_ci(posts, now: datetime, half_life: float, n_boot: int = 300, seed: int = 7) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(posts)
    if n < 2:
        return 0.0, 1.0
    scores = []
    for _ in range(n_boot):
        sample = [posts[rng.randrange(n)] for _ in range(n)]
        scores.append(_components(sample, now, half_life)["score"])
    scores.sort()
    lo = scores[int(0.025 * n_boot)]
    hi = scores[min(n_boot - 1, int(0.975 * n_boot))]
    return round(lo, 4), round(hi, 4)


def score_account(session: Session, author_id: str, *, days: int = 365, now: datetime | None = None) -> AccountScore | None:
    settings = get_settings()
    now = now or datetime.utcnow()
    author = session.get(AuthorRow, author_id)
    if author is None:
        return None
    since = now - timedelta(days=days)
    rows = session.execute(
        select(PostRow.created_at, ClassificationRow.final_toxicity, ClassificationRow.severity_level, ClassificationRow.target_labels, ClassificationRow.duplicate_of)
        .join(ClassificationRow, ClassificationRow.post_id == PostRow.id)
        .where(PostRow.author_id == author_id, PostRow.created_at >= since, PostRow.deleted_upstream.is_(False))
    ).all()
    posts = [((now - ts).total_seconds() / 86400.0, tox, sev, labels or []) for ts, tox, sev, labels, _dup in rows]
    n = len(posts)
    reliable = n >= settings.account_min_posts
    comps = _components(posts, now, settings.account_decay_half_life_days)
    ci_lo, ci_hi = bootstrap_ci(posts, now, settings.account_decay_half_life_days) if n >= 2 else (None, None)
    result = AccountScore(
        author_id=author_id,
        username=author.username,
        n_posts=n,
        reliable=reliable,
        score=round(comps["score"], 4) if n else None,
        ci_low=ci_lo,
        ci_high=ci_hi,
        avg_severity=round(comps["avg_severity"] * 4, 3),
        toxic_ratio=round(comps["toxic_ratio"], 4),
        target_diversity=round(comps["target_diversity"], 4),
        recency_weight=round(comps["recency_weight"], 4),
        bot_probability=author.bot_probability,
        disclaimer=DISCLAIMER,
    )
    author.account_score = result.score if reliable else None
    author.account_score_json = result.model_dump(mode="json")
    return result


def score_all_accounts(session: Session, min_posts: int = 1) -> int:
    ids = session.scalars(select(AuthorRow.id)).all()
    n = 0
    for aid in ids:
        r = score_account(session, aid)
        if r and r.n_posts >= min_posts:
            n += 1
    return n


def confidence_label(score: AccountScore) -> str:
    if not score.reliable:
        return f"insufficient sample (< {get_settings().account_min_posts} posts)"
    width = (score.ci_high or 0) - (score.ci_low or 0)
    if width < 0.1:
        return "high confidence"
    if width < 0.2:
        return "moderate confidence"
    return "low confidence"


__all__ = ["DISCLAIMER", "WEIGHTS", "bootstrap_ci", "confidence_label", "score_account", "score_all_accounts", "math"]
