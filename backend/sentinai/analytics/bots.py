"""Module 10 — Bot & inauthentic behaviour detection.

Feature set: account age, posting frequency, follower/following ratio, profile completeness,
tweet timing regularity, content repetitiveness (+ a few cheap extras: username digit ratio,
default avatar, retweet ratio).

Model: a scikit-learn RandomForest trained on TwiBot-22-style features via
``scripts/train_bot_detector.py`` and stored at ``models/bots/random_forest.joblib``.  When no
trained model exists, a calibrated logistic scoring function over the same features is used
(coefficients set from the published TwiBot-20/22 feature-importance literature) so the module
is always functional.  Output: ``P_bot ∈ [0, 1]``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from sentinai.config import get_settings
from sentinai.schemas import BotScore
from sentinai.storage.models import AuthorRow, PostRow

FEATURE_NAMES = [
    "account_age_days",
    "posts_per_day",
    "follower_following_ratio",
    "profile_completeness",
    "timing_regularity",
    "content_repetitiveness",
    "username_digit_ratio",
    "default_avatar",
    "retweet_ratio",
    "reply_ratio",
    "night_activity_ratio",
    "burstiness",
]

# Logistic fallback weights (positive → more bot-like).  Features are transformed to roughly
# [0, 1] or log scale in ``_transform`` before applying these.
_FALLBACK_WEIGHTS = {
    "bias": -2.4,
    "account_age_days": -1.4,  # log-scaled, younger → higher
    "posts_per_day": 1.6,
    "follower_following_ratio": -1.0,
    "profile_completeness": -1.3,
    "timing_regularity": 2.2,
    "content_repetitiveness": 2.4,
    "username_digit_ratio": 1.2,
    "default_avatar": 0.9,
    "retweet_ratio": 1.1,
    "reply_ratio": -0.3,
    "night_activity_ratio": 0.6,
    "burstiness": 0.8,
}


@dataclass
class AuthorActivity:
    author: AuthorRow
    timestamps: list[datetime]
    texts: list[str]
    retweets: int
    replies: int


def compute_features(act: AuthorActivity, now: datetime | None = None) -> dict[str, float]:
    now = now or datetime.utcnow()
    a = act.author
    age_days = max(1.0, (now - a.created_at).total_seconds() / 86400) if a.created_at else 365.0
    n = len(act.timestamps)
    ts = sorted(act.timestamps)
    span_days = max(1.0, (ts[-1] - ts[0]).total_seconds() / 86400) if n > 1 else 1.0
    lifetime_rate = (a.tweet_count or n) / age_days
    observed_rate = n / span_days if n > 1 else float(n)
    posts_per_day = max(lifetime_rate, observed_rate)

    ratio = (a.followers_count + 1) / (a.following_count + 1)
    completeness = sum([bool(a.description), bool(a.location), bool(a.display_name), not a.has_default_profile_image, bool(a.verified)]) / 5.0

    # timing regularity: 1 - coefficient of variation of inter-arrival gaps (bounded)
    if n > 2:
        gaps = [(ts[i + 1] - ts[i]).total_seconds() for i in range(n - 1)]
        mean = sum(gaps) / len(gaps)
        sd = (sum((g - mean) ** 2 for g in gaps) / len(gaps)) ** 0.5
        cv = sd / mean if mean > 0 else 0.0
        regularity = max(0.0, 1.0 - min(cv, 1.5) / 1.5)
        burstiness = (sd - mean) / (sd + mean) if (sd + mean) > 0 else 0.0  # Goh & Barabási
        burstiness = (burstiness + 1) / 2
    else:
        regularity, burstiness = 0.0, 0.5

    # content repetitiveness: 1 - (unique normalised texts / texts), plus mean pairwise jaccard on a sample
    if n > 1:
        norm = [re.sub(r"\W+", " ", t.lower()).strip() for t in act.texts]
        uniq_ratio = len(set(norm)) / n
        sample = norm[:40]
        sets = [set(s.split()) for s in sample]
        sims = []
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                u = sets[i] | sets[j]
                if u:
                    sims.append(len(sets[i] & sets[j]) / len(u))
        jacc = sum(sims) / len(sims) if sims else 0.0
        repetitiveness = max(1.0 - uniq_ratio, jacc)
    else:
        repetitiveness = 0.0

    uname = a.username or ""
    digit_ratio = sum(c.isdigit() for c in uname) / max(1, len(uname))
    night = sum(1 for t in ts if 1 <= t.hour < 5) / n if n else 0.0

    return {
        "account_age_days": round(age_days, 2),
        "posts_per_day": round(posts_per_day, 3),
        "follower_following_ratio": round(ratio, 3),
        "profile_completeness": round(completeness, 3),
        "timing_regularity": round(regularity, 3),
        "content_repetitiveness": round(repetitiveness, 3),
        "username_digit_ratio": round(digit_ratio, 3),
        "default_avatar": 1.0 if a.has_default_profile_image else 0.0,
        "retweet_ratio": round(act.retweets / n, 3) if n else 0.0,
        "reply_ratio": round(act.replies / n, 3) if n else 0.0,
        "night_activity_ratio": round(night, 3),
        "burstiness": round(burstiness, 3),
    }


def _transform(f: dict[str, float]) -> dict[str, float]:
    return {
        "account_age_days": min(1.0, math.log1p(f["account_age_days"]) / math.log1p(3650)),
        "posts_per_day": min(1.0, math.log1p(f["posts_per_day"]) / math.log1p(200)),
        "follower_following_ratio": min(1.0, math.log1p(f["follower_following_ratio"]) / math.log1p(100)),
        "profile_completeness": f["profile_completeness"],
        "timing_regularity": f["timing_regularity"],
        "content_repetitiveness": f["content_repetitiveness"],
        "username_digit_ratio": min(1.0, f["username_digit_ratio"] * 2),
        "default_avatar": f["default_avatar"],
        "retweet_ratio": f["retweet_ratio"],
        "reply_ratio": f["reply_ratio"],
        "night_activity_ratio": f["night_activity_ratio"],
        "burstiness": f["burstiness"],
    }


@lru_cache
def _trained_model():
    path = Path(get_settings().model_cache_dir) / "bots" / "random_forest.joblib"
    if not path.exists():
        return None
    try:
        import joblib  # type: ignore

        return joblib.load(path)
    except Exception:
        return None


def score_features(f: dict[str, float]) -> tuple[float, str, list[str]]:
    model = _trained_model()
    if model is not None:
        import numpy as np

        p = float(model.predict_proba(np.asarray([[f[k] for k in FEATURE_NAMES]]))[0][1])
        importances = getattr(model, "feature_importances_", None)
        top = [FEATURE_NAMES[i] for i in sorted(range(len(FEATURE_NAMES)), key=lambda i: -importances[i])[:3]] if importances is not None else []
        return p, "random-forest(twibot-22)", top
    t = _transform(f)
    contrib = {k: _FALLBACK_WEIGHTS[k] * v for k, v in t.items()}
    z = _FALLBACK_WEIGHTS["bias"] + sum(contrib.values())
    p = 1.0 / (1.0 + math.exp(-z))
    # top signals = largest positive contributions (young age counts as positive when age is small)
    signals = dict(contrib)
    signals["account_age_days"] = -contrib["account_age_days"] - 1.4  # young account → positive
    signals["profile_completeness"] = -contrib["profile_completeness"] - 1.3
    signals["follower_following_ratio"] = -contrib["follower_following_ratio"] - 1.0
    top = [k for k, v in sorted(signals.items(), key=lambda kv: -kv[1]) if v > 0.15][:3]
    return p, "heuristic-logistic", top


def score_author(session: Session, author_id: str, days: int = 90) -> BotScore | None:
    a = session.get(AuthorRow, author_id)
    if a is None:
        return None
    since = datetime.utcnow() - timedelta(days=days)
    rows = session.execute(select(PostRow.created_at, PostRow.text, PostRow.retweeted_id, PostRow.parent_id).where(PostRow.author_id == author_id, PostRow.created_at >= since)).all()
    act = AuthorActivity(author=a, timestamps=[r[0] for r in rows], texts=[r[1] for r in rows], retweets=sum(1 for r in rows if r[2]), replies=sum(1 for r in rows if r[3]))
    f = compute_features(act)
    p, model, top = score_features(f)
    a.bot_probability = round(p, 4)
    a.bot_features = f
    return BotScore(author_id=author_id, probability=round(p, 4), features=f, top_signals=top, model=model)


def score_all_authors(session: Session, days: int = 90) -> int:
    ids = session.scalars(select(AuthorRow.id)).all()
    n = 0
    for aid in ids:
        if score_author(session, aid, days) is not None:
            n += 1
    return n
