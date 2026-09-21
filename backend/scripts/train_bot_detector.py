#!/usr/bin/env python
"""Train the Module 10 bot detector (RandomForest / XGBoost) on TwiBot-22-style data.

Expected input: a CSV/JSONL with one row per account containing the SentinAI feature columns
(``sentinai.analytics.bots.FEATURE_NAMES``) plus a binary ``label`` column (1 = bot). Use
``--from-twibot <dir>`` to derive those features from a TwiBot-22 dump (``user.json`` +
``tweet_*.json`` + ``label.csv``) with the *same* feature code used at inference time, so
training and serving cannot drift.

Output: ``<model-cache>/bots/random_forest.joblib`` — picked up automatically by
``sentinai.analytics.bots.score_features`` (no config change required).

Only scikit-learn (core dependency) is needed; ``--xgboost`` uses xgboost when installed.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from sentinai.analytics.bots import FEATURE_NAMES, AuthorActivity, compute_features
from sentinai.config import get_settings
from sentinai.storage.models import AuthorRow


def _parse_ts(s: str) -> datetime:
    for fmt in ("%Y-%m-%d %H:%M:%S%z", "%a %b %d %H:%M:%S %z %Y", "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=None)
        except ValueError:
            continue
    return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)


def features_from_twibot(root: Path, max_users: int | None = None) -> pd.DataFrame:
    """Derive SentinAI features from a TwiBot-22 release directory."""
    labels = pd.read_csv(root / "label.csv").set_index("id")["label"].map({"bot": 1, "human": 0})
    users = {u["id"]: u for u in json.loads((root / "user.json").read_text(encoding="utf-8"))}
    tweets_by_user: dict[str, list[dict]] = {}
    for f in sorted(root.glob("tweet_*.json")):
        for t in json.loads(f.read_text(encoding="utf-8")):
            tweets_by_user.setdefault(t.get("author_id"), []).append(t)
    rows = []
    for uid, lab in labels.items():
        u = users.get(uid)
        if u is None:
            continue
        pm = u.get("public_metrics") or {}
        author = AuthorRow(
            id=uid,
            username=u.get("username") or "",
            display_name=u.get("name"),
            description=u.get("description"),
            location=u.get("location"),
            created_at=_parse_ts(u["created_at"]) if u.get("created_at") else None,
            followers_count=int(pm.get("followers_count") or 0),
            following_count=int(pm.get("following_count") or 0),
            tweet_count=int(pm.get("tweet_count") or 0),
            verified=bool(u.get("verified")),
            has_default_profile_image="default_profile" in (u.get("profile_image_url") or ""),
        )
        tw = tweets_by_user.get(uid, [])
        act = AuthorActivity(
            author=author,
            timestamps=[_parse_ts(t["created_at"]) for t in tw if t.get("created_at")],
            texts=[t.get("text") or "" for t in tw],
            retweets=sum(1 for t in tw if (t.get("text") or "").startswith("RT @")),
            replies=sum(1 for t in tw if t.get("in_reply_to_user_id")),
        )
        f = compute_features(act)
        f["label"] = int(lab)
        rows.append(f)
        if max_users and len(rows) >= max_users:
            break
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, help="CSV/JSONL with FEATURE_NAMES columns + label")
    ap.add_argument("--from-twibot", type=Path, help="TwiBot-22 directory (user.json, tweet_*.json, label.csv)")
    ap.add_argument("--max-users", type=int, default=None)
    ap.add_argument("--xgboost", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    if args.from_twibot:
        df = features_from_twibot(args.from_twibot, args.max_users)
    elif args.data:
        df = pd.read_json(args.data, lines=True) if args.data.suffix in {".jsonl", ".json"} else pd.read_csv(args.data)
    else:
        raise SystemExit("provide --data or --from-twibot")
    missing = [c for c in FEATURE_NAMES + ["label"] if c not in df.columns]
    if missing:
        raise SystemExit(f"missing columns: {missing}")

    from sklearn.metrics import classification_report, roc_auc_score
    from sklearn.model_selection import train_test_split

    X, y = df[FEATURE_NAMES].fillna(0.0).to_numpy(), df["label"].astype(int).to_numpy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=args.seed, stratify=y)
    if args.xgboost:
        from xgboost import XGBClassifier  # type: ignore

        model = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.9, colsample_bytree=0.8, eval_metric="logloss", random_state=args.seed)
    else:
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=args.seed)
    model.fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:, 1]
    print(f"ROC-AUC {roc_auc_score(yte, proba):.4f}")
    print(classification_report(yte, (proba >= 0.5).astype(int), target_names=["human", "bot"]))

    import joblib

    out = (args.out or Path(get_settings().model_cache_dir)) / "bots"
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "random_forest.joblib")
    (out / "features.json").write_text(json.dumps(FEATURE_NAMES))
    print(f"saved → {out / 'random_forest.joblib'}")


if __name__ == "__main__":
    main()
