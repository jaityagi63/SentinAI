#!/usr/bin/env python
"""Fine-tune the Task A (toxicity) and Task C (severity, CORAL ordinal) transformer heads.

Requires ``pip install -e ".[ml]"`` and a GPU for the large backbones (DeBERTa-v3-large /
HateBERT); small backbones (``distilroberta-base``) run on CPU for smoke tests.

Input: JSONL rows ``{"text": ..., "label": "hate_speech", "severity": 3, "group": "aave"}``.
The ``severity`` and ``group`` fields are optional. Retraining batches exported by the HITL
loop (``sentinai retrain-batch`` → ``data/retrain/batch_*.jsonl``) are directly compatible.

Outputs (Hugging Face layout, loadable by ``sentinai.classification.transformer``):

* ``<model-cache>/toxicity_en/``   – 4-way softmax head (config.id2label = SentinAI labels)
* ``<model-cache>/severity/``      – 4-logit CORAL head (P(level > k), k = 0..3)

Bias mitigation (Module 11): ``--adversarial`` adds a gradient-reversal adversary that tries
to predict the dialect group from the pooled representation; the classifier is trained to make
that impossible, reducing dialect-correlated false positives (AAVE, Chicano English, …).

Example::

    python scripts/train_toxicity.py --data data/retrain/batch_20260101_120000.jsonl \
        --base microsoft/deberta-v3-large --epochs 3 --adversarial
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

LABELS = ["non_toxic", "offensive", "hate_speech", "violent_extremism"]
GROUPS = ["sae", "aave", "chicano", "hinglish", "arabizi", "british", "other"]


def load_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    if r.get("text") and r.get("label") in LABELS:
                        rows.append(r)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", type=Path, required=True)
    ap.add_argument("--base", default="microsoft/deberta-v3-large", help="backbone (or GroNLP/hateBERT, microsoft/mdeberta-v3-base for multilingual)")
    ap.add_argument("--out", type=Path, default=None, help="model cache dir (default: settings.model_cache_dir)")
    ap.add_argument("--head", choices=["toxicity_en", "toxicity_multi", "severity"], default="toxicity_en")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--adversarial", action="store_true", help="gradient-reversal dialect adversary (needs `group` labels or uses tag_dialect)")
    ap.add_argument("--adv-lambda", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    from sentinai.analytics.fairness import tag_dialect
    from sentinai.config import get_settings

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows = load_rows(args.data)
    if not rows:
        raise SystemExit("no usable rows")
    random.shuffle(rows)
    out_dir = (args.out or Path(get_settings().model_cache_dir)) / args.head
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{len(rows)} rows · backbone={args.base} · head={args.head} · device={device} · adversarial={args.adversarial}")

    tok = AutoTokenizer.from_pretrained(args.base)
    ordinal = args.head == "severity"
    if ordinal:
        model = AutoModelForSequenceClassification.from_pretrained(args.base, num_labels=4)  # CORAL: K-1 cumulative logits
        model.config.id2label = {i: f"gt_{i}" for i in range(4)}
        model.config.problem_type = "coral_ordinal"
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.base, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)})  # noqa: E741
    model.to(device)

    class GradReverse(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):  # type: ignore[override]
            return x.view_as(x)

        @staticmethod
        def backward(ctx, g):  # type: ignore[override]
            return -args.adv_lambda * g

    hidden = model.config.hidden_size
    adversary = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU(), nn.Linear(128, len(GROUPS))).to(device) if args.adversarial else None

    def encode(batch: list[dict]):
        enc = tok([r["text"] for r in batch], padding=True, truncation=True, max_length=args.max_len, return_tensors="pt")
        if ordinal:
            sev = torch.tensor([int(r.get("severity") or (0 if r["label"] == "non_toxic" else 2)) for r in batch])
            y = torch.stack([(sev > k).float() for k in range(4)], dim=1)  # cumulative targets
        else:
            y = torch.tensor([LABELS.index(r["label"]) for r in batch])
        g = torch.tensor([GROUPS.index(r.get("group") or tag_dialect(r["text"])) if (r.get("group") or tag_dialect(r["text"])) in GROUPS else GROUPS.index("other") for r in batch])
        return enc, y, g

    loader = DataLoader(rows, batch_size=args.batch_size, shuffle=True, collate_fn=encode)
    params = list(model.parameters()) + (list(adversary.parameters()) if adversary else [])
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * len(loader) * args.epochs), len(loader) * args.epochs)
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(args.epochs):
        tot = 0.0
        for enc, y, g in loader:
            enc = {k: v.to(device) for k, v in enc.items()}
            y, g = y.to(device), g.to(device)
            out = model(**enc, output_hidden_states=bool(adversary))
            loss = bce(out.logits, y) if ordinal else ce(out.logits, y)
            if adversary is not None:
                pooled = out.hidden_states[-1][:, 0]  # [CLS]
                adv_logits = adversary(GradReverse.apply(pooled))
                loss = loss + ce(adv_logits, g)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            tot += float(loss)
        print(f"epoch {epoch + 1}/{args.epochs} · loss {tot / max(1, len(loader)):.4f}")

    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    (out_dir / "training_meta.json").write_text(json.dumps({"base": args.base, "rows": len(rows), "epochs": args.epochs, "adversarial": args.adversarial, "labels": LABELS if not ordinal else "coral(4)"}, indent=2))
    print(f"saved → {out_dir}")


if __name__ == "__main__":
    main()
