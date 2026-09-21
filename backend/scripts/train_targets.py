#!/usr/bin/env python
"""Fine-tune the Task B multi-label target head (XLM-RoBERTa-large by default).

Input JSONL: ``{"text": ..., "targets": ["religion:islam", "nationality:pakistani"]}``.
Non-toxic rows may have an empty ``targets`` list (they teach the head to stay silent).
Mixing in code-switched data (LinCE Spanglish / Hinglish converted to this format) is the
recommended recipe for Module 7.

Output: ``<model-cache>/targets/`` (HF layout) + ``target_labels.json`` listing the
``[category, label]`` pairs in logit order, exactly what ``TransformerClassifier`` expects.

Requires ``pip install -e ".[ml]"``.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import yaml

from sentinai.config import RESOURCES_DIR


def taxonomy_labels() -> list[tuple[str, str]]:
    data = yaml.safe_load((RESOURCES_DIR / "lexicons" / "targets.yaml").read_text(encoding="utf-8"))
    out: list[tuple[str, str]] = []
    for category, groups in data.items():
        if not isinstance(groups, dict):
            continue
        for label in groups:
            out.append((category, label))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", type=Path, required=True)
    ap.add_argument("--base", default="xlm-roberta-large")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1.5e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=192)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from sentinai.config import get_settings

    labels = taxonomy_labels()
    index = {f"{c}:{l}": i for i, (c, l) in enumerate(labels)}  # noqa: E741
    rows = []
    for p in args.data:
        for line in open(p, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if r.get("text"):
                    rows.append(r)
    random.seed(args.seed)
    random.shuffle(rows)
    if not rows:
        raise SystemExit("no rows")

    out_dir = (args.out or Path(get_settings().model_cache_dir)) / "targets"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(args.base, num_labels=len(labels), problem_type="multi_label_classification", id2label={i: f"{c}/{l}" for i, (c, l) in enumerate(labels)}).to(device)  # noqa: E741

    def encode(batch: list[dict]):
        enc = tok([r["text"] for r in batch], padding=True, truncation=True, max_length=args.max_len, return_tensors="pt")
        y = torch.zeros(len(batch), len(labels))
        for i, r in enumerate(batch):
            for t in r.get("targets") or []:
                if t in index:
                    y[i, index[t]] = 1.0
        return enc, y

    loader = DataLoader(rows, batch_size=args.batch_size, shuffle=True, collate_fn=encode)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    # positive-class up-weighting: targets are sparse
    pos = torch.zeros(len(labels))
    for r in rows:
        for t in r.get("targets") or []:
            if t in index:
                pos[index[t]] += 1
    pos_weight = ((len(rows) - pos) / pos.clamp(min=1)).clamp(max=50.0).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model.train()
    for epoch in range(args.epochs):
        tot = 0.0
        for enc, y in loader:
            enc = {k: v.to(device) for k, v in enc.items()}
            loss = loss_fn(model(**enc).logits, y.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            tot += float(loss)
        print(f"epoch {epoch + 1}/{args.epochs} · loss {tot / max(1, len(loader)):.4f}")

    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    (out_dir / "target_labels.json").write_text(json.dumps([list(x) for x in labels]))
    print(f"saved → {out_dir} ({len(labels)} target labels)")


if __name__ == "__main__":
    main()
