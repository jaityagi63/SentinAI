#!/usr/bin/env python
"""Train the custom YOLOv8 hate-symbol detector (Module 5).

Dataset layout (Ultralytics format)::

    <dataset>/
      images/{train,val}/*.jpg
      labels/{train,val}/*.txt      # one "class cx cy w h" line per box (normalised)

Class ids must follow ``sentinai.classification.multimodal.SYMBOL_CLASSES`` — this script
writes the matching ``data.yaml`` for you. Sources used in practice: the ADL Hate Symbols
Database (reference imagery), Hateful Memes (FAIR) crops, and in-house annotated screenshots.
Always keep the dataset out of Git.

Output: ``<model-cache>/symbols/yolov8_hate_symbols.pt`` (loaded lazily by
``multimodal.detect_symbols``). Requires ``pip install -e ".[vision]"``.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml

from sentinai.classification.multimodal import SYMBOL_CLASSES
from sentinai.config import get_settings


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--base", default="yolov8s.pt", help="pretrained YOLOv8 weights to fine-tune")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    from ultralytics import YOLO  # type: ignore

    data_yaml = args.dataset / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({"path": str(args.dataset.resolve()), "train": "images/train", "val": "images/val", "names": dict(enumerate(SYMBOL_CLASSES))}))
    model = YOLO(args.base)
    results = model.train(data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, project=str(args.dataset / "runs"), name="hate_symbols", exist_ok=True)
    best = Path(results.save_dir) / "weights" / "best.pt"
    out = (args.out or Path(get_settings().model_cache_dir)) / "symbols"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, out / "yolov8_hate_symbols.pt")
    print(f"saved → {out / 'yolov8_hate_symbols.pt'}")


if __name__ == "__main__":
    main()
