"""Module 5 — Multimodal analysis (text + image / meme).

Pipeline for a post with media:

    image ─┬─► OCR (Tesseract v5 → EasyOCR fallback) ─► text classifier (Task A/B/C on OCR text)
           ├─► Vision-language model (CLIP zero-shot over hateful-imagery prompts)
           └─► Hate-symbol detector (YOLOv8 custom weights: swastika, SS bolts, KKK hood, 1488,
                burning cross, noose, celtic-cross variants, black sun, Pepe/Happy-Merchant memes …)

    visual_toxicity = max(vlm_hateful, symbol_confidence, 0.9 * ocr_text_toxicity)
    final = (1 - w) * text_toxicity + w * visual_toxicity        (w = settings.visual_ensemble_weight)
    juxtaposition flag when caption is benign but image/OCR text is hateful (or vice-versa).

Each stage is optional: when a dependency or weight file is missing the stage is skipped and
reported in ``VisualResult`` so the dashboard can show which signals were available.  Alt-text
supplied by the platform is always analysed (it is free, and frequently contains the meme text).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from sentinai.config import get_settings
from sentinai.schemas import MediaAttachment, VisualResult

log = logging.getLogger(__name__)

# Zero-shot CLIP prompts (Module 5 vision-language stage).
CLIP_PROMPTS: dict[str, str] = {
    "hateful_symbol": "a photo containing a nazi swastika, ss bolts, or other hate symbol",
    "extremist_iconography": "extremist propaganda imagery with militant or white supremacist iconography",
    "kkk_imagery": "people wearing ku klux klan hoods or a burning cross",
    "dehumanizing_meme": "a racist meme comparing people to animals or vermin",
    "violent_threat": "a threatening image with weapons aimed at a group of people",
    "benign": "an ordinary everyday photo with nothing offensive",
    "news_screenshot": "a screenshot of a news article or a social media post",
    "text_meme": "a meme consisting mostly of text",
}
HATEFUL_PROMPTS = ("hateful_symbol", "extremist_iconography", "kkk_imagery", "dehumanizing_meme", "violent_threat")

# Symbol classes of the custom YOLOv8 detector (training recipe: scripts/train_symbols.py).
SYMBOL_CLASSES = [
    "swastika", "ss_bolts", "kkk_hood", "burning_cross", "noose", "1488", "black_sun", "celtic_cross_hate",
    "wolfsangel", "totenkopf", "confederate_flag", "iron_cross_hate", "happy_merchant", "echo_brackets",
    "isis_flag", "okay_hand_wp", "skull_mask", "boogaloo_flag", "triple_parentheses",
]
# Hate-code text tokens frequently found in memes (OCR post-processing).
_HATE_CODE_RE = re.compile(r"\b(1488|14/88|14 88|88|hh|wpww|rahowa|goyim know|6mwe|13/52|13/90|jq|zog|gtkrwn|tnd|kalergi)\b", re.I)


@dataclass
class OCRResult:
    text: str
    engine: str | None


# --------------------------------------------------------------------------------------------
# OCR
# --------------------------------------------------------------------------------------------


def run_ocr(path: Path) -> OCRResult:
    """Tesseract v5 (pytesseract) first, EasyOCR fallback, else empty."""
    try:
        import pytesseract  # type: ignore
        from PIL import Image  # type: ignore

        txt = pytesseract.image_to_string(Image.open(path))
        if txt.strip():
            return OCRResult(txt.strip(), "tesseract")
    except Exception as exc:  # pragma: no cover - env specific
        log.debug("tesseract unavailable: %s", exc)
    try:
        reader = _easyocr_reader()
        if reader is not None:
            txt = " ".join(r[1] for r in reader.readtext(str(path)))
            if txt.strip():
                return OCRResult(txt.strip(), "easyocr")
    except Exception as exc:  # pragma: no cover
        log.debug("easyocr failed: %s", exc)
    return OCRResult("", None)


@lru_cache
def _easyocr_reader():  # pragma: no cover - heavy optional dependency
    try:
        import easyocr  # type: ignore

        return easyocr.Reader(["en", "es", "fr", "pt", "de", "ar", "hi"], gpu=False)
    except Exception:
        return None


# --------------------------------------------------------------------------------------------
# CLIP zero-shot
# --------------------------------------------------------------------------------------------


@lru_cache
def _clip():  # pragma: no cover - heavy optional dependency
    try:
        import open_clip  # type: ignore
        import torch  # type: ignore

        model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        model.eval()
        with torch.no_grad():
            text_feats = model.encode_text(tokenizer(list(CLIP_PROMPTS.values())))
            text_feats /= text_feats.norm(dim=-1, keepdim=True)
        return model, preprocess, text_feats
    except Exception as exc:
        log.debug("CLIP unavailable: %s", exc)
        return None


def clip_scores(path: Path) -> dict[str, float]:  # pragma: no cover - heavy optional dependency
    bundle = _clip()
    if bundle is None:
        return {}
    import torch
    from PIL import Image

    model, preprocess, text_feats = bundle
    img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        f = model.encode_image(img)
        f /= f.norm(dim=-1, keepdim=True)
        probs = (100.0 * f @ text_feats.T).softmax(dim=-1)[0].tolist()
    return {k: round(float(p), 4) for k, p in zip(CLIP_PROMPTS.keys(), probs, strict=True)}


# --------------------------------------------------------------------------------------------
# YOLOv8 hate-symbol detector
# --------------------------------------------------------------------------------------------


@lru_cache
def _yolo():  # pragma: no cover - heavy optional dependency
    weights = Path(get_settings().model_cache_dir) / "symbols" / "yolov8_hate_symbols.pt"
    if not weights.exists():
        return None
    try:
        from ultralytics import YOLO  # type: ignore

        return YOLO(str(weights))
    except Exception as exc:
        log.debug("YOLO unavailable: %s", exc)
        return None


def detect_symbols(path: Path) -> list[dict]:  # pragma: no cover - heavy optional dependency
    model = _yolo()
    if model is None:
        return []
    out = []
    for r in model.predict(str(path), verbose=False, conf=0.35):
        for b in r.boxes:
            cls = int(b.cls[0])
            out.append({"label": model.names.get(cls, str(cls)), "confidence": round(float(b.conf[0]), 4), "box": [round(float(x), 1) for x in b.xyxy[0].tolist()]})
    return out


# --------------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------------


class MultimodalAnalyzer:
    def __init__(self, text_scorer):
        """``text_scorer(text) -> float`` returns a toxicity score for OCR / alt text."""
        self.text_scorer = text_scorer
        self.weight = get_settings().visual_ensemble_weight

    def analyze(self, media: list[MediaAttachment], caption_toxicity: float) -> VisualResult | None:
        if not media:
            return None
        ocr_texts: list[str] = []
        engine: str | None = None
        vlm: dict[str, float] = {}
        symbols: list[dict] = []
        for m in media:
            if m.alt_text:
                ocr_texts.append(m.alt_text)
            if m.local_path and Path(m.local_path).exists():
                p = Path(m.local_path)
                ocr = run_ocr(p)
                if ocr.text:
                    ocr_texts.append(ocr.text)
                    engine = ocr.engine
                for k, v in clip_scores(p).items():
                    vlm[k] = max(vlm.get(k, 0.0), v)
                symbols.extend(detect_symbols(p))

        ocr_text = " ".join(ocr_texts).strip()
        ocr_tox = self.text_scorer(ocr_text) if ocr_text else 0.0
        if ocr_text and _HATE_CODE_RE.search(ocr_text):
            ocr_tox = max(ocr_tox, 0.7)
        vlm_hateful = max((vlm.get(k, 0.0) for k in HATEFUL_PROMPTS), default=0.0)
        sym_conf = max((s["confidence"] for s in symbols), default=0.0)
        visual = max(vlm_hateful, sym_conf, 0.9 * ocr_tox)
        juxt = (caption_toxicity < 0.3 and visual >= 0.6) or (caption_toxicity >= 0.6 and ocr_text != "" and ocr_tox < 0.2 and vlm_hateful < 0.3)
        return VisualResult(
            ocr_text=ocr_text[:2000],
            ocr_engine=engine if engine else ("alt-text" if ocr_text else None),
            visual_toxicity=round(visual, 4),
            symbols=symbols,
            vlm_labels=vlm,
            juxtaposition_flag=juxt,
        )

    def ensemble(self, text_toxicity: float, visual: VisualResult | None) -> float:
        if visual is None:
            return text_toxicity
        w = self.weight
        # Weighted ensemble, but a strong visual signal should never be diluted below itself.
        return round(max((1 - w) * text_toxicity + w * visual.visual_toxicity, min(visual.visual_toxicity, 0.95) if visual.visual_toxicity >= 0.8 else 0.0), 4)
