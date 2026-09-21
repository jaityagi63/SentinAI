"""Module 2 — Text preprocessing pipeline.

normalize -> emoji parsing -> obfuscation detection -> language detection -> transliteration
-> near-duplicate detection (MinHash / LSH).
"""

from sentinai.preprocessing.pipeline import Preprocessor, preprocess

__all__ = ["Preprocessor", "preprocess"]
