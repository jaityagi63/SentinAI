"""Near-duplicate detection with MinHash + LSH (Module 2).

Uses ``datasketch`` when installed; otherwise a compact pure-Python MinHash/LSH implementation
with the same behaviour (banding over the signature) is used so the pipeline has no hard
third-party dependency.
"""

from __future__ import annotations

import hashlib
import re
import struct
from collections.abc import Iterable

try:
    from datasketch import MinHash as _DSMinHash
    from datasketch import MinHashLSH as _DSMinHashLSH
except Exception:  # pragma: no cover - optional dependency
    _DSMinHash = None
    _DSMinHashLSH = None

_MERSENNE = (1 << 61) - 1
_MAX_HASH = (1 << 32) - 1
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def shingles(text: str, k: int = 3) -> set[str]:
    """Word k-shingles (falls back to the token set for very short texts)."""
    toks = _TOKEN_RE.findall(text.lower())
    if len(toks) <= k:
        return set(toks)
    return {" ".join(toks[i : i + k]) for i in range(len(toks) - k + 1)}


def _hash32(s: str) -> int:
    return struct.unpack("<I", hashlib.sha1(s.encode("utf-8")).digest()[:4])[0]


class MinHash:
    """Pure-Python MinHash with universal hashing (a*x+b mod p)."""

    def __init__(self, num_perm: int = 128, seed: int = 1):
        self.num_perm = num_perm
        import random

        rng = random.Random(seed)
        self._a = [rng.randint(1, _MERSENNE - 1) for _ in range(num_perm)]
        self._b = [rng.randint(0, _MERSENNE - 1) for _ in range(num_perm)]
        self.hashvalues = [_MAX_HASH] * num_perm

    def update(self, item: str) -> None:
        x = _hash32(item)
        for i in range(self.num_perm):
            h = ((self._a[i] * x + self._b[i]) % _MERSENNE) & _MAX_HASH
            if h < self.hashvalues[i]:
                self.hashvalues[i] = h

    def jaccard(self, other: MinHash) -> float:
        eq = sum(1 for a, b in zip(self.hashvalues, other.hashvalues, strict=True) if a == b)
        return eq / self.num_perm

    def digest(self) -> list[int]:
        return list(self.hashvalues)


class MinHashLSH:
    """Banded LSH index over MinHash signatures."""

    def __init__(self, threshold: float = 0.8, num_perm: int = 128):
        self.threshold = threshold
        self.num_perm = num_perm
        self.b, self.r = self._optimal_params(threshold, num_perm)
        self._tables: list[dict[bytes, set[str]]] = [dict() for _ in range(self.b)]
        self._keys: dict[str, list[bytes]] = {}

    @staticmethod
    def _optimal_params(threshold: float, num_perm: int) -> tuple[int, int]:
        best = (1, num_perm)
        best_err = float("inf")
        for b in range(1, num_perm + 1):
            if num_perm % b:
                continue
            r = num_perm // b
            # approximate threshold of the S-curve: (1/b)^(1/r)
            t = (1.0 / b) ** (1.0 / r)
            err = abs(t - threshold)
            if err < best_err:
                best_err, best = err, (b, r)
        return best

    def _bands(self, hv: list[int]) -> list[bytes]:
        return [struct.pack(f"<{self.r}I", *hv[i * self.r : (i + 1) * self.r]) for i in range(self.b)]

    def insert(self, key: str, mh: MinHash) -> None:
        bands = self._bands(mh.hashvalues)
        self._keys[key] = bands
        for t, band in zip(self._tables, bands, strict=True):
            t.setdefault(band, set()).add(key)

    def query(self, mh: MinHash) -> list[str]:
        out: set[str] = set()
        for t, band in zip(self._tables, self._bands(mh.hashvalues), strict=True):
            out.update(t.get(band, ()))
        return list(out)

    def __contains__(self, key: str) -> bool:
        return key in self._keys


class Deduplicator:
    """Stateful near-duplicate detector.

    >>> d = Deduplicator(threshold=0.8)
    >>> d.check("1", "kill all of them now")   # returns None (first occurrence)
    >>> d.check("2", "kill all of them now!!") # returns "1"
    """

    def __init__(self, threshold: float = 0.8, num_perm: int = 128, k: int = 3):
        self.threshold = threshold
        self.num_perm = num_perm
        self.k = k
        self._use_ds = _DSMinHash is not None
        self._lsh = _DSMinHashLSH(threshold=threshold, num_perm=num_perm) if self._use_ds else MinHashLSH(threshold, num_perm)
        self._sigs: dict[str, object] = {}
        self._texts: dict[str, str] = {}

    def signature(self, text: str):
        if self._use_ds:
            mh = _DSMinHash(num_perm=self.num_perm)
            for s in shingles(text, self.k):
                mh.update(s.encode("utf-8"))
            return mh
        mh = MinHash(self.num_perm)
        for s in shingles(text, self.k):
            mh.update(s)
        return mh

    @staticmethod
    def _digest(mh) -> list[int]:
        return [int(x) for x in mh.hashvalues]

    def check(self, key: str, text: str) -> str | None:
        """Register ``key`` and return the key of a near-duplicate seen before (or None)."""
        mh = self.signature(text)
        cands = self._lsh.query(mh)
        best: tuple[float, str] | None = None
        for c in cands:
            if c == key:
                continue
            other = self._sigs[c]
            sim = mh.jaccard(other)
            if sim >= self.threshold and (best is None or sim > best[0]):
                best = (sim, c)
        if key not in self._sigs:
            self._sigs[key] = mh
            self._texts[key] = text
            try:
                self._lsh.insert(key, mh)
            except ValueError:  # datasketch raises on duplicate keys
                pass
        return best[1] if best else None

    def check_with_signature(self, key: str, text: str) -> tuple[str | None, list[int]]:
        dup = self.check(key, text)
        return dup, self._digest(self._sigs[key])

    def bulk(self, items: Iterable[tuple[str, str]]) -> dict[str, str | None]:
        return {k: self.check(k, t) for k, t in items}

    def __len__(self) -> int:
        return len(self._sigs)
