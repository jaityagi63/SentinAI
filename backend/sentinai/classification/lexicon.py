"""Lexicon loader shared by the heuristic classifier, the explainability layer and the
preprocessing (known-word) step.

Compiles the YAML resources under ``sentinai/resources/lexicons`` into regexes:
  * ``targets.yaml``   → group terms per (category, label)
  * ``slurs.yaml``     → slur terms per (category, label) with weights / ambiguity flags
  * ``hostility.yaml`` → hostility patterns per severity category, with ``{G}`` expansion
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from sentinai.config import RESOURCES_DIR

LEXICON_DIR = RESOURCES_DIR / "lexicons"

_CATEGORY_SEVERITY = {"stereotype": 1, "dehumanization": 2, "harassment": 3, "incitement": 4}


@dataclass(frozen=True)
class GroupTerm:
    term: str
    category: str  # ethnicity | religion | nationality
    label: str
    weight: float = 0.0  # >0 → slur
    ambiguous: bool = False

    @property
    def is_slur(self) -> bool:
        return self.weight > 0


@dataclass
class HostilityPattern:
    category: str  # stereotype | dehumanization | harassment | incitement | offensive | counterspeech
    weight: float
    regex: re.Pattern[str]
    source: str
    targeted: bool  # pattern contains {G}

    @property
    def severity(self) -> int:
        return _CATEGORY_SEVERITY.get(self.category, 0)


def _boundary(term: str) -> str:
    """Word boundaries that work for Latin, Arabic and Devanagari text."""
    esc = re.escape(term)
    if re.search(r"[\u0600-\u06FF\u0900-\u097F]", term):
        return rf"(?<![\w\u0600-\u06FF\u0900-\u097F]){esc}(?![\w\u0600-\u06FF\u0900-\u097F])"
    if term.startswith("("):  # echo brackets
        return esc
    return rf"(?<![\w]){esc}(?![\w])"


@dataclass
class Lexicon:
    groups: list[GroupTerm] = field(default_factory=list)
    slurs: list[GroupTerm] = field(default_factory=list)
    patterns: list[HostilityPattern] = field(default_factory=list)
    group_regex: re.Pattern[str] | None = None
    slur_regex: re.Pattern[str] | None = None
    _term_index: dict[str, GroupTerm] = field(default_factory=dict)

    # ------------------------------------------------------------------------------------
    @property
    def vocabulary(self) -> set[str]:
        """Every single-word lexicon token (used by the obfuscation repair step)."""
        vocab: set[str] = set()
        for t in self.groups + self.slurs:
            for w in t.term.split():
                if w.isalpha() and len(w) > 2:
                    vocab.add(w)
        # add hostility verbs frequently obfuscated
        vocab.update(
            "kill murder shoot hang lynch burn gas exterminate deport racist racism nazi hitler terrorist "
            "rape genocide bomb attack slaughter die death hate scum vermin cockroach parasite invaders".split()
        )
        return vocab

    def lookup(self, term: str) -> GroupTerm | None:
        return self._term_index.get(term.lower())

    def find_groups(self, text: str) -> list[tuple[GroupTerm, int, int]]:
        """All neutral-group and slur mentions with spans, longest match first at each position."""
        out: list[tuple[GroupTerm, int, int]] = []
        if self.group_regex is None:
            return out
        for m in self.group_regex.finditer(text):
            term = self.lookup(m.group(0))
            if term:
                out.append((term, m.start(), m.end()))
        return out

    def find_slurs(self, text: str) -> list[tuple[GroupTerm, int, int]]:
        if self.slur_regex is None:
            return []
        out: list[tuple[GroupTerm, int, int]] = []
        for m in self.slur_regex.finditer(text):
            term = self.lookup(m.group(0))
            if term and term.is_slur:
                out.append((term, m.start(), m.end()))
        return out

    def extend(self, terms: list[GroupTerm]) -> None:
        """Add terms at runtime (e.g. licensed lexicons) and recompile."""
        for t in terms:
            (self.slurs if t.is_slur else self.groups).append(t)
        self._compile()

    # ------------------------------------------------------------------------------------
    def _compile(self) -> None:
        all_terms = self.groups + self.slurs
        self._term_index = {t.term.lower(): t for t in all_terms}
        by_len = sorted({t.term.lower() for t in all_terms}, key=len, reverse=True)
        if by_len:
            self.group_regex = re.compile("|".join(_boundary(t) for t in by_len), re.I)
        slur_terms = sorted({t.term.lower() for t in self.slurs}, key=len, reverse=True)
        if slur_terms:
            self.slur_regex = re.compile("|".join(_boundary(t) for t in slur_terms), re.I)


def _group_alternation(terms: list[GroupTerm]) -> str:
    # Sort by length so longer phrases win; include a generic "them"/"people" fallback so that
    # anaphoric hostility still scores (the target is then resolved from group mentions nearby).
    uniq = sorted({t.term.lower() for t in terms}, key=len, reverse=True)
    alts = [re.escape(t) for t in uniq]
    alts.extend([r"them", r"these people", r"those people", r"you people", r"your kind", r"your people", r"immigrants?", r"migrants?", r"refugees?", r"foreigners?", r"minorit(?:y|ies)"])
    return "(?:" + "|".join(alts) + ")"


def load_lexicon(lexicon_dir: Path = LEXICON_DIR) -> Lexicon:
    lex = Lexicon()

    with open(lexicon_dir / "targets.yaml", encoding="utf-8") as fh:
        targets = yaml.safe_load(fh) or {}
    for category, labels in targets.items():
        for label, terms in (labels or {}).items():
            for term in terms or []:
                lex.groups.append(GroupTerm(term=str(term).lower(), category=category, label=label))

    with open(lexicon_dir / "slurs.yaml", encoding="utf-8") as fh:
        slurs = yaml.safe_load(fh) or {}
    for term, meta in slurs.items():
        meta = meta or {}
        lex.slurs.append(
            GroupTerm(
                term=str(term).lower(),
                category=meta.get("category", "ethnicity"),
                label=meta.get("label", "unknown"),
                weight=float(meta.get("weight", 0.8)),
                ambiguous=bool(meta.get("ambiguous", False)),
            )
        )

    lex._compile()

    g_alt = _group_alternation(lex.groups + lex.slurs)
    with open(lexicon_dir / "hostility.yaml", encoding="utf-8") as fh:
        hostility = yaml.safe_load(fh) or {}
    for category, entries in hostility.items():
        for entry in entries or []:
            src = entry["pattern"]
            targeted = "{G}" in src
            pat = src.replace("{G}", g_alt)
            try:
                rx = re.compile(pat, re.I | re.UNICODE)
            except re.error as exc:  # pragma: no cover - resource validation
                raise ValueError(f"Invalid hostility pattern in category {category!r}: {src!r}: {exc}") from exc
            lex.patterns.append(
                HostilityPattern(category=category, weight=float(entry.get("weight", 0.5)), regex=rx, source=src, targeted=targeted)
            )
    return lex


@lru_cache
def get_lexicon() -> Lexicon:
    return load_lexicon()
