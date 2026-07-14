"""NER — spaCy en_core_web_sm (CLAUDE.md §5, FS9).

If spaCy/the model is unavailable, a labelled regex fallback extracts capitalised proper-noun
runs so the KG plumbing still populates in a minimal environment. `eval` reports NER as
degraded when the fallback is used — it is not the real detector and must not be scored as one.
"""

from __future__ import annotations

import re

from .._deps import optional
from ..contracts import Entity

_SPACY_MAP = {
    "PERSON": "PERSON", "ORG": "ORG", "GPE": "GPE", "LOC": "LOC",
    "PRODUCT": "PRODUCT", "EVENT": "EVENT", "DATE": "DATE", "TIME": "TIME",
}
_PROPER = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b")


class NerRunner:
    def __init__(self, cfg=None) -> None:
        self._nlp = None
        spacy = optional("spacy")
        model = (cfg.get("tier2.ner.model") if cfg else None) or "en_core_web_sm"
        if spacy is not None:
            try:
                self._nlp = spacy.load(model)
                self.backend = "spacy"
            except Exception:  # noqa: BLE001
                self._nlp = None
        if self._nlp is None:
            self.backend = "regex_fallback"

    @property
    def is_real(self) -> bool:
        return self.backend == "spacy"

    def run(self, text: str) -> list[Entity]:
        if not text:
            return []
        if self._nlp is not None:
            doc = self._nlp(text)
            return [Entity(text=e.text, type=_SPACY_MAP.get(e.label_, "OTHER"),
                           span=(e.start_char, e.end_char)) for e in doc.ents]
        out: list[Entity] = []
        for m in _PROPER.finditer(text):
            out.append(Entity(text=m.group(1), type="OTHER", span=(m.start(1), m.end(1))))
        return out
