"""Embedding — all-MiniLM-L6-v2, d=384, L2-normalised (CLAUDE.md §5, FROZEN interface).

d=384 is the P3<->P4 contract and is asserted. If sentence-transformers is not installed, a
DETERMINISTIC HASHING embedder is used so the store/retrieval plumbing runs end to end — but
it is loudly labelled `hash_fallback` and is NOT a semantic model. `eval/retrieval` refuses to
report retrieval quality numbers computed on the fallback (they would be meaningless).
"""

from __future__ import annotations

import hashlib

import numpy as np

from .._deps import optional
from ..contracts import EMBEDDING_DIM


class Embedder:
    def __init__(self, cfg=None) -> None:
        model_name = (cfg.get("tier2.embed.model") if cfg else None) or "all-MiniLM-L6-v2"
        self.model_name = model_name
        self._model = None
        st = optional("sentence_transformers")
        if st is not None:
            try:
                self._model = st.SentenceTransformer(model_name)
                self.backend = "sentence_transformers"
            except Exception:  # noqa: BLE001
                self._model = None
        if self._model is None:
            self.backend = "hash_fallback"

    @property
    def is_real(self) -> bool:
        return self.backend == "sentence_transformers"

    def embed(self, text: str) -> list[float]:
        if self._model is not None:
            vec = self._model.encode([text], normalize_embeddings=True)[0]
            vec = np.asarray(vec, dtype=np.float32)
        else:
            vec = self._hash_embed(text)
        if vec.shape[0] != EMBEDDING_DIM:
            raise ValueError(f"embedding dim {vec.shape[0]} != frozen {EMBEDDING_DIM}")
        return vec.astype(float).tolist()

    @staticmethod
    def _hash_embed(text: str) -> np.ndarray:
        """Deterministic bag-of-token-hashes projected to 384 dims, L2-normalised. Non-semantic."""
        vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        for tok in (text or "").lower().split():
            h = int(hashlib.sha1(tok.encode()).hexdigest(), 16)
            vec[h % EMBEDDING_DIM] += 1.0
        n = np.linalg.norm(vec)
        if n < 1e-9:
            vec[0] = 1.0  # avoid the zero vector
            n = 1.0
        return vec / n
