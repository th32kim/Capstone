"""Vector index: FAISS IndexHNSWFlat (derived cache) with an exact numpy fallback.

CLAUDE.md §6: HNSW (M=16, efConstruction=200, efSearch=64) over L2-normalised vectors
(cosine == inner product), wrapped so `embedding_id` is ours, with an exact IndexFlat kept
as the recall ORACLE. FAISS has no in-place vector removal, so deletion is handled upstream
(SQLite tombstone + retrieval filter + periodic rebuild) — never `remove_ids` on HNSW.

When faiss is not installed, an exact numpy flat index is used for BOTH the main index and
the oracle. That is honest: search is exact, so recall@k vs the oracle is trivially 1.0, and
the index-latency benchmark prints `faiss unavailable` rather than a fake HNSW number.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .._deps import optional
from ..contracts import EMBEDDING_DIM


def _l2norm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, 1e-12, None)


class _NumpyFlat:
    """Exact inner-product index over L2-normalised vectors. Backend-free."""

    backend = "numpy_flat_exact"

    def __init__(self, d: int = EMBEDDING_DIM) -> None:
        self.d = d
        self._ids: list[int] = []
        self._mat = np.zeros((0, d), dtype=np.float32)

    def add(self, emb_id: int, vec: np.ndarray) -> None:
        self._ids.append(int(emb_id))
        self._mat = np.vstack([self._mat, _l2norm(vec).reshape(1, -1)])

    def search(self, q: np.ndarray, k: int) -> list[tuple[int, float]]:
        if not self._ids:
            return []
        sims = (_l2norm(q).reshape(1, -1) @ self._mat.T).ravel()
        order = np.argsort(-sims)[:k]
        return [(self._ids[i], float(sims[i])) for i in order]

    def rebuild(self, ids: list[int], vecs: np.ndarray) -> None:
        self._ids = [int(i) for i in ids]
        self._mat = _l2norm(vecs).astype(np.float32) if len(ids) else np.zeros((0, self.d), np.float32)

    def vectors_for(self, ids: list[int]) -> np.ndarray:
        pos = {i: k for k, i in enumerate(self._ids)}
        rows = [self._mat[pos[i]] for i in ids if i in pos]
        return np.vstack(rows) if rows else np.zeros((0, self.d), np.float32)

    def __len__(self) -> int:
        return len(self._ids)

    def save(self, path: Path) -> None:
        np.savez(path, ids=np.array(self._ids, dtype=np.int64), mat=self._mat)

    @classmethod
    def load(cls, path: Path, d: int = EMBEDDING_DIM) -> "_NumpyFlat":
        idx = cls(d)
        data = np.load(path)
        idx._ids = data["ids"].tolist()
        idx._mat = data["mat"].astype(np.float32)
        return idx


class VectorIndex:
    """Facade over faiss (if present) or the numpy fallback. d is FROZEN at 384."""

    def __init__(self, cfg=None, d: int = EMBEDDING_DIM) -> None:
        if d != EMBEDDING_DIM:
            raise ValueError(f"d={d} != frozen EMBEDDING_DIM {EMBEDDING_DIM}")
        self.d = d
        self.cfg = cfg
        self._faiss = optional("faiss")
        self._impl = _NumpyFlat(d)          # main index
        self._oracle = _NumpyFlat(d)        # exact recall oracle (always numpy-exact)
        self.backend = self._impl.backend
        if self._faiss is not None and cfg is not None:
            self._build_faiss(cfg)

    def _build_faiss(self, cfg) -> None:  # pragma: no cover - needs faiss
        faiss = self._faiss
        fc = cfg.get("store.faiss")
        base = faiss.IndexHNSWFlat(self.d, int(fc.get("M", 16)), faiss.METRIC_INNER_PRODUCT)
        base.hnsw.efConstruction = int(fc.get("ef_construction", 200))
        base.hnsw.efSearch = int(fc.get("ef_search", 64))
        self._impl = faiss.IndexIDMap2(base)
        self.backend = "faiss_hnsw"

    def add(self, emb_id: int, vec: np.ndarray) -> None:
        self._oracle.add(emb_id, vec)
        if self.backend == "faiss_hnsw":  # pragma: no cover
            self._impl.add_with_ids(_l2norm(vec).reshape(1, -1), np.array([emb_id], dtype=np.int64))
        else:
            self._impl.add(emb_id, vec)

    def search(self, q: np.ndarray, k: int, *, oracle: bool = False) -> list[tuple[int, float]]:
        idx = self._oracle if oracle else self._impl
        if oracle or self.backend != "faiss_hnsw":
            return idx.search(q, k)
        D, I = self._impl.search(_l2norm(q).reshape(1, -1), k)  # pragma: no cover
        return [(int(i), float(d)) for i, d in zip(I[0], D[0]) if i != -1]

    def rebuild(self, ids: list[int], vecs: np.ndarray) -> None:
        self._oracle.rebuild(ids, vecs)
        if self.backend == "faiss_hnsw":  # pragma: no cover
            self._build_faiss(self.cfg)
            if len(ids):
                self._impl.add_with_ids(_l2norm(vecs), np.array(ids, dtype=np.int64))
        else:
            self._impl.rebuild(ids, vecs)

    def vectors_for(self, ids: list[int]) -> np.ndarray:
        """Vectors (from the exact oracle store) for the given ids — used to rebuild from live rows."""
        return self._oracle.vectors_for(ids)

    def __len__(self) -> int:
        return len(self._oracle)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        if self.backend == "faiss_hnsw":  # pragma: no cover
            self._faiss.write_index(self._impl, str(path))
        else:
            self._impl.save(path.with_suffix(".npz"))

    @classmethod
    def load(cls, path: str | Path, cfg=None, d: int = EMBEDDING_DIM) -> "VectorIndex":
        """Load a persisted index. numpy .npz is the portable form; both indexes get the vectors."""
        path = Path(path)
        idx = cls(cfg=cfg, d=d)
        npz = path.with_suffix(".npz")
        if idx.backend != "faiss_hnsw" and npz.exists():
            flat = _NumpyFlat.load(npz, d)
            idx._impl = flat
            idx._oracle = _NumpyFlat.load(npz, d)
        elif idx._faiss is not None and path.exists():  # pragma: no cover
            idx._impl = idx._faiss.read_index(str(path))
            idx.backend = "faiss_hnsw"
        return idx
