"""eval/index_bench — Fig 3.4-2 (latency vs N) and Fig 3.4-4 (efSearch knee).

Exact-Flat single-query latency at N in {1k,10k,100k} is MEASURED here on SYNTHETIC vectors with
realistic cluster structure (a Gaussian mixture — NOT uniform random draws, which make ANN look
artificially easy; M9). HNSW / IVF-Flat / IVF-PQ and the efSearch sweep require faiss; when faiss
is absent they print NOT MEASURED rather than a fabricated curve.

Everything is labelled SYNTHETIC — in the caption and here in code.
"""

from __future__ import annotations

import time

import numpy as np

from .._deps import optional
from ..contracts import EMBEDDING_DIM


def _synthetic_corpus(n: int, d: int = EMBEDDING_DIM, n_clusters: int = 50, seed: int = 1337):
    """Clustered unit vectors (NOT uniform random). Labelled SYNTHETIC everywhere it is used."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 1, size=(n_clusters, d))
    assign = rng.integers(0, n_clusters, size=n)
    vecs = centers[assign] + rng.normal(0, 0.35, size=(n, d))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs.astype(np.float32)


def flat_latency(sizes=(1000, 10000, 100000), n_queries: int = 100) -> list[dict]:
    rng = np.random.default_rng(7)
    out = []
    for n in sizes:
        corpus = _synthetic_corpus(n)
        qs = corpus[rng.integers(0, n, size=n_queries)]
        t0 = time.perf_counter()
        for q in qs:
            sims = corpus @ q
            np.argpartition(-sims, 10)[:10]
        ms = (time.perf_counter() - t0) / n_queries * 1000
        out.append({"N": n, "index": "Flat(exact,numpy)", "ms_per_query": round(ms, 4),
                    "synthetic": True})
    return out


def report() -> dict:
    res = {"flat_synthetic": flat_latency()}
    if optional("faiss") is None:
        res["hnsw_ivf_pq"] = "NOT MEASURED (faiss not installed)"
        res["efsearch_sweep"] = "NOT MEASURED (faiss not installed)"
    else:  # pragma: no cover - needs faiss
        res["hnsw_ivf_pq"] = "run with faiss present to populate Fig 3.4-2/3.4-4"
    return res
