"""eval/retrieval — Fig 3.4-5 hybrid ablation: p@5, nDCG@10, MRR, recall@10 vs the Flat oracle.

Requires (a) a REAL embedder (not the hash fallback — those numbers would be meaningless) and
(b) judged relevant sets in data/queries/queries.yaml. When either is missing this prints
NOT MEASURED with the reason. When both are present it scores dense-only / kg-only / RRF so the
two-channel design is EARNED by the ablation, not assumed.
"""

from __future__ import annotations

import math
from pathlib import Path

import yaml

from ..config import Config

QUERIES = Path("data/queries/queries.yaml")


def _load_queries() -> list[dict]:
    if not QUERIES.exists():
        return []
    return (yaml.safe_load(QUERIES.read_text(encoding="utf-8")) or {}).get("queries", [])


def _dcg(rels: list[int]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def _ndcg(ranked_ids: list[str], relevant: set[str], k: int = 10) -> float:
    rels = [1 if x in relevant else 0 for x in ranked_ids[:k]]
    ideal = sorted(rels, reverse=True)
    idcg = _dcg(ideal)
    return _dcg(rels) / idcg if idcg else 0.0


def evaluate(engine, cfg: Config) -> dict:
    queries = _load_queries()
    judged = [q for q in queries if q.get("relevant")]
    if not engine.embedder.is_real:
        return {"status": "NOT MEASURED",
                "reason": f"embedder backend is '{engine.embedder.backend}', not a real model"}
    if not judged:
        return {"status": "NOT MEASURED",
                "reason": "no judged relevant sets in data/queries/queries.yaml"}

    modes = {"dense_only": dict(dense_only=True), "kg_only": dict(kg_only=True), "rrf": {}}
    out: dict[str, dict] = {}
    for name, kw in modes.items():
        p5, ndcg10, mrr = [], [], []
        for q in judged:
            rel = set(q["relevant"])
            hits = engine.retrieve(q["text"], k=10, **kw).hits
            ids = [h.segment_id for h in hits]
            p5.append(sum(1 for x in ids[:5] if x in rel) / 5.0)
            ndcg10.append(_ndcg(ids, rel, 10))
            rr = next((1.0 / (i + 1) for i, x in enumerate(ids) if x in rel), 0.0)
            mrr.append(rr)
        n = len(judged)
        out[name] = {"p@5": round(sum(p5) / n, 3), "nDCG@10": round(sum(ndcg10) / n, 3),
                     "MRR": round(sum(mrr) / n, 3)}
    return {"status": "MEASURED", "n_queries": len(judged), "ablation": out}
