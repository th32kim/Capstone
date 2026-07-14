"""Reciprocal Rank Fusion (k=60). Fuses RANKS, not scores (CLAUDE.md §6).

Cosine distance and graph recency x degree are incommensurable; RRF sidesteps fragile
per-channel normalisation by combining ranks: score(d) = sum_c 1 / (k + rank_c(d)).
"""

from __future__ import annotations

from collections import defaultdict


def rrf_fuse(channels: dict[str, list[int]], k: int = 60) -> list[tuple[int, float]]:
    """channels: name -> ranked list of ids (rank 0 = best). Returns fused (id, score) desc."""
    score: dict[int, float] = defaultdict(float)
    for ranked in channels.values():
        for rank, doc_id in enumerate(ranked):
            score[doc_id] += 1.0 / (k + rank)
    return sorted(score.items(), key=lambda kv: -kv[1])


def rank_of(ranked: list[int], doc_id: int) -> int | None:
    try:
        return ranked.index(doc_id)
    except ValueError:
        return None
