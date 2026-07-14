"""Extractive TextRank summariser — the DEFAULT because it cannot hallucinate (CLAUDE.md §5).

k=2 sentences, damping d=0.85, <= 30 power-iterations. Pure numpy; no model, no network,
zero-hallucination: it can only *select* sentences that were actually in the source text, so
it can never invent a memory that did not happen. That property matters more here than fluency.
"""

from __future__ import annotations

import re

import numpy as np

_SENT = re.compile(r"(?<=[.!?])\s+|\n+")


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(text or "") if s.strip()]


def _similarity(a: str, b: str) -> float:
    wa, wb = set(a.lower().split()), set(b.lower().split())
    if not wa or not wb:
        return 0.0
    inter = len(wa & wb)
    denom = np.log(len(wa) + 1) + np.log(len(wb) + 1)
    return inter / denom if denom > 0 else 0.0


def extractive_summary(text: str, k: int = 2, damping: float = 0.85, max_iter: int = 30) -> str:
    sents = _sentences(text)
    if len(sents) <= k:
        return " ".join(sents)
    n = len(sents)
    sim = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            s = _similarity(sents[i], sents[j])
            sim[i, j] = sim[j, i] = s
    row_sums = sim.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    transition = sim / row_sums

    scores = np.ones(n) / n
    for _ in range(max_iter):
        new = (1 - damping) / n + damping * (transition.T @ scores)
        if np.allclose(new, scores, atol=1e-6):
            scores = new
            break
        scores = new
    top = sorted(sorted(range(n), key=lambda i: -scores[i])[:k])  # keep source order
    return " ".join(sents[i] for i in top)
