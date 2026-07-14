"""retrieve(): dense ⊕ KG → RRF(k=60) → constraints → top-k (CLAUDE.md §6).

The dense channel is live-filtered against SQLite BEFORE fusion — this is where deletion takes
effect immediately (a tombstoned embedding_id never reaches the results, even though its vector
is still in the index until the next rebuild). Ablations `--dense-only` / `--kg-only` are
first-class (CLAUDE.md §1.8). Per-stage latency is measured and returned as QueryResult.latency_ms
(this IS Table 3.4-7).
"""

from __future__ import annotations

import time
from datetime import datetime

from ..config import Config
from ..contracts import MemoryHit, QueryResult
from ..store.faiss_index import VectorIndex
from ..store.kg import KgView
from ..store.sqlite_store import SqliteStore
from ..tier2.embed import Embedder
from ..tier2.ner import NerRunner
from .constraints import apply_constraints, parse_constraints
from .rrf import rank_of, rrf_fuse


class RetrievalEngine:
    def __init__(self, store: SqliteStore, index: VectorIndex, cfg: Config,
                 embedder: Embedder | None = None, ner: NerRunner | None = None) -> None:
        self.store = store
        self.index = index
        self.cfg = cfg
        self.embedder = embedder or Embedder(cfg)
        self.ner = ner or NerRunner(cfg)
        self.k = int(cfg.get("retrieval.k", 5))
        self.dense_candidates = int(cfg.get("retrieval.dense_candidates", 50))
        self.rrf_k = int(cfg.get("retrieval.rrf_k", 60))

    def _session_start(self) -> datetime | None:
        row = self.store.conn.execute("SELECT started_at FROM session ORDER BY session_id LIMIT 1").fetchone()
        if not row:
            return None
        try:
            return datetime.fromisoformat(row["started_at"])
        except Exception:  # noqa: BLE001
            return None

    def retrieve(self, query: str, k: int | None = None, *, dense_only: bool = False,
                 kg_only: bool = False, now: datetime | None = None) -> QueryResult:
        k = k or self.k
        lat: dict[str, float] = {}

        t = time.perf_counter()
        qvec = self.embedder.embed(query)
        lat["embed"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        q_entities = self.ner.run(query)
        lat["ner"] = (time.perf_counter() - t) * 1000

        channels: dict[str, list[int]] = {}

        if not kg_only:
            t = time.perf_counter()
            live = self.store.live_embedding_ids()
            raw = self.index.search(qvec, self.dense_candidates)
            dense = [eid for eid, _ in raw if eid in live]  # <-- deletion takes effect here
            lat["ann"] = (time.perf_counter() - t) * 1000
            channels["dense"] = dense
        if not dense_only:
            t = time.perf_counter()
            kg = KgView(self.store)
            seeds: set[int] = set()
            for e in q_entities:
                eid = kg.resolve(e.text, e.type)
                if eid is not None:
                    seeds |= kg.expand(eid, hops=self.cfg.get("store.kg.max_hops", 2),
                                       branching_cap=self.cfg.get("store.kg.branching_cap", 8))
            kg_hits = [sid for sid, _ in kg.rank_segments(seeds)]
            lat["kg"] = (time.perf_counter() - t) * 1000
            channels["kg"] = kg_hits

        t = time.perf_counter()
        fused = rrf_fuse(channels, k=self.rrf_k)
        lat["rrf"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        hydrated = []
        for sid, fscore in fused:
            h = self.store.hydrate(sid)
            if h is None or not h["live"]:
                continue
            h["_score"] = fscore
            h["_dense_rank"] = rank_of(channels.get("dense", []), sid)
            h["_kg_rank"] = rank_of(channels.get("kg", []), sid)
            hydrated.append(h)
        cons = parse_constraints(query, now=now or datetime.now(), entities=q_entities)
        hydrated = apply_constraints(hydrated, cons, session_start=self._session_start())
        lat["sqlite"] = (time.perf_counter() - t) * 1000

        hits = tuple(
            MemoryHit(segment_id=h["segment_id"], score=h["_score"],
                      dense_rank=h["_dense_rank"], kg_rank=h["_kg_rank"],
                      t_start=h["t_start"], t_end=h["t_end"], summary=h["summary"],
                      entities=h["entities"], thumbnail_ref=None)
            for h in hydrated[:k])
        return QueryResult(query=query, hits=hits, latency_ms=lat)
