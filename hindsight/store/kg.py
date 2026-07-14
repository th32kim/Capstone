"""Knowledge-graph view: O(1) entity resolution + bounded BFS expansion (CLAUDE.md §6).

ENTITY.canonical_name is a unique index -> resolve is a dict lookup. RELATION(CO_OCCURS) is
the always-populated edge set; WORKS_AT/LOCATED_IN are left empty rather than faked. The
neighbourhood is materialised into an in-memory graph (NetworkX if present, else a plain
adjacency dict) for bounded traversal at query time.

The KG channel returns *segment* ids ranked by recency x degree, over LIVE segments only.
"""

from __future__ import annotations

from collections import defaultdict, deque

from .sqlite_store import SqliteStore, _canonical


class KgView:
    def __init__(self, store: SqliteStore) -> None:
        self.store = store
        self._adj: dict[int, dict[int, float]] = defaultdict(dict)
        self._canon: dict[tuple[str, str], int] = {}
        self._degree: dict[int, float] = defaultdict(float)
        self._load()

    def _load(self) -> None:
        conn = self.store.conn
        for r in conn.execute("SELECT entity_id, canonical_name, type FROM entity"):
            self._canon[(r["canonical_name"], r["type"])] = int(r["entity_id"])
        for r in conn.execute("SELECT head_id, tail_id, weight FROM relation WHERE predicate='CO_OCCURS'"):
            h, t, w = int(r["head_id"]), int(r["tail_id"]), float(r["weight"])
            self._adj[h][t] = w
            self._adj[t][h] = w
            self._degree[h] += w
            self._degree[t] += w

    def resolve(self, text: str, type_hint: str | None = None) -> int | None:
        canon = _canonical(text)
        if type_hint and (canon, type_hint) in self._canon:
            return self._canon[(canon, type_hint)]
        for (c, _t), eid in self._canon.items():
            if c == canon:
                return eid
        return None

    def expand(self, entity_id: int, hops: int = 2, branching_cap: int = 8) -> set[int]:
        """Bounded BFS: <= branching_cap neighbours per node, up to `hops` deep."""
        seen = {entity_id}
        frontier = deque([(entity_id, 0)])
        while frontier:
            node, depth = frontier.popleft()
            if depth >= hops:
                continue
            neighbours = sorted(self._adj[node].items(), key=lambda kv: -kv[1])[:branching_cap]
            for nb, _w in neighbours:
                if nb not in seen:
                    seen.add(nb)
                    frontier.append((nb, depth + 1))
        return seen

    def rank_segments(self, entity_ids: set[int]) -> list[tuple[int, float]]:
        """Live segments mentioning any expanded entity, ranked by recency x degree."""
        if not entity_ids:
            return []
        conn = self.store.conn
        placeholders = ",".join("?" * len(entity_ids))
        rows = conn.execute(
            f"SELECT m.segment_id AS sid, m.entity_id AS eid, s.end_ts AS ts "
            f"FROM mention m JOIN live_segment s ON s.segment_id=m.segment_id "
            f"WHERE m.entity_id IN ({placeholders})", tuple(entity_ids)).fetchall()
        if not rows:
            return []
        max_ts = max(r["ts"] for r in rows) or 1.0
        scores: dict[int, float] = defaultdict(float)
        for r in rows:
            recency = (r["ts"] / max_ts) if max_ts else 1.0
            scores[int(r["sid"])] += recency * (1.0 + self._degree.get(int(r["eid"]), 0.0))
        return sorted(scores.items(), key=lambda kv: -kv[1])
