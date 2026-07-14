"""Retention / deletion — the project's headline privacy argument (CLAUDE.md §1.4/§6).

Deletion is NOT a FAISS operation (HNSW has no in-place removal). It is:
  1. tombstone in SQLite (one transaction): live = 0,
  2. retrieval filters out any embedding_id whose segment is not live (immediate effect),
  3. the index is rebuilt from live rows on a schedule (purges the vector for good).

Step 2 is what makes a deleted memory unretrievable *before* the rebuild — the property the
`test_deleted_memory_unretrievable_before_rebuild` test pins down.
"""

from __future__ import annotations

from dataclasses import dataclass

from .faiss_index import VectorIndex
from .sqlite_store import SqliteStore


def delete_memory(store: SqliteStore, segment_ref) -> None:
    """Tombstone only. The vector is still in the index until the next rebuild — retrieval
    filters it out in the meantime (that is the point)."""
    store.tombstone(segment_ref)


def rebuild_from_live(store: SqliteStore, index: VectorIndex) -> int:
    """Rebuild the vector index from live rows only, purging tombstoned vectors. Returns count."""
    live_ids = store.all_live_vectors_meta()
    vecs = index.vectors_for(live_ids)
    index.rebuild(live_ids, vecs)
    return len(live_ids)


@dataclass
class RetentionPolicy:
    rebuild_every_n_deletes: int = 50
    _deletes_since_rebuild: int = 0

    @classmethod
    def from_config(cls, cfg) -> "RetentionPolicy":
        return cls(int(cfg.get("store.retention.rebuild_every_n_deletes", 50)))

    def on_delete(self, store: SqliteStore, index: VectorIndex, segment_ref) -> bool:
        """Delete one memory; rebuild if the threshold is reached. Returns True if a rebuild ran."""
        delete_memory(store, segment_ref)
        self._deletes_since_rebuild += 1
        if self._deletes_since_rebuild >= self.rebuild_every_n_deletes:
            rebuild_from_live(store, index)
            self._deletes_since_rebuild = 0
            return True
        return False
