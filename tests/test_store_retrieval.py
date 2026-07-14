"""Store ingest/hydrate, the deletion privacy guarantee, and retrieval fusion."""

from __future__ import annotations

import pytest

from hindsight.contracts import Entity, MemoryRecord
from hindsight.retrieval.engine import RetrievalEngine
from hindsight.retrieval.rrf import rrf_fuse
from hindsight.store import SqliteStore, VectorIndex, rebuild_from_live
from hindsight.store.retention import delete_memory
from tests.conftest import make_record


def _store_with(records, cfg, tmp_path):
    store = SqliteStore(tmp_path / "h.db", config_hash=cfg.config_hash)
    index = VectorIndex(cfg=cfg)
    sid = store.create_session("uri", "clip", "2026-01-01T09:00:00", 100.0, 12.0, 16000)
    for r in records:
        store.ingest(r, sid, index)
    return store, index


def test_ingest_and_hydrate(cfg, embedder, tmp_path):
    recs = [
        make_record("c_seg000", "Mike discussed the data contract", embedder,
                    entities=[Entity("Mike", "PERSON", (0, 4))]),
        make_record("c_seg001", "the whiteboard had a block diagram", embedder, t_start=20, t_end=30),
    ]
    store, _index = _store_with(recs, cfg, tmp_path)
    sid = store._to_int("c_seg000")
    h = store.hydrate(sid)
    assert h["segment_id"] == "c_seg000"
    assert any("mike" in e.lower() for e in h["entities"])


def test_embedding_dim_enforced(cfg, tmp_path):
    r = MemoryRecord("x", 0.0, 10.0, "src", on_device=True)
    r.embedding = [0.0] * 100  # wrong dim
    store = SqliteStore(tmp_path / "h.db", config_hash=cfg.config_hash)
    sid = store.create_session("u", "clip", "2026-01-01T09:00:00", 10.0, 12.0, 16000)
    with pytest.raises(ValueError):
        store.ingest(r, sid)


def test_deleted_memory_unretrievable_before_rebuild(cfg, embedder, tmp_path):
    """M7 headline: delete a memory, do NOT rebuild the index, and it is gone from results —
    while its vector is still physically in the index (proving the live-filter, not a rebuild)."""
    recs = [
        make_record("c_seg000", "apple apple apple orchard fruit", embedder),
        make_record("c_seg001", "diesel engine turbine machinery", embedder, t_start=20, t_end=30),
        make_record("c_seg002", "mountain river forest hiking", embedder, t_start=40, t_end=50),
    ]
    store, index = _store_with(recs, cfg, tmp_path)
    engine = RetrievalEngine(store, index, cfg)

    before = [h.segment_id for h in engine.retrieve("apple orchard fruit", k=3).hits]
    assert "c_seg000" in before

    n_before = len(index)
    delete_memory(store, "c_seg000")          # tombstone only; NO rebuild

    after = [h.segment_id for h in engine.retrieve("apple orchard fruit", k=3).hits]
    assert "c_seg000" not in after            # unretrievable immediately
    assert len(index) == n_before             # vector is still physically present

    removed = rebuild_from_live(store, index)  # now purge it
    assert removed == 2 and len(index) == 2


def test_rrf_fuses_ranks():
    fused = rrf_fuse({"dense": [10, 20, 30], "kg": [20, 40]}, k=60)
    top = fused[0][0]
    assert top == 20  # appears in both channels -> highest fused score


def test_retrieval_ablations(cfg, embedder, tmp_path):
    recs = [make_record("c_seg000", "quantum computing lecture qubits", embedder),
            make_record("c_seg001", "gardening tomatoes soil", embedder, t_start=20, t_end=30)]
    store, index = _store_with(recs, cfg, tmp_path)
    engine = RetrievalEngine(store, index, cfg)
    dense = engine.retrieve("qubits quantum", k=2, dense_only=True).hits
    assert dense and dense[0].segment_id == "c_seg000"
    assert dense[0].kg_rank is None  # kg channel not run in dense-only
