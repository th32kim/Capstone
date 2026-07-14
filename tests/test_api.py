"""Localhost API: query + the deletion-before-rebuild privacy guarantee over HTTP."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from hindsight.config import load_config  # noqa: E402
from hindsight.store import SqliteStore, VectorIndex  # noqa: E402
from tests.conftest import make_record  # noqa: E402


def _seed(cfg, embedder, tmp_path):
    store = SqliteStore(tmp_path / "h.db", config_hash=cfg.config_hash)
    index = VectorIndex(cfg=cfg)
    sid = store.create_session("u", "clip", "2026-01-01T09:00:00", 100.0, 12.0, 16000)
    for i, txt in enumerate(["apple orchard fruit", "diesel engine turbine"]):
        store.ingest(make_record(f"a_seg{i:03d}", txt, embedder, t_start=i * 20.0, t_end=i * 20.0 + 10), sid, index)
    index.save(tmp_path / "index.faiss")
    store.close()


def test_api_query_and_delete(tmp_path, embedder):
    cfg = load_config("default").with_overrides(**{
        "store.db_path": str(tmp_path / "h.db"), "store.index_path": str(tmp_path / "index.faiss")})
    _seed(cfg, embedder, tmp_path)

    import hindsight.api.server as srv

    client = TestClient(srv.create_app(cfg))
    before = client.post("/query", json={"query": "apple orchard", "k": 2}).json()
    assert before["hits"][0]["segment_id"] == "a_seg000"

    assert client.delete("/memory/a_seg000").status_code == 200
    after = [h["segment_id"] for h in client.post("/query", json={"query": "apple orchard", "k": 2}).json()["hits"]]
    assert "a_seg000" not in after                      # gone immediately, before any rebuild
    assert client.get("/memory/a_seg000").status_code == 404
