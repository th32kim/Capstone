"""FastAPI server — bound to localhost ONLY (CLAUDE.md §8, NFS4).

    POST   /query          {query, k, dense_only?, kg_only?} -> ranked hits + per-stage latency
    GET    /memory/{id}    -> one hydrated memory
    DELETE /memory/{id}    -> tombstone (unretrievable immediately, before index rebuild)

The DELETE endpoint is the privacy story made operational: it tombstones in SQLite and returns;
the very next /query cannot surface it. Host is forced to 127.0.0.1 in `run()`.
"""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from ..config import Config, load_config
from ..store.faiss_index import VectorIndex
from ..store.retention import delete_memory
from ..store.sqlite_store import SqliteStore
from ..retrieval.engine import RetrievalEngine


class QueryReq(BaseModel):
    # module-level (not local to create_app) so FastAPI resolves it as the request body
    query: str
    k: Optional[int] = None
    dense_only: bool = False
    kg_only: bool = False


def create_app(cfg: Config | None = None):
    from fastapi import FastAPI, HTTPException

    cfg = cfg or load_config("default")
    store = SqliteStore(cfg.get("store.db_path"), config_hash=cfg.config_hash)
    index = VectorIndex.load(cfg.get("store.index_path"), cfg=cfg)
    engine = RetrievalEngine(store, index, cfg)

    app = FastAPI(title="Hindsight", version="0.1.0")

    @app.post("/query")
    def query_ep(req: QueryReq):
        res = engine.retrieve(req.query, req.k, dense_only=req.dense_only, kg_only=req.kg_only)
        return {
            "query": res.query,
            "hits": [
                {"segment_id": h.segment_id, "score": h.score, "t_start": h.t_start,
                 "t_end": h.t_end, "summary": h.summary, "entities": list(h.entities),
                 "dense_rank": h.dense_rank, "kg_rank": h.kg_rank}
                for h in res.hits
            ],
            "latency_ms": res.latency_ms,
        }

    @app.get("/memory/{segment_id}")
    def get_memory(segment_id: str):
        try:
            sid = store._to_int(segment_id)
        except KeyError:
            raise HTTPException(404, "unknown memory")
        h = store.hydrate(sid)
        if h is None or not h["live"]:
            raise HTTPException(404, "unknown or deleted memory")
        return h

    @app.delete("/memory/{segment_id}")
    def delete_memory_ep(segment_id: str):
        try:
            delete_memory(store, segment_id)
        except KeyError:
            raise HTTPException(404, "unknown memory")
        return {"deleted": segment_id, "note": "tombstoned; unretrievable immediately"}

    return app


def run(cfg: Config | None = None, port: int = 8099) -> None:  # pragma: no cover - server
    import uvicorn

    uvicorn.run(create_app(cfg), host="127.0.0.1", port=port)  # localhost only
