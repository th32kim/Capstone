"""SqliteStore — SQLite is the SOURCE OF TRUTH; FAISS is a derived cache (CLAUDE.md §1.4/§6).

Ingest is ONE transaction: segment -> SEGMENT; entities -> ENTITY + MENTION; co-mentions ->
RELATION(CO_OCCURS). The vector is appended to the FAISS sidecar under `embedding_id` after
the SQLite commit succeeds.

The frozen schema keys SEGMENT by an INTEGER id; the pipeline keys by a string id
("clip01_seg000"). We let SQLite assign the integer PK, reuse it as `embedding_id`, and map
int<->string in the `meta` table — so `schema.sql` stays verbatim and the string ids the
contracts use survive round-trip.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from ..contracts import EMBEDDING_DIM, MemoryRecord

SCHEMA = Path(__file__).with_name("schema.sql")


def _canonical(name: str) -> str:
    return " ".join(name.strip().lower().split())


class SqliteStore:
    def __init__(self, db_path: str | Path, config_hash: str = "") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: the localhost API serves requests from a worker thread; access
        # is serialised by the single-user prototype, so this is safe (and standard for the pattern).
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA.read_text())
        self.conn.commit()
        if config_hash:
            self._set_meta("config_hash", config_hash)
        self._set_meta("schema_version", self._get_meta("schema_version") or "1")
        self._set_meta("embedding_dim", str(EMBEDDING_DIM))

    # -- meta --------------------------------------------------------------
    def _set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value))
        self.conn.commit()

    def _get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    # -- sessions ----------------------------------------------------------
    def create_session(self, source_uri: str, source_kind: str, started_at: str,
                        duration_s: float, fps: float | None, sample_rate: int | None) -> int:
        cur = self.conn.execute(
            "INSERT INTO session(source_uri,source_kind,started_at,duration_s,fps,sample_rate) "
            "VALUES(?,?,?,?,?,?)",
            (source_uri, source_kind, started_at, duration_s, fps, sample_rate))
        self.conn.commit()
        return int(cur.lastrowid)

    # -- ingest ------------------------------------------------------------
    def ingest(self, rec: MemoryRecord, session_id: int, index=None) -> int:
        """Insert one MemoryRecord in a single transaction. Returns the integer segment id."""
        rec.validate()  # d==384, t_end>t_start — loud failure, never silent coercion
        conn = self.conn
        try:
            conn.execute("BEGIN")
            cur = conn.execute(
                "INSERT INTO segment(session_id,start_ts,end_ts,salience_score,validator_verdict,"
                "validator_confidence,validator_reason,summary,transcript_ref,thumbnail_ref,"
                "embedding_id,live) VALUES(?,?,?,?,?,?,?,?,?,?,?,1)",
                (session_id, rec.t_start, rec.t_end, rec.salience_peak, rec.validator_verdict,
                 rec.validator_confidence, rec.validator_reason, rec.summary, rec.transcript,
                 None, None))
            seg_int = int(cur.lastrowid)
            conn.execute("UPDATE segment SET embedding_id=? WHERE segment_id=?", (seg_int, seg_int))
            conn.execute("INSERT INTO meta(key,value) VALUES(?,?) "
                         "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                         (f"extid:{seg_int}", rec.segment_id))
            conn.execute("INSERT INTO meta(key,value) VALUES(?,?) "
                         "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                         (f"intid:{rec.segment_id}", str(seg_int)))

            entity_ids = self._upsert_entities(rec, seg_int)
            self._insert_comentions(entity_ids, rec.t_end)
            conn.commit()
        except Exception:
            conn.rollback()
            raise

        if index is not None and rec.embedding:
            index.add(seg_int, rec.embedding)  # FAISS sidecar, AFTER the SQLite commit
        return seg_int

    def _modality_for_span(self, rec: MemoryRecord, span: tuple[int, int]) -> str:
        # text = transcript "\n" caption "\n" ocr_text  (tier2 aggregation order)
        n_t = len(rec.transcript)
        n_c = len(rec.caption)
        start = span[0]
        if start <= n_t:
            return "transcript"
        if start <= n_t + 1 + n_c:
            return "caption"
        return "ocr"

    def _upsert_entities(self, rec: MemoryRecord, seg_int: int) -> list[int]:
        conn = self.conn
        ids: list[int] = []
        for ent in rec.entities:
            canon = _canonical(ent.text)
            row = conn.execute("SELECT entity_id FROM entity WHERE canonical_name=? AND type=?",
                               (canon, ent.type)).fetchone()
            if row:
                eid = int(row["entity_id"])
                conn.execute("UPDATE entity SET last_ts=? WHERE entity_id=?", (rec.t_end, eid))
            else:
                cur = conn.execute(
                    "INSERT INTO entity(canonical_name,type,first_ts,last_ts) VALUES(?,?,?,?)",
                    (canon, ent.type, rec.t_start, rec.t_end))
                eid = int(cur.lastrowid)
            conn.execute(
                "INSERT OR IGNORE INTO mention(segment_id,entity_id,modality,span_start,span_end) "
                "VALUES(?,?,?,?,?)",
                (seg_int, eid, self._modality_for_span(rec, ent.span), ent.span[0], ent.span[1]))
            ids.append(eid)
        return ids

    def _insert_comentions(self, entity_ids: list[int], ts: float) -> None:
        uniq = sorted(set(entity_ids))
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                h, t = uniq[i], uniq[j]
                self.conn.execute(
                    "INSERT INTO relation(head_id,tail_id,predicate,weight,last_ts) "
                    "VALUES(?,?, 'CO_OCCURS', 1.0, ?) "
                    "ON CONFLICT(head_id,tail_id,predicate) "
                    "DO UPDATE SET weight = weight + 1.0, last_ts=excluded.last_ts",
                    (h, t, ts))

    # -- deletion (the privacy differentiator) -----------------------------
    def tombstone(self, segment_ref: str | int) -> None:
        """Set live=0 in one transaction. Retrieval filters non-live before the index rebuilds."""
        seg_int = self._to_int(segment_ref)
        self.conn.execute("BEGIN")
        self.conn.execute("UPDATE segment SET live=0 WHERE segment_id=?", (seg_int,))
        self.conn.commit()

    def _to_int(self, ref: str | int) -> int:
        if isinstance(ref, int):
            return ref
        v = self._get_meta(f"intid:{ref}")
        if v is None:
            raise KeyError(f"unknown segment id {ref!r}")
        return int(v)

    def ext_id(self, seg_int: int) -> str:
        return self._get_meta(f"extid:{seg_int}") or str(seg_int)

    # -- queries -----------------------------------------------------------
    def live_embedding_ids(self) -> set[int]:
        rows = self.conn.execute("SELECT embedding_id FROM live_segment WHERE embedding_id IS NOT NULL")
        return {int(r["embedding_id"]) for r in rows}

    def all_live_vectors_meta(self) -> list[int]:
        rows = self.conn.execute("SELECT embedding_id FROM live_segment "
                                 "WHERE embedding_id IS NOT NULL ORDER BY embedding_id")
        return [int(r["embedding_id"]) for r in rows]

    def is_live(self, seg_int: int) -> bool:
        row = self.conn.execute("SELECT live FROM segment WHERE segment_id=?", (seg_int,)).fetchone()
        return bool(row and row["live"])

    def hydrate(self, seg_int: int) -> dict | None:
        row = self.conn.execute("SELECT * FROM segment WHERE segment_id=?", (seg_int,)).fetchone()
        if row is None:
            return None
        ents = self.conn.execute(
            "SELECT e.canonical_name, e.type FROM mention m JOIN entity e ON e.entity_id=m.entity_id "
            "WHERE m.segment_id=?", (seg_int,)).fetchall()
        return {
            "segment_id": self.ext_id(seg_int), "int_id": seg_int,
            "t_start": row["start_ts"], "t_end": row["end_ts"],
            "summary": row["summary"] or "", "live": bool(row["live"]),
            "entities": tuple(f"{e['canonical_name']} ({e['type']})" for e in ents),
        }

    def close(self) -> None:
        self.conn.close()
