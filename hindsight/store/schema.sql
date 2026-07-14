-- Hindsight memory store — SQLite is the SOURCE OF TRUTH.
-- The FAISS sidecar is a DERIVED CACHE. It holds vectors keyed by embedding_id and nothing else.
--
-- Why this split: faiss.IndexHNSWFlat has no in-place vector removal. Deletion therefore cannot be
-- a single cross-file transaction. Instead: tombstone here (one SQLite transaction), have retrieval
-- filter out any embedding_id that is no longer live, and rebuild the index from live rows on a
-- schedule. Consistency comes from making SQLite authoritative — not from a transaction that FAISS
-- cannot join. This is the project's headline privacy argument; it must actually be true.

PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS meta (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL          -- includes next_embedding_id, config_hash, schema_version
);

CREATE TABLE IF NOT EXISTS session (
    session_id      INTEGER PRIMARY KEY,
    source_uri      TEXT    NOT NULL,
    source_kind     TEXT    NOT NULL,  -- clip | webcam | glasses | gopro
    started_at      TEXT    NOT NULL,  -- ISO-8601
    duration_s      REAL    NOT NULL,
    fps             REAL,
    sample_rate     INTEGER,
    notes           TEXT
);

CREATE TABLE IF NOT EXISTS segment (
    segment_id           INTEGER PRIMARY KEY,
    session_id           INTEGER NOT NULL REFERENCES session(session_id) ON DELETE CASCADE,
    start_ts             REAL    NOT NULL,   -- master monotonic clock (P1 contract)
    end_ts               REAL    NOT NULL,
    salience_score       REAL    NOT NULL,   -- the fused Tier-1 salience that gated this segment
    validator_verdict    TEXT,               -- auto_accept | ai_keep | null   (ai_drop never lands here)
    validator_confidence REAL,
    validator_reason     TEXT,
    summary              TEXT,
    transcript_ref       TEXT,               -- handle to the full transcript
    thumbnail_ref        TEXT,               -- handle to a representative keyframe
    embedding_id         INTEGER UNIQUE,     -- cross-store handle into the FAISS sidecar
    live                 INTEGER NOT NULL DEFAULT 1   -- 0 = tombstoned. RETRIEVAL MUST FILTER ON THIS.
);
CREATE INDEX IF NOT EXISTS idx_segment_time ON segment(start_ts, end_ts);
CREATE INDEX IF NOT EXISTS idx_segment_live ON segment(live);

CREATE TABLE IF NOT EXISTS entity (
    entity_id       INTEGER PRIMARY KEY,
    canonical_name  TEXT    NOT NULL,
    type            TEXT    NOT NULL,        -- PERSON | ORG | GPE | LOC | PRODUCT | EVENT | ...
    first_ts        REAL,
    last_ts         REAL
);
-- O(1) entity resolution. This index IS the "kg.resolve()" in the retrieval pseudocode.
CREATE UNIQUE INDEX IF NOT EXISTS idx_entity_canon ON entity(canonical_name, type);

-- SEGMENT <-> ENTITY, many-to-many, carrying the modality the entity was seen in.
CREATE TABLE IF NOT EXISTS mention (
    segment_id  INTEGER NOT NULL REFERENCES segment(segment_id) ON DELETE CASCADE,
    entity_id   INTEGER NOT NULL REFERENCES entity(entity_id)   ON DELETE CASCADE,
    modality    TEXT    NOT NULL,            -- transcript | caption | ocr
    span_start  INTEGER NOT NULL,
    span_end    INTEGER NOT NULL,
    PRIMARY KEY (segment_id, entity_id, modality, span_start)
);
CREATE INDEX IF NOT EXISTS idx_mention_entity ON mention(entity_id);

-- ENTITY <-> ENTITY, the typed weighted graph edge.
--   CO_OCCURS  — derived free from shared mentions. ALWAYS populated.
--   WORKS_AT / LOCATED_IN — need relation extraction beyond NER. Populated opportunistically by an
--   entity-type-gated rule pass over co-mentions, and LEFT EMPTY rather than promised.
CREATE TABLE IF NOT EXISTS relation (
    head_id     INTEGER NOT NULL REFERENCES entity(entity_id) ON DELETE CASCADE,
    tail_id     INTEGER NOT NULL REFERENCES entity(entity_id) ON DELETE CASCADE,
    predicate   TEXT    NOT NULL,
    weight      REAL    NOT NULL DEFAULT 1.0,
    last_ts     REAL,
    PRIMARY KEY (head_id, tail_id, predicate)
);
CREATE INDEX IF NOT EXISTS idx_relation_head ON relation(head_id);

-- Query-time view: only live segments are ever retrievable. Retrieval joins through this,
-- never through `segment` directly.
CREATE VIEW IF NOT EXISTS live_segment AS
    SELECT * FROM segment WHERE live = 1;
