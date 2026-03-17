-- sql/init.sql

CREATE EXTENSION IF NOT EXISTS vector;

-- ── Dokumente ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS documents (
    id              BIGSERIAL    PRIMARY KEY,
    content         TEXT,
    file_data       TEXT,
    content_type    VARCHAR(20)  NOT NULL,
    source          TEXT,
    title           TEXT,                        -- Referenz: Dokumenttitel
    chunk_index     INT          DEFAULT 0,      -- Referenz: Chunk-Nummer
    chunk_total     INT          DEFAULT 1,      -- Referenz: Gesamtzahl Chunks
    metadata        JSONB        DEFAULT '{}',   -- Referenz: flexibles Metadaten-JSON
    transcript      TEXT,                        -- Whisper-Volltranskript
    timestamp_start FLOAT,                       -- Sekunden (Audio/Video)
    timestamp_end   FLOAT,
    embedding       vector(3072),
    created_at      TIMESTAMPTZ  DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_documents_content_type ON documents (content_type);
CREATE INDEX IF NOT EXISTS idx_documents_source       ON documents (source);
CREATE INDEX IF NOT EXISTS idx_documents_created_at   ON documents (created_at DESC);

-- ── Chat-Sessions ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS chat_sessions (
    id          BIGSERIAL    PRIMARY KEY,
    name        TEXT         NOT NULL,
    created_at  TIMESTAMPTZ  DEFAULT NOW(),
    updated_at  TIMESTAMPTZ  DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id          BIGSERIAL    PRIMARY KEY,
    session_id  BIGINT       NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role        VARCHAR(10)  NOT NULL,
    content     TEXT         NOT NULL,
    sources     JSONB,
    created_at  TIMESTAMPTZ  DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages (session_id, created_at);

-- ── RPC: match_documents ─────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION match_documents(
    query_embedding   vector(3072),
    match_count       INT     DEFAULT 5,
    similarity_thresh FLOAT   DEFAULT 0.75,
    filter_type       TEXT    DEFAULT NULL
)
RETURNS TABLE (
    id              BIGINT,
    content         TEXT,
    file_data       TEXT,
    content_type    VARCHAR(20),
    source          TEXT,
    title           TEXT,
    chunk_index     INT,
    chunk_total     INT,
    metadata        JSONB,
    transcript      TEXT,
    timestamp_start FLOAT,
    timestamp_end   FLOAT,
    similarity      FLOAT
)
LANGUAGE plpgsql AS $$
BEGIN
    RETURN QUERY
    SELECT
        d.id, d.content, d.file_data, d.content_type, d.source,
        d.title, d.chunk_index, d.chunk_total, d.metadata,
        d.transcript, d.timestamp_start, d.timestamp_end,
        (1 - (d.embedding <=> query_embedding))::FLOAT AS similarity
    FROM documents d
    WHERE
        (filter_type IS NULL OR d.content_type = filter_type)
        AND (1 - (d.embedding <=> query_embedding)) >= similarity_thresh
    ORDER BY d.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;