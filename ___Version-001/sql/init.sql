-- sql/init.sql
-- Wird beim ersten Start des pgvector-db-Containers automatisch ausgeführt.
-- Pfad im Container: /docker-entrypoint-initdb.d/init.sql

-- ── Extension ────────────────────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS vector;

-- ── Tabelle ──────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS documents (
    id           BIGSERIAL    PRIMARY KEY,
    content      TEXT,                         -- Textinhalt (NULL für Bild/Audio/Video)
    file_data    TEXT,                         -- Base64-kodierte Bilddaten
    content_type VARCHAR(20)  NOT NULL,        -- text | pdf | image | audio | video
    source       TEXT,                         -- Originaldateiname
    embedding    vector(3072),                 -- Gemini-Embedding, 3072 Dimensionen
    created_at   TIMESTAMPTZ  DEFAULT NOW()
);

-- Hinweis: Kein HNSW-Index — pgvector unterstützt HNSW nur bis 2000 Dimensionen.
-- Bei 3072 Dims wird Exact Search (Sequential Scan) verwendet.
-- IVFFlat als Alternative:
--   CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
--   Erfordert VACUUM ANALYZE nach größeren Bulk-Inserts.

-- ── Index auf häufig gefilterte Spalten ──────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_documents_content_type
    ON documents (content_type);

CREATE INDEX IF NOT EXISTS idx_documents_source
    ON documents (source);

CREATE INDEX IF NOT EXISTS idx_documents_created_at
    ON documents (created_at DESC);

-- ── RPC: match_documents ─────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION match_documents(
    query_embedding   vector(3072),
    match_count       INT     DEFAULT 5,
    similarity_thresh FLOAT   DEFAULT 0.75,
    filter_type       TEXT    DEFAULT NULL     -- NULL = alle Content-Types
)
RETURNS TABLE (
    id           BIGINT,
    content      TEXT,
    file_data    TEXT,
    content_type VARCHAR(20),
    source       TEXT,
    similarity   FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        d.id,
        d.content,
        d.file_data,
        d.content_type,
        d.source,
        (1 - (d.embedding <=> query_embedding))::FLOAT AS similarity
    FROM documents d
    WHERE
        (filter_type IS NULL OR d.content_type = filter_type)
        AND (1 - (d.embedding <=> query_embedding)) >= similarity_thresh
    ORDER BY
        d.embedding <=> query_embedding   -- aufsteigend = beste zuerst
    LIMIT match_count;
END;
$$;

-- ── Hilfsfunktion: Statistiken ───────────────────────────────────────────────
CREATE OR REPLACE FUNCTION get_document_stats()
RETURNS TABLE (content_type VARCHAR(20), chunk_count BIGINT)
LANGUAGE sql
AS $$
    SELECT content_type, COUNT(*) AS chunk_count
    FROM documents
    GROUP BY content_type
    ORDER BY chunk_count DESC;
$$;