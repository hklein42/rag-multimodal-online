"""
lib/db.py
pgvector Datenbankschicht.

Operationen
───────────
insert_document()     Chunk + Embedding speichern
match_documents()     Cosine Similarity Suche via RPC
list_documents()      Paginiertes Browse
delete_document()     Löschen per ID
get_stats()           Aggregierte Zählungen pro content_type
"""

from __future__ import annotations

import os
from typing import Any

import psycopg2
import psycopg2.extras
import psycopg2.errors
import psycopg2.extensions
from pgvector.psycopg2 import register_vector

_conn: psycopg2.extensions.connection | None = None


# ── Connection ─────────────────────────────────────────────────────────────────

def _get_conn() -> psycopg2.extensions.connection:
    """Gibt eine (gecachte) Datenbankverbindung zurück. Reconnect bei Fehler."""
    global _conn
    if _conn is None or _conn.closed:
        url = os.environ["DATABASE_URL"]
        _conn = psycopg2.connect(url)
        register_vector(_conn)
        return _conn

    # Abgebrochene Transaktion zurückrollen (z. B. nach Python-Fehler im UI)
    if _conn.status == psycopg2.extensions.STATUS_IN_TRANSACTION:
        try:
            _conn.rollback()
        except Exception:
            _conn = None
            return _get_conn()

    return _conn


def _cursor() -> psycopg2.extensions.cursor:
    conn = _get_conn()
    try:
        return conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    except psycopg2.errors.InFailedSqlTransaction:
        conn.rollback()
        return conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    except psycopg2.OperationalError:
        # Verbindung komplett neu aufbauen
        global _conn
        _conn = None
        return _get_conn().cursor(cursor_factory=psycopg2.extras.RealDictCursor)


# ── Write ──────────────────────────────────────────────────────────────────────

def insert_document(
    content: str | None,
    file_data: str | None,
    content_type: str,
    source: str,
    embedding: list[float],
) -> int:
    """
    Fügt einen Dokument-Chunk in die `documents`-Tabelle ein.

    Args:
        content      : Textinhalt (oder None für binäre Chunks).
        file_data    : Base64-kodierte Bilddaten; None für text/audio/video.
        content_type : "text" | "pdf" | "image" | "audio" | "video"
        source       : Originaldateiname.
        embedding    : 3072-dimensionaler Float-Vektor.

    Returns:
        ID der eingefügten Zeile.
    """
    conn = _get_conn()
    with _cursor() as cur:
        cur.execute(
            """
            INSERT INTO documents (content, file_data, content_type, source, embedding)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
            """,
            (content, file_data, content_type, source, embedding),
        )
        row_id: int = cur.fetchone()["id"]
    conn.commit()
    return row_id


# ── Read ───────────────────────────────────────────────────────────────────────

def match_documents(
    query_embedding: list[float],
    match_count: int = 5,
    similarity_threshold: float = 0.75,
    filter_type: str | None = None,
) -> list[dict[str, Any]]:
    """
    Vektorähnlichkeitssuche via match_documents() PostgreSQL RPC.

    Args:
        query_embedding      : 3072-dim Query-Vektor.
        match_count          : Max. Ergebnisse (Sidebar "Top AI Results").
        similarity_threshold : Min. Cosine-Ähnlichkeit (0–1).
        filter_type          : Einschränkung auf content_type; None = alle.

    Returns:
        Liste von Dicts mit: id, content, file_data, content_type, source, similarity.
    """
    with _cursor() as cur:
        cur.execute(
            """
            SELECT * FROM match_documents(
                query_embedding   := %s::vector,
                match_count       := %s,
                similarity_thresh := %s,
                filter_type       := %s
            )
            """,
            (query_embedding, match_count, similarity_threshold, filter_type),
        )
        return [dict(row) for row in cur.fetchall()]


def list_documents(limit: int = 200) -> list[dict[str, Any]]:
    """Gibt die neuesten Dokumente zurück (ohne Embeddings — nur für Anzeige)."""
    with _cursor() as cur:
        cur.execute(
            """
            SELECT id, content_type, source,
                   LEFT(content, 200) AS content,
                   created_at
            FROM documents
            ORDER BY id DESC
            LIMIT %s
            """,
            (limit,),
        )
        return [dict(row) for row in cur.fetchall()]


def get_stats() -> dict[str, Any]:
    """Gibt aggregierte Chunk-Zählungen pro content_type zurück."""
    with _cursor() as cur:
        cur.execute(
            """
            SELECT
                COUNT(*)                                       AS "Total Chunks",
                COUNT(*) FILTER (WHERE content_type = 'text')  AS "Text",
                COUNT(*) FILTER (WHERE content_type = 'pdf')   AS "PDF",
                COUNT(*) FILTER (WHERE content_type = 'image') AS "Images",
                COUNT(*) FILTER (WHERE content_type = 'audio') AS "Audio",
                COUNT(*) FILTER (WHERE content_type = 'video') AS "Video"
            FROM documents
            """
        )
        return dict(cur.fetchone())


# ── Delete ─────────────────────────────────────────────────────────────────────

def delete_document(doc_id: int) -> None:
    """Löscht ein einzelnes Dokument per Primary Key."""
    conn = _get_conn()
    with _cursor() as cur:
        cur.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
    conn.commit()


def delete_all_by_source(source: str) -> int:
    """Löscht alle Chunks eines bestimmten Quell-Dateinamens."""
    conn = _get_conn()
    with _cursor() as cur:
        cur.execute(
            "DELETE FROM documents WHERE source = %s RETURNING id",
            (source,),
        )
        deleted = cur.rowcount
    conn.commit()
    return deleted