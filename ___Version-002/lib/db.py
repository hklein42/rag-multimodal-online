"""
lib/db.py
pgvector Datenbankschicht + Chat-Session-Verwaltung.

Ergänzungen ggü. Referenzprojekt:
- Felder: title, chunk_index, chunk_total, metadata (wie Referenz)
- Zusätzlich: timestamp_start, timestamp_end, transcript (unsere Erweiterung)
- Chat-Sessions für Prompt-Verlauf (unsere Erweiterung)
- Direkt psycopg2 statt Supabase Client (self-hosted Docker)
"""

from __future__ import annotations

import json
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
    global _conn
    if _conn is None or _conn.closed:
        _conn = psycopg2.connect(os.environ["DATABASE_URL"])
        register_vector(_conn)
        return _conn
    if _conn.status == psycopg2.extensions.STATUS_IN_TRANSACTION:
        try:
            _conn.rollback()
        except Exception:
            _conn = None
            return _get_conn()
    return _conn


def _cursor():
    conn = _get_conn()
    try:
        return conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    except (psycopg2.errors.InFailedSqlTransaction, psycopg2.OperationalError):
        conn.rollback()
        return conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)


# ── Documents: Write ───────────────────────────────────────────────────────────

def insert_document(
    content: str | None,
    file_data: str | None,
    content_type: str,
    source: str,
    embedding: list[float],
    title: str | None = None,
    chunk_index: int = 0,
    chunk_total: int = 1,
    metadata: dict | None = None,
    transcript: str | None = None,
    timestamp_start: float | None = None,
    timestamp_end: float | None = None,
) -> int:
    conn = _get_conn()
    with _cursor() as cur:
        cur.execute(
            """
            INSERT INTO documents
                (content, file_data, content_type, source, embedding,
                 title, chunk_index, chunk_total, metadata,
                 transcript, timestamp_start, timestamp_end)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (content, file_data, content_type, source, embedding,
             title or source, chunk_index, chunk_total,
             json.dumps(metadata or {}),
             transcript, timestamp_start, timestamp_end),
        )
        row_id = cur.fetchone()["id"]
    conn.commit()
    return row_id


def source_exists(source: str) -> bool:
    """Duplikat-Erkennung: Quelldatei bereits eingebettet?"""
    with _cursor() as cur:
        cur.execute("SELECT 1 FROM documents WHERE source = %s LIMIT 1", (source,))
        return cur.fetchone() is not None


# ── Documents: Read ────────────────────────────────────────────────────────────

def match_documents(
    query_embedding: list[float],
    match_count: int = 5,
    similarity_threshold: float = 0.75,
    filter_type: str | None = None,
) -> list[dict[str, Any]]:
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
    with _cursor() as cur:
        cur.execute(
            """
            SELECT id, content_type, source, title,
                   chunk_index, chunk_total,
                   LEFT(content, 200) AS content,
                   timestamp_start, timestamp_end, created_at
            FROM documents ORDER BY id DESC LIMIT %s
            """,
            (limit,),
        )
        return [dict(row) for row in cur.fetchall()]


def get_stats() -> dict[str, Any]:
    with _cursor() as cur:
        cur.execute(
            """
            SELECT
                COUNT(*)                                        AS "Total Chunks",
                COUNT(*) FILTER (WHERE content_type = 'text')  AS "Text",
                COUNT(*) FILTER (WHERE content_type = 'pdf')   AS "PDF",
                COUNT(*) FILTER (WHERE content_type = 'image') AS "Images",
                COUNT(*) FILTER (WHERE content_type = 'audio') AS "Audio",
                COUNT(*) FILTER (WHERE content_type = 'video') AS "Video"
            FROM documents
            """
        )
        return dict(cur.fetchone())


# ── Documents: Delete ──────────────────────────────────────────────────────────

def delete_document(doc_id: int) -> None:
    conn = _get_conn()
    with _cursor() as cur:
        cur.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
    conn.commit()


def delete_all_by_source(source: str) -> int:
    conn = _get_conn()
    with _cursor() as cur:
        cur.execute("DELETE FROM documents WHERE source = %s RETURNING id", (source,))
        deleted = cur.rowcount
    conn.commit()
    return deleted


# ── Chat Sessions ──────────────────────────────────────────────────────────────

def list_sessions() -> list[dict[str, Any]]:
    with _cursor() as cur:
        cur.execute(
            "SELECT id, name, updated_at FROM chat_sessions ORDER BY updated_at DESC"
        )
        return [dict(r) for r in cur.fetchall()]


def create_session(name: str) -> int:
    conn = _get_conn()
    with _cursor() as cur:
        cur.execute(
            "INSERT INTO chat_sessions (name) VALUES (%s) RETURNING id", (name,)
        )
        sid = cur.fetchone()["id"]
    conn.commit()
    return sid


def rename_session(session_id: int, name: str) -> None:
    conn = _get_conn()
    with _cursor() as cur:
        cur.execute(
            "UPDATE chat_sessions SET name=%s, updated_at=NOW() WHERE id=%s",
            (name, session_id),
        )
    conn.commit()


def delete_session(session_id: int) -> None:
    conn = _get_conn()
    with _cursor() as cur:
        cur.execute("DELETE FROM chat_sessions WHERE id = %s", (session_id,))
    conn.commit()


def get_messages(session_id: int) -> list[dict[str, Any]]:
    with _cursor() as cur:
        cur.execute(
            """
            SELECT id, role, content, sources, created_at
            FROM chat_messages WHERE session_id=%s ORDER BY created_at ASC
            """,
            (session_id,),
        )
        rows = [dict(r) for r in cur.fetchall()]
        for r in rows:
            if r["sources"] and isinstance(r["sources"], str):
                r["sources"] = json.loads(r["sources"])
        return rows


def add_message(
    session_id: int,
    role: str,
    content: str,
    sources: list[dict] | None = None,
) -> int:
    conn = _get_conn()
    with _cursor() as cur:
        cur.execute(
            """
            INSERT INTO chat_messages (session_id, role, content, sources)
            VALUES (%s, %s, %s, %s) RETURNING id
            """,
            (session_id, role, content, json.dumps(sources) if sources else None),
        )
        mid = cur.fetchone()["id"]
        cur.execute(
            "UPDATE chat_sessions SET updated_at=NOW() WHERE id=%s", (session_id,)
        )
    conn.commit()
    return mid