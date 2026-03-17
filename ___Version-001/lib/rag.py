"""
lib/rag.py
Orchestriert die vollständige RAG-Pipeline.

Ingest  : Typ erkennen → Chunken → Einbetten → Speichern → Quelldatei löschen
Query   : Query einbetten → Vektorsuche → Reasoning via o4-mini
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Iterator

from lib.chunker import Chunk, chunk_file
from lib.db import insert_document, match_documents
from lib.embedder import embed_bytes, embed_document_text, embed_query
from lib.codex import generate_answer


# ── Ingest ─────────────────────────────────────────────────────────────────────

def ingest_file(file_path: str) -> int:
    """
    Vollständige Ingest-Pipeline für eine einzelne Datei.
    Convenience-Wrapper um ingest_file_progress() für nicht-UI-Aufrufe.

    Returns:
        Anzahl erfolgreich gespeicherter Chunks.
    """
    stored = 0
    for _, _, _ in ingest_file_progress(file_path):
        stored += 1
    return stored


def ingest_file_progress(file_path: str) -> Iterator[tuple[int, int, float]]:
    """
    Generator-basierte Ingest-Pipeline mit Chunk-genauem Fortschritt.

    Yieldet nach jedem Chunk:
        (chunks_done: int, chunks_total: int, pct: float)

    Verwendung in Streamlit:
        for done, total, pct in ingest_file_progress(path):
            progress_bar.progress(pct, text=f"Chunk {done}/{total}")

    Löscht die Quelldatei aus /media nach dem Processing (immer, auch bei Fehler).
    """
    path = Path(file_path)

    try:
        chunks: list[Chunk] = chunk_file(file_path)
        total = len(chunks)

        for i, chunk in enumerate(chunks, 1):
            embedding = _embed_chunk(chunk)
            file_data = _extract_file_data(chunk)

            insert_document(
                content=chunk.text,
                file_data=file_data,
                content_type=chunk.content_type,
                source=str(path.name),
                embedding=embedding,
            )

            pct = i / total
            yield i, total, pct

    finally:
        if path.exists():
            path.unlink()


def _embed_chunk(chunk: Chunk) -> list[float]:
    """Leitet einen Chunk an die korrekte Embedding-Funktion weiter."""
    if chunk.content_type in {"text", "pdf"}:
        return embed_document_text(chunk.text)

    if chunk.content_type in {"image", "audio", "video"}:
        return embed_bytes(chunk.data, chunk.mime_type)

    raise ValueError(f"Unbekannter content_type: {chunk.content_type}")


def _extract_file_data(chunk: Chunk) -> str | None:
    """Base64-kodiert Bilddaten für die Datenbank; None für alle anderen Typen."""
    if chunk.content_type == "image" and chunk.data:
        return base64.b64encode(chunk.data).decode("utf-8")
    return None


# ── Query ──────────────────────────────────────────────────────────────────────

def query_rag(
    query: str,
    top_k: int = 5,
    similarity_threshold: float = 0.75,
    filter_type: str | None = None,
) -> dict[str, Any]:
    """
    Vollständige RAG Query-Pipeline.

    1. Query einbetten (RETRIEVAL_QUERY Task-Typ)
    2. Cosine Similarity Suche in pgvector
    3. Abgerufenen Kontext an o4-mini übergeben

    Args:
        query               : Natürlichsprachige Frage des Nutzers.
        top_k               : Max. Anzahl abgerufener Chunks.
        similarity_threshold: Min. Cosine-Ähnlichkeit für einen Chunk.
        filter_type         : Optionaler Content-Type-Filter.

    Returns:
        {
            "answer":  str,           # Generierte Antwort mit Quellenangaben
            "sources": list[dict],    # Rohe abgerufene Dokumente
        }
    """
    query_vec = embed_query(query)

    sources = match_documents(
        query_embedding=query_vec,
        match_count=top_k,
        similarity_threshold=similarity_threshold,
        filter_type=filter_type,
    )

    answer = generate_answer(query=query, retrieved_docs=sources)

    return {
        "answer": answer,
        "sources": sources,
    }