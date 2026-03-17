"""
lib/rag.py
RAG-Pipeline: Ingest + Query.

Ergänzungen ggü. Referenzprojekt:
- MIME_MAP aus Referenz übernommen (vollständige Typ-Erkennung)
- Whisper-Transkription + Timestamps (unsere Erweiterung)
- Originaldateien (Audio/Video) bleiben erhalten in /media
- Duplikat-Erkennung via source_exists()
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Iterator

from lib.chunker import Chunk, chunk_file
from lib.db import insert_document, match_documents, source_exists
from lib.embedder import embed_bytes, embed_document_text, embed_query
from lib.codex import generate_answer

# ── MIME-Map aus Referenzprojekt übernommen ────────────────────────────────────
MIME_MAP = {
    "image/png":       "image",
    "image/jpeg":      "image",
    "image/jpg":       "image",
    "image/webp":      "image",
    "image/gif":       "image",
    "image/bmp":       "image",
    "application/pdf": "pdf",
    "audio/mpeg":      "audio",
    "audio/mp3":       "audio",
    "audio/wav":       "audio",
    "audio/x-wav":     "audio",
    "audio/ogg":       "audio",
    "audio/flac":      "audio",
    "audio/mp4":       "audio",
    "video/mp4":       "video",
    "video/quicktime": "video",
    "video/x-msvideo": "video",
    "video/webm":      "video",
    "text/plain":      "text",
    "text/csv":        "text",
    "application/json":"text",
}

EXT_MAP = {
    "txt": "text",  "md": "text",  "csv": "text",
    "json": "text", "xml": "text", "html": "text",
    "png": "image", "jpg": "image", "jpeg": "image",
    "webp": "image", "gif": "image", "bmp": "image",
    "pdf": "pdf",
    "mp3": "audio", "wav": "audio", "ogg": "audio",
    "flac": "audio", "m4a": "audio", "aac": "audio",
    "mp4": "video", "mov": "video", "avi": "video",
    "mkv": "video", "webm": "video",
}

MEDIA_EXTENSIONS = {
    ".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac",
    ".mp4", ".mov", ".avi", ".mkv", ".webm",
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp",
}


def detect_content_type(mime: str, filename: str) -> str:
    """MIME-basierte Erkennung mit Extension-Fallback (wie Referenzprojekt)."""
    if mime in MIME_MAP:
        return MIME_MAP[mime]
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return EXT_MAP.get(ext, "text")


# ── Ingest ─────────────────────────────────────────────────────────────────────

def ingest_file(file_path: str, force: bool = False) -> int:
    stored = 0
    for _, _, _ in ingest_file_progress(file_path, force=force):
        stored += 1
    return stored


def ingest_file_progress(
    file_path: str,
    force: bool = False,
) -> Iterator[tuple[int, int, float]]:
    """
    Generator-basierte Ingest-Pipeline mit Chunk-genauem Fortschritt.
    Yields: (chunks_done, chunks_total, pct)

    Audio/Video-Originale bleiben in /media erhalten.
    Text/PDF/Bild-Dateien werden nach dem Embedding gelöscht.
    """
    path = Path(file_path)
    is_media = path.suffix.lower() in MEDIA_EXTENSIONS

    # Duplikat-Erkennung
    if not force and source_exists(path.name):
        raise ValueError(
            f"'{path.name}' wurde bereits eingebettet. "
            "Checkbox 'Erneut verarbeiten' aktivieren oder Chunks zuerst löschen."
        )

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
                source=path.name,
                embedding=embedding,
                title=path.stem,
                chunk_index=i - 1,
                chunk_total=total,
                metadata={
                    "original_path": str(path),
                    "mime_type": chunk.mime_type or "",
                },
                transcript=chunk.transcript,
                timestamp_start=chunk.timestamp_start,
                timestamp_end=chunk.timestamp_end,
            )
            yield i, total, i / total

    finally:
        # Audio/Video/Bild-Originale NICHT löschen — für Wiedergabe in /media
        if not is_media and path.exists():
            path.unlink()


def _embed_chunk(chunk: Chunk) -> list[float]:
    if chunk.content_type in {"text", "pdf", "audio", "video"}:
        text = chunk.text or chunk.transcript or ""
        if text:
            return embed_document_text(text)
    if chunk.data and chunk.mime_type:
        return embed_bytes(chunk.data, chunk.mime_type)
    return embed_document_text(chunk.text or "")


def _extract_file_data(chunk: Chunk) -> str | None:
    if chunk.content_type == "image" and chunk.data:
        return base64.b64encode(chunk.data).decode("utf-8")
    return None


# ── Query ──────────────────────────────────────────────────────────────────────

def query_rag(
    query: str,
    top_k: int = 5,
    similarity_threshold: float = 0.75,
    filter_type: str | None = None,
    use_ai: bool = True,
    reasoning_backend: str | None = None,
) -> dict[str, Any]:
    """
    RAG Query-Pipeline.

    Args:
        use_ai            : False = nur Quellen, kein Reasoning.
        reasoning_backend : "gemini" | "openai" — überschreibt .env-Einstellung.
    """
    query_vec = embed_query(query)
    sources   = match_documents(
        query_embedding=query_vec,
        match_count=top_k,
        similarity_threshold=similarity_threshold,
        filter_type=filter_type,
    )

    answer = None
    if use_ai:
        answer = generate_answer(
            query=query,
            retrieved_docs=sources,
            backend=reasoning_backend,
        )

    return {"answer": answer, "sources": sources}