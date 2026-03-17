"""
lib/chunker.py
Splits oversized content into embeddable chunks.

Strategien
──────────
Text  → ~6 000 Token Chunks (≈ 24 000 Zeichen) mit 200-Zeichen Overlap
PDF   → 5-Seiten-Chunks via PyMuPDF  (gibt Text pro Chunk zurück)
Audio → 75-Sekunden-Segmente via pydub (gibt Bytes pro Segment zurück)
Video → 120-Sekunden-Segmente via ffmpeg subprocess (gibt Bytes zurück)
Image → wird als einzelner Bytes-Chunk zurückgegeben (kein Splitting nötig)
"""

from __future__ import annotations

import io
import json
import math
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

# ── Konstanten ─────────────────────────────────────────────────────────────────
TEXT_CHUNK_CHARS    = 24_000      # ~6 000 Tokens @ 4 Zeichen/Token
TEXT_OVERLAP_CHARS  = 200
PDF_PAGES_PER_CHUNK = 5
AUDIO_CHUNK_SEC     = 75
VIDEO_CHUNK_SEC     = 120

FFMPEG  = "/usr/bin/ffmpeg"
FFPROBE = "/usr/bin/ffprobe"


@dataclass
class Chunk:
    """Eine einzelne einbettbare Inhaltseinheit."""
    content_type: str       # "text" | "image" | "audio" | "video" | "pdf"
    text: str | None        # befüllt für text / pdf Chunks
    data: bytes | None      # befüllt für binäre Chunks
    mime_type: str | None   # z. B. "audio/mp3", "video/mp4"
    index: int              # Chunk-Index innerhalb der Quelldatei
    source: str             # Originaldateipfad


# ── Öffentlicher Einstiegspunkt ────────────────────────────────────────────────

def chunk_file(file_path: str) -> list[Chunk]:
    """
    Erkennt den Inhaltstyp und gibt eine Liste von Chunk-Objekten zurück.

    Args:
        file_path: Absoluter Pfad zur Datei auf dem Datenträger.

    Returns:
        List[Chunk]
    """
    path = Path(file_path)
    ext  = path.suffix.lower()

    if ext in {".txt", ".md", ".csv", ".json", ".xml", ".html"}:
        return list(_chunk_text_file(path))

    if ext == ".pdf":
        return list(_chunk_pdf(path))

    if ext in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}:
        return list(_chunk_image(path))

    if ext in {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac"}:
        return list(_chunk_audio(path))

    if ext in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
        return list(_chunk_video(path))

    raise ValueError(f"Nicht unterstützter Dateityp: {ext}")


# ── Text ───────────────────────────────────────────────────────────────────────

def _chunk_text_file(path: Path) -> Iterator[Chunk]:
    text = path.read_text(encoding="utf-8", errors="replace")
    yield from _split_text(text, source=str(path), content_type="text")


def _split_text(
    text: str,
    source: str,
    content_type: str = "text",
) -> Iterator[Chunk]:
    """Sliding-Window-Chunker mit Overlap."""
    start  = 0
    idx    = 0
    length = len(text)

    while start < length:
        end        = min(start + TEXT_CHUNK_CHARS, length)
        chunk_text = text[start:end].strip()
        if chunk_text:
            yield Chunk(
                content_type=content_type,
                text=chunk_text,
                data=None,
                mime_type=None,
                index=idx,
                source=source,
            )
        idx += 1
        if end == length:
            break
        start = end - TEXT_OVERLAP_CHARS  # Overlap


# ── PDF ────────────────────────────────────────────────────────────────────────

def _chunk_pdf(path: Path) -> Iterator[Chunk]:
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("PyMuPDF nicht installiert. Ausführen: pip install pymupdf")

    doc         = fitz.open(str(path))
    total_pages = len(doc)
    idx         = 0

    for page_start in range(0, total_pages, PDF_PAGES_PER_CHUNK):
        page_end = min(page_start + PDF_PAGES_PER_CHUNK, total_pages)
        texts    = []

        for page_num in range(page_start, page_end):
            page = doc[page_num]
            texts.append(page.get_text())

        combined = "\n\n".join(texts).strip()
        if combined:
            yield Chunk(
                content_type="pdf",
                text=combined,
                data=None,
                mime_type=None,
                index=idx,
                source=str(path),
            )
        idx += 1

    doc.close()


# ── Image ──────────────────────────────────────────────────────────────────────

def _chunk_image(path: Path) -> Iterator[Chunk]:
    mime_map = {
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png":  "image/png",
        ".gif":  "image/gif",
        ".webp": "image/webp",
        ".bmp":  "image/bmp",
    }
    mime = mime_map.get(path.suffix.lower(), "image/jpeg")
    data = path.read_bytes()

    yield Chunk(
        content_type="image",
        text=None,
        data=data,
        mime_type=mime,
        index=0,
        source=str(path),
    )


# ── Audio ──────────────────────────────────────────────────────────────────────

def _chunk_audio(path: Path) -> Iterator[Chunk]:
    try:
        from pydub import AudioSegment
    except ImportError:
        raise ImportError("pydub nicht installiert. Ausführen: pip install pydub")

    audio    = AudioSegment.from_file(str(path))
    total_ms = len(audio)
    chunk_ms = AUDIO_CHUNK_SEC * 1000

    ext      = path.suffix.lower().lstrip(".")
    mime_map = {
        "mp3":  "audio/mpeg",
        "wav":  "audio/wav",
        "ogg":  "audio/ogg",
        "flac": "audio/flac",
        "m4a":  "audio/mp4",
        "aac":  "audio/aac",
    }
    mime          = mime_map.get(ext, "audio/mpeg")
    export_format = "mp3" if ext in {"m4a", "aac"} else ext
    idx           = 0

    for start_ms in range(0, total_ms, chunk_ms):
        end_ms  = min(start_ms + chunk_ms, total_ms)
        segment = audio[start_ms:end_ms]

        buf = io.BytesIO()
        segment.export(buf, format=export_format)
        buf.seek(0)

        yield Chunk(
            content_type="audio",
            text=None,
            data=buf.read(),
            mime_type=mime,
            index=idx,
            source=str(path),
        )
        idx += 1


# ── Video ──────────────────────────────────────────────────────────────────────

def _chunk_video(path: Path) -> Iterator[Chunk]:
    """
    Schneidet Video in Segmente via ffmpeg subprocess.
    Kein moviepy — funktioniert zuverlässig auf ARM64 / macOS M3 / Docker.

    Strategie: ffprobe ermittelt Gesamtdauer, ffmpeg schneidet Segmente
    mit Stream-Copy (kein Re-Encode → sehr schnell).
    """
    # ── Gesamtdauer ermitteln ──────────────────────────────────────────
    probe = subprocess.run(
        [
            FFPROBE, "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        raise RuntimeError(f"ffprobe fehlgeschlagen:\n{probe.stderr}")

    info     = json.loads(probe.stdout)
    duration = float(info["format"]["duration"])

    # ── Segmente schneiden ─────────────────────────────────────────────
    idx = 0
    for start in range(0, math.ceil(duration), VIDEO_CHUNK_SEC):
        end = min(start + VIDEO_CHUNK_SEC, duration)
        if end <= start:
            break

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            result = subprocess.run(
                [
                    FFMPEG, "-y",
                    "-ss", str(start),
                    "-t",  str(end - start),
                    "-i",  str(path),
                    "-c",  "copy",           # Stream-Copy — kein Re-Encode
                    "-movflags", "+faststart",
                    tmp_path,
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg Segment {idx} fehlgeschlagen:\n"
                    f"{result.stderr[-800:]}"
                )

            data = Path(tmp_path).read_bytes()

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        yield Chunk(
            content_type="video",
            text=None,
            data=data,
            mime_type="video/mp4",
            index=idx,
            source=str(path),
        )
        idx += 1