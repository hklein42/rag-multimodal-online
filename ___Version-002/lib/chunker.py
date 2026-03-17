"""
lib/chunker.py
Inhalte in einbettbare Chunks aufteilen.

Strategien
──────────
Text  → Sliding-Window ~24 000 Zeichen, 200 Overlap
PDF   → 5-Seiten-Chunks via PyMuPDF
Image → Einzelner Bytes-Chunk
Audio → Whisper-Transkription + 75s-Segmente mit Timestamps
Video → Whisper-Transkription (Audio-Spur) + 120s-Segmente mit Timestamps

Wichtig: Audio- und Video-Originaldateien werden NICHT gelöscht —
         sie bleiben in /media für die direkte Wiedergabe.
"""

from __future__ import annotations

import io
import json
import math
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

# ── Konstanten ─────────────────────────────────────────────────────────────────
TEXT_CHUNK_CHARS    = 24_000
TEXT_OVERLAP_CHARS  = 200
PDF_PAGES_PER_CHUNK = 5
AUDIO_CHUNK_SEC     = 75
VIDEO_CHUNK_SEC     = 120

FFMPEG  = "/usr/bin/ffmpeg"
FFPROBE = "/usr/bin/ffprobe"


@dataclass
class Chunk:
    content_type:    str
    text:            str | None
    data:            bytes | None
    mime_type:       str | None
    index:           int
    source:          str
    transcript:      str | None       = None   # Vollständiges Whisper-Transkript
    timestamp_start: float | None     = None   # Sekunden
    timestamp_end:   float | None     = None   # Sekunden


# ── Einstiegspunkt ─────────────────────────────────────────────────────────────

def chunk_file(file_path: str) -> list[Chunk]:
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
    transcript: str | None = None,
    timestamp_start: float | None = None,
    timestamp_end: float | None = None,
) -> Iterator[Chunk]:
    start = 0
    idx   = 0
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
                transcript=transcript,
                timestamp_start=timestamp_start,
                timestamp_end=timestamp_end,
            )
        idx += 1
        if end == length:
            break
        start = end - TEXT_OVERLAP_CHARS


# ── PDF ────────────────────────────────────────────────────────────────────────

def _chunk_pdf(path: Path) -> Iterator[Chunk]:
    try:
        import fitz
    except ImportError:
        raise ImportError("pip install pymupdf")

    doc = fitz.open(str(path))
    idx = 0
    for page_start in range(0, len(doc), PDF_PAGES_PER_CHUNK):
        page_end = min(page_start + PDF_PAGES_PER_CHUNK, len(doc))
        combined = "\n\n".join(
            doc[p].get_text() for p in range(page_start, page_end)
        ).strip()
        if combined:
            yield Chunk(
                content_type="pdf", text=combined, data=None,
                mime_type=None, index=idx, source=str(path),
            )
        idx += 1
    doc.close()


# ── Image ──────────────────────────────────────────────────────────────────────

def _chunk_image(path: Path) -> Iterator[Chunk]:
    mime_map = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png",  ".gif":  "image/gif",
        ".webp": "image/webp", ".bmp": "image/bmp",
    }
    yield Chunk(
        content_type="image", text=None,
        data=path.read_bytes(),
        mime_type=mime_map.get(path.suffix.lower(), "image/jpeg"),
        index=0, source=str(path),
    )


# ── Whisper-Hilfsfunktion ──────────────────────────────────────────────────────

def _transcribe(audio_path: str) -> list[dict]:
    """
    Transkribiert eine Audio-Datei mit OpenAI Whisper.
    Gibt Liste von Segmenten zurück: [{start, end, text}, ...]
    """
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",   # Liefert Timestamps pro Segment
            timestamp_granularities=["segment"],
        )

    segments = []
    for seg in (response.segments or []):
        segments.append({
            "start": float(seg.start),
            "end":   float(seg.end),
            "text":  seg.text.strip(),
        })
    return segments


def _segments_to_chunks(
    segments: list[dict],
    source: str,
    content_type: str,
    chunk_sec: float,
) -> Iterator[Chunk]:
    """
    Fasst Whisper-Segmente zu Chunks der Länge chunk_sec zusammen.
    Jeder Chunk bekommt timestamp_start/timestamp_end.
    """
    if not segments:
        return

    full_transcript = " ".join(s["text"] for s in segments)
    chunk_start = segments[0]["start"]
    chunk_texts = []
    chunk_end   = chunk_start

    for seg in segments:
        chunk_texts.append(seg["text"])
        chunk_end = seg["end"]

        if (chunk_end - chunk_start) >= chunk_sec:
            text = " ".join(chunk_texts).strip()
            if text:
                yield Chunk(
                    content_type=content_type,
                    text=text,
                    data=None, mime_type=None,
                    index=0, source=source,
                    transcript=full_transcript,
                    timestamp_start=chunk_start,
                    timestamp_end=chunk_end,
                )
            chunk_start = chunk_end
            chunk_texts = []

    # Letzter Rest-Chunk
    if chunk_texts:
        text = " ".join(chunk_texts).strip()
        if text:
            yield Chunk(
                content_type=content_type,
                text=text,
                data=None, mime_type=None,
                index=0, source=source,
                transcript=full_transcript,
                timestamp_start=chunk_start,
                timestamp_end=chunk_end,
            )


# ── Audio ──────────────────────────────────────────────────────────────────────

def _chunk_audio(path: Path) -> Iterator[Chunk]:
    """
    Audio-Verarbeitung:
    1. Whisper → Transkript mit Timestamps
    2. Text-Chunks mit Timestamps für semantische Suche
    3. Originaldatei bleibt erhalten in /media
    """
    from pydub import AudioSegment

    audio    = AudioSegment.from_file(str(path))
    total_ms = len(audio)

    ext      = path.suffix.lower().lstrip(".")
    mime_map = {
        "mp3": "audio/mpeg", "wav": "audio/wav",
        "ogg": "audio/ogg",  "flac": "audio/flac",
        "m4a": "audio/mp4",  "aac": "audio/aac",
    }
    mime = mime_map.get(ext, "audio/mpeg")

    # Whisper braucht mp3
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        whisper_path = tmp.name
    try:
        audio.export(whisper_path, format="mp3")
        segments = _transcribe(whisper_path)
    finally:
        if os.path.exists(whisper_path):
            os.unlink(whisper_path)

    # Text-Chunks aus Whisper-Segmenten
    yield from _segments_to_chunks(
        segments, source=path.name,
        content_type="audio", chunk_sec=AUDIO_CHUNK_SEC,
    )

    # Falls Transkription leer: Fallback auf Binär-Chunks
    if not segments:
        chunk_ms = AUDIO_CHUNK_SEC * 1000
        idx = 0
        for start_ms in range(0, total_ms, chunk_ms):
            segment = audio[start_ms:min(start_ms + chunk_ms, total_ms)]
            buf = io.BytesIO()
            segment.export(buf, format="mp3")
            buf.seek(0)
            ts = start_ms / 1000
            te = min(start_ms + chunk_ms, total_ms) / 1000
            yield Chunk(
                content_type="audio", text=None,
                data=buf.read(), mime_type=mime,
                index=idx, source=path.name,
                timestamp_start=ts, timestamp_end=te,
            )
            idx += 1


# ── Video ──────────────────────────────────────────────────────────────────────

def _chunk_video(path: Path) -> Iterator[Chunk]:
    """
    Video-Verarbeitung:
    1. ffprobe → Gesamtdauer
    2. ffmpeg  → Audio-Spur extrahieren (16kHz mono mp3 für Whisper)
    3. Whisper → Transkript mit Timestamps
    4. Text-Chunks mit Timestamps → content_type="video", text=Transkript-Chunk
    5. Originaldatei bleibt erhalten in /media (NICHT löschen)
    """
    # ── Gesamtdauer ────────────────────────────────────────────────────
    probe = subprocess.run(
        [FFPROBE, "-v", "quiet", "-print_format", "json", "-show_format", str(path)],
        capture_output=True, text=True,
    )
    if probe.returncode != 0:
        raise RuntimeError(f"ffprobe fehlgeschlagen:\n{probe.stderr}")
    duration = float(json.loads(probe.stdout)["format"]["duration"])

    # ── Audio für Whisper extrahieren ──────────────────────────────────
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        audio_path = tmp.name
    try:
        result = subprocess.run(
            [FFMPEG, "-y", "-i", str(path),
             "-vn", "-ar", "16000", "-ac", "1", "-q:a", "0",
             audio_path],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Audio-Extraktion fehlgeschlagen:\n{result.stderr[-500:]}")

        segments = _transcribe(audio_path)
    finally:
        if os.path.exists(audio_path):
            os.unlink(audio_path)

    # ── Text-Chunks aus Whisper-Segmenten ─────────────────────────────
    if segments:
        yield from _segments_to_chunks(
            segments, source=path.name,
            content_type="video", chunk_sec=VIDEO_CHUNK_SEC,
        )
    else:
        # Fallback: Binär-Segmente ohne Transkript
        idx = 0
        for start in range(0, math.ceil(duration), VIDEO_CHUNK_SEC):
            end = min(start + VIDEO_CHUNK_SEC, duration)
            if end <= start:
                break

            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                res = subprocess.run(
                    [FFMPEG, "-y", "-ss", str(start), "-t", str(end - start),
                     "-i", str(path), "-c", "copy", "-movflags", "+faststart", tmp_path],
                    capture_output=True, text=True,
                )
                if res.returncode != 0:
                    raise RuntimeError(f"ffmpeg Segment {idx}:\n{res.stderr[-500:]}")
                data = Path(tmp_path).read_bytes()
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

            yield Chunk(
                content_type="video", text=None, data=data,
                mime_type="video/mp4", index=idx, source=path.name,
                timestamp_start=float(start), timestamp_end=float(end),
            )
            idx += 1