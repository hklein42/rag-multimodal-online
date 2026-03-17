"""
lib/embedder.py
Wraps die neue google-genai SDK für Gemini Embeddings.

SDK     : google-genai >= 1.0.0  (google.generativeai ist deprecated seit v0.8)
Model   : models/gemini-embedding-2-preview
Dims    : 3072  (L2-normalisiert)
"""

import os
from typing import Literal

from google import genai
from google.genai import types as genai_types

# Client einmalig initialisieren
_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client


EMBEDDING_MODEL = "models/gemini-embedding-2-preview"
EMBEDDING_DIMS  = 3072

TaskType = Literal["RETRIEVAL_QUERY", "RETRIEVAL_DOCUMENT"]


def embed_text(text: str, task_type: TaskType = "RETRIEVAL_DOCUMENT") -> list[float]:
    """
    Embed einen Text-String.

    Args:
        text      : Zu embettender Text (max ~8 000 Tokens empfohlen).
        task_type : "RETRIEVAL_DOCUMENT" für gespeicherte Inhalte,
                    "RETRIEVAL_QUERY"    für Suchanfragen.

    Returns:
        Liste von 3072 Float-Werten (L2-normalisiert).
    """
    client = _get_client()
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=genai_types.EmbedContentConfig(task_type=task_type),
    )
    return response.embeddings[0].values


def embed_bytes(
    data: bytes,
    mime_type: str,
    task_type: TaskType = "RETRIEVAL_DOCUMENT",
) -> list[float]:
    """
    Embed binäre Inhalte (Bild, Audio-Segment, Video-Segment …).

    Args:
        data      : Rohe Bytes des Inhalts.
        mime_type : z. B. "image/jpeg", "audio/mpeg", "video/mp4".
        task_type : Siehe embed_text().

    Returns:
        Liste von 3072 Float-Werten.
    """
    client = _get_client()
    part = genai_types.Part.from_bytes(data=data, mime_type=mime_type)
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=part,
        config=genai_types.EmbedContentConfig(task_type=task_type),
    )
    return response.embeddings[0].values


def embed_query(text: str) -> list[float]:
    """Convenience-Wrapper: Suchanfrage einbetten."""
    return embed_text(text, task_type="RETRIEVAL_QUERY")


def embed_document_text(text: str) -> list[float]:
    """Convenience-Wrapper: Dokument-Chunk einbetten."""
    return embed_text(text, task_type="RETRIEVAL_DOCUMENT")