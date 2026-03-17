"""
lib/embedder.py
Google Gemini Embeddings — neue google-genai SDK >= 1.0.0.

Änderungen ggü. Referenzprojekt:
- L2-Normalisierung übernommen (wie Referenz, explizit via numpy)
- Generisches embed_bytes() statt spezialisierter Funktionen (simpler)
- Modell: models/gemini-embedding-2-preview (3072 dims)
"""

import os
from typing import Literal

import numpy as np
from google import genai
from google.genai import types as genai_types

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client


def _normalize(vec: list[float]) -> list[float]:
    """L2-Normalisierung — identisch zum Referenzprojekt."""
    a    = np.array(vec, dtype=np.float64)
    norm = np.linalg.norm(a)
    if norm > 0:
        a = a / norm
    return a.tolist()


EMBEDDING_MODEL = "models/gemini-embedding-2-preview"
EMBEDDING_DIMS  = 3072

TaskType = Literal["RETRIEVAL_QUERY", "RETRIEVAL_DOCUMENT"]


def embed_text(text: str, task_type: TaskType = "RETRIEVAL_DOCUMENT") -> list[float]:
    response = _get_client().models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=genai_types.EmbedContentConfig(task_type=task_type),
    )
    return _normalize(response.embeddings[0].values)


def embed_bytes(
    data: bytes,
    mime_type: str,
    task_type: TaskType = "RETRIEVAL_DOCUMENT",
) -> list[float]:
    """Binäre Inhalte einbetten (Bild, Audio, Video, PDF-Bytes)."""
    part = genai_types.Part.from_bytes(data=data, mime_type=mime_type)
    response = _get_client().models.embed_content(
        model=EMBEDDING_MODEL,
        contents=part,
        config=genai_types.EmbedContentConfig(task_type=task_type),
    )
    return _normalize(response.embeddings[0].values)


def embed_query(text: str) -> list[float]:
    return embed_text(text, task_type="RETRIEVAL_QUERY")


def embed_document_text(text: str) -> list[float]:
    return embed_text(text, task_type="RETRIEVAL_DOCUMENT")