"""
lib/codex.py
Sends the query + retrieved context to OpenAI o4-mini for reasoning.

Note: "codex" is a legacy filename — the model used is o4-mini (not Codex).
"""

from __future__ import annotations

import os
from typing import Any

from openai import OpenAI

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


# ── System Prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Du bist ein präziser KI-Assistent für eine multimodale RAG-Anwendung.

Du erhältst eine Nutzeranfrage sowie eine Liste von abgerufenen Dokumenten als Kontext.
Jedes Dokument enthält:
- content_type: text | pdf | image | audio | video
- source: Dateiname der Originalquelle
- similarity: Kosinus-Ähnlichkeit (0–1)
- content: Textinhalt (kann leer sein bei Bild/Audio/Video)

Regeln:
1. Beantworte die Frage ausschließlich auf Basis der bereitgestellten Dokumente.
2. Zitiere Quellen mit [Quelle: <Dateiname>, Typ: <content_type>].
3. Wenn mehrere Quellen relevant sind, erwähne alle.
4. Falls der Kontext keine ausreichenden Informationen enthält, antworte:
   "Die vorliegenden Dokumente enthalten keine ausreichenden Informationen zu dieser Frage."
5. Antworte auf Deutsch, sofern die Frage auf Deutsch gestellt wurde.
6. Gib keine Informationen aus deinem Trainings-Wissen weiter — nur aus dem Kontext.
"""


def generate_answer(query: str, retrieved_docs: list[dict[str, Any]]) -> str:
    """
    Generate a grounded answer using o4-mini with the retrieved context.

    Args:
        query         : The user's original question.
        retrieved_docs: List of dicts from match_documents() — each with
                        content, file_data, content_type, source, similarity.

    Returns:
        The model's answer as a markdown-formatted string.
    """
    if not retrieved_docs:
        return (
            "Es wurden keine relevanten Dokumente gefunden. "
            "Bitte passe den Similarity-Threshold an oder lade weitere Inhalte hoch."
        )

    context_blocks = []
    for i, doc in enumerate(retrieved_docs, 1):
        block = (
            f"--- Dokument {i} ---\n"
            f"Typ: {doc.get('content_type', 'unbekannt')}\n"
            f"Quelle: {doc.get('source', 'unbekannt')}\n"
            f"Ähnlichkeit: {doc.get('similarity', 0):.3f}\n"
        )
        if doc.get("content"):
            # Truncate very long chunks to avoid context overflow
            content_preview = doc["content"][:3000]
            if len(doc["content"]) > 3000:
                content_preview += "\n[… Inhalt gekürzt …]"
            block += f"Inhalt:\n{content_preview}\n"
        else:
            block += "Inhalt: [Binärdaten — kein Textinhalt verfügbar]\n"

        context_blocks.append(block)

    context_text = "\n\n".join(context_blocks)

    user_message = (
        f"Anfrage: {query}\n\n"
        f"Kontext:\n{context_text}"
    )

    client = _get_client()

    response = client.chat.completions.create(
        model="o4-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        max_completion_tokens=2048,
    )

    return response.choices[0].message.content or ""