"""
lib/codex.py
Reasoning-Schicht — wählbar zwischen o4-mini und Gemini Flash Lite.

Referenzprojekt nutzt Gemini 3.1 Flash Lite (günstiger).
Unsere Version bietet beide Modelle per REASONING_MODEL env-Variable.
"""

from __future__ import annotations

import os
from typing import Any

_openai_client  = None
_gemini_client  = None

# Konfigurierbar via .env: "gemini" oder "openai" (Standard: gemini)
REASONING_BACKEND = os.environ.get("REASONING_BACKEND", "gemini").lower()

GEMINI_REASONING_MODEL = "gemini-2.5-flash-lite"
OPENAI_REASONING_MODEL = "o4-mini"


def _get_openai():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _openai_client


def _get_gemini():
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        _gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _gemini_client


SYSTEM_PROMPT = """Du bist ein präziser KI-Assistent für eine multimodale RAG-Anwendung.

Du erhältst eine Nutzeranfrage sowie abgerufene Dokumente als Kontext.
Jedes Dokument enthält:
- content_type : text | pdf | image | audio | video
- source       : Dateiname
- similarity   : Kosinus-Ähnlichkeit (0–1)
- content      : Textinhalt / Transkript-Chunk
- timestamp_start / timestamp_end : Zeitstempel in Sekunden (bei Audio/Video)

Regeln:
1. Beantworte die Frage ausschließlich auf Basis der bereitgestellten Dokumente.
2. Zitiere Quellen: [Quelle: <Dateiname>, Typ: <content_type>]
3. Bei Audio/Video: nenne den Zeitstempel der Fundstelle, z. B. [02:15 – 03:45]
4. Falls keine ausreichenden Informationen vorliegen, antworte:
   "Die Dokumente enthalten keine ausreichenden Informationen zu dieser Frage."
5. Antworte auf Deutsch wenn die Frage auf Deutsch gestellt wurde.
6. Kein Trainingswissen — nur aus dem gegebenen Kontext antworten.
"""


def _build_context(retrieved_docs: list[dict[str, Any]]) -> str:
    blocks = []
    for i, doc in enumerate(retrieved_docs, 1):
        ts = ""
        if doc.get("timestamp_start") is not None:
            def fmt(s):
                s = int(s); return f"{s//60:02d}:{s%60:02d}"
            ts = f"\nZeitstempel: {fmt(doc['timestamp_start'])} – {fmt(doc['timestamp_end'])}"

        content = (doc.get("content") or "")[:3000]
        if len(doc.get("content") or "") > 3000:
            content += "\n[… gekürzt …]"

        blocks.append(
            f"--- Dokument {i} ---\n"
            f"Typ: {doc.get('content_type','?')}\n"
            f"Quelle: {doc.get('source','?')}\n"
            f"Ähnlichkeit: {doc.get('similarity',0):.3f}{ts}\n"
            f"Inhalt:\n{content}"
        )
    return "\n\n".join(blocks)


def generate_answer(
    query: str,
    retrieved_docs: list[dict[str, Any]],
    backend: str | None = None,
) -> str:
    """
    Erzeugt eine Antwort via Gemini Flash Lite (Standard) oder o4-mini.

    Args:
        backend: "gemini" | "openai" — überschreibt REASONING_BACKEND env-Variable.
    """
    if not retrieved_docs:
        return (
            "Es wurden keine relevanten Dokumente gefunden. "
            "Bitte Similarity-Threshold senken oder weitere Inhalte hochladen."
        )

    ctx     = _build_context(retrieved_docs)
    used    = (backend or REASONING_BACKEND).lower()
    prompt  = f"Anfrage: {query}\n\nKontext:\n{ctx}"

    if used == "openai":
        return _reason_openai(prompt)
    else:
        return _reason_gemini(prompt)


def _reason_gemini(prompt: str) -> str:
    from google.genai import types as genai_types
    response = _get_gemini().models.generate_content(
        model=GEMINI_REASONING_MODEL,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=2048,
        ),
    )
    return response.text or ""


def _reason_openai(prompt: str) -> str:
    response = _get_openai().chat.completions.create(
        model=OPENAI_REASONING_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        max_completion_tokens=2048,
    )
    return response.choices[0].message.content or ""


def summarize_chunk(content: str, content_type: str, source: str) -> str:
    """Kurze Zusammenfassung eines einzelnen Chunks (2–3 Sätze)."""
    prompt = (
        f"Erstelle eine kurze Zusammenfassung (2-3 Sätze) des folgenden "
        f"{content_type}-Inhalts aus '{source}':\n\n{content[:2000]}"
    )
    used = REASONING_BACKEND.lower()
    if used == "openai":
        r = _get_openai().chat.completions.create(
            model=OPENAI_REASONING_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=256,
        )
        return r.choices[0].message.content or ""
    else:
        from google.genai import types as genai_types
        r = _get_gemini().models.generate_content(
            model=GEMINI_REASONING_MODEL,
            contents=prompt,
            config=genai_types.GenerateContentConfig(max_output_tokens=256),
        )
        return r.text or ""