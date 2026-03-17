"""
multimodal-rag — app.py
Streamlit GUI mit:
  Tab 1: Upload & Embed    (Duplikat-Erkennung, Fortschritt)
  Tab 2: Suche             (Ergebnisanzeige mit Timestamps, Video-Player, KI optional)
  Tab 3: Prompt-Verlauf    (Sessions speichern, umbenennen, löschen)
  Tab 4: Browse            (Dokumente anzeigen & löschen)
"""

import base64
import os
from pathlib import Path

import streamlit as st

from lib.rag import ingest_file_progress, query_rag
from lib.db import (
    get_stats, delete_document, list_documents,
    list_sessions, create_session, rename_session,
    delete_session, get_messages, add_message,
    delete_all_by_source,
)
from lib.codex import summarize_chunk

MEDIA_DIR = Path("/app/media")
MEDIA_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_TYPES = {
    "text":  [".txt", ".md", ".csv", ".json", ".xml", ".html"],
    "pdf":   [".pdf"],
    "image": [".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"],
    "audio": [".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac"],
    "video": [".mp4", ".mov", ".avi", ".mkv", ".webm"],
}
ALL_EXTENSIONS = [e for exts in SUPPORTED_TYPES.values() for e in exts]

MEDIA_TYPES = {
    ".mp3": "audio", ".wav": "audio", ".ogg": "audio",
    ".flac": "audio", ".m4a": "audio", ".aac": "audio",
    ".mp4": "video", ".mov": "video", ".avi": "video",
    ".mkv": "video", ".webm": "video",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def fmt_ts(seconds: float | None) -> str:
    """Sekunden → MM:SS"""
    if seconds is None:
        return "–"
    s = int(seconds)
    return f"{s // 60:02d}:{s % 60:02d}"


def media_path(source: str) -> Path | None:
    """Gibt den Pfad zur Originaldatei zurück, falls vorhanden."""
    p = MEDIA_DIR / source
    return p if p.exists() else None


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multimodal RAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔍 Multimodal RAG")
st.caption("Text · Bilder · Video · Audio · PDF — Gemini Embeddings + OpenAI o4-mini")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Suchfilter")

    top_k = st.slider("Top AI Results", 1, 25, 5)

    similarity_threshold = st.slider(
        "Similarity Threshold", 0.0, 1.0, 0.35, 0.01, format="%.2f"
    )

    content_type_options = ["All", "text", "pdf", "image", "audio", "video"]
    selected_type = st.selectbox("Content Type", content_type_options)
    filter_type = None if selected_type == "All" else selected_type

    use_ai = st.toggle("🤖 KI-Antwort", value=True)

    reasoning_backend = st.radio(
        "Reasoning-Modell",
        options=["gemini", "openai"],
        index=0,
        horizontal=True,
        help="gemini = Gemini Flash Lite (günstiger) | openai = o4-mini",
        disabled=not use_ai,
    )

    st.divider()
    st.subheader("📊 Datenbank")
    try:
        stats = get_stats()
        for k, v in stats.items():
            st.metric(k, v)
    except Exception as e:
        st.error(f"DB nicht erreichbar: {e}")


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_upload, tab_search, tab_history, tab_browse = st.tabs([
    "📤 Upload & Embed",
    "🔎 Suche",
    "💬 Prompt-Verlauf",
    "📁 Browse",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Upload & Embed
# ══════════════════════════════════════════════════════════════════════════════
with tab_upload:
    st.subheader("Dateien hochladen & einbetten")
    st.info(
        "**Unterstützte Formate:** Text, PDF, Bilder, Audio, Video  \n"
        "**Max. Dateigröße:** 250 MB  \n"
        "**Audio/Video-Originale bleiben erhalten** für die direkte Wiedergabe.  \n"
        "Text/PDF/Bild-Dateien werden nach dem Embedding gelöscht."
    )

    force_reembed = st.checkbox("Bereits eingebettete Dateien erneut verarbeiten", value=False)

    uploaded_files = st.file_uploader(
        "Dateien auswählen",
        accept_multiple_files=True,
        type=[e.lstrip(".") for e in ALL_EXTENSIONS],
    )

    if uploaded_files and st.button("▶ Verarbeiten & Einbetten", type="primary"):
        total_files   = len(uploaded_files)
        success_count = 0
        error_count   = 0

        st.markdown("**Gesamt-Fortschritt**")
        overall_bar   = st.progress(0)
        overall_label = st.empty()

        for file_idx, uf in enumerate(uploaded_files):
            overall_label.caption(f"Datei {file_idx+1} / {total_files}: **{uf.name}**")

            st.markdown(f"↳ `{uf.name}`")
            file_slot  = st.empty()
            label_slot = st.empty()

            dest = MEDIA_DIR / uf.name
            try:
                with open(dest, "wb") as f:
                    f.write(uf.read())

                chunks_stored = 0
                for done, total_c, pct in ingest_file_progress(
                    str(dest), force=force_reembed
                ):
                    chunks_stored = done
                    file_slot.progress(
                        pct,
                        text=f"Chunk {done} / {total_c} — {pct*100:.0f} %"
                    )

                file_slot.progress(1.0, text="✅ Fertig")
                label_slot.caption(f"{chunks_stored} Chunks gespeichert")
                success_count += 1

            except ValueError as e:
                # Duplikat-Warnung
                file_slot.progress(0.0, text="⚠️ Übersprungen")
                label_slot.caption(str(e))
                error_count += 1
                # Duplikat-Datei wieder löschen wenn kein Media-Typ
                if dest.exists() and dest.suffix.lower() not in MEDIA_TYPES:
                    dest.unlink()

            except Exception as e:
                file_slot.progress(0.0, text="❌ Fehler")
                label_slot.caption(str(e))
                error_count += 1
                if dest.exists() and dest.suffix.lower() not in MEDIA_TYPES:
                    dest.unlink()

            overall_bar.progress(
                (file_idx + 1) / total_files,
                text=f"Datei {file_idx+1} / {total_files} abgeschlossen"
            )

        overall_label.caption(
            f"✅ Abgeschlossen — {success_count} erfolgreich, {error_count} Fehler/Übersprungen"
        )
        col1, col2 = st.columns(2)
        col1.metric("Erfolgreich", success_count)
        col2.metric("Fehler/Übersprungen", error_count)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Suche
# ══════════════════════════════════════════════════════════════════════════════
with tab_search:
    st.subheader("Semantische Suche")

    # Aktive Session für Verlauf
    sessions = list_sessions()
    session_names = ["— Kein Verlauf speichern —"] + [s["name"] for s in sessions]
    session_ids   = [None] + [s["id"] for s in sessions]

    col_q, col_s = st.columns([3, 1])
    with col_q:
        query_input = st.text_area("Suchanfrage", placeholder="Was möchtest du wissen?", height=80)
    with col_s:
        selected_session_idx = st.selectbox(
            "In Session speichern",
            range(len(session_names)),
            format_func=lambda i: session_names[i],
        )
        active_session_id = session_ids[selected_session_idx]

        new_session_name = st.text_input("Neue Session anlegen")
        if st.button("➕ Session") and new_session_name.strip():
            sid = create_session(new_session_name.strip())
            st.success(f"Session '{new_session_name}' angelegt.")
            st.rerun()

    if st.button("🔍 Suchen", type="primary") and query_input.strip():
        with st.spinner("Einbetten & Suchen …"):
            try:
                results = query_rag(
                    query=query_input,
                    top_k=top_k,
                    similarity_threshold=similarity_threshold,
                    filter_type=filter_type,
                    use_ai=use_ai,
                    reasoning_backend=reasoning_backend if use_ai else None,
                )
            except Exception as e:
                st.error(f"Fehler: {e}")
                results = None

        if results:
            # Session-Verlauf speichern
            if active_session_id:
                add_message(active_session_id, "user", query_input)
                if results["answer"]:
                    add_message(
                        active_session_id, "assistant",
                        results["answer"], sources=results["sources"]
                    )

            # ── KI-Antwort ─────────────────────────────────────────────
            if use_ai and results["answer"]:
                st.divider()
                st.subheader("💬 KI-Antwort")
                st.markdown(results["answer"])

            # ── Quellen ────────────────────────────────────────────────
            st.divider()
            st.subheader(f"📌 Quellen ({len(results['sources'])} Treffer)")

            for idx, src in enumerate(results["sources"], 1):
                ct      = src.get("content_type", "?")
                sim     = src.get("similarity", 0)
                source  = src.get("source", "unbekannt")
                ts_s    = src.get("timestamp_start")
                ts_e    = src.get("timestamp_end")
                mp      = media_path(source)

                # Zeitstempel-Label
                ts_label = ""
                if ts_s is not None:
                    ts_label = f" · ⏱ {fmt_ts(ts_s)} – {fmt_ts(ts_e)}"

                with st.expander(
                    f"#{idx} · {ct.upper()} · {sim:.3f}{ts_label} · {source}",
                    expanded=False,
                ):
                    # ── Zusammenfassung ────────────────────────────────
                    content_text = src.get("content") or ""
                    if content_text:
                        with st.spinner("Zusammenfassung …"):
                            try:
                                summary = summarize_chunk(content_text, ct, source)
                                st.info(f"**Zusammenfassung:** {summary}")
                            except Exception:
                                st.caption(content_text[:300])
                    else:
                        st.caption("Kein Textinhalt verfügbar.")

                    # ── Bild anzeigen ──────────────────────────────────
                    if ct == "image" and src.get("file_data"):
                        try:
                            img_bytes = base64.b64decode(src["file_data"])
                            st.image(img_bytes, use_container_width=True)
                        except Exception:
                            st.warning("Bild konnte nicht dekodiert werden.")

                    # ── Video-Player mit Zeitstempel ───────────────────
                    if ct == "video" and mp:
                        st.markdown("**🎬 Original-Video**")
                        # Start 10s vor Fundstelle
                        start_at = max(0, int(ts_s or 0) - 10)
                        st.video(str(mp), start_time=start_at)
                        if ts_s is not None:
                            st.caption(
                                f"Fundstelle: {fmt_ts(ts_s)} – {fmt_ts(ts_e)} "
                                f"(Wiedergabe ab {fmt_ts(start_at)})"
                            )
                        # Download-Link
                        with open(mp, "rb") as vf:
                            st.download_button(
                                "⬇ Video herunterladen",
                                data=vf.read(),
                                file_name=source,
                                mime="video/mp4",
                                key=f"dl_v_{idx}",
                            )

                    # ── Audio-Player mit Zeitstempel ───────────────────
                    elif ct == "audio" and mp:
                        st.markdown("**🎵 Original-Audio**")
                        start_at = max(0, int(ts_s or 0) - 10)
                        st.audio(str(mp), start_time=start_at)
                        if ts_s is not None:
                            st.caption(
                                f"Fundstelle: {fmt_ts(ts_s)} – {fmt_ts(ts_e)} "
                                f"(Wiedergabe ab {fmt_ts(start_at)})"
                            )

                    # ── Kein Original vorhanden ────────────────────────
                    elif ct in {"video", "audio"} and not mp:
                        st.warning(
                            f"Original '{source}' nicht in /media gefunden. "
                            "Datei neu hochladen um Wiedergabe zu aktivieren."
                        )

                    # ── Vollständiger Transkript-Text ──────────────────
                    if src.get("transcript") and ct in {"video", "audio"}:
                        with st.expander("📝 Vollständiges Transkript"):
                            st.text(src["transcript"][:3000])

        elif results is not None:
            st.warning(
                "Keine Treffer. Similarity-Threshold senken "
                "oder andere Suchbegriffe verwenden."
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Prompt-Verlauf
# ══════════════════════════════════════════════════════════════════════════════
with tab_history:
    st.subheader("💬 Prompt-Verlauf")

    sessions = list_sessions()

    if not sessions:
        st.info("Noch keine Sessions vorhanden. Im Such-Tab eine Session anlegen.")
    else:
        # Session-Auswahl
        session_labels = {s["id"]: s["name"] for s in sessions}
        selected_id = st.selectbox(
            "Session wählen",
            options=[s["id"] for s in sessions],
            format_func=lambda i: session_labels[i],
        )

        col_r, col_d = st.columns([3, 1])
        with col_r:
            new_name = st.text_input("Session umbenennen", value=session_labels[selected_id])
            if st.button("✏️ Umbenennen"):
                rename_session(selected_id, new_name)
                st.success("Umbenannt.")
                st.rerun()
        with col_d:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🗑 Session löschen", type="secondary"):
                delete_session(selected_id)
                st.success("Session gelöscht.")
                st.rerun()

        st.divider()

        # Nachrichten anzeigen (Chat-Stil)
        messages = get_messages(selected_id)
        if not messages:
            st.caption("Keine Nachrichten in dieser Session.")
        else:
            for msg in messages:
                role    = msg["role"]
                content = msg["content"]
                sources = msg.get("sources") or []

                with st.chat_message(role):
                    st.markdown(content)

                    # Quellen kompakt anzeigen
                    if sources:
                        with st.expander(f"📌 {len(sources)} Quellen"):
                            for s in sources:
                                ts = ""
                                if s.get("timestamp_start") is not None:
                                    ts = f" · {fmt_ts(s['timestamp_start'])}–{fmt_ts(s['timestamp_end'])}"
                                st.caption(
                                    f"**{s.get('content_type','?').upper()}** · "
                                    f"{s.get('similarity',0):.3f}{ts} · "
                                    f"{s.get('source','?')}"
                                )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Browse
# ══════════════════════════════════════════════════════════════════════════════
with tab_browse:
    st.subheader("Gespeicherte Dokumente")

    col_ref, col_info = st.columns([1, 4])
    with col_ref:
        if st.button("🔄 Aktualisieren"):
            st.rerun()
    with col_info:
        st.caption("Löschen entfernt Chunks aus der DB. Audio/Video-Originale in /media bleiben erhalten.")

    try:
        docs = list_documents(limit=200)
    except Exception as e:
        st.error(f"Fehler: {e}")
        docs = []

    if not docs:
        st.info("Keine Dokumente in der Datenbank.")
    else:
        st.caption(f"{len(docs)} Einträge")

        # Spaltenköpfe
        hc = st.columns([1, 2, 3, 2, 2, 1])
        for col, label in zip(hc, ["ID", "Typ", "Inhalt", "Quelle", "Zeitstempel", "Aktion"]):
            col.markdown(f"**{label}**")

        for doc in docs:
            c1, c2, c3, c4, c5, c6 = st.columns([1, 2, 3, 2, 2, 1])
            c1.text(str(doc["id"]))
            c2.text(doc.get("content_type", "–"))
            content_text = doc.get("content") or ""
            c3.text(content_text[:60] + ("…" if len(content_text) > 60 else ""))
            c4.text((doc.get("source") or "–")[-35:])

            ts_s = doc.get("timestamp_start")
            ts_e = doc.get("timestamp_end")
            c5.text(f"{fmt_ts(ts_s)} – {fmt_ts(ts_e)}" if ts_s is not None else "–")

            with c6:
                if st.button("🗑", key=f"del_{doc['id']}"):
                    try:
                        delete_document(doc["id"])
                        st.success(f"#{doc['id']} gelöscht.")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

    # ── Alle Chunks einer Quelle löschen ──────────────────────────────
    st.divider()
    st.markdown("**Alle Chunks einer Quelldatei löschen**")
    source_to_delete = st.text_input("Dateiname (z. B. video.mp4)")
    if st.button("🗑 Alle Chunks löschen") and source_to_delete:
        n = delete_all_by_source(source_to_delete)
        st.success(f"{n} Chunks gelöscht.")
        st.rerun()