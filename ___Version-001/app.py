"""
multimodal-rag — app.py
Streamlit GUI: Upload & Embed | Search | Browse
"""

import streamlit as st
import os
from pathlib import Path

from lib.rag import ingest_file_progress, query_rag
from lib.db import get_stats, delete_document, list_documents

MEDIA_DIR = Path("/app/media")
MEDIA_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_TYPES = {
    "text":  [".txt", ".md", ".csv", ".json", ".xml", ".html"],
    "pdf":   [".pdf"],
    "image": [".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"],
    "audio": [".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac"],
    "video": [".mp4", ".mov", ".avi", ".mkv", ".webm"],
}

ALL_EXTENSIONS = [ext for exts in SUPPORTED_TYPES.values() for ext in exts]


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multimodal RAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔍 Multimodal RAG")
st.caption("Text · Images · Video · Audio · PDF — powered by Gemini Embeddings + OpenAI o4-mini")

# ── Sidebar — Search filters ───────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Search Filters")

    top_k = st.slider(
        "Top AI Results",
        min_value=1,
        max_value=25,
        value=5,
        step=1,
        help="Maximale Anzahl zurückgegebener Treffer",
    )

    similarity_threshold = st.slider(
        "Similarity Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.75,
        step=0.01,
        format="%.2f",
        help="Minimale Cosine-Ähnlichkeit (0 = alles, 1 = exakt)",
    )

    content_type_options = ["All", "text", "pdf", "image", "audio", "video"]
    selected_type = st.selectbox(
        "Content Type",
        options=content_type_options,
        index=0,
        help="Einschränkung auf einen bestimmten Medientyp",
    )
    filter_type = None if selected_type == "All" else selected_type

    st.divider()
    st.subheader("📊 Database Stats")
    try:
        stats = get_stats()
        for k, v in stats.items():
            st.metric(k, v)
    except Exception as e:
        st.error(f"DB nicht erreichbar: {e}")


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_upload, tab_search, tab_browse = st.tabs(
    ["📤 Upload & Embed", "🔎 Search", "📁 Browse"]
)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — Upload & Embed
# ──────────────────────────────────────────────────────────────────────────────
with tab_upload:
    st.subheader("Dateien hochladen & einbetten")
    st.info(
        "Unterstützte Formate: Text, PDF, Bilder, Audio (mp3/wav/…), Video (mp4/mov/…)  \n"
        "Maximale Dateigröße: **250 MB** pro Datei  \n"
        "Dateien werden nach erfolgreichem Embedding automatisch gelöscht."
    )

    uploaded_files = st.file_uploader(
        "Dateien auswählen",
        accept_multiple_files=True,
        type=[ext.lstrip(".") for ext in ALL_EXTENSIONS],
    )

    if uploaded_files:
        if st.button("▶ Verarbeiten & Einbetten", type="primary"):
            total_files = len(uploaded_files)
            success_count = 0
            error_count = 0

            # ── Gesamt-Fortschritt (alle Dateien) ─────────────────────────
            st.markdown("**Gesamt-Fortschritt**")
            overall_bar   = st.progress(0)
            overall_label = st.empty()

            for file_idx, uf in enumerate(uploaded_files):

                overall_label.caption(
                    f"Datei {file_idx + 1} / {total_files}: **{uf.name}**"
                )

                # ── Datei-Fortschritt (Chunks) — st.empty() für Live-Updates
                st.markdown(f"↳ `{uf.name}`")
                file_progress_slot = st.empty()
                file_label_slot    = st.empty()

                dest_path = MEDIA_DIR / uf.name
                try:
                    with open(dest_path, "wb") as f:
                        f.write(uf.read())

                    chunks_stored = 0
                    for done, total_chunks, pct in ingest_file_progress(str(dest_path)):
                        chunks_stored = done
                        file_progress_slot.progress(
                            pct,
                            text=f"Chunk {done} / {total_chunks} — {pct * 100:.0f} %"
                        )

                    file_progress_slot.progress(1.0, text="✅ Fertig")
                    file_label_slot.caption(f"{chunks_stored} Chunks gespeichert")
                    success_count += 1

                except Exception as e:
                    file_progress_slot.progress(0.0, text="❌ Fehler")
                    file_label_slot.caption(str(e))
                    error_count += 1

                finally:
                    if dest_path.exists():
                        dest_path.unlink()

                # Gesamt-Fortschritt nach jeder abgeschlossenen Datei
                overall_pct = (file_idx + 1) / total_files
                overall_bar.progress(
                    overall_pct,
                    text=f"Datei {file_idx + 1} / {total_files} abgeschlossen"
                )

            overall_label.caption(
                f"✅ Alle Dateien verarbeitet — "
                f"{success_count} erfolgreich, {error_count} Fehler"
            )

            st.divider()
            col1, col2 = st.columns(2)
            col1.metric("Erfolgreich", success_count)
            col2.metric("Fehler", error_count)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — Search
# ──────────────────────────────────────────────────────────────────────────────
with tab_search:
    st.subheader("Semantische Suche")

    query_input = st.text_area(
        "Suchanfrage",
        placeholder="Was möchtest du wissen?",
        height=100,
    )

    if st.button("🔍 Suchen", type="primary") and query_input.strip():
        with st.spinner("Einbetten & Suchen …"):
            try:
                results = query_rag(
                    query=query_input,
                    top_k=top_k,
                    similarity_threshold=similarity_threshold,
                    filter_type=filter_type,
                )
            except Exception as e:
                st.error(f"Fehler bei der Suche: {e}")
                results = None

        if results:
            st.divider()
            st.subheader("💬 Antwort")
            st.markdown(results["answer"])

            st.divider()
            st.subheader(f"📌 Quellen ({len(results['sources'])} Treffer)")

            for idx, src in enumerate(results["sources"], 1):
                with st.expander(
                    f"#{idx} · {src['content_type'].upper()} · "
                    f"Similarity: {src['similarity']:.3f} · {src.get('source', 'unbekannt')}"
                ):
                    if src["content_type"] == "image" and src.get("file_data"):
                        import base64
                        try:
                            img_bytes = base64.b64decode(src["file_data"])
                            st.image(img_bytes, use_container_width=True)
                        except Exception:
                            st.warning("Bild konnte nicht dekodiert werden.")
                    elif src.get("content"):
                        st.text(src["content"][:2000])
                    else:
                        st.caption("Kein Textinhalt verfügbar.")
        elif results is not None:
            st.warning("Keine passenden Dokumente gefunden. Threshold senken oder anderen Content Type wählen.")


# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — Browse
# ──────────────────────────────────────────────────────────────────────────────
with tab_browse:
    st.subheader("Gespeicherte Dokumente")

    if st.button("🔄 Aktualisieren"):
        st.rerun()

    try:
        docs = list_documents(limit=200)
    except Exception as e:
        st.error(f"Fehler beim Laden: {e}")
        docs = []

    if not docs:
        st.info("Noch keine Dokumente in der Datenbank.")
    else:
        st.caption(f"{len(docs)} Einträge (max. 200 angezeigt)")

        for doc in docs:
            col1, col2, col3, col4, col5 = st.columns([1, 2, 3, 2, 1])
            with col1:
                st.text(str(doc["id"]))
            with col2:
                st.text(doc.get("content_type", "–"))
            with col3:
                # None-safe: content kann NULL sein bei Audio/Video-Chunks
                content_text = doc.get("content") or ""
                preview = content_text[:80]
                st.text(preview + ("…" if len(content_text) > 80 else ""))
            with col4:
                st.text(doc.get("source", "–")[-40:])
            with col5:
                if st.button("🗑", key=f"del_{doc['id']}"):
                    try:
                        delete_document(doc["id"])
                        st.success(f"Dokument {doc['id']} gelöscht.")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))