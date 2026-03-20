"""
3dv.py — Vektor-Visualisierung & DB-Statistiken
Eigenständige Streamlit-App (Port 8502).

Starten: streamlit run 3dv.py --server.port 8502
Oder via docker-compose als Service "viz".
"""

import os
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go

from lib.db import get_embeddings_for_viz, get_db_stats
from lib.viz import reduce_dimensions, build_3d_figure

MEDIA_DIR = Path("/app/media")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Visualisierung",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔬 Vektor-Visualisierung & Statistiken")
st.caption("UMAP 3D · pgvector Datenbank · /media Dateisystem")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ UMAP Parameter")
    max_points = st.slider(
        "Max. Datenpunkte",
        min_value=50, max_value=2000, value=500, step=50,
        help="Mehr Punkte = genauere Darstellung, aber langsamer",
    )
    n_neighbors = st.slider(
        "n_neighbors",
        min_value=5, max_value=50, value=15, step=5,
        help="Lokale Nachbarschaftsgröße — klein = lokale Struktur, groß = globale Struktur",
    )
    min_dist = st.slider(
        "min_dist",
        min_value=0.01, max_value=0.99, value=0.1, step=0.01,
        format="%.2f",
        help="Minimaler Abstand zwischen Punkten — klein = dichtere Cluster",
    )
    st.divider()
    st.link_button("← Zurück zur RAG-App", url="http://localhost:8501")


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_viz, tab_stats, tab_media, tab_saved, tab_explain = st.tabs([
    "🌐 3D Vektor-Map",
    "📊 DB-Statistiken",
    "💾 Speicher & Dateien",
    "📁 Saved 3D-VG",
    "🧠 RAG Erklärer",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — 3D Visualisierung
# ══════════════════════════════════════════════════════════════════════════════
with tab_viz:
    st.subheader("3D Vektor-Embedding Map")
    st.caption(
        "Jeder Punkt = ein Chunk. Farbe = Content-Typ. "
        "Semantisch ähnliche Inhalte clustern zusammen. "
        "Hover für Details."
    )

    col_btn, col_info = st.columns([2, 5])
    with col_btn:
        calc_btn = st.button("🔄 Visualisierung berechnen", type="primary")
    with col_info:
        st.info(
            "UMAP-Berechnung dauert je nach Datenmenge 10–60 Sekunden. "
            "Ergebnis wird im Session-Cache gehalten."
        )

    if calc_btn or "umap_fig" in st.session_state:
        if calc_btn or "umap_fig" not in st.session_state:
            with st.spinner(f"Lade {max_points} Embeddings aus DB …"):
                try:
                    rows = get_embeddings_for_viz(limit=max_points)
                except Exception as e:
                    st.error(f"DB-Fehler: {e}")
                    rows = []

            if not rows:
                st.warning("Keine Embeddings in der DB.")
            else:
                embeddings = [r["embedding"] for r in rows]
                metadata   = [
                    {k: v for k, v in r.items() if k != "embedding"}
                    for r in rows
                ]

                st.info(f"{len(embeddings)} Embeddings geladen — starte UMAP …")

                with st.spinner("UMAP 3D-Reduktion läuft …"):
                    try:
                        coords = reduce_dimensions(
                            embeddings,
                            n_neighbors=n_neighbors,
                            min_dist=min_dist,
                        )
                        fig = build_3d_figure(
                            coords, metadata,
                            title=f"Vektor-Map — {len(embeddings)} Chunks",
                        )
                        st.session_state["umap_fig"]      = fig
                        st.session_state["umap_n"]        = len(embeddings)
                        st.session_state["umap_params"]   = (n_neighbors, min_dist)
                    except ImportError as e:
                        st.error(str(e))
                        st.code("pip install umap-learn")

        if "umap_fig" in st.session_state:
            params = st.session_state.get("umap_params", ())
            if params and params != (n_neighbors, min_dist):
                st.warning(
                    "Parameter geändert — bitte 'Visualisierung berechnen' erneut klicken."
                )
            st.plotly_chart(
                st.session_state["umap_fig"],
                use_container_width=True,
            )
            st.caption(
                f"Dargestellt: {st.session_state.get('umap_n', 0)} Chunks · "
                f"n_neighbors={n_neighbors} · min_dist={min_dist}  ·  "
                f"💡 Kamera-Icon in der Chart-Toolbar oben rechts = PNG-Export"
            )

            # ── Download-Buttons ───────────────────────────────────────
            import plotly.io as pio
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            dl1, dl2 = st.columns(2)

            with dl1:
                # HTML-Export — immer verfügbar, kein kaleido/Chrome nötig
                html_str = pio.to_html(
                    st.session_state["umap_fig"],
                    full_html=True,
                    include_plotlyjs=True,
                )
                st.download_button(
                    label="⬇ HTML (interaktiv, empfohlen)",
                    data=html_str.encode("utf-8"),
                    file_name=f"umap_3d_{timestamp}.html",
                    mime="text/html",
                    key="dl_html",
                    help="Vollständig interaktive 3D-Ansicht — im Browser öffnen, drehen, zoomen",
                )

            with dl2:
                # JSON in /app/snapshots speichern (persistent)
                import json
                from pathlib import Path
                SNAP_DIR = Path("/app/snapshots")
                SNAP_DIR.mkdir(parents=True, exist_ok=True)

                snap_name = st.text_input(
                    "Snapshot-Name",
                    value=f"umap_{timestamp}",
                    key="snap_name",
                    help="Name unter dem die 3D-Momentaufnahme gespeichert wird",
                )
                if st.button("💾 Snapshot speichern", key="btn_save_snap"):
                    if snap_name.strip():
                        fig_json = st.session_state["umap_fig"].to_json()
                        meta = {
                            "name":       snap_name.strip(),
                            "timestamp":  timestamp,
                            "n_chunks":   st.session_state.get("umap_n", 0),
                            "n_neighbors": n_neighbors,
                            "min_dist":   min_dist,
                            "figure":     json.loads(fig_json),
                        }
                        save_path = SNAP_DIR / f"{snap_name.strip()}.json"
                        save_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
                        st.success(f"Snapshot gespeichert: {save_path.name}")
                        st.rerun()
                    else:
                        st.warning("Bitte einen Namen eingeben.")

                # Auch als Download anbieten
                fig_json = st.session_state["umap_fig"].to_json()
                st.download_button(
                    label="⬇ JSON herunterladen",
                    data=fig_json.encode("utf-8"),
                    file_name=f"umap_3d_{timestamp}.json",
                    mime="application/json",
                    key="dl_json",
                    help="Plotly-Figure als JSON — für eigene Weiterverarbeitung",
                )

            st.caption(
                "💡 PNG/SVG-Export: Kamera-Icon in der Chart-Toolbar oben rechts "
                "(direkt im Browser, ohne Installation)"
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DB-Statistiken
# ══════════════════════════════════════════════════════════════════════════════
with tab_stats:
    st.subheader("Datenbankstatistiken")

    if st.button("🔄 Statistiken laden", key="load_stats"):
        st.session_state.pop("db_stats", None)

    if "db_stats" not in st.session_state:
        with st.spinner("Lade DB-Statistiken …"):
            try:
                st.session_state["db_stats"] = get_db_stats()
            except Exception as e:
                st.error(f"DB-Fehler: {e}")

    if "db_stats" in st.session_state:
        stats = st.session_state["db_stats"]

        # ── Größen ────────────────────────────────────────────────────
        st.markdown("#### Datenbankgröße")
        sizes = stats["sizes"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Gesamt-DB", sizes.get("total_db_size", "–"))
        c2.metric("documents", sizes.get("documents_size", "–"))
        c3.metric("chat_messages", sizes.get("chat_size", "–"))

        import shutil
        disk     = shutil.disk_usage("/")
        total_gb = disk.total / 1024**3
        free_gb  = disk.free  / 1024**3
        used_db  = sizes.get("total_db_bytes", 0)
        used_pct = min(used_db / disk.total, 1.0)

        # Hinweis: Docker Desktop VM-Disk, nicht physische Mac-Festplatte
        st.caption(
            f"ℹ️ Docker Desktop Virtual Disk: {total_gb:.0f} GB gesamt · "
            f"{free_gb:.0f} GB frei · "
            f"Größe anpassen: Docker Desktop → Settings → Resources → Disk image size"
        )
        st.progress(
            used_pct,
            text=f"DB-Größe: {used_db/1024**2:.1f} MB von {total_gb:.0f} GB VM-Disk ({used_pct*100:.3f}%)"
        )

        st.divider()

        # ── Chunks pro Typ ────────────────────────────────────────────
        st.markdown("#### Chunks pro Content-Typ")
        import pandas as pd

        by_type = stats.get("by_type", [])
        if by_type:
            df_type = pd.DataFrame(by_type)
            df_type.columns = ["Typ", "Chunks", "Quellen", "Ø Inhaltslänge (Zeichen)"]
            st.dataframe(df_type, use_container_width=True, hide_index=True)

            # Donut Chart
            fig_donut = go.Figure(go.Pie(
                labels=[r["Typ"] for r in df_type.to_dict("records")],
                values=[r["Chunks"] for r in df_type.to_dict("records")],
                hole=0.5,
                marker=dict(colors=["#4F8EF7","#F7A24F","#4FF7A2","#F74F8E","#A24FF7"]),
            ))
            fig_donut.update_layout(
                paper_bgcolor="#0E1117",
                plot_bgcolor="#0E1117",
                font=dict(color="#FAFAFA"),
                margin=dict(l=0, r=0, t=20, b=0),
                height=300,
                showlegend=True,
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        st.divider()

        # ── Top Quellen ───────────────────────────────────────────────
        st.markdown("#### Top 10 Quellen nach Chunk-Anzahl")
        top = stats.get("top_sources", [])
        if top:
            df_top = pd.DataFrame(top)
            df_top.columns = ["Quelle", "Typ", "Chunks", "Start", "Ende"]
            st.dataframe(df_top, use_container_width=True, hide_index=True)

        st.divider()

        # ── Chat-Statistiken ──────────────────────────────────────────
        st.markdown("#### Prompt-Verlauf")
        cs = stats.get("chat_stats", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sessions",          cs.get("sessions", 0))
        c2.metric("Nachrichten gesamt", cs.get("total_messages", 0))
        c3.metric("Fragen (User)",      cs.get("user_messages", 0))
        c4.metric("Antworten (KI)",     cs.get("ai_messages", 0))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Speicher & Dateien
# ══════════════════════════════════════════════════════════════════════════════
with tab_media:
    st.subheader("Speicher & /media Verzeichnis")

    if st.button("🔄 Dateien einlesen", key="load_media"):
        st.session_state.pop("media_files", None)

    if "media_files" not in st.session_state:
        files = []
        if MEDIA_DIR.exists():
            for f in sorted(MEDIA_DIR.iterdir()):
                if f.is_file() and not f.name.startswith("."):
                    size_mb = f.stat().st_size / 1024**2
                    ext     = f.suffix.lower().lstrip(".")
                    ct      = (
                        "video" if ext in {"mp4","mov","avi","mkv","webm"} else
                        "audio" if ext in {"mp3","wav","ogg","flac","m4a","aac"} else
                        "image" if ext in {"jpg","jpeg","png","gif","webp","bmp"} else
                        "pdf"   if ext == "pdf" else "text"
                    )
                    files.append({
                        "Dateiname":  f.name,
                        "Typ":        ct,
                        "Größe (MB)": round(size_mb, 2),
                    })
        st.session_state["media_files"] = files

    files = st.session_state.get("media_files", [])

    if not files:
        st.info("/media ist leer oder nicht erreichbar.")
    else:
        import pandas as pd

        df_files = pd.DataFrame(files)
        total_mb = df_files["Größe (MB)"].sum()

        # Metriken
        c1, c2, c3 = st.columns(3)
        c1.metric("Dateien gesamt", len(files))
        c2.metric("Gesamtgröße",    f"{total_mb:.1f} MB")
        c3.metric("Ø Dateigröße",   f"{total_mb/len(files):.1f} MB")

        # Tatsächlichen Festplattenplatz des /media Volumes ermitteln
        import shutil
        disk_media = shutil.disk_usage(str(MEDIA_DIR))
        total_vol_gb = disk_media.total / 1024**3
        free_vol_gb  = disk_media.free  / 1024**3
        used_p = min(total_mb / (disk_media.total / 1024**2), 1.0)
        st.progress(
            used_p,
            text=(
                f"/media belegt: {total_mb:.1f} MB  ·  "
                f"Volume gesamt: {total_vol_gb:.0f} GB  ·  "
                f"Frei: {free_vol_gb:.0f} GB  ·  "
                f"Auslastung: {used_p*100:.2f}%"
            )
        )

        st.divider()

        # Tabelle
        st.dataframe(
            df_files,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Dateiname":  st.column_config.TextColumn("Dateiname",  width="large"),
                "Typ":        st.column_config.TextColumn("Typ",        width="small"),
                "Größe (MB)": st.column_config.NumberColumn("Größe (MB)", format="%.2f MB"),
            },
        )

        # Balkendiagramm Dateigröße pro Typ
        import plotly.express as px
        df_by_type = df_files.groupby("Typ")["Größe (MB)"].sum().reset_index()
        fig_bar = px.bar(
            df_by_type,
            x="Typ", y="Größe (MB)",
            color="Typ",
            color_discrete_map={
                "video": "#A24FF7", "audio": "#F74F8E",
                "image": "#4FF7A2", "pdf":   "#F7A24F", "text": "#4F8EF7",
            },
            title="Speicherverbrauch nach Typ",
        )
        fig_bar.update_layout(
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            font=dict(color="#FAFAFA"),
            showlegend=False,
            height=300,
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Saved 3D-VG
# ══════════════════════════════════════════════════════════════════════════════
with tab_saved:
    st.subheader("Gespeicherte 3D-Momentaufnahmen")

    import json
    import pandas as pd
    from pathlib import Path
    import plotly.io as pio

    SNAP_DIR = Path("/app/snapshots")
    SNAP_DIR.mkdir(parents=True, exist_ok=True)

    col_ref2, col_info2 = st.columns([1, 4])
    with col_ref2:
        if st.button("🔄 Aktualisieren", key="refresh_snaps"):
            st.session_state.pop("snap_selection", None)
            st.rerun()
    with col_info2:
        st.caption(
            "Snapshots werden in /app/snapshots gespeichert. "
            "Zeile anklicken → 3D-Chart laden."
        )

    # Alle JSON-Dateien einlesen
    snap_files = sorted(SNAP_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)

    if not snap_files:
        st.info("Noch keine Snapshots gespeichert. Im Tab '3D Vektor-Map' einen Snapshot anlegen.")
    else:
        # Metadaten-Tabelle aufbauen
        rows = []
        for f in snap_files:
            try:
                meta = json.loads(f.read_text())
                rows.append({
                    "Dateiname":   f.name,
                    "Name":        meta.get("name", f.stem),
                    "Gespeichert": meta.get("timestamp", "–"),
                    "Chunks":      meta.get("n_chunks", "–"),
                    "n_neighbors": meta.get("n_neighbors", "–"),
                    "min_dist":    meta.get("min_dist", "–"),
                    "Größe (KB)":  round(f.stat().st_size / 1024, 1),
                })
            except Exception:
                rows.append({
                    "Dateiname": f.name,
                    "Name": f.stem,
                    "Gespeichert": "–", "Chunks": "–",
                    "n_neighbors": "–", "min_dist": "–",
                    "Größe (KB)": round(f.stat().st_size / 1024, 1),
                })

        df_snaps = pd.DataFrame(rows)

        st.caption(f"{len(rows)} Snapshot(s) vorhanden")

        selection = st.dataframe(
            df_snaps,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            column_config={
                "Dateiname":   st.column_config.TextColumn("Dateiname",   width="medium"),
                "Name":        st.column_config.TextColumn("Name",        width="medium"),
                "Gespeichert": st.column_config.TextColumn("Gespeichert", width="medium"),
                "Chunks":      st.column_config.NumberColumn("Chunks",    width="small"),
                "n_neighbors": st.column_config.NumberColumn("n_neighbors", width="small"),
                "min_dist":    st.column_config.NumberColumn("min_dist",  width="small", format="%.2f"),
                "Größe (KB)":  st.column_config.NumberColumn("KB",        width="small", format="%.1f"),
            },
        )

        selected_rows = selection.selection.rows if selection and selection.selection else []

        if selected_rows:
            sel_idx  = selected_rows[0]
            sel_file = snap_files[sel_idx]
            sel_row  = rows[sel_idx]

            st.divider()

            act1, act2, act3 = st.columns([2, 2, 2])

            with act1:
                if st.button("🌐 3D-Chart laden", type="primary", key="btn_load_snap"):
                    try:
                        meta = json.loads(sel_file.read_text())
                        fig  = pio.from_json(json.dumps(meta["figure"]))
                        st.session_state["loaded_snap_fig"]  = fig
                        st.session_state["loaded_snap_name"] = sel_row["Name"]
                        st.session_state["loaded_snap_meta"] = sel_row
                    except Exception as e:
                        st.error(f"Laden fehlgeschlagen: {e}")

            with act2:
                # Download des gespeicherten JSON
                try:
                    raw = sel_file.read_bytes()
                    st.download_button(
                        label="⬇ JSON herunterladen",
                        data=raw,
                        file_name=sel_file.name,
                        mime="application/json",
                        key="dl_snap_json",
                    )
                except Exception:
                    pass

            with act3:
                if st.button("🗑 Snapshot löschen", type="secondary", key="btn_del_snap"):
                    sel_file.unlink()
                    st.session_state.pop("loaded_snap_fig", None)
                    st.success(f"'{sel_row['Name']}' gelöscht.")
                    st.rerun()

        # ── Geladener Chart ────────────────────────────────────────────
        if "loaded_snap_fig" in st.session_state:
            st.divider()
            meta_info = st.session_state.get("loaded_snap_meta", {})
            st.markdown(
                f"**{st.session_state['loaded_snap_name']}** · "
                f"{meta_info.get('Chunks','?')} Chunks · "
                f"n_neighbors={meta_info.get('n_neighbors','?')} · "
                f"min_dist={meta_info.get('min_dist','?')}"
            )
            st.plotly_chart(
                st.session_state["loaded_snap_fig"],
                use_container_width=True,
                key="saved_chart",
            )

            # HTML-Export des geladenen Snapshots
            html_out = pio.to_html(
                st.session_state["loaded_snap_fig"],
                full_html=True,
                include_plotlyjs=True,
            )
            st.download_button(
                label="⬇ Als HTML exportieren",
                data=html_out.encode("utf-8"),
                file_name=f"{st.session_state['loaded_snap_name']}.html",
                mime="text/html",
                key="dl_loaded_html",
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — RAG Erklärer
# ══════════════════════════════════════════════════════════════════════════════
with tab_explain:
    st.subheader("🧠 RAG-Pipeline Erklärer")
    st.caption("Klicke auf eine Stufe um Details zu sehen")

    # ── Stufen-Daten ──────────────────────────────────────────────────
    STUFEN = [
        {
            "nr": 1, "emoji": "🟣", "phase": "Input",
            "titel": "Nutzeranfrage & Tokenisierung",
            "kurz": "Text wird in Tokens (Teilwörter) zerlegt — die atomaren Einheiten des Modells.",
            "detail": """
**Was passiert:**
Die Frage *"Was kann ich in der SU24 pflegen?"* wird tokenisiert.

**Tokenisierung-Beispiel:**

| Token | Was | kann | ich | in | der | SU | ##24 | pfl | ##eg | ##en | ? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ID | 2001 | 1234 | 456 | 189 | 234 | 9821 | 1024 | 4521 | 892 | 341 | 136 |

**Wichtig:** `SU24` → `SU` + `##24` (## = Fortsetzungstoken). Das Modell arbeitet nur mit Token-IDs — nie mit Rohtext.
""",
            "farbe": "#534AB7",
        },
        {
            "nr": 2, "emoji": "🟣", "phase": "Embed",
            "titel": "Query-Embedding (3072 Dims)",
            "kurz": "Gemini wandelt die Tokens in einen 3072-dimensionalen Vektor um.",
            "detail": """
**Was passiert:**
`embed_query()` → `gemini-embedding-2-preview` → **[0.42, -0.17, 0.83, … × 3072]**

**Schlüssel-Parameter:**
- `task_type="RETRIEVAL_QUERY"` → optimiert für Suche, nicht für Dokumentspeicherung
- **L2-Normalisierung** via numpy → Cosinus-Ähnlichkeit = Skalarprodukt (präziser)
- **3072 Dimensionen** → jede kodiert ein semantisches Merkmal

**Warum wichtig:** RETRIEVAL_QUERY vs RETRIEVAL_DOCUMENT erzeugen unterschiedliche Vektoren für dieselbe Phrase — das verbessert die Suche erheblich.
""",
            "farbe": "#534AB7",
        },
        {
            "nr": 3, "emoji": "🟢", "phase": "Search",
            "titel": "Cosinus-Suche in pgvector",
            "kurz": "Query-Vektor wird mit allen gespeicherten Embeddings verglichen.",
            "detail": """
**Was passiert:**
```sql
SELECT * FROM match_documents(
    query_embedding   := $1::vector,
    match_count       := 5,
    similarity_thresh := 0.35,
    filter_type       := NULL
)
```

**Cosinus-Ähnlichkeit:** `1 - (embedding <=> query_embedding) >= threshold`

**Threshold (Standard: 0.35):**
Nur Chunks mit Cosinus-Ähnlichkeit ≥ 0.35 werden zurückgegeben — alles darunter wird verworfen.

| Chunk | Ähnlichkeit | Status |
|---|---|---|
| SU24 Berechtigungsobjekte | 0.76 | ✅ Treffer |
| Rollenbau Best Practice | 0.72 | ✅ Treffer |
| DSAG Prüfleitfaden | 0.44 | ✅ Treffer |
| Windows Installation | 0.21 | ❌ Verworfen |

**Kein HNSW-Index:** pgvector-Limit = 2000 Dims, unsere Embeddings haben 3072 → Exact Search (Sequential Scan). Bei >10.000 Chunks: IVFFlat als Alternative.
""",
            "farbe": "#1D9E75",
        },
        {
            "nr": 4, "emoji": "🟢", "phase": "Retrieve",
            "titel": "Chunk-Retrieval mit Timestamps",
            "kurz": "Top-K Chunks werden geladen — Transkripttext, Zeitstempel, Quelle.",
            "detail": """
**Jeder Chunk enthält:**

| Feld | Beispiel |
|---|---|
| `content` | "...Berechtigungsvorschlagswerte in der SU24 definieren..." |
| `timestamp_start` | 127.3 → **02:07** |
| `timestamp_end` | 254.1 → **04:14** |
| `source` | 001-Best_Practice_...SU24_Xiting_Xpert-1.mp4 |
| `similarity` | 0.76 |
| `content_type` | video |

**Video-Player:** Der Such-Tab öffnet `st.video(start_time=ts_start - 10)` → Wiedergabe startet 10 Sekunden vor der Fundstelle für Kontext.
""",
            "farbe": "#1D9E75",
        },
        {
            "nr": 5, "emoji": "🟡", "phase": "Assemble",
            "titel": "Kontext-Assembly & System-Prompt",
            "kurz": "Chunks + Frage werden strukturiert an das LLM übergeben.",
            "detail": """
**System-Prompt (Auszug):**
> Beantworte die Frage ausschließlich auf Basis der bereitgestellten Dokumente.
> Zitiere Quellen: [Quelle: Datei, Typ, Zeitstempel]
> Kein Trainingswissen — nur aus dem gegebenen Kontext antworten.

**User-Message Struktur:**
```
Anfrage: Was kann ich in der SU24 pflegen?

--- Dokument 1 ---
Typ: video | Quelle: 001-Best_...mp4
Ähnlichkeit: 0.76
Zeitstempel: 02:07 – 04:14
Inhalt: ...Berechtigungsvorschlagswerte in der SU24...

--- Dokument 2 ---
...
```

**Modell-Wahl:** Gemini Flash Lite (Standard, günstiger) oder o4-mini (optional) — umschaltbar in der Sidebar.
""",
            "farbe": "#BA7517",
        },
        {
            "nr": 6, "emoji": "🟡", "phase": "Generate",
            "titel": "Grounded Antwort mit Zeitstempel-Zitaten",
            "kurz": "Das Modell antwortet nur aus dem Kontext — mit präzisen Zeitstempel-Zitaten.",
            "detail": """
**Beispiel-Antwort:**
> In der Transaktion SU24 können Sie für jede SAP-Anwendung definieren, welche Berechtigungsobjekte geprüft werden und bereits Werte hinterlegen
> *[Quelle: 001-Best_Practice...mp4, Typ: video, 02:07 – 04:14]*
>
> Sie können auch einstellen, ob ein Objekt geprüft werden soll.
> *[Quelle: 001-Best_Practice...mp4, Typ: video, 08:20 – 10:24]*

**Kernvorteil RAG vs. reines LLM:**

| | Reines LLM | RAG |
|---|---|---|
| Quelle | Trainingsdaten (unklar) | Deine Videos/PDFs |
| Nachprüfbar | ❌ | ✅ mit Zeitstempel |
| Aktuell | ❌ (Cutoff) | ✅ (deine Uploads) |
| Halluzination | Möglich | Stark reduziert |
""",
            "farbe": "#BA7517",
        },
        {
            "nr": 7, "emoji": "🔴", "phase": "Ingest",
            "titel": "Ingest-Pipeline (Upload → Vektor)",
            "kurz": "Whisper transkribiert, Chunking teilt auf, Gemini bettet ein, pgvector speichert.",
            "detail": """
**Pipeline-Schritte:**

```
MP4 Upload (/media)
    ↓
ffmpeg → Audio-Spur extrahieren (16kHz Mono MP3)
    ↓
Whisper-1 → Transkription mit Segment-Timestamps
    ↓
_segments_to_chunks() → 120s Text-Chunks mit ts_start/ts_end
    ↓
embed_document_text() → Gemini 3072-dim Vektor
    ↓
insert_document() → pgvector (content + embedding + timestamps)
    ↓
Originaldatei bleibt in /media (NICHT gelöscht!)
```

**Design-Entscheid:** Videos werden nicht als Binärdaten eingebettet — zu groß, semantisch schwach. Stattdessen: Whisper transkribiert die Sprache. Der Text wird eingebettet, das Original bleibt für `st.video()`.

**Unterstützte Formate:** MP4, MOV, AVI (Video) · MP3, WAV, M4A (Audio) · PDF · TXT, MD, CSV · JPG, PNG, WebP
""",
            "farbe": "#993C1D",
        },
    ]

    # ── Navigation: Dots ──────────────────────────────────────────────
    if "erklaerer_stufe" not in st.session_state:
        st.session_state["erklaerer_stufe"] = 0

    # Dot-Navigation
    dot_cols = st.columns(len(STUFEN))
    for i, s in enumerate(STUFEN):
        with dot_cols[i]:
            is_active = st.session_state["erklaerer_stufe"] == i
            is_done   = st.session_state["erklaerer_stufe"] > i
            if is_done:
                bg = "#0F6E56"; fg = "#9FE1CB"
            elif is_active:
                bg = s["farbe"]; fg = "#FFFFFF"
            else:
                bg = "#1C2033"; fg = "#888888"
            if st.button(
                str(s["nr"]),
                key=f"dot_{i}",
                use_container_width=True,
                help=s["titel"],
            ):
                st.session_state["erklaerer_stufe"] = i
                st.rerun()

    st.divider()

    # ── Aktive Stufe ──────────────────────────────────────────────────
    idx_s = st.session_state["erklaerer_stufe"]
    s     = STUFEN[idx_s]

    # Header
    phase_colors = {"Input":"#534AB7","Embed":"#534AB7","Search":"#1D9E75",
                    "Retrieve":"#1D9E75","Assemble":"#BA7517","Generate":"#BA7517","Ingest":"#993C1D"}
    pc = phase_colors.get(s["phase"], "#888")

    col_badge, col_title = st.columns([1, 8])
    with col_badge:
        st.markdown(
            f'<div style="background:{pc};color:white;padding:4px 12px;'
            f'border-radius:20px;font-size:12px;font-weight:600;text-align:center;margin-top:4px">'
            f'{s["phase"]}</div>',
            unsafe_allow_html=True,
        )
    with col_title:
        st.markdown(f"### {s['emoji']} Stufe {s['nr']} — {s['titel']}")

    st.caption(s["kurz"])
    st.markdown(s["detail"])

    # ── Navigations-Buttons ───────────────────────────────────────────
    st.divider()
    nav1, nav2, nav3 = st.columns([2, 3, 2])
    with nav1:
        if idx_s > 0:
            if st.button("← Zurück", key="erklaerer_back", use_container_width=True):
                st.session_state["erklaerer_stufe"] -= 1
                st.rerun()
    with nav2:
        st.caption(f"Stufe {idx_s+1} von {len(STUFEN)}")
    with nav3:
        if idx_s < len(STUFEN) - 1:
            if st.button("Weiter →", key="erklaerer_next",
                         type="primary", use_container_width=True):
                st.session_state["erklaerer_stufe"] += 1
                st.rerun()
        else:
            if st.button("↺ Neu starten", key="erklaerer_reset",
                         type="primary", use_container_width=True):
                st.session_state["erklaerer_stufe"] = 0
                st.rerun()