"""
lib/viz.py
UMAP-Dimensionsreduktion + Plotly 3D Visualisierung.

Eingabe : Embeddings (3072 dims) aus pgvector
Ausgabe : Plotly Figure mit 3D Scatter — eingefärbt nach Content-Typ
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

# Farben pro Content-Typ
TYPE_COLORS = {
    "text":  "#4F8EF7",   # Blau
    "pdf":   "#F7A24F",   # Orange
    "image": "#4FF7A2",   # Grün
    "audio": "#F74F8E",   # Pink
    "video": "#A24FF7",   # Lila
}
DEFAULT_COLOR = "#888888"


def reduce_dimensions(
    embeddings: list[list[float]],
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """
    Reduziert Embeddings von 3072 auf 3 Dimensionen via UMAP.

    Args:
        embeddings  : Liste von Float-Vektoren (3072 dims)
        n_neighbors : UMAP — lokale Nachbarschaftsgröße (15 = ausgewogen)
        min_dist    : UMAP — minimaler Abstand zwischen Punkten im 3D-Raum
        random_state: Reproduzierbarkeit

    Returns:
        numpy Array shape (N, 3)
    """
    try:
        import umap
    except ImportError:
        raise ImportError(
            "umap-learn nicht installiert. "
            "Ausführen: pip install umap-learn"
        )

    arr = np.array(embeddings, dtype=np.float32)

    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=random_state,
        verbose=False,
    )
    return reducer.fit_transform(arr)


def build_3d_figure(
    coords: np.ndarray,
    metadata: list[dict],
    title: str = "Vektor-Embeddings 3D",
) -> go.Figure:
    """
    Erstellt eine interaktive Plotly 3D Scatter Figure.

    Args:
        coords   : (N, 3) Array aus reduce_dimensions()
        metadata : Liste von Dicts mit content_type, source, preview, id

    Returns:
        Plotly Figure
    """
    # Nach Content-Typ gruppieren für separate Traces (Legende)
    by_type: dict[str, list] = {}
    for i, meta in enumerate(metadata):
        ct = meta.get("content_type", "unknown")
        if ct not in by_type:
            by_type[ct] = []
        by_type[ct].append(i)

    fig = go.Figure()

    for ct, indices in by_type.items():
        x = coords[indices, 0]
        y = coords[indices, 1]
        z = coords[indices, 2]

        hover_texts = []
        for idx in indices:
            m = metadata[idx]
            ts = ""
            if m.get("timestamp_start") is not None:
                def fmt(s):
                    s = int(float(s)); return f"{s//60:02d}:{s%60:02d}"
                ts = f"<br>⏱ {fmt(m['timestamp_start'])} – {fmt(m['timestamp_end'])}"
            preview = (m.get("preview") or "")[:60]
            hover_texts.append(
                f"<b>ID {m.get('id')}</b> · {ct.upper()}{ts}"
                f"<br>{m.get('source','?')}"
                f"<br><i>{preview}</i>"
            )

        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode="markers",
            name=ct.upper(),
            marker=dict(
                size=5,
                color=TYPE_COLORS.get(ct, DEFAULT_COLOR),
                opacity=0.85,
                line=dict(width=0.5, color="white"),
            ),
            text=hover_texts,
            hovertemplate="%{text}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        scene=dict(
            xaxis=dict(title="UMAP-1", showgrid=True, gridcolor="#333"),
            yaxis=dict(title="UMAP-2", showgrid=True, gridcolor="#333"),
            zaxis=dict(title="UMAP-3", showgrid=True, gridcolor="#333"),
            bgcolor="#0E1117",
        ),
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font=dict(color="#FAFAFA"),
        legend=dict(
            bgcolor="#1C2033",
            bordercolor="#4F8EF7",
            borderwidth=1,
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=600,
    )
    return fig