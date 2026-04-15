# RAG Multimodal — SAP Knowledge Assistant

Ein multimodales Retrieval-Augmented Generation (RAG) System für SAP-Berechtigungsthemen. Verarbeitet Videos, PDFs, Bilder und Texte — durchsucht sie semantisch und beantwortet Fragen mit Zeitstempel-genauen Quellenangaben.

---

## Features

- **Multimodales Ingest** — MP4, MP3, PDF, TXT, MD, CSV, JPG, PNG
- **Whisper-Transkription** — automatische Spracherkennung mit Segment-Timestamps
- **Gemini Embeddings** — 3072-dimensionale Vektoren (gemini-embedding-2-preview)
- **pgvector Vektorsuche** — Cosinus-Ähnlichkeit mit konfigurierbarem Threshold
- **Video-Player** — springt direkt zur Fundstelle (−10s Kontext)
- **3D Vektor-Visualisierung** — UMAP-Reduktion mit Plotly, Snapshot-Export
- **RAG Erklärer** — interaktiver Stepper durch alle 7 Pipeline-Stufen
- **Transformer Explainer** — GPT-2 live im Browser (poloclub.github.io)
- **Prompt-Verlauf** — Sessions mit Export, Quellen-Wiedergabe
- **Browse & Verwaltung** — DataFrame mit Checkbox-Selektion, Bulk-Delete

---

## Architektur

```
┌─────────────────────────────────────────────────────┐
│  RAG Multimodal App (Port 8501)                     │
│  Upload · Suche · Prompt-Verlauf · Browse           │
├─────────────────────────────────────────────────────┤
│  RAG Visualisierung (Port 8502)                     │
│  3D Vektor-Map · DB-Statistiken · RAG Erklärer      │
├─────────────────────────────────────────────────────┤
│  Transformer Explainer (Port 8503)                  │
│  GPT-2 live · Attention · MLP · Softmax animiert    │
├─────────────────────────────────────────────────────┤
│  pgvector / PostgreSQL 16 (Port 5432)               │
│  Embeddings · Chat-Sessions · Dokumente             │
└─────────────────────────────────────────────────────┘
```

---

## Voraussetzungen

- Docker Desktop (macOS/Linux)
- Python 3.11+
- Node.js 20+ (nur für Transformer Explainer Build)
- API-Keys: Google Gemini, OpenAI

---

## Setup

### 1. Repository klonen

```bash
git clone https://github.com/hklein42/rag-multimodal-online.git
cd rag-multimodal-online
```

### 2. Umgebungsvariablen konfigurieren

```bash
cp .env.example .env
```

`.env` befüllen:

```env
GEMINI_API_KEY=dein-gemini-api-key
OPENAI_API_KEY=dein-openai-api-key
POSTGRES_USER=postgres
POSTGRES_PASSWORD=mysecretpassword42
POSTGRES_DB=rag_db
DATABASE_URL=postgresql://postgres:mysecretpassword42@db:5432/rag_db
REASONING_BACKEND=gemini
```

### 3. Transformer Explainer lokal bauen (einmalig)

```bash
cd transformer-explainer
npm install --legacy-peer-deps
npx svelte-kit sync
npm run build
cd ..
```

### 4. Docker Container starten

```bash
docker-compose up -d
```

Beim ersten Start werden alle Images gebaut — das dauert ca. 5–10 Minuten.

---

## Container & Ports

| Container | Image | Port | Beschreibung |
|---|---|---|---|
| `rag-multimodal` | python:3.11-slim | 8501 | RAG Haupt-App (Streamlit) |
| `rag-viz` | python:3.11-slim | 8502 | Visualisierung & Statistiken |
| `rag-transformer-explainer` | nginx:alpine | 8503 | Transformer Explainer |
| `rag-pgvector-db` | pgvector/pgvector:pg16 | 5432 | PostgreSQL mit pgvector |

---

## URLs nach dem Start

| URL | Beschreibung |
|---|---|
| http://localhost:8501 | RAG Multimodal App |
| http://localhost:8502 | 3D Visualisierung & Statistiken |
| http://localhost:8503 | Transformer Explainer (GPT-2) |

---

## Verzeichnisstruktur

```
rag-multimodal/
├── app.py                          # Streamlit RAG-App (4 Tabs)
├── 3dv.py                          # Visualisierungs-App (5 Tabs)
├── Dockerfile                      # Python/Streamlit Container
├── transformer-explainer.Dockerfile # Nginx Container für SvelteKit
├── docker-compose.yml              # 4 Services
├── requirements.txt                # Python Dependencies
├── .env.example                    # Umgebungsvariablen Vorlage
├── lib/
│   ├── chunker.py                  # Text/PDF/Audio/Video Chunking
│   ├── db.py                       # pgvector DB-Operationen
│   ├── embedder.py                 # Gemini Embedding (3072 dims)
│   ├── rag.py                      # Ingest-Pipeline + Query
│   ├── viz.py                      # UMAP + Plotly 3D
│   └── codex.py                    # LLM Reasoning (Gemini/OpenAI)
├── sql/
│   └── init.sql                    # DB-Schema + match_documents()
├── transformer-explainer/          # SvelteKit App (GPT-2)
│   ├── src/
│   ├── static/
│   │   ├── model-v2/               # GPT-2 ONNX (63 Chunks)
│   │   └── xenova-gpt2/            # Tokenizer (lokal)
│   └── build/                      # Fertiger Build (nach npm run build)
└── .streamlit/
    └── config.toml                 # maxUploadSize=300MB
```

---

## Unterstützte Dateiformate

| Typ | Formate |
|---|---|
| Video | MP4, MOV, AVI, MKV, WebM |
| Audio | MP3, WAV, OGG, FLAC, M4A, AAC |
| Dokument | PDF, TXT, MD, CSV |
| Bild | JPG, JPEG, PNG, GIF, WebP, BMP |

---

## RAG-Pipeline

```
Upload → ffmpeg (Audio) → Whisper-1 (Transkription + Timestamps)
      → Chunking (120s Segmente) → Gemini embed_document (3072 dims)
      → pgvector INSERT (content + embedding + ts_start/ts_end)
      → Originaldatei bleibt in /media

Suche → Gemini embed_query → pgvector Cosinus-Suche
      → Top-K Chunks → LLM (Gemini Flash / o4-mini)
      → Antwort mit Zeitstempel-Zitaten → Video-Player
```

---

## Nützliche Befehle

```bash
# Status prüfen
docker-compose ps

# Logs anzeigen
docker-compose logs -f app

# App neu laden (ohne Rebuild)
docker cp app.py rag-multimodal:/app/app.py
docker-compose restart app

# Datenbank zurücksetzen
docker-compose down
docker volume rm rag-multimodal_db_data
docker-compose up -d

# Alle Container stoppen
docker-compose down

# Komplett neu bauen
docker-compose build --no-cache
docker-compose up -d
```

---

## Git Workflow

```bash
# Änderungen committen
git add .
git commit -m "Beschreibung der Änderung"

# Zu GitHub pushen
git push origin main

# Zu Gitea pushen (Self-Hosted)
git push gitea main
```

---

## Technologie-Stack

| Komponente | Technologie |
|---|---|
| Frontend | Streamlit 1.40+ |
| Embedding | Google Gemini (gemini-embedding-2-preview) |
| Reasoning | Gemini Flash Lite / OpenAI o4-mini |
| Transkription | OpenAI Whisper-1 |
| Vektordatenbank | PostgreSQL 16 + pgvector |
| 3D Visualisierung | UMAP + Plotly |
| Transformer Erklärer | SvelteKit + ONNX Runtime Web (GPT-2) |
| Container | Docker Compose |
| Reverse Proxy | Nginx (alpine) |

---

## Lizenz

MIT License — entwickelt mit Claude (Anthropic) als KI-Assistent.