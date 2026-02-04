# CITADEL — High-Performance RAG Pipeline

Modular Retrieval-Augmented Generation backend built on FastAPI, PostgreSQL with
pgvector, and async-first Python. CITADEL handles the full document lifecycle:
ingestion, chunking, embedding, and semantic retrieval with LLM-powered answers.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-blue.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Features

- **📄 Document Ingestion**: Upload PDF and Markdown files with automatic text extraction
- **🔪 Smart Chunking**: LangChain-based text splitting with configurable overlap
- **🧠 Local Embeddings**: sentence-transformers (all-MiniLM-L6-v2) — no API costs
- **🔍 Semantic Search**: pgvector HNSW index for fast cosine similarity
- **🤖 LLM Integration**: Ollama (Mistral 7B) with graceful fallback to mock mode
- **🖥️ Web Interface**: Streamlit chat UI with source citations
- **🔒 Deduplication**: SHA-256 content hashing prevents duplicate ingestion
- **⚡ Async-First**: Full async/await architecture with SQLAlchemy 2.0

## Tech Stack

| Layer | Technology |
|-------|------------|
| **API Framework** | FastAPI 0.109+ with Pydantic v2 |
| **Database** | PostgreSQL 16 with pgvector extension |
| **ORM** | SQLAlchemy 2.0 (async) + Alembic migrations |
| **Ingestion** | PyMuPDF (PDF), standard lib (Markdown) |
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2, 384 dims) |
| **LLM** | Ollama (Mistral 7B, local) with graceful fallback |
| **Frontend** | Streamlit with session-based chat history |
| **Cache** | Redis (prepared, not yet active) |
| **Testing** | pytest + pytest-asyncio + httpx |
| **Code Quality** | Ruff + mypy (strict) + pre-commit |

## 🚀 Quickstart

### Prerequisites
- Docker Desktop
- Python 3.11+
- Make
- (Optional) Ollama for full LLM responses

### 1. Setup

```bash
# Clone and configure
git clone https://github.com/yourusername/citadel-rag.git
cd citadel-rag

# Create environment file
cp .env.example .env
```

### 2. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# or: venv\Scripts\activate  # Windows

# Install with dev dependencies
make install
pre-commit install
```

### 3. Choose Your Development Mode

#### Option A: Full Docker (Recommended for testing)

Everything runs in containers — simplest setup.

```bash
# Start all services
make up

# Run migrations
make mig-up

# (Optional) Start Ollama for real LLM responses
# Install: brew install ollama (macOS) or see https://ollama.ai
ollama serve &
ollama pull mistral
```

**Access:** http://localhost:8501

#### Option B: Hybrid Mode (Recommended for development)

DB in Docker, API locally — enables hot-reload and debugging.

```bash
# Start only DB + Redis
make deps

# Run migrations
make mig-up

# Terminal 1: Start RAG API
make run-citadel

# Terminal 2: Start UI
make run-ui

# Terminal 3 (Optional): Start Ollama
ollama serve
```

**Access:** http://localhost:8501

### 4. Verify Everything Works

```bash
# Check all services are healthy
curl http://localhost:8001/health      # RAG API
curl http://localhost:11434/api/tags   # Ollama (if running)

# Open Web UI
open http://localhost:8501
```

| Indicator | Meaning |
|-----------|---------|
| 🟢 API Connected | RAG API is reachable |
| ⚠️ Mock Mode | Ollama not running (retrieval works, LLM mocked) |
| 🔴 API Unreachable | Docker stack not started |

### 5. Service URLs

| Service | URL | Description |
|---------|-----|-------------|
| **Web UI** | http://localhost:8501 | Streamlit chat interface |
| **RAG API** | http://localhost:8001 | FastAPI backend |
| **API Docs** | http://localhost:8001/docs | Swagger UI |
| **Atlas API** | http://localhost:8000 | Legacy notes API |
| **Ollama** | http://localhost:11434 | LLM server (if running) |

## 🖥️ Web Interface

CITADEL includes a Streamlit-based chat interface for easy interaction:

### Features

- **📁 Document Upload**: Drag & drop PDF/Markdown files
- **💬 Chat Interface**: Conversational Q&A with history
- **📖 Source Citations**: View retrieved chunks with relevance scores
- **⚙️ Settings**: Adjust number of context chunks (k)
- **🟢 Status Indicator**: Real-time API connection status
- **⚠️ Mock Mode Warning**: Visual alert when Ollama is unavailable

### Running the UI

```bash
# Via Docker (recommended)
docker compose up rag-ui

# Or locally
cd ui && streamlit run main.py
```

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         CITADEL RAG Pipeline                         │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────┐    ┌───────────────┐    ┌──────────┐    ┌──────────┐  │
│  │  Upload  │───▶│ FileProcessor │───▶│ Chunker  │───▶│ Embedder │  │
│  │  (API)   │    │  (PDF / MD)   │    │(LangChain)│   │ (MiniLM) │  │
│  └──────────┘    └───────────────┘    └──────────┘    └──────────┘  │
│        │                                                    │        │
│        ▼                                                    ▼        │
│  ┌─────────────┐                                  ┌──────────────┐  │
│  │  SHA-256    │                                  │   pgvector   │  │
│  │  Dedup Gate │                                  │   Storage    │  │
│  └─────────────┘                                  └──────────────┘  │
│                                                          │          │
│  ┌──────────┐    ┌───────────────┐    ┌──────────┐      │          │
│  │  Answer  │◀───│  LLM Service  │◀───│ Retriever│◀─────┘          │
│  │  (API)   │    │   (Ollama)    │    │(Semantic)│                  │
│  └──────────┘    └───────────────┘    └──────────┘                  │
│                         │                                            │
│                         ▼                                            │
│                  ┌─────────────┐                                     │
│                  │ Mock Mode   │  ← Fallback if Ollama unavailable  │
│                  │ (Graceful)  │                                     │
│                  └─────────────┘                                     │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    Streamlit Web UI                           │   │
│  │  [Upload] [Chat History] [Sources] [Settings]                 │   │
│  └──────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
```

**Pipeline Flow**:

1. **Ingest**: Upload PDF/Markdown → Extract text → Compute SHA-256 hash
2. **Chunk**: Split into overlapping segments (500 chars, 100 overlap)
3. **Embed**: Generate 384-dim vectors via MiniLM (local, no API calls)
4. **Store**: Persist to PostgreSQL with pgvector HNSW index
5. **Search**: Cosine similarity search on query embedding
6. **Generate**: LLM synthesizes answer from retrieved context

## 📡 API Endpoints

### CITADEL RAG API (Port 8001)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/api/v1/rag/ingest` | Upload file for ingestion (202 Accepted) |
| `POST` | `/api/v1/rag/search` | Semantic search across documents |
| `POST` | `/api/v1/rag/ask` | **Full RAG**: retrieve + generate answer |

### Atlas API (Port 8000) — Legacy

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/api/v1/notes/` | Create note (triggers async embedding) |
| `GET` | `/api/v1/notes/` | List notes |
| `POST` | `/api/v1/notes/search` | Semantic search |

### Example: Full RAG Query

```bash
# 1. Ingest a document
curl -X POST http://localhost:8001/api/v1/rag/ingest \
  -F "file=@document.pdf"

# 2. Ask a question
curl -X POST http://localhost:8001/api/v1/rag/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the main topic?", "k": 5}'

# Response:
{
  "answer": "Based on the context provided...",
  "sources": [
    {
      "filename": "document.pdf",
      "chunk_index": 0,
      "score": 0.85,
      "preview": "First 100 characters..."
    }
  ],
  "is_mocked": false,
  "query": "What is the main topic?"
}
```

> **Note**: If `is_mocked: true`, Ollama is not running. Retrieval still works,
> but the answer is a placeholder. Start Ollama for full LLM responses.

## ⚙️ Configuration

Environment variables loaded from `.env`:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `POSTGRES_USER` | Yes | - | Database username |
| `POSTGRES_PASSWORD` | Yes | - | Database password |
| `POSTGRES_HOST` | Yes | `db` (Docker) / `localhost` (local) | Database host |
| `POSTGRES_DB` | Yes | - | Database name |
| `OLLAMA_BASE_URL` | No | `http://host.docker.internal:11434` | Ollama API URL |
| `OLLAMA_MODEL` | No | `mistral` | LLM model name |
| `OLLAMA_TIMEOUT` | No | `30.0` | Request timeout (seconds) |
| `LOG_LEVEL` | No | `INFO` | Logging level |

## 🛠️ Development Commands

| Command | Description |
|---------|-------------|
| **Docker** | |
| `make up` | Start full Docker stack (all services) |
| `make deps` | Start only DB + Redis (for hybrid dev) |
| `make down` | Stop Docker stack |
| `make rebuild` | Rebuild and restart containers |
| `make logs` | Tail all container logs |
| `make logs-rag` | Tail RAG API logs |
| `make logs-ui` | Tail Streamlit UI logs |
| **Local Dev** | |
| `make run` | Run Atlas API locally (port 8000) |
| `make run-citadel` | Run CITADEL RAG API locally (port 8001) |
| `make run-ui` | Run Streamlit UI locally (port 8501) |
| **Testing** | |
| `make test` | Run all tests |
| `make test-live` | Run integration tests (requires Docker) |
| `make test-rag` | Run RAG integration tests |
| **Code Quality** | |
| `make lint` | Run Ruff + mypy |
| `make format` | Format code with Ruff |
| **Database** | |
| `make mig-up` | Apply pending migrations |
| `make mig-rev m="message"` | Generate new migration |
| `make db-shell` | Open psql shell |
| `make db-tables` | List database tables |

## 📁 Project Structure

```
citadel-rag/
├── app/                          # CITADEL RAG Pipeline
│   ├── api/v1/rag.py             # REST endpoints (/ingest, /search, /ask)
│   ├── core/
│   │   ├── config.py             # Ollama settings
│   │   └── database.py           # Async SQLAlchemy setup
│   ├── models/
│   │   ├── orm.py                # DocumentRecord, ChunkRecord (pgvector)
│   │   └── schemas.py            # Pydantic models for pipeline
│   ├── repositories/rag.py       # Data access + vector search
│   ├── schemas/rag.py            # API request/response DTOs
│   └── services/
│       ├── chunking.py           # LangChain text splitter
│       ├── ingestion.py          # PDF/Markdown extraction
│       ├── llm.py                # Ollama client + mock fallback
│       ├── rag_pipeline.py       # Orchestrator (ingest/search/ask)
│       └── vector.py             # MiniLM embedding service
├── ui/                           # Streamlit Frontend
│   ├── main.py                   # Chat interface application
│   ├── Dockerfile                # Frontend container
│   └── requirements.txt          # Streamlit dependencies
├── src/atlas_template/           # Legacy Atlas API
├── tests/
│   ├── test_ingestion.py         # Ingestion unit tests
│   ├── test_chunking.py          # Chunking unit tests
│   ├── unit/                     # Unit tests (mocked)
│   └── integration/              # Live tests (Docker required)
│       └── test_rag_flow.py      # Full RAG E2E tests
├── migrations/                   # Alembic migrations
├── docker-compose.yml            # Multi-service orchestration
├── Dockerfile                    # Backend container
├── Makefile                      # Development commands
└── pyproject.toml                # Python project config
```

## 🛡️ Graceful Degradation

CITADEL is designed to work even without all services running:

| Service | Status | Behavior |
|---------|--------|----------|
| PostgreSQL | ❌ Down | API fails to start (required) |
| Ollama | ❌ Down | `/ask` returns `is_mocked: true` with context preview |
| Ollama | ✅ Running | `/ask` returns full LLM-generated answers |

This ensures anyone can test the system without installing Ollama.

## ✅ Code Quality

Pre-commit hooks run automatically on every commit:

```bash
pre-commit install    # Setup (once)
pre-commit run --all  # Manual run
```

**Quality Tools**:
- **Ruff**: Linting and formatting (replaces flake8, isort, black)
- **mypy**: Static type checking (strict mode for `app/`)
- **pytest**: Unit and integration tests with coverage

## 🧪 Complete Test Workflow

### Before Committing

```bash
# 1. Format and lint
make format
make lint

# 2. Run unit tests (no Docker needed)
pytest tests/test_*.py tests/unit/ -v

# 3. Run pre-commit hooks
pre-commit run --all
```

### Full Integration Test

```bash
# 1. Start Ollama (Terminal 1)
ollama serve

# 2. Verify Ollama has the model
ollama list                           # Should show 'mistral'
ollama pull mistral                   # If not present

# 3. Start Docker stack (Terminal 2)
docker compose down                   # Clean slate
docker compose up -d --build
docker compose ps                     # All containers UP?

# 4. Run migrations
make mig-up

# 5. Health checks
curl http://localhost:8001/health     # {"status":"ok",...}
curl http://localhost:11434/api/tags  # {"models":[{"name":"mistral"...}]}

# 6. Open UI and test
open http://localhost:8501
# - Upload a PDF
# - Ask a question
# - Verify NO "Mock Mode" warning
```

### CI Pipeline (GitHub Actions)

The CI runs automatically on push/PR:

1. **unit-tests**: Fast tests without Docker
2. **quality**: Ruff + mypy + pre-commit
3. **integration-tests**: Full Docker stack (runs only if unit-tests pass)

## 🔧 Troubleshooting

### Port already in use

```bash
# Kill processes on CITADEL ports
kill -9 $(lsof -t -i :8000 -i :8001 -i :8501) 2>/dev/null
```

### Database connection refused

```bash
# Ensure PostgreSQL is running
docker compose up -d db
docker compose ps

# Check logs
docker compose logs db
```

### Migrations out of sync

```bash
# Reset database (WARNING: deletes data)
make down
docker volume rm citadel-rag_postgres_data
make up && make mig-up
```

### Ollama connection issues

```bash
# For Docker on macOS/Windows
# Set in .env: OLLAMA_BASE_URL=http://host.docker.internal:11434

# For Docker on Linux
# Set in .env: OLLAMA_BASE_URL=http://172.17.0.1:11434

# For local development
# Set in .env: OLLAMA_BASE_URL=http://localhost:11434
```

### Mock responses even with Ollama running

```bash
# Verify Ollama is accessible
curl http://localhost:11434/api/tags

# Check from Docker container
docker compose exec rag-api curl http://host.docker.internal:11434/api/tags
```

## 📄 License


## 👤 Author

**Youssef Chaouki** — AI/ML Engineer
