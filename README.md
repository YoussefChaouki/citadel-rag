# CITADEL — High-Performance RAG Pipeline

Modular Retrieval-Augmented Generation backend built on FastAPI, PostgreSQL with
pgvector, and async-first Python. CITADEL handles the full document lifecycle:
ingestion, chunking, embedding, and semantic retrieval with LLM-powered answers.

## Tech Stack

- **Framework**: FastAPI 0.109+ with Pydantic v2
- **Database**: PostgreSQL 16 with pgvector extension
- **ORM**: SQLAlchemy 2.0 (async) + Alembic migrations
- **Ingestion**: PyMuPDF (PDF), standard lib (Markdown)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2, local)
- **LLM**: Ollama (Mistral 7B, local) with graceful fallback
- **Cache**: Redis (prepared, not yet active)
- **Testing**: pytest + pytest-asyncio + httpx
- **Code Quality**: Ruff (linter/formatter) + mypy (strict) + pre-commit

## Quickstart

**Prerequisites**: Docker, Python 3.11+, Make
```bash
# 1. Clone and configure
cp .env.example .env
# Edit .env if needed (defaults work out of the box)

# 2. Install dependencies
make install
pre-commit install

# 3. Start Docker stack (PostgreSQL + Redis + APIs)
make up

# 4. Run database migrations
make mig-up

# 5. Verify setup
curl http://localhost:8000/health  # Atlas API
curl http://localhost:8001/health  # CITADEL RAG API
```

### Optional: Enable Full LLM Responses

Without Ollama, the `/ask` endpoint returns mock responses (graceful degradation).
To enable full AI-generated answers:
```bash
# Install Ollama (macOS)
brew install ollama

# Start Ollama server
ollama serve

# Download Mistral model (~4GB)
ollama pull mistral

# Verify
curl http://localhost:11434/api/tags
```

## Architecture
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
└──────────────────────────────────────────────────────────────────────┘
```

**Pipeline flow**:
1. **Ingest**: Upload PDF/Markdown → Extract text → Compute SHA-256 hash
2. **Chunk**: Split into overlapping segments (500 chars, 100 overlap)
3. **Embed**: Generate 384-dim vectors via MiniLM (local, no API calls)
4. **Store**: Persist to PostgreSQL with pgvector HNSW index
5. **Search**: Cosine similarity search on query embedding
6. **Generate**: LLM synthesizes answer from retrieved context

## API Endpoints

### Atlas API (Port 8000)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/api/v1/notes/` | Create note (triggers async embedding) |
| `GET` | `/api/v1/notes/` | List notes |
| `GET` | `/api/v1/notes/{id}` | Get note by ID |
| `POST` | `/api/v1/notes/search` | Semantic search |

### CITADEL RAG API (Port 8001)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/api/v1/rag/ingest` | Upload file for ingestion (202 Accepted) |
| `POST` | `/api/v1/rag/search` | Semantic search across documents |
| `POST` | `/api/v1/rag/ask` | **Full RAG**: retrieve + generate answer |

### Example: Full RAG Query
```bash
# 1. Ingest a document
curl -X POST http://localhost:8001/api/v1/rag/ingest \
  -F "file=@document.pdf"

# 2. Ask a question
curl -X POST http://localhost:8001/api/v1/rag/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is quantum computing?", "k": 5}'

# Response:
{
  "answer": "Based on the context, quantum computing uses qubits...",
  "sources": [
    {"filename": "quantum.pdf", "chunk_index": 0, "score": 0.85, "preview": "..."}
  ],
  "is_mocked": false,
  "query": "What is quantum computing?"
}
```

> **Note**: If `is_mocked: true`, Ollama is not running. The retrieval still works,
> but the answer is a placeholder. Start Ollama for full responses.

## Configuration

Environment variables loaded from `.env`:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `POSTGRES_USER` | Yes | - | Database username |
| `POSTGRES_PASSWORD` | Yes | - | Database password |
| `POSTGRES_HOST` | Yes | `db` | Database host |
| `POSTGRES_DB` | Yes | - | Database name |
| `OPENAI_API_KEY` | No | `mock` | OpenAI key (legacy, not used by CITADEL) |
| `OLLAMA_BASE_URL` | No | `http://host.docker.internal:11434` | Ollama API URL |
| `OLLAMA_MODEL` | No | `mistral` | LLM model name |
| `OLLAMA_TIMEOUT` | No | `30.0` | Request timeout (seconds) |
| `LOG_LEVEL` | No | `INFO` | Logging level |

## Development Commands

| Command | Description |
|---------|-------------|
| `make up` | Start Docker stack (detached) |
| `make down` | Stop Docker stack |
| `make rebuild` | Rebuild and restart containers |
| `make logs` | Tail all container logs |
| `make logs-rag` | Tail RAG API logs |
| `make run` | Run Atlas API locally |
| `make run-citadel` | Run CITADEL RAG API locally |
| `make test` | Run all tests |
| `make test-live` | Run integration tests (requires Docker) |
| `make test-rag` | Run RAG integration tests |
| `make lint` | Run Ruff + mypy |
| `make format` | Format code with Ruff |
| `make mig-up` | Apply pending migrations |
| `make mig-rev m="message"` | Generate new migration |
| `make db-shell` | Open psql shell |

## Project Structure
```
app/                          # CITADEL RAG Pipeline
├── api/v1/rag.py             # REST endpoints (/ingest, /search, /ask)
├── core/
│   ├── config.py             # Ollama settings
│   └── database.py           # Async SQLAlchemy setup
├── models/
│   ├── orm.py                # DocumentRecord, ChunkRecord (pgvector)
│   └── schemas.py            # Pydantic models for pipeline
├── repositories/rag.py       # Data access + vector search
├── schemas/rag.py            # API request/response DTOs
├── services/
│   ├── chunking.py           # LangChain text splitter
│   ├── ingestion.py          # PDF/Markdown extraction
│   ├── llm.py                # Ollama client + mock fallback
│   ├── rag_pipeline.py       # Orchestrator (ingest/search/ask)
│   └── vector.py             # MiniLM embedding service
src/atlas_template/           # Legacy Atlas API
tests/
├── test_ingestion.py         # Ingestion unit tests
├── test_chunking.py          # Chunking unit tests
├── unit/                     # Unit tests (mocked)
├── integration/              # Live tests (Docker required)
│   └── test_rag_flow.py      # Full RAG E2E tests
migrations/                   # Alembic migrations
```

## Graceful Degradation

CITADEL is designed to work even without all services running:

| Service | Status | Behavior |
|---------|--------|----------|
| PostgreSQL | ❌ Down | API fails to start (required) |
| Ollama | ❌ Down | `/ask` returns `is_mocked: true` with context preview |
| Ollama | ✅ Running | `/ask` returns full LLM-generated answers |

This ensures a recruiter can test the system without installing Ollama.

## Code Quality

Pre-commit hooks run automatically:
```bash
pre-commit install    # Setup (once)
pre-commit run --all  # Manual run
```

Quality tools:
- **Ruff**: Linting and formatting (replaces flake8, isort, black)
- **mypy**: Static type checking (strict mode for `app/`)
- **pytest**: Unit and integration tests with coverage

## Troubleshooting

**Ollama connection refused (inside Docker)**
```bash
# Check OLLAMA_BASE_URL in .env
# For Docker on macOS/Windows: http://host.docker.internal:11434
# For Docker on Linux: http://172.17.0.1:11434
```

**Mock responses even with Ollama running**
```bash
# Verify Ollama is accessible from Docker
docker compose exec rag-api curl http://host.docker.internal:11434/api/tags
```

**Database connection refused**
```bash
docker compose ps
docker compose logs db
```

**Migrations out of sync**
```bash
make down
docker volume rm citadel-rag_postgres_data
make up && make mig-up
```
