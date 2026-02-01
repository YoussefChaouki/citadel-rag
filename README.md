# CITADEL — High-Performance RAG Pipeline

Modular Retrieval-Augmented Generation backend built on FastAPI, PostgreSQL with
pgvector, and async-first Python. CITADEL handles the full document lifecycle:
ingestion, chunking, embedding, and semantic retrieval.

## Tech Stack

- **Framework**: FastAPI 0.109+ with Pydantic v2
- **Database**: PostgreSQL 16 with pgvector extension
- **ORM**: SQLAlchemy 2.0 (async) + Alembic migrations
- **Ingestion**: PyMuPDF (PDF), standard lib (Markdown)
- **Embeddings**: OpenAI text-embedding-3-small (mock mode for dev)
- **Cache**: Redis (prepared, not yet active)
- **Testing**: pytest + pytest-asyncio + httpx
- **Code Quality**: Ruff (linter/formatter) + mypy (strict) + pre-commit

## Quickstart

**Prerequisites**: Docker, Python 3.11+, Make

```bash
# 1. Clone and configure
cp .env.example .env
# Edit .env (OPENAI_API_KEY optional — mock mode works without it)

# 2. Install dependencies
make install
pre-commit install

# 3. Start Docker stack (PostgreSQL + Redis)
make up

# 4. Run database migrations
make mig-up

# 5. Verify setup
curl http://localhost:8000/health
```

## Architecture

```
┌──────────┐    ┌───────────────┐    ┌──────────┐    ┌──────────┐
│  Upload  │───▶│ FileProcessor │───▶│ Chunker  │───▶│ Embedder │
│  (API)   │    │  (Ingestion)  │    │ (planned)│    │ (OpenAI) │
└──────────┘    └───────────────┘    └──────────┘    └──────────┘
                       │                                    │
                       ▼                                    ▼
                ┌─────────────┐                    ┌──────────────┐
                │  SHA-256    │                    │   pgvector   │
                │  Dedup Gate │                    │   Storage    │
                └─────────────┘                    └──────────────┘
```

**Ingestion pipeline** (this release):
1. Accept PDF or Markdown files via API or direct path.
2. Compute SHA-256 hash for idempotent processing — skip duplicates.
3. Extract text content (PyMuPDF for PDFs, UTF-8 decode for Markdown).
4. Return a structured `Document` with metadata (filename, size, page count).

## Development Commands

| Command | Description |
|---------|-------------|
| `make up` | Start Docker stack (detached) |
| `make down` | Stop Docker stack |
| `make rebuild` | Rebuild and restart containers |
| `make logs` | Tail all container logs |
| `make logs-api` | Tail API container logs |
| `make run` | Run API locally (outside Docker) |
| `make test` | Run all tests |
| `make test-live` | Run integration tests (requires Docker) |
| `make lint` | Run Ruff + mypy |
| `make format` | Format code with Ruff |
| `make mig-up` | Apply pending migrations |
| `make mig-rev m="message"` | Generate new migration |
| `make db-shell` | Open psql shell in database container |

## Configuration

Environment variables are loaded from `.env`. See `.env.example` for all options.

| Variable | Required | Description |
|----------|----------|-------------|
| `POSTGRES_USER` | Yes | Database username |
| `POSTGRES_PASSWORD` | Yes | Database password |
| `POSTGRES_HOST` | Yes | Database host (`db` in Docker, `localhost` for local) |
| `POSTGRES_DB` | Yes | Database name |
| `OPENAI_API_KEY` | No | OpenAI API key (leave empty or set to `mock` for dev) |
| `LOG_LEVEL` | No | Logging level (default: `INFO`) |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/api/v1/notes/` | Create note (triggers async embedding) |
| `GET` | `/api/v1/notes/` | List notes |
| `GET` | `/api/v1/notes/{id}` | Get note by ID |
| `POST` | `/api/v1/notes/search` | Semantic search |

## Project Structure

```
app/
├── models/
│   └── schemas.py        # Document, Chunk, DocumentMetadata (Pydantic)
├── services/
│   └── ingestion.py      # FileProcessor — PDF & Markdown extraction
src/atlas_template/
├── api/v1/               # Route handlers
├── core/                 # Config, database, logging
├── models/               # SQLAlchemy ORM models
├── repositories/         # Data access layer
├── schemas/              # Pydantic request/response models
├── services/             # Business logic (AI, embeddings)
tests/
├── test_ingestion.py     # Ingestion unit tests
├── unit/                 # Unit tests (mocked dependencies)
├── integration/          # Live tests (require Docker)
migrations/               # Alembic migration files
scripts/                  # Utility scripts
```

> **Note**: `app/` contains the new CITADEL ingestion layer. The existing
> `src/atlas_template/` code (notes API, vector search) will be migrated
> into `app/` in an upcoming session.

## Code Quality

Pre-commit hooks run automatically on commit:

```bash
pre-commit install    # Setup (once)
pre-commit run --all  # Manual run
```

Quality tools:
- **Ruff**: Linting and formatting (replaces flake8, isort, black)
- **mypy**: Static type checking (strict mode for `app/`)
- **pytest**: Unit and integration tests with coverage

## Troubleshooting

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

**API health check fails**
```bash
make logs-api
make db-shell
```
