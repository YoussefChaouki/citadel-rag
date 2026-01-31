# Atlas Backend Template

FastAPI backend template with async SQLAlchemy, PostgreSQL vector search (pgvector),
and background task processing for AI embeddings.

## Tech Stack

- **Framework**: FastAPI 0.109+ with Pydantic v2
- **Database**: PostgreSQL 16 with pgvector extension
- **ORM**: SQLAlchemy 2.0 (async) + Alembic migrations
- **Cache**: Redis (prepared, not yet implemented)
- **AI**: OpenAI embeddings (text-embedding-3-small) with mock mode for development
- **Testing**: pytest + pytest-asyncio + httpx
- **Code Quality**: Ruff (linter/formatter) + mypy + pre-commit

## Quickstart

**Prerequisites**: Docker, Python 3.11+, Make

```bash
# 1. Clone and configure
cp .env.example .env
# Edit .env if needed (OPENAI_API_KEY optional - mock mode works without it)

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

## Code Quality

Pre-commit hooks run automatically on commit:

```bash
pre-commit install    # Setup (once)
pre-commit run --all  # Manual run
```

Quality tools:
- **Ruff**: Linting and formatting (replaces flake8, isort, black)
- **mypy**: Static type checking
- **pytest**: Unit and integration tests with coverage

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/api/v1/notes/` | Create note (triggers async embedding) |
| `GET` | `/api/v1/notes/` | List notes |
| `GET` | `/api/v1/notes/{id}` | Get note by ID |
| `POST` | `/api/v1/notes/search` | Semantic search |

## Troubleshooting

**Database connection refused**
```bash
# Ensure Docker is running and database is healthy
docker compose ps
docker compose logs db
```

**Migrations out of sync**
```bash
# Reset and re-apply migrations (dev only - destroys data)
make down
docker volume rm atlas-backend-template_postgres_data
make up
make mig-up
```

**API health check fails**
```bash
# Check API logs for startup errors
make logs-api
# Verify database is accessible
make db-shell
```

## Project Structure

```
src/atlas_template/
    api/v1/          # Route handlers
    core/            # Config, database, logging
    models/          # SQLAlchemy ORM models
    repositories/    # Data access layer
    schemas/         # Pydantic request/response models
    services/        # Business logic (AI, embeddings)
tests/
    unit/            # Unit tests (mocked dependencies)
    integration/     # Live tests (require Docker)
migrations/          # Alembic migration files
scripts/             # Utility scripts (backfill, etc.)
```
