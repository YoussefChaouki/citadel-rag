# ==============================================================================
# CITADEL RAG Pipeline - Development Commands
# Common tasks for local development, testing, and deployment
# ==============================================================================

.PHONY: install run run-citadel run-ui test test-live test-rag lint format \
        docker-build up down rebuild logs logs-api logs-rag logs-ui \
        db-shell db-tables mig-up mig-rev eval seed-eval check check-full

# --- Local Development ---

install:
	pip install -e ".[dev]"

run:
	uvicorn atlas_template.main:app --host 0.0.0.0 --port 8000 --reload

run-citadel:
	uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload

run-ui:
	cd ui && streamlit run main.py --server.port 8501

# --- Testing ---

test:
	pytest tests/ -v

test-live:
	# Requires Docker stack running (make up)
	pytest tests/integration -v

test-rag:
	# Requires Docker stack with rag-api running
	pytest tests/integration/test_rag_flow.py -v

# --- Evaluation ---

eval:
	# RAG Retrieval Evaluation (requires API running)
	python scripts/evaluate_rag.py

eval-verbose:
	# Evaluation with k=10 results
	python scripts/evaluate_rag.py --k 10

seed-eval:
	# Seed ML sample documents for evaluation
	python scripts/seed_eval_docs.py

seed-eval-clean:
	# Clean and reseed evaluation documents
	python scripts/seed_eval_docs.py --clean

# --- Pre-Push Validation ---

check:
	# Quick validation (format, lint, unit tests)
	@chmod +x scripts/pre-push-check.sh
	@./scripts/pre-push-check.sh

check-full:
	# Full validation (includes integration tests)
	@chmod +x scripts/pre-push-check.sh
	@./scripts/pre-push-check.sh --full

# --- Code Quality ---

lint:
	ruff check --fix .
	mypy src/
	mypy app/ --strict

format:
	ruff format .

# --- Docker Operations ---

# Start ONLY dependencies (DB + Redis) - for local development
deps:
	docker compose up -d db redis
	@echo "✅ DB + Redis ready. Now run: make run-citadel"

# Start FULL Docker stack (all services)
up:
	docker compose up -d
	@echo "✅ Full stack ready:"
	@echo "   - UI:  http://localhost:8501"
	@echo "   - API: http://localhost:8001"

down:
	docker compose down

rebuild:
	docker compose down
	docker compose up -d --build

logs:
	docker compose logs -f

logs-api:
	docker compose logs -f api

logs-rag:
	docker compose logs -f rag-api

logs-ui:
	docker compose logs -f rag-ui

# --- Database Tools ---

db-shell:
	docker compose exec db psql -U atlas -d atlas_db

db-tables:
	docker compose exec db psql -U atlas -d atlas_db -c "\dt"

# --- Migrations (Alembic) ---

mig-rev:
	# Usage: make mig-rev m="add users table"
	POSTGRES_HOST=localhost alembic revision --autogenerate -m "$(m)"

mig-up:
	POSTGRES_HOST=localhost alembic upgrade head
