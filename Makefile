# ==============================================================================
# CITADEL RAG Pipeline — Developer Task Automation
# Usage: make <target>
# Run `make help` to see all available targets.
# ==============================================================================

.PHONY: help setup install run run-citadel run-ui \
        test test-unit test-live test-rag \
        lint format check check-full \
        up down deps rebuild logs logs-api logs-rag logs-ui \
        db-shell db-tables mig-up mig-rev \
        eval eval-verbose seed-eval seed-eval-clean \
        clean

# ==============================================================================
# Help
# ==============================================================================

help: ## Show this help message
	@echo "🏰 CITADEL — Available Targets"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ==============================================================================
# Setup & Installation
# ==============================================================================

setup: install ## Full dev setup: install deps + pre-commit hooks
	pre-commit install
	@echo "✅ Dev environment ready"

install: ## Install project in editable mode with dev dependencies
	pip install -e ".[dev]"

# ==============================================================================
# Local Development (no Docker required for API)
# ==============================================================================

run: ## Start legacy Atlas API on :8000 (hot-reload)
	uvicorn atlas_template.main:app --host 0.0.0.0 --port 8000 --reload

run-citadel: ## Start CITADEL RAG API on :8001 (hot-reload)
	uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload

run-ui: ## Start Streamlit UI on :8501
	cd ui && streamlit run main.py --server.port 8501

# ==============================================================================
# Testing
# ==============================================================================

test: ## Run all tests (unit + integration markers)
	pytest tests/ -v

test-unit: ## Run unit tests only (no Docker needed)
	pytest tests/test_*.py tests/unit/ -v --tb=short

test-live: ## Run integration tests (requires: make up)
	pytest tests/integration/ -v

test-rag: ## Run RAG-specific integration tests (requires: make up)
	pytest tests/integration/test_rag_flow.py -v

test-cov: ## Run tests with coverage report
	pytest tests/ -v --cov=app --cov=src --cov-report=term-missing --cov-report=html

# ==============================================================================
# Code Quality
# ==============================================================================

lint: ## Run linter (ruff) + type checker (mypy)
	ruff check --fix .
	mypy src/
	mypy app/ --strict

format: ## Auto-format code with ruff
	ruff format .

check: ## Quick pre-push validation (format + lint + unit tests)
	@echo "━━━ Formatting ━━━"
	ruff format --check .
	@echo "━━━ Linting ━━━"
	ruff check .
	@echo "━━━ Type Checking ━━━"
	mypy src/
	mypy app/ --strict
	@echo "━━━ Unit Tests ━━━"
	pytest tests/test_*.py tests/unit/ -v --tb=short
	@echo ""
	@echo "✅ All checks passed"

check-full: check ## Full validation including integration tests
	@echo "━━━ Integration Tests ━━━"
	pytest tests/integration/ -v --tb=short
	@echo ""
	@echo "✅ Full validation passed"

# ==============================================================================
# Docker Operations
# ==============================================================================

deps: ## Start dependencies only (DB + Redis) for local dev
	docker compose up -d db redis
	@echo "✅ DB + Redis ready → run: make run-citadel"

up: ## Start full Docker stack (API + UI + DB + Redis)
	docker compose up -d
	@echo ""
	@echo "✅ CITADEL is running:"
	@echo "   UI:   http://localhost:8501"
	@echo "   API:  http://localhost:8001"
	@echo "   Docs: http://localhost:8001/docs"

down: ## Stop all Docker services
	docker compose down

rebuild: ## Rebuild and restart all Docker services
	docker compose down
	docker compose up -d --build

logs: ## Tail all Docker service logs
	docker compose logs -f

logs-api: ## Tail legacy Atlas API logs
	docker compose logs -f api

logs-rag: ## Tail CITADEL RAG API logs
	docker compose logs -f rag-api

logs-ui: ## Tail Streamlit UI logs
	docker compose logs -f rag-ui

# ==============================================================================
# Database
# ==============================================================================

db-shell: ## Open psql shell in the database container
	docker compose exec db psql -U atlas -d atlas_db

db-tables: ## List all database tables
	docker compose exec db psql -U atlas -d atlas_db -c "\dt"

mig-up: ## Apply all pending Alembic migrations
	POSTGRES_HOST=localhost alembic upgrade head

mig-rev: ## Create new migration (usage: make mig-rev m="description")
	POSTGRES_HOST=localhost alembic revision --autogenerate -m "$(m)"

# ==============================================================================
# Evaluation
# ==============================================================================

seed-eval: ## Ingest sample ML documents for evaluation
	python scripts/seed_eval_docs.py

seed-eval-clean: ## Clean and re-ingest evaluation documents
	python scripts/seed_eval_docs.py --clean

eval: ## Run RAG evaluation harness (requires: make up + make seed-eval)
	python scripts/evaluate_rag.py

eval-verbose: ## Run evaluation with k=10 and verbose output
	python scripts/evaluate_rag.py --k 10

# ==============================================================================
# Cleanup
# ==============================================================================

clean: ## Remove build artifacts, caches, and temp files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf dist/ build/ htmlcov/ .coverage
	rm -f evaluation_results_*.json evaluation_report_*.md
	@echo "✅ Cleaned"
