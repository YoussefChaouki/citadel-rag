# ==============================================================================
# Atlas Backend - Development Commands
# Common tasks for local development, testing, and deployment
# ==============================================================================

.PHONY: install run test test-live lint format docker-build up down rebuild logs logs-api db-shell db-tables mig-up mig-rev

# --- Local Development ---

install:
	pip install -e ".[dev]"

run:
	uvicorn atlas_template.main:app --host 0.0.0.0 --port 8000 --reload

# --- Testing ---

test:
	pytest tests/ -v

test-live:
	# Requires Docker stack running (make up)
	pytest tests/integration -v

# --- Code Quality ---

lint:
	ruff check --fix .
	mypy src/

format:
	ruff format .

# --- Docker Operations ---

up:
	docker compose up -d

down:
	docker compose down

rebuild:
	# Full rebuild: stops containers, rebuilds images, restarts
	docker compose down
	docker compose up -d --build

logs:
	docker compose logs -f

logs-api:
	docker compose logs -f api

# --- Database Tools ---

db-shell:
	# Opens psql shell inside the database container
	docker compose exec db psql -U atlas -d atlas_db

db-tables:
	docker compose exec db psql -U atlas -d atlas_db -c "\dt"

# --- Migrations (Alembic) ---

mig-rev:
	# Usage: make mig-rev m="add users table"
	# POSTGRES_HOST=localhost required for host-to-container connection
	POSTGRES_HOST=localhost alembic revision --autogenerate -m "$(m)"

mig-up:
	POSTGRES_HOST=localhost alembic upgrade head
