# ==============================================================================
# Atlas Backend - Production Dockerfile
# Multi-stage optimized build for FastAPI + async SQLAlchemy application
# ==============================================================================

# Base image: Python 3.11 slim for minimal footprint (~150MB vs ~900MB full)
FROM python:3.11-slim

# Python environment configuration
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PYTHONPATH=/app/src

WORKDIR /app

# Copy dependency manifest first (enables Docker layer caching)
COPY pyproject.toml .

# Install dependencies before copying source code (cache optimization)
# Heavy deps installed explicitly to maximize layer reuse across builds
RUN pip install --upgrade pip && \
    pip install fastapi uvicorn[standard] sqlalchemy asyncpg alembic pydantic pydantic-settings openai pgvector

COPY src/ src/

# Install package in editable mode to resolve 'atlas_template' imports
RUN pip install .

# Security: Run as non-root user in production
RUN useradd -m atlas
USER atlas

EXPOSE 8000

# Production entrypoint (override with --reload for development)
CMD ["uvicorn", "atlas_template.main:app", "--host", "0.0.0.0", "--port", "8000"]
