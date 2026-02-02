# ==============================================================================
# CITADEL RAG Pipeline - Production Dockerfile
# Multi-stage optimized build for FastAPI + async SQLAlchemy application
# Includes: Atlas API (legacy) + CITADEL RAG API (new)
# ==============================================================================

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PYTHONPATH=/app/src:/app

WORKDIR /app

# Copy dependency manifest first (Docker layer caching)
COPY pyproject.toml .

# Install dependencies
# 1. CPU-only PyTorch (saves ~1.5GB vs full CUDA build)
# 2. Core FastAPI + DB stack
# 3. RAG pipeline dependencies (sentence-transformers, langchain, pymupdf)
RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install \
        fastapi "uvicorn[standard]" sqlalchemy asyncpg alembic \
        pydantic pydantic-settings openai pgvector pymupdf \
        langchain-text-splitters sentence-transformers

# Copy source code
COPY src/ src/
COPY app/ app/

# Install atlas_template package (editable mode for import resolution)
RUN pip install .

# Pre-download embedding model (cached in image layer, ~90MB)
# Avoids cold-start download on first request
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Security: Run as non-root user in production
RUN useradd -m atlas
USER atlas

EXPOSE 8000

# Default: Atlas legacy API. Override in docker-compose for CITADEL.
CMD ["uvicorn", "atlas_template.main:app", "--host", "0.0.0.0", "--port", "8000"]
