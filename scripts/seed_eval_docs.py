#!/usr/bin/env python3
"""
Seed ML Documents for RAG Evaluation

Ingests the sample ML documents into CITADEL for evaluation testing.

Usage:
    python scripts/seed_eval_docs.py
    python scripts/seed_eval_docs.py --clean  # Delete all docs first
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx

DEFAULT_API_URL = "http://localhost:8001"
DOCS_DIR = Path("tests/data/sample_docs")
TIMEOUT = 60.0


def log_info(msg: str) -> None:
    print(f"ℹ {msg}")


def log_success(msg: str) -> None:
    print(f"✓ {msg}")


def log_error(msg: str) -> None:
    print(f"✗ {msg}")


def check_api(api_url: str) -> bool:
    try:
        r = httpx.get(f"{api_url}/health", timeout=5.0)
        return r.status_code == 200
    except httpx.RequestError:
        return False


def delete_all_documents(api_url: str) -> bool:
    """Delete all existing documents."""
    try:
        r = httpx.delete(
            f"{api_url}/api/v1/rag/documents",
            params={"confirm": "true"},
            timeout=TIMEOUT,
        )
        if r.status_code == 200:
            data = r.json()
            log_success(f"Deleted {data.get('deleted_count', 0)} documents")
            return True
        elif r.status_code == 404:
            log_info("Delete endpoint not available (skipping)")
            return True  # Not a failure, just not implemented
        else:
            log_error(f"Delete failed: {r.status_code}")
            return False
    except httpx.RequestError as e:
        log_error(f"Delete failed: {e}")
        return False


def ingest_document(api_url: str, filepath: Path) -> bool:
    """Ingest a single document."""
    try:
        with filepath.open("rb") as f:
            files = {"file": (filepath.name, f, "text/markdown")}
            r = httpx.post(
                f"{api_url}/api/v1/rag/ingest",
                files=files,
                timeout=TIMEOUT,
            )

        # 200 = sync success, 202 = accepted for async processing
        if r.status_code in (200, 202):
            data = r.json()
            chunks = data.get("chunks_created") or data.get("chunks_count") or 0
            status = data.get("status", "ok")
            if status == "processing":
                log_success(f"Queued {filepath.name} for processing")
            else:
                log_success(f"Ingested {filepath.name} ({chunks} chunks)")
            return True
        else:
            log_error(f"Failed {filepath.name}: {r.status_code} - {r.text}")
            return False

    except httpx.RequestError as e:
        log_error(f"Failed {filepath.name}: {e}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed ML documents for evaluation")
    parser.add_argument(
        "--clean", action="store_true", help="Delete all documents first"
    )
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="API base URL")
    parser.add_argument(
        "--wait", type=int, default=5, help="Seconds to wait for async processing"
    )

    args = parser.parse_args()
    api_url = args.api_url

    print("\n🏰 CITADEL Document Seeder\n")

    # Check API
    if not check_api(api_url):
        log_error(f"API not available at {api_url}")
        return 1
    log_success("API connected")

    # Clean if requested
    if args.clean:
        log_info("Cleaning existing documents...")
        delete_all_documents(api_url)

    # Find documents
    if not DOCS_DIR.exists():
        log_error(f"Documents directory not found: {DOCS_DIR}")
        return 1

    docs = list(DOCS_DIR.glob("*.md"))
    if not docs:
        log_error("No markdown documents found")
        return 1

    log_info(f"Found {len(docs)} documents to ingest")
    print()

    # Ingest each
    success = 0
    for doc in sorted(docs):
        if ingest_document(api_url, doc):
            success += 1

    print()

    if success == len(docs):
        log_success(f"Ingested {success}/{len(docs)} documents")
        if args.wait > 0:
            import time

            log_info(f"Waiting {args.wait}s for async processing...")
            time.sleep(args.wait)
            log_success("Ready for evaluation!")
        return 0
    else:
        log_error(f"Ingested {success}/{len(docs)} documents")
        return 1


if __name__ == "__main__":
    sys.exit(main())
