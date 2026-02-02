"""
Pytest Configuration and Fixtures

Shared fixtures for integration tests requiring a running Docker stack.
Session-scoped fixtures ensure API readiness before test execution.
"""

import os

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Test environment defaults — MUST be before any atlas_template imports.
#
# 1. Load .env first so that Docker-matching credentials are available.
#    This mirrors what docker-compose does when reading the .env file.
# 2. setdefault fills in anything still missing (CI runners, fresh clones
#    without a .env file) so that pydantic Settings validation doesn't crash.
# ---------------------------------------------------------------------------
load_dotenv()  # .env → os.environ (no-op if file is missing)

_test_env = {
    "POSTGRES_USER": "atlas",
    "POSTGRES_PASSWORD": "atlas_password",
    "POSTGRES_HOST": "localhost",
    "POSTGRES_PORT": "5432",
    "POSTGRES_DB": "atlas_db",
    "OPENAI_API_KEY": "mock",
}
for _key, _value in _test_env.items():
    os.environ.setdefault(_key, _value)

# ---------------------------------------------------------------------------
# Imports (safe now that env vars are set)
# ---------------------------------------------------------------------------
import time  # noqa: E402
from collections.abc import Generator  # noqa: E402

import httpx  # noqa: E402
import pytest  # noqa: E402

BASE_URL = "http://localhost:8000"


@pytest.fixture(scope="session")
def wait_for_api():
    """
    Block until the API is ready or timeout expires.

    Polls /health endpoint with 1s intervals for up to 30s.
    Fails the test session if API is unreachable (Docker likely not running).

    Scope:
        session - runs once before all tests that depend on it.
    """
    url = f"{BASE_URL}/health"
    timeout = 30
    start = time.time()

    print("\n[Test] Waiting for API...")
    while time.time() - start < timeout:
        try:
            res = httpx.get(url, timeout=1.0)
            if res.status_code == 200:
                print("API Ready")
                return
        except httpx.RequestError:
            time.sleep(1)

    pytest.fail("API unreachable. Docker is likely down.")


@pytest.fixture(scope="session")
def api_client(wait_for_api) -> Generator[httpx.Client, None, None]:
    """
    Pre-configured HTTP client for integration tests.

    Depends on wait_for_api to ensure API is ready.
    Base URL points to /api/v1 for cleaner test assertions.

    Yields:
        httpx.Client: Session-scoped client, automatically closed after tests.
    """
    with httpx.Client(base_url=f"{BASE_URL}/api/v1", timeout=10.0) as client:
        yield client
