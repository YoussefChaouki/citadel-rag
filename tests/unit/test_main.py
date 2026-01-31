"""
Main Application Unit Tests

Tests for application startup and health endpoints with mocked infrastructure.
Runs without Docker - uses mocks for database and Redis connections.
"""

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from atlas_template.main import app


def test_health_check():
    """
    Verify /health endpoint returns correct response structure.

    Mocks infrastructure checks to isolate from Docker dependencies.
    TestClient triggers the lifespan handler, so DB/Redis checks must be mocked.
    """
    with (
        patch("atlas_template.main.wait_for_db", new_callable=AsyncMock) as mock_db,
        patch("atlas_template.main.check_redis", new_callable=AsyncMock) as mock_redis,
    ):
        mock_db.return_value = True
        mock_redis.return_value = True

        with TestClient(app) as client:
            response = client.get("/health")

            assert response.status_code == 200
            data = response.json()

            assert data["status"] == "ok"
            assert data["service"] == "atlas-template"

            # Verify infrastructure keys exist (values are static, not live checks)
            assert "db" in data
            assert "redis" in data
            assert "environment" in data
