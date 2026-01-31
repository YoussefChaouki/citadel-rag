"""
Live API Integration Tests

Tests against a running Docker stack (make up).
Marked with @pytest.mark.live for selective execution.

Run with: pytest tests/integration/test_live_api.py -m live
"""

import pytest


@pytest.mark.live
def test_lifecycle_create_read(api_client):
    """
    Test full CRUD lifecycle: Create -> List -> Get.

    Verifies the happy path for note creation and retrieval.
    """
    payload = {
        "title": "Live Test",
        "content": "Running inside Docker",
        "is_active": True,
    }
    res_post = api_client.post("/notes/", json=payload)
    assert res_post.status_code == 201
    data = res_post.json()
    assert data["title"] == payload["title"]
    note_id = data["id"]

    res_list = api_client.get("/notes/")
    assert res_list.status_code == 200
    assert len(res_list.json()) >= 1  # At least the note we just created

    res_get = api_client.get(f"/notes/{note_id}")
    assert res_get.status_code == 200
    assert res_get.json()["id"] == note_id


@pytest.mark.live
def test_not_found(api_client):
    """Verify 404 response for non-existent note ID."""
    res = api_client.get("/notes/999999999")
    assert res.status_code == 404


@pytest.mark.live
def test_validation_error(api_client):
    """Verify 422 response when payload fails Pydantic validation (title < 3 chars)."""
    payload = {"title": "No", "content": "Valid Content"}
    res = api_client.post("/notes/", json=payload)
    assert res.status_code == 422
