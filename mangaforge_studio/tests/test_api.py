"""
Smoke test da API REST usando o TestClient do FastAPI + MockPipeline
(sem GPU). Garante que os endpoints principais respondem e que o
Hardware Manager é exposto em /hardware.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from mangaforge_studio.api.main import app

client = TestClient(app)


def test_dashboard_served_at_root() -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "MangaForge Studio" in resp.text


def test_health() -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["pipeline"] == "mock"


def test_hardware_endpoint_exposes_selected_profile() -> None:
    resp = client.get("/hardware")
    assert resp.status_code == 200
    body = resp.json()
    assert "selected_profile" in body
    assert body["selected_profile"]["name"] in {"8gb", "12gb", "16gb", "24gb", "48gb"}


def test_create_character_and_build_chapter() -> None:
    resp = client.post(
        "/characters",
        json={"name": "Kaito", "description": "young ronin, spiky black hair"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] in {"draft", "ready"}

    resp = client.post(
        "/chapters/build",
        json={
            "title": "Cap 1",
            "beats": [{"scene_description": "rainy village", "panel_prompts": ["wide shot"]}],
            "panels_per_page": 2,
        },
    )
    assert resp.status_code == 200
    assert len(resp.json()["page_ids"]) == 1
