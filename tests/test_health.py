"""Smoke-тест веб-приложения (текущий web/app.py)."""
from __future__ import annotations

from fastapi.testclient import TestClient

from web.app import app

client = TestClient(app)


def test_status() -> None:
    r = client.get("/api/status")
    assert r.status_code == 200
    data = r.json()
    assert "data_loaded" in data


def test_index() -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert b"html" in r.content.lower() or b"DOCTYPE" in r.content
