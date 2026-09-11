import pytest
from fastapi.testclient import TestClient
from backend.api.app import app

client = TestClient(app)

def test_health_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "database_active_mode" in data

def test_cameras_endpoint():
    res = client.get("/cameras")
    assert res.status_code == 200
    data = res.json()
    assert "cameras" in data

def test_alerts_endpoint():
    res = client.get("/alerts")
    assert res.status_code == 200
    data = res.json()
    assert "alerts" in data

def test_coverage_endpoint():
    res = client.get("/coverage")
    assert res.status_code == 200
    data = res.json()
    assert "coverage_tier_labels" in data
