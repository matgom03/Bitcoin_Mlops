# tests/test_api.py
import numpy as np
import pytest
from fastapi.testclient import TestClient
from app.api import app

LAG = 30
N_FEATURES = 6
N_STEPS_FORECAST = 7


@pytest.fixture(scope="module")
def client():
    """Activa el lifespan (carga modelos) antes de los tests."""
    with TestClient(app) as c:
        yield c


def test_predict_endpoint(client):
    """Test endpoint /predict"""
    lags = np.random.rand(LAG * N_FEATURES).tolist()
    response = client.post(
        "/predict",
        json={
            "lags": lags,
            "lag_minutes": LAG
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], list)
    assert len(data["prediction"]) == N_STEPS_FORECAST


def test_predict_invalid_length(client):
    """Test input inválido"""
    lags = np.random.rand(10).tolist()
    response = client.post(
        "/predict",
        json={
            "lags": lags,
            "lag_minutes": 30
        }
    )
    assert response.status_code == 422


def test_health_endpoint(client):
    """Test health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()


def test_root_endpoint(client):
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "status" in response.json()