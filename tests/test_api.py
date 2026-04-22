# tests/test_api.py
import numpy as np
import pytest
from fastapi.testclient import TestClient
from app.api import app
N_FEATURES = 6
N_STEPS_FORECAST = 7


@pytest.fixture(scope="module")
def client():
    """Activa el lifespan (carga modelos) antes de los tests."""
    with TestClient(app) as c:
        yield c


# Reemplaza la generación de lags en test_predict_endpoint
def test_predict_endpoint(client):
    """Test con valores en escala y orden correctos"""
    lag = 15
    lags = []
    for _ in range(lag):
        lags.extend([
            0.52,      # Volatility_daily
           -0.65,      # log_Volatility_daily
            0.00031,   # log_ret
            0.00142,   # hl_range
           12.847,     # log_volume
           -0.00018,   # ret_lag1min
        ])

    response = client.post(
        "/predict",
        json={"lags": lags, "lag_minutes": lag}
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["prediction"]) == 7
    assert all(isinstance(v, float) for v in body["prediction"])
    assert all(v >= 0 for v in body["prediction"])

def test_predict_invalid_length(client):
    """Test input inválido"""
    lags = np.random.rand(10).tolist()
    response = client.post(
        "/predict",
        json={
            "lags": lags,
            "lag_minutes": 15
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