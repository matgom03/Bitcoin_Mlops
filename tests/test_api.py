from fastapi.testclient import TestClient
from app.api import app

import numpy as np

client = TestClient(app)

LAG = 15
N_FEATURES = 6
N_STEPS_FORECAST = 7


def test_predict_endpoint():
    """Test endpoint /predict"""

    # Generar input válido (15 × 6 = 90 valores)
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


def test_predict_invalid_length():
    """Test input inválido"""

    # tamaño incorrecto
    lags = np.random.rand(10).tolist()

    response = client.post(
        "/predict",
        json={
            "lags": lags,
            "lag_minutes": 15
        }
    )

    assert response.status_code == 422


def test_health_endpoint():
    """Test health endpoint"""

    response = client.get("/health")

    assert response.status_code == 200
    assert "status" in response.json()


def test_root_endpoint():
    """Test root endpoint"""

    response = client.get("/")

    assert response.status_code == 200
    assert "status" in response.json()