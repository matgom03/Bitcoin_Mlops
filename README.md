# Bitcoin Volatility Predictor — MLOps

Predice la volatilidad de Bitcoin a 7 minutos hacia adelante usando un MLP
entrenado con velas de 1 minuto (2023–2026).

## Arquitectura

datos BTC (1min) → EDA → features → CV → MLP → API FastAPI → Docker

## Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/` | Estado y lags disponibles |
| GET | `/health` | Health check |
| GET | `/info` | Metadata del pipeline |
| GET | `/example` | Ejemplo de request |
| POST | `/predict` | Predicción de volatilidad |

## Uso rápido

### Con Docker

```bash
docker build -t btc-predictor .
docker run -p 8000:8000 btc-predictor
```

### Sin Docker

```bash
pip install -r requirements.txt
python download_models.py
uvicorn app.api:app --reload
```

### Ejemplo de predicción

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "lag_minutes": 15,
    "lags": [0.025, 0.0003, -3.68, 0.0012, 12.5, 0.0002,
             0.025, -0.0001, -3.68, 0.001, 12.4, 0.0003,
             0.026, 0.0005, -3.64, 0.0015, 12.6, -0.0001,
             0.025, 0.0002, -3.68, 0.0011, 12.3, 0.0005,
             0.025, -0.0003, -3.68, 0.0009, 12.5, 0.0002,
             0.026, 0.0004, -3.64, 0.0013, 12.7, -0.0003,
             0.025, 0.0001, -3.68, 0.001, 12.4, 0.0004,
             0.025, -0.0002, -3.68, 0.0012, 12.5, 0.0001,
             0.026, 0.0003, -3.64, 0.0014, 12.6, -0.0002,
             0.025, 0.0, -3.68, 0.0011, 12.3, 0.0003,
             0.025, -0.0004, -3.68, 0.001, 12.5, 0.0,
             0.026, 0.0005, -3.64, 0.0016, 12.7, -0.0004,
             0.025, 0.0002, -3.68, 0.0012, 12.4, 0.0005,
             0.025, -0.0001, -3.68, 0.0009, 12.5, 0.0002,
             0.026, 0.0003, -3.64, 0.0013, 12.6, -0.0001]
  }'
```

### Respuesta esperada

```json
json{
  "prediction": [0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06],
  "lag_minutes": 15,
  "horizons": ["h=1 min", "h=2 min", "h=3 min", "h=4 min", "h=5 min", "h=6 min", "h=7 min"],
  "model_info": {
    "arch": "(32, 64)",
    "dropout": "(False, False)",
    "rmse_test": 0.003059,
    "lag_minutes": 15
  }
}
```

## Input

El campo `lags` debe contener `lag_minutes × 6` valores en este orden por paso:

[Volatility_daily, log_ret, log_Volatility_daily, hl_range, log_volume, ret_lag1min]

| lag_minutes | valores esperados |
|-------------|-------------------|
| 15 | 90 |
| 30 | 180 |
| 60 | 360 |
| 90 | 540 |

## Tests

```bash
pytest tests/
```

## CI/CD

El repositorio incluye un pipeline de GitHub Actions que:
1. Instala dependencias
2. Descarga modelos desde GitHub Releases
3. Corre linter (flake8)
4. Corre tests unitarios (pytest)
