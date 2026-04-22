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

## Uso

### Levantar el servidor

```bash
# Sin Docker
uvicorn app.api:app --port 8000

# Con Docker
docker build -t btc-predictor .
docker run -p 8000:8000 btc-predictor
```

### Documentación interactiva

```
http://localhost:8000/docs
```

### Ejemplo de predicción (lag=15, valores típicos reales)

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "lag_minutes": 15,
    "lags": [
      0.52, -0.65, 0.00031, 0.00142, 12.847, -0.00018,
      0.52, -0.65, 0.00031, 0.00142, 12.847, -0.00018,
      0.52, -0.65, 0.00031, 0.00142, 12.847, -0.00018,
      0.52, -0.65, 0.00031, 0.00142, 12.847, -0.00018,
      0.52, -0.65, 0.00031, 0.00142, 12.847, -0.00018,
      0.52, -0.65, 0.00031, 0.00142, 12.847, -0.00018,
      0.52, -0.65, 0.00031, 0.00142, 12.847, -0.00018,
      0.52, -0.65, 0.00031, 0.00142, 12.847, -0.00018,
      0.52, -0.65, 0.00031, 0.00142, 12.847, -0.00018,
      0.52, -0.65, 0.00031, 0.00142, 12.847, -0.00018,
      0.52, -0.65, 0.00031, 0.00142, 12.847, -0.00018,
      0.52, -0.65, 0.00031, 0.00142, 12.847, -0.00018,
      0.52, -0.65, 0.00031, 0.00142, 12.847, -0.00018,
      0.52, -0.65, 0.00031, 0.00142, 12.847, -0.00018,
      0.52, -0.65, 0.00031, 0.00142, 12.847, -0.00018
    ]
  }'
```

### Estructura del input

El campo `lags` repite estas 6 features por cada minuto de historia, en este orden:

| Feature | Rango típico | Descripción |
|---|---|---|
| `Volatility_daily` | [0.30, 1.20] | Volatilidad anualizada sqrt(1440×365) |
| `log_Volatility_daily` | [-1.05, -0.22] | Log de la volatilidad diaria |
| `log_ret` | [-0.005, 0.005] | Log-retorno del minuto |
| `hl_range` | [0.0005, 0.003] | Rango High-Low normalizado |
| `log_volume` | [11.5, 13.5] | Log del volumen |
| `ret_lag1min` | [-0.005, 0.005] | Retorno del minuto anterior |

| `lag_minutes` | Total valores |
|---|---|
| 15 | 90 |
| 30 | 180 |
| 60 | 360 |
| 90 | 540 |

### Respuesta esperada

```json
{
  "prediction": [0.58, 0.57, 0.56, 0.55, 0.54, 0.53, 0.52],
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
> También puedes consultar el ejemplo completo con rangos típicos en:
> `GET http://localhost:8000/example`

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
