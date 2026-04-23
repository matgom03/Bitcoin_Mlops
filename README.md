# Bitcoin Volatility Predictor — MLOps

Predice la volatilidad de Bitcoin a 7 minutos hacia adelante usando un MLP
entrenado con velas de 1 minuto (2023–2026).

## Arquitectura
datos BTC (1min) → EDA → features → CV → MLP → API FastAPI → Docker

## Resultados del modelo

| Lag (min) | RMSE | MAPE | BDS p-val | iid |
|-----------|------|------|-----------|-----|
| 15 | 0.002427 | 0.3471% | 0.0446 | ✗ |
| **30** | **0.003059** | **0.4750%** | **0.1652** | **✓** |
| 60 | 0.004288 | 0.6598% | 0.4386 | ✓ |
| 90 | 0.005490 | 0.9207% | 0.6352 | ✓ |

> **Configuración recomendada: Lag = 30 min** — mejor balance entre
> precisión (RMSE) y calidad de residuos (BDS > 0.05 en todos los folds).
> Lag=15 tiene mejor RMSE pero sus residuos muestran estructura remanente
> en los períodos más recientes (folds 4 y 5, régimen 2025).

## Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/` | Estado y lags disponibles |
| GET | `/health` | Health check |
| GET | `/info` | Metadata del pipeline |
| GET | `/example` | Ejemplo de request con valores reales |
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
http://localhost:8000/docs

### Estructura del input

El campo `lags` contiene `lag_minutes × 6` valores aplanados.
Orden de features por cada minuto de historia:

| # | Feature | Rango típico | Descripción |
|---|---------|-------------|-------------|
| 0 | `Volatility_daily` | [0.30, 1.20] | Volatilidad anualizada sqrt(1440×365) |
| 1 | `log_Volatility_daily` | [-1.05, -0.22] | log(Volatility_daily) |
| 2 | `log_ret` | [-0.005, 0.005] | Log-retorno del minuto |
| 3 | `hl_range` | [0.0005, 0.003] | (high-low)/close |
| 4 | `log_volume` | [11.5, 13.5] | log(volumen) |
| 5 | `ret_lag1min` | [-0.005, 0.005] | Log-retorno del minuto anterior |

| `lag_minutes` | Total valores |
|---------------|---------------|
| 15 | 90 |
| 30 | 180 |
| 60 | 360 |
| 90 | 540 |

### Ejemplo de predicción (lag=15)

Valores representativos de 15 minutos consecutivos de BTC
con volatilidad en tendencia bajista suave:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
  "lags": [
    0.51368,
    -0.666155,
    0.001222,
    0.001109,
    13.0663,
    0.000376,
    0.520355,
    -0.653244,
    -0.001165,
    0.001089,
    12.6897,
    -0.000213,
    0.523891,
    -0.646472,
    8.6e-05,
    0.001171,
    12.7662,
    -0.00062,
    0.518454,
    -0.656904,
    0.000889,
    0.001537,
    12.5388,
    -0.0014,
    0.511971,
    -0.669487,
    -0.000374,
    0.001273,
    12.6638,
    0.000852,
    0.498293,
    -0.696567,
    -0.000454,
    0.001156,
    12.7359,
    0.000965,
    0.502499,
    -0.688162,
    -0.001053,
    0.001643,
    12.3909,
    -0.000836,
    0.5022,
    -0.688757,
    -0.001122,
    0.00164,
    12.9371,
    0.000112,
    0.500493,
    -0.692162,
    0.000679,
    0.001715,
    12.9267,
    -0.000429,
    0.508082,
    -0.677112,
    -1.8e-05,
    0.001262,
    12.4377,
    -0.00011,
    0.509494,
    -0.674337,
    -0.000284,
    0.001284,
    12.9377,
    -0.001542,
    0.492336,
    -0.708594,
    0.00095,
    0.001965,
    12.7368,
    0.000872,
    0.497616,
    -0.697927,
    -0.000793,
    0.001051,
    12.2277,
    -1.7e-05,
    0.493262,
    -0.706715,
    -0.000466,
    0.001437,
    13.0938,
    0.000525,
    0.489949,
    -0.713454,
    -0.000844,
    0.001083,
    12.4614,
    -0.001027
  ],
  "lag_minutes": 15
}'
```

### Respuesta esperada

```json
{
  "prediction": [
    0.544116,
    0.54646,
    0.551525,
    0.553674,
    0.546469,
    0.562522,
    0.549704
  ],
  "lag_minutes": 15,
  "horizons": [
    "h=1 min",
    "h=2 min",
    "h=3 min",
    "h=4 min",
    "h=5 min",
    "h=6 min",
    "h=7 min"
  ],
  "model_info": {
    "arch": "(32, 64)",
    "dropout": "(False, False)",
    "rmse_test": 0.0024268850684165955,
    "lag_minutes": 15
  }
}
```

> La predicción indica una volatilidad anualizada de ~52% sostenida
> durante los próximos 7 minutos, coherente con el input de baja
> volatilidad relativa para BTC.
> Para el ejemplo completo con todos los lags disponibles:
> `GET http://localhost:8000/example`

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