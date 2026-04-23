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
    "lag_minutes": 15,
    "lags": [
      0.520341, -0.652814,  0.000423, 0.001287, 12.6341, -0.000318,
      0.514892, -0.663987, -0.000891, 0.001542, 12.8821,  0.000423,
      0.511203, -0.670731,  0.001204, 0.001198, 12.4517, -0.000891,
      0.507841, -0.677374, -0.000324, 0.001876, 12.7234,  0.001204,
      0.503129, -0.686761,  0.000567, 0.001421, 12.5891, -0.000324,
      0.498734, -0.695532, -0.000143, 0.001654, 12.9102,  0.000567,
      0.496218, -0.700587,  0.000892, 0.001312, 12.3847, -0.000143,
      0.491543, -0.710064, -0.000671, 0.001789, 12.6723,  0.000892,
      0.487921, -0.717447,  0.000234, 0.001098, 12.8134, -0.000671,
      0.484312, -0.724914, -0.000456, 0.001567, 12.5421,  0.000234,
      0.479856, -0.734138,  0.000789, 0.001234, 12.7891, -0.000456,
      0.476234, -0.741573, -0.000312, 0.001678, 12.4123,  0.000789,
      0.472891, -0.748627,  0.000545, 0.001345, 12.9234, -0.000312,
      0.469123, -0.756412, -0.000678, 0.001892, 12.6512,  0.000545,
      0.465734, -0.763891,  0.000123, 0.001456, 12.8341, -0.000678
    ]
  }'
```

### Respuesta esperada

```json
{
  "prediction": [0.521515, 0.523108, 0.518612, 0.52634, 0.516498, 0.522788, 0.528303],
  "lag_minutes": 15,
  "horizons": ["h=1 min", "h=2 min", "h=3 min", "h=4 min", "h=5 min", "h=6 min", "h=7 min"],
  "model_info": {
    "arch": "(32, 64)",
    "dropout": "(False, False)",
    "rmse_test": 0.002427,
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