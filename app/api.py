# ─────────────────────────────────────────────────────────────────────────────
# app/api.py — API de predicción de volatilidad BTC
# ─────────────────────────────────────────────────────────────────────────────
# Recibe: ventana de lags en minutos (15, 30, 60 o 90) con todas las features
# Devuelve: 7 predicciones de volatilidad anualizada (h=1..7 minutos)
#
# Escala de salida: volatilidad anualizada con factor sqrt(1440 * 365)
# consistente con el ejemplo del enunciado: [0.06, 0.12]
# ─────────────────────────────────────────────────────────────────────────────

import os
import joblib
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from typing import Literal
from contextlib import asynccontextmanager
from .schemas import PredictRequest, PredictResponse

# ── Constantes del pipeline (deben coincidir con notebook 2) ──────────────────
DIR_MODELS = "app/models"
LAGS_LIST        = [15, 30, 60, 90]
BEST_LAG         = 30
N_STEPS_FORECAST = 7
N_FEATURES       = 6      # número de columnas en cols_ordered
SCALE_CORRECTION = 1.0
COLS_ORDERED = [
    "Volatility_daily",
    "log_Volatility_daily",
    "log_ret",
    "hl_range",
    "log_volume",
    "ret_lag1min",
]                                  

# ── Recursos globales (cargados una sola vez al iniciar) ──────────────────────
_models   = {}   # { lag: keras_model }
_scalers  = {}   # { lag: (scaler_x, scaler_y) }
_meta     = {}   # pipeline_meta


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga modelos y scalers al arrancar la API."""
    global _models, _scalers, _meta

    print("Cargando pipeline_meta...")
    _meta = joblib.load(os.path.join(DIR_MODELS, "pipeline_meta.pkl"))

    for lag in LAGS_LIST:
        model_path    = os.path.join(DIR_MODELS, f"mlp_lag{lag}min.keras")
        scaler_x_path = os.path.join(DIR_MODELS, f"scaler_x_lag{lag}min.pkl")
        scaler_y_path = os.path.join(DIR_MODELS, f"scaler_y_lag{lag}min.pkl")

        if not all(os.path.exists(p) for p in
                   [model_path, scaler_x_path, scaler_y_path]):
            print(f"Archivos de lag={lag} no encontrados — se omite")
            continue

        _models[lag]  = tf.keras.models.load_model(model_path)
        _scalers[lag] = (
            joblib.load(scaler_x_path),
            joblib.load(scaler_y_path),
        )
        print(f"Lag={lag}min cargado")

    print(f"API lista. Lags disponibles: {list(_models.keys())}")
    yield
    # Limpieza al apagar
    _models.clear()
    _scalers.clear()


app = FastAPI(
    title="BTC Volatility Predictor",
    description=(
        "Predice la volatilidad de Bitcoin a 7 minutos hacia adelante "
        "usando un MLP entrenado con datos de velas de 1 minuto (2023–2026). "
        "La volatilidad se devuelve anualizada con factor sqrt(1440×365)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)
# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status":       "ok",
        "description":  "BTC Volatility Predictor API",
        "lags_available": list(_models.keys()),
        "best_lag":     _meta.get("best_lag", "N/A"),
        "n_outputs":    N_STEPS_FORECAST,
        "scale":        "volatilidad anualizada sqrt(1440*365)",
    }


@app.get("/health")
def health():
    return {
        "status":        "ok",
        "models_loaded": list(_models.keys()),
    }


@app.get("/info")
def info():
    """Devuelve metadata del pipeline entrenado."""
    return {
        "lags_list":             _meta.get("LAGS_LIST"),
        "n_steps_forecast":      _meta.get("N_STEPS_FORECAST"),
        "cols_ordered":          _meta.get("cols_ordered"),
        "best_lag": BEST_LAG,
        "best_arch_per_lag":     _meta.get("best_arch_per_lag"),
        "rmse_test_avg_per_lag": _meta.get("rmse_test_avg_per_lag"),
        "annual_factor":         "sqrt(1440 * 365)",
        "scale_correction":      float(SCALE_CORRECTION),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(data: PredictRequest):
    """
    Predice la volatilidad de BTC para los próximos 7 minutos.

    El input debe ser la ventana de historia aplanada:
    lag_minutes × n_features valores en el orden de cols_ordered.
    """
    lag = data.lag_minutes

    # Verificar que el modelo está disponible
    if lag not in _models:
        raise HTTPException(
            status_code=404,
            detail=f"Modelo para lag={lag}min no disponible. "
                   f"Disponibles: {list(_models.keys())}"
        )

    model, (scaler_x, scaler_y) = _models[lag], _scalers[lag]

    # ── Preprocesamiento ──────────────────────────────────────────────────────
    try:
        # Reconstruir shape 3D: (1, n_steps_input, n_features)
        X_raw = np.array(data.lags, dtype=np.float32).reshape(1, lag, N_FEATURES)

        # Escalar X con el scaler del fold 0 (igual que en entrenamiento)
        X_sc  = scaler_x.transform(X_raw.reshape(-1, N_FEATURES)).reshape(1, lag * N_FEATURES)

        # Convertir a tensor
        X_t   = tf.constant(X_sc, dtype=tf.float32)

    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Error en preprocesamiento: {str(e)}"
        )

    # ── Predicción ────────────────────────────────────────────────────────────
    try:
        y_sc  = model.predict(X_t, verbose=0)        # (1, 7) escalado
        print(f"X_raw rango    : [{X_raw.min():.4f}, {X_raw.max():.4f}]")
        print(f"X_sc rango     : [{X_sc.min():.4f}, {X_sc.max():.4f}]")  # debe ser ≈ [-3, 3]
        print(f"y_sc (modelo)  : {y_sc.flatten()}")                        # debe ser ≈ [-2, 2]
        print(f"scaler_y mean_ : {scaler_y.mean_}")                        # media del target
        print(f"scaler_y scale_: {scaler_y.scale_}")                       # std del target
        y_inv = scaler_y.inverse_transform(y_sc.reshape(-1, 1)) # des-escalar
        y_inv = y_inv.reshape(1, N_STEPS_FORECAST)
        print(f"y_inv          : {y_inv.flatten()}") 

        # Aplicar corrección de escala: sqrt(365) → sqrt(1440*365)
        y_out = (y_inv * SCALE_CORRECTION).flatten().tolist()

        # Clip de seguridad: volatilidad no puede ser negativa
        y_out = [max(0.0, round(float(v), 6)) for v in y_out]

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error en predicción: {str(e)}"
        )

    # ── Respuesta ─────────────────────────────────────────────────────────────
    arch_info = _meta.get("best_arch_per_lag", {}).get(lag, {})

    return PredictResponse(
    prediction  = y_out,
    lag_minutes = int(lag),                    # int() explícito
    horizons    = [f"h={h} min" for h in range(1, N_STEPS_FORECAST + 1)],
    model_info  = {
        "arch":        str(arch_info.get("arch", "N/A")),
        "dropout":     str(arch_info.get("dmask", "N/A")),
        "rmse_test":   float(_meta.get("rmse_test_avg_per_lag", {}).get(lag, 0)),  # float() explícito
        "lag_minutes": int(lag),               # int() explícito
    },
    )


# ── Ejemplo de uso (para documentación automática de FastAPI) ─────────────────
@app.get("/example")
def example():
    lag = 15
    example_lags = []
    for _ in range(lag):
        example_lags.extend([
            0.52,      # Volatility_daily  [0.30, 1.20]
           -0.65,      # log_Volatility_daily [-1.05, -0.22]
            0.00031,   # log_ret
            0.00142,   # hl_range
           12.847,     # log_volume
           -0.00018,   # ret_lag1min
        ])

    return {
        "endpoint":      "POST /predict",
        "cols_ordered":  COLS_ORDERED,
        "rangos_tipicos": {
            "Volatility_daily":     "[0.30, 1.20] — volatilidad anualizada sqrt(1440×365)",
            "log_Volatility_daily": "[-1.05, -0.22]",
            "log_ret":              "[-0.005, 0.005]",
            "hl_range":             "[0.0005, 0.003]",
            "log_volume":           "[11.5, 13.5]",
            "ret_lag1min":          "[-0.005, 0.005]",
        },
        "request_body": {
            "lags":        example_lags,
            "lag_minutes": lag,
        },
        "expected_response": {
            "prediction":  [0.58, 0.57, 0.56, 0.55, 0.54, 0.53, 0.52],
            "lag_minutes": lag,
            "horizons":    [f"h={h} min" for h in range(1, 8)],
        },
        "note": (
            "Los valores de 'lags' deben seguir el orden de cols_ordered, "
            "repitiendo las 6 features por cada minuto de historia."
        ),
    }