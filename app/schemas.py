from pydantic import BaseModel, field_validator
from typing import Literal

N_FEATURES = 6
N_STEPS_FORECAST = 7

# ── Schemas ───────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    """
    Entrada del endpoint /predict.

    lags: matriz aplanada de shape (n_steps_input × n_features).
          Orden de features (igual que cols_ordered del notebook 2):
            [Volatility_daily, log_ret, log_Volatility_daily,
             hl_range, log_volume, ret_lag1min]

    lag_minutes: tamaño de la ventana de input en minutos.
                 Valores válidos: 15, 30, 60, 90.
    """
    lags: list[float]
    lag_minutes: Literal[15, 30, 60, 90] = 30

    @field_validator("lags")
    @classmethod
    def validate_lags(cls, v, info):
        lag_minutes = info.data.get("lag_minutes", 30)
        expected    = lag_minutes * N_FEATURES
        if len(v) != expected:
            raise ValueError(
                f"Se esperan {expected} valores "
                f"({lag_minutes} pasos × {N_FEATURES} features), "
                f"pero se recibieron {len(v)}."
            )
        return v


class PredictResponse(BaseModel):
    """
    Salida del endpoint /predict.

    prediction: lista de 7 floats con la volatilidad predicha
                anualizada para h=1..7 minutos.
                Escala: sqrt(1440×365) — rango típico BTC [0.06, 0.80]

    lag_minutes: lag usado para la predicción.
    horizons:    etiquetas de cada horizonte.
    model_info:  arquitectura del modelo usado.
    """
    prediction:  list[float]
    lag_minutes: int
    horizons:    list[str]
    model_info:  dict

