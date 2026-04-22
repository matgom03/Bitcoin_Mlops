from pydantic import BaseModel, model_validator
from typing import Literal

N_FEATURES = 6
N_STEPS_FORECAST = 7


class PredictRequest(BaseModel):
    lags: list[float]
    lag_minutes: Literal[15, 30, 60, 90] = 30

    @model_validator(mode="after")
    def validate_lags(self):
        expected = self.lag_minutes * N_FEATURES

        if len(self.lags) != expected:
            raise ValueError(
                f"Se esperan {expected} valores "
                f"({self.lag_minutes} pasos × {N_FEATURES} features), "
                f"pero se recibieron {len(self.lags)}."
            )

        return self


class PredictResponse(BaseModel):
    prediction:  list[float]
    lag_minutes: int
    horizons:    list[str]
    model_info:  dict