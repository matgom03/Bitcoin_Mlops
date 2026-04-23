# Ejecuta esto para generar un ejemplo realista para lag=15
import numpy as np, json

np.random.seed(0)
lag = 15

# Simular una serie temporal con tendencia y ruido coherente
# como si fueran 15 minutos consecutivos de BTC
vol_base = 0.52
lags = []

for i in range(lag):
    # Volatilidad con leve tendencia decreciente + ruido
    vol      = round(vol_base - i * 0.002 + np.random.normal(0, 0.005), 6)
    log_vol  = round(np.log(max(vol, 0.01)), 6)
    # Retornos con clustering de volatilidad
    log_ret  = round(np.random.normal(0, 0.0008), 6)
    hl       = round(abs(np.random.normal(0.0014, 0.0003)), 6)
    log_volm = round(np.random.uniform(12.2, 13.1), 4)
    ret_lag  = round(np.random.normal(0, 0.0008), 6)

    lags.extend([vol, log_vol, log_ret, hl, log_volm, ret_lag])

print(json.dumps({"lags": lags, "lag_minutes": lag}, indent=2))