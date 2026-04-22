import numpy as np
import joblib
import tensorflow as tf

DIR_MODELS   = "app/models"
LAG          = 30
N_FEATURES   = 6
N_STEPS_FORE = 7
ANNUAL_FACTOR = np.sqrt(1440 * 365)   # ≈ 724.98 — escala real del scaler

scaler_x = joblib.load(f"{DIR_MODELS}/scaler_x_lag{LAG}min.pkl")
scaler_y = joblib.load(f"{DIR_MODELS}/scaler_y_lag{LAG}min.pkl")
model    = tf.keras.models.load_model(f"{DIR_MODELS}/mlp_lag{LAG}min.keras")

# ── Input corregido con orden y escala correctos ──────────────────────────────
# orden: ['Volatility_daily', 'log_Volatility_daily', 'log_ret',
#          'hl_range', 'log_volume', 'ret_lag1min']
# Volatility_daily ahora en escala sqrt(1440*365) → rango [0.30, 1.20]

lags = []
for _ in range(LAG):
    vol_daily = np.random.uniform(0.35, 0.80)        # Volatility_daily anualizada real
    log_vol   = np.log(vol_daily)                    # log_Volatility_daily ≈ [-1.05, -0.22]
    log_ret   = np.random.uniform(-0.003, 0.003)     # log_ret
    hl        = np.random.uniform(0.0005, 0.003)     # hl_range
    log_volum = np.random.uniform(11.5, 13.5)        # log_volume
    ret_lag   = np.random.uniform(-0.003, 0.003)     # ret_lag1min

    lags.extend([
        round(float(vol_daily), 6),   # col 0: Volatility_daily
        round(float(log_vol),   6),   # col 1: log_Volatility_daily
        round(float(log_ret),   6),   # col 2: log_ret
        round(float(hl),        6),   # col 3: hl_range
        round(float(log_volum), 4),   # col 4: log_volume
        round(float(ret_lag),   6),   # col 5: ret_lag1min
    ])

X_raw = np.array(lags, dtype=np.float32).reshape(1, LAG, N_FEATURES)
X_sc  = scaler_x.transform(X_raw.reshape(-1, N_FEATURES)).reshape(1, LAG * N_FEATURES)
X_t   = tf.constant(X_sc, dtype=tf.float32)
y_sc  = model.predict(X_t, verbose=0)
y_inv = scaler_y.inverse_transform(y_sc.reshape(-1, 1)).reshape(1, N_STEPS_FORE)

print(f"X_raw rango    : [{X_raw.min():.4f}, {X_raw.max():.4f}]")
print(f"X_sc rango     : [{X_sc.min():.4f}, {X_sc.max():.4f}]")   # debe ser ≈ [-3, 3]
print(f"y_sc (modelo)  : {y_sc.flatten().round(4)}")               # debe ser ≈ [-2, 2]
print(f"y_inv          : {y_inv.flatten().round(6)}")               # debe ser ≈ [0.30, 0.80]
print(f"\nRango esperado y_inv: [0.30, 0.80]")
print(f"Resultado OK   : {all(0.05 < v < 2.0 for v in y_inv.flatten())}")