import os
import joblib
import numpy as np
import tensorflow as tf

DIR_MODELS = "app/models"
LAGS_LIST = [15, 30, 60, 90]
N_FEATURES = 6
N_STEPS_FORECAST = 7


def test_models_load():
    """Verifica que todos los modelos cargan correctamente"""
    for lag in LAGS_LIST:
        model_path = os.path.join(DIR_MODELS, f"mlp_lag{lag}min.keras")
        assert os.path.exists(model_path)

        model = tf.keras.models.load_model(model_path)
        assert model is not None


def test_model_prediction_shape():
    """Verifica shape de predicción"""
    lag = 15

    model = tf.keras.models.load_model(
        os.path.join(DIR_MODELS, f"mlp_lag{lag}min.keras")
    )

    scaler_x = joblib.load(
        os.path.join(DIR_MODELS, f"scaler_x_lag{lag}min.pkl")
    )

    # Input simulado
    X = np.random.rand(1, lag, N_FEATURES)

    # Flatten como en API
    X_sc = scaler_x.transform(
        X.reshape(-1, N_FEATURES)
    ).reshape(1, lag * N_FEATURES)

    X_t = tf.constant(X_sc, dtype=tf.float32)

    y_pred = model.predict(X_t)

    assert y_pred.shape == (1, N_STEPS_FORECAST)


def test_prediction_values():
    """Verifica valores válidos"""
    lag = 15

    model = tf.keras.models.load_model(
        os.path.join(DIR_MODELS, f"mlp_lag{lag}min.keras")
    )

    X = np.random.rand(1, lag * N_FEATURES)
    X = tf.constant(X, dtype=tf.float32)

    y_pred = model.predict(X)[0]

    assert len(y_pred) == N_STEPS_FORECAST
    assert all(isinstance(v, (float, np.floating)) for v in y_pred)