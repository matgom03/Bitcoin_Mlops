# download_models.py
import urllib.request
import os
import sys

BASE_URL = "https://github.com/matgom03/Bitcoin_Mlops/releases/download/v1.0-models"

FILES = [
    "pipeline_meta.pkl",
    "mlp_lag15min.keras",
    "mlp_lag30min.keras",
    "mlp_lag60min.keras",
    "mlp_lag90min.keras",
    "scaler_x_lag15min.pkl",
    "scaler_x_lag30min.pkl",
    "scaler_x_lag60min.pkl",
    "scaler_x_lag90min.pkl",
    "scaler_y_lag15min.pkl",
    "scaler_y_lag30min.pkl",
    "scaler_y_lag60min.pkl",
    "scaler_y_lag90min.pkl",
]

os.makedirs("app/models", exist_ok=True)

for filename in FILES:
    dest = os.path.join("app/models", filename)
    if os.path.exists(dest):
        print(f"Ya existe: {filename}")
        continue
    print(f"Descargando {filename}...")
    try:
        urllib.request.urlretrieve(f"{BASE_URL}/{filename}", dest)
        print(f"{filename}")
    except Exception as e:
        print(f"ERROR descargando {filename}: {e}")
        sys.exit(1)

# Verificación final
missing = [f for f in FILES if not os.path.exists(f"app/models/{f}")]
if missing:
    print(f"\nERROR — archivos faltantes: {missing}")
    sys.exit(1)

print("\nTodos los modelos descargados correctamente.")