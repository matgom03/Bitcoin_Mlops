# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Dependencias del sistema para TensorFlow
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Código de la app
COPY app/ ./app/
COPY download_models.py .

# Descargar modelos al construir la imagen
RUN python download_models.py

EXPOSE 8000

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]