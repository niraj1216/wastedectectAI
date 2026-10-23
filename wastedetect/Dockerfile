# ---------- stage 1: export the zero-shot model to ONNX (heavy, thrown away) ----------
FROM python:3.11-slim AS builder
ARG EXPORT_IMGSZ=640
ENV EXPORT_IMGSZ=${EXPORT_IMGSZ} PIP_NO_CACHE_DIR=1 YOLO_CONFIG_DIR=/tmp/Ultralytics
RUN apt-get update && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY requirements-export.txt .
RUN pip install -r requirements-export.txt
COPY prompts.py export_onnx.py ./
RUN python export_onnx.py

# ---------- stage 2: tiny runtime image (no torch) ----------
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=1
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY prompts.py waste_core.py app.py api.py ./
COPY samples ./samples
COPY --from=builder /build/models ./models

EXPOSE 10000
CMD streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 \
    --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false \
    --server.fileWatcherType=none --server.maxUploadSize=10 --browser.gatherUsageStats=false
