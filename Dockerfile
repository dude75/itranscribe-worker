# TARGET=cpu — slim Python, CPU wheels of torch.
# TARGET=gpu — NVIDIA CUDA 12.6 + cu126 torch (needs nvidia-container-toolkit at run).
ARG TARGET=cpu

FROM python:3.12-slim-bookworm AS cpu-base
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        build-essential \
        libsndfile1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04 AS gpu-base
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
        python3-dev \
        ffmpeg \
        git \
        build-essential \
        libsndfile1 \
        libsndfile1-dev \
        libgomp1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

FROM ${TARGET}-base AS runtime

ARG TARGET=cpu
ARG DEVICE=cpu
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG TORCH_VERSION=2.10.0
ARG TORCHAUDIO_VERSION=2.10.0

WORKDIR /app

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    DATA_DIR=/data \
    MODELS_DIR=/data/models \
    SQLITE_PATH=/data/tasks.db \
    LOG_DIR=/data/logs \
    PERFORMANCE_LOG=/data/logs/performance_log.csv \
    WORKERS=1 \
    DEVICE=${DEVICE}

COPY requirements.txt requirements-ml.txt ./
RUN python3 -m venv /opt/venv \
    && pip install --no-cache-dir -U pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir \
        "torch==${TORCH_VERSION}" \
        "torchaudio==${TORCHAUDIO_VERSION}" \
        --index-url "${TORCH_INDEX_URL}" \
    && grep -vE '^(torch|torchaudio)==' requirements-ml.txt > /tmp/requirements-ml-notorch.txt \
    && pip install --no-cache-dir -r /tmp/requirements-ml-notorch.txt \
    && rm /tmp/requirements-ml-notorch.txt

COPY version.txt ./
COPY app ./app

EXPOSE 8000
VOLUME ["/data"]

CMD ["sh", "-c", "uvicorn app.main:app --host ${HOST:-0.0.0.0} --port ${PORT:-8000} --workers 1"]
