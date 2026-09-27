# syntax=docker/dockerfile:1
# Foresight ml-service image (build context: repository root): python -m ml.service, HTTP 8003.
# CPU only: CatBoost + ONNX Runtime CPU (extra `ml`), FastAPI (extra `backend`); no PyTorch (extra `train` is
# for training on the GPU server only). The model versions of the repository (models/<version>/) are baked in;
# FORESIGHT_MODEL_VERSION picks one (default: the newest). The ignore file of this image is
# ml/Dockerfile.dockerignore: the root .dockerignore leaves models/ out of the other images.

# ---- build: resolve the locked dependencies into a virtualenv (uv stays in this stage) ----------------
FROM python:3.12-slim AS build

COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# README.md нужен только метаданным пакета: пустой, чтобы правки документации не пересобирали образ
COPY pyproject.toml uv.lock ./
RUN touch README.md
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra ml --extra backend --no-install-project

# ---- runtime ----------------------------------------------------------------------------------------------
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY --from=build /app/.venv /app/.venv
COPY shared ./shared
COPY ml ./ml
COPY models ./models
# the service must import, find a model and have no torch (the image is CPU-only by design)
RUN /app/.venv/bin/python -m compileall -q shared ml \
    && PYTHONPATH=/app /app/.venv/bin/python -c "import importlib.util as u, ml.service; \
assert u.find_spec('torch') is None, 'torch must not be in the ml-service image'" \
    && ls models/*/manifest.json \
    && useradd --system --uid 10001 foresight \
    && chown -R foresight /app/models  # retraining on the stream writes new versions here (ml/online.py)
USER foresight

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app \
    FORESIGHT_MODELS_DIR=/app/models \
    FORESIGHT_ML_PORT=8003

EXPOSE 8003

HEALTHCHECK --interval=5s --timeout=3s --start-period=30s --retries=10 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8003/health', timeout=2)"]

CMD ["python", "-m", "ml.service"]
