# syntax=docker/dockerfile:1
# Foresight replayer: historical traffic.csv -> real NDTP over TCP, control API on 8010 (build context: repo root)
#   python -m replayer serve   HTTP API (+ autostart with FORESIGHT_REPLAY_AUTOSTART=1), default command
#   python -m replayer run     replay without the API

# ---- build: resolve the locked dependencies into a virtualenv (uv stays in this stage) ----------------
FROM python:3.12-slim AS build

COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# dependencies only (the project itself is not installed): this layer is rebuilt only when the lock changes;
# the backend extra brings FastAPI, uvicorn and prometheus-client. The replayer never imports pyarrow (pandas
# works without it), asyncpg or redis, so they stay out of the image (checked below by importing the entry
# points).
# README.md нужен только метаданным пакета: пустой, чтобы правки документации не пересобирали образ
COPY pyproject.toml uv.lock ./
RUN touch README.md
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra backend --no-install-project \
        --no-install-package pyarrow --no-install-package asyncpg --no-install-package redis

# ---- runtime: the same base (the venv links to its interpreter), the venv and two packages -----------
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY --from=build /app/.venv /app/.venv
# the replayer needs only these two packages
COPY shared ./shared
COPY replayer ./replayer
RUN /app/.venv/bin/python -m compileall -q shared replayer \
    && PYTHONPATH=/app /app/.venv/bin/python -c "import replayer.__main__, replayer.api, uvicorn" \
    && useradd --system --uid 10001 foresight
USER foresight

# The dataset is mounted read-only at /app/dataset (the default of shared.data.dataset_dir);
# FORESIGHT_DATASET_DIR or MT_DATASET_DIR point elsewhere. FORESIGHT_REPLAY_* configure the replay;
# change the API port with FORESIGHT_REPLAY_HTTP_PORT (the healthcheck reads it too), not with --http-port.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app \
    FORESIGHT_REPLAY_HTTP_PORT=8010

EXPOSE 8010

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('FORESIGHT_REPLAY_HTTP_PORT', '8010') + '/health', timeout=2)"]

CMD ["python", "-m", "replayer", "serve"]
