# syntax=docker/dockerfile:1
# Foresight backend image: one image, three services (build context: repository root)
#   python -m backend.ingest     NDTP TCP 9201 + HTTP 8001
#   python -m backend.predictor  HTTP 8002
#   python -m backend.api        HTTP 8000 (default command)

# ---- build: resolve the locked dependencies into a virtualenv (uv stays in this stage) ----------------
FROM python:3.12-slim AS build

COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# dependencies only (the project itself is not installed): this layer is rebuilt only when the lock changes;
# the data stack (numpy, pandas, pyarrow) is here for the predictor: it runs the offline stop detector and
# feature code (shared/stops.py, shared/features.py) on the stream. No CatBoost / ONNX: the model is ml-service.
# README.md нужен только метаданным пакета: пустой, чтобы правки документации не пересобирали образ
COPY pyproject.toml uv.lock ./
RUN touch README.md
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra backend --no-install-project

# ---- runtime: the same base (the venv links to its interpreter), the venv and two packages -----------
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY --from=build /app/.venv /app/.venv
# the services need these two packages and the feature labels of ml/ (causes of the forecasts); the rest of
# ml/, the replayer and the dataset stay out of the image
COPY shared ./shared
COPY backend ./backend
COPY ml/__init__.py ml/feature_labels.py ./ml/
RUN /app/.venv/bin/python -m compileall -q shared backend ml \
    && PYTHONPATH=/app /app/.venv/bin/python -c "import backend.ingest, backend.predictor, backend.api" \
    && useradd --system --uid 10001 foresight
USER foresight

# the dataset (unit_id -> tr_id map, the plan schedule of the predictor) is mounted read-only at /app/dataset
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app \
    FORESIGHT_UNIT_MAP_DIR=/app/dataset

EXPOSE 8000 8001 8002 9201

CMD ["python", "-m", "backend.api"]
