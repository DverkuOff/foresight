"""Проверка «поток = офлайн»: онлайн-прогнозы Foresight против разметки и той же модели офлайн.

Запуск — на сервере, из корня репозитория (``uv run --all-extras python scripts/validate_online.py …``).

Подкоманды:

* ``simulate`` — прогон **того же движка predictor** (``backend.forecast.ForecastEngine`` под
  ``backend.predictor.PredictorCore``) по телеметрии сплита в порядке ``receive_time``, как её отдаёт
  replayer: часы потока, окна треков, тик каждые 30 с, детектор на потоке, признаки, прогнозы. Модель — в
  процессе: версия реестра (``--model v1``), ансамбль v1, обученный только на train (``--model holdout``:
  именно он дал test MAE 75.8), или только fallback (``--model fallback``). Результат —
  ``artifacts/online/<name>/{predictions,updates,passages,alerts,ticks}.parquet``;
* ``export`` — то же из PostgreSQL работающего стека (``docker compose exec postgres psql``) плюс статистика
  predictor и ml-service → ``artifacts/online/<name>/``;
* ``evaluate`` — сопоставление онлайн-прогнозов с ``labels_<split>`` (точки, где ``(tr_id, target_stop_id)``
  совпадают, а тик прогноза = ``T``, т.е. lead ∈ [10, 15] мин): онлайн-MAE против офлайн-MAE той же модели
  на тех же точках (с настоящим ``cur_dev_s`` и с его онлайн-аналогом), baseline, распределение lead,
  «задним числом», доля fallback, длительность тика; печатает markdown для ``docs/online-validation.md``;
* ``detector`` — онлайн-детектор (проходы прогона) против офлайн ``shared.stops.detect_all`` и факта
  расписания;
* ``fallback-coef`` — коэффициенты fallback-формулы predictor ``a + k · phys`` (MAE-оптимум на train,
  реальные ТС, те же строки, что у обучения v1).

Факты расписания и разметка используются только для оценки, никогда — как вход движка.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.bus import TelemetryEvent  # noqa: E402
from backend.config import Settings  # noqa: E402
from backend.db import UPSERT_KEYS  # noqa: E402
from backend.forecast import ForecastEngine  # noqa: E402
from backend.mlclient import MLPrediction, MLResult  # noqa: E402
from backend.predictor import PredictorCore, TickContext  # noqa: E402
from backend.runtime import DependencyStatus  # noqa: E402
from backend.schedule import Schedule  # noqa: E402
from shared.data import dataset_dir, load_points, load_schedule, load_traffic  # noqa: E402
from shared.features import FEATURE_NAMES, build_features  # noqa: E402

OUT = ROOT / "artifacts" / "online"
HOLDOUT_DIR = OUT / "holdout_model"
MODELS = ROOT / "models"
SEQ_ONNX_TRAIN = Path("/root/wt/mlv2/artifacts/v2/onnx")
"""Train-only ONNX of the v2 sequence model (``ml/train_v2.py`` final stage: ``<best>_train.*.onnx``)."""
LEAD_MIN_S, LEAD_MAX_S = 600.0, 900.0
CUR_DEV_MODES = ["median3", "last_stop", "last_regular", "nan"]

Predict = Callable[[pd.DataFrame], np.ndarray]


# ---- models in process -----------------------------------------------------------------------------------


class LocalModel:
    """The forecast model of the simulation: a registry bundle (or a prediction function) in process instead
    of ml-service — the same outputs (P10–P90, p_late for v2) and, for a model with a sequence component, the
    same telemetry sequences the predictor sends to ml-service."""

    def __init__(self, version: str, fn: Predict | None, bundle: Any = None, sequences: bool = True) -> None:
        self.fn = fn
        self.bundle = bundle
        self.model_version: str | None = version
        self.status = DependencyStatus("ml-service")
        self.status.ok = fn is not None or bundle is not None
        self.sequence_shape = bundle.sequence_shape if bundle is not None and sequences else None
        self.seconds: list[float] = []

    async def predict(
        self, rows: Sequence[Mapping[str, float]], sequences: Sequence[np.ndarray] | None = None
    ) -> MLResult | None:
        if self.fn is None and self.bundle is None:
            return None
        t0 = time.perf_counter()
        frame = pd.DataFrame(list(rows)).reindex(columns=FEATURE_NAMES).astype(np.float64)
        if self.bundle is not None:
            seq = np.stack(list(sequences)) if sequences is not None else None
            out = self.bundle.predict(frame, sequences=seq)
            preds = [
                MLPrediction(
                    float(r.pred_delay_s),
                    _opt(r.p10),
                    _opt(r.p50),
                    _opt(r.p90),
                    _opt(r.p_late),
                    _opt(r.expected_abs_error_s),
                )
                for r in out.itertuples(index=False)
            ]
            precision = self.bundle.precision
        else:
            assert self.fn is not None
            preds = [MLPrediction(float(p)) for p in np.asarray(self.fn(frame), dtype=np.float64)]
            precision = "fp32"
        self.seconds.append(time.perf_counter() - t0)
        return MLResult(str(self.model_version), precision, None, preds)


def _opt(x: Any) -> float | None:
    return float(x) if x is not None and np.isfinite(x) else None


def holdout_members() -> list[Any]:
    """Ансамбль v1, обученный только на train (как для test MAE 75.8 в ``ml/train.py``), с кэшем."""
    from catboost import CatBoostRegressor

    from ml.dataset import build_split, leak_mask
    from ml.train import ENSEMBLE, PARAMS, SEEDS, fit_ensemble

    meta = HOLDOUT_DIR / "members.json"
    if meta.exists():
        members = []
        for item in json.loads(meta.read_text()):
            model = CatBoostRegressor()
            model.load_model(str(HOLDOUT_DIR / item["file"]))
            members.append((model, next(c for c in ENSEMBLE if c.name == item["config"])))
        return members
    train_all = build_split("train")
    leak = leak_mask(train_all, [load_points("test"), load_points("validate")])
    train = train_all[~leak].reset_index(drop=True)
    print(f"training the holdout ensemble on {len(train)} train rows ({len(ENSEMBLE)} × {len(SEEDS)})")
    members = fit_ensemble(train, dict(PARAMS))
    HOLDOUT_DIR.mkdir(parents=True, exist_ok=True)
    items = []
    for i, (model, cfg) in enumerate(members):
        name = f"holdout_{i}.cbm"
        model.save_model(str(HOLDOUT_DIR / name))
        items.append({"file": name, "config": cfg.name})
    meta.write_text(json.dumps(items))
    return members


def load_model(name: str) -> tuple[str, Predict | None, Any]:
    """Модель по имени: версия реестра (``v1``, ``v2-holdout`` из ``models/``), ``holdout`` (ансамбль v1 на
    train в процессе) или ``fallback``.

    Returns:
        Версия, функция прогноза ``df → pred`` (без последовательностей) и бандл (``None`` — не из реестра).
    """
    if name == "fallback":
        return "fallback", None, None
    if name == "holdout":
        from ml.train import predict_ensemble

        members = holdout_members()
        return "holdout", lambda df: predict_ensemble(members, df), None
    from ml.inference import load_bundle

    path = MODELS / name
    bundle = load_bundle(path if path.is_dir() else name)
    return bundle.version, lambda df: bundle.predict(df)["pred_delay_s"].to_numpy(), bundle


def model_fn(name: str) -> tuple[str, Predict | None]:
    """Функция прогноза по имени (см. :func:`load_model`)."""
    version, fn, _ = load_model(name)
    return version, fn


def offline_predict(
    name: str, split: str, frames: Sequence[pd.DataFrame], sequences: bool = True
) -> list[np.ndarray]:
    """Офлайн-прогноз модели на таблицах точек сплита (``sample_id`` + признаки), с последовательностями
    ``ml.dataset.build_sequences_split`` для версий с последовательной моделью (как при обучении), если
    ``sequences`` (так же, как прогон на потоке)."""
    _, fn, bundle = load_model(name)
    assert fn is not None
    if bundle is None or bundle.sequence_shape is None or not sequences:
        return [np.asarray(fn(f), dtype=np.float64) for f in frames]
    from ml.dataset import build_sequences_split, build_split

    order = {int(s): i for i, s in enumerate(build_split(split)["sample_id"])}
    seq = build_sequences_split(split)
    out = []
    for f in frames:
        idx = [order[int(s)] for s in f["sample_id"]]
        out.append(bundle.predict(f, sequences=seq[idx])["pred_delay_s"].to_numpy(dtype=np.float64))
    return out


# ---- simulation ------------------------------------------------------------------------------------------


class Collector:
    """Sink of the engine: upserted tables keep the latest row per key, the others every row."""

    def __init__(self) -> None:
        self.rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.latest: dict[str, dict[Any, dict[str, Any]]] = defaultdict(dict)

    def write(self, table: str, row: Mapping[str, Any]) -> None:
        key = UPSERT_KEYS.get(table)
        if key is not None and key in row:
            self.latest[table][row[key]] = dict(row)
        else:
            self.rows[table].append(dict(row))

    def frame(self, table: str) -> pd.DataFrame:
        rows = list(self.latest[table].values()) if table in self.latest else self.rows.get(table, [])
        return pd.DataFrame(rows)


def replay_data(split: str, start: str | None, until: str | None) -> Any:
    """Телеметрия сплита как у replayer: массивы в порядке ``receive_time`` (``ReplayData``)."""
    from replayer.source import ReplayData

    data = ReplayData.from_frame(pd.read_csv(dataset_dir() / split / "traffic.csv"), split)
    return data.select(start=start, until=until) if start or until else data


def _event(data: Any, i: int) -> TelemetryEvent:
    """Row ``i`` as the ingest publishes it (NDTP precision: 1e-7 degree, whole seconds, km/h)."""
    tr = int(data.tr_id[i])
    return TelemetryEvent(
        unit_id=int(data.unit_id[i]),
        tr_id=tr if tr >= 0 else None,
        ts=float(data.event_s[i]),
        lon=round(float(data.lon[i]), 7),
        lat=round(float(data.lat[i]), 7),
        speed=int(data.speed[i]),
        course=int(data.course[i]),
        valid=bool(data.valid[i]),
    )


async def run_simulation(args: argparse.Namespace) -> Path:
    version, fn, bundle = load_model(args.model)
    model = LocalModel(version, fn if bundle is None else None, bundle, sequences=args.sequences)
    settings = Settings(
        database_url="",
        unit_map_splits="",
        ml_url="",
        schedule_split=args.split,
        cur_dev_mode=args.cur_dev,
        cur_dev_lag_s=args.cur_dev_lag,
        log_updates=True,
        ml_sequences=args.sequences,
        alert_confirm_ticks=args.confirm_ticks,
        alert_hysteresis_s=args.hysteresis,
        incident_clear_ticks=args.clear_ticks,
    )
    schedule = Schedule.load(dataset_dir(), args.split)
    sink = Collector()
    engine = ForecastEngine(settings, schedule, model, sink)
    ticks: list[dict[str, float]] = []
    feature_rows: list[pd.DataFrame] = []

    def on_features(t: float, targets: list[tuple[int, int]], features: pd.DataFrame) -> None:
        plans = schedule.vehicles
        head = pd.DataFrame(
            {
                "tick_at": pd.to_datetime(np.full(len(targets), round(t * 1e9), dtype=np.int64), unit="ns"),
                "tr_id": [tr for tr, _ in targets],
                "target_stop_id": [int(plans[tr].stop_ids[i]) for tr, i in targets],
            }
        )
        feature_rows.append(pd.concat([head, features.reset_index(drop=True)], axis=1))

    engine.on_features = on_features

    async def on_tick(ctx: TickContext) -> None:
        await engine.tick(ctx)
        tm = engine.timings
        ticks.append(
            {
                "tick_at": ctx.stream_time.timestamp(),
                "total_s": tm.total,
                "detector_s": tm.detector,
                "features_s": tm.features,
                "ml_s": tm.ml,
                "active": engine.last_active,
                "targets": engine.last_targets,
            }
        )

    core = PredictorCore(
        window_s=settings.track_window_s, tick_period_s=settings.tick_period_s, on_tick=on_tick
    )
    data = replay_data(args.split, args.start, args.until)
    recv = data.recv
    edges = np.searchsorted(recv, np.arange(recv[0], recv[-1] + 5.0, 5.0), side="left")
    edges = np.unique(np.r_[edges, len(recv)])
    t0 = time.perf_counter()
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        if b > a:  # events of 5 s of receive time: never more than one tick boundary per batch
            await core.handle([(f"{i}-0", _event(data, i)) for i in range(a, b)])
    wall = time.perf_counter() - t0
    out = OUT / args.name
    out.mkdir(parents=True, exist_ok=True)
    tables = {
        "predictions": "predictions",
        "prediction_updates": "updates",
        "stop_passages": "passages",
        "alerts": "alerts",
        "incidents": "incidents",
    }
    for table, file in tables.items():
        _save(sink.frame(table), out / f"{file}.parquet")
    pd.DataFrame(ticks).to_parquet(out / "ticks.parquet", index=False)
    if feature_rows:
        pd.concat(feature_rows, ignore_index=True).to_parquet(out / "features.parquet", index=False)
    info = {
        "mode": "simulate",
        "split": args.split,
        "model": version,
        "sequences": model.sequence_shape is not None,
        "cur_dev_mode": args.cur_dev,
        "cur_dev_lag_s": args.cur_dev_lag,
        "alerts": {
            "confirm_ticks": args.confirm_ticks,
            "hysteresis_s": args.hysteresis,
            "clear_ticks": args.clear_ticks,
        },
        "window": [args.start, args.until],
        "events": core.events,
        "ticks": core.ticks_run,
        "wall_s": round(wall, 1),
        "engine": engine.stats(),
    }
    (out / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2, default=str))
    print(f"simulated {core.events} events, {core.ticks_run} ticks in {wall:.0f} s → {out}")
    return out


def _save(df: pd.DataFrame, path: Path) -> None:
    for col in df.columns:
        if df[col].map(lambda v: isinstance(v, dict | list)).any():
            df[col] = df[col].map(lambda v: json.dumps(v, ensure_ascii=False, default=str) if v else None)
    df.to_parquet(path, index=False)


# ---- export from the stack -------------------------------------------------------------------------------

EXPORT_QUERIES = {
    "predictions": "SELECT * FROM predictions",
    "updates": "SELECT * FROM prediction_updates",
    "passages": "SELECT * FROM stop_passages",
    "alerts": "SELECT * FROM alerts",
    "incidents": "SELECT id, kind, status, tr_id, related_tr_id, risk, cause, opened_at, closed_at "
    "FROM incidents",
}
EXPORT_HTTP = (
    ("predictor", 8002, "/api/predictor/stats"),
    ("predictor", 8002, "/metrics"),
    ("ml-service", 8003, "/model/info"),
    ("ml-service", 8003, "/metrics"),
)


def run_export(args: argparse.Namespace) -> Path:
    out = OUT / args.name
    out.mkdir(parents=True, exist_ok=True)
    compose = ["docker", "compose", *(["-p", args.project] if args.project else [])]
    psql = [*compose, "exec", "-T", "postgres", "psql", "-U", "foresight", "-d", "foresight", "-c"]
    for name, query in EXPORT_QUERIES.items():
        sql = f"\\copy ({query}) TO STDOUT WITH CSV HEADER"
        csv = subprocess.run([*psql, sql], check=True, capture_output=True, text=True).stdout
        (out / f"{name}.csv").write_text(csv, encoding="utf-8")
        print(f"{name}: {max(csv.count(chr(10)) - 1, 0)} rows")
    for service, port, path in EXPORT_HTTP:
        url = f"http://127.0.0.1:{port}{path}"
        code = f"import urllib.request; print(urllib.request.urlopen('{url}').read().decode())"
        cmd = [*compose, "exec", "-T", service, "python", "-c", code]
        res = subprocess.run(cmd, capture_output=True, text=True)
        (out / f"{service}{path.replace('/', '_')}.txt").write_text(res.stdout, encoding="utf-8")
    info = {"mode": "stack", "project": args.project, "split": args.split}
    (out / "info.json").write_text(json.dumps(info))
    print(f"exported → {out}")
    return out


# ---- evaluation ------------------------------------------------------------------------------------------


def _read(path: Path) -> pd.DataFrame:
    for suffix, reader in ((".parquet", pd.read_parquet), (".csv", pd.read_csv)):
        if path.with_suffix(suffix).exists():
            return reader(path.with_suffix(suffix))
    return pd.DataFrame()


def _naive(values: pd.Series) -> pd.Series:
    """Timestamps (aware or naive, text or datetime) → naive UTC datetime64[ns] (the dataset's convention)."""
    s = pd.to_datetime(values, utc=True, format="mixed")
    return s.dt.tz_localize(None).astype("datetime64[ns]")


TIME_COLUMNS = {
    "predictions": ("target_time_begin", "issued_at", "updated_at", "pass_time", "closed_at"),
    "updates": ("target_time_begin", "tick_at"),
    "passages": ("time_begin", "pass_time", "confirmed_at"),
    "alerts": ("target_time_begin", "issued_at", "closed_at"),
    "incidents": ("opened_at", "closed_at"),
}
RUN_TABLES = ("predictions", "updates", "passages", "alerts", "incidents", "ticks")


def _truthy(values: pd.Series) -> pd.Series:
    """Booleans of a column read from parquet (bool) or CSV of psql (``t`` / ``f``)."""
    return values.map(lambda v: str(v).lower() in ("true", "t", "1"))


def alert_quality(data: Mapping[str, pd.DataFrame], split: str) -> dict[str, Any]:
    """Точность и полнота алертов прогона (одинаково для алертов «на прогноз» и «на инцидент»).

    * точность — доля красных алертов, чей факт на целевой остановке алерта > 120 с (жёлтых — ≥ 60 с); факт —
      проход детектора (онлайн-разметка системы) и, для сверки, факт расписания датасета;
    * полнота — доля опозданий > 120 с (закрытые прогнозы), у ТС которых был красный (любой) уровень, пока
      остановка была в окне 10–15 мин: колонка ``alert_level`` прогноза; у прогона без неё (алерты на
      каждый прогноз) — у самого прогноза был красный (любой) алерт;
    * алертов на инцидент, заблаговременность алерта до прохода его остановки.
    """
    alerts, pred, pas = data["alerts"], data["predictions"], data["passages"]
    inc = data.get("incidents", pd.DataFrame())
    kind = alerts["kind"] if "kind" in alerts else pd.Series("delay", index=alerts.index)
    delay = alerts[kind == "delay"]
    matched = pas[_truthy(pas["matched"]) & pas["delay_s"].notna()] if len(pas) else pas
    det = {(int(r.tr_id), int(r.stop_id)): float(r.delay_s) for r in matched.itertuples()}
    passed = {(int(r.tr_id), int(r.stop_id)): r.pass_time for r in matched.itertuples()}
    sched = load_schedule(split)
    fact: dict[tuple[int, int], float] = {}
    if "time_fact_begin" in sched:
        s = sched.dropna(subset=["time_fact_begin"])
        delays = (s["time_fact_begin"] - s["time_begin"]).dt.total_seconds().to_numpy()
        fact = dict(zip(zip(s["tr_id"], s["tt_action_item_id"], strict=True), delays, strict=True))

    def precision(level: str, facts: Mapping[tuple[int, int], float]) -> tuple[float | None, int]:
        sub = delay[delay["level"] == level]
        vals = [facts.get((int(tr), int(st))) for tr, st in zip(sub["tr_id"], sub["stop_id"], strict=True)]
        known = np.array([v for v in vals if v is not None], dtype=np.float64)
        if not len(known):
            return None, 0
        hit = known > 120.0 if level == "red" else known >= 60.0
        return float(np.mean(hit)), int(len(known))

    res: dict[str, Any] = {
        "alerts_delay": int(len(delay)),
        "alerts_bunching": int(len(alerts) - len(delay)),
        "red": int((delay["level"] == "red").sum()),
        "yellow": int((delay["level"] == "yellow").sum()),
    }
    for name, facts in (("detector", det), ("fact", fact)):
        res[f"red_precision_{name}"], res[f"red_checked_{name}"] = precision("red", facts)
        res[f"yellow_precision_{name}"], _ = precision("yellow", facts)
    closed = pred[pred["status"] == "closed"]
    late = closed[closed["actual_delay_s"] > 120.0]
    res["late_stops"] = int(len(late))
    if "alert_level" in late and late["alert_level"].notna().any():
        red = late["alert_level"] == "red"
        anyl = late["alert_level"].isin(["yellow", "red"])
        res["recall_basis"] = "vehicle level while the stop was 10–15 min ahead"
    else:
        red = late["id"].isin(set(delay.loc[delay["level"] == "red", "prediction_id"]))
        anyl = late["id"].isin(set(delay["prediction_id"]))
        res["recall_basis"] = "an alert of the forecast itself"
    res["recall_red"] = float(red.mean()) if len(late) else None
    res["recall_any"] = float(anyl.mean()) if len(late) else None
    if len(inc) and "kind" in inc:
        n_inc = int((inc["kind"] == "delay").sum())
        res["delay_incidents"] = n_inc
        res["alerts_per_incident"] = round(len(delay) / n_inc, 2) if n_inc else None
    leads = []
    for tr, st, issued in zip(delay["tr_id"], delay["stop_id"], delay["issued_at"], strict=True):
        key = (int(tr), int(st))
        if key in passed:
            leads.append((passed[key] - issued).total_seconds())
    if leads:
        arr = np.asarray(leads)
        res["alert_lead_quantiles_s"] = {
            str(q): round(float(np.quantile(arr, q)), 1) for q in (0.0, 0.05, 0.5, 0.95)
        }
        res["alert_lead_ge_10min"] = float(np.mean(arr >= LEAD_MIN_S))
        res["alert_retroactive"] = int(np.sum(arr <= 0))
    return res


def run_sequences(run: Path, info: Mapping[str, Any]) -> bool:
    """Слал ли прогон последовательности в модель (simulate: ``info.json``; стек: статистика predictor)."""
    if "sequences" in info:
        return bool(info["sequences"])
    stats = run / "predictor_api_predictor_stats.txt"
    try:
        return json.loads(stats.read_text())["forecast"]["sequences_sent"] > 0
    except (OSError, ValueError, KeyError, TypeError):
        return True


def load_run(run: Path) -> dict[str, pd.DataFrame]:
    """Прогнозы, обновления, проходы, алерты, инциденты и тики прогона ``simulate`` или ``export``."""
    data = {name: _read(run / name) for name in RUN_TABLES}
    for name, cols in TIME_COLUMNS.items():
        df = data[name]
        for col in cols:
            if col in df.columns:
                df[col] = _naive(df[col])
    return data


def offline_features(split: str, analog: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Офлайн-признаки точек сплита: с ``cur_dev_s`` датасета и с онлайн-аналогом ``cur_dev_s``.

    Args:
        split: Сплит.
        analog: ``sample_id``, ``cur_dev_online`` — онлайн-аналог ``cur_dev_s`` точек.

    Returns:
        Таблица ``ml.dataset.build_split`` и признаки тех же точек с аналогом (с колонкой ``sample_id``).
    """
    from ml.dataset import build_context, build_split

    feats = build_split(split)
    pts = load_points(split).merge(analog, on="sample_id", how="inner")
    pts = pts.assign(cur_dev_s=pts["cur_dev_online"].to_numpy(dtype=np.float64))
    with_analog = build_features(pts, build_context(split))
    with_analog.insert(0, "sample_id", pts["sample_id"].to_numpy())
    return feats, with_analog


def compare_features(online: pd.DataFrame, offline: pd.DataFrame, tol: float = 0.5) -> list[dict[str, Any]]:
    """Признаки потока против офлайн на одних точках: доля расхождений (> ``tol``) и средний модуль.

    Args:
        online: ``sample_id`` + признаки, посчитанные движком на тике ``T``.
        offline: ``sample_id`` + офлайн-признаки тех же точек (с тем же ``cur_dev_s``).
        tol: Допуск (секунды / метры / доли).

    Returns:
        По признакам с расхождениями, по убыванию доли.
    """
    m = online.merge(offline, on="sample_id", suffixes=("_on", "_off"))
    rows = []
    for name in FEATURE_NAMES:
        a = m[f"{name}_on"].to_numpy(dtype=np.float64)
        b = m[f"{name}_off"].to_numpy(dtype=np.float64)
        nan_diff = np.isnan(a) != np.isnan(b)
        both = ~np.isnan(a) & ~np.isnan(b)
        diff = np.abs(a - b)
        bad = nan_diff | (both & (diff > tol))
        if bad.any():
            rows.append(
                {
                    "feature": name,
                    "share": round(float(bad.mean()), 3),
                    "nan_mismatch": int(nan_diff.sum()),
                    "mean_abs": round(float(diff[both].mean()), 2) if both.any() else None,
                }
            )
    return sorted(rows, key=lambda r: -r["share"])


def mae(a: Any, b: Any) -> float:
    return float(np.mean(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def histogram(values: np.ndarray, edges: Sequence[float]) -> list[tuple[str, int]]:
    """Гистограмма по минутам: ``[(интервал, число)]``."""
    counts, _ = np.histogram(values, bins=list(edges))
    return [(f"{edges[i] / 60:g}–{edges[i + 1] / 60:g}", int(c)) for i, c in enumerate(counts)]


def match_labels(upd: pd.DataFrame, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Точки разметки в окне прогона и их онлайн-прогнозы (тик = ``T``, та же целевая остановка)."""
    labels = load_points(split)
    lab = labels[(labels["T"] >= upd["tick_at"].min()) & (labels["T"] <= upd["tick_at"].max())].copy()
    online = upd.rename(
        columns={"tick_at": "T", "pred_delay_s": "online_pred", "cur_dev_s": "cur_dev_online"}
    )
    cols = ["tr_id", "T", "target_stop_id", "online_pred", "cur_dev_online", "source"]
    cols += [c for c in ("p10", "p90", "p_late") if c in online.columns]
    online = online[cols].drop_duplicates(["tr_id", "T", "target_stop_id"], keep="last")
    return lab, lab.merge(online, on=["tr_id", "T", "target_stop_id"], how="left")


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    run = OUT / args.name
    info = json.loads((run / "info.json").read_text()) if (run / "info.json").exists() else {}
    data = load_run(run)
    upd, pred = data["updates"], data["predictions"]
    lab, m = match_labels(upd, args.split)
    covered = m["online_pred"].notna()
    mm = m[covered].copy()
    y = mm["target_delay_s"].to_numpy(dtype=np.float64)
    res: dict[str, Any] = {
        "run": args.name,
        "info": info,
        "labels_in_window": int(len(lab)),
        "matched": int(covered.sum()),
        "coverage": float(covered.mean()) if len(m) else math.nan,
        "online_mae": mae(mm["online_pred"], y),
        "baseline_official_mae": mae(mm["cur_dev_s"], y),
        "baseline_online_mae": mae(mm["cur_dev_online"].fillna(0.0), y),
        "zero_mae": mae(0.0 * y, y),
        "fallback_share_matched": float((mm["source"] == "fallback").mean()) if len(mm) else math.nan,
        "cur_dev_online_nan_share": float(mm["cur_dev_online"].isna().mean()) if len(mm) else math.nan,
    }
    if {"p10", "p90", "p_late"} <= set(mm.columns) and mm["p10"].notna().any():
        q = mm.dropna(subset=["p10", "p90", "p_late"])
        yq = q["target_delay_s"].to_numpy(dtype=np.float64)
        res["interval_points"] = int(len(q))
        res["interval_coverage"] = float(np.mean((yq >= q["p10"]) & (yq <= q["p90"])))
        res["interval_mean_width_s"] = float(np.mean(q["p90"] - q["p10"]))
        late = (yq > 120.0).astype(np.float64)
        res["p_late_brier"] = float(np.mean((q["p_late"].to_numpy(dtype=np.float64) - late) ** 2))
        res["p_late_brier_climatology"] = float(np.mean((late.mean() - late) ** 2))
    if args.model:
        feats, with_analog = offline_features(args.split, mm[["sample_id", "cur_dev_online"]])
        use_seq = run_sequences(run, info)
        pred_off, pred_analog = offline_predict(args.model, args.split, [feats, with_analog], use_seq)
        res["offline_sequences"] = use_seq
        off = pd.DataFrame({"sample_id": feats["sample_id"].to_numpy(), "offline_pred": pred_off})
        extra = pd.DataFrame(
            {"sample_id": with_analog["sample_id"].to_numpy(), "offline_pred_analog": pred_analog}
        )
        off = off.merge(extra, on="sample_id", how="left")
        mo = mm.merge(off, on="sample_id")
        res["offline_model"] = args.model
        res["offline_mae"] = mae(mo["offline_pred"], mo["target_delay_s"])
        res["offline_analog_mae"] = mae(mo["offline_pred_analog"], mo["target_delay_s"])
        res["online_vs_offline_analog_median_abs_s"] = float(
            np.median(np.abs(mo["online_pred"] - mo["offline_pred_analog"]))
        )
        res["online_vs_offline_analog_p90_abs_s"] = float(
            np.quantile(np.abs(mo["online_pred"] - mo["offline_pred_analog"]), 0.9)
        )
        all_window = lab.merge(off[["sample_id", "offline_pred"]], on="sample_id")
        res["offline_mae_all_window"] = mae(all_window["offline_pred"], all_window["target_delay_s"])
        online_feats = _read(run / "features")
        if len(online_feats):
            online_feats["tick_at"] = _naive(online_feats["tick_at"])
            keys = mm[["sample_id", "tr_id", "T", "target_stop_id"]]
            on = keys.merge(
                online_feats.rename(columns={"tick_at": "T"}), on=["tr_id", "T", "target_stop_id"]
            ).drop(columns=["tr_id", "T", "target_stop_id"])
            res["feature_mismatch"] = compare_features(on, with_analog)[:20]
    res.update(stream_honesty(pred, upd))
    alerts = data["alerts"]
    if len(alerts):
        res["alerts"] = int(len(alerts))
        retro = alerts["retroactive"] if "retroactive" in alerts else pd.Series(dtype=object)
        res["alerts_retroactive"] = int(retro.map(lambda v: str(v).lower() in ("true", "t", "1")).sum())
        res["alert_quality"] = alert_quality(data, args.split)
    ticks = data["ticks"]
    if len(ticks):
        res["tick_ms"] = {
            col: {q: round(float(np.quantile(ticks[col], q)) * 1000, 1) for q in (0.5, 0.95, 0.99, 1.0)}
            for col in ("total_s", "detector_s", "features_s", "ml_s")
        }
        res["tick_targets_mean"] = float(ticks["targets"].mean())
        res["tick_active_mean"] = float(ticks["active"].mean())
    print_report(res)
    (run / "report.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str))
    return res


def stream_honesty(pred: pd.DataFrame, upd: pd.DataFrame) -> dict[str, Any]:
    """Горизонт и честность на всех прогнозах потока, закрытых фактом детектора."""
    res: dict[str, Any] = {"forecasts": int(len(pred))}
    if not len(pred):
        return res
    res["status_counts"] = {str(k): int(v) for k, v in pred["status"].value_counts().items()}
    lead_plan = (pred["target_time_begin"] - pred["issued_at"]).dt.total_seconds().to_numpy()
    res["lead_plan_range_s"] = [float(lead_plan.min()), float(lead_plan.max())]
    res["lead_plan_outside"] = int(np.sum((lead_plan <= LEAD_MIN_S) | (lead_plan > LEAD_MAX_S)))
    res["lead_plan_hist"] = histogram(lead_plan, [600, 660, 720, 780, 840, 870, 900])
    closed = pred[pred["status"] == "closed"]
    if len(closed):
        lead = (closed["pass_time"] - closed["issued_at"]).dt.total_seconds().to_numpy()
        res["retroactive"] = int(np.sum(lead <= 0))
        res["actual_lead_quantiles_s"] = {
            str(q): round(float(np.quantile(lead, q)), 1) for q in (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)
        }
        res["actual_lead_hist"] = histogram(lead, [-1e9, 0, 300, 600, 720, 840, 900, 1080, 1260, 1e9])
        res["share_lead_10_15"] = float(np.mean((lead >= LEAD_MIN_S) & (lead <= LEAD_MAX_S)))
        res["share_lead_ge_10"] = float(np.mean(lead >= LEAD_MIN_S))
        res["online_mae_detector"] = mae(closed["pred_delay_s"], closed["actual_delay_s"])
        res["online_first_mae_detector"] = mae(closed["first_pred_delay_s"], closed["actual_delay_s"])
        base = closed["cur_dev_s"].fillna(0.0) if "cur_dev_s" in closed else 0.0 * closed["actual_delay_s"]
        res["baseline_online_mae_detector"] = mae(base, closed["actual_delay_s"])
    if "source" in upd and len(upd):
        res["fallback_share"] = float((upd["source"] == "fallback").mean())
        res["updates"] = int(len(upd))
    return res


def _f(x: Any) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.1f}"


def print_report(res: Mapping[str, Any]) -> None:
    """Markdown-сводка прогона."""
    lines = [
        f"### {res['run']}",
        "",
        f"- точек разметки в окне: {res['labels_in_window']}, с онлайн-прогнозом на тике `T`: "
        f"{res['matched']} ({100 * res['coverage']:.1f}%)",
        f"- онлайн-MAE (модель на потоке): **{_f(res['online_mae'])}** с",
    ]
    if "offline_mae" in res:
        lines += [
            f"- та же модель офлайн на тех же точках: с `cur_dev_s` датасета {_f(res['offline_mae'])} с, "
            f"с онлайн-аналогом `cur_dev_s` {_f(res['offline_analog_mae'])} с; "
            f"|онлайн − офлайн-аналог|: медиана {_f(res['online_vs_offline_analog_median_abs_s'])} с, "
            f"p90 {_f(res['online_vs_offline_analog_p90_abs_s'])} с",
        ]
    lines += [
        f"- baseline `cur_dev_s` датасета: {_f(res['baseline_official_mae'])} с; baseline онлайн-аналог: "
        f"{_f(res['baseline_online_mae'])} с; ноль: {_f(res['zero_mae'])} с",
        f"- прогнозов: {res['forecasts']}, статусы: {res.get('status_counts')}",
        f"- lead «план − выдача»: {res.get('lead_plan_range_s')} с, "
        f"вне (600, 900]: {res.get('lead_plan_outside')}",
    ]
    if "retroactive" in res:
        lines += [
            f"- задним числом (выдан в момент прохода или позже): **{res['retroactive']}**; "
            f"lead «проход − выдача», квантили: {res['actual_lead_quantiles_s']}",
            f"- онлайн-MAE по факту детектора: последнее значение {_f(res['online_mae_detector'])} с, "
            f"первая выдача {_f(res['online_first_mae_detector'])} с, "
            f"baseline {_f(res['baseline_online_mae_detector'])} с",
        ]
    if "fallback_share" in res:
        lines.append(f"- доля fallback: {100 * res['fallback_share']:.2f}% из {res['updates']} значений")
    if "alerts" in res:
        lines.append(f"- алертов: {res['alerts']}, задним числом: {res['alerts_retroactive']}")
    if "tick_ms" in res:
        lines.append(
            f"- тик, мс (p50/p95/p99/max): {res['tick_ms']}; активных ТС {_f(res['tick_active_mean'])}, "
            f"точек прогноза на тик {_f(res['tick_targets_mean'])}"
        )
    if res.get("feature_mismatch"):
        lines.append("- признаки потока ≠ офлайн (с тем же `cur_dev_s`), доля точек:")
        lines += [f"  - `{r['feature']}`: {r}" for r in res["feature_mismatch"][:12]]
    print("\n" + "\n".join(lines))


# ---- detector ----------------------------------------------------------------------------------------------


def compare_detector(args: argparse.Namespace) -> dict[str, Any]:
    from shared.stops import detect_all

    run = OUT / args.name
    online = load_run(run)["passages"]
    schedule = load_schedule(args.split)
    plan = schedule.drop(columns=["time_fact_begin", "manual_fill"], errors="ignore")
    offline = detect_all(load_traffic(args.split), plan).rename(columns={"tt_action_item_id": "stop_id"})
    offline = offline[offline["confirmed_at"].notna()]
    m = online.merge(offline, on=["tr_id", "stop_id"], suffixes=("_on", "_off"), how="inner")
    both = m["pass_time_on"].notna() & m["pass_time_off"].notna()
    diff = (m.loc[both, "pass_time_on"] - m.loc[both, "pass_time_off"]).dt.total_seconds().abs()
    res: dict[str, Any] = {
        "online_decisions": int(len(online)),
        "offline_decisions": int(len(offline)),
        "compared": int(len(m)),
        "same_decision": float((m["pass_time_on"].notna() == m["pass_time_off"].notna()).mean()),
        "both_matched": int(both.sum()),
        "pass_time_within_1s": float((diff <= 1).mean()) if len(diff) else math.nan,
        "pass_time_abs_diff_p50_p90_p99_s": [round(float(np.quantile(diff, q)), 1) for q in (0.5, 0.9, 0.99)]
        if len(diff)
        else [],
        "confirmed_after_pass_median_s": float(
            (online["confirmed_at"] - online["pass_time"]).dt.total_seconds().median()
        ),
    }
    if "time_fact_begin" in schedule:
        fact = schedule[["tr_id", "tt_action_item_id", "time_fact_begin"]]
        fact = fact.rename(columns={"tt_action_item_id": "stop_id"})
        mf = online.merge(fact, on=["tr_id", "stop_id"]).dropna(subset=["pass_time", "time_fact_begin"])
        err = (mf["pass_time"] - mf["time_fact_begin"]).dt.total_seconds()
        res["online_vs_fact_median_abs_s"] = float(err.abs().median())
        res["online_vs_fact_p10_p90_s"] = [float(err.quantile(0.1)), float(err.quantile(0.9))]
    print(json.dumps(res, ensure_ascii=False, indent=2))
    (run / "detector.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    return res


# ---- online analogue of cur_dev_s ------------------------------------------------------------------------

ANALOG_LAGS = (0, 60, 120, 180, 300)


def analog_candidates(ctx: Any, tr_id: int, t: float) -> dict[str, float]:
    """Кандидаты онлайн-аналога ``cur_dev_s`` на момент ``t`` по проходам детектора (``confirmed_at ≤ t``)."""
    from shared.stops import MODE_MIN, _stop_modes

    v = ctx.vehicles.get(int(tr_id))
    out = {f"last_lag{lag}": math.nan for lag in ANALOG_LAGS}
    out.update(last_regular=math.nan, median3=math.nan, plan_last=math.nan)
    if v is None:
        return out
    matched = np.isfinite(v.pass_s) & np.isfinite(v.conf_s)
    # plan_last: the definition of the hint (the last stop planned by t) made causal — its delay if the
    # detector confirmed the pass by t, otherwise at least t − plan (and not less than the last delay)
    valid = np.flatnonzero(np.isfinite(v.slon) & np.isfinite(v.slat) & (v.tb <= t))
    if len(valid):
        k = valid[-1]
        known = np.flatnonzero(matched & (v.conf_s <= t))
        last = float(v.pass_s[known[-1]] - v.tb[known[-1]]) if len(known) else math.nan
        if v.conf_s[k] <= t:  # decided by t (NaN compares False)
            out["plan_last"] = float(v.pass_s[k] - v.tb[k]) if np.isfinite(v.pass_s[k]) else last
        else:
            out["plan_last"] = t - v.tb[k] if math.isnan(last) else max(t - v.tb[k], last)
    for lag in ANALOG_LAGS:
        idx = np.flatnonzero(matched & (v.conf_s <= t - lag))
        if len(idx):
            last = idx[np.lexsort((idx, v.conf_s[idx]))[-1]]  # the latest decision (then the latest stop)
            out[f"last_lag{lag}"] = float(v.pass_s[last] - v.tb[last])
    idx = np.flatnonzero(matched & (v.conf_s <= t))
    if len(idx):
        out["median3"] = float(np.median((v.pass_s - v.tb)[idx[-3:]]))
        valid = np.isfinite(v.slon) & np.isfinite(v.slat)
        modes = np.full(len(v.tb), -1)
        modes[valid] = _stop_modes(v.tb[valid], 300.0)
        regular = idx[modes[idx] == MODE_MIN]
        if len(regular):
            out["last_regular"] = float(v.pass_s[regular[-1]] - v.tb[regular[-1]])
    return out


def analog_study(args: argparse.Namespace) -> dict[str, Any]:
    """Какой онлайн-аналог ближе к ``cur_dev_s`` датасета: выбор на train (реальные ТС), отчёт на test."""
    from ml.dataset import build_context, real_vehicle_ids
    from shared.features import to_seconds

    res: dict[str, Any] = {}
    for split in ("train", "test"):
        pts = load_points(split)
        if split == "train":
            pts = pts[pts["tr_id"].isin(real_vehicle_ids())].reset_index(drop=True)
        ctx = build_context(split)
        ts = to_seconds(pts["T"])
        cand = pd.DataFrame([analog_candidates(ctx, tr, t) for tr, t in zip(pts["tr_id"], ts, strict=True)])
        official = pts["cur_dev_s"].to_numpy(dtype=np.float64)
        target = pts["target_delay_s"].to_numpy(dtype=np.float64)
        rows = {}
        for col in cand.columns:
            x = cand[col].to_numpy(dtype=np.float64)
            ok = np.isfinite(x)
            rows[col] = {
                "nan_share": round(float(1 - ok.mean()), 3),
                "mae_vs_cur_dev_s": round(mae(x[ok], official[ok]), 1),
                "equal_within_5s": round(float(np.mean(np.abs(x[ok] - official[ok]) <= 5)), 3),
                "mae_vs_target": round(mae(np.nan_to_num(x), target), 1),
            }
        res[split] = {"points": int(len(pts)), "cur_dev_s_mae_vs_target": round(mae(official, target), 1)}
        res[split]["hint"] = hint_definition(split, pts)
        res[split]["candidates"] = rows
    res["closest_to_hint_on_train"] = min(
        res["train"]["candidates"], key=lambda c: res["train"]["candidates"][c]["mae_vs_cur_dev_s"]
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return res


def hint_definition(split: str, pts: pd.DataFrame) -> dict[str, Any]:
    """Что такое ``cur_dev_s`` датасета (офлайн-проверка по факту расписания, только для анализа).

    Сверяет подсказку с ``time_fact_begin − time_begin`` последней остановки, **плановое** время которой
    ``≤ T``, и считает, у скольких точек факт этой остановки наступил **после** ``T`` (подсказка несёт
    информацию из будущего, на потоке её нет).
    """
    sch = load_schedule(split).sort_values(["tr_id", "time_begin", "tt_action_item_id"], kind="stable")
    by_tr = {int(k): g for k, g in sch.groupby("tr_id")}
    equal, future, err_future, err_past = [], [], [], []
    for tr, t, hint, target in zip(
        pts["tr_id"], pts["T"], pts["cur_dev_s"], pts["target_delay_s"], strict=True
    ):
        rows = by_tr.get(int(tr))
        planned = rows[rows["time_begin"] <= t] if rows is not None else None
        if planned is None or not len(planned) or pd.isna(planned["time_fact_begin"].iloc[-1]):
            continue
        last = planned.iloc[-1]
        dev = (last["time_fact_begin"] - last["time_begin"]).total_seconds()
        equal.append(abs(dev - hint) <= 1)
        after = last["time_fact_begin"] > t
        future.append(after)
        (err_future if after else err_past).append(abs(hint - target))
    return {
        "equal_fact_at_last_planned_stop": round(float(np.mean(equal)), 3),
        "fact_after_T_share": round(float(np.mean(future)), 3),
        "hint_mae_when_fact_after_T": round(float(np.mean(err_future)), 1) if err_future else None,
        "hint_mae_when_fact_before_T": round(float(np.mean(err_past)), 1) if err_past else None,
    }


# ---- fallback coefficient ----------------------------------------------------------------------------------


FALLBACK_BASES = ("dev_1", "dev_med5", "cur_best", "phys", "phys_own", "pos_delay")
"""Candidate deviations of the fallback formula (online features; ``cur_dev_s`` of the dataset is not)."""
MAX_FALLBACK_NAN = 0.05


def fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """MAE-optimal ``y ≈ a + k·x``: grid over ``k`` ∈ [0, 1.5], ``a`` = median of the residual."""
    best = (math.inf, 0.0, 0.0)
    for k in np.arange(0.0, 1.501, 0.01):
        a = float(np.median(y - k * x))
        err = float(np.mean(np.abs(y - a - k * x)))
        if err < best[0]:
            best = (err, float(k), a)
    return best


def fallback_coef(args: argparse.Namespace) -> dict[str, Any]:
    from ml.dataset import TARGET, build_split, leak_mask

    train_all = build_split("train")
    leak = leak_mask(train_all, [load_points("test"), load_points("validate")])
    train = train_all[~leak & train_all["is_real"].to_numpy()].reset_index(drop=True)
    test = build_split("test")
    y, yt = train[TARGET].to_numpy(dtype=np.float64), test[TARGET].to_numpy(dtype=np.float64)
    rows = {}
    for base in FALLBACK_BASES:
        x = np.nan_to_num(train[base].to_numpy(dtype=np.float64))
        xt = np.nan_to_num(test[base].to_numpy(dtype=np.float64))
        err, k, a = fit_line(x, y)
        rows[base] = {
            "coef": round(k, 2),
            "intercept_s": round(a, 1),
            "train_mae": round(err, 2),
            "test_mae": round(mae(a + k * xt, yt), 2),
            "test_mae_raw": round(mae(xt, yt), 2),
            "nan_share_train": round(float(train[base].isna().mean()), 3),
            "nan_share_test": round(float(test[base].isna().mean()), 3),
        }
    # chosen on train only, among the deviations known almost always (a missing one leaves the intercept)
    usable = [b for b in rows if rows[b]["nan_share_train"] <= MAX_FALLBACK_NAN]
    best = min(usable, key=lambda b: rows[b]["train_mae"])
    res = {
        "chosen": best,
        **rows[best],
        "train_rows": int(len(train)),
        "test_mae_zero": round(mae(0.0 * yt, yt), 2),
        "test_mae_cur_dev_s": round(mae(test["cur_dev_s"], yt), 2),
        "candidates": rows,
    }
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return res


# ---- holdout versions of the registry --------------------------------------------------------------------


def _test_metrics(bundle: Any, test: pd.DataFrame, seq: np.ndarray | None) -> dict[str, float]:
    """Метрики версии на test (модели не видели test): MAE, покрытие P10–P90, Brier ``p_late``."""
    from ml.dataset import TARGET

    y = test[TARGET].to_numpy(dtype=np.float64)
    out = bundle.predict(test, sequences=seq)
    res = {"test_mae": round(mae(out["pred_delay_s"], y), 6)}
    if bundle.sequence_shape is not None:
        res["test_mae_without_sequences"] = round(mae(bundle.predict(test)["pred_delay_s"], y), 6)
    if out["p10"].notna().all():
        res["interval_coverage_test"] = round(float(np.mean((y >= out["p10"]) & (y <= out["p90"]))), 6)
    if out["p_late"].notna().all():
        late = (y > bundle.late_threshold_s).astype(np.float64)
        res["p_late_brier_test"] = round(float(np.mean((out["p_late"].to_numpy() - late) ** 2)), 6)
    return res


def pack_holdout(args: argparse.Namespace) -> None:
    """Версии ``<base>-holdout``: тот же рецепт, что у ``<base>``, но модели обучены только на train.

    Нужны для честной онлайн-оценки на потоке test (стенд проигрывает test, а боевые v1 / v2 обучены на
    train + test). CatBoost задержки — :func:`holdout_members` (ансамбль v1 на train без синтетических копий
    периодов оценки, как в ``ml/train.py``); квантили и классификатор опоздания — ``ml.train_v2`` на том же
    train; последовательная модель — экспорт ONNX, обученный только на train (``ml/train_v2.py``, стадия
    final: ``<best>_train.*.onnx``); калибровка ``p_late``, вес ансамбля и поправка интервала — из манифеста
    базы (подобраны по OOF train); ожидаемая ошибка — ``abs_error_k · (p90 − p10)`` (член ``abs_error`` базы
    обучен на ошибках test, в holdout его нет). Метрики test пересчитываются на собранной версии.
    """
    from ml.inference import load_bundle
    from ml.registry import git_head, utc_iso, write_bundle
    from ml.train_v2 import fit_late, fit_quantiles, load_data

    data = load_data()
    root = Path(args.root) if args.root else MODELS
    stage = OUT / "holdout_pack"
    stage.mkdir(parents=True, exist_ok=True)
    delay_models = holdout_members()
    extra: dict[str, Any] = {}
    for base in args.bases.split(","):
        tmpl = json.loads((root / base / "manifest.json").read_text(encoding="utf-8"))
        version = f"{base}-holdout"
        files: dict[str, Path] = {}
        members: list[dict[str, Any]] = []
        delay_items = [m for m in tmpl["members"] if m["component"] == "catboost" and m["target"] == "delay"]
        if len(delay_items) != len(delay_models):
            raise SystemExit(
                f"{base}: {len(delay_items)} CatBoost delay members, holdout has {len(delay_models)}"
            )
        for item, (model, cfg) in zip(delay_items, delay_models, strict=True):
            if item.get("base") != cfg.base:
                raise SystemExit(f"{base}: member {item['file']} base {item.get('base')} != {cfg.base}")
            path = stage / f"{version}_{item['file']}"
            model.save_model(str(path))
            files[item["file"]] = path
            members.append(dict(item))
        for item in tmpl["members"]:
            comp, target = item["component"], item["target"]
            if comp == "catboost" and target in ("quantiles", "p_late"):
                if target not in extra:
                    print(f"training {target} on train ({len(data.train)} rows)", flush=True)
                    fit = fit_quantiles if target == "quantiles" else fit_late
                    extra[target] = fit(data.train)
                path = stage / f"{version}_{item['file']}"
                extra[target].save_model(str(path))
                files[item["file"]] = path
                members.append(dict(item))
            elif comp != "catboost":  # the sequence model: its train-only export
                name = str(item["name"])
                files[item["file"]] = Path(args.seq_onnx) / f"{name}_train.int8.onnx"
                if item.get("fp32_file"):
                    files[item["fp32_file"]] = Path(args.seq_onnx) / f"{name}_train.fp32.onnx"
                members.append(dict(item))
        for path in files.values():
            if not path.is_file():
                raise SystemExit(f"{version}: missing {path}")
        metrics = {k: v for k, v in tmpl.get("metrics", {}).items() if not k.startswith("test_")}
        drop = ("interval_coverage_test", "p_late_brier_test", "platform_score")  # the score is of the base
        metrics = {k: v for k, v in metrics.items() if k not in drop}
        metrics = {k: v for k, v in metrics.items() if not k.startswith(("p_late_ece_test", "expected_abs"))}
        manifest = {k: v for k, v in tmpl.items() if k not in ("format", "files")}
        manifest.update(
            version=version,
            created_at=utc_iso(),
            packed_at=utc_iso(),
            packed_commit=git_head(),
            description=f"{tmpl.get('description', '')} — HOLDOUT: все модели обучены только на train "
            "(test не видели); для онлайн-оценки на потоке test. Модель сабмита — "
            f"{base} (train + test).",
            members=members,
            metrics=metrics,
            train={
                "rows": int(len(data.train)),
                "dropped_synthetic_rows": data.dropped,
                "holdout": True,
                "base": base,
                "note": "train only (ml.dataset.leak_mask removes synthetic copies of test/validate periods)",
            },
        )
        write_bundle(root / version, manifest, files, overwrite=True)
        bundle = load_bundle(root / version)
        seq = data.seq_test if bundle.sequence_shape is not None else None
        manifest["metrics"] = {**metrics, **_test_metrics(bundle, data.test, seq)}
        write_bundle(root / version, manifest, {n: root / version / n for n in files}, overwrite=True)
        bundle = load_bundle(root / version)
        print(f"packed {bundle!r}: {json.dumps(bundle.metrics, ensure_ascii=False)}", flush=True)


def alerts_report(args: argparse.Namespace) -> dict[str, Any]:
    """Качество алертов готовых прогонов (для сравнения «до / после»)."""
    out = {}
    for name in args.names.split(","):
        data = load_run(OUT / name)
        out[name] = alert_quality(data, args.split)
        info = OUT / name / "info.json"
        if info.exists():
            out[name]["info"] = json.loads(info.read_text()).get("alerts")
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    return out


# ---- CLI -------------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("simulate", help="движок predictor по телеметрии сплита, модель в процессе")
    p.add_argument("--name", required=True, help="имя прогона: artifacts/online/<name>")
    p.add_argument("--split", default="test")
    p.add_argument(
        "--model", default="v2-holdout", help="версия реестра (v2-holdout, v1), holdout или fallback"
    )
    p.add_argument(
        "--sequences",
        action="store_true",
        help="передавать последовательности (CatBoost + GRU; как FORESIGHT_ML_SEQUENCES=1 в стеке)",
    )
    p.add_argument("--cur-dev", default="median3", choices=CUR_DEV_MODES)
    p.add_argument("--cur-dev-lag", type=float, default=0.0)
    p.add_argument("--confirm-ticks", type=int, default=Settings().alert_confirm_ticks)
    p.add_argument("--hysteresis", type=float, default=Settings().alert_hysteresis_s)
    p.add_argument("--clear-ticks", type=int, default=Settings().incident_clear_ticks)
    p.add_argument("--start", default=None, help="окно по receive_time, HH:MM")
    p.add_argument("--until", default=None)

    p = sub.add_parser("export", help="прогнозы из PostgreSQL работающего стека")
    p.add_argument("--name", required=True)
    p.add_argument("--project", default="", help="compose-проект (-p)")
    p.add_argument("--split", default="test")

    p = sub.add_parser("evaluate", help="онлайн-прогнозы против разметки и офлайн-модели")
    p.add_argument("--name", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--model", default="", help="модель для офлайн-сравнения (v1, holdout)")

    p = sub.add_parser("detector", help="онлайн-детектор против офлайн detect_all и факта")
    p.add_argument("--name", required=True)
    p.add_argument("--split", default="test")

    sub.add_parser("fallback-coef", help="коэффициенты fallback-формулы на train")
    sub.add_parser("analog", help="онлайн-аналог cur_dev_s: кандидаты по детектору против cur_dev_s датасета")

    p = sub.add_parser("pack-holdout", help="версии <base>-holdout: модели только на train → models/")
    p.add_argument("--bases", default="v1,v2")
    p.add_argument("--root", default="", help="каталог моделей (по умолчанию models/ репозитория)")
    p.add_argument("--seq-onnx", default=str(SEQ_ONNX_TRAIN), help="train-only ONNX последовательной модели")

    p = sub.add_parser("alerts", help="точность и полнота алертов готовых прогонов")
    p.add_argument("--names", required=True, help="прогоны через запятую")
    p.add_argument("--split", default="test")

    args = ap.parse_args(argv)
    if args.cmd == "simulate":
        asyncio.run(run_simulation(args))
    elif args.cmd == "export":
        run_export(args)
    elif args.cmd == "evaluate":
        evaluate(args)
    elif args.cmd == "detector":
        compare_detector(args)
    elif args.cmd == "analog":
        analog_study(args)
    elif args.cmd == "pack-holdout":
        pack_holdout(args)
    elif args.cmd == "alerts":
        alerts_report(args)
    else:
        fallback_coef(args)


if __name__ == "__main__":
    main()
