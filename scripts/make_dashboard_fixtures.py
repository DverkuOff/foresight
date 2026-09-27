"""Фикстуры mock-режима дашборда Foresight из настоящих данных ``dataset/test``.

Скрипт строит ``dashboard/src/mocks/fixtures/world.json`` и ``honesty.json``:

* маршруты — ТС с одинаковым набором остановок объединяются в маршрут (контракт, §1);
  последовательность остановок — самый частый рейс между отстоями (разрыв плана > 5 мин),
  рейсы «туда» и «обратно» склеиваются в круг; линия — по дорогам (кэш геометрии ``shared.routes``);
* треки ТС — валидные GPS-точки ``traffic.csv`` вокруг момента (по умолчанию 07:30), шаг ~15 с;
* расписание ТС — план и **факт** (``time_fact_begin``) остановок: для моков фронта это допустимо,
  это только демо-данные (в живом режиме факт восстанавливает детектор, прогнозы даёт модель);
* демо-плотность — к каждому реальному ТС добавляются «клоны» с тем же треком со сдвигом по времени
  (``clone: true``, tr_id 8000001+); на одном маршруте — пара «сбивка» (лидер опаздывает, ведомый догоняет);
* ``honesty.json`` — закрытые прогнозы по точкам ``labels/labels_test.csv`` до момента: факт и
  baseline ``cur_dev_s`` настоящие, прогноз синтетический (откалиброван под CV-оценку модели v1).

Запуск (на сервере, из корня репозитория)::

    uv run python scripts/make_dashboard_fixtures.py
    uv run python scripts/make_dashboard_fixtures.py --moment 08:15 --window-min 60

Время датасета наивное и трактуется как UTC (как во всём проекте).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shared.data import load_traffic  # noqa: E402
from shared.routes import RouteMap, load_segments  # noqa: E402

DAY = "2026-01-06"
LAYOVER_GAP_S = 300
THIN_S = 15
CLONE_OFFSETS_S = (-600, 600, 1200)
# Цвета линий маршрутов: сине-фиолетовая гамма, не пересекается с цветами риска (зелёный / жёлтый / красный).
ROUTE_COLORS = [
    "#4C8DF6",
    "#8B7CF6",
    "#2BB5C8",
    "#C07CF0",
    "#5E9FD8",
    "#7A8CFF",
    "#48A6A6",
    "#A58BFF",
    "#3E7CB1",
    "#9DA7FF",
    "#6FB7E0",
    "#B38CD9",
    "#5C7CFA",
]
CLONE_TR_BASE = 8_000_000
CLONE_UNIT_BASE = 8_100_000
SYNTH_PULL = 0.5
# (d_track, d_plan), с: лидер опаздывает на 6 мин и идёт в 4 мин перед реальным ТС (сбивка при
# плановом интервале 10 мин), обычный клон через 20 мин, клон на 10 мин позади и замыкающий с опозданием
# 5 мин (разрыв интервала).
BUNCHING_CLONES = ((240, 600), (1200, 1200), (-600, -600), (-1500, -1200))
SYNTH_NOISE_S = 90.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--moment", default="07:30", help="момент начала симуляции (HH:MM, время потока)")
    parser.add_argument("--window-min", type=int, default=60, help="длительность симуляции, мин")
    parser.add_argument("--history-min", type=int, default=45, help="история до момента, мин")
    parser.add_argument("--out", default=str(ROOT / "dashboard" / "src" / "mocks" / "fixtures"))
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args(argv)


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    r = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def load_schedule(moment: pd.Timestamp) -> pd.DataFrame:
    s = pd.read_csv(ROOT / "dataset" / "test" / "schedule.csv")
    s["plan"] = pd.to_datetime(s["time_begin"])
    s["fact"] = pd.to_datetime(s["time_fact_begin"])
    xy = s["geom"].str.extract(r"POINT\s*\(\s*(\S+)\s+(\S+)\s*\)").astype(float)
    s["lon"], s["lat"] = xy[0], xy[1]
    s["stop_key"] = s["lon"].round(5).map("{:.5f}".format) + "," + s["lat"].round(5).map("{:.5f}".format)
    s["name"] = s["building_address"].fillna("").astype(str).str.strip()
    s["plan_s"] = (s["plan"] - moment).dt.total_seconds().astype(int)
    s["fact_s"] = (s["fact"] - moment).dt.total_seconds()
    return s.sort_values(["tr_id", "plan", "tt_action_item_id"]).reset_index(drop=True)


def split_trips(rows: pd.DataFrame) -> list[pd.DataFrame]:
    gap = rows["plan"].diff().dt.total_seconds().fillna(1e9).to_numpy()
    trip = np.cumsum(gap > LAYOVER_GAP_S)
    return [g for _, g in rows.groupby(trip, sort=True)]


def route_sequence(rows: pd.DataFrame) -> list[str]:
    """Последовательность stop_key маршрута: самый частый рейс (+ обратный, если он не круговой)."""
    trips = [tuple(t["stop_key"]) for t in split_trips(rows) if len(t) > 3]
    counts = Counter(trips)
    main, _ = counts.most_common(1)[0]
    if main[0] == main[-1]:
        return list(main)
    back = [t for t, _ in counts.most_common() if t[0] == main[-1] and t != main]
    if back:
        return list(main) + list(back[0][1:])
    return list(main)


def assign_seq(rows: pd.DataFrame, seq_keys: list[str]) -> np.ndarray:
    """Порядковый номер остановки в маршруте для каждой строки плана (последовательное сопоставление)."""
    out = np.full(len(rows), -1, dtype=int)
    for trip in split_trips(rows):
        pos = None
        for i, key in zip(trip.index, trip["stop_key"], strict=True):
            candidates = [j for j, k in enumerate(seq_keys) if k == key]
            if not candidates:
                continue
            if pos is None:
                pos = candidates[0]
            else:
                ahead = [j for j in candidates if j > pos]
                pos = ahead[0] if ahead else candidates[0]
            out[rows.index.get_loc(i)] = pos
    return out


def stop_names(schedule: pd.DataFrame) -> dict[str, str]:
    named = schedule[schedule["name"] != ""]
    return named.groupby("stop_key")["name"].agg(lambda x: x.mode().iloc[0]).to_dict()


ROAD_MAP = RouteMap(segments=load_segments())
"""Кэш геометрии участков репозитория (backend/assets/route_segments.json)."""


def build_routes(schedule: pd.DataFrame) -> tuple[list[dict], dict[int, dict]]:
    names = stop_names(schedule)
    coords = schedule.groupby("stop_key")[["lon", "lat"]].first()
    groups: dict[frozenset, list[int]] = {}
    for tr_id, rows in schedule.groupby("tr_id"):
        groups.setdefault(frozenset(rows["stop_key"]), []).append(int(tr_id))
    routes: list[dict] = []
    by_tr: dict[int, dict] = {}
    for idx, (_, tr_ids) in enumerate(sorted(groups.items(), key=lambda kv: min(kv[1]))):
        route_id = f"R{idx + 1}"
        rows = schedule[schedule["tr_id"] == tr_ids[0]]
        seq_keys = route_sequence(rows)
        stops = []
        for seq, key in enumerate(seq_keys):
            lon, lat = coords.loc[key, "lon"], coords.loc[key, "lat"]
            stops.append(
                {
                    "stop_key": key,
                    "name": names.get(key) or f"Остановка №{seq + 1}",
                    "lat": round(float(lat), 6),
                    "lon": round(float(lon), 6),
                    "seq": seq,
                }
            )
        first = stops[0]
        if seq_keys[0] == seq_keys[-1]:
            far = max(stops, key=lambda s: haversine_m(first["lon"], first["lat"], s["lon"], s["lat"]))
        else:
            far = stops[-1]
        route = {
            "route_id": route_id,
            "name": f"Маршрут {route_id}: {first['name']} — {far['name']}",
            "tr_ids": sorted(tr_ids),
            "stops": stops,
            # та же дорожная геометрия, что в живом режиме: участки по GPS-трекам и OSRM (shared.routes)
            "line": ROAD_MAP.path(seq_keys),
            "color": ROUTE_COLORS[idx % len(ROUTE_COLORS)],
        }
        routes.append(route)
        for tr in tr_ids:
            by_tr[tr] = {"route": route, "seq_keys": seq_keys}
    return routes, by_tr


def thin_track(track: pd.DataFrame, moment: pd.Timestamp) -> list[list[float]]:
    out: list[list[float]] = []
    last_t = -1e18
    for t, lon, lat, heading, speed in zip(
        track["event_time"], track["lon"], track["lat"], track["heading"], track["speed"], strict=True
    ):
        ts = (t - moment).total_seconds()
        if ts - last_t < THIN_S:
            continue
        last_t = ts
        course = 0 if pd.isna(heading) else int(round(heading)) % 360
        spd = 0 if pd.isna(speed) else int(round(speed))
        out.append([int(round(ts)), round(float(lon), 5), round(float(lat), 5), course, spd])
    return out


def p_late_of(pred: float, sigma: float) -> float:
    return 0.5 * (1.0 + math.erf((pred - 120.0) / (sigma * math.sqrt(2.0))))


def risk_of(pred: float, p_late: float) -> str:
    if pred > 120 or p_late > 0.6:
        return "red"
    if pred < 60 and p_late < 0.3:
        return "green"
    return "yellow"


def build_honesty(
    moment: pd.Timestamp,
    schedule: pd.DataFrame,
    by_tr: dict[int, dict],
    units: dict[int, int],
    seed: int,
) -> dict:
    labels = pd.read_csv(ROOT / "dataset" / "labels" / "labels_test.csv")
    labels["T"] = pd.to_datetime(labels["T"])
    labels["target_time_begin"] = pd.to_datetime(labels["target_time_begin"])
    # Синтетический прогноз, откалиброванный под честную оценку модели v1 на кросс-валидации (MAE ≈ 77 с).
    # Сохранённая модель v1 обучена в том числе на test, поэтому её прогнозы по test были бы оптимистичны.
    rng = np.random.default_rng(seed)
    actual = labels["target_delay_s"].to_numpy(dtype=float)
    base = labels["cur_dev_s"].to_numpy(dtype=float)
    pred = base + SYNTH_PULL * (actual - base) + rng.normal(0.0, SYNTH_NOISE_S, len(labels))
    labels["pred"] = pred
    source = f"synthetic: cur_dev_s + {SYNTH_PULL} · (факт − cur_dev_s) + N(0, {SYNTH_NOISE_S:.0f} с)"
    names = schedule.set_index("tt_action_item_id")["name"].to_dict()
    closed_at = labels["target_time_begin"] + pd.to_timedelta(labels["target_delay_s"], unit="s")
    labels = labels[closed_at <= moment].copy()
    labels["closed_at"] = closed_at[closed_at <= moment]
    items = []
    for row in labels.sort_values("closed_at").itertuples(index=False):
        p = float(row.pred)
        sigma = 55.0 + 0.25 * abs(p)
        pl = p_late_of(p, sigma)
        route = by_tr.get(int(row.tr_id))
        items.append(
            {
                "prediction_id": f"p-{row.sample_id}",
                "tr_id": int(row.tr_id),
                "unit_id": units.get(int(row.tr_id)),
                "route_id": route["route"]["route_id"] if route else None,
                "target_stop_id": int(row.target_stop_id),
                "target_stop_name": names.get(row.target_stop_id) or "Остановка",
                "planned_at": row.target_time_begin.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "issued_at": row.T.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "lead_s": int((row.target_time_begin - row.T).total_seconds()),
                "pred_delay_s": round(p, 1),
                "p10": round(p - 1.2816 * sigma, 1),
                "p50": round(p, 1),
                "p90": round(p + 1.2816 * sigma, 1),
                "p_late": round(pl, 3),
                "risk": risk_of(p, pl),
                "model_version": "mock",
                "source": "model",
                "status": "closed",
                "actual_delay_s": float(row.target_delay_s),
                "abs_error_s": round(abs(float(row.target_delay_s) - p), 1),
                "closed_at": row.closed_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "baseline_delay_s": float(row.cur_dev_s),
            }
        )
    return {"meta": {"source": "dataset/labels/labels_test.csv", "model_source": source}, "closed": items}


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    moment = pd.Timestamp(f"{DAY} {args.moment}")
    window_s = args.window_min * 60
    history_s = args.history_min * 60
    lo_s = -history_s + min(CLONE_OFFSETS_S) - 600
    hi_s = window_s + max(CLONE_OFFSETS_S) + 900

    schedule = load_schedule(moment)
    routes, by_tr = build_routes(schedule)

    raw = pd.read_csv(ROOT / "dataset" / "test" / "traffic.csv", usecols=["tr_id", "unit_id"])
    units = raw.drop_duplicates("tr_id").set_index("tr_id")["unit_id"].astype(int).to_dict()
    traffic = load_traffic("test")
    traffic = traffic[
        (traffic["event_time"] >= moment + pd.Timedelta(seconds=lo_s))
        & (traffic["event_time"] <= moment + pd.Timedelta(seconds=hi_s))
    ]

    sources = []
    vehicles = []
    route_by_id = {r["route_id"]: r for r in routes}
    clone_idx = 0
    for tr_id, track in traffic.groupby("tr_id"):
        tr_id = int(tr_id)
        pts = thin_track(track, moment)
        in_window = [p for p in pts if -60 <= p[0] <= window_s]
        if len(in_window) < 20:
            continue
        info = by_tr.get(tr_id)
        stops_out: list[list[float | int | None]] = []
        if info is not None:
            rows = schedule[schedule["tr_id"] == tr_id]
            seqs = assign_seq(rows, info["seq_keys"])
            for (_, row), seq in zip(rows.iterrows(), seqs, strict=True):
                if seq < 0 or not (lo_s - 1800 <= row["plan_s"] <= hi_s + 1800):
                    continue
                fact = None if pd.isna(row["fact_s"]) else int(round(row["fact_s"]))
                stops_out.append([int(row["tt_action_item_id"]), int(seq), int(row["plan_s"]), fact])
        route_id = info["route"]["route_id"] if info else None
        sources.append({"tr_id": tr_id, "route_id": route_id, "track": pts, "stops": stops_out})
        vehicles.append(
            {
                "unit_id": units.get(tr_id, tr_id),
                "tr_id": tr_id,
                "source": tr_id,
                "d_track": 0,
                "d_plan": 0,
                "clone": False,
            }
        )
        if info is None or not stops_out:
            continue
        for d in CLONE_OFFSETS_S:
            clone_idx += 1
            vehicles.append(
                {
                    "unit_id": CLONE_UNIT_BASE + clone_idx,
                    "tr_id": CLONE_TR_BASE + clone_idx,
                    "source": tr_id,
                    "d_track": d,
                    "d_plan": d,
                    "clone": True,
                }
            )

    # Сбивка и разрыв интервала на маршруте с самым длинным кругом: вместо обычных клонов — BUNCHING_CLONES.
    scheduled = [s for s in sources if s["route_id"] and s["stops"]]
    if scheduled:
        target = max(scheduled, key=lambda s: len(route_by_id[s["route_id"]]["stops"]))
        vehicles = [v for v in vehicles if not (v["clone"] and v["source"] == target["tr_id"])]
        for d_track, d_plan in BUNCHING_CLONES:
            clone_idx += 1
            vehicles.append(
                {
                    "unit_id": CLONE_UNIT_BASE + clone_idx,
                    "tr_id": CLONE_TR_BASE + clone_idx,
                    "source": target["tr_id"],
                    "d_track": d_track,
                    "d_plan": d_plan,
                    "clone": True,
                }
            )
    src_route = {s["tr_id"]: s["route_id"] for s in sources}
    for route in routes:
        route["tr_ids"] = sorted(
            v["tr_id"] for v in vehicles if src_route.get(v["source"]) == route["route_id"]
        )

    world = {
        "meta": {
            "source": "dataset/test (traffic.csv, schedule.csv)",
            "moment": moment.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "window_s": window_s,
            "history_s": history_s,
            "note": "Демо-данные mock-режима: треки и факт из dataset/test, клоны ТС (tr_id 8000001+) — "
            "сдвиг трека по времени для плотности демо; прогнозы в симуляторе синтетические.",
        },
        "routes": routes,
        "sources": sources,
        "vehicles": vehicles,
    }
    honesty = build_honesty(moment, schedule, by_tr, units, args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "world.json").write_text(
        json.dumps(world, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    (out / "honesty.json").write_text(
        json.dumps(honesty, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    n_real = sum(1 for v in vehicles if not v["clone"])
    closed = honesty["closed"]
    mae = np.mean([c["abs_error_s"] for c in closed]) if closed else float("nan")
    base_mae = (
        np.mean([abs(c["actual_delay_s"] - c["baseline_delay_s"]) for c in closed])
        if closed
        else float("nan")
    )
    print(
        f"routes={len(routes)} sources={len(sources)} vehicles={len(vehicles)} (real {n_real}) "
        f"closed={len(closed)} mae={mae:.1f} baseline={base_mae:.1f} -> {out}"
    )


if __name__ == "__main__":
    main()
