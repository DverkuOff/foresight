# Foresight · контракт API

> Статус: **утверждён 25.09.2026**. Общий контракт между `api` (backend), `ml-service`, `predictor` и `dashboard`.
> Изменения — только с записью в журнал в конце файла. Метрики Prometheus — в [`observability.md`](observability.md),
> архитектура — в [`architecture.md`](architecture.md).

## 0. Общие правила

- JSON, UTF-8, `snake_case`. Все времена — ISO 8601 в UTC (`2026-01-06T08:15:00Z`).
- Два разных времени: **время потока** (`stream_time`, `issued_at`, `planned_at` — время данных; при replayer идёт с
  ускорением и в прошлом) и **серверное время** (`server_time`, `received_at`). Дашборд показывает время потока.
- Задержки — секунды, `+` опоздание, `−` опережение. Координаты — WGS-84 градусы.
- Ошибки: `{"detail": "..."}` с HTTP 4xx/5xx (формат FastAPI). Пустые списки — `[]`, а не 404.
- Каждый ответ списка содержит `stream_time` и `degraded` (true, если часть данных из кэша/fallback).
- Все эндпоинты описаны в Swagger (`/docs`) с pydantic-схемами и примерами.

## 1. Справочники

### Уровень риска `risk`
`green | yellow | red | unknown`. Пороги (по умолчанию, меняются в админке, хранятся в `settings`):
- `red`: `pred_delay_s > 120` **или** `p_late > 0.6`;
- `green`: `pred_delay_s < 60` **и** (`p_late` отсутствует **или** `p_late < 0.3`);
- `yellow`: остальное; `unknown`: прогноза нет (нет данных/ТС вне рейса).

### Причины `cause`
| code | Текст для диспетчера | Рекомендация по умолчанию |
|---|---|---|
| `dwell_long` | Длительный простой на остановке | Связаться с водителем, проверить посадку/высадку |
| `slow_segment` | Низкая скорость на участке | Проверить участок (ДТП, затор, ремонт), рассмотреть объезд |
| `layover` | Отстой на конечной сдвигает отправление | Скорректировать время отправления с конечной |
| `accumulated_delay` | Накопленное опоздание по рейсу | Выпуск резервного ТС / сокращённый рейс |
| `bunching` | Сбивка с соседним ТС маршрута | Придержать ТС на остановке / перераспределить интервал |
| `gps_lost` | Потеря GPS-сигнала | Проверить терминал, прогноз по последнему состоянию |
| `unknown` | Причина не определена | — |

`cause` содержит `code`, `text`, `recommendation` и `factors` — топ-3 вклада признаков
(`[{"feature": "dwell_s", "label": "Простой на остановке", "contribution_s": 42.0}]`).

### Маршрут `route_id`
В расписании нет номера маршрута. `route_id` выводится backend-ом (`shared/routes.py`): ТС с одинаковой (с точностью
до порядка обхода) последовательностью остановок объединяются в маршрут `R1`, `R2`, …; `name` — «Маршрут R1:
<первая конечная> — <вторая конечная>» по адресам конечных. Остановка маршрута идентифицируется `stop_key`
(округлённые координаты точки остановки), а плановое прибытие конкретного ТС — `stop_id` = `tt_action_item_id`.

## 2. api (порт 8000, Swagger «Foresight · API»)

### Уже есть (сохраняются без ломающих изменений)
- `GET /health`, `GET /metrics`, `GET /api/ingest/stats`.
- `GET /api/vehicles`, `GET /api/vehicles/{unit_id}` — `VehicleOut` **дополняется** полями (nullable, пока нет прогнозов):
  `route_id`, `risk`, `current_delay_s` (текущее отклонение по детектору), `pred_delay_s` (ближайший прогноз),
  `p_late`, `next_stop` (`{stop_id, stop_key, name, planned_at}`), `incident_id` (активный инцидент ТС или null),
  `p10`, `p90`, `forecast_source` (`model|fallback`), `scheduled` (ТС есть в расписании), `position_time`, `position_age_s`
  (последняя ВАЛИДНАЯ позиция; lat/lon = null, если её нет), производные признаки критерия 3: `segment_speed_kmh` (средняя
  скорость на текущем сегменте), `dwell_s` (время стоянки на остановке), `idle_s` (текущий простой).

### Новые
| Метод и путь | Ответ | Назначение |
|---|---|---|
| `GET /api/routes` | `[{route_id, name, tr_ids[], stops[{stop_key, name, lat, lon, seq}], line: [[lon, lat], ...], color, directions: [{direction, name, stops[], line, segments, gps_segments}]}]`; `line` = прямое направление; геометрия — по реальным GPS-трекам (кэш `backend/assets/route_segments.json`), участки без GPS — `source` straight/osrm | линии маршрутов на карте |
| `GET /api/incidents?status=active\|all&limit=` | `{stream_time, degraded, items: IncidentOut[]}`, сортировка: `red` → `yellow`, затем по `p_late`/`pred_delay_s` убыв. | список проблемных ТС |
| `GET /api/incidents/{incident_id}` | `IncidentOut` + `history` (отклонение ТС за 30 мин: `[{t, delay_s}]`) + `forecast` (`[{stop_id, name, planned_at, pred_delay_s, p10, p90}]` на 15 мин вперёд) | карточка инцидента |
| `GET /api/predictions?tr_id=&status=open\|closed&limit=` | `{items: PredictionOut[]}` | журнал прогнозов |
| `GET /api/alerts?since=&level=&limit=` | `{items: AlertOut[]}` | лента алертов |
| `GET /api/stringline?route_id=&from=&to=` | `{route_id, stops: [{stop_key, name, seq}], trips: [{tr_id, planned: [{t, seq}], actual: [{t, seq}], forecast: [{t, seq, p10, p90}]}], stream_time}` | график «нитка» (время × остановки) |
| `GET /api/metrics/horizon?window_s=3600` | `{closed, online_mae_s, baseline_mae_s, warned_share, retroactive, lead_hist: [{from_s, to_s, count}], mae_by_hour: [{hour, mae_s, baseline_s, n}]}` | страница «Честность прогноза» |
| `GET /api/metrics/perf` | `{ingest_pps, e2e_p95_s, inference_p95_ms, tick_p95_ms, consumer_lag, vehicles_online, deps: {redis, postgres, ml, ingest, predictor}}` | страница «Производительность» |
| `POST /api/whatif` | запрос `{route_id, at?, action: "add_vehicle"\|"hold", params: {from_stop_key?, depart_at?, hold_s?, tr_id?}, horizon_min: 60}` → `{baseline: Scenario, scenario: Scenario, delta: {mean_wait_s, max_gap_s, late_stops, bunching_pairs}}`, где `Scenario = {headways: [{stop_key, t, gap_s}], mean_wait_s, max_gap_s, late_stops, bunching_pairs, vehicles: [{tr_id, arrivals: [{stop_key, t}]}]}` | What-if |
| `GET /api/replay/status`, `POST /api/replay/start\|stop\|speed` | прокси к replayer (+ сброс часов потока при start) | управление демо-потоком |
| `GET/PUT /api/admin/settings` | `{risk: {red_delay_s, red_p_late, green_delay_s, green_p_late}, alert: {min_level, min_p_late}}` | пороги |
| `GET /api/admin/models`, `POST /api/admin/models/{version}/activate`, `POST /api/admin/models/retrain` | версии моделей с метриками (CV, test, онлайн-MAE; у дообученных — поправка и её проверка), активация, запуск переобучения на потоке (202, async; состояние — поле `retrain` ответа GET), fallback-формула и её доля среди прогнозов (`fallback`) | модели |
| `GET /api/admin/journal?kind=alerts\|predictions&from=&to=&format=json\|csv` | журнал текущей линии времени, последние записи первыми; фильтры `from`/`to` (время потока), `tr_id`, `status` (прогнозы); CSV-экспорт | журнал |
| `GET /api/admin/services` | здоровье всех сервисов и зависимостей | здоровье |
| `GET /api/admin/units` | `[{unit_id, tr_id, route_id, scheduled, status, risk, last_packet_at}]`; маршрут — из плана, даже без прогноза сейчас | справочник ТС |

### Схемы
```
PredictionOut = {prediction_id, tr_id, unit_id, route_id, target_stop_id, target_stop_name, planned_at,
                 issued_at, lead_s, pred_delay_s, p10, p50, p90, p_late, risk, model_version, source: "model"|"fallback",
                 status: "open"|"closed", actual_delay_s, abs_error_s, closed_at}
AlertOut      = {alert_id, kind: "delay"|"bunching", incident_id, prediction_id, tr_id, route_id, level: "yellow"|"red",
                 escalated_from, cause, target_stop_id, target_stop_name, issued_at, planned_at, pred_delay_s, p_late,
                 related_tr_id, acknowledged: bool}   # алерт — на уровне инцидента (1 алерт на инцидент + эскалация)
IncidentOut   = {incident_id, kind: "delay"|"bunching", tr_id, unit_id, route_id, route_name, risk,
                 target_stop: {stop_id, name, lat, lon, planned_at}, pred_delay_s, p10, p90, p_late, cause,
                 segment: {from_stop, to_stop, line: [[lon, lat], ...]}, issued_at, time_to_event_s,
                 vehicle: {lat, lon, course_deg, speed_kmh, current_delay_s}, related_tr_id (для bunching)}
```

### WebSocket `/ws`
Сообщения `{"type": ..., "stream_time": ..., ...}`:
- `snapshot` / `delta` — ТС (как сейчас, `VehicleOut` с новыми полями);
- `alert` — новый или обновлённый `AlertOut`;
- `incident` — `{action: "open"|"update"|"close", incident: IncidentOut}`;
- `prediction_closed` — `PredictionOut` со сверкой факта (для «Честности прогноза»);
- `clock` — `{stream_time, epoch}` раз в секунду и при сбросе часов.

## 3. ml-service (порт 8003, Swagger «Foresight · ML»)

| Метод и путь | Назначение |
|---|---|
| `POST /predict` | запрос `{rows: [{row_id, features: {<имя>: число\|null}, sequence_id?}], sequences?: {<id>: base64 float32 LE (SEQ_LEN×N_CHANNELS) \| вложенные списки}}` (последовательности опциональны; без них — CatBoost-часть), в ответе также `sequences_used`; → `{model_version, precision, latency_ms, predictions: [{row_id, pred_delay_s, p10, p50, p90, p_late, expected_abs_error_s, factors: [{feature, contribution_s}]}]}`. Пакет до 1000 строк. Отсутствующие признаки = null (модель обрабатывает). |
| `GET /model/info` | `{version, created_at, features[], metrics: {cv_mae, test_mae}, precision: "fp32"\|"int8", components: ["catboost", "gru"]}` |
| `POST /model/reload` | `{version?}` — перезагрузить активную/указанную версию без остановки |
| `POST /model/retrain`, `GET /model/retrain` | дообучение на потоке в фоне (202; 409, пока идёт): поправка `a + b · прогноз` активной версии по её закрытым прогнозам из `predictions` → новая версия `<база>-online<N>` (не включается сама), если поправка снизила MAE на поздних 30 % (`improved`); GET — состояние последнего запуска (`idle / running / done / error`, MAE до/после на поздних 30 %) |
| `GET /health`, `GET /metrics` | здоровье, метрики по контракту observability |

`p10/p50/p90/p_late/expected_abs_error_s` могут быть `null`, если активная модель их не поддерживает (v1) —
predictor тогда считает риск только по `pred_delay_s`.

## 4. Библиотека инференса `ml/inference.py`

Единый интерфейс для ml-service, офлайн-сабмита и тестов:
```python
bundle = load_bundle(path_or_version)          # артефакты: artifacts/models/<version>/manifest.json + файлы моделей
bundle.version, bundle.features, bundle.precision, bundle.capabilities   # {"quantiles", "p_late", "factors", "sequence"}
out = bundle.predict(features_df, sequences=None)   # DataFrame: pred_delay_s, p10, p50, p90, p_late, expected_abs_error_s
factors = bundle.explain(features_df, top=3)        # вклад признаков (CatBoost SHAP), секунды
```
Инференс работает на CPU (CatBoost + ONNX Runtime CPU); GPU — опционально.

## Журнал изменений

| Дата | Изменение |
|---|---|
| 25.09.2026 | Первая версия |
| 27.09.2026 | По факту реализации ядра прогнозов: VehicleOut (+p10/p90, forecast_source, scheduled, position_*, segment_speed_kmh, dwell_s, idle_s), RouteOut.directions, AlertOut уровня инцидента, последовательности в POST /predict, /health ml-service — capabilities |
| 27.09.2026 | Реализованы журнальные представления (`backend/journal.py`, `backend/stringline.py`, `backend/perf.py`): `/api/incidents` (+`{id}`: `history`, `forecast`, `alerts`), `/api/alerts`, `/api/predictions` (+`outcome`, `actual_lead_s`, `retroactive`), `/api/stringline` (`p10`/`p90` — ISO-время границ прибытия), `/api/metrics/horizon` (+`total`, `warned_red_share`, `late_stops`), `/api/metrics/perf` (p95 за 5 мин по `/metrics` predictor и ml-service, без Prometheus; `deps.*` — `up/down/degraded/unknown/disabled`), прокси `/api/replay/*`; `/ws` пересылает `alert`/`incident`/`prediction_closed` из pub/sub и шлёт `clock` раз в секунду. Строки старых эпох потока отсекаются: прогнозы по `epoch`, остальное по `created_at ≥ epoch` (epoch — мс настенного времени скачка часов) |
