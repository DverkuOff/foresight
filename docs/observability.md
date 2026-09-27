# Foresight · наблюдаемость: метрики, дашборды, алерты

Документ — контракт наблюдаемости Foresight и доказательная база критерия 5 («Производительность и надёжность:
честная оценка с подтверждением метриками»). Архитектура — [`architecture.md`](architecture.md) §8.

Состав:

| Компонент | Где | Что делает |
|---|---|---|
| Prometheus 3.15 (`prom/prometheus:v3.15.0`) | `deploy/prometheus/prometheus.yml` | скрейп всех сервисов каждые 5 с, правила каждые 5 с, хранение 7 дней |
| правила алертов | `deploy/prometheus/alerts.yml`, тесты `alerts_test.yml` | 25 правил в 5 группах (ниже) |
| Grafana 13.2 (`grafana/grafana:13.2.2`) | `deploy/grafana/provisioning`, `deploy/grafana/dashboards` | источник данных и 5 дашбордов из репозитория, папка «Foresight» |
| redis_exporter 1.92 (`oliver006/redis_exporter:v1.92.0-alpine`) | `docker-compose.yml` | Redis: доступность, память, клиенты, команды, поток `foresight:telemetry` и его consumer group |
| postgres_exporter 0.20 (`prometheuscommunity/postgres-exporter:v0.20.1`) | `docker-compose.yml` | PostgreSQL: доступность, соединения, транзакции, строки, размер БД |

## 1. Запуск и доступ

Prometheus, Grafana и экспортёры — часть основного стека (`docker-compose.yml`):

```bash
make demo               # при первом запуске создаёт .env со случайным паролем администратора Grafana
# без make: cp .env.example .env (поменять GF_SECURITY_ADMIN_PASSWORD) и docker compose up -d --build
```

| Что | Адрес | Доступ |
|---|---|---|
| Grafana | http://localhost:3000/grafana/ (и через дашборд: http://localhost:8080/grafana/) | анонимно — только просмотр (роль Viewer), домашняя страница — «Foresight · Обзор системы»; администратор — `admin` / `GF_SECURITY_ADMIN_PASSWORD` из `.env` |
| Prometheus | http://localhost:9090 | цели — `/targets`, правила — `/rules`, активные алерты — `/alerts` |

- Порты меняются переменными `GRAFANA_PORT` и `PROMETHEUS_PORT` (можно с адресом: `127.0.0.1:13000`).
- Пароль администратора Grafana обязателен и берётся только из окружения (`.env` в `.gitignore`); `make up`
  создаёт `.env` со случайным паролем, без make compose останавливается с подсказкой скопировать `.env.example`.
- Grafana работает под sub-path `/grafana/` (`GF_SERVER_SERVE_FROM_SUB_PATH`) и разрешает встраивание
  (`GF_SECURITY_ALLOW_EMBEDDING`): дашборд диспетчера показывает её панели на странице «Производительность».
- Дашборды и источник данных провижинятся из репозитория и не сохраняются из UI (`allowUiUpdates: false`);
  правка — в `deploy/grafana/build_dashboards.py`, затем `python3 deploy/grafana/build_dashboards.py`.
- Grafana работает офлайн: предустановка плагинов, проверки обновлений и новостная лента выключены.

## 2. Сбор метрик

- `scrape_interval` и `evaluation_interval` — 5 с, `scrape_timeout` — 4 с; хранение — 7 дней
  (`--storage.tsdb.retention.time=7d`).
- Сервисы приложения находятся через DNS compose-сети (`dns_sd_configs`, записи A по имени сервиса):

  | job | Имя в DNS : порт | Когда есть |
  |---|---|---|
  | `ingest` | `ingest:8001` | всегда |
  | `predictor` | `predictor:8002` | всегда; N реплик (`--scale predictor=N`) находятся автоматически |
  | `api` | `api:8000` | всегда |
  | `replayer` | `replayer:8010` | когда сервис replayer в стеке |
  | `ml-service` | `ml-service:8003` | всегда (с 26.09) |
  | `redis-exporter` | `redis-exporter:9121` (статически) | observability |
  | `postgres-exporter` | `postgres-exporter:9187` (статически) | observability |
  | `prometheus` | `localhost:9090` (статически) | observability |

- Метки целей: `job` и `service` — имя compose-сервиса, `instance` — `IP:порт` контейнера (различает реплики).
- Сервиса нет в стеке — 0 целей и никаких алертов. Остановленный контейнер пропадает из DNS: Docker пересылает
  запрос во внешний DNS; ответ NXDOMAIN убирает цель (обязательные ingest / predictor / api ловит
  `ServiceMissing`), ошибка или таймаут (как на стенде WSL) оставляет последние цели — их ловит `ServiceDown`
  по `up == 0`. Для replayer и ml-service DNS опрашивается раз в 15 с: резолв отсутствующего имени медленный
  и пишет ошибку в лог Prometheus.
- Экспортёры настроены на быстрый отказ (`REDIS_EXPORTER_CONNECTION_TIMEOUT=2s`, `connect_timeout=2`):
  при остановленной базе скрейп укладывается в `scrape_timeout` и честно отдаёт `redis_up 0` / `pg_up 0`,
  а не «экспортёр не отвечает».

## 3. Метрики сервисов (экспозиция)

Имена — ровно как в выводе `/metrics` (снято с живого стека 25.09.2026). Счётчики `CounterMetricFamily` получают
суффикс `_total`; гистограммы — ряды `_bucket{le}`, `_sum`, `_count`, `_created`. Ко всем рядам Prometheus
добавляет `job`, `service`, `instance`. В PromQL счётчики берутся только через `rate` / `increase`.

### 3.1. ingest (`ingest:8001/metrics`)

| Имя в экспозиции | Тип | Labels | Смысл |
|---|---|---|---|
| `foresight_ndtp_connections_total` | counter | | принятые NDTP-соединения |
| `foresight_ndtp_disconnects_total` | counter | | закрытые NDTP-соединения |
| `foresight_ndtp_read_timeouts_total` | counter | | соединения, закрытые по молчанию |
| `foresight_ndtp_received_bytes_total` | counter | | принятые байты |
| `foresight_ndtp_discarded_bytes_total` | counter | | байты, пропущенные при ресинхронизации потока |
| `foresight_ndtp_frames_total` | counter | | кадры с верной CRC |
| `foresight_ndtp_handshakes_total` | counter | | кадры handshake |
| `foresight_ndtp_realtime_packets_total` | counter | | realtime-пакеты |
| `foresight_ndtp_nav_records_total` | counter | | сохранённые записи Nav00 |
| `foresight_ndtp_crc_errors_total` | counter | | кадры, отброшенные из-за CRC |
| `foresight_ndtp_bad_headers_total` | counter | | отвергнутые неправдоподобные заголовки NPL (мусор) |
| `foresight_ndtp_parse_errors_total` | counter | | кадры с битым телом |
| `foresight_ndtp_unknown_cells_total` | counter | | пакеты, остановленные на неизвестной ячейке |
| `foresight_ndtp_abusive_disconnects_total` | counter | | анти-DoS: закрыто за превышение бюджета ошибок CRC |
| `foresight_ndtp_connections_active` | gauge | | открытые NDTP-соединения |
| `foresight_ndtp_packets_per_second` | gauge | | пакетов/с за последние 10 с (снимок; на дашбордах — `rate` счётчика) |
| `foresight_ndtp_listening` | gauge | | 1 — NDTP-сервер принимает соединения |
| `foresight_vehicles` | gauge | `status` = `online` / `stale` / `offline` | ТС по статусу связи |
| `foresight_stream_time_seconds` | gauge | | часы потока (Unix-секунды); нет ряда до первой точки |
| `foresight_stream_clock_epoch` | gauge | | эпоха часов потока (меняется при перезапуске источника) |
| `foresight_stream_clock_resets_total` | counter | | скачки часов потока (перезапуск источника) |
| `foresight_stream_clock_garbage_total` | counter | | точки с неправдоподобным временем |
| `foresight_stream_clock_off_timeline_total` | counter | | точки вне линии времени потока |
| `foresight_bus_published_total` | counter | | события, записанные в поток `foresight:telemetry` |
| `foresight_bus_buffered` | gauge | | события в памяти, ещё не записанные в Redis |
| `foresight_bus_evicted_total` | counter | | события, вытесненные из переполненного буфера (потеря) |
| `foresight_bus_flushes_total` | counter | | успешные pipeline в Redis |
| `foresight_bus_flush_errors_total` | counter | | неудачные pipeline |
| `foresight_bus_command_errors_total` | counter | | команды, отклонённые Redis |
| `foresight_bus_state_writes_total` | counter | | записи горячего состояния ТС (hash) |
| `foresight_bus_unmapped_events_total` | counter | | события устройств без `tr_id` |
| `foresight_bus_last_flush_seconds` | gauge | | длительность последнего pipeline, с |

### 3.2. predictor (`predictor:8002/metrics`, по ряду на реплику)

| Имя в экспозиции | Тип | Labels | Смысл |
|---|---|---|---|
| `foresight_consumer_events_total` | counter | | обработанные события потока |
| `foresight_consumer_acked_total` | counter | | подтверждённые (XACK) записи |
| `foresight_consumer_events_per_second` | gauge | | событий/с за 10 с (снимок) |
| `foresight_consumer_malformed_total` | counter | | пропущенные битые записи |
| `foresight_consumer_handler_errors_total` | counter | | упавшие обработчики пачек |
| `foresight_consumer_recovered_total` | counter | | свои pending-записи, перечитанные после сбоя |
| `foresight_consumer_claimed_total` | counter | | pending-записи, забранные у других потребителей |
| `foresight_consumer_read_errors_total` | counter | | неудачные чтения потока |
| `foresight_consumer_history_total` | counter | | обработанные записи, перечитанные при старте (заполнение окон) |
| `foresight_consumer_removed_consumers_total` | counter | | неактивные потребители, удалённые из группы |
| `foresight_consumer_lag` | gauge | | лаг consumer group (недоставленные записи); общий для группы |
| `foresight_consumer_pending` | gauge | | доставленные, но не подтверждённые записи |
| `foresight_stream_length` | gauge | | длина потока `foresight:telemetry` |
| `foresight_predictor_events_total` | counter | | события, обработанные ядром |
| `foresight_predictor_unmapped_events_total` | counter | | события без `tr_id` (не попадают в окна) |
| `foresight_predictor_late_points_total` | counter | | поздние точки, вставленные в трек по времени |
| `foresight_predictor_duplicate_points_total` | counter | | точные дубли (отброшены) |
| `foresight_predictor_expired_points_total` | counter | | точки старше окна (отброшены) |
| `foresight_predictor_ahead_points_total` | counter | | точки из будущего относительно часов потока (отложены) |
| `foresight_predictor_restored_points_total` | counter | | точки, восстановленные из потока при старте |
| `foresight_predictor_tracks` | gauge | | ТС с окном трека |
| `foresight_predictor_window_points` | gauge | | точек во всех окнах |
| `foresight_predictor_held_points` | gauge | | отложенные точки вне линии времени |
| `foresight_predictor_stream_time_seconds` | gauge | | часы потока predictor (Unix-секунды) |
| `foresight_predictor_clock_epoch` | gauge | | эпоха часов потока |
| `foresight_predictor_clock_resets_total` | counter | | скачки часов потока |
| `foresight_predictor_clock_garbage_total` | counter | | события с неправдоподобным временем |
| `foresight_predictor_off_timeline_total` | counter | | события вне линии времени |
| `foresight_predictor_ticks_total` | counter | | выполненные тики прогнозов |
| `foresight_predictor_ticks_skipped_total` | counter | | пропущенные границы тиков (скачок потока или отставание) |
| `foresight_predictor_ticks_lagging_total` | counter | | тики, пропущенные при отставании: пачка уже дошла до следующей границы, а событие старше 1 с (`FORESIGHT_TICK_LAG_S`) — очередь не копится |
| `foresight_predictor_tick_errors_total` | counter | | упавшие тики |
| `foresight_predictor_tick_duration_seconds` | gauge | | длительность последнего тика, с |
| `foresight_predictor_tick_seconds` | histogram | `le`: 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5 | длительность каждого тика, с (доработка §8 п. 1) |
| `foresight_predictor_event_latency_seconds` | histogram | `le`: 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60 | приём пакета в ingest → обработка в predictor, с |
| `foresight_predictions_total` | counter | `source` = `model` / `fallback` | выданные прогнозы (точка «ТС × целевая остановка» на тике: первая выдача и каждое обновление); оба ряда есть с нуля |
| `foresight_prediction_lead_seconds` | histogram | `kind` = `prediction` (каждый закрытый прогноз) / `alert` (закрытый прогноз, о котором был алерт; наблюдается дополнительно); `le`: 0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 630, 660, 690, 720, 750, 780, 810, 840, 870, 900, 960, 1020, 1080, 1140, 1200 | заблаговременность: проход целевой остановки по детектору − момент первой выдачи (у `alert` — момент алерта), с; ≤ 0 — бакет `le="0"` |
| `foresight_prediction_abs_error_seconds` | histogram | `source`; `le`: 5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 300, 450, 600, 900 | \|прогноз − факт\| задержки при закрытии прогноза (последнее значение прогноза), с |
| `foresight_online_mae_seconds` | gauge | | онлайн-MAE закрытых прогнозов за последние 60 мин потокового времени; ряд есть после ≥ 30 закрытых |
| `foresight_online_baseline_mae_seconds` | gauge | | MAE baseline «прогноз = онлайн-`cur_dev_s` в момент выдачи» (отклонение на последней подтверждённой детектором остановке; нет — 0) на тех же прогнозах |
| `foresight_alerts_total` | counter | `level` = `warning` (жёлтый) / `critical` (красный); `cause` = `dwell` / `slow_segment` / `layover` / `accumulated_delay` / `bunching` / `gps_loss` / `other` | алерты уровня инцидента: открытие инцидента задержки ТС (уровень ТС — риск ближайшей целевой остановки окна с гистерезисом `FORESIGHT_ALERT_HYSTERESIS_S`) и его эскалация до красного, сбивка ТС; повторный алерт по ТС — только после закрытия инцидента (`FORESIGHT_INCIDENT_CLEAR_TICKS` тиков в зелёном); все 14 рядов есть с нуля |
| `foresight_alerts_retroactive_total` | counter | | алерты, выданные в момент фактического прохождения или позже; должно оставаться 0 |
| `foresight_predictions_open` | gauge | | прогнозы, ожидающие факта прохождения |

Зависимость `ml-service` predictor отдаёт тем же `DependencyStatus`: `foresight_dependency_up{dependency="ml-service"}`
(проба `GET /health` раз в 5 с и каждый `POST /predict`) и `DependencyDown` покрывают отказ ML без новых правил.

### 3.3. api (`api:8000/metrics`)

| Имя в экспозиции | Тип | Labels | Смысл |
|---|---|---|---|
| `foresight_vehicles` | gauge | `status` | ТС по статусу связи в кэше api (на дашбордах — ряд `service="ingest"`) |
| `foresight_api_ws_clients` | gauge | | открытые WebSocket-соединения |
| `foresight_api_degraded` | gauge | | 1 — Redis недоступен, api отдаёт последнее известное состояние |

### 3.4. Общие для ingest, predictor, api

| Имя в экспозиции | Тип | Labels | Смысл |
|---|---|---|---|
| `foresight_db_buffered` | gauge | | строки журнала в памяти, ожидающие PostgreSQL |
| `foresight_db_dropped_total` | counter | | потерянные строки (переполнение буфера или БД выключена) |
| `foresight_db_written_total` | counter | | записанные строки |
| `foresight_db_rejected_total` | counter | | строки, отклонённые PostgreSQL (некорректные данные) |
| `foresight_db_errors_total` | counter | | неудачные операции с БД |
| `foresight_dependency_up` | gauge | `dependency` = `redis` / `postgres` (у predictor ещё `ml-service`) | 1 — зависимость отвечает; 0 — отказ **или ещё не проверена после старта** |
| `foresight_dependency_outages_total` | counter | `dependency` | переходы «доступна → недоступна» с запуска |
| `process_cpu_seconds_total`, `process_resident_memory_bytes`, `process_open_fds`, `python_info` … | | | стандартные метрики процесса prometheus-client |

### 3.5. replayer (`replayer:8010/metrics`)

| Имя в экспозиции | Тип | Labels | Смысл |
|---|---|---|---|
| `foresight_replayer_packets_dispatched_total` | counter | | пакеты, выданные планировщиком |
| `foresight_replayer_packets_sent_total` | counter | | realtime-кадры, записанные в сокет |
| `foresight_replayer_packets_dropped_total` | counter | `reason` = `overflow` / `disconnected` / `stopped` | неотправленные пакеты |
| `foresight_replayer_handshakes_total` | counter | | записанные handshake |
| `foresight_replayer_connects_total` | counter | | успешные TCP-подключения к приёмнику |
| `foresight_replayer_reconnects_total` | counter | | подключения после потери связи |
| `foresight_replayer_disconnects_total` | counter | | неожиданные обрывы |
| `foresight_replayer_connect_failures_total` | counter | | неудачные попытки подключения |
| `foresight_replayer_cycles_total` | counter | | запуски воспроизведения (старт, рестарт, круг петли) |
| `foresight_replayer_bridge_posts_total` | counter | | конфигурации, принятые эмулятором (режим bridge) |
| `foresight_replayer_bridge_errors_total` | counter | | ошибки запросов к эмулятору |
| `foresight_replayer_connections_active` | gauge | | открытые NDTP-соединения |
| `foresight_replayer_running` | gauge | | 1 — воспроизведение ждёт или идёт |
| `foresight_replayer_speed` | gauge | | скорость, секунд данных за секунду |
| `foresight_replayer_lag_seconds` | gauge | | возраст старейшего неотправленного пакета, с |
| `foresight_replayer_backlog_packets` | gauge | | пакеты в очередях отправки |
| `foresight_replayer_last_send_lag_seconds` | gauge | | опоздание последнего пакета относительно расписания, с |
| `foresight_replayer_epoch` | gauge | | номер прохода; смена сбрасывает часы потока |
| `foresight_replayer_data_time_seconds` | gauge | | позиция часов данных (Unix-секунды) |

### 3.6. Экспортёры (используемые на дашбордах и в алертах)

| Метрика | Экспортёр | Смысл |
|---|---|---|
| `redis_up` | redis_exporter | 1 — Redis отвечает |
| `redis_memory_used_bytes`, `redis_memory_used_rss_bytes` | redis_exporter | память |
| `redis_connected_clients`, `redis_blocked_clients` | redis_exporter | клиенты (заблокированные — XREADGROUP BLOCK) |
| `redis_commands_processed_total` | redis_exporter | команды (ops/s через `rate`) |
| `redis_net_input_bytes_total`, `redis_net_output_bytes_total` | redis_exporter | сеть |
| `redis_db_keys`, `redis_db_keys_expiring` | redis_exporter | ключи `db0` |
| `redis_stream_length{stream}`, `redis_stream_group_lag{group}`, `redis_stream_group_messages_pending{group}`, `redis_stream_group_consumers{group}` | redis_exporter | поток `foresight:telemetry` и группа `predictors` (сверка с метриками predictor) |
| `pg_up` | postgres_exporter | 1 — PostgreSQL отвечает |
| `pg_stat_database_numbackends{datname}`, `pg_stat_activity_count{datname,state}` | postgres_exporter | соединения |
| `pg_stat_database_xact_commit`, `pg_stat_database_xact_rollback` | postgres_exporter | транзакции (счётчики без `_total`) |
| `pg_stat_database_tup_inserted`, `…_tup_updated`, `…_tup_deleted` | postgres_exporter | строки |
| `pg_database_size_bytes{datname}` | postgres_exporter | размер БД |
| `up`, `ALERTS` | Prometheus | доступность целей, активные алерты |

### 3.7. ml-service (`ml-service:8003/metrics`)

Своя `CollectorRegistry` (`ml/service.py`), без метрик процесса.

| Имя в экспозиции | Тип | Labels | Смысл |
|---|---|---|---|
| `foresight_ml_request_duration_seconds` | histogram | `endpoint` — шаблон маршрута: `/predict`, `/model/info`, `/model/reload`, `/health` (ряды всех четырёх есть с нуля); `le`: 0.001, 0.0025, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.25, 0.5, 1, 2.5 | время обработки запроса от приёма до готового ответа, с; для `/predict` — весь батч (признаки → прогноз → SHAP) |
| `foresight_ml_batch_size` | histogram | `le`: 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000 | прогнозных точек в одном `POST /predict` |
| `foresight_ml_model_info` | gauge = 1 | `version`, `model`, `precision` | активная модель: ровно один ряд, после `POST /model/reload` — ряд новой версии; до загрузки модели ряда нет |

## 4. Дашборды

Все — в папке «Foresight», обновление 5 с, диапазон по умолчанию 15 минут, общий курсор, ссылки друг на друга,
аннотации «Алерты Prometheus» (интервалы firing) и «Сброс часов потока». Источник данных — `Foresight Prometheus`,
uid `foresight-prometheus` (по умолчанию).

| Дашборд | uid | Что показывает |
|---|---|---|
| Foresight · Обзор системы (домашний) | `foresight-overview` | up сервисов и зависимостей, алерты firing, ТС на связи, пакеты/с, задержка обработки p95 (цель < 1 с), лаг очереди (≈ 0), время тика (< 100 мс), буферы шины и БД; ТС по статусам, пропускная способность «приём → шина → predictor», квантили задержки, лаг и pending, тики, буферы деградации; часы потока против реального времени и скорость часов; список активных алертов; CPU и RSS процессов |
| Foresight · Приём NDTP | `foresight-ingest` | соединения, приём, пакеты/с, трафик, CRC, анти-DoS; кадры/с, ошибки протокола (CRC, мусорные заголовки, разбор, неизвестные ячейки), таймауты, байты, ТС по статусам, часы потока ingest, устройства без `tr_id`; replayer: состояние, скорость, время данных, отставание, отправлено/отброшено/с, очередь, переподключения, мост эмулятора. Переменная `$instance` (реплики ingest) |
| Foresight · Поток и прогнозы | `foresight-stream` | шина: запись/с, буфер и вытеснения, ошибки Redis, длительность pipeline, длина потока; consumer group: реплики, потребители, лаг, pending, события/с по репликам, восстановление (XCLAIM, pending, история, GC), ошибки; predictor: события/с, поздние / дубли / устаревшие / будущие точки, треки и окна, тики, длительность тика p50/p95, heatmap и квантили задержки события, сбросы часов, время потока. Переменная `$instance` (реплики predictor) |
| Foresight · Хранилище и деградация | `foresight-storage` | state timeline `foresight_dependency_up` по сервисам, экспортёры Redis / PostgreSQL, режим api (норма / последнее состояние), сбои зависимостей, WS-клиенты; журнал PostgreSQL: записано/с, буфер, ошибки / отброшено / отклонено; PostgreSQL: соединения, транзакции/с, строки/с, размер; Redis: память, клиенты, команды/с, сеть, ключи, поток; api HTTP (контракт п. 7). Переменная `$service` |
| Foresight · Модель | `foresight-model` | контракт п. 7: версия модели, инференс p50/p95/p99 (цель < 100 мс), размер батча, GPU-память, прогнозы/с по источнику, доля fallback, онлайн-MAE против baseline, открытые прогнозы, распределение заблаговременности алертов, алерты по уровню и причине, алерты задним числом (0), распределение ошибки |

Единицы: секунды (`s`), байты (`bytes`, `Bps`), операции/с (`ops`, `pps`, `reqps`), доли (`percentunit`).
Пороги цветов — по целям архитектуры: задержка события — жёлтый с 0,5 с, красный с 1 с; тик и инференс —
жёлтый со 100 мс; лаг — жёлтый со 100, красный с 500; алерты задним числом — красный с 1.

## 5. Алерты

Правила — `deploy/prometheus/alerts.yml`, у каждого `severity` (`critical` / `warning`), `summary` и
`description` на русском. Правила chaos-сценариев покрыты юнит-тестами `deploy/prometheus/alerts_test.yml`.

| Алерт | severity | Условие | for | Смысл |
|---|---|---|---|---|
| `ServiceDown` | critical | `up == 0` | 15s | цель не отдаёт `/metrics` |
| `ServiceMissing` | critical | `absent(up{job="ingest"})` (и predictor, api) | 30s | обязательного сервиса нет в DNS |
| `DependencyDown` | critical | `foresight_dependency_up == 0 and on (job, instance, dependency) foresight_dependency_outages_total > 0` | 5s | сервис не видит Redis / PostgreSQL (состояние «не проверена после старта» не считается) |
| `RedisDown` | critical | `redis_up == 0` | 5s | Redis недоступен (экспортёр) |
| `PostgresDown` | critical | `pg_up == 0` | 5s | PostgreSQL недоступен (экспортёр) |
| `NdtpNotListening` | critical | `foresight_ndtp_listening == 0` | 15s | ingest не принимает NDTP |
| `NoTelemetry` | warning | `sum(increase(foresight_ndtp_realtime_packets_total[5m])) == 0` | 1m | ни одного пакета 5 минут |
| `NoNavRecords` | warning | соединения > 0 и `increase(foresight_ndtp_nav_records_total[2m]) == 0` | 1m | соединения есть, навигации нет |
| `CRCErrorSpike` | warning | `increase(foresight_ndtp_crc_errors_total[1m]) > 10` | — | всплеск битых кадров |
| `AbusiveClientDisconnected` | warning | `increase(foresight_ndtp_abusive_disconnects_total[5m]) > 0` | — | анти-DoS отключил клиента |
| `ReplayerLagging` | warning | `foresight_replayer_lag_seconds > 5` при `running == 1` | 30s | replayer отстаёт от расписания |
| `ReplayerDisconnected` | warning | `foresight_replayer_connections_active == 0` при `running == 1` | 15s | replayer не подключён к ingest |
| `ConsumerLagHigh` | warning | `max(foresight_consumer_lag) > 500` | 30s | predictor не успевает за потоком |
| `EventLatencyHigh` | warning | p95 `foresight_predictor_event_latency_seconds` за 1m > 1 с | 2m | задержка обработки выше цели |
| `PredictorTickSlow` | warning | медиана `foresight_predictor_tick_duration_seconds` за 2m > 1 с | 1m | тики прогнозов запаздывают |
| `PredictorErrors` | warning | `increase` ошибок тиков или обработчиков за 5m > 0 | — | ошибки в predictor |
| `BusBuffering` | warning | `foresight_bus_buffered > 0` при `dependency_up{dependency="redis"} == 0` | 5s | деградация: ingest копит события в памяти |
| `DbBuffering` | warning | `foresight_db_buffered > 0` при `dependency_up{dependency="postgres"} == 0` | 5s | деградация: журнал копится в памяти |
| `ApiDegraded` | warning | `foresight_api_degraded == 1` | 5s | api отдаёт последнее известное состояние |
| `DataLoss` | critical | `increase(foresight_bus_evicted_total[5m]) > 0 or increase(foresight_db_dropped_total[5m]) > 0` | — | буфер переполнился, данные потеряны |
| `DbRowsRejected` | warning | `increase(foresight_db_rejected_total[15m]) > 0` | — | PostgreSQL отклоняет строки |
| `MLInferenceSlow` | warning | p95 `foresight_ml_request_duration_seconds{endpoint="/predict"}` за 1m > 0,1 с | 1m | контракт: инференс медленнее цели |
| `MLFallbackActive` | warning | доля `foresight_predictions_total{source="fallback"}` за 1m > 0,5 | 30s | контракт: прогнозы по fallback |
| `OnlineAccuracyDegraded` | warning | `foresight_online_mae_seconds > foresight_online_baseline_mae_seconds` | 10m | контракт: модель хуже baseline |
| `RetroactiveAlerts` | critical | `increase(foresight_alerts_retroactive_total[15m]) > 0` | — | контракт: алерт задним числом |

Как подобраны пороги:

- chaos-алерты требуют двух оценок подряд (`for: 5s`): сервису нужно до 5 с, чтобы заметить отказ (таймауты
  клиента Redis / пула PostgreSQL), ещё до 5 с — скрейп и до 5 с — оценка правила;
- на здоровом стеке `foresight_bus_buffered` бывает 1–5 в момент скрейпа (пачка ждёт 10 мс), поэтому буферные
  алерты привязаны к отказу зависимости, а не к `> 0`;
- `DependencyDown` не срабатывает на «ещё не проверена после старта» (экспортируется как 0) — отличается
  по счётчику сбоев;
- `EventLatencyHigh`: окно 1m и `for: 2m` — догоняющий всплеск задержки после восстановления Redis (события,
  пролежавшие в буфере ingest) уходит из окна за минуту и не даёт ложного warning.

## 6. Chaos-демо

Сценарий (стек поднят, идёт нагрузка replayer x30):

```bash
docker compose stop redis    # отказ Redis
#   Grafana «Обзор системы»: Зависимости → DOWN, «Буферы деградации» растут, алерты firing;
#   Prometheus /alerts: DependencyDown (ingest, predictor, api), RedisDown, BusBuffering, ApiDegraded
docker compose start redis   # восстановление
#   буфер ingest дописывается в поток, алерты гаснут

docker compose stop postgres # отказ PostgreSQL
#   DependencyDown (ingest, predictor, api), PostgresDown, DbBuffering
docker compose start postgres
```

Замеры на стенде (25.09.2026, чистый стек, replayer x30 по `test`, 07:00–10:00; время от команды
`stop` / `start` до смены состояния алерта в `/api/v1/alerts`, опрос раз в 0,5 с):

| Сценарий | Алерт | firing через | resolved через |
|---|---|---|---|
| stop / start redis | `DependencyDown` (ingest, predictor, api) | 13,0 с | 19,3 с |
| | `RedisDown` | 13,0 с | 8,8 с |
| | `BusBuffering` | 14,0 с | 15,1 с |
| | `ApiDegraded` | 14,0 с | 15,1 с |
| stop / start postgres | `DependencyDown` | 14,5 с | 14,1 с |
| | `PostgresDown` | 14,5 с | 8,8 с |
| | `DbBuffering` | 15,5 с | 15,1 с |
| stop / start api | `ServiceDown` | 22,2 с | 9,9 с |

По двум прогонам сценариев Redis / PostgreSQL с итоговой конфигурацией: firing — 11–23 с, resolved — 9–21 с
(цель ≤ 30 с). Других алертов во время и после сценариев не было (в том числе `EventLatencyHigh` после догоняющего всплеска задержки
до 30 с и `DataLoss`: вытеснений и потерь нет, только `foresight_db_errors_total` во время отказа). Здоровый стек
под нагрузкой (replayer x30, плюс эмулятор с тремя устройствами) — 0 алертов. `--scale predictor=2` —
вторая цель появляется в Prometheus за ≤ 25 с и исчезает после `--scale predictor=1`.

## 7. Метрики следующего блока (обязательны к реализации)

Метрики ml-service и predictor из этого раздела реализованы (26.09.2026) и перенесены в §3.2 и §3.7 — с теми же
именами, типами, labels и бакетами; дашборд «Модель» и правила `foresight-ml-contract` работают на них. Здесь
остаётся то, чего ещё нет: панели показывают *No data*, правила валидны и молчат. Метрики регистрируются так же,
как существующие: своя `CollectorRegistry` на приложение, счётчики экспонируются с `_total`.

### 7.1. ml-service

| Имя | Тип | Labels | Бакеты | Смысл |
|---|---|---|---|---|
| `foresight_ml_gpu_memory_bytes` | gauge | `device` — индекс GPU (`"0"`) | — | память GPU, занятая процессом; без GPU ряд не экспортируется (образ ml-service — CPU, ряда нет) |

### 7.3. api

| Имя | Тип | Labels | Бакеты | Смысл |
|---|---|---|---|---|
| `foresight_http_request_duration_seconds` | histogram | `method`; `route` — шаблон маршрута (`/api/vehicles/{unit_id}`, не сырой путь); `status` — код строкой (`"200"`) | 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5 | время обработки HTTP-запроса, с (кроме `/metrics`; WebSocket `/ws` — только handshake) |

## 8. Доработки сервисов

Чего не хватало в экспорте сервисов на 25.09 (пп. 1 и 4 закрыты 26.09 вместе с прогнозами):

1. ✅ **Длительность тика — гейдж последнего тика.** При x30 тик идёт раз в секунду, скрейп — раз в 5 с, поэтому
   p50/p95 на дашборде — квантили по снимкам (`quantile_over_time`), 4 из 5 тиков не видны. Добавлена гистограмма
   `foresight_predictor_tick_seconds` (бакеты 0.001…5 с; гейдж `…_tick_duration_seconds` остался).
2. **Нет end-to-end «пакет → дашборд».** `foresight_predictor_event_latency_seconds` покрывает участок
   ingest → predictor. Для цели < 1 с нужна гистограмма в api `foresight_api_push_latency_seconds`
   (`received_at` события → отправка дельты в WebSocket).
3. **`foresight_dependency_up` = 0 для «ещё не проверена».** Правило обходит это счётчиком сбоев, но панели
   сразу после старта показывают DOWN. Лучше не экспортировать ряд до первой проверки.
4. ✅ **Во время отказа PostgreSQL буфер журнала непуст только у сервисов, которым есть что писать** (было —
   лишь события сервиса), поэтому `DbBuffering` загорался по одному-двум сервисам, а не по всем трём
   (в замерах — api или ingest). Predictor теперь пишет прогнозы, алерты, инциденты и проходы остановок
   каждый тик — при отказе PostgreSQL его буфер растёт сразу.
5. **Отказ зависимости сервисы замечают за 1–5 с** (таймауты клиента Redis и пула PostgreSQL, проба PostgreSQL
   раз в 5 с) — это основная часть времени до firing; сократить можно только в сервисах.

## 9. Проверки

```bash
python3 deploy/grafana/build_dashboards.py --check                               # JSON дашбордов = генератор
python3 scripts/check_dashboards.py --promtool --no-panels                       # promtool check config / rules, test rules
python3 scripts/check_dashboards.py --prometheus http://localhost:9090 --allow-empty  # все панели с данными
uv run pytest tests/test_observability.py                                        # статические проверки
```

- `check_dashboards.py` выполняет каждый запрос каждой панели (range 15 мин, шаг 15 с, `$__rate_interval` = 20s,
  переменные = `.*`), выражения аннотаций и запросы переменных; код возврата 1 при ошибке PromQL или пустой
  панели. С `--allow-empty` пустыми могут быть только панели, все метрики которых из раздела 7.
- `tests/test_observability.py`: JSON дашбордов валиден и совпадает с генератором, uid уникальны, у всех панелей
  с запросами источник `foresight-prometheus`, все метрики `foresight_*` в дашбордах и правилах либо
  экспортируются кодом `backend/` / `replayer/`, либо объявлены в разделе 7, имена алертов уникальны.
- Прогон 25.09.2026 на живом стеке (replayer x30): все 7 целей up, 93 панели с данными, 19 пустых — только
  контракт раздела 7, ошибок PromQL нет, алертов на здоровом стеке нет.
