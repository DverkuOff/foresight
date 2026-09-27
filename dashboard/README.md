# Форсайт · Диспетчер (Foresight)

Диспетчерский дашборд Foresight (критерий 4 ТЗ): React 19 + Vite 8 + TypeScript (strict), карта MapLibre GL 6.11,
Ant Design 6.6 (светлая тема: белый и светло-голубой, русская локаль), графики ECharts 6, REST — TanStack Query, поток — свой WebSocket-хук
с переподключением. Работает строго по контракту [`docs/api-contract.md`](../docs/api-contract.md).

| Страница               | Путь                        | Что показывает                                                                                                     |
| ---------------------- | --------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| Оперативная обстановка | `/`                         | KPI, карта сети с ТС по риску, проблемные ТС, карточка инцидента (`?incident=<id>`, `?incident=top`)               |
| График движения        | `/stringline?route=R1&tr=…` | график «время × остановки»: план, факт, прогноз 15 мин с P10–P90, опоздания, сбивка                                |
| Честность прогноза     | `/honesty`                  | заблаговременность (окно 10–15 мин), алерты задним числом, онлайн-MAE против `cur_dev_s`, лента закрытых прогнозов |
| What-if                | `/whatif?route=R1`          | «выпустить резерв» / «придержать ТС»: до / после, подбор лучшего действия                                          |
| Производительность     | `/perf`                     | `GET /api/metrics/perf`, зависимости, панели Grafana (`/grafana/d-solo/…`)                                         |
| Администрирование      | `/admin`                    | поток и приём NDTP, пороги риска, модели и переобучение, журнал с CSV, здоровье сервисов, справочники              |

## Режимы данных

- **mock** — всё в браузере: симулятор на фикстурах из `dataset/test` (`src/mocks/fixtures`, строит
  `scripts/make_dashboard_fixtures.py`), REST — обработчики MSW по путям контракта, WebSocket — mock-сокет
  (ТС едут по реальным трекам, инциденты открываются и закрываются, часы потока идут с ускорением).
  Параметры адреса: `?speed=6` (ускорение), `?fleet=300` (нагрузочная проверка карты),
  `?chaos=redis` (деградация) и `?chaos=offline` (через 5 с пропадает связь).
- **live** — реальные `/api`, `/health`, `/ws` и `/grafana` через nginx-прокси образа.

Приоритет выбора: `?mode=mock|live` → runtime-конфиг `/config.js` (в Docker — `FORESIGHT_API_MODE`) →
`VITE_API_MODE` при сборке → по умолчанию dev = mock, production = live.

## Разработка и проверки (на сервере, в контейнере node:22)

```bash
docker run --rm -v "$PWD":/app -w /app -v /root/.npm:/root/.npm node:22 bash -c \
  "npm ci && npm run lint && npm run typecheck && npm test && npm run build && npx prettier --check ."
```

`npm run dev` — Vite на :5173 (mock по умолчанию; live-прокси на `VITE_API_TARGET`, `VITE_GRAFANA_TARGET`).

## Docker

```bash
docker build -t foresight-dashboard dashboard
docker run --rm -p 8080:8080 -e FORESIGHT_API_MODE=mock foresight-dashboard   # демо без backend
```

Переменные образа: `FORESIGHT_API_MODE` (`live`), `FORESIGHT_API_UPSTREAM` (`api:8000`),
`FORESIGHT_GRAFANA_UPSTREAM` (`grafana:3000`), `FORESIGHT_GRAFANA_URL` (`/grafana`), `FORESIGHT_TILES_URL`
(шаблоны тайлов через `|`, `none` — без подложки), `FORESIGHT_MOCK_SPEED`. nginx: SPA-fallback, gzip, вечный кэш
`/assets/`, прокси `/api/`, `/health`, `/ws` (upgrade), `/grafana/`; имена upstream резолвятся при запросе, поэтому
дашборд стартует и без backend. В стеке это сервис `dashboard` основного `docker-compose.yml` (порт 8080).

Подложка — векторный стиль OpenFreeMap positron (без ключей), перекрашенный в палитру интерфейса; не загрузился —
растровые тайлы OpenStreetMap; без интернета карта работает без подложки: линии маршрутов и ТС видны.

## Скриншоты

```bash
dashboard/scripts/screenshots.sh http://127.0.0.1:8080 /tmp/shots   # headless Chrome по CDP, 1920×1080 и 1366×768
```
