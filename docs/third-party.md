# Сторонние компоненты и лицензии

Foresight использует только открытые компоненты и открытые данные; все лицензии допускают использование в
решении. Ниже — прямые зависимости с ролью в проекте, образы Docker, внешние данные и сервисы, затем полные
списки пакетов (включая транзитивные), собранные из окружений сборки 27.09.2026 (`pip-licenses`,
`license-checker`).

## 1. Прямые зависимости

### Python (`pyproject.toml`)

| Пакет | Лицензия | Для чего |
|---|---|---|
| numpy, pandas, pyarrow | BSD-3-Clause, BSD-3-Clause, Apache-2.0 | данные, признаки |
| catboost | Apache-2.0 | модель градиентного бустинга (MAE, квантили, p_late) |
| scikit-learn | BSD-3-Clause | кросс-валидация, калибровка |
| onnxruntime | MIT | инференс GRU (ONNX, INT8) на CPU |
| torch, onnx, onnxscript | BSD-3-Clause, Apache-2.0, MIT | обучение GRU и экспорт в ONNX (только обучение, не в образах сервисов) |
| fastapi, uvicorn, pydantic, pydantic-settings | MIT, BSD-3-Clause, MIT, MIT | HTTP API, WebSocket, Swagger |
| redis (redis-py) | MIT | клиент Redis Streams / pub/sub |
| asyncpg | Apache-2.0 | клиент PostgreSQL |
| prometheus-client | Apache-2.0 | метрики |
| pytest, ruff, fakeredis, httpx | MIT, MIT, BSD-3-Clause, BSD-3-Clause | тесты и линтер (только разработка) |
| sphinx, furo, myst-parser | BSD-2-Clause, MIT, MIT | сайт документации (только сборка документации) |

### Дашборд (`dashboard/package.json`)

| Пакет | Лицензия | Для чего |
|---|---|---|
| react, react-dom, react-router | MIT | интерфейс |
| antd, @ant-design/icons | MIT | компоненты, администрирование |
| maplibre-gl | BSD-3-Clause | карта (WebGL) |
| echarts | Apache-2.0 | графики |
| @tanstack/react-query | MIT | запросы к API |
| msw | MIT | демо-режим без backend (mock API) |
| vite, typescript, eslint, prettier, vitest, @testing-library/* | MIT / Apache-2.0 | сборка и тесты (только разработка) |

## 2. Образы Docker

| Образ | Лицензия | Роль |
|---|---|---|
| python:3.12-slim | PSF-2.0 (Python), Debian — свободные лицензии | база образов backend, ml-service, replayer |
| ghcr.io/astral-sh/uv | Apache-2.0 / MIT | установка зависимостей при сборке |
| node:22 | MIT | сборка дашборда (не входит в итоговый образ) |
| nginx:1.31-alpine | BSD-2-Clause | раздача дашборда, прокси |
| redis:8 | RSALv2 / SSPLv1 / AGPLv3 (на выбор) | шина событий и горячее состояние — используется без изменений |
| postgres:18 | PostgreSQL License | журнал прогнозов и алертов |
| prom/prometheus:v3.15.0 | Apache-2.0 | метрики и алерты |
| grafana/grafana:13.2.2 | AGPL-3.0 | дашборды мониторинга — используется без изменений |
| oliver006/redis_exporter | MIT | метрики Redis |
| prometheuscommunity/postgres-exporter | Apache-2.0 | метрики PostgreSQL |
| ndtp-telemetry-emulator:1.0 | материалы организаторов хакатона | эмулятор NDTP (из датасета, в репозиторий не входит) |

## 3. Данные и внешние сервисы

| Что | Лицензия / условия | Где используется |
|---|---|---|
| датасет организаторов (телеметрия, расписание, разметка) | условия хакатона | обучение и демо-поток; скачивается `make dataset`, в репозиторий не входит |
| данные OpenStreetMap | ODbL 1.0, © участники OpenStreetMap | подложка карты, дорожная геометрия OSRM |
| OpenFreeMap (векторные тайлы, стиль) | бесплатно, без ключа; схема OpenMapTiles (BSD-3-Clause / CC-BY 4.0) | подложка карты дашборда (атрибуция на карте) |
| тайлы tile.openstreetmap.org | Tile Usage Policy OSMF | запасная подложка, если векторная недоступна |
| OSRM (движок; свой экземпляр в Docker `ghcr.io/project-osrm/osrm-backend` на выгрузке OSM Москвы и области с download.openstreetmap.fr) | движок BSD-2-Clause, данные ODbL | один раз при сборке кэша геометрии маршрутов: привязка GPS-треков целых направлений к дорогам (match, `python -m shared.routes trace`) и путь по дорогам для участков без трека (`snap`); сервисы к нему не обращаются |
| swagger-ui-dist через jsDelivr | Apache-2.0 | статическая страница OpenAPI на GitHub Pages |

## 4. Полный список Python-пакетов (образы backend и ml-service)

Лицензии: MIT — 20, BSD-3-Clause — 10, BSD — 5, Apache-2.0 — 2, Apache, Version 2.0 — 1, Apache Software — 1, Python Software Foundation — 1, BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 — 1, Apache-2.0 OR BSD-2-Clause — 1, MIT-CMU — 1, Apache-2.0 AND BSD-2-Clause — 1, 3-Clause BSD — 1, Apache Software; BSD — 1, PSF-2.0 — 1, Apache Software; MIT — 1.

| Пакет | Версия | Лицензия | Ссылка |
|---|---|---|---|
| PyYAML | 6.0.3 | MIT | https://pyyaml.org/ |
| annotated-doc | 0.0.5 | MIT | https://github.com/fastapi/annotated-doc |
| annotated-types | 0.8.0 | MIT | https://github.com/annotated-types/annotated-types |
| anyio | 4.15.1 | MIT | https://anyio.readthedocs.io/en/stable/versionhistory.html |
| asyncpg | 0.31.0 | Apache-2.0 | — |
| catboost | 1.2.10 | Apache, Version 2.0 | https://catboost.ai |
| click | 8.5.0 | BSD-3-Clause | https://github.com/pallets/click/ |
| cloudpickle | 3.1.2 | BSD | https://github.com/cloudpipe/cloudpickle |
| contourpy | 1.4.0 | BSD-3-Clause | https://github.com/contourpy/contourpy |
| cycler | 0.12.1 | BSD | https://matplotlib.org/cycler/ |
| fastapi | 0.141.1 | MIT | https://github.com/fastapi/fastapi |
| flatbuffers | 25.12.19 | Apache Software | https://google.github.io/flatbuffers/ |
| fonttools | 4.66.0 | MIT | http://github.com/fonttools/fonttools |
| graphviz | 0.21 | MIT | https://github.com/xflr6/graphviz |
| h11 | 0.16.0 | MIT | https://github.com/python-hyper/h11 |
| httptools | 0.8.0 | MIT | https://github.com/MagicStack/httptools |
| idna | 3.20 | BSD-3-Clause | https://github.com/kjd/idna |
| joblib | 1.6.0 | BSD-3-Clause | https://joblib.readthedocs.io |
| kiwisolver | 1.5.1 | BSD | https://github.com/nucleic/kiwi |
| matplotlib | 3.11.2 | Python Software Foundation | https://matplotlib.org |
| narwhals | 2.26.0 | MIT | https://github.com/narwhals-dev/narwhals |
| numpy | 2.5.3 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | https://numpy.org |
| onnxruntime | 1.30.0 | MIT | https://onnxruntime.ai |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause | https://github.com/pypa/packaging |
| pandas | 3.0.6 | BSD | https://pandas.pydata.org |
| pillow | 12.3.0 | MIT-CMU | https://python-pillow.github.io |
| plotly | 7.1.0 | MIT | https://plotly.com/python/ |
| prometheus_client | 0.26.0 | Apache-2.0 AND BSD-2-Clause | https://github.com/prometheus/client_python |
| protobuf | 7.36.2 | 3-Clause BSD | https://developers.google.com/protocol-buffers/ |
| pyarrow | 25.0.1 | Apache-2.0 | https://arrow.apache.org/ |
| pydantic | 2.13.5 | MIT | https://github.com/pydantic/pydantic |
| pydantic-settings | 2.15.0 | MIT | https://github.com/pydantic/pydantic-settings |
| pydantic_core | 2.46.5 | MIT | https://github.com/pydantic |
| pyparsing | 3.3.3 | MIT | https://github.com/pyparsing/pyparsing/ |
| python-dateutil | 2.9.0.post0 | Apache Software; BSD | https://github.com/dateutil/dateutil |
| python-dotenv | 1.2.3 | BSD-3-Clause | https://github.com/theskumar/python-dotenv |
| redis | 8.1.0 | MIT | https://github.com/redis/redis-py |
| scikit-learn | 1.9.1 | BSD-3-Clause | https://scikit-learn.org |
| scipy | 1.18.1 | BSD | https://scipy.org/ |
| six | 1.17.0 | MIT | https://github.com/benjaminp/six |
| starlette | 1.7.0 | BSD-3-Clause | https://github.com/Kludex/starlette |
| threadpoolctl | 3.7.0 | BSD-3-Clause | https://github.com/joblib/threadpoolctl |
| typing-inspection | 0.4.4 | MIT | https://github.com/pydantic/typing-inspection |
| typing_extensions | 4.16.0 | PSF-2.0 | https://github.com/python/typing_extensions |
| uvicorn | 0.54.0 | BSD-3-Clause | https://uvicorn.dev/ |
| uvloop | 0.22.1 | Apache Software; MIT | — |
| watchfiles | 1.3.0 | MIT | https://github.com/samuelcolvin/watchfiles |
| websockets | 17.1 | BSD-3-Clause | https://github.com/python-websockets/websockets |

## 5. Полный список npm-пакетов дашборда (production-сборка)

Лицензии: MIT — 129, ISC — 16, BSD-3-Clause — 5, BSD-2-Clause — 2, Apache-2.0 — 2, (MIT OR Apache-2.0) — 1, 0BSD — 1, (MIT OR CC0-1.0) — 1.

| Пакет | Версия | Лицензия | Репозиторий |
|---|---|---|---|
| @ant-design/colors | 8.0.1 | MIT | https://github.com/ant-design/ant-design-colors |
| @ant-design/cssinjs-utils | 2.1.2 | MIT | https://github.com/ant-design/cssinjs-util |
| @ant-design/cssinjs | 2.1.2 | MIT | https://github.com/ant-design/cssinjs |
| @ant-design/fast-color | 3.0.1 | MIT | https://github.com/ant-design/fast-color |
| @ant-design/icons-svg | 4.6.0 | MIT | https://github.com/ant-design/ant-design-icons |
| @ant-design/icons | 6.3.4 | MIT | https://github.com/ant-design/ant-design-icons.git#master |
| @ant-design/react-slick | 2.0.0 | MIT | https://github.com/ant-design/react-slick |
| @babel/runtime | 7.29.7 | MIT | https://github.com/babel/babel |
| @babel/runtime | 8.0.5 | MIT | https://github.com/babel/babel |
| @emotion/hash | 0.8.0 | MIT | https://github.com/emotion-js/emotion.git#master |
| @emotion/unitless | 0.7.5 | MIT | https://github.com/emotion-js/emotion.git#master |
| @inquirer/ansi | 2.0.8 | MIT | https://github.com/SBoudrias/Inquirer.js |
| @inquirer/confirm | 6.3.2 | MIT | https://github.com/SBoudrias/Inquirer.js |
| @inquirer/core | 12.0.3 | MIT | https://github.com/SBoudrias/Inquirer.js |
| @inquirer/figures | 2.0.9 | MIT | https://github.com/SBoudrias/Inquirer.js |
| @inquirer/type | 4.1.1 | MIT | https://github.com/SBoudrias/Inquirer.js |
| @mapbox/jsonlint-lines-primitives | 2.0.3 | MIT | https://github.com/mapbox/jsonlint |
| @mapbox/point-geometry | 1.1.0 | ISC | https://github.com/mapbox/point-geometry |
| @mapbox/tiny-sdf | 2.2.0 | BSD-2-Clause | https://github.com/mapbox/tiny-sdf |
| @mapbox/unitbezier | 1.0.0 | BSD-2-Clause | https://github.com/mapbox/unitbezier |
| @mapbox/vector-tile | 3.0.0 | BSD-3-Clause | https://github.com/mapbox/vector-tile-js |
| @maplibre/geojson-vt | 6.1.1 | ISC | https://github.com/maplibre/geojson-vt |
| @maplibre/maplibre-gl-style-spec | 26.4.4 | ISC | https://github.com/maplibre/maplibre-style-spec |
| @maplibre/mlt | 1.3.0 | (MIT OR Apache-2.0) | https://github.com/maplibre/maplibre-tile-spec |
| @maplibre/vt-pbf | 4.3.2 | MIT | https://github.com/maplibre/vt-pbf |
| @mswjs/interceptors | 0.41.9 | MIT | https://github.com/mswjs/interceptors |
| @open-draft/deferred-promise | 2.2.0 | MIT | https://github.com/open-draft/deferred-promise |
| @open-draft/deferred-promise | 3.0.0 | MIT | https://github.com/open-draft/deferred-promise |
| @open-draft/logger | 0.3.0 | MIT | https://github.com/open-draft/logger |
| @open-draft/until | 2.1.0 | MIT | https://github.com/open-draft/until |
| @rc-component/async-validator | 6.0.0 | MIT | https://github.com/react-component/async-validator |
| @rc-component/cascader | 1.22.0 | MIT | https://github.com/react-component/cascader |
| @rc-component/checkbox | 2.0.0 | MIT | https://github.com/react-component/checkbox |
| @rc-component/collapse | 1.2.0 | MIT | https://github.com/react-component/collapse |
| @rc-component/color-picker | 3.1.1 | MIT | https://github.com/react-component/color-picker |
| @rc-component/context | 2.0.2 | MIT | https://github.com/react-component/context |
| @rc-component/dialog | 1.10.0 | MIT | https://github.com/react-component/dialog |
| @rc-component/drawer | 1.4.2 | MIT | https://github.com/react-component/drawer |
| @rc-component/dropdown | 1.0.3 | MIT | https://github.com/react-component/dropdown |
| @rc-component/form | 1.8.6 | MIT | https://github.com/react-component/field-form |
| @rc-component/image | 1.10.0 | MIT | https://github.com/react-component/image |
| @rc-component/input-number | 1.6.2 | MIT | https://github.com/react-component/input-number |
| @rc-component/input | 1.3.1 | MIT | https://github.com/react-component/input |
| @rc-component/listy | 1.2.3 | MIT | https://github.com/react-component/listy |
| @rc-component/mentions | 1.12.0 | MIT | https://github.com/react-component/mentions |
| @rc-component/menu | 1.5.0 | MIT | https://github.com/react-component/menu |
| @rc-component/mini-decimal | 1.1.4 | MIT | https://github.com/react-component/mini-decimal |
| @rc-component/motion | 1.3.3 | MIT | https://github.com/react-component/motion |
| @rc-component/mutate-observer | 2.0.1 | MIT | https://github.com/react-component/mutate-observer |
| @rc-component/notification | 2.0.8 | MIT | https://github.com/react-component/notification |
| @rc-component/overflow | 1.0.1 | MIT | https://github.com/react-component/overflow |
| @rc-component/pagination | 1.4.0 | MIT | https://github.com/react-component/pagination |
| @rc-component/picker | 1.12.2 | MIT | https://github.com/react-component/picker |
| @rc-component/portal | 2.2.1 | MIT | https://github.com/react-component/portal |
| @rc-component/progress | 1.0.3 | MIT | https://github.com/react-component/progress |
| @rc-component/qrcode | 2.0.0 | MIT | https://github.com/react-component/qrcode |
| @rc-component/rate | 1.0.1 | MIT | https://github.com/react-component/rate |
| @rc-component/resize-observer | 1.1.2 | MIT | https://github.com/react-component/resize-observer |
| @rc-component/segmented | 1.4.0 | MIT | https://github.com/react-component/segmented |
| @rc-component/select | 1.10.1 | MIT | https://github.com/react-component/select |
| @rc-component/slider | 1.1.1 | MIT | https://github.com/react-component/slider |
| @rc-component/steps | 1.2.3 | MIT | https://github.com/react-component/steps |
| @rc-component/switch | 1.0.3 | MIT | https://github.com/react-component/switch |
| @rc-component/table | 1.11.1 | MIT | https://github.com/react-component/table |
| @rc-component/tabs | 1.13.0 | MIT | https://github.com/react-component/tabs |
| @rc-component/tooltip | 1.5.2 | MIT | https://github.com/react-component/tooltip |
| @rc-component/tour | 2.4.0 | MIT | https://github.com/react-component/tour |
| @rc-component/tree-select | 1.16.1 | MIT | https://github.com/react-component/tree-select |
| @rc-component/tree | 1.4.0 | MIT | https://github.com/react-component/tree |
| @rc-component/trigger | 3.10.1 | MIT | https://github.com/react-component/trigger |
| @rc-component/upload | 1.1.1 | MIT | https://github.com/react-component/upload |
| @rc-component/util | 1.13.0 | MIT | https://github.com/react-component/util |
| @rc-component/virtual-list | 1.5.2 | MIT | https://github.com/react-component/virtual-list |
| @remix-run/route-pattern | 0.22.1 | MIT | https://github.com/remix-run/remix |
| @tanstack/query-core | 5.103.2 | MIT | https://github.com/TanStack/query |
| @tanstack/react-query | 5.103.2 | MIT | https://github.com/TanStack/query |
| @types/geojson | 7946.0.16 | MIT | https://github.com/DefinitelyTyped/DefinitelyTyped |
| @types/node | 22.20.4 | MIT | https://github.com/DefinitelyTyped/DefinitelyTyped |
| @types/set-cookie-parser | 2.4.10 | MIT | https://github.com/DefinitelyTyped/DefinitelyTyped |
| @types/statuses | 2.0.6 | MIT | https://github.com/DefinitelyTyped/DefinitelyTyped |
| ansi-regex | 5.0.1 | MIT | https://github.com/chalk/ansi-regex |
| ansi-styles | 4.3.0 | MIT | https://github.com/chalk/ansi-styles |
| antd | 6.6.5 | MIT | https://github.com/ant-design/ant-design |
| bidi-js | 1.1.0 | MIT | https://github.com/lojjic/bidi-js |
| cli-width | 4.1.0 | ISC | https://github.com/knownasilya/cli-width |
| cliui | 8.0.1 | ISC | https://github.com/yargs/cliui |
| clsx | 2.1.1 | MIT | https://github.com/lukeed/clsx |
| color-convert | 2.0.1 | MIT | https://github.com/Qix-/color-convert |
| color-name | 1.1.4 | MIT | https://github.com/colorjs/color-name |
| compute-scroll-into-view | 3.1.1 | MIT | https://github.com/scroll-into-view/compute-scroll-into-view |
| cookie-es | 3.1.1 | MIT | https://github.com/unjs/cookie-es |
| cookie | 1.1.1 | MIT | https://github.com/jshttp/cookie |
| csstype | 3.2.3 | MIT | https://github.com/frenic/csstype |
| dayjs | 1.11.23 | MIT | https://github.com/iamkun/dayjs |
| earcut | 3.2.3 | ISC | https://github.com/mapbox/earcut |
| echarts | 6.1.0 | Apache-2.0 | https://github.com/apache/echarts |
| emoji-regex | 8.0.0 | MIT | https://github.com/mathiasbynens/emoji-regex |
| escalade | 3.2.0 | MIT | https://github.com/lukeed/escalade |
| fast-string-truncated-width | 3.0.3 | MIT | https://github.com/fabiospampinato/fast-string-truncated-width |
| fast-string-width | 3.0.2 | MIT | https://github.com/fabiospampinato/fast-string-width |
| fast-wrap-ansi | 0.2.2 | MIT | https://github.com/43081j/fast-wrap-ansi |
| get-caller-file | 2.0.5 | ISC | https://github.com/stefanpenner/get-caller-file |
| gl-matrix | 3.4.4 | MIT | https://github.com/toji/gl-matrix |
| graphql | 16.14.2 | MIT | https://github.com/graphql/graphql-js |
| headers-polyfill | 5.0.1 | MIT | https://github.com/mswjs/headers-polyfill |
| is-fullwidth-code-point | 3.0.0 | MIT | https://github.com/sindresorhus/is-fullwidth-code-point |
| is-mobile | 5.0.0 | MIT | https://github.com/juliangruber/is-mobile |
| is-node-process | 1.2.0 | MIT | https://github.com/mswjs/is-node-process |
| json-stringify-pretty-compact | 4.0.0 | MIT | https://github.com/lydell/json-stringify-pretty-compact |
| json2mq | 0.2.0 | MIT | https://github.com/akiran/json2mq |
| kdbush | 4.1.0 | ISC | https://github.com/mourner/kdbush |
| maplibre-gl | 6.11.2 | BSD-3-Clause | https://github.com/maplibre/maplibre-gl-js |
| minimist | 1.2.8 | MIT | https://github.com/minimistjs/minimist |
| msw | 2.15.0 | MIT | https://github.com/mswjs/msw |
| murmurhash-js | 1.0.0 | MIT | https://github.com/mikolalysenko/murmurhash-js |
| mute-stream | 3.0.0 | ISC | https://github.com/npm/mute-stream |
| outvariant | 1.4.3 | MIT | https://github.com/open-draft/outvariant |
| path-to-regexp | 6.3.0 | MIT | https://github.com/pillarjs/path-to-regexp |
| pbf | 5.1.2 | BSD-3-Clause | https://github.com/mapbox/pbf |
| picocolors | 1.1.1 | ISC | https://github.com/alexeyraspopov/picocolors |
| potpack | 2.1.0 | ISC | https://github.com/mapbox/potpack |
| protocol-buffers-schema | 3.6.1 | MIT | https://github.com/mafintosh/protocol-buffers-schema |
| quickselect | 3.0.0 | ISC | https://github.com/mourner/quickselect |
| react-dom | 19.3.0 | MIT | https://github.com/react/react |
| react-is | 19.3.0 | MIT | https://github.com/react/react |
| react-router | 8.4.0 | MIT | https://github.com/remix-run/react-router |
| react | 19.3.0 | MIT | https://github.com/react/react |
| require-directory | 2.1.1 | MIT | https://github.com/troygoode/node-require-directory |
| require-from-string | 2.0.2 | MIT | https://github.com/floatdrop/require-from-string |
| resolve-protobuf-schema | 2.1.0 | MIT | https://github.com/mafintosh/resolve-protobuf-schema |
| rettime | 0.11.12 | MIT | https://github.com/kettanaito/rettime |
| scheduler | 0.28.0 | MIT | https://github.com/react/react |
| scroll-into-view-if-needed | 3.1.0 | MIT | https://github.com/scroll-into-view/scroll-into-view-if-needed |
| set-cookie-parser | 3.1.2 | MIT | https://github.com/nfriedly/set-cookie-parser |
| signal-exit | 4.1.0 | ISC | https://github.com/tapjs/signal-exit |
| statuses | 2.0.2 | MIT | https://github.com/jshttp/statuses |
| strict-event-emitter | 0.5.1 | MIT | https://github.com/open-draft/strict-event-emitter |
| string-convert | 0.2.1 | MIT | https://github.com/akiran/string-convert |
| string-width | 4.2.3 | MIT | https://github.com/sindresorhus/string-width |
| strip-ansi | 6.0.1 | MIT | https://github.com/chalk/strip-ansi |
| stylis | 4.4.0 | MIT | https://github.com/thysultan/stylis.js |
| tagged-tag | 1.0.0 | MIT | https://github.com/sindresorhus/tagged-tag |
| throttle-debounce | 5.0.2 | MIT | https://github.com/niksy/throttle-debounce |
| tinyqueue | 3.0.0 | ISC | https://github.com/mourner/tinyqueue |
| tldts-core | 7.4.15 | MIT | https://github.com/remusao/tldts |
| tldts | 7.4.15 | MIT | https://github.com/remusao/tldts |
| tough-cookie | 6.0.2 | BSD-3-Clause | https://github.com/salesforce/tough-cookie |
| tslib | 2.3.0 | 0BSD | https://github.com/Microsoft/tslib |
| type-fest | 5.10.0 | (MIT OR CC0-1.0) | https://github.com/sindresorhus/type-fest |
| typescript | 6.0.3 | Apache-2.0 | https://github.com/microsoft/TypeScript |
| undici-types | 6.21.0 | MIT | https://github.com/nodejs/undici |
| until-async | 3.0.2 | MIT | https://github.com/kettanaito/until-async |
| wrap-ansi | 7.0.0 | MIT | https://github.com/chalk/wrap-ansi |
| y18n | 5.0.8 | ISC | https://github.com/yargs/y18n |
| yargs-parser | 21.1.1 | ISC | https://github.com/yargs/yargs-parser |
| yargs | 17.7.3 | MIT | https://github.com/yargs/yargs |
| zrender | 6.1.0 | BSD-3-Clause | https://github.com/ecomfe/zrender |
