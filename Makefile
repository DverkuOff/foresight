# Foresight · запуск демо одной командой. Нужны только make, curl и Docker Engine ≥ 25 с Docker Compose ≥ 2.21
# (проверено на Docker 29.2 и Compose v5.0.2; `make up` проверяет версии сам).
#
#   make dataset   скачать датасет организаторов (один раз)
#   make demo      собрать и поднять стек, дождаться healthy, запустить replayer (test, x30, с 06:00, по кругу)
#   make status    кратко: /health сервисов, часы потока, лаг consumer group, ТС, replayer
#   make chaos     отказы и восстановление: ingest, Redis, PostgreSQL, ML-сервис, replayer (scripts/chaos.sh)
#   make help      все цели и переменные
#
# Переменные задаются в командной строке (make demo REPLAY_SPEED=60 REPLAY_START=07:00 REPLAY_UNTIL=10:00)
# или в .env: его читают и make, и docker compose. Порты — те же API_PORT, INGEST_PORT, … что и в
# docker-compose.yml (номер порта или адрес:порт).

MAKEFILE := $(lastword $(MAKEFILE_LIST))
-include .env

SHELL := /bin/bash
.DEFAULT_GOAL := help

COMPOSE        ?= docker compose
DEMO_HOST      ?= localhost
API_PORT       ?= 8000
INGEST_PORT    ?= 8001
PREDICTOR_PORT ?= 8002
ML_PORT        ?= 8003
REPLAYER_PORT  ?= 8010
EMULATOR_PORT  ?= 18080
NDTP_PORT      ?= 9201
DASHBOARD_PORT ?= 8080
GRAFANA_PORT   ?= 3000
PROMETHEUS_PORT ?= 9090
DATASET_DIR    ?= ./dataset
WAIT_TIMEOUT   ?= 300

# воспроизведение для make demo / make replay-start (и для автостарта replayer: REPLAY_AUTOSTART=1)
REPLAY_SPLIT   ?= test
REPLAY_SPEED   ?= 30
REPLAY_START   ?= 06:00
REPLAY_UNTIL   ?=
REPLAY_UNITS   ?=
REPLAY_LOOP    ?= 1
REPLAY_MODE    ?= ndtp
# Автостарт replayer: контейнер сам запускает воспроизведение REPLAY_* при каждом старте (перезапуск контейнера,
# перезагрузка хоста). `make demo` включает его, `make emulator` выключает; пусто — `make up` сохраняет значение
# работающего контейнера (без контейнера — выключен).
REPLAY_AUTOSTART ?=

# make chaos: сценарии scripts/chaos.sh (all, ingest, ingest-kill, redis, postgres, ml, replayer) и длительность отказа
SCENARIO       ?= all
DOWN_S         ?= 15

# make up проверяет версии Docker и Compose (DOCKER_CHECK=0 — пропустить)
DOCKER_CHECK   ?= 1

# make emulator: устройства из датасета (ingest знает их tr_id) и доп. аргументы scripts/emulator_demo.py
EMU_UNITS      ?= 664030,794446,663271
EMU_ARGS       ?=
EMULATOR_IMAGE ?= ndtp-telemetry-emulator:1.0

# make logs
SERVICES       ?=
TAIL           ?= 100

export API_PORT INGEST_PORT PREDICTOR_PORT ML_PORT REPLAYER_PORT EMULATOR_PORT NDTP_PORT DASHBOARD_PORT GRAFANA_PORT \
       PROMETHEUS_PORT DATASET_DIR DEMO_HOST COMPOSE
export REPLAY_SPLIT REPLAY_SPEED REPLAY_START REPLAY_UNTIL REPLAY_UNITS REPLAY_LOOP REPLAY_MODE DOWN_S
# Сборка без attestation-манифеста. С хранилищем образов containerd (Docker 29) он получает новый digest при каждой
# сборке, и `make up`/`make demo` на работающем стеке пересоздавали бы контейнеры, хотя ничего не изменилось.
# `provenance: false` в docker-compose.yml compose v5 не передаёт в сборку, а старые версии эту переменную игнорируют.
export BUILDX_NO_DEFAULT_ATTESTATIONS := 1

addr          = $(if $(findstring :,$(1)),$(1),$(DEMO_HOST):$(1))
API_URL       = http://$(call addr,$(API_PORT))
INGEST_URL    = http://$(call addr,$(INGEST_PORT))
PREDICTOR_URL = http://$(call addr,$(PREDICTOR_PORT))
ML_URL        = http://$(call addr,$(ML_PORT))
REPLAYER_URL  = http://$(call addr,$(REPLAYER_PORT))
EMULATOR_URL  = http://$(call addr,$(EMULATOR_PORT))
DASHBOARD_URL = http://$(call addr,$(DASHBOARD_PORT))
GRAFANA_URL   = http://$(call addr,$(GRAFANA_PORT))/grafana
PROMETHEUS_URL = http://$(call addr,$(PROMETHEUS_PORT))

json_bool   = $(if $(filter 1 true yes on,$(1)),true,false)
REPLAY_BODY = {"split":"$(REPLAY_SPLIT)","speed":$(REPLAY_SPEED),"start":"$(REPLAY_START)","until":"$(REPLAY_UNTIL)","units":[$(REPLAY_UNITS)],"loop":$(call json_bool,$(REPLAY_LOOP)),"mode":"$(REPLAY_MODE)"}
EMULATOR_OFF = {"targetHost":"ingest","targetPort":9201,"units":[]}

# scripts/emulator_demo.py (только стандартная библиотека) выполняется python-ом контейнера replayer:
# на хосте python не нужен, эмулятор и ingest видны по именам сети compose
EMULATOR_SCRIPT = $(COMPOSE) exec -T replayer python - --emulator http://emulator:18080 --target-host ingest --target-port 9201

# shell-помощники рецептов: `jv поле` печатает первое значение поля JSON со stdin (строку или скаляр),
# `get url` печатает тело ответа и " HTTP<код>" (000 — нет ответа)
SH = jv() { grep -oE "\"$$1\":(\"[^\"]*\"|[^,{}]*)" | head -n1 | sed -E 's/^"[^"]*"://; s/^"//; s/"$$//'; }; \
     dep() { grep -oE "\"$$1\":\{\"state\":\"[a-z]+\"" | head -n1 | sed -E 's/.*:"//; s/"$$//'; }; \
     get() { curl -s --max-time 3 -w ' HTTP%{http_code}' "$$1"; }

.PHONY: help dataset submission docs check-docker check-dataset check-env emulator-image up demo replay-ensure replay-start replay-stop emulator \
        emulator-stop status chaos urls logs down clean

help: ## эта справка
	@echo "Foresight · make <цель> [ПЕРЕМЕННАЯ=значение …]"
	@echo
	@awk -F':.*## ' '/^[a-z-]+:.*## /{printf "  make %-14s %s\n", $$1, $$2}' $(MAKEFILE)
	@echo
	@echo "Воспроизведение: REPLAY_SPLIT=$(REPLAY_SPLIT) REPLAY_SPEED=$(REPLAY_SPEED) REPLAY_START=$(REPLAY_START)" \
	  "REPLAY_UNTIL=$(REPLAY_UNTIL) REPLAY_UNITS=$(REPLAY_UNITS) REPLAY_LOOP=$(REPLAY_LOOP) REPLAY_MODE=$(REPLAY_MODE)" \
	  "REPLAY_AUTOSTART=$(REPLAY_AUTOSTART)"
	@echo "Эмулятор: EMU_UNITS=$(EMU_UNITS) EMU_ARGS=$(EMU_ARGS)"
	@echo "Chaos: SCENARIO=$(SCENARIO) DOWN_S=$(DOWN_S)"
	@echo "Стек: DATASET_DIR=$(DATASET_DIR) DEMO_HOST=$(DEMO_HOST) WAIT_TIMEOUT=$(WAIT_TIMEOUT) API_PORT=$(API_PORT)" \
	  "INGEST_PORT=$(INGEST_PORT) PREDICTOR_PORT=$(PREDICTOR_PORT) ML_PORT=$(ML_PORT) REPLAYER_PORT=$(REPLAYER_PORT)" \
	  "EMULATOR_PORT=$(EMULATOR_PORT) NDTP_PORT=$(NDTP_PORT) DASHBOARD_PORT=$(DASHBOARD_PORT)" \
	  "GRAFANA_PORT=$(GRAFANA_PORT) PROMETHEUS_PORT=$(PROMETHEUS_PORT) DOCKER_CHECK=$(DOCKER_CHECK)"

dataset: ## скачать датасет организаторов (Яндекс.Диск из ТЗ) в DATASET_DIR с проверкой sha256
	@bash scripts/get_dataset.sh "$(DATASET_DIR)"

# сабмит по validate в образе ml-service (Python на хосте не нужен): BUNDLE=v1 — модель сабмита 0.87636, v2 — кандидат
BUNDLE ?= v1
submission: check-env check-dataset ## прогноз по validate → artifacts/submission.csv (sample_id;prediction), модель BUNDLE=v1|v2
	@mkdir -p artifacts
	@$(COMPOSE) build -q ml-service
	@$(COMPOSE) run --rm --no-deps --entrypoint python --user "$$(id -u):$$(id -g)" -e MT_DATASET_DIR=/data \
	  -v "$(abspath $(DATASET_DIR)):/data:ro" -v "$(CURDIR)/artifacts:/out" \
	  ml-service -m ml.submit --bundle $(BUNDLE) --out /out/submission.csv
	@echo "готово: artifacts/submission.csv ($$(($$(wc -l < artifacts/submission.csv) - 1)) строк, модель $(BUNDLE))"

# обучение в том же образе (Python на хосте не нужен): рецепт v1 — CV по семействам ТС, оценка на test, финал на
# train + test; ~1 мин на CPU. Упаковать в реестр: uv run python -m ml.registry pack --meta artifacts/model_v1.json
train: check-env check-dataset ## обучение CatBoost (рецепт модели сабмита) в образе ml-service → artifacts/model_v1*.cbm, отчёт artifacts/model_v1.json
	@mkdir -p artifacts
	@$(COMPOSE) build -q ml-service
	@$(COMPOSE) run --rm --no-deps --entrypoint python --user "$$(id -u):$$(id -g)" -e MT_DATASET_DIR=/data \
	  -v "$(abspath $(DATASET_DIR)):/data:ro" -v "$(CURDIR)/artifacts:/app/artifacts" \
	  ml-service -m ml.train --skip-grid --tag docker
	@echo "готово: artifacts/model_v1.json (метрики CV и test) и artifacts/model_v1_*.cbm"

docs: ## сайт документации в site/ (Sphinx + справочник по коду + OpenAPI всех сервисов; нужен uv)
	@rm -rf site && uv run --group docs --extra ml --extra backend sphinx-build -q -b html docs/sphinx site -d .sphinx-cache
	@uv run --group docs --extra ml --extra backend python scripts/export_openapi.py site/api
	@echo "готово: site/index.html, OpenAPI — site/api/index.html"

# healthcheck start_interval нужен Docker Engine 25+, `up --wait --wait-timeout` — Compose 2.21+
check-docker:
	@[ "$(DOCKER_CHECK)" = 0 ] && exit 0; \
	dv=$$(docker version --format '{{.Server.Version}}' 2>/dev/null); \
	cv=$$($(COMPOSE) version --short 2>/dev/null | sed 's/^v//'); \
	if [ -z "$$dv" ] || [ -z "$$cv" ]; then \
	  echo "нет Docker или плагина compose: docker version / $(COMPOSE) version не отвечают (демон запущен?)" >&2; \
	  exit 1; \
	fi; \
	dmaj=$${dv%%.*}; cmaj=$${cv%%.*}; cmin=$${cv#*.}; cmin=$${cmin%%.*}; \
	if ! [ "$$dmaj" -ge 25 ] 2>/dev/null || ! { [ "$$cmaj" -gt 2 ] || { [ "$$cmaj" -eq 2 ] && [ "$$cmin" -ge 21 ]; }; } 2>/dev/null; then \
	  echo "нужны Docker Engine ≥ 25 и Docker Compose ≥ 2.21, установлены Engine $$dv и Compose $$cv" >&2; \
	  echo "(проверено на 29.2 / v5.0.2; DOCKER_CHECK=0 — пропустить проверку)" >&2; \
	  exit 1; \
	fi

# .env с паролем администратора Grafana: при первом запуске — копия .env.example со случайным паролем
check-env:
	@if [ ! -f .env ]; then \
	  pw=$$(LC_ALL=C tr -dc 'A-Za-z0-9' < /dev/urandom | head -c 20); \
	  sed "s/^GF_SECURITY_ADMIN_PASSWORD=.*/GF_SECURITY_ADMIN_PASSWORD=$$pw/" .env.example > .env; \
	  echo "создан .env: пароль администратора Grafana (admin) — GF_SECURITY_ADMIN_PASSWORD в .env"; \
	elif ! grep -q '^GF_SECURITY_ADMIN_PASSWORD=.' .env; then \
	  pw=$$(LC_ALL=C tr -dc 'A-Za-z0-9' < /dev/urandom | head -c 20); \
	  echo "GF_SECURITY_ADMIN_PASSWORD=$$pw" >> .env; \
	  echo "в .env добавлен пароль администратора Grafana (GF_SECURITY_ADMIN_PASSWORD)"; \
	fi

check-dataset:
	@test -f "$(DATASET_DIR)/$(REPLAY_SPLIT)/traffic.csv" || { \
	  echo "нет датасета: $(DATASET_DIR)/$(REPLAY_SPLIT)/traffic.csv" >&2; \
	  echo "скачайте его: make dataset (или задайте DATASET_DIR=/путь/к/dataset)" >&2; exit 1; }

emulator-image:
	@docker image inspect $(EMULATOR_IMAGE) >/dev/null 2>&1 || { \
	  tar="$(DATASET_DIR)/ndtp-telemetry-emulator.tar"; \
	  test -f "$$tar" || { echo "нет образа $(EMULATOR_IMAGE) и файла $$tar" >&2; exit 1; }; \
	  echo "загружаю образ эмулятора из $$tar"; docker load -i "$$tar"; }

up: check-docker check-env check-dataset emulator-image ## собрать образы, поднять стек и дождаться healthy всех сервисов
	@t0=$$(date +%s); \
	auto="$(REPLAY_AUTOSTART)"; \
	if [ -z "$$auto" ]; then \
	  id=$$($(COMPOSE) ps -a -q replayer 2>/dev/null | head -n1); \
	  [ -n "$$id" ] && auto=$$(docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' "$$id" \
	    2>/dev/null | sed -n 's/^FORESIGHT_REPLAY_AUTOSTART=//p' | head -n1); \
	fi; \
	export REPLAY_AUTOSTART=$${auto:-0}; \
	$(COMPOSE) build || exit 1; \
	t1=$$(date +%s); \
	if ! $(COMPOSE) up -d --wait --wait-timeout $(WAIT_TIMEOUT); then \
	  $(COMPOSE) ps -a; echo "стек не стал healthy за $(WAIT_TIMEOUT) с: make logs" >&2; exit 1; \
	fi; \
	t2=$$(date +%s); \
	$(COMPOSE) ps --format 'table {{.Service}}\t{{.Status}}'; \
	echo "сборка образов $$((t1 - t0)) с, старт до healthy $$((t2 - t1)) с;" \
	  "автостарт replayer: $$([ "$$REPLAY_AUTOSTART" = 1 ] && echo включён || echo выключен)"

demo: ## up с автостартом replayer (test, x30, с 06:00, по кругу) + адреса; демо переживает перезапуски
	@$(MAKE) --no-print-directory up REPLAY_AUTOSTART=1
	@$(MAKE) --no-print-directory replay-ensure
	@$(MAKE) --no-print-directory urls

# один источник за раз: остановить эмулятор; запустить воспроизведение, если replayer его не ведёт (автостарт
# уже идёт или был остановлен make replay-stop / make emulator); дождаться ТС в api
replay-ensure:
	@$(SH); \
	if curl -s --max-time 3 $(EMULATOR_URL)/api/config | grep -q '"unitId"'; then \
	  curl -s --max-time 10 -o /dev/null -H 'Content-Type: application/json' -X POST -d '$(EMULATOR_OFF)' \
	    $(EMULATOR_URL)/api/config && echo "эмулятор остановлен: один источник за раз (у ingest одни часы потока)"; \
	fi; \
	out=$$(curl -s --max-time 5 $(REPLAYER_URL)/replay/status); state=$$(echo "$$out" | jv state); \
	case "$$state" in \
	  running|waiting|loading|created) echo "replayer: $$state (автостарт), epoch $$(echo "$$out" | jv epoch)";; \
	  *) echo "replayer: $${state:-не отвечает} → POST /replay/start" '$(REPLAY_BODY)'; \
	     out=$$(curl -sS --fail-with-body --max-time 120 -H 'Content-Type: application/json' -X POST \
	       -d '$(REPLAY_BODY)' $(REPLAYER_URL)/replay/start) || { echo "replayer не запустился: $$out" >&2; exit 1; };; \
	esac; \
	n=0; for i in $$(seq 1 30); do \
	  n=$$(curl -s --max-time 2 $(API_URL)/api/vehicles | grep -oE '"tr_id":[0-9]+' | wc -l | tr -d ' '); \
	  [ "$$n" -gt 0 ] && break; sleep 1; \
	done; \
	out=$$(curl -s --max-time 5 $(REPLAYER_URL)/replay/status); \
	echo "replayer: $$(echo "$$out" | jv state) $$(echo "$$out" | jv split) x$$(echo "$$out" | jv speed)," \
	  "ТС $$(echo "$$out" | jv units), данные $$(echo "$$out" | jv data_first) … $$(echo "$$out" | jv data_last)"; \
	echo "api: ТС с tr_id — $$n (через $$i с), часы потока $$(curl -s --max-time 2 $(API_URL)/api/vehicles | jv stream_time)"

replay-start: check-env ## (пере)запустить воспроизведение с параметрами REPLAY_* (стек уже поднят)
	@$(SH); \
	if curl -s --max-time 3 $(EMULATOR_URL)/api/config | grep -q '"unitId"'; then \
	  curl -s --max-time 10 -o /dev/null -H 'Content-Type: application/json' -X POST -d '$(EMULATOR_OFF)' \
	    $(EMULATOR_URL)/api/config && echo "эмулятор остановлен: один источник за раз (у ingest одни часы потока)"; \
	fi; \
	echo 'replayer: POST /replay/start $(REPLAY_BODY)'; \
	out=$$(curl -sS --fail-with-body --max-time 120 -H 'Content-Type: application/json' -X POST \
	  -d '$(REPLAY_BODY)' $(REPLAYER_URL)/replay/start) || { echo "replayer не запустился: $$out" >&2; exit 1; }; \
	echo "replayer: $$(echo "$$out" | jv state), epoch $$(echo "$$out" | jv epoch), ТС $$(echo "$$out" | jv units)," \
	  "пакетов $$(echo "$$out" | jv packets_total), данные $$(echo "$$out" | jv data_first) … $$(echo "$$out" | jv data_last)"; \
	n=0; for i in $$(seq 1 30); do \
	  n=$$(curl -s --max-time 2 $(API_URL)/api/vehicles | grep -oE '"tr_id":[0-9]+' | wc -l | tr -d ' '); \
	  [ "$$n" -gt 0 ] && break; sleep 1; \
	done; \
	echo "api: ТС с tr_id — $$n (через $$i с), часы потока $$(curl -s --max-time 2 $(API_URL)/api/vehicles | jv stream_time)"

replay-stop: check-env ## остановить воспроизведение (стек продолжает работать)
	@$(SH); \
	out=$$(curl -sS --fail-with-body --max-time 30 -X POST $(REPLAYER_URL)/replay/stop) \
	  || { echo "replayer: $$out" >&2; exit 1; }; \
	echo "replayer: $$(echo "$$out" | jv state), отправлено $$(echo "$$out" | jv packets_sent)"

emulator: check-env ## остановить replayer (и его автостарт) и пустить живой поток эмулятора в ingest (EMU_UNITS, EMU_ARGS)
	@curl -s --max-time 30 -o /dev/null -X POST $(REPLAYER_URL)/replay/stop \
	  && echo "replayer остановлен: один источник за раз (у ingest одни часы потока)" || true
	@# без автостарта: перезапуск контейнера replayer не должен вернуть второй источник
	@REPLAY_AUTOSTART=0 $(COMPOSE) up -d --no-deps --wait --wait-timeout $(WAIT_TIMEOUT) replayer >/dev/null 2>&1 \
	  || echo "replayer: не удалось пересоздать без автостарта (make logs SERVICES=replayer)" >&2
	@$(EMULATOR_SCRIPT) $(if $(EMU_UNITS),--unit-ids $(EMU_UNITS)) $(EMU_ARGS) < scripts/emulator_demo.py

emulator-stop: check-env ## остановить поток эмулятора
	@$(EMULATOR_SCRIPT) --stop < scripts/emulator_demo.py

status: check-env ## кратко: контейнеры, /health сервисов, часы потока, лаг consumer group, ТС, replayer
	@$(COMPOSE) ps -a --format 'table {{.Service}}\t{{.Status}}'
	@$(SH); echo; warn=(); \
	for s in api:$(API_URL) ingest:$(INGEST_URL) predictor:$(PREDICTOR_URL); do \
	  name=$${s%%:*}; out=$$(get "$${s#*:}/health"); body=$${out% HTTP*}; \
	  printf '%-10s HTTP %s  %-8s redis %s, postgres %s%s\n' "$$name" "$${out##* HTTP}" "$$(echo "$$body" | jv status)" \
	    "$$(echo "$$body" | dep redis)" "$$(echo "$$body" | dep postgres)" \
	    "$$(echo "$$body" | grep -q '"ndtp_listening":true' && echo ', NDTP принимает')$$(echo "$$body" | grep -q '"ingest":{"state"' && echo ", ingest $$(echo "$$body" | dep ingest)")"; \
	  if [ "$${out##* HTTP}" = 000 ]; then \
	    st=$$($(COMPOSE) ps -a --format '{{.State}}' "$$name" 2>/dev/null); \
	    warn+=("$$name не отвечает (контейнер: $${st:-нет}); поднять: $(COMPOSE) up -d $$name или make up$$([ "$$st" = exited ] && echo '. Остановленный вручную контейнер (docker kill / stop) Docker сам не перезапускает')"); \
	  fi; \
	done; \
	out=$$(get $(ML_URL)/health); body=$${out% HTTP*}; \
	printf '%-10s HTTP %s  %-8s %s\n' ml-service "$${out##* HTTP}" "$$(echo "$$body" | jv status)" \
	  "модель $$(echo "$$body" | jv model_version), реестр PostgreSQL $$(echo "$$body" | jv database)"; \
	if [ "$${out##* HTTP}" != 200 ]; then \
	  warn+=("ml-service не готов (HTTP $${out##* HTTP}): predictor выдаёт прогнозы по fallback; поднять: $(COMPOSE) up -d ml-service, логи: make logs SERVICES=ml-service"); \
	fi; \
	out=$$(get $(REPLAYER_URL)/health); body=$${out% HTTP*}; rstate=$$(echo "$$body" | jv state); \
	printf '%-10s HTTP %s  %-8s %s\n' replayer "$${out##* HTTP}" "$$(echo "$$body" | jv status)" \
	  "воспроизведение $$rstate, epoch $$(echo "$$body" | jv epoch)"; \
	out=$$(get $(EMULATOR_URL)/api/config); body=$${out% HTTP*}; \
	n=$$(echo "$$body" | grep -o '"unitId"' | wc -l | tr -d ' '); \
	printf '%-10s HTTP %s  %s\n' emulator "$${out##* HTTP}" \
	  "$$([ "$$n" -gt 0 ] && echo "поток $$n ТС → $$(echo "$$body" | jv targetHost):$$(echo "$$body" | jv targetPort)" || echo 'без потока')"; \
	case "$$rstate" in running|waiting|loading|created) ;; \
	  *) [ "$$n" -gt 0 ] || warn+=("поток телеметрии не идёт: replayer $${rstate:-не отвечает}, эмулятор без потока; запустить: make replay-start (или make demo)");; \
	esac; \
	for w in "$${warn[@]}"; do echo "ВНИМАНИЕ: $$w"; done; \
	echo; \
	v=$$(curl -s --max-time 5 $(API_URL)/api/vehicles); \
	echo "ТС (api)     всего $$(echo "$$v" | jv count), с tr_id $$(echo "$$v" | grep -oE '"tr_id":[0-9]+' | wc -l | tr -d ' ')," \
	  "$$(echo "$$v" | grep -oE '"status_counts":\{[^}]*\}' | sed -E 's/"status_counts"://; s/[{}"]//g; s/:/ /g; s/,/, /g')," \
	  "degraded $$(echo "$$v" | jv degraded)"; \
	echo "часы потока  $$(echo "$$v" | jv stream_time)"; \
	g=$$($(COMPOSE) exec -T redis redis-cli XINFO GROUPS foresight:telemetry 2>/dev/null | awk \
	  'k != "" {v[k] = $$0; k = ""; next} {k = $$0} END {if (v["name"] != "") printf "группа %s: lag %s, pending %s, consumers %s", v["name"], v["lag"], v["pending"], v["consumers"]}'); \
	echo "очередь      $${g:-Redis недоступен или группы ещё нет}, длина потока $$($(COMPOSE) exec -T redis redis-cli XLEN foresight:telemetry 2>/dev/null || echo '?')"; \
	i=$$(curl -s --max-time 3 $(INGEST_URL)/api/ingest/stats); \
	echo "ingest       соединений $$(echo "$$i" | jv connections_active), пакетов/с $$(echo "$$i" | jv packets_per_s)," \
	  "nav $$(echo "$$i" | jv nav_records), CRC-ошибок $$(echo "$$i" | jv crc_errors), в буфере $$(echo "$$i" | jv bus_buffered)," \
	  "эпоха часов $$(echo "$$i" | jv clock_epoch)"; \
	p=$$(curl -s --max-time 3 $(PREDICTOR_URL)/api/predictor/stats); \
	echo "predictor    треков $$(echo "$$p" | jv tracks), точек в окнах $$(echo "$$p" | jv window_points)," \
	  "тиков $$(echo "$$p" | jv ticks) (пропущено при отставании $$(echo "$$p" | jv ticks_lagging)), событий/с $$(echo "$$p" | jv events_per_s)"; \
	echo "прогнозы     открыто $$(echo "$$p" | jv predictions_open), закрыто фактом $$(echo "$$p" | jv closed)," \
	  "источник $$(echo "$$p" | jv last_source), задним числом $$(echo "$$p" | jv retroactive), алертов $$(echo "$$p" | jv alerts)," \
	  "онлайн-MAE $$(echo "$$p" | jv online_mae_s) с (baseline $$(echo "$$p" | jv online_baseline_mae_s) с)," \
	  "тик $$(echo "$$p" | grep -oE '"tick_ms":\{"total":[0-9.]+' | sed 's/.*://') мс"; \
	r=$$(curl -s --max-time 3 $(REPLAYER_URL)/replay/status); \
	echo "replayer     $$(echo "$$r" | jv state) $$(echo "$$r" | jv split) x$$(echo "$$r" | jv speed)," \
	  "данные $$(echo "$$r" | jv data_time), отправлено $$(echo "$$r" | jv packets_sent), отставание $$(echo "$$r" | jv lag_s) с," \
	  "соединений $$(echo "$$r" | jv connections_active)/$$(echo "$$r" | jv units), переподключений $$(echo "$$r" | jv reconnects)"

chaos: check-env ## отказы и восстановление на работающем демо (SCENARIO="ingest ingest-kill redis postgres ml replayer", DOWN_S=15)
	@bash scripts/chaos.sh $(SCENARIO)

urls: ## адреса сервисов
	@echo "Foresight:"
	@echo "  ДАШБОРД          $(DASHBOARD_URL)   (карта, инциденты, нитка, честность прогноза, What-if, админка)"
	@echo "  Grafana          $(GRAFANA_URL)/   (просмотр без входа; admin — пароль в .env)   Prometheus $(PROMETHEUS_URL)"
	@echo "  API и Swagger    $(API_URL)/docs"
	@echo "  ТС: REST, WS     $(API_URL)/api/vehicles   ws://$(call addr,$(API_PORT))/ws"
	@echo "  ingest           $(INGEST_URL)/api/ingest/stats   NDTP tcp://$(call addr,$(NDTP_PORT))   Swagger $(INGEST_URL)/docs"
	@echo "  predictor        $(PREDICTOR_URL)/api/predictor/stats   метрики $(PREDICTOR_URL)/metrics"
	@echo "  ml-service       $(ML_URL)/model/info   Swagger $(ML_URL)/docs"
	@echo "  replayer         $(REPLAYER_URL)/replay/status   Swagger $(REPLAYER_URL)/docs"
	@echo "  эмулятор         $(EMULATOR_URL)/api/config   (make emulator)"
	@echo "  метрики          /metrics у api, ingest, predictor, ml-service и replayer"
	@echo "Дальше: make status · make logs · make chaos · make replay-stop · make emulator · make down"

logs: check-env ## логи с продолжением (SERVICES="ingest replayer", TAIL=100); выход — Ctrl+C
	$(COMPOSE) logs -f --tail=$(TAIL) $(SERVICES)

down: check-env ## остановить стек (данные Redis и PostgreSQL сохраняются)
	$(COMPOSE) down --remove-orphans

clean: check-env ## остановить стек и удалить тома Redis и PostgreSQL
	$(COMPOSE) down -v --remove-orphans
