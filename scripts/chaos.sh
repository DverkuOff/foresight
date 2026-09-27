#!/usr/bin/env bash
# Foresight · chaos-сценарии (docs/architecture.md §8): отказ и восстановление сервисов на работающем демо-стеке
# (make demo). Нужны только docker (compose), curl, grep, sed и awk. Печатает ok / FAIL по каждой проверке и время
# восстановления; код выхода 0 — все проверки прошли.
#
#   make chaos                               все сценарии по очереди (~4 мин)
#   make chaos SCENARIO="ingest redis"       выбранные; DOWN_S=30 — длительность отказа (по умолчанию 15 с)
#   scripts/chaos.sh ingest-kill             то же без make
#
# Сценарии:
#   ingest       docker compose stop ingest на DOWN_S с, затем start. api замечает отказ за ~10 с (ТС offline,
#                /health degraded), replayer копит пакеты и после start переподключается и досылает очередь;
#                часы потока сохраняют эпоху (досылка — опоздавшие точки, а не перезапуск источника)
#   ingest-kill  docker compose kill ingest (SIGKILL). Docker не перезапускает контейнер, остановленный вручную
#                (docker kill / docker stop): restart: unless-stopped срабатывает только при падении процесса.
#                Сценарий поднимает его сам — docker compose up -d ingest; то же делает `make up`
#   redis        stop redis на DOWN_S с: ingest принимает NDTP и копит события в памяти, api отдаёт последнее
#                состояние (degraded); после start буфер дописан, лаг 0, потерь нет
#   postgres     stop postgres на DOWN_S с: журнал копится в памяти и дописывается после start
#   replayer     docker compose restart replayer: после make demo воспроизведение продолжается само (автостарт)
#   ml           stop ml-service на DOWN_S с: predictor прогнозирует по резервной формуле, тики не прерываются;
#                после start прогнозы снова от модели
#   all          ingest ingest-kill redis postgres ml replayer (по умолчанию)
#
# Переменные окружения: COMPOSE (по умолчанию «docker compose»), DEMO_HOST, API_PORT, INGEST_PORT, PREDICTOR_PORT,
# REPLAYER_PORT (как в Makefile), DOWN_S.
set -uo pipefail

COMPOSE=${COMPOSE:-docker compose}
HOST=${DEMO_HOST:-localhost}
DOWN_S=${DOWN_S:-15}

url() { if [[ $1 == *:* ]]; then echo "http://$1"; else echo "http://$HOST:$1"; fi; }
API=$(url "${API_PORT:-8000}")
INGEST=$(url "${INGEST_PORT:-8001}")
PREDICTOR=$(url "${PREDICTOR_PORT:-8002}")
REPLAYER=$(url "${REPLAYER_PORT:-8010}")

failures=0

# ---- helpers ------------------------------------------------------------------------------------------------

jv() { grep -oE "\"$1\":(\"[^\"]*\"|[^,{}]*)" | head -n1 | sed -E 's/^"[^"]*"://; s/^"//; s/"$//'; }
dep() { grep -oE "\"$1\":\{\"state\":\"[a-z]+\"" | head -n1 | sed -E 's/.*:"//; s/"$//'; }
get() { curl -s --max-time 3 "$1"; }
say() { echo "[$(date +%H:%M:%S)] $*"; }
dc() { $COMPOSE "$@" >/dev/null 2>&1; }

pass() { echo "  ok    $*"; }
fail() { echo "  FAIL  $*"; failures=$((failures + 1)); }

# check "что проверяем" команда…: одна проверка
check() {
  local what=$1
  shift
  if "$@"; then pass "$what"; else fail "$what"; fi
}

# wait_for секунд "что ждём" команда…: ждать, пока команда не выполнится (раз в секунду)
wait_for() {
  local limit=$1 what=$2 start=$SECONDS
  shift 2
  until "$@"; do
    if ((SECONDS - start >= limit)); then
      fail "$what (не дождались за $limit с)"
      return 1
    fi
    sleep 1
  done
  pass "$what (за $((SECONDS - start)) с)"
}

ingest_stat() { get "$INGEST/api/ingest/stats" | jv "$1"; }
predictor_stat() { get "$PREDICTOR/api/predictor/stats" | jv "$1"; }
replayer_stat() { get "$REPLAYER/replay/status" | jv "$1"; }
connected() { get "$API/api/vehicles" | grep -o '"connected":true' | wc -l | tr -d ' '; }
group_lag() {
  $COMPOSE exec -T redis redis-cli XINFO GROUPS foresight:telemetry 2>/dev/null |
    awk 'k != "" {v[k] = $0; k = ""; next} {k = $0} END {print v["lag"]}'
}
clock_reset_rows() {
  $COMPOSE exec -T postgres psql -U foresight -d foresight -tAc \
    "select count(*) from service_events where kind = 'clock_reset'" 2>/dev/null | tr -d ' \r'
}

is() { [[ $1 == "$2" ]]; }
api_ingest_is() { is "$(get "$API/health" | dep ingest)" "$1"; }
health_is() { is "$(get "$1/health" | jv status)" "$2"; }
nobody_connected() { is "$(connected)" 0; }
somebody_connected() { (($(connected) > 0)); }
replayer_all_connected() {
  local body
  body=$(get "$REPLAYER/replay/status")
  [[ -n $(echo "$body" | jv units) ]] && is "$(echo "$body" | jv connections_active)" "$(echo "$body" | jv units)"
}
forecast_source() { get "$PREDICTOR/api/predictor/stats" | jv last_source; }
replayer_running() { is "$(replayer_stat state)" running; }
replayer_caught_up() { awk -v lag="$(replayer_stat lag_s)" 'BEGIN {exit !(lag != "" && lag < 2)}'; }
lag_zero() { is "$(group_lag)" 0; }
ingest_buffer_empty() { is "$(ingest_stat bus_buffered)" 0; }
ingest_buffering() {
  local n
  n=$(ingest_stat bus_buffered)
  ((${n:-0} > 0))
}
predictor_db_empty() { is "$(predictor_stat db_buffered)" 0; }
service_up() { $COMPOSE up -d --no-deps --wait --wait-timeout 90 "$1" >/dev/null 2>&1; }

# ---- scenarios ----------------------------------------------------------------------------------------------

precheck() {
  if ! replayer_running || ! somebody_connected; then
    echo "поток не идёт (replayer: $(replayer_stat state), ТС на связи: $(connected)): сначала make demo" >&2
    exit 2
  fi
}

scenario_ingest() {
  local how=$1 epoch p_resets rows units t0
  say "ingest: $how на $DOWN_S с (replayer x$(replayer_stat speed), $(connected) ТС на связи)"
  epoch=$(ingest_stat clock_epoch)
  p_resets=$(predictor_stat clock_resets)
  rows=$(clock_reset_rows)
  units=$(replayer_stat units)
  t0=$SECONDS
  if is "$how" kill; then dc kill ingest; else dc stop ingest; fi
  wait_for 20 "api видит отказ: /health ingest down" api_ingest_is down
  check "api: /health degraded (HTTP 200), все ТС offline" eval 'health_is "$API" degraded && nobody_connected'
  check "predictor жив: /health отвечает" eval '[[ -n $(get "$PREDICTOR/health" | jv status) ]]'
  if is "$how" kill; then
    echo "        контейнер ingest: $($COMPOSE ps -a --format '{{.State}}' ingest) (docker kill: Docker его не поднимет)"
  fi
  sleep $((DOWN_S - (SECONDS - t0) > 0 ? DOWN_S - (SECONDS - t0) : 0))
  t0=$SECONDS
  check "ingest поднят: $COMPOSE up -d ingest" service_up ingest
  say "ingest healthy через $((SECONDS - t0)) с после команды"
  wait_for 30 "replayer переподключился: соединений $units из $units" replayer_all_connected
  wait_for 30 "api: ingest up, ТС снова на связи" eval 'api_ingest_is up && somebody_connected'
  wait_for 60 "replayer дослал очередь (отставание < 2 с)" replayer_caught_up
  wait_for 60 "лаг consumer group 0" lag_zero
  check "часы потока: та же эпоха ($epoch), ложного «перезапуска источника» нет" \
    eval 'is "$(ingest_stat clock_epoch)" "$epoch" && is "$(ingest_stat clock_resets)" 0'
  check "predictor: сбросов часов не прибавилось ($p_resets)" eval 'is "$(predictor_stat clock_resets)" "$p_resets"'
  check "журнал: новых clock_reset в service_events нет ($rows)" eval 'is "$(clock_reset_rows)" "$rows"'
}

scenario_redis() {
  local evicted t0
  say "redis: stop на $DOWN_S с"
  evicted=$(ingest_stat bus_evicted)
  t0=$SECONDS
  dc stop redis
  wait_for 20 "все сервисы видят отказ: /health degraded (HTTP 200)" \
    eval 'health_is "$API" degraded && health_is "$INGEST" degraded && health_is "$PREDICTOR" degraded'
  wait_for 10 "ingest принимает NDTP и копит события в памяти" ingest_buffering
  check "api отдаёт последнее известное состояние (degraded: true)" \
    eval 'get "$API/api/vehicles" | grep -q "\"degraded\":true"'
  sleep $((DOWN_S - (SECONDS - t0) > 0 ? DOWN_S - (SECONDS - t0) : 0))
  t0=$SECONDS
  dc start redis
  wait_for 60 "все сервисы снова ok" \
    eval 'health_is "$API" ok && health_is "$INGEST" ok && health_is "$PREDICTOR" ok'
  say "восстановление за $((SECONDS - t0)) с после start"
  wait_for 30 "буфер ingest дописан в поток" ingest_buffer_empty
  wait_for 60 "лаг consumer group 0" lag_zero
  check "потерь нет: bus_evicted не вырос ($evicted)" eval 'is "$(ingest_stat bus_evicted)" "$evicted"'
}

scenario_postgres() {
  local dropped t0
  say "postgres: stop на $DOWN_S с"
  dropped=$(predictor_stat db_dropped)
  t0=$SECONDS
  dc stop postgres
  wait_for 30 "сервисы видят отказ: /health degraded (HTTP 200)" \
    eval 'health_is "$API" degraded && health_is "$PREDICTOR" degraded'
  check "поток идёт: ТС на связи" somebody_connected
  sleep $((DOWN_S - (SECONDS - t0) > 0 ? DOWN_S - (SECONDS - t0) : 0))
  t0=$SECONDS
  dc start postgres
  wait_for 90 "все сервисы снова ok" \
    eval 'health_is "$API" ok && health_is "$INGEST" ok && health_is "$PREDICTOR" ok'
  say "восстановление за $((SECONDS - t0)) с после start"
  wait_for 30 "журнал predictor дописан" predictor_db_empty
  check "потерь журнала нет: db_dropped не вырос ($dropped)" eval 'is "$(predictor_stat db_dropped)" "$dropped"'
}

scenario_ml() {
  local ticks t0
  say "ml-service: stop на $DOWN_S с"
  t0=$SECONDS
  dc stop ml-service
  wait_for 30 "predictor прогнозирует по резервной формуле (source fallback)" eval 'is "$(forecast_source)" fallback'
  ticks=$(predictor_stat ticks)
  sleep 5
  check "тики прогнозов не прерываются" eval '(($(predictor_stat ticks) > ticks))'
  check "api отвечает (HTTP 200)" eval 'health_is "$API" ok || health_is "$API" degraded'
  sleep $((DOWN_S - (SECONDS - t0) > 0 ? DOWN_S - (SECONDS - t0) : 0))
  t0=$SECONDS
  dc start ml-service
  wait_for 90 "прогнозы снова от модели (source model)" eval 'is "$(forecast_source)" model'
  say "восстановление за $((SECONDS - t0)) с после start"
}

scenario_replayer() {
  local epoch t0
  say "replayer: docker compose restart replayer"
  epoch=$(ingest_stat clock_epoch)
  t0=$SECONDS
  dc restart replayer
  wait_for 60 "воспроизведение продолжилось само (автостарт make demo)" replayer_running
  wait_for 30 "replayer подключил все ТС" replayer_all_connected
  wait_for 30 "ТС на связи" somebody_connected
  say "демо снова идёт через $((SECONDS - t0)) с, с начала окна"
  # the replay starts over: the stream clock follows it back once most devices are on the new pass
  # (unless the stream was within 5 min of the window start anyway)
  sleep 5
  echo "        часы потока: $(ingest_stat stream_time), эпоха $epoch → $(ingest_stat clock_epoch)" \
    "(predictor: $(predictor_stat clock_epoch))"
}

run() {
  case $1 in
    ingest) scenario_ingest stop ;;
    ingest-kill) scenario_ingest kill ;;
    redis) scenario_redis ;;
    postgres) scenario_postgres ;;
    replayer) scenario_replayer ;;
    ml) scenario_ml ;;
    all) for s in ingest ingest-kill redis postgres ml replayer; do run "$s"; done ;;
    *)
      echo "неизвестный сценарий: $1 (ingest, ingest-kill, redis, postgres, ml, replayer, all)" >&2
      exit 2
      ;;
  esac
}

precheck
for scenario in "${@:-all}"; do
  run "$scenario"
done
echo
if ((failures)); then
  say "chaos: проверок не прошло — $failures"
  exit 1
fi
say "chaos: все проверки прошли"
