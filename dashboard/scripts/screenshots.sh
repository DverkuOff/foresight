#!/usr/bin/env bash
# Скриншоты страниц дашборда для материалов сдачи. Headless Chrome (образ zenika/alpine-chrome) управляется по
# DevTools Protocol скриптом cdp-shot.mjs (Node 22 в Docker): ожидание в реальном времени — карта, WebSocket и
# графики успевают отрисоваться (у `chrome --screenshot` виртуальное время, приложение не успевает стартовать).
#
#   dashboard/scripts/screenshots.sh http://127.0.0.1:15180 /tmp/fs-dash-shots
#
# Дашборд должен быть запущен (например, образ foresight-dashboard с FORESIGHT_API_MODE=mock).
# Файлы: dashboard-<страница>-<ширина>x<высота>.png. Для карты нужен WebGL — программный SwiftShader.
set -euo pipefail

base="${1:?usage: screenshots.sh <base-url> <out-dir>}"
out="${2:?usage: screenshots.sh <base-url> <out-dir>}"
image="${CHROME_IMAGE:-zenika/alpine-chrome:124}"
port="${CDP_PORT:-19222}"
wait_ms="${WAIT_MS:-9000}"
sizes="${SIZES:-1920x1080,1366x768}"
here="$(cd "$(dirname "$0")" && pwd)"
chrome="fs-dash-chrome-$$"

mkdir -p "$out"
docker run -d --name "$chrome" --network host --shm-size=1g "$image" \
  --no-sandbox --hide-scrollbars --lang=ru-RU \
  --remote-debugging-address=127.0.0.1 --remote-debugging-port="$port" \
  --use-gl=angle --use-angle=swiftshader --enable-unsafe-swiftshader --ignore-gpu-blocklist \
  about:blank >/dev/null
trap 'docker rm -f "$chrome" >/dev/null 2>&1 || true' EXIT

for _ in $(seq 1 40); do
  curl -sf "http://127.0.0.1:$port/json/version" >/dev/null && break
  sleep 0.5
done

docker run --rm --network host -v "$here/cdp-shot.mjs:/cdp-shot.mjs:ro" -v "$out:/out" node:22 \
  node /cdp-shot.mjs --cdp "http://127.0.0.1:$port" --base "$base" --out /out --wait "$wait_ms" --sizes "$sizes" \
  overview=/ incident='/?incident=top' stringline=/stringline honesty=/honesty whatif=/whatif perf=/perf admin=/admin
