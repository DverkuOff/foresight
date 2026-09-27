#!/usr/bin/env bash
# Скачать датасет организаторов (Яндекс.Диск из ТЗ кейса) в ./dataset с проверкой sha256.
#
#   scripts/get_dataset.sh [каталог]      (make dataset; по умолчанию ./dataset)
#
# Нужны curl и одно из: unzip, python3 или Docker (распаковка). Уже скачанный датасет не перекачивается.
set -euo pipefail

DIR="${1:-dataset}"
PUBLIC_URL="https://disk.yandex.ru/d/CA6tsj4aJJ4Aaw"
ZIP_SHA256="fce28b4121dbaae793eee855b8efa523484c91ac203d782ff33853884c527760"
API="https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key=${PUBLIC_URL}&path=/dataset.zip"

if [ -f "$DIR/test/traffic.csv" ] && [ -f "$DIR/train/traffic.csv" ] && [ -f "$DIR/ndtp-telemetry-emulator.tar" ]; then
    echo "датасет уже на месте: $DIR"
    exit 0
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

echo "датасет: $PUBLIC_URL (dataset.zip, 147 МБ)"
href="$(curl -fsS "$API" | sed -n 's/.*"href":"\([^"]*\)".*/\1/p')"
[ -n "$href" ] || { echo "Яндекс.Диск не отдал ссылку на скачивание: $API" >&2; exit 1; }
curl -fL --progress-bar -o "$tmp/dataset.zip" "$href"

if command -v sha256sum >/dev/null 2>&1; then
    sum="$(sha256sum "$tmp/dataset.zip" | cut -d' ' -f1)"
else
    sum="$(shasum -a 256 "$tmp/dataset.zip" | cut -d' ' -f1)"
fi
if [ "$sum" != "$ZIP_SHA256" ]; then
    echo "sha256 не совпадает: $sum (ожидался $ZIP_SHA256) — архив изменён или скачан не полностью" >&2
    exit 1
fi
echo "sha256 совпадает"

mkdir -p "$DIR"
if command -v unzip >/dev/null 2>&1; then
    unzip -q -o "$tmp/dataset.zip" -d "$DIR"
elif command -v python3 >/dev/null 2>&1; then
    python3 -m zipfile -e "$tmp/dataset.zip" "$DIR"
else
    docker run --rm -v "$tmp:/in:ro" -v "$(cd "$DIR" && pwd):/out" python:3.12-slim \
        python -m zipfile -e /in/dataset.zip /out
fi
echo "готово: $DIR ($(find "$DIR" -type f | wc -l | tr -d ' ') файлов)"
