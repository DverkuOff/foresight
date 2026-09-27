#!/bin/sh
# Runtime-конфигурация дашборда из переменных окружения (без пересборки образа): пишет /config.js,
# который index.html загружает до приложения. Переменные: FORESIGHT_API_MODE (live | mock),
# FORESIGHT_TILES_URL (шаблоны растровых тайлов через «|», none — без подложки), FORESIGHT_BASEMAP_STYLE (URL
# векторного стиля, none — без него), FORESIGHT_GRAFANA_URL, FORESIGHT_MOCK_SPEED.
set -eu

# изменения в администрировании (POST/PUT /api/admin/*, /api/replay/*) — под паролем, если он задан
mkdir -p /etc/nginx/foresight
if [ -n "${FORESIGHT_ADMIN_PASSWORD:-}" ]; then
    printf '%s:{PLAIN}%s\n' "${FORESIGHT_ADMIN_USER:-admin}" "$FORESIGHT_ADMIN_PASSWORD" > /etc/nginx/foresight/htpasswd
    chmod 600 /etc/nginx/foresight/htpasswd
    chown nginx /etc/nginx/foresight/htpasswd 2>/dev/null || true
    cat > /etc/nginx/foresight/admin-auth.conf <<'AUTH'
limit_except GET HEAD {
    auth_basic "Foresight: administration";
    auth_basic_user_file /etc/nginx/foresight/htpasswd;
}
AUTH
    admin="под паролем (${FORESIGHT_ADMIN_USER:-admin})"
else
    : > /etc/nginx/foresight/admin-auth.conf
    admin="открыто"
fi

escape() {
    printf '%s' "$1" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' -e 's/</\\u003c/g'
}

target=/usr/share/nginx/html/config.js
cat > "$target" <<CONFIG
// Сгенерировано при старте контейнера (docker/40-foresight-config.sh)
window.__FORESIGHT_CONFIG__ = {
  apiMode: "$(escape "${FORESIGHT_API_MODE:-live}")",
  tilesUrl: "$(escape "${FORESIGHT_TILES_URL:-}")",
  basemapStyle: "$(escape "${FORESIGHT_BASEMAP_STYLE:-}")",
  grafanaUrl: "$(escape "${FORESIGHT_GRAFANA_URL:-/grafana}")",
  mockSpeed: "$(escape "${FORESIGHT_MOCK_SPEED:-6}")"
}
CONFIG
echo "40-foresight-config.sh: apiMode=${FORESIGHT_API_MODE:-live}, api=${FORESIGHT_API_UPSTREAM:-}, grafana=${FORESIGHT_GRAFANA_UPSTREAM:-}, администрирование: $admin"
