# syntax=docker/dockerfile:1
# Foresight · Диспетчер: сборка SPA в node:22 → статика и обратный прокси на nginx (порт 8080).
# Режим данных задаётся при запуске: FORESIGHT_API_MODE=live (по умолчанию, /api и /ws через прокси) или mock.

FROM node:22 AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN --mount=type=cache,target=/root/.npm npm ci --no-audit --no-fund
COPY . .
RUN npm run build

FROM nginx:1.31-alpine
ENV FORESIGHT_API_MODE=live \
    FORESIGHT_API_UPSTREAM=api:8000 \
    FORESIGHT_GRAFANA_UPSTREAM=grafana:3000 \
    FORESIGHT_GRAFANA_URL=/grafana \
    FORESIGHT_TILES_URL= \
    FORESIGHT_BASEMAP_STYLE= \
    FORESIGHT_ADMIN_USER=admin \
    FORESIGHT_ADMIN_PASSWORD= \
    FORESIGHT_MOCK_SPEED=6 \
    NGINX_ENTRYPOINT_LOCAL_RESOLVERS=1 \
    NGINX_ENVSUBST_FILTER=^(FORESIGHT_|NGINX_LOCAL_RESOLVERS)
RUN rm -f /etc/nginx/conf.d/default.conf
COPY nginx.conf /etc/nginx/templates/default.conf.template
COPY --chmod=755 docker/40-foresight-config.sh /docker-entrypoint.d/40-foresight-config.sh
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD wget -q -O /dev/null http://127.0.0.1:8080/healthz || exit 1
