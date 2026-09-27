"""Export the OpenAPI specs of all Foresight services and a static Swagger UI page (for GitHub Pages).

    python scripts/export_openapi.py site/api

Writes ``<out>/<service>.json`` for api, ingest, predictor, ml-service and replayer and ``<out>/index.html`` —
Swagger UI (from jsDelivr) with a service switcher; «Try it out» is off (there is no backend behind Pages).
No service is started: the apps are only built to render their schemas (no Redis, PostgreSQL or dataset
needed).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SERVICES = [
    ("api", "api — дашборд: ТС, инциденты, алерты, нитка, метрики, What-if, админка, WebSocket"),
    ("ingest", "ingest — приём NDTP, статистика потока"),
    ("predictor", "predictor — признаки, тики прогнозов, состояние"),
    ("ml-service", "ml-service — модель: прогноз, P10–P90, p_late, факторы"),
    ("replayer", "replayer — демо-поток из исторического CSV"),
]

PAGE = """<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Foresight · API (OpenAPI / Swagger)</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css" />
  <style>
    body { margin: 0; background: #fafafa; }
    .fs-head {
      padding: 14px 24px; background: #0b1220; color: #e2e8f0; font: 15px/1.4 system-ui, sans-serif;
    }
    .fs-head a { color: #93c5fd; }
  </style>
</head>
<body>
  <div class="fs-head">
    <b>Foresight</b> · спецификации OpenAPI всех сервисов (выбор сервиса — справа вверху).
    На развёрнутом стенде тот же Swagger с кнопкой «Try it out»: <code>/docs</code> у каждого сервиса.
    <a href="../">Документация</a>
  </div>
  <div id="swagger"></div>
  <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-standalone-preset.js"></script>
  <script>
    window.ui = SwaggerUIBundle({
      urls: __URLS__,
      'urls.primaryName': __PRIMARY__,
      dom_id: '#swagger',
      deepLinking: true,
      supportedSubmitMethods: [],
      presets: [SwaggerUIBundle.presets.apis, SwaggerUIStandalonePreset],
      layout: 'StandaloneLayout',
    })
  </script>
</body>
</html>
"""


def build_apps() -> dict[str, object]:
    from backend import api, ingest, predictor
    from backend.config import Settings
    from ml import service
    from replayer import api as replayer_api

    settings = Settings(  # type: ignore[call-arg]
        database_url="", unit_map_splits="", forecast_enabled=False, api_schedule=False
    )
    return {
        "api": api.create_app(settings),
        "ingest": ingest.create_app(settings),
        "predictor": predictor.create_app(settings),
        "ml-service": service.create_app(load=False),
        "replayer": replayer_api.create_app(),
    }


def main(out: Path) -> None:
    logging.basicConfig(level=logging.WARNING)
    out.mkdir(parents=True, exist_ok=True)
    apps = build_apps()
    urls = []
    for name, label in SERVICES:
        spec = apps[name].openapi()  # type: ignore[attr-defined]
        (out / f"{name}.json").write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
        urls.append({"url": f"{name}.json", "name": label})
        print(f"{name}: {len(spec.get('paths', {}))} paths")
    page = PAGE.replace("__URLS__", json.dumps(urls, ensure_ascii=False)).replace(
        "__PRIMARY__", json.dumps(urls[0]["name"], ensure_ascii=False)
    )
    (out / "index.html").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "site/api"))
