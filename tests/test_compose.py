"""Static checks of the demo stack: ``docker-compose.yml`` and the ``Makefile`` (no Docker needed).

``make up`` runs ``docker compose up --wait``, so every service needs a healthcheck. The replayer must wait
for a healthy ingest and read the dataset read-only. The Makefile builds its URLs and the ``make demo``
request from the same variables and defaults as the compose file, and the request must fit the replayer API.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

yaml = pytest.importorskip("yaml")

REPO = Path(__file__).resolve().parents[1]
COMPOSE = REPO / "docker-compose.yml"
MAKEFILE = REPO / "Makefile"

PORT_VARS = (
    "API_PORT",
    "INGEST_PORT",
    "PREDICTOR_PORT",
    "ML_PORT",
    "REPLAYER_PORT",
    "EMULATOR_PORT",
    "NDTP_PORT",
    "DASHBOARD_PORT",
    "GRAFANA_PORT",
    "PROMETHEUS_PORT",
)


@pytest.fixture(scope="module")
def compose() -> dict[str, Any]:
    return yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def makefile() -> str:
    return MAKEFILE.read_text(encoding="utf-8")


def _default(value: str) -> tuple[str, str]:
    """Variable name and default of a compose ``${NAME:-default}`` value."""
    match = re.fullmatch(r"\$\{(\w+):-(.*)\}", value)
    assert match, value
    return match.group(1), match.group(2)


def _make_default(makefile: str, name: str) -> str:
    match = re.search(rf"^{name}[ \t]*\?=[ \t]*(.*)$", makefile, re.M)
    assert match, f"{name} is not a ?= variable of the Makefile"
    return match.group(1).strip()


def _recipe(makefile: str, target: str) -> list[str]:
    """Lines of a rule body: from the ``target:`` line to the next blank line."""
    lines = makefile.splitlines()
    start = next(i for i, line in enumerate(lines) if re.match(rf"{re.escape(target)}:", line))
    body = []
    for line in lines[start + 1 :]:
        if not line.strip():
            break
        body.append(line)
    return body


def test_every_service_is_healthchecked_and_restarts(compose: dict[str, Any]) -> None:
    for name, svc in compose["services"].items():
        assert svc.get("healthcheck", {}).get("test"), f"{name}: `up --wait` cannot tell when it is ready"
        assert svc.get("restart") == "unless-stopped", name


def test_unchanged_rebuild_keeps_containers(compose: dict[str, Any], makefile: str) -> None:
    # with the containerd image store a provenance attestation changes the image id on every build, and
    # `make demo` on a running stack would re-create the containers although nothing changed
    builds = {name for name, svc in compose["services"].items() if "build" in svc}
    assert builds == {"ingest", "replayer", "ml-service", "dashboard"}
    assert re.search(r"^export BUILDX_NO_DEFAULT_ATTESTATIONS :?= 1$", makefile, re.M)


def test_replayer_service(compose: dict[str, Any]) -> None:
    svc = compose["services"]["replayer"]
    assert svc["build"]["dockerfile"] == "replayer/Dockerfile"
    assert svc["image"].startswith("foresight-replayer:")
    assert svc["depends_on"]["ingest"]["condition"] == "service_healthy"
    env = svc["environment"]
    assert (env["FORESIGHT_REPLAY_TARGET_HOST"], env["FORESIGHT_REPLAY_TARGET_PORT"]) == ("ingest", "9201")
    assert env["FORESIGHT_REPLAY_EMULATOR_URL"] == "http://emulator:18080"
    assert _default(env["FORESIGHT_REPLAY_AUTOSTART"]) == ("REPLAY_AUTOSTART", "0")  # make demo starts it
    assert "http://127.0.0.1:8010/health" in " ".join(svc["healthcheck"]["test"])
    assert any(port.endswith(":8010") for port in svc["ports"])


def test_dataset_is_mounted_read_only(compose: dict[str, Any], makefile: str) -> None:
    for name in ("ingest", "predictor", "replayer"):
        mounts = [v for v in compose["services"][name]["volumes"] if ":/app/dataset" in v]
        assert len(mounts) == 1, name
        source, target, mode = mounts[0].rsplit(":", 2)
        assert (target, mode) == ("/app/dataset", "ro"), name
        assert _default(source) == ("DATASET_DIR", _make_default(makefile, "DATASET_DIR")), name


def test_replay_defaults_agree(compose: dict[str, Any], makefile: str) -> None:
    # REPLAY_AUTOSTART=1 (compose) and make demo replay the same thing by default
    env = compose["services"]["replayer"]["environment"]
    for var in ("SPLIT", "SPEED", "START", "UNTIL", "LOOP"):
        name, default = _default(env[f"FORESIGHT_REPLAY_{var}"])
        assert name == f"REPLAY_{var}"
        assert default == _make_default(makefile, name), var


def test_published_ports_agree(compose: dict[str, Any], makefile: str) -> None:
    published: dict[str, str] = {}
    for svc in compose["services"].values():
        for port in svc.get("ports", []):
            host, container = port.rsplit(":", 1)
            name, default = _default(host)
            assert default == container, port
            published[name] = default
    assert set(published) == set(PORT_VARS)
    for var in PORT_VARS:
        assert _make_default(makefile, var) == published[var], var


def test_make_targets(makefile: str) -> None:
    targets = set(re.findall(r"^([a-z-]+):", makefile, re.M))
    required = {"up", "demo", "replay-start", "replay-stop", "emulator", "emulator-stop", "status", "logs"}
    assert required | {"down", "clean", "help"} <= targets
    for target in targets:
        for line in _recipe(makefile, target):
            assert line.startswith("\t"), f"{target}: recipe lines must start with a tab: {line!r}"
    assert "up -d --wait" in "\n".join(_recipe(makefile, "up"))
    down, clean = "\n".join(_recipe(makefile, "down")), "\n".join(_recipe(makefile, "clean"))
    assert "down" in down and " -v" not in down  # make down keeps the data
    assert "down -v" in clean


def test_demo_request_fits_the_replayer_api(makefile: str) -> None:
    body = re.search(r"^REPLAY_BODY = (.*)$", makefile, re.M)
    assert body
    text = re.sub(r"\$\(call json_bool,\$\(REPLAY_LOOP\)\)", "true", body.group(1))
    text = re.sub(r"\$\((REPLAY_\w+)\)", lambda m: _make_default(makefile, m.group(1)), text)
    request = json.loads(text)
    assert request == {
        "split": "test",
        "speed": 30,
        "start": "06:00",
        "until": "",
        "units": [],
        "loop": True,
        "mode": "ndtp",
    }
    pytest.importorskip("fastapi")
    from replayer.api import StartIn

    assert StartIn.model_validate(request).model_dump(exclude_unset=True) == request
