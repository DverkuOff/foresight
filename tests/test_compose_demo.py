"""Static checks of the demo lifecycle: autostart of the replayer, the chaos script, the Docker version check.

``make demo`` must leave a replayer that resumes the demo by itself after a restart, ``make emulator`` must
switch that off (one source at a time), the autostarted replay must use the same parameters as the
``make demo`` request, and ``make chaos`` must run a syntactically valid script.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

yaml = pytest.importorskip("yaml")

REPO = Path(__file__).resolve().parents[1]
MAKEFILE = REPO / "Makefile"
CHAOS = REPO / "scripts" / "chaos.sh"


@pytest.fixture(scope="module")
def compose() -> dict[str, Any]:
    return yaml.safe_load((REPO / "docker-compose.yml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def makefile() -> str:
    return MAKEFILE.read_text(encoding="utf-8")


def _recipe(makefile: str, target: str) -> str:
    lines = makefile.splitlines()
    start = next(i for i, line in enumerate(lines) if re.match(rf"{re.escape(target)}:", line))
    body = []
    for line in lines[start + 1 :]:
        if not line.strip():
            break
        body.append(line)
    return "\n".join(body)


def _make_default(makefile: str, name: str) -> str:
    match = re.search(rf"^{name}[ \t]*\?=[ \t]*(.*)$", makefile, re.M)
    assert match, name
    return match.group(1).strip()


def test_autostarted_replay_uses_the_demo_parameters(compose: dict[str, Any], makefile: str) -> None:
    env = compose["services"]["replayer"]["environment"]
    for var in ("UNITS", "MODE"):  # the rest is checked by test_compose.test_replay_defaults_agree
        match = re.fullmatch(r"\$\{(\w+):-(.*)\}", env[f"FORESIGHT_REPLAY_{var}"])
        assert match and match.group(1) == f"REPLAY_{var}", var
        assert match.group(2) == _make_default(makefile, f"REPLAY_{var}"), var
    exported = " ".join(re.findall(r"^export ([A-Z_ ]+)$", makefile, re.M)).split()
    assert {"REPLAY_UNITS", "REPLAY_MODE", "REPLAY_SPEED", "REPLAY_START"} <= set(exported)


def test_demo_turns_the_replayer_autostart_on_and_emulator_off(makefile: str) -> None:
    assert "up REPLAY_AUTOSTART=1" in _recipe(makefile, "demo")
    assert "replay-ensure" in _recipe(makefile, "demo")
    emulator = _recipe(makefile, "emulator")
    assert "REPLAY_AUTOSTART=0" in emulator and "up -d --no-deps" in emulator
    up = _recipe(makefile, "up")
    assert "FORESIGHT_REPLAY_AUTOSTART" in up  # make up keeps the autostart of the running replayer
    assert re.search(r"^up: check-docker ", makefile, re.M)


def test_status_warns_when_the_stream_is_down(makefile: str) -> None:
    status = _recipe(makefile, "status")
    assert "ВНИМАНИЕ" in status and "make replay-start" in status and "up -d" in status


def test_chaos_script(makefile: str) -> None:
    assert "scripts/chaos.sh $(SCENARIO)" in _recipe(makefile, "chaos")
    text = CHAOS.read_text(encoding="utf-8")
    for scenario in ("ingest", "ingest-kill", "redis", "postgres", "replayer", "all"):
        assert re.search(rf"^\s+{re.escape(scenario)}\)", text, re.M), scenario
    assert os.access(CHAOS, os.X_OK)
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is not available")
    subprocess.run([bash, "-n", str(CHAOS)], check=True)
