"""Smoke test: packages import."""

import backend
import ml
import replayer
import shared


def test_packages_import() -> None:
    assert all(m.__doc__ for m in (shared, ml, backend, replayer))
