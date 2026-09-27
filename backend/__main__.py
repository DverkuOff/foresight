"""Entry point: ``python -m backend <service>`` runs one of the services (``ingest``, ``predictor``, ``api``).

Equivalent to ``python -m backend.ingest`` / ``python -m backend.predictor`` / ``python -m backend.api``.
"""

from __future__ import annotations

import sys

SERVICES = ("ingest", "predictor", "api")


def main(argv: list[str] | None = None) -> int:
    """Run the service named in ``argv`` (default: ``sys.argv[1:]``).

    Returns:
        Exit code: 2 for an unknown or missing service name.
    """
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1 or args[0] not in SERVICES:
        print(f"usage: python -m backend {{{','.join(SERVICES)}}}", file=sys.stderr)
        return 2
    name = args[0]
    if name == "ingest":
        from backend.ingest import main as run
    elif name == "predictor":
        from backend.predictor import main as run
    else:
        from backend.api import main as run
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
