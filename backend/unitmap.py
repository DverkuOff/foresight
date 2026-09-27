"""``unit_id -> tr_id`` map: which schedule vehicle an NDTP device belongs to.

The map is read from the ``unit_id`` and ``tr_id`` columns of ``<dataset>/<split>/traffic.csv`` (the pairs
are one-to-one in the hackathon data). The standard ``csv`` module streams the files, so the ingest image
needs no pandas at runtime for this.
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Iterable
from pathlib import Path

log = logging.getLogger(__name__)


def read_pairs(path: Path) -> Iterable[tuple[int, int]]:
    """Yield distinct ``(unit_id, tr_id)`` pairs from one telemetry CSV.

    Args:
        path: CSV with ``unit_id`` and ``tr_id`` columns.

    Raises:
        ValueError: The header lacks one of the columns.
    """
    seen: set[tuple[str, str]] = set()
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader, [])
        try:
            iu, it = header.index("unit_id"), header.index("tr_id")
        except ValueError as exc:
            raise ValueError(f"{path}: no unit_id/tr_id columns") from exc
        width = max(iu, it)
        for row in reader:
            if len(row) <= width:
                continue
            pair = (row[iu].strip(), row[it].strip())
            if pair in seen or not pair[0] or not pair[1]:
                continue
            seen.add(pair)
            try:
                yield int(pair[0]), int(float(pair[1]))
            except ValueError:
                continue


def load_unit_map(dataset_dir: Path, splits: Iterable[str]) -> dict[int, int]:
    """Build the ``unit_id -> tr_id`` map from several splits (first split wins on conflicts).

    Missing files are skipped with a warning: the ingest still works, events just carry an empty ``tr_id``.

    Args:
        dataset_dir: Dataset root with ``<split>/traffic.csv``.
        splits: Splits to read, in priority order.

    Returns:
        The map.
    """
    mapping: dict[int, int] = {}
    conflicts = 0
    for split in splits:
        path = Path(dataset_dir) / split / "traffic.csv"
        if not path.is_file():
            log.warning("unit map: %s not found, skipped", path)
            continue
        try:
            for unit_id, tr_id in read_pairs(path):
                known = mapping.setdefault(unit_id, tr_id)
                if known != tr_id:
                    conflicts += 1
        except (OSError, ValueError, csv.Error) as exc:  # a broken file must not stop the ingest
            log.error("unit map: %s unreadable, skipped: %s", path, exc)
    if conflicts:
        log.warning("unit map: %d conflicting unit_id -> tr_id pairs ignored", conflicts)
    log.info("unit map: %d devices from %s", len(mapping), dataset_dir)
    return mapping
