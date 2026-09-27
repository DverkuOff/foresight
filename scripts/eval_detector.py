"""Качество детектора прохождения остановок: pass_time против time_fact_begin.

Запуск (на сервере): uv run python scripts/eval_detector.py [split ...]
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from shared.data import load_schedule, load_traffic
from shared.stops import detect_all


def _report(name: str, df: pd.DataFrame) -> None:
    if df.empty:
        print(f"  {name:22s} нет остановок")
        return
    matched = df["pass_time"].notna()
    err = (df.loc[matched, "pass_time"] - df.loc[matched, "time_fact_begin"]).dt.total_seconds()
    ae = err.abs()
    print(
        f"  {name:22s} stops={len(df):6d} matched={matched.mean():.3f} "
        f"|err| med={ae.median():5.1f}s p90={ae.quantile(0.9):6.1f}s mean={ae.mean():6.1f}s "
        f"≤15s={(ae <= 15).mean():.3f} bias(med)={err.median():+.1f}s"
    )


def evaluate(split: str, real_ids: set[int]) -> None:
    t0 = time.perf_counter()
    traffic = load_traffic(split)
    schedule = load_schedule(split)
    t1 = time.perf_counter()
    res = detect_all(traffic, schedule)
    t2 = time.perf_counter()
    df = res.merge(
        schedule[["tt_action_item_id", "tr_id", "time_fact_begin", "manual_fill"]],
        on=["tr_id", "tt_action_item_id"],
        how="left",
    )
    print(
        f"[{split}] точек={len(traffic)} остановок={len(schedule)} ТС={schedule['tr_id'].nunique()} "
        f"загрузка={t1 - t0:.1f}s detect_all={t2 - t1:.1f}s"
    )
    real = df["tr_id"].isin(real_ids)
    manual = df["manual_fill"].astype(bool)
    for gname, mask in {"real": real, "synthetic": ~real}.items():
        if not mask.any():
            continue
        _report(gname, df[mask])
        _report(f"{gname}, auto fact", df[mask & ~manual])
        _report(f"{gname}, manual fact", df[mask & manual])
    conf = df.dropna(subset=["pass_time"])
    lag = (conf["confirmed_at"] - conf["pass_time"]).dt.total_seconds()
    print(f"  задержка подтверждения: med={lag.median():.0f}s p90={lag.quantile(0.9):.0f}s")


def main(argv: list[str]) -> None:
    splits = argv or ["test", "train"]
    real_ids = {int(x) for x in np.unique(load_traffic("test")["tr_id"])}
    for split in splits:
        evaluate(split, real_ids)


if __name__ == "__main__":
    main(sys.argv[1:])
