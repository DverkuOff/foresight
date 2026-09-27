"""Read side of the journal for the api: incidents, alerts, forecasts and stop passages in PostgreSQL.

The predictor writes the rows (``backend.forecast``); the api reads them for the dashboard. Rows of an older
stream timeline (the replayer restarted, the clock jumped) are left out: forecasts carry their ``epoch``,
the other tables are filtered by ``created_at`` — the epoch is the wall-clock millisecond of the jump
(:class:`backend.clock.StreamClock`), so the rows of the current timeline were created after it.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from backend.db import Database

RISK_ORDER = {"red": 0, "yellow": 1, "green": 2, "unknown": 3}
"""Sort order of incidents: red first."""

_EPOCH_MIN = 10**12
"""Smaller epochs are not wall-clock milliseconds (tests, a clock without jumps): no time filter."""
_EPOCH_SLACK_S = 5.0


def iso(value: datetime | None) -> str | None:
    """Aware (or naive UTC) datetime → ISO 8601 UTC with ``Z``."""
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def num(x: Any, digits: int = 1) -> float | None:
    """Rounded finite number or ``None``."""
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return round(f, digits) if math.isfinite(f) else None


def epoch_start(epoch: int | None) -> datetime | None:
    """Wall-clock start of a stream epoch (a few seconds early), ``None`` if the epoch is not a timestamp."""
    if not epoch or epoch < _EPOCH_MIN:
        return None
    return datetime.fromtimestamp(epoch / 1000 - _EPOCH_SLACK_S, UTC)


def _json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str | bytes):
        try:
            parsed = json.loads(value)
        except ValueError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _cause(code: str | None, detail: Mapping[str, Any] | None, recommendation: str | None = None) -> dict:
    """``cause`` object of the contract (§1) from the stored detail, or a minimal one from the code."""
    if detail and detail.get("code"):
        return dict(detail)
    return {"code": code or "unknown", "text": "", "recommendation": recommendation or "", "factors": []}


def alert_out(row: Mapping[str, Any]) -> dict[str, Any]:
    """An ``alerts`` row → ``AlertOut``."""
    d = _json(row.get("details"))
    return {
        "alert_id": row["id"],
        "prediction_id": row.get("prediction_id"),
        "incident_id": row.get("incident_id") or d.get("incident_id"),
        "kind": row.get("kind") or "delay",
        "tr_id": row.get("tr_id"),
        "unit_id": row.get("unit_id"),
        "route_id": row.get("route_id"),
        "level": row.get("level"),
        "status": row.get("status"),
        "cause": _cause(row.get("cause"), d.get("cause"), row.get("recommendation")),
        "issued_at": iso(row.get("issued_at")),
        "planned_at": iso(row.get("target_time_begin")),
        "target_stop_id": row.get("stop_id"),
        "target_stop_name": d.get("stop_name"),
        "pred_delay_s": num(row.get("pred_delay_s")),
        "p10": num(row.get("p10")),
        "p90": num(row.get("p90")),
        "p_late": num(row.get("p_late"), 3),
        "escalated_from": d.get("escalated_from"),
        "related_tr_id": d.get("related_tr_id"),
        "acknowledged": bool(row.get("acknowledged")),
        "retroactive": row.get("retroactive"),
        "actual_delay_s": num(row.get("actual_delay_s")),
        "closed_at": iso(row.get("closed_at")),
    }


def prediction_out(row: Mapping[str, Any]) -> dict[str, Any]:
    """A ``predictions`` row → ``PredictionOut`` (plus the check against the fact)."""
    status = row.get("status") or "open"
    return {
        "prediction_id": row["id"],
        "tr_id": row.get("tr_id"),
        "unit_id": row.get("unit_id"),
        "route_id": row.get("route_id"),
        "target_stop_id": row.get("target_stop_id"),
        "target_stop_name": row.get("target_stop_name"),
        "planned_at": iso(row.get("target_time_begin")),
        "issued_at": iso(row.get("issued_at")),
        "updated_at": iso(row.get("updated_at")),
        "updates": row.get("updates") or 0,
        "lead_s": num(row.get("lead_s")),
        "pred_delay_s": num(row.get("pred_delay_s")),
        "first_pred_delay_s": num(row.get("first_pred_delay_s")),
        "p10": num(row.get("p10")),
        "p50": num(row.get("p50")),
        "p90": num(row.get("p90")),
        "p_late": num(row.get("p_late"), 3),
        "risk": row.get("risk") or "unknown",
        "cause": row.get("cause"),
        "model_version": row.get("model_version"),
        "source": row.get("source") or "model",
        "status": "open" if status == "open" else "closed",
        "outcome": status,
        "cur_dev_s": num(row.get("cur_dev_s")),
        "actual_delay_s": num(row.get("actual_delay_s")),
        "abs_error_s": num(row.get("abs_error_s")),
        "pass_time": iso(row.get("pass_time")),
        "actual_lead_s": num(row.get("actual_lead_s")),
        "retroactive": row.get("retroactive"),
        "alert_level": row.get("alert_level"),
        "closed_at": iso(row.get("closed_at")),
    }


def incident_out(row: Mapping[str, Any]) -> dict[str, Any]:
    """An ``incidents`` row → ``IncidentOut`` (the predictor stores it whole in ``details``)."""
    out = _json(row.get("details"))
    out.setdefault("incident_id", row["id"])
    out.setdefault("kind", row.get("kind"))
    out.setdefault("tr_id", row.get("tr_id"))
    out.setdefault("unit_id", row.get("unit_id"))
    out.setdefault("route_id", row.get("route_id"))
    out.setdefault("risk", row.get("risk") or "unknown")
    out.setdefault("pred_delay_s", num(row.get("pred_delay_s")))
    out.setdefault("p_late", num(row.get("p_late"), 3))
    out.setdefault("related_tr_id", row.get("related_tr_id"))
    if not isinstance(out.get("cause"), dict):
        out["cause"] = _cause(row.get("cause"), None)
    out["status"] = row.get("status") or out.get("status") or "open"
    out["opened_at"] = iso(row.get("opened_at")) or out.get("opened_at")
    out["updated_at"] = iso(row.get("updated_at"))
    out["closed_at"] = iso(row.get("closed_at"))
    return out


def incident_sort_key(incident: Mapping[str, Any]) -> tuple[int, float, float]:
    """Red → yellow → the rest, then by ``p_late`` and the forecast delay, both descending."""
    return (
        RISK_ORDER.get(str(incident.get("risk")), 3),
        -(num(incident.get("p_late"), 3) or 0.0),
        -(num(incident.get("pred_delay_s")) or 0.0),
    )


def lead_histogram(values: Iterable[float], bucket_s: int = 60, max_s: int = 1800) -> list[dict[str, int]]:
    """Buckets ``(from_s, to_s]`` of lead times (the horizon window is ``(600, 900]``); zero and negatives
    (at or after the fact) in ``(-bucket, 0]``."""
    counts: dict[int, int] = {}
    for v in values:
        f = num(v, 3)
        if f is None:
            continue
        b = -1 if f <= 0 else min(math.ceil(f / bucket_s) - 1, max_s // bucket_s - 1)
        counts[b] = counts.get(b, 0) + 1
    if not counts:
        return []
    lo, hi = min(counts), max(counts)
    return [
        {"from_s": b * bucket_s, "to_s": (b + 1) * bucket_s, "count": counts.get(b, 0)}
        for b in range(lo, hi + 1)
    ]


class Journal:
    """Queries of the journal (each fails fast when PostgreSQL is not connected).

    Args:
        db: The api's database (connected by its writer loop); ``None`` — no journal.
        timeout_s: Statement timeout.
    """

    def __init__(self, db: Database | None, timeout_s: float = 3.0) -> None:
        self.db = db
        self.timeout_s = timeout_s

    @property
    def available(self) -> bool:
        """A pool exists (the writer loop connected it)."""
        return self.db is not None and self.db.connected

    async def fetch(self, sql: str, *args: Any) -> list[Mapping[str, Any]]:
        """Run a query.

        Raises:
            Exception: Not connected, unreachable or the query failed.
        """
        if self.db is None:
            raise RuntimeError("no database configured")
        return await self.db.pool.fetch(sql, *args, timeout=self.timeout_s)

    # ---- incidents ------------------------------------------------------------------------------

    async def incidents_by_id(self, ids: Sequence[int]) -> dict[int, dict[str, Any]]:
        """Incidents with these ids."""
        if not ids:
            return {}
        rows = await self.fetch("SELECT * FROM incidents WHERE id = ANY($1::bigint[])", list(ids))
        return {int(r["id"]): incident_out(r) for r in rows}

    async def incidents(
        self, *, since: datetime | None, status: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Incidents of the timeline, newest first."""
        rows = await self.fetch(
            "SELECT * FROM incidents WHERE ($1::timestamptz IS NULL OR created_at >= $1) "
            "AND ($2::text IS NULL OR status = $2) ORDER BY opened_at DESC LIMIT $3",
            since,
            status,
            limit,
        )
        return [incident_out(r) for r in rows]

    # ---- alerts ---------------------------------------------------------------------------------

    async def alerts(
        self,
        *,
        since: datetime | None,
        issued_after: datetime | None = None,
        issued_before: datetime | None = None,
        level: str | None = None,
        tr_id: int | None = None,
        incident_id: int | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Alerts of the timeline, newest first."""
        rows = await self.fetch(
            "SELECT * FROM alerts WHERE ($1::timestamptz IS NULL OR created_at >= $1) "
            "AND ($2::timestamptz IS NULL OR issued_at > $2) AND ($3::text IS NULL OR level = $3) "
            "AND ($4::bigint IS NULL OR tr_id = $4) AND ($5::bigint IS NULL OR incident_id = $5) "
            "AND ($7::timestamptz IS NULL OR issued_at <= $7) ORDER BY issued_at DESC, id DESC LIMIT $6",
            since,
            issued_after,
            level,
            tr_id,
            incident_id,
            limit,
            issued_before,
        )
        return [alert_out(r) for r in rows]

    # ---- forecasts ------------------------------------------------------------------------------

    async def predictions(
        self,
        *,
        epoch: int | None,
        tr_id: int | None = None,
        tr_ids: Sequence[int] | None = None,
        status: str | None = None,
        planned_from: datetime | None = None,
        planned_to: datetime | None = None,
        limit: int = 100,
        newest_first: bool = False,
    ) -> list[dict[str, Any]]:
        """Forecasts of the timeline: open ones by plan time, closed ones newest first.

        Args:
            epoch: Timeline (``None``: all).
            tr_id: One vehicle.
            tr_ids: Several vehicles.
            status: ``open`` or ``closed`` (closed with the fact; skipped / expired / reset stops are not).
            planned_from: Plan time of the target stop from.
            planned_to: ... to.
            limit: Rows.
            newest_first: By plan time, the latest first (the journal of the admin: the last ``limit`` rows).
        """
        order = "closed_at DESC, id DESC" if status == "closed" else "target_time_begin, id"
        if newest_first:
            order = "target_time_begin DESC, id DESC"
        rows = await self.fetch(
            "SELECT * FROM predictions WHERE ($1::bigint IS NULL OR epoch = $1) "
            "AND ($2::bigint IS NULL OR tr_id = $2) AND ($3::bigint[] IS NULL OR tr_id = ANY($3)) "
            "AND ($4::text IS NULL OR status = $4) "
            "AND ($5::timestamptz IS NULL OR target_time_begin >= $5) "
            f"AND ($6::timestamptz IS NULL OR target_time_begin <= $6) ORDER BY {order} LIMIT $7",
            epoch,
            tr_id,
            list(tr_ids) if tr_ids is not None else None,
            status,
            planned_from,
            planned_to,
            limit,
        )
        return [prediction_out(r) for r in rows]

    # ---- stop passages (the fact of the detector) -----------------------------------------------

    async def passages(
        self, tr_ids: Sequence[int], lo: datetime, hi: datetime, *, since: datetime | None
    ) -> list[Mapping[str, Any]]:
        """Passes of the stops by these vehicles with the pass time in ``[lo, hi]``, in time order."""
        if not tr_ids:
            return []
        return await self.fetch(
            "SELECT DISTINCT ON (tr_id, stop_id) tr_id, stop_id, time_begin, pass_time, delay_s "
            "FROM stop_passages WHERE tr_id = ANY($1::bigint[]) AND matched AND pass_time BETWEEN $2 AND $3 "
            "AND ($4::timestamptz IS NULL OR created_at >= $4) ORDER BY tr_id, stop_id, confirmed_at DESC",
            list(tr_ids),
            lo,
            hi,
            since,
        )

    # ---- «honesty» of the forecasts -------------------------------------------------------------

    async def horizon(
        self,
        *,
        epoch: int | None,
        since: datetime | None,
        closed_after: datetime | None,
        late_s: float,
    ) -> dict[str, Any]:
        """Closed forecasts of the timeline checked against the fact (``GET /api/metrics/horizon``).

        Args:
            epoch: Timeline of the forecasts.
            since: Wall-clock start of the timeline (alerts).
            closed_after: Stream time from which closed forecasts count for the window figures.
            late_s: A stop is late when its fact exceeds this (the red threshold).
        """
        window = await self.fetch(
            "SELECT count(*) AS closed, avg(abs_error_s) AS mae, "
            "avg(abs(coalesce(cur_dev_s, 0) - actual_delay_s)) AS base, "
            "count(*) FILTER (WHERE actual_delay_s > $3) AS late, "
            "count(*) FILTER (WHERE actual_delay_s > $3 AND alert_level IN ('yellow', 'red')) "
            "AS late_warned, "
            "count(*) FILTER (WHERE actual_delay_s > $3 AND alert_level = 'red') AS late_red, "
            "array_agg(lead_s) AS leads, array_agg(actual_lead_s) AS actual_leads "
            "FROM predictions WHERE ($1::bigint IS NULL OR epoch = $1) AND status = 'closed' "
            "AND ($2::timestamptz IS NULL OR closed_at >= $2)",
            epoch,
            closed_after,
            late_s,
        )
        hours = await self.fetch(
            "SELECT extract(hour FROM target_time_begin AT TIME ZONE 'UTC')::int AS hour, count(*) AS n, "
            "avg(abs_error_s) AS mae, avg(abs(coalesce(cur_dev_s, 0) - actual_delay_s)) AS base "
            "FROM predictions WHERE ($1::bigint IS NULL OR epoch = $1) AND status = 'closed' "
            "GROUP BY 1 ORDER BY 1",
            epoch,
        )
        total = await self.fetch(
            "SELECT count(*) FILTER (WHERE status = 'closed') AS closed, "
            "avg(abs_error_s) FILTER (WHERE status = 'closed') AS mae, "
            "avg(abs(coalesce(cur_dev_s, 0) - actual_delay_s)) FILTER (WHERE status = 'closed') AS base, "
            "count(*) AS issued, count(*) FILTER (WHERE retroactive) AS retroactive, "
            "count(*) FILTER (WHERE source = 'fallback') AS fallback, "
            "min(lead_s) AS lead_min, max(lead_s) AS lead_max "
            "FROM predictions WHERE ($1::bigint IS NULL OR epoch = $1)",
            epoch,
        )
        alerts = await self.fetch(
            "SELECT count(*) AS alerts, count(*) FILTER (WHERE retroactive) AS retroactive, "
            "count(DISTINCT incident_id) AS incidents FROM alerts "
            "WHERE ($1::timestamptz IS NULL OR created_at >= $1)",
            since,
        )
        w, t, a = window[0], total[0], alerts[0]
        late = int(w["late"] or 0)
        return {
            "closed": int(w["closed"] or 0),
            "online_mae_s": num(w["mae"]),
            "baseline_mae_s": num(w["base"]),
            "warned_share": round(int(w["late_warned"]) / late, 3) if late else None,
            "warned_red_share": round(int(w["late_red"]) / late, 3) if late else None,
            "late_stops": late,
            "retroactive": int(t["retroactive"] or 0) + int(a["retroactive"] or 0),
            "retroactive_predictions": int(t["retroactive"] or 0),
            "retroactive_alerts": int(a["retroactive"] or 0),
            "lead_hist": lead_histogram(w["leads"] or []),
            "actual_lead_hist": lead_histogram(w["actual_leads"] or []),
            "mae_by_hour": [
                {
                    "hour": int(r["hour"]),
                    "mae_s": num(r["mae"]),
                    "baseline_s": num(r["base"]),
                    "n": int(r["n"]),
                }
                for r in hours
            ],
            "total": {
                "issued": int(t["issued"] or 0),
                "closed": int(t["closed"] or 0),
                "mae_s": num(t["mae"]),
                "baseline_mae_s": num(t["base"]),
                "fallback": int(t["fallback"] or 0),
                "lead_min_s": num(t["lead_min"]),
                "lead_max_s": num(t["lead_max"]),
                "alerts": int(a["alerts"] or 0),
                "incidents_alerted": int(a["incidents"] or 0),
            },
        }


def window_start(stream_time: datetime | None, seconds: float) -> datetime | None:
    """``stream_time − seconds`` (``None`` without a clock)."""
    return None if stream_time is None else stream_time - timedelta(seconds=seconds)
