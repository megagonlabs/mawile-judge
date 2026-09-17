"""Small helpers for looking up persisted judge-call traces in reports."""

from __future__ import annotations

from typing import Any


def trace_by_result(
    traces: list[dict[str, Any]],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    mapped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for trace in traces:
        key = (
            str(trace.get("item_id")),
            str(trace.get("variant_id")),
            str(trace.get("run_id")),
        )
        mapped[key] = trace
        mapped.setdefault((key[0], key[1], "*"), trace)
    return mapped


def first_field_change(
    trace: dict[str, Any] | None,
) -> tuple[str | None, Any, Any]:
    changes = (trace or {}).get("field_changes") or []
    if not changes:
        return None, None, None
    first = changes[0]
    return first.get("field"), first.get("before"), first.get("after")


__all__ = ["first_field_change", "trace_by_result"]
