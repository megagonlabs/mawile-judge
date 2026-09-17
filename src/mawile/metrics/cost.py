from __future__ import annotations

from typing import Any

from mawile.pricing import call_cost_usd, call_price_complete
from mawile.schemas import ItemRisk


_CATEGORY_CALL_ALIASES = {
    "judge": "judge_calls",
    "planning": "perturbation_planning_calls",
    "perturbation_generation": "perturbation_generation_calls",
    "equivalence_validation": "equivalence_validation_calls",
    "reporting": "reporting_calls",
}


def compute_cost(
    item_risks: list[ItemRisk],
    call_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Price every recorded LLM call and break the bill out by role.

    ``call_records`` are captured at the API-call boundary and contain a
    category, provider, model, status, and normalized usage mapping. A priced
    subtotal is still reported when a provider/model has no table entry or a
    call returned no usage, but ``pricing_complete`` makes that lower-bound
    status explicit.

    The ledger records actual audit calls and tokens only.  It deliberately does
    not project a speculative "repeats to stable" production cost from a small
    audit sample.
    """

    buckets: dict[str, dict[str, Any]] = {}
    for record in call_records:
        category = str(record.get("category") or "uncategorized")
        bucket = buckets.setdefault(category, _empty_bucket())
        bucket["calls"] += 1

        usage = record.get("usage")
        if not isinstance(usage, dict) or not any(
            key in usage for key in ("input_tokens", "output_tokens")
        ):
            bucket["missing_usage_calls"] += 1
            continue

        input_tokens = _token_count(usage.get("input_tokens"))
        output_tokens = _token_count(usage.get("output_tokens"))
        cached_input_tokens = min(
            _token_count(usage.get("cached_input_tokens")), input_tokens
        )
        cache_write_tokens = min(
            _token_count(usage.get("cache_write_tokens")),
            input_tokens - cached_input_tokens,
        )
        bucket["input_tokens"] += input_tokens
        bucket["cached_input_tokens"] += cached_input_tokens
        bucket["cache_write_tokens"] += cache_write_tokens
        bucket["output_tokens"] += output_tokens

        provider = str(record.get("provider") or "unknown")
        model = str(record.get("model") or "unknown")
        cost = call_cost_usd(
            provider,
            model,
            input_tokens,
            output_tokens,
            cached_input_tokens=cached_input_tokens,
            cache_write_tokens=cache_write_tokens,
        )
        pricing_complete = call_price_complete(
            provider,
            model,
            cached_input_tokens=cached_input_tokens,
            cache_write_tokens=cache_write_tokens,
        )
        if cost is None:
            bucket["unpriced_calls"] += 1
            bucket["unpriced_models"].add(f"{provider}/{model}")
        elif not pricing_complete:
            # Retain the known token-class subtotal, but expose that it is only
            # a lower bound instead of silently substituting an incorrect rate.
            bucket["unpriced_calls"] += 1
            bucket["unpriced_models"].add(f"{provider}/{model}")
            bucket["_total_usd"] += cost
        else:
            bucket["priced_calls"] += 1
            bucket["_total_usd"] += cost

    by_category = {
        category: _finalize_bucket(bucket)
        for category, bucket in sorted(buckets.items())
    }
    experiment = _aggregate_experiment(buckets)
    for category, alias in _CATEGORY_CALL_ALIASES.items():
        experiment[alias] = int(buckets.get(category, {}).get("calls", 0))

    return {
        "experiment": {**experiment, "by_category": by_category},
    }


def _empty_bucket() -> dict[str, Any]:
    return {
        "calls": 0,
        "priced_calls": 0,
        "unpriced_calls": 0,
        "missing_usage_calls": 0,
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "cache_write_tokens": 0,
        "output_tokens": 0,
        "_total_usd": 0.0,
        "unpriced_models": set(),
    }


def _finalize_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    return {
        "calls": bucket["calls"],
        "priced_calls": bucket["priced_calls"],
        "unpriced_calls": bucket["unpriced_calls"],
        "missing_usage_calls": bucket["missing_usage_calls"],
        "pricing_complete": not bucket["unpriced_calls"]
        and not bucket["missing_usage_calls"],
        "input_tokens": bucket["input_tokens"],
        "cached_input_tokens": bucket["cached_input_tokens"],
        "cache_write_tokens": bucket["cache_write_tokens"],
        "output_tokens": bucket["output_tokens"],
        "total_tokens": bucket["input_tokens"] + bucket["output_tokens"],
        "total_usd": round(bucket["_total_usd"], 6),
        "unpriced_models": sorted(bucket["unpriced_models"]),
    }


def _aggregate_experiment(buckets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    unpriced_models: set[str] = set()
    for bucket in buckets.values():
        unpriced_models.update(bucket["unpriced_models"])

    input_tokens = sum(bucket["input_tokens"] for bucket in buckets.values())
    output_tokens = sum(bucket["output_tokens"] for bucket in buckets.values())
    unpriced_calls = sum(bucket["unpriced_calls"] for bucket in buckets.values())
    missing_usage_calls = sum(
        bucket["missing_usage_calls"] for bucket in buckets.values()
    )
    return {
        "total_calls": sum(bucket["calls"] for bucket in buckets.values()),
        "priced_calls": sum(bucket["priced_calls"] for bucket in buckets.values()),
        "unpriced_calls": unpriced_calls,
        "missing_usage_calls": missing_usage_calls,
        "pricing_complete": not unpriced_calls and not missing_usage_calls,
        "input_tokens": input_tokens,
        "cached_input_tokens": sum(
            bucket["cached_input_tokens"] for bucket in buckets.values()
        ),
        "cache_write_tokens": sum(
            bucket["cache_write_tokens"] for bucket in buckets.values()
        ),
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "total_usd": round(
            sum(bucket["_total_usd"] for bucket in buckets.values()), 6
        ),
        "unpriced_models": sorted(unpriced_models),
    }


def _token_count(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0
