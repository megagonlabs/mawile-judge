from __future__ import annotations

import copy
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mawile.io import load_items
from mawile.judges.prompting import build_judge_instructions
from mawile.measurement import validate_items_for_judge
from mawile.schemas import AuditRunConfig, Item

_MAX_EXAMPLE_CHARS = 6000
_TRUNCATION_MIN_CHARS = 80


@dataclass(frozen=True)
class SuggestionContext:
    text: str
    metadata: dict[str, Any]


def build_suggestion_context(
    config: AuditRunConfig,
    *,
    rng: Any | None = None,
) -> SuggestionContext:
    """Build bounded, audit-side context shared by both suggestion agents."""

    items = load_items(config.data)
    validate_items_for_judge(items, config.judge)
    budget = config.planner_agent.dataset_context_max_chars
    summary_budget = min(10000, max(1000, budget // 4))
    document_budget = min(10000, max(0, budget // 4))

    summary_raw = json.dumps(
        _dataset_summary(config, items),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    summary, summary_truncated = _truncate_text(summary_raw, summary_budget)

    document = ""
    document_original_chars = 0
    document_truncated = False
    if config.data.context_path is not None:
        document_raw = config.data.context_path.read_text(encoding="utf-8")
        document_original_chars = len(document_raw)
        document, document_truncated = _truncate_text(document_raw, document_budget)

    fixed_sections = [
        "Dataset structure and aggregate statistics:\n" + summary,
    ]
    if config.data.context_path is not None:
        fixed_sections.append(
            "Explicit dataset context document "
            f"({config.data.context_path.name}):\n{document}"
        )
    fixed_text = "\n\n".join(fixed_sections)
    samples_budget = max(0, budget - len(fixed_text) - 64)
    sampling_seed = config.planner_agent.sampling_seed
    samples, sampled_ids, all_included, truncated_item_ids, examples_truncated = _sample_items(
        items,
        samples_budget,
        rng=rng or random.Random(sampling_seed),
    )
    sample_text = "\n\n".join(samples) if samples else "(no dataset items)"
    dataset_text = (
        f"{fixed_text}\n\n"
        "Sampled canonical items:\n"
        f"{sample_text}"
    )
    if len(dataset_text) > budget:
        # Header arithmetic and very small configured budgets can leave a small
        # overrun. Enforce the public limit exactly as a final guardrail.
        dataset_text, final_truncated = _truncate_text(dataset_text, budget)
    else:
        final_truncated = False

    metadata = {
        "dataset_context_max_chars": budget,
        "dataset_context_chars": len(dataset_text),
        "item_count": len(items),
        "sampled_item_count": len(sampled_ids),
        "sampled_item_ids": sampled_ids,
        "truncated_item_ids": truncated_item_ids,
        "all_items_included": all_included,
        "context_path": str(config.data.context_path) if config.data.context_path else None,
        "summary_chars": len(summary),
        "summary_truncated": summary_truncated,
        "context_document_chars": len(document),
        "context_document_original_chars": document_original_chars,
        "context_document_truncated": document_truncated,
        "examples_truncated": examples_truncated,
        "final_truncated": final_truncated,
        "sampling_method": "all_items" if all_included else "seeded_random",
        "sampling_seed": sampling_seed if rng is None else None,
        "truncated": bool(
            summary_truncated
            or document_truncated
            or examples_truncated
            or final_truncated
        ),
    }
    return SuggestionContext(text=dataset_text, metadata=metadata)


def build_audit_context(config: AuditRunConfig) -> str:
    """Render the complete suggestion-relevant audit and judge configuration."""

    custom = [
        item.model_dump(mode="json")
        for item in config.audit.custom_perturbations
        if item.enabled
    ]
    audit = config.audit.model_dump(mode="json")
    audit["custom_perturbations"] = custom
    payload = {
        "judge": config.judge.model_dump(mode="json"),
        "assembled_judge_instructions": build_judge_instructions(config.judge),
        "data_mapping": {
            "items_file": config.data.items_path.name,
            "context_file": (
                config.data.context_path.name if config.data.context_path is not None else None
            ),
            "input_field": config.data.input_field,
            "output_field": config.data.output_field,
            "gold_field": config.data.gold_field,
        },
        "audit": audit,
        "agents": {
            "perturbation": config.perturbation_agent.model_dump(mode="json"),
            "planner": {
                "provider": config.resolved_planner_provider(),
                "model": config.resolved_planner_model(),
                "decoding_params": config.resolved_planner_decoding_params(),
            },
            "validator": {
                "provider": config.resolved_validator_provider(),
                "model": config.resolved_validator_model(),
                "decoding_params": config.resolved_validator_decoding_params(),
            },
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def _dataset_summary(config: AuditRunConfig, items: list[Item]) -> dict[str, Any]:
    field_names = (
        "input",
        "output",
        "metadata",
        "gold_label",
        "human_label_distribution",
    )
    fields: dict[str, Any] = {}
    for name in field_names:
        present = [getattr(item, name) for item in items if _is_present(getattr(item, name))]
        lengths = [len(_json_text(value)) for value in present]
        fields[name] = {
            "present": len(present),
            "missing": len(items) - len(present),
            "coverage": (len(present) / len(items)) if items else 0.0,
            "types": dict(sorted(Counter(_type_name(value) for value in present).items())),
            "length_chars": _numeric_summary(lengths),
        }

    metadata_values: dict[str, list[Any]] = defaultdict(list)
    for item in items:
        for key, value in item.metadata.items():
            metadata_values[str(key)].append(value)

    return {
        "items_file": config.data.items_path.name,
        "item_count": len(items),
        "item_shape": _item_shape(items),
        "fields": fields,
        "metadata": {
            key: _value_summary(values, total=len(items))
            for key, values in sorted(metadata_values.items())
        },
        "gold_labels": _value_summary(
            [item.gold_label for item in items if item.gold_label is not None],
            total=len(items),
        ),
        "human_label_distributions": _value_summary(
            [
                item.human_label_distribution
                for item in items
                if item.human_label_distribution is not None
            ],
            total=len(items),
        ),
    }


def _sample_items(
    items: list[Item],
    budget: int,
    *,
    rng: Any | None,
) -> tuple[list[str], list[str], bool, list[str], bool]:
    if not items or budget <= 0:
        return [], [], not items, [], bool(items)

    full = [_item_json(item) for item in items]
    separator_cost = max(0, len(items) - 1) * 2
    if sum(len(text) for text in full) + separator_cost <= budget:
        return full, [item.item_id for item in items], True, [], False

    order = list(range(len(items)))
    (rng or random.SystemRandom()).shuffle(order)
    rendered: list[str] = []
    sampled_ids: list[str] = []
    truncated_ids: list[str] = []
    remaining = budget
    any_truncated = False
    for index in order:
        separator = 2 if rendered else 0
        if remaining - separator < _TRUNCATION_MIN_CHARS:
            break
        item_budget = min(_MAX_EXAMPLE_CHARS, remaining - separator)
        text, truncated = _bounded_item_json(items[index], item_budget)
        if len(text) + separator > remaining:
            continue
        rendered.append(text)
        sampled_ids.append(items[index].item_id)
        if truncated:
            truncated_ids.append(items[index].item_id)
        remaining -= len(text) + separator
        any_truncated = any_truncated or truncated
    return (
        rendered,
        sampled_ids,
        False,
        truncated_ids,
        any_truncated or len(sampled_ids) < len(items),
    )


def _item_json(item: Item) -> str:
    return json.dumps(
        _item_payload(item),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )


def _bounded_item_json(item: Item, budget: int) -> tuple[str, bool]:
    payload = _item_payload(item)
    raw = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if len(raw) <= budget:
        return raw, False

    longest = max(_string_lengths(payload), default=0)
    low, high = 8, max(8, longest)
    best: str | None = None
    while low <= high:
        limit = (low + high) // 2
        candidate = json.dumps(
            _truncate_strings(payload, limit),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        if len(candidate) <= budget:
            best = candidate
            low = limit + 1
        else:
            high = limit - 1
    if best is not None:
        return best, True

    preview, _ = _truncate_text(raw, max(16, budget // 2))
    fallback = json.dumps(
        {
            "item_id": _truncate_text(item.item_id, max(8, budget // 4))[0],
            "_truncated": True,
            "preview": preview,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    if len(fallback) > budget:
        fallback = json.dumps({"_truncated": True}, separators=(",", ":"))
    return fallback, True


def _item_payload(item: Item) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "item_id": item.item_id,
        "input": item.input,
        "output": item.output,
    }
    if item.metadata:
        payload["metadata"] = item.metadata
    if item.gold_label is not None:
        payload["gold_label"] = item.gold_label
    if item.human_label_distribution is not None:
        payload["human_label_distribution"] = item.human_label_distribution
    return payload


def _item_shape(items: list[Item]) -> str:
    if not items:
        return "empty"
    pairwise = sum(1 for item in items if item.is_pairwise)
    if pairwise == len(items) and items:
        return "pairwise"
    if pairwise == 0:
        return "pointwise"
    return "mixed"


def _value_summary(values: list[Any], *, total: int) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "present": len(values),
        "missing": total - len(values),
        "types": dict(sorted(Counter(_type_name(value) for value in values).items())),
    }
    numeric = [float(value) for value in values if _finite_number(value)]
    if numeric and len(numeric) == len(values):
        summary["numeric"] = _numeric_summary(numeric)
        return summary
    encoded = [_display_value(value) for value in values]
    counts = Counter(encoded)
    summary["unique"] = len(counts)
    if len(counts) <= 20:
        summary["counts"] = dict(sorted(counts.items()))
    else:
        summary["most_common"] = dict(counts.most_common(10))
    return summary


def _numeric_summary(values: list[float | int]) -> dict[str, Any] | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    p90_index = min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1)
    return {
        "min": ordered[0],
        "median": statistics.median(ordered),
        "p90": ordered[p90_index],
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


def _truncate_strings(value: Any, limit: int) -> Any:
    if isinstance(value, str):
        return _truncate_text(value, limit)[0]
    if isinstance(value, list):
        return [_truncate_strings(item, limit) for item in value]
    if isinstance(value, dict):
        return {str(key): _truncate_strings(item, limit) for key, item in value.items()}
    return copy.deepcopy(value)


def _string_lengths(value: Any) -> list[int]:
    if isinstance(value, str):
        return [len(value)]
    if isinstance(value, list):
        return [length for item in value for length in _string_lengths(item)]
    if isinstance(value, dict):
        return [length for item in value.values() for length in _string_lengths(item)]
    return []


def _truncate_text(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    if limit <= 0:
        return "", True
    marker = f"\n…<truncated {len(text) - limit} chars>…\n"
    if len(marker) >= limit:
        compact_marker = "<truncated>"
        return compact_marker[:limit], True
    available = limit - len(marker)
    head = (available + 1) // 2
    tail = available // 2
    return text[:head] + marker + (text[-tail:] if tail else ""), True


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _display_value(value: Any) -> str:
    return value if isinstance(value, str) else _json_text(value)


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _is_present(value: Any) -> bool:
    return value is not None and value != "" and value != {} and value != []


__all__ = [
    "SuggestionContext",
    "build_audit_context",
    "build_suggestion_context",
]
