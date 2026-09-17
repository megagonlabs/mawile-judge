from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Literal

from mawile.schemas import (
    ExpectedRelation,
    Item,
    JudgeConfig,
    Perturbation,
    PerturbationExpectedEffect,
    PerturbationFamily,
)

# A validation_kind names the question the validator must answer for a generated
# or deterministic candidate. Construction mode never implies scientific validity.
ValidationKind = Literal[
    "meaning_preserve",
    "typo_preserve",
    "sample_plausibility",
    "same_answer",
    "facts_preserve",
    "pair_coherence",
    "judge_prompt_preserve",
    "judge_rubric_preserve",
    "judge_rubric_rescale",
    "directional_degradation",
    "format_degradation",
    "unmet_requirement",
    "refusal_degradation",
]


@dataclass(frozen=True)
class DimensionSpec:
    """One perturbation dimension, declared in the registry.

    ``det`` dimensions carry a ``builder`` that emits variants offline; ``llm``
    dimensions carry an ``instruction`` the perturbation agent runs and a
    ``validation_kind`` gating the result. ``applies_to`` says whether the edit
    lands on the judge artifacts (once) or on each item. ``default_effect`` is
    the expected relation recorded in the operator catalog and run manifest.
    """

    operator: str
    dimension: str
    family: PerturbationFamily
    kind: Literal["det", "llm"]
    target: str
    applies_to: Literal["judge", "item"]
    touches: tuple[str, ...] = ()
    default_effect: PerturbationExpectedEffect = PerturbationExpectedEffect.SAME_VERDICT
    expected_relation: ExpectedRelation = ExpectedRelation.INVARIANT
    builder: Callable[["DimensionSpec", Any, list[Item]], list[Perturbation]] | None = None
    instruction: str | None = None
    validation_kind: ValidationKind | None = None
    paired: bool = False
    applicability: Callable[[Any], bool] | None = None
    description: str | None = None
    limitations: tuple[str, ...] = ()


def build_variant(
    spec: DimensionSpec,
    *,
    variant_id: str,
    summary: str,
    item_id: str = "*",
    changed_fields: list[str] | None = None,
    judge_config_overrides: dict[str, Any] | None = None,
    item_overrides: dict[str, Any] | None = None,
    extra_metadata: dict[str, Any] | None = None,
    validity_status: Literal["accepted", "rejected", "unavailable", "not_needed"] = "not_needed",
) -> Perturbation:
    metadata: dict[str, Any] = {
        "summary": summary,
        "dimension": spec.dimension,
        "intended_operator": spec.operator,
        "declared_relation": spec.expected_relation.value,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    return Perturbation(
        variant_id=variant_id,
        item_id=item_id,
        family=spec.family,
        operator=spec.operator,
        expected_effect=spec.default_effect,
        expected_relation=spec.expected_relation,
        changed_fields=changed_fields if changed_fields is not None else [spec.target],
        validity_status=validity_status,
        metadata=metadata,
        judge_config_overrides=judge_config_overrides or {},
        item_overrides=item_overrides or {},
    )


def stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


ITEM_FIELDS = ("input", "output")


def labeled_item(item: Item) -> str:
    """Render the present transcript sections, each wrapped in a named tag.

    Used to give the perturbation agent (and the pair-coherence validator) the
    whole transcript as read-only context.
    """

    parts = []
    for name in ITEM_FIELDS:
        value = getattr(item, name)
        if value is not None:
            parts.append(f"<{name}>\n{stringify(value)}\n</{name}>")
    return "\n\n".join(parts)


def agent_response_params(decoding_params: dict[str, Any]) -> dict[str, Any]:
    """Decoding options accepted by the standalone Responses-based data scripts."""

    allowed = {"max_output_tokens", "reasoning", "service_tier", "temperature", "top_p"}
    return {
        key: value
        for key, value in decoding_params.items()
        if key in allowed and value is not None
    }


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Parse one JSON object from a generation response.

    Generation output is a machine interface.  Accept a JSON code fence for
    provider formatting, but reject prose wrapped around an object so malformed
    units become visible skips instead of silently producing a probe.
    """

    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) < 3 or lines[0].strip().casefold() not in {"```", "```json"} or lines[-1].strip() != "```":
            return None
        cleaned = "\n".join(lines[1:-1]).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def apply_judge_overrides(config: JudgeConfig, overrides: dict) -> JudgeConfig:
    if not overrides:
        return config

    payload = config.model_dump()
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(payload.get(key), dict):
            merged = deepcopy(payload[key])
            merged.update(value)
            payload[key] = merged
        else:
            payload[key] = value
    return JudgeConfig.model_validate(payload)


def apply_item_overrides(item: Item, overrides: dict) -> Item:
    if not overrides:
        return item

    payload = item.model_dump()
    payload.update(overrides)
    return Item.model_validate(payload)
