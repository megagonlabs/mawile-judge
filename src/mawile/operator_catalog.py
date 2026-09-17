from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

from mawile.perturbations import build_perturbations
from mawile.applications import resolve_generation_units
from mawile.perturbations.base import (
    DimensionSpec,
    apply_item_overrides,
    apply_judge_overrides,
)
from mawile.perturbations.registry import (
    DIMENSIONS,
    DIRECTIONAL_DIMENSIONS,
    enabled_dimensions,
)
from mawile.providers import provider_has_api_key, resolve_provider
from mawile.schemas import AuditRunConfig, Item, Perturbation
from mawile.validation.validators import validation_prompt, validation_prompts


@dataclass(frozen=True)
class DimensionRow:
    family: str
    operator: str
    dimension: str
    kind: str
    target: str
    applies_to: str
    touches: tuple[str, ...]
    expected_effect: str
    expected_relation: str
    status: str
    validation_kind: str | None


@dataclass(frozen=True)
class CatalogOperatorRow:
    family: str
    operator: str
    dimension: str
    description: str | None
    kind: str
    target: str
    applies_to: str
    touches: tuple[str, ...]
    expected_effect: str
    expected_relation: str
    instruction: str | None
    validation_kind: str | None
    validation_prompt: str | None
    limitations: tuple[str, ...]


@dataclass(frozen=True)
class ValidationCatalogRow:
    kind: str
    prompt: str
    operators: tuple[str, ...]


@dataclass(frozen=True)
class PerturbationPreview:
    family: str
    operator: str
    dimension: str
    kind: str
    target: str
    applies_to: str
    expected_effect: str
    expected_relation: str
    variant_id: str | None
    item_id: str | None
    summary: str
    before: Any
    after: Any
    changed_field: str
    status: str
    instruction: str | None = None
    validation_kind: str | None = None


def dimension_rows(
    config: AuditRunConfig,
    *,
    has_api_key: bool = False,
    include_not_runnable: bool = True,
) -> list[DimensionRow]:
    rows: list[DimensionRow] = []
    for spec in enabled_dimensions(config):
        status = _dimension_status(spec, config, has_api_key=has_api_key)
        if not include_not_runnable and status != "Runs in this environment":
            continue
        rows.append(
            DimensionRow(
                family=spec.family.value,
                operator=spec.operator,
                dimension=spec.dimension,
                kind=spec.kind,
                target=spec.target,
                applies_to=spec.applies_to,
                touches=spec.touches,
                expected_effect=spec.default_effect.value,
                expected_relation=spec.expected_relation.value,
                status=status,
                validation_kind=spec.validation_kind,
            )
        )
    return rows


def catalog_operator_rows(specs: Sequence[DimensionSpec]) -> list[CatalogOperatorRow]:
    return [
        CatalogOperatorRow(
            family=spec.family.value,
            operator=spec.operator,
            dimension=spec.dimension,
            description=spec.description,
            kind=spec.kind,
            target=spec.target,
            applies_to=spec.applies_to,
            touches=spec.touches,
            expected_effect=spec.default_effect.value,
            expected_relation=spec.expected_relation.value,
            instruction=spec.instruction,
            validation_kind=spec.validation_kind,
            validation_prompt=validation_prompt(spec.validation_kind),
            limitations=spec.limitations,
        )
        for spec in specs
    ]


def validation_catalog_rows(
    specs: Sequence[DimensionSpec] | None = None,
) -> list[ValidationCatalogRow]:
    specs = specs or [*DIMENSIONS, *DIRECTIONAL_DIMENSIONS]
    operators_by_kind: dict[str, list[str]] = defaultdict(list)
    for spec in specs:
        if spec.validation_kind:
            operators_by_kind[spec.validation_kind].append(spec.operator)
    return [
        ValidationCatalogRow(
            kind=kind,
            prompt=prompt,
            operators=tuple(sorted(operators_by_kind.get(kind, []))),
        )
        for kind, prompt in validation_prompts().items()
    ]


def operator_status(
    spec: DimensionSpec,
    config: AuditRunConfig,
    *,
    has_api_key: bool = False,
) -> str:
    return _dimension_status(spec, config, has_api_key=has_api_key)


def build_perturbation_previews(
    config: AuditRunConfig,
    items: list[Item],
    *,
    has_api_key: bool = False,
    max_examples_per_dimension: int = 1,
    include_not_runnable: bool = True,
) -> list[PerturbationPreview]:
    return build_perturbation_previews_for_specs(
        config,
        items,
        enabled_dimensions(config),
        has_api_key=has_api_key,
        max_examples_per_dimension=max_examples_per_dimension,
        include_not_runnable=include_not_runnable,
    )


def build_perturbation_previews_for_specs(
    config: AuditRunConfig,
    items: list[Item],
    specs: Sequence[DimensionSpec],
    *,
    has_api_key: bool = False,
    max_examples_per_dimension: int = 1,
    include_not_runnable: bool = True,
) -> list[PerturbationPreview]:
    previews: list[PerturbationPreview] = []
    for spec in specs:
        status = _dimension_status(spec, config, has_api_key=has_api_key)
        if not include_not_runnable and status != "Runs in this environment":
            continue
        if spec.kind == "det" and spec.builder is not None:
            variants = spec.builder(spec, config, items)
            for perturbation in variants[:max_examples_per_dimension]:
                previews.append(
                    _preview_from_perturbation(
                        spec=spec,
                        config=config,
                        items=items,
                        perturbation=perturbation,
                        status=status,
                    )
                )
            if not variants:
                previews.append(_empty_preview(spec, config=config, status=status, items=items))
            continue
        previews.append(_llm_preview(spec, config=config, items=items, status=status))
    return previews


def estimate_run_size(config: AuditRunConfig, items: list[Item]) -> dict[str, int]:
    deterministic = build_perturbations(config, items)
    original_calls = len(items) * config.audit.repeats
    deterministic_calls = _call_count_for_perturbations(
        deterministic,
        item_count=len(items),
        repeats=config.audit.repeats,
    )
    units = resolve_generation_units(enabled_dimensions(config), items)
    llm_units = [unit for unit in units if unit.spec.kind == "llm"]
    # This is an upper bound over selected scopes, even when prerequisites block
    # the actual generation (for example a mock agent in the UI).
    llm_generation_units = len(llm_units)
    llm_judge_calls = sum(
        (len(items) if unit.spec.applies_to == "judge" else 1)
        * config.audit.repeats
        for unit in llm_units
    )
    known_judge_calls = original_calls + deterministic_calls
    return {
        "items": len(items),
        "deterministic_perturbations": len(deterministic),
        "generation_units_upper_bound": len(units),
        "original_judge_calls": original_calls,
        "deterministic_judge_calls": deterministic_calls,
        "llm_generation_units": llm_generation_units,
        "llm_judge_calls": llm_judge_calls,
        "known_judge_calls": known_judge_calls,
        "max_judge_calls": known_judge_calls + llm_judge_calls,
    }


def _dimension_status(
    spec: DimensionSpec,
    config: AuditRunConfig,
    *,
    has_api_key: bool,
) -> str:
    if spec.kind == "det" and not spec.validation_kind:
        return "Runs in this environment"
    if spec.kind == "llm" and config.perturbation_agent.model == "mock":
        return "Blocked: perturbation agent model is mock"
    # Generator and validator prerequisites are independent.  Deterministic
    # operators with semantic validation do not need a generator credential.
    if spec.kind == "llm" and not has_api_key:
        provider = resolve_provider(config, config.perturbation_agent.provider)
        return f"Needs {provider.api_key_env}"
    if spec.validation_kind and config.resolved_validator_model() == "mock":
        return "Blocked: independent validator model is mock"
    if spec.validation_kind and not provider_has_api_key(
        config, config.resolved_validator_provider()
    ):
        provider = resolve_provider(config, config.resolved_validator_provider())
        return f"Needs {provider.api_key_env}"
    return "Runs in this environment"


def _preview_from_perturbation(
    *,
    spec: DimensionSpec,
    config: AuditRunConfig,
    items: list[Item],
    perturbation: Perturbation,
    status: str,
) -> PerturbationPreview:
    item = _item_for_perturbation(items, perturbation)
    submitted_config = apply_judge_overrides(config.judge, perturbation.judge_config_overrides)
    submitted_item = apply_item_overrides(item, perturbation.item_overrides)
    before, after, changed_field = _field_change_values(
        spec=spec,
        item=item,
        submitted_item=submitted_item,
        config=config,
        submitted_config=submitted_config,
        perturbation=perturbation,
    )
    return PerturbationPreview(
        family=perturbation.family.value,
        operator=perturbation.operator,
        dimension=str(perturbation.metadata.get("dimension", spec.dimension)),
        kind=spec.kind,
        target=spec.target,
        applies_to=spec.applies_to,
        expected_effect=perturbation.expected_effect.value,
        expected_relation=perturbation.expected_relation.value,
        variant_id=perturbation.variant_id,
        item_id=perturbation.item_id,
        summary=str(perturbation.metadata.get("summary", "")),
        before=before,
        after=after,
        changed_field=changed_field,
        status=status,
        validation_kind=spec.validation_kind,
    )


def _empty_preview(
    spec: DimensionSpec,
    *,
    config: AuditRunConfig,
    status: str,
    items: list[Item],
) -> PerturbationPreview:
    before = _sample_before(spec, config, items)
    return PerturbationPreview(
        family=spec.family.value,
        operator=spec.operator,
        dimension=spec.dimension,
        kind=spec.kind,
        target=spec.target,
        applies_to=spec.applies_to,
        expected_effect=spec.default_effect.value,
        expected_relation=spec.expected_relation.value,
        variant_id=None,
        item_id=None,
        summary=_no_preview_summary(spec),
        before=before,
        after=None,
        changed_field=spec.target,
        status=status,
        validation_kind=spec.validation_kind,
    )


def _no_preview_summary(spec: DimensionSpec) -> str:
    return (
        "No concrete example was produced for the current sample because the "
        "operator did not find a safe, applicable field to rewrite."
    )


def _llm_preview(
    spec: DimensionSpec,
    *,
    config: AuditRunConfig,
    items: list[Item],
    status: str,
) -> PerturbationPreview:
    before = _sample_before(spec, config, items)
    return PerturbationPreview(
        family=spec.family.value,
        operator=spec.operator,
        dimension=spec.dimension,
        kind=spec.kind,
        target=spec.target,
        applies_to=spec.applies_to,
        expected_effect=spec.default_effect.value,
        expected_relation=spec.expected_relation.value,
        variant_id=None,
        item_id=items[0].item_id if items and spec.applies_to == "item" else "*",
        summary="Generated at run time by the perturbation agent.",
        before=before,
        after=None,
        changed_field=spec.target,
        status=status,
        instruction=spec.instruction,
        validation_kind=spec.validation_kind,
    )


def _sample_before(
    spec: DimensionSpec,
    config: AuditRunConfig | None,
    items: list[Item],
) -> Any:
    if spec.target == "judge.prompt_template" and config is not None:
        return config.judge.prompt_template
    if spec.target == "judge.rubric" and config is not None:
        return config.judge.rubric
    if spec.target == "judge.prompt_block_order" and config is not None:
        return config.judge.prompt_block_order
    if items and spec.target.startswith("item."):
        return _nested_lookup(
            items[0].model_dump(mode="json"), spec.target.split(".", 1)[1]
        )
    return None


def _item_for_perturbation(items: list[Item], perturbation: Perturbation) -> Item:
    if perturbation.item_id == "*" or not items:
        return items[0]
    by_id = {item.item_id: item for item in items}
    return by_id.get(perturbation.item_id, items[0])


def _field_change_values(
    *,
    spec: DimensionSpec,
    item: Item,
    submitted_item: Item,
    config: AuditRunConfig,
    submitted_config: Any,
    perturbation: Perturbation,
) -> tuple[Any, Any, str]:
    changed_field = perturbation.changed_fields[0] if perturbation.changed_fields else spec.target
    before_payloads = {
        "item": item.model_dump(mode="json"),
        "judge": config.judge.model_dump(mode="json"),
    }
    after_payloads = {
        "item": submitted_item.model_dump(mode="json"),
        "judge": submitted_config.model_dump(mode="json"),
    }
    root, _, path = changed_field.partition(".")
    return (
        _nested_lookup(before_payloads.get(root, {}), path),
        _nested_lookup(after_payloads.get(root, {}), path),
        changed_field,
    )


def _nested_lookup(payload: Any, dotted_path: str) -> Any:
    current = payload
    for part in dotted_path.split("."):
        if not part:
            continue
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def _call_count_for_perturbations(
    perturbations: list[Perturbation],
    *,
    item_count: int,
    repeats: int,
) -> int:
    total = 0
    for perturbation in perturbations:
        target_count = item_count if perturbation.item_id == "*" else 1
        total += target_count * repeats
    return total


__all__ = [
    "CatalogOperatorRow",
    "DimensionRow",
    "PerturbationPreview",
    "ValidationCatalogRow",
    "build_perturbation_previews",
    "build_perturbation_previews_for_specs",
    "catalog_operator_rows",
    "dimension_rows",
    "estimate_run_size",
    "operator_status",
    "validation_catalog_rows",
]
