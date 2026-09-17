"""Canonical accounting for selected perturbation applications.

This module deliberately models only the units a generator works on and the
operator/item applications an audit measures.  It has no workflow or I/O
responsibilities, so both the pipeline and tests can reconcile generated,
validated, and judged variants through the same strict rules.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from hashlib import sha256
import json
from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence

from mawile.measurement import result_is_analyzable
from mawile.schemas import (
    ExpectedRelation,
    Item,
    JudgeConfig,
    JudgeResult,
    Perturbation,
    PerturbationExpectedEffect,
)

if TYPE_CHECKING:
    from mawile.perturbations.base import DimensionSpec
else:
    DimensionSpec = Any

GenerationStatus = Literal["generated", "skipped", "failed"]
ApplicationStatus = Literal[
    "baseline_ineligible",
    "generation_skipped",
    "generation_failed",
    "validation_rejected",
    "validation_unavailable",
    "accepted",
]


@dataclass(frozen=True)
class GenerationUnit:
    """One generator invocation scope; global judge edits have ``item=None``."""

    spec: DimensionSpec
    item: Item | None
    target_item_ids: list[str]
    unit_id: str

    @property
    def item_id(self) -> str:
        return "*" if self.item is None else self.item.item_id


@dataclass(frozen=True)
class GenerationOutcome:
    operator: str
    unit_id: str
    item_id: str
    status: GenerationStatus
    reason: str | None = None
    variant_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ApplicationOutcome:
    operator: str
    item_id: str
    status: ApplicationStatus
    reason: str | None = None
    unit_id: str | None = None
    variant_ids: list[str] = field(default_factory=list)
    accepted_variant_ids: list[str] = field(default_factory=list)
    rejected_variant_ids: list[str] = field(default_factory=list)
    expected_judge_calls: int = 0
    attempted_judge_calls: int = 0
    analyzable_judge_calls: int = 0
    unavailable_variant_ids: list[str] = field(default_factory=list)


def resolve_generation_units(
    specs: Sequence[DimensionSpec],
    items: Sequence[Item],
    *,
    directional_eligible_item_ids: set[str] | None = None,
) -> list[GenerationUnit]:
    """Return the exact generation scopes for selected specifications.

    An item-side directional spec has no unit for baseline-ineligible items;
    ``resolve_applications`` nevertheless emits a row for every selected
    operator/item pair so those exclusions are measured rather than inferred.
    """

    _validate_specs_and_items(specs, items)
    item_ids = [item.item_id for item in items]
    eligible = set(item_ids) if directional_eligible_item_ids is None else set(directional_eligible_item_ids)
    unknown_eligible = eligible - set(item_ids)
    if unknown_eligible:
        raise ValueError("Directional eligibility names unknown items: " + ", ".join(sorted(unknown_eligible)))

    units: list[GenerationUnit] = []
    for spec in specs:
        if spec.applies_to == "judge":
            targets = [item_id for item_id in item_ids if not _is_directional(spec) or item_id in eligible]
            if targets:
                units.append(_unit(spec, None, targets))
            continue
        for item in items:
            if _is_directional(spec) and item.item_id not in eligible:
                continue
            units.append(_unit(spec, item, [item.item_id]))
    return units


def resolve_applications(
    specs: Sequence[DimensionSpec],
    items: Sequence[Item],
    generation_units: Sequence[GenerationUnit],
    generation_outcomes: Sequence[GenerationOutcome],
    validated_perturbations: Sequence[Perturbation],
    validation_report: Sequence[Mapping[str, Any]],
    *,
    directional_eligible_item_ids: set[str] | None = None,
    directional_ineligible_reasons: Mapping[str, str] | None = None,
    repeats: int = 1,
) -> list[ApplicationOutcome]:
    """Strictly reconcile generation and validation into one row per application."""

    _validate_specs_and_items(specs, items)
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    expected_units = resolve_generation_units(
        specs, items, directional_eligible_item_ids=directional_eligible_item_ids
    )
    _validate_units(generation_units, expected_units)
    outcomes_by_unit = _validate_generation_outcomes(generation_outcomes, generation_units)
    variants_by_id = _validate_variants(specs, generation_units, generation_outcomes, validated_perturbations)
    rejection_reasons = _validation_reasons(validation_report, variants_by_id)

    item_ids = {item.item_id for item in items}
    eligible = item_ids if directional_eligible_item_ids is None else set(directional_eligible_item_ids)
    reasons = directional_ineligible_reasons or {}
    unknown_reason_ids = set(reasons) - item_ids
    if unknown_reason_ids:
        raise ValueError("Directional ineligibility reasons name unknown items: " + ", ".join(sorted(unknown_reason_ids)))
    units_by_scope = {(unit.spec.operator, unit.item_id): unit for unit in generation_units}

    applications: list[ApplicationOutcome] = []
    for spec in specs:
        for item in items:
            if _is_directional(spec) and item.item_id not in eligible:
                applications.append(
                    ApplicationOutcome(
                        operator=spec.operator,
                        item_id=item.item_id,
                        status="baseline_ineligible",
                        reason=reasons.get(item.item_id, "baseline_ineligible"),
                    )
                )
                continue
            source_id = "*" if spec.applies_to == "judge" else item.item_id
            unit = units_by_scope[(spec.operator, source_id)]
            outcome = outcomes_by_unit[unit.unit_id]
            if outcome.status == "skipped":
                applications.append(_application_from_generation(unit, item.item_id, outcome, "generation_skipped"))
                continue
            if outcome.status == "failed":
                applications.append(_application_from_generation(unit, item.item_id, outcome, "generation_failed"))
                continue
            variants = [variants_by_id[variant_id] for variant_id in outcome.variant_ids]
            accepted = [variant.variant_id for variant in variants if variant.validity_status in {"accepted", "not_needed"}]
            rejected = [variant.variant_id for variant in variants if variant.validity_status == "rejected"]
            unavailable = [variant.variant_id for variant in variants if variant.validity_status == "unavailable"]
            status: ApplicationStatus = "accepted" if accepted else "validation_rejected"
            reason = None
            if not accepted and unavailable:
                status = "validation_unavailable"
                reason = _first_reason(unavailable, rejection_reasons, "validation_unavailable")
            elif status == "validation_rejected":
                reason = _first_reason(rejected, rejection_reasons, "validation_rejected")
            applications.append(
                ApplicationOutcome(
                    operator=spec.operator,
                    item_id=item.item_id,
                    status=status,
                    reason=reason,
                    unit_id=unit.unit_id,
                    variant_ids=list(outcome.variant_ids),
                    accepted_variant_ids=accepted,
                    rejected_variant_ids=rejected,
                    unavailable_variant_ids=unavailable,
                    expected_judge_calls=len(accepted) * repeats,
                )
            )
    return applications


def record_judge_outcomes(
    applications: Sequence[ApplicationOutcome],
    judge_results: Sequence[JudgeResult],
    judge_config: JudgeConfig,
) -> list[ApplicationOutcome]:
    """Attach attempted and analyzable calls to canonical application rows."""

    by_operator_item = {(row.operator, row.item_id): row for row in applications}
    if len(by_operator_item) != len(applications):
        raise ValueError("Duplicate application outcomes")
    variant_owners: dict[str, list[ApplicationOutcome]] = defaultdict(list)
    for row in applications:
        for variant_id in row.accepted_variant_ids:
            variant_owners[variant_id].append(row)

    counts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for result in judge_results:
        owners = variant_owners.get(result.variant_id)
        if not owners:
            raise ValueError(f"Judge result references unaccepted or unknown variant_id {result.variant_id!r}")
        owner = next((row for row in owners if row.item_id == result.item_id), None)
        if owner is None:
            raise ValueError(
                f"Judge result item {result.item_id!r} is out of scope for variant {result.variant_id!r}"
            )
        bucket = counts[(owner.operator, owner.item_id)]
        bucket[0] += 1
        bucket[1] += int(result_is_analyzable(result, judge_config.output_type))

    return [
        replace(
            row,
            attempted_judge_calls=counts[(row.operator, row.item_id)][0],
            analyzable_judge_calls=counts[(row.operator, row.item_id)][1],
        )
        for row in applications
    ]


def summarize_application_outcomes(
    applications: Sequence[ApplicationOutcome],
    generation_outcomes: Sequence[GenerationOutcome],
) -> dict[str, Any]:
    """Return conservation-friendly totals and per-operator rollups."""

    app_counts = Counter(row.status for row in applications)
    generation_counts = Counter(row.status for row in generation_outcomes)
    per_operator: dict[str, dict[str, Any]] = {}
    for operator in sorted({row.operator for row in applications} | {row.operator for row in generation_outcomes}):
        rows = [row for row in applications if row.operator == operator]
        generation = [row for row in generation_outcomes if row.operator == operator]
        per_operator[operator] = {
            "applications": len(rows),
            "application_statuses": dict(Counter(row.status for row in rows)),
            "generation_units": len(generation),
            "generation_statuses": dict(Counter(row.status for row in generation)),
            "unique_variants": len({variant_id for row in generation for variant_id in row.variant_ids}),
            "variant_item_pairs": sum(len(row.variant_ids) for row in rows),
            "accepted_variants": len({variant_id for row in rows for variant_id in row.accepted_variant_ids}),
            "rejected_variants": len({variant_id for row in rows for variant_id in row.rejected_variant_ids}),
            "unavailable_variants": len({variant_id for row in rows for variant_id in row.unavailable_variant_ids}),
            "accepted_variant_item_pairs": sum(len(row.accepted_variant_ids) for row in rows),
            "rejected_variant_item_pairs": sum(len(row.rejected_variant_ids) for row in rows),
            "unavailable_variant_item_pairs": sum(len(row.unavailable_variant_ids) for row in rows),
            "expected_judge_calls": sum(row.expected_judge_calls for row in rows),
            "attempted_judge_calls": sum(row.attempted_judge_calls for row in rows),
            "analyzable_judge_calls": sum(row.analyzable_judge_calls for row in rows),
        }
    return {
        "applications": len(applications),
        "application_statuses": dict(app_counts),
        "generation_units": len(generation_outcomes),
        "generation_statuses": dict(generation_counts),
        "unique_variants": len({variant_id for row in generation_outcomes for variant_id in row.variant_ids}),
        "variant_item_pairs": sum(len(row.variant_ids) for row in applications),
        "accepted_variants": len({variant_id for row in applications for variant_id in row.accepted_variant_ids}),
        "rejected_variants": len({variant_id for row in applications for variant_id in row.rejected_variant_ids}),
        "unavailable_variants": len({variant_id for row in applications for variant_id in row.unavailable_variant_ids}),
        "accepted_variant_item_pairs": sum(len(row.accepted_variant_ids) for row in applications),
        "rejected_variant_item_pairs": sum(len(row.rejected_variant_ids) for row in applications),
        "unavailable_variant_item_pairs": sum(len(row.unavailable_variant_ids) for row in applications),
        "expected_judge_calls": sum(row.expected_judge_calls for row in applications),
        "attempted_judge_calls": sum(row.attempted_judge_calls for row in applications),
        "analyzable_judge_calls": sum(row.analyzable_judge_calls for row in applications),
        "per_operator": per_operator,
    }


def _unit(spec: DimensionSpec, item: Item | None, target_item_ids: list[str]) -> GenerationUnit:
    source = "*" if item is None else item.item_id
    identity = json.dumps([spec.operator, source, target_item_ids], separators=(",", ":"), ensure_ascii=True)
    return GenerationUnit(spec, item, list(target_item_ids), f"unit_{sha256(identity.encode()).hexdigest()[:20]}")


def _is_directional(spec: DimensionSpec) -> bool:
    return (
        spec.default_effect == PerturbationExpectedEffect.WORSE_VERDICT
        or spec.expected_relation == ExpectedRelation.DIRECTIONAL
    )


def _validate_specs_and_items(specs: Sequence[DimensionSpec], items: Sequence[Item]) -> None:
    operators = [spec.operator for spec in specs]
    if len(operators) != len(set(operators)):
        raise ValueError("Selected specifications contain duplicate operators")
    item_ids = [item.item_id for item in items]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("Items contain duplicate item IDs")


def _validate_units(
    units: Sequence[GenerationUnit], expected: Sequence[GenerationUnit]
) -> None:
    actual_by_id = {unit.unit_id: unit for unit in units}
    if len(actual_by_id) != len(units):
        raise ValueError("Duplicate generation unit IDs")
    expected_by_id = {unit.unit_id: unit for unit in expected}
    if set(actual_by_id) != set(expected_by_id):
        raise ValueError("Generation units do not exactly match the selected operator/item scopes")
    for unit_id, unit in actual_by_id.items():
        canonical = expected_by_id[unit_id]
        if unit.spec != canonical.spec or unit.item_id != canonical.item_id or unit.target_item_ids != canonical.target_item_ids:
            raise ValueError(f"Generation unit {unit_id!r} does not match its canonical scope")


def _validate_generation_outcomes(
    outcomes: Sequence[GenerationOutcome], units: Sequence[GenerationUnit]
) -> dict[str, GenerationOutcome]:
    by_unit = {outcome.unit_id: outcome for outcome in outcomes}
    if len(by_unit) != len(outcomes):
        raise ValueError("Duplicate generation outcomes for a unit")
    units_by_id = {unit.unit_id: unit for unit in units}
    if set(by_unit) != set(units_by_id):
        raise ValueError("Generation outcomes must contain exactly one outcome for every generation unit")
    seen_variants: set[str] = set()
    for unit_id, outcome in by_unit.items():
        unit = units_by_id[unit_id]
        if outcome.status not in {"generated", "skipped", "failed"}:
            raise ValueError(f"Unknown generation outcome status {outcome.status!r}")
        if outcome.operator != unit.spec.operator or outcome.item_id != unit.item_id:
            raise ValueError(f"Generation outcome {unit_id!r} does not match its unit operator/source")
        if len(outcome.variant_ids) != len(set(outcome.variant_ids)):
            raise ValueError(f"Generation outcome {unit_id!r} repeats a variant ID")
        if outcome.status == "generated" and not outcome.variant_ids:
            raise ValueError(f"Generated outcome {unit_id!r} must name at least one variant")
        if outcome.status != "generated" and outcome.variant_ids:
            raise ValueError(f"{outcome.status} outcome {unit_id!r} cannot name variants")
        overlap = seen_variants.intersection(outcome.variant_ids)
        if overlap:
            raise ValueError("A variant ID belongs to multiple generation outcomes: " + ", ".join(sorted(overlap)))
        seen_variants.update(outcome.variant_ids)
    return by_unit


def _validate_variants(
    specs: Sequence[DimensionSpec],
    units: Sequence[GenerationUnit],
    outcomes: Sequence[GenerationOutcome],
    variants: Sequence[Perturbation],
) -> dict[str, Perturbation]:
    by_id = {variant.variant_id: variant for variant in variants}
    if len(by_id) != len(variants):
        raise ValueError("Duplicate returned variant IDs")
    declared_ids = {variant_id for outcome in outcomes for variant_id in outcome.variant_ids}
    if set(by_id) != declared_ids:
        missing = declared_ids - set(by_id)
        extra = set(by_id) - declared_ids
        bits = []
        if missing:
            bits.append("missing IDs: " + ", ".join(sorted(missing)))
        if extra:
            bits.append("undeclared IDs: " + ", ".join(sorted(extra)))
        raise ValueError("Returned variants do not match generation outcomes (" + "; ".join(bits) + ")")
    specs_by_operator = {spec.operator: spec for spec in specs}
    units_by_id = {unit.unit_id: unit for unit in units}
    outcome_by_variant = {
        variant_id: outcome for outcome in outcomes for variant_id in outcome.variant_ids
    }
    for variant_id, variant in by_id.items():
        spec = specs_by_operator.get(variant.operator)
        if spec is None:
            raise ValueError(f"Returned variant {variant_id!r} uses unselected operator {variant.operator!r}")
        outcome = outcome_by_variant[variant_id]
        unit = units_by_id[outcome.unit_id]
        if (
            variant.operator != outcome.operator
            or unit.spec.operator != outcome.operator
            or unit.unit_id != outcome.unit_id
        ):
            raise ValueError(f"Returned variant {variant_id!r} is assigned to another operator's outcome")
        if variant.item_id != unit.item_id:
            raise ValueError(f"Returned variant {variant_id!r} is out of its generation unit scope")
        if (
            variant.family != spec.family
            or variant.expected_effect != spec.default_effect
            or variant.expected_relation != spec.expected_relation
        ):
            raise ValueError(f"Returned variant {variant_id!r} does not match selected operator identity")
    return by_id


def _validation_reasons(
    report: Sequence[Mapping[str, Any]], variants_by_id: Mapping[str, Perturbation]
) -> dict[str, str]:
    reasons: dict[str, str] = {}
    seen: set[str] = set()
    for row in report:
        variant_id = row.get("variant_id")
        if variant_id is None:
            continue
        if not isinstance(variant_id, str) or variant_id not in variants_by_id:
            raise ValueError("Validation report references an unknown variant ID")
        if variant_id in seen:
            raise ValueError(f"Validation report repeats variant_id {variant_id!r}")
        seen.add(variant_id)
        status = row.get("status")
        if status not in {"accepted", "rejected", "unavailable"}:
            raise ValueError(f"Validation report has invalid status for variant_id {variant_id!r}")
        validity = variants_by_id[variant_id].validity_status
        expected_status = "accepted" if validity == "not_needed" else validity
        if status != expected_status:
            raise ValueError(f"Validation report status disagrees with variant validity for {variant_id!r}")
        if status in {"rejected", "unavailable"}:
            reasons[variant_id] = str(row.get("reason") or f"validation_{status}")
    return reasons


def _application_from_generation(
    unit: GenerationUnit,
    item_id: str,
    outcome: GenerationOutcome,
    status: ApplicationStatus,
) -> ApplicationOutcome:
    return ApplicationOutcome(
        operator=unit.spec.operator,
        item_id=item_id,
        status=status,
        reason=outcome.reason or outcome.status,
        unit_id=unit.unit_id,
    )


def _first_reason(ids: Sequence[str], reasons: Mapping[str, str], fallback: str) -> str:
    return next((reasons[variant_id] for variant_id in ids if variant_id in reasons), fallback)


__all__ = [
    "ApplicationOutcome",
    "ApplicationStatus",
    "GenerationOutcome",
    "GenerationStatus",
    "GenerationUnit",
    "record_judge_outcomes",
    "resolve_applications",
    "resolve_generation_units",
    "summarize_application_outcomes",
]
