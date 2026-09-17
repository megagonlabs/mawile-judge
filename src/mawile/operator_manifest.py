from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

from mawile.applications import (
    ApplicationOutcome,
    GenerationOutcome,
    summarize_application_outcomes,
)
from mawile.perturbations.base import DimensionSpec
from mawile.schemas import Perturbation
from mawile.validation.validators import validation_prompt


def build_operator_manifest(specs: list[DimensionSpec]) -> dict[str, Any]:
    """Describe the selected operators and their expected relations."""

    return {"operators": [_manifest_row(spec) for spec in specs]}


def build_perturbation_ledger(
    specs: list[DimensionSpec],
    attempted_perturbations: list[Perturbation],
    accepted_perturbations: list[Perturbation],
    validation_report: list[dict[str, Any]],
    *,
    application_outcomes: Sequence[ApplicationOutcome] | None = None,
    generation_outcomes: Sequence[GenerationOutcome] | None = None,
) -> list[dict[str, Any]]:
    """Build new-run rollups from canonical application records.

    Positional perturbation lists remain accepted for call compatibility and
    examples only.  Historical artifacts already carry their rendered ledger;
    rebuilding old inferred rows is intentionally not supported.
    """

    del accepted_perturbations
    if application_outcomes is None or generation_outcomes is None:
        raise ValueError(
            "New perturbation ledgers require canonical application_outcomes and generation_outcomes"
        )
    return _ledger_from_application_outcomes(
        specs,
        application_outcomes,
        generation_outcomes,
        attempted_perturbations,
        validation_report,
    )


def _ledger_from_application_outcomes(
    specs: list[DimensionSpec],
    applications: Sequence[ApplicationOutcome],
    generation: Sequence[GenerationOutcome],
    perturbations: Sequence[Perturbation],
    validation_report: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected = {spec.operator for spec in specs}
    unexpected = {row.operator for row in applications} | {row.operator for row in generation}
    if unexpected - selected:
        raise ValueError("Application records include unselected operators: " + ", ".join(sorted(unexpected - selected)))
    totals = summarize_application_outcomes(applications, generation)
    validation_by_variant = {
        str(row.get("variant_id")): row
        for row in validation_report
        if row.get("variant_id") is not None
    }
    examples = _examples_by_operator(list(perturbations), validation_by_variant)
    rows: list[dict[str, Any]] = []
    for spec in specs:
        rollup = totals["per_operator"].get(spec.operator, {})
        statuses = rollup.get("application_statuses", {})
        generation_statuses = rollup.get("generation_statuses", {})
        skipped = sum(
            statuses.get(status, 0)
            for status in ("baseline_ineligible", "generation_skipped")
        )
        rows.append(
            {
                **_manifest_row(spec),
                # Legacy aliases remain unique-variant counts. Explicit pair
                # counts below identify the replicated judge work for globals.
                "attempted": rollup.get("unique_variants", 0),
                "accepted": rollup.get("accepted_variants", 0),
                "rejected": rollup.get("rejected_variants", 0),
                "unavailable": rollup.get("unavailable_variants", 0),
                "skipped": skipped,
                "skipped_reason": _rollup_skip_reason(statuses),
                "failed": statuses.get("generation_failed", 0),
                "applications": rollup.get("applications", 0),
                "application_statuses": statuses,
                "generation_units": rollup.get("generation_units", 0),
                "generation_statuses": generation_statuses,
                "unique_variants": rollup.get("unique_variants", 0),
                "accepted_variants": rollup.get("accepted_variants", 0),
                "rejected_variants": rollup.get("rejected_variants", 0),
                "unavailable_variants": rollup.get("unavailable_variants", 0),
                "variant_item_pairs": rollup.get("variant_item_pairs", 0),
                "expected_judge_calls": rollup.get("expected_judge_calls", 0),
                "attempted_judge_calls": rollup.get("attempted_judge_calls", 0),
                "analyzable_judge_calls": rollup.get("analyzable_judge_calls", 0),
                "examples": examples.get(spec.operator, [])[:3],
            }
        )
    return rows


def _rollup_skip_reason(statuses: Mapping[str, int]) -> str | None:
    for status in ("baseline_ineligible", "generation_skipped"):
        if statuses.get(status):
            return status
    return None


def operator_manifest_from_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    """Read the operator manifest from a current MAWILE artifact."""
    manifest = artifact.get("operator_manifest")
    return manifest if isinstance(manifest, dict) else {}


def _manifest_row(spec: DimensionSpec) -> dict[str, Any]:
    return {
        "operator": spec.operator,
        "dimension": spec.dimension,
        "family": spec.family.value,
        "target": spec.target,
        "applies_to": spec.applies_to,
        "touches": list(spec.touches),
        "expected_effect": spec.default_effect.value,
        "expected_relation": spec.expected_relation.value,
        "generation_method": "rule" if spec.kind == "det" else "llm",
        "generation_instruction": spec.instruction,
        "validation_kind": spec.validation_kind,
        "validation_prompt": validation_prompt(spec.validation_kind),
        "limitations": list(spec.limitations),
    }


def _examples_by_operator(
    perturbations: list[Perturbation],
    validation_by_variant: dict[str, dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for perturbation in perturbations:
        validation = validation_by_variant.get(perturbation.variant_id, {})
        examples[perturbation.operator].append(
            {
                "variant_id": perturbation.variant_id,
                "item_id": perturbation.item_id,
                "changed_fields": perturbation.changed_fields,
                "expected_effect": perturbation.expected_effect.value,
                "expected_relation": perturbation.expected_relation.value,
                "validation_status": perturbation.validity_status,
                "validation_reason": validation.get("reason"),
                "summary": perturbation.metadata.get("summary"),
            }
        )
    return examples
