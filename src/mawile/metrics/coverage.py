from __future__ import annotations

from typing import Any

from mawile.measurement import result_is_analyzable
from mawile.schemas import AuditRunConfig, ItemRisk, JudgeResult, Perturbation


def compute_coverage(
    *,
    config: AuditRunConfig,
    source_item_count: int,
    attempted_perturbations: list[Perturbation],
    accepted_perturbations: list[Perturbation],
    validation_report: list[dict[str, Any]],
    judge_results: list[JudgeResult],
    item_risks: list[ItemRisk],
    application_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    judge_attempts = len(judge_results)
    transport_failures = sum(
        result.call_status == "transport_error" for result in judge_results
    )
    completed_calls = judge_attempts - transport_failures
    parse_failures = sum(
        result.call_status == "ok" and result.parse_status != "ok"
        for result in judge_results
    )
    analyzable = sum(
        result_is_analyzable(result, config.judge.output_type)
        for result in judge_results
    )
    rejected = sum(
        str(record.get("status")) == "rejected" for record in validation_report
    )
    unavailable = sum(
        str(record.get("status")) == "unavailable" for record in validation_report
    )
    semantic_mean = _mean([risk.flip_risk for risk in item_risks if risk.flip_risk is not None])
    conservative_mean = _mean(
        [
            risk.all_attempt_flip_risk
            for risk in item_risks
            if risk.all_attempt_flip_risk is not None
        ]
    )
    summary = application_summary or {}
    coverage = {
        "source_items_sampled": source_item_count,
        # These are total calls over baseline and perturbations.
        "judge_calls_attempted": judge_attempts,
        "judge_calls_completed": completed_calls,
        "transport_failures": transport_failures,
        "parse_failures": parse_failures,
        "analyzable_verdicts": analyzable,
        "completion_rate": _rate(completed_calls, judge_attempts),
        "parse_rate": _rate(analyzable, completed_calls),
        "analyzable_rate": _rate(analyzable, judge_attempts),
        "semantic_mean_invariant_risk": semantic_mean,
        "conservative_all_attempt_mean_invariant_risk": conservative_mean,
    }
    if summary:
        coverage.update(
            {
                "generation_units": summary["generation_units"],
                "generation_statuses": summary["generation_statuses"],
                "operator_applications_planned": summary["applications"],
                "application_statuses": summary["application_statuses"],
                "variants_generated": summary["unique_variants"],
                "variants_rejected_by_validation": summary["rejected_variants"],
                "variants_validation_unavailable": summary.get("unavailable_variants", unavailable),
                "variants_accepted": summary["accepted_variants"],
                "perturbation_judge_calls_expected": summary["expected_judge_calls"],
                "perturbation_judge_calls_attempted": summary["attempted_judge_calls"],
                "perturbation_analyzable_verdicts": summary["analyzable_judge_calls"],
            }
        )
    else:
        coverage.update(
            {
                "variants_generated": len(attempted_perturbations),
                "variants_rejected_by_validation": rejected,
                "variants_validation_unavailable": unavailable,
                "variants_accepted": len(accepted_perturbations),
            }
        )
    return coverage


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _mean(values: list[float | None]) -> float | None:
    return sum(values) / len(values) if values else None
