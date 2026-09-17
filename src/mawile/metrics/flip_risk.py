from __future__ import annotations

from collections import defaultdict
from typing import Any

from mawile.measurement import baselines_by_item, invariant_disagrees, normalized_score
from mawile.metrics.routing import rank_item_risks
from mawile.schemas import (
    AuditRunConfig,
    ExpectedRelation,
    Item,
    ItemRisk,
    JudgeResult,
    Perturbation,
    OutputType,
)


def compute_item_risks(
    items: list[Item],
    results: list[JudgeResult],
    perturbations: list[Perturbation],
    config: AuditRunConfig,
    noise_floor: dict[str, Any],
) -> list[ItemRisk]:
    perturbation_by_id = {perturbation.variant_id: perturbation for perturbation in perturbations}
    variants_by_item: dict[str, list[JudgeResult]] = defaultdict(list)

    for result in results:
        if result.variant_id != "original":
            variants_by_item[result.item_id].append(result)
    baselines = baselines_by_item(results)

    risks: list[ItemRisk] = []
    for item in items:
        baseline = baselines.get(item.item_id)
        original_label = baseline.label if baseline is not None else None
        original_score = baseline.score if baseline is not None else None

        invariant_results = [
            result
            for result in variants_by_item[item.item_id]
            if result.variant_id in perturbation_by_id
            and perturbation_by_id[result.variant_id].expected_relation
            == ExpectedRelation.INVARIANT
        ]
        outcomes = [
            invariant_disagrees(
                result,
                baseline_label=original_label,
                baseline_score=original_score,
                judge=config.judge,
                score=normalized_score(result, perturbation_by_id[result.variant_id]),
            )
            for result in invariant_results
        ]
        analyzable_outcomes = [outcome for outcome in outcomes if outcome is not None]
        flips = sum(1 for outcome in analyzable_outcomes if outcome)
        missing = len(outcomes) - len(analyzable_outcomes)
        # A zero rate is evidence of stability only when at least one comparison
        # succeeded.  No probes or all failed calls are unavailable, not stable.
        flip_risk = flips / len(analyzable_outcomes) if analyzable_outcomes else None
        all_attempt_flip_risk = (
            (flips + missing) / len(outcomes) if analyzable_outcomes else None
        )
        noise_risk = noise_floor["items"].get(item.item_id, {}).get("noise_flip_rate")
        family_risks = _compute_family_risks(
            invariant_results,
            perturbation_by_id,
            original_label,
            original_score,
            config,
        )
        family_signed_shifts = _compute_family_signed_shifts(
            invariant_results,
            perturbation_by_id,
            original_score,
            config,
        )
        boundary_proximity = _boundary_proximity(
            original_score,
            threshold=config.judge.threshold,
            width=config.audit.scalar_delta * 2,
        )

        original_verdict: Any = original_label
        if config.judge.output_type in {OutputType.SCALAR, OutputType.ORDINAL}:
            original_verdict = original_label if original_label is not None else original_score
        judge_error = _judge_error(
            original_label,
            original_score,
            item.gold_label,
            config,
        )
        risks.append(
            ItemRisk(
                item_id=item.item_id,
                original_verdict=original_verdict,
                original_score=original_score,
                gold_label=item.gold_label,
                noise_risk=noise_risk,
                flip_risk=flip_risk,
                all_attempt_flip_risk=all_attempt_flip_risk,
                excess_over_noise=(
                    max(0.0, flip_risk - noise_risk)
                    if flip_risk is not None and noise_risk is not None
                    else None
                ),
                invariant_attempts=len(outcomes),
                invariant_analyzable=len(analyzable_outcomes),
                boundary_proximity=boundary_proximity,
                family_risks=family_risks,
                family_signed_shifts=family_signed_shifts,
                judge_correct=None if judge_error is None else not judge_error,
                judge_error=judge_error,
                gold_error=judge_error,
            )
        )

    return rank_item_risks(risks)


def _compute_family_risks(
    invariant_results: list[JudgeResult],
    perturbation_by_id: dict[str, Perturbation],
    original_label: str | None,
    original_score: float | None,
    config: AuditRunConfig,
) -> dict[str, float]:
    family_totals: dict[str, int] = defaultdict(int)
    family_flips: dict[str, int] = defaultdict(int)

    for result in invariant_results:
        perturbation = perturbation_by_id[result.variant_id]
        outcome = invariant_disagrees(
            result,
            baseline_label=original_label,
            baseline_score=original_score,
            judge=config.judge,
            score=normalized_score(result, perturbation),
        )
        if outcome is None:
            continue
        family = perturbation.family.value
        family_totals[family] += 1
        if outcome:
            family_flips[family] += 1

    return {
        family: family_flips[family] / total
        for family, total in family_totals.items()
        if total
    }


def _compute_family_signed_shifts(
    invariant_results: list[JudgeResult],
    perturbation_by_id: dict[str, Perturbation],
    original_score: float | None,
    config: AuditRunConfig,
) -> dict[str, float]:
    if original_score is None:
        return {}
    family_shifts: dict[str, list[float]] = defaultdict(list)
    for result in invariant_results:
        perturbation = perturbation_by_id[result.variant_id]
        score = normalized_score(result, perturbation)
        if score is None:
            continue
        family = perturbation.family.value
        family_shifts[family].append(score - original_score)
    return {
        family: sum(shifts) / len(shifts)
        for family, shifts in family_shifts.items()
        if shifts
    }


def _boundary_proximity(
    score: float | None,
    threshold: float | None,
    width: float,
) -> float:
    if score is None or threshold is None:
        return 0.0
    distance = abs(score - threshold)
    return max(0.0, 1.0 - min(distance / max(width, 1e-9), 1.0))


def _judge_error(
    original_label: str | None,
    original_score: float | None,
    gold_label: Any | None,
    config: AuditRunConfig,
) -> bool | None:
    if gold_label is None:
        return None
    if config.judge.output_type in {OutputType.SCALAR, OutputType.ORDINAL}:
        if original_score is None or config.judge.gold_tolerance is None:
            return None
        try:
            return abs(original_score - float(gold_label)) > config.judge.gold_tolerance
        except (TypeError, ValueError):
            return None
    if original_label is None:
        return None
    return str(original_label).casefold() != str(gold_label).casefold()
