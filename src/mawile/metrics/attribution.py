from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from typing import Any

from mawile.measurement import (
    baselines_by_item,
    invariant_disagrees,
    normalized_score,
)
from mawile.schemas import ExpectedRelation, JudgeConfig, JudgeResult, Perturbation


def summarize_attribution(
    results: list[JudgeResult],
    perturbations: list[Perturbation],
    noise_floor: dict[str, Any],
    judge: JudgeConfig | None = None,
) -> list[dict[str, Any]]:
    family_totals: dict[str, int] = defaultdict(int)
    family_flips: dict[str, int] = defaultdict(int)
    family_shifts: dict[str, list[float]] = defaultdict(list)
    family_signed_shifts: dict[str, list[float]] = defaultdict(list)

    for _item_id, perturbation, outcome, score, baseline_score in _invariant_observations(
        results, perturbations, judge
    ):
        family = perturbation.family.value
        family_totals[family] += 1
        if outcome:
            family_flips[family] += 1
        if score is not None and baseline_score is not None:
            family_shifts[family].append(abs(score - baseline_score))
            family_signed_shifts[family].append(score - baseline_score)

    mean_noise = noise_floor.get("mean_noise_flip_rate")
    summaries: list[dict[str, Any]] = []
    for family, total in family_totals.items():
        flip_rate = family_flips[family] / total if total else 0.0
        shifts = family_shifts[family]
        signed = family_signed_shifts[family]
        summaries.append(
            {
                "family": family,
                "num_results": total,
                "family_flip_rate": flip_rate,
                "family_mean_shift": sum(shifts) / len(shifts) if shifts else 0.0,
                "family_mean_signed_shift": sum(signed) / len(signed) if signed else 0.0,
                "family_excess_over_noise": (
                    max(0.0, flip_rate - mean_noise)
                    if mean_noise is not None
                    else None
                ),
            }
        )

    summaries.append(
        {
            "family": "noise",
            "num_results": sum(1 for result in results if result.variant_id == "original"),
            "family_flip_rate": mean_noise,
            "family_mean_shift": noise_floor.get("mean_noise_score_sd", 0.0),
            "family_mean_signed_shift": 0.0,
            "family_excess_over_noise": None,
        }
    )
    return sorted(
        summaries,
        key=lambda row: row["family_flip_rate"] if row["family_flip_rate"] is not None else -1.0,
        reverse=True,
    )


def summarize_perturbation_attribution(
    results: list[JudgeResult],
    perturbations: list[Perturbation],
    noise_floor: dict[str, Any],
    judge: JudgeConfig | None = None,
) -> list[dict[str, Any]]:
    """Summarize flip/shift metrics at the perturbation-operator level."""

    type_totals: dict[tuple[str, str, str], int] = defaultdict(int)
    type_flips: dict[tuple[str, str, str], int] = defaultdict(int)
    type_shifts: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    type_signed_shifts: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    type_variants: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    type_items: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    type_changed_fields: dict[tuple[str, str, str], set[str]] = defaultdict(set)

    for item_id, perturbation, outcome, score, baseline_score in _invariant_observations(
        results, perturbations, judge
    ):
        key = _perturbation_key(perturbation)
        type_totals[key] += 1
        type_variants[key].add(perturbation.variant_id)
        type_items[key].add(item_id)
        type_changed_fields[key].update(perturbation.changed_fields)
        if outcome:
            type_flips[key] += 1
        if score is not None and baseline_score is not None:
            type_shifts[key].append(abs(score - baseline_score))
            type_signed_shifts[key].append(score - baseline_score)

    summaries: list[dict[str, Any]] = []
    for key, total in type_totals.items():
        family, operator, dimension = key
        flip_rate = type_flips[key] / total if total else 0.0
        shifts = type_shifts[key]
        signed = type_signed_shifts[key]
        summaries.append(
            {
                "family": family,
                "operator": operator,
                "dimension": dimension,
                "num_results": total,
                "num_variants": len(type_variants[key]),
                "num_items": len(type_items[key]),
                "changed_fields": sorted(type_changed_fields[key]),
                "flip_rate": flip_rate,
                "mean_shift": sum(shifts) / len(shifts) if shifts else 0.0,
                "mean_signed_shift": sum(signed) / len(signed) if signed else 0.0,
                "excess_over_noise": max(
                    0.0,
                    flip_rate - float(noise_floor.get("mean_noise_flip_rate") or 0.0),
                ),
            }
        )
    return sorted(
        summaries,
        key=lambda row: (row["flip_rate"], abs(row["mean_signed_shift"]), row["operator"]),
        reverse=True,
    )


def _invariant_observations(
    results: list[JudgeResult],
    perturbations: list[Perturbation],
    judge: JudgeConfig | None,
) -> Iterator[tuple[str, Perturbation, bool, float | None, float | None]]:
    """Yield the existing pooled invariant observations for both rollups."""

    perturbation_by_id = {perturbation.variant_id: perturbation for perturbation in perturbations}
    baselines = baselines_by_item(results)
    for result in results:
        if result.variant_id == "original":
            continue
        perturbation = perturbation_by_id[result.variant_id]
        if perturbation.expected_relation != ExpectedRelation.INVARIANT:
            continue
        baseline = baselines.get(result.item_id)
        baseline_label = baseline.label if baseline is not None else None
        baseline_score = baseline.score if baseline is not None else None
        score = normalized_score(result, perturbation)
        outcome = invariant_disagrees(
            result,
            baseline_label=baseline_label,
            baseline_score=baseline_score,
            judge=judge,
            score=score,
        )
        if outcome is not None:
            yield result.item_id, perturbation, outcome, score, baseline_score


def _perturbation_key(perturbation: Perturbation) -> tuple[str, str, str]:
    return (
        perturbation.family.value,
        perturbation.operator,
        str(perturbation.metadata.get("dimension", "")),
    )
