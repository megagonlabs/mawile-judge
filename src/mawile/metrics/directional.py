from __future__ import annotations

from collections import defaultdict
from typing import Any

from mawile.measurement import (
    baselines_by_item,
    directional_baseline_has_room,
    quality_delta,
    summarize_verdicts,
)
from mawile.schemas import (
    AuditRunConfig,
    JudgeResult,
    Perturbation,
    PerturbationExpectedEffect,
)


def case_detected(case: dict[str, Any]) -> bool | None:
    """Whether the degradation registered, on whichever basis the case carries."""

    if "detected" in case:
        return case["detected"]
    return case.get("score_detected")


def case_contradicted(case: dict[str, Any]) -> bool | None:
    if "detected" in case:
        return case.get("contradiction")
    return case.get("contradiction") if case.get("score_delta") is not None else None


def compute_directional_metrics(
    results: list[JudgeResult],
    perturbations: list[Perturbation],
    noise_floor: dict[str, Any],
    config: AuditRunConfig,
) -> dict[str, Any]:
    """Measure whether intentional degradations move a judge's verdict downward.

    Numeric judges compare quality-normalized score shifts; judges that emit
    canonical classes compare movement between those classes. Both bases feed
    the same detection and contradiction rates, so the headline is comparable
    across output shapes; score-only quantities stay separate.
    """

    directional_by_id = {
        perturbation.variant_id: perturbation
        for perturbation in perturbations
        if perturbation.expected_effect == PerturbationExpectedEffect.WORSE_VERDICT
    }
    if not directional_by_id:
        return _empty_summary()

    baselines = baselines_by_item(results)
    variant_results: dict[tuple[str, str], list[JudgeResult]] = defaultdict(list)
    for result in results:
        if result.variant_id in directional_by_id:
            variant_results[(result.item_id, result.variant_id)].append(result)

    cases: list[dict[str, Any]] = []
    for (item_id, variant_id), case_results in sorted(variant_results.items()):
        baseline_summary = baselines.get(item_id)
        baseline = (
            (baseline_summary.label, baseline_summary.score)
            if baseline_summary is not None
            else (None, None)
        )
        eligible, _ = directional_baseline_has_room(*baseline, config.judge)
        if not eligible:
            continue
        cases.append(
            _case_summary(
                item_id=item_id,
                perturbation=directional_by_id[variant_id],
                results=case_results,
                baseline=baseline,
                noise_floor=noise_floor,
                config=config,
            )
        )
    if not cases:
        return _empty_summary()
    score_cases = [case for case in cases if case["score_delta"] is not None]
    label_cases = [case for case in cases if case["detection_basis"] == "label"]
    analyzable_cases = [case for case in cases if case["detected"] is not None]
    threshold_cases = [
        case for case in cases if case["threshold_worsened"] is not None
    ]
    label_movement = _label_movement_summary(cases)

    return {
        "has_directional": True,
        "expected_effect": PerturbationExpectedEffect.WORSE_VERDICT.value,
        "score_supported": bool(score_cases),
        "detection_basis": _summary_detection_basis(score_cases, label_cases),
        "num_cases": len(cases),
        "num_results": sum(case["num_results"] for case in cases),
        "num_score_cases": len(score_cases),
        "num_label_cases": len(label_cases),
        "num_analyzable_cases": len(analyzable_cases),
        "degradation_detection_rate": _rate(
            sum(1 for case in analyzable_cases if case["detected"]),
            len(analyzable_cases),
        ),
        "noise_adjusted_detection_rate": _noise_adjusted_detection_rate(
            analyzable_cases
        ),
        "threshold_worsening_rate": _rate(
            sum(1 for case in threshold_cases if case["threshold_worsened"]),
            len(threshold_cases),
        ),
        "contradiction_rate": _rate(
            sum(1 for case in analyzable_cases if case["contradiction"]),
            len(analyzable_cases),
        ),
        "mean_score_delta": _mean(
            [case["score_delta"] for case in score_cases if case["score_delta"] is not None]
        ),
        "mean_quality_delta": _mean(
            [case["quality_delta"] for case in score_cases if case["quality_delta"] is not None]
        ),
        "score_direction": config.judge.score_direction.value,
        "label_movement": label_movement,
        "operator_summaries": _operator_summaries(cases),
        "top_missed_degradations": _top_missed_degradations(cases),
        "cases": cases,
    }


def _summary_detection_basis(
    score_cases: list[dict[str, Any]],
    label_cases: list[dict[str, Any]],
) -> str | None:
    if score_cases and label_cases:
        return "mixed"
    if score_cases:
        return "score"
    if label_cases:
        return "label"
    return None


def _empty_summary() -> dict[str, Any]:
    return {
        "has_directional": False,
        "expected_effect": PerturbationExpectedEffect.WORSE_VERDICT.value,
        "score_supported": False,
        "detection_basis": None,
        "num_cases": 0,
        "num_results": 0,
        "num_score_cases": 0,
        "num_label_cases": 0,
        "num_analyzable_cases": 0,
        "degradation_detection_rate": None,
        "noise_adjusted_detection_rate": None,
        "threshold_worsening_rate": None,
        "contradiction_rate": None,
        "mean_score_delta": None,
        "mean_quality_delta": None,
        "score_direction": None,
        "label_movement": {
            "total": 0,
            "worsened": 0,
            "unchanged": 0,
            "improved": 0,
            "unknown": 0,
            "worsening_rate": None,
        },
        "operator_summaries": [],
        "top_missed_degradations": [],
        "cases": [],
    }


def _case_summary(
    *,
    item_id: str,
    perturbation: Perturbation,
    results: list[JudgeResult],
    baseline: tuple[str | None, float | None],
    noise_floor: dict[str, Any],
    config: AuditRunConfig,
) -> dict[str, Any]:
    baseline_label, baseline_score = baseline
    perturbed = summarize_verdicts(results, perturbation)
    perturbed_label = perturbed.label
    perturbed_score = perturbed.score
    score_delta = (
        perturbed_score - baseline_score
        if perturbed_score is not None and baseline_score is not None
        else None
    )
    semantic_delta = (
        quality_delta(score_delta, config.judge) if score_delta is not None else None
    )
    item_noise = noise_floor.get("items", {}).get(item_id, {})
    raw_noise_sd = item_noise.get("noise_score_sd", 0.0)
    raw_noise_flip_rate = item_noise.get("noise_flip_rate", 0.0)
    noise_sd = float(raw_noise_sd) if raw_noise_sd is not None else None
    noise_flip_rate = float(raw_noise_flip_rate) if raw_noise_flip_rate is not None else None
    threshold_worsened = _threshold_worsened(
        baseline_score=baseline_score,
        perturbed_score=perturbed_score,
        threshold=config.judge.threshold,
        higher_is_better=config.judge.score_direction.value == "higher_is_better",
    )
    movement = label_movement(baseline_label, perturbed_label)
    basis, detected, contradiction = _detection(semantic_delta, movement)
    noise_adjusted_detected = (
        None if semantic_delta is None or noise_sd is None else semantic_delta < -noise_sd
    )
    return {
        "item_id": item_id,
        "variant_id": perturbation.variant_id,
        "operator": perturbation.operator,
        "family": perturbation.family.value,
        "dimension": perturbation.metadata.get("dimension", ""),
        "changed_fields": list(perturbation.changed_fields),
        "num_results": len(results),
        "baseline_label": baseline_label,
        "perturbed_label": perturbed_label,
        "baseline_score": baseline_score,
        "perturbed_score": perturbed_score,
        "score_delta": score_delta,
        "quality_delta": semantic_delta,
        "noise_score_sd": noise_sd,
        "noise_flip_rate": noise_flip_rate,
        "detection_basis": basis,
        "detected": detected,
        "score_detected": None if semantic_delta is None else semantic_delta < 0,
        "noise_adjusted_detected": noise_adjusted_detected,
        "noise_adjusted_detection_credit": _noise_adjusted_detection_credit(
            basis=basis,
            detected=detected,
            noise_adjusted_detected=noise_adjusted_detected,
            noise_flip_rate=noise_flip_rate,
        ),
        "threshold_worsened": threshold_worsened,
        "contradiction": contradiction,
        "label_movement": movement,
    }


def _detection(
    semantic_delta: float | None,
    movement: str,
) -> tuple[str | None, bool | None, bool | None]:
    """Resolve the detection verdict, preferring the finer numeric basis."""

    if semantic_delta is not None:
        return "score", semantic_delta < 0, semantic_delta > 0
    if movement == "unknown":
        return None, None, None
    return "label", movement == "worsened", movement == "improved"


def _threshold_worsened(
    *,
    baseline_score: float | None,
    perturbed_score: float | None,
    threshold: float | None,
    higher_is_better: bool,
) -> bool | None:
    """Whether a declared decision threshold was crossed in the worse direction."""

    if threshold is None or baseline_score is None or perturbed_score is None:
        return None
    if higher_is_better:
        return baseline_score >= threshold and perturbed_score < threshold
    return baseline_score <= threshold and perturbed_score > threshold


def label_movement(baseline_label: str | None, perturbed_label: str | None) -> str:
    """Direction of travel between two canonical classes."""

    if baseline_label is None or perturbed_label is None:
        return "unknown"
    baseline = baseline_label.casefold()
    perturbed = perturbed_label.casefold()
    if baseline == perturbed:
        return "unchanged"
    if baseline == "pass" and perturbed == "fail":
        return "worsened"
    if baseline == "fail" and perturbed == "pass":
        return "improved"
    return "unknown"


def _label_movement_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {"worsened": 0, "unchanged": 0, "improved": 0, "unknown": 0}
    for case in cases:
        counts[case["label_movement"]] += 1
    total = len(cases)
    return {
        "total": total,
        **counts,
        "worsening_rate": _rate(counts["worsened"], total),
    }


def _operator_summaries(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[case["operator"]].append(case)

    rows: list[dict[str, Any]] = []
    for operator, operator_cases in grouped.items():
        score_cases = [case for case in operator_cases if case["score_delta"] is not None]
        analyzable_cases = [case for case in operator_cases if case["detected"] is not None]
        threshold_cases = [
            case for case in operator_cases if case["threshold_worsened"] is not None
        ]
        rows.append(
            {
                "operator": operator,
                "family": operator_cases[0]["family"],
                "dimension": operator_cases[0]["dimension"],
                "changed_fields": sorted(
                    {
                        field
                        for case in operator_cases
                        for field in case["changed_fields"]
                    }
                ),
                "num_cases": len(operator_cases),
                "num_results": sum(case["num_results"] for case in operator_cases),
                "num_analyzable_cases": len(analyzable_cases),
                "degradation_detection_rate": _rate(
                    sum(1 for case in analyzable_cases if case["detected"]),
                    len(analyzable_cases),
                ),
                "noise_adjusted_detection_rate": _noise_adjusted_detection_rate(
                    analyzable_cases
                ),
                "threshold_worsening_rate": _rate(
                    sum(1 for case in threshold_cases if case["threshold_worsened"]),
                    len(threshold_cases),
                ),
                "contradiction_rate": _rate(
                    sum(1 for case in analyzable_cases if case["contradiction"]),
                    len(analyzable_cases),
                ),
                "mean_score_delta": _mean(
                    [
                        case["score_delta"]
                        for case in score_cases
                        if case["score_delta"] is not None
                    ]
                ),
                "mean_quality_delta": _mean(
                    [
                        case["quality_delta"]
                        for case in score_cases
                        if case["quality_delta"] is not None
                    ]
                ),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            row["degradation_detection_rate"] is None,
            -(row["degradation_detection_rate"] or 0.0),
            row["operator"],
        ),
    )


def _top_missed_degradations(cases: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    missed = [case for case in cases if case["detected"] is False]
    return [
        {
            "item_id": case["item_id"],
            "variant_id": case["variant_id"],
            "operator": case["operator"],
            "detection_basis": case["detection_basis"],
            "baseline_label": case["baseline_label"],
            "perturbed_label": case["perturbed_label"],
            "label_movement": case["label_movement"],
            "baseline_score": case["baseline_score"],
            "perturbed_score": case["perturbed_score"],
            "score_delta": case["score_delta"],
            "noise_score_sd": case["noise_score_sd"],
            "noise_flip_rate": case["noise_flip_rate"],
        }
        for case in sorted(missed, key=_missed_sort_key)[:limit]
    ]


def _missed_sort_key(case: dict[str, Any]) -> tuple[Any, ...]:
    """Rank contradictions above flat responses on either detection basis."""

    semantic_delta = case["quality_delta"]
    return (
        case["contradiction"] is not True,
        -(semantic_delta if semantic_delta is not None else 0.0),
        case["detection_basis"] == "label",
        case["operator"],
        case["item_id"],
    )


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _noise_adjusted_detection_credit(
    *,
    basis: str | None,
    detected: bool | None,
    noise_adjusted_detected: bool | None,
    noise_flip_rate: float | None,
) -> float | None:
    """Per-case contribution to detection after the matching noise baseline.

    Scores retain their existing SD comparison. Class movements subtract the
    item's repeated-baseline flip probability; averaging those contributions
    yields detection rate minus mean categorical noise, floored at zero.
    """

    if basis == "score":
        return None if noise_adjusted_detected is None else float(noise_adjusted_detected)
    if basis == "label":
        return (
            None
            if detected is None or noise_flip_rate is None
            else float(detected) - noise_flip_rate
        )
    return None


def _noise_adjusted_detection_rate(cases: list[dict[str, Any]]) -> float | None:
    credits = [
        float(case["noise_adjusted_detection_credit"])
        for case in cases
        if case.get("noise_adjusted_detection_credit") is not None
    ]
    mean_credit = _mean(credits)
    return None if mean_credit is None else max(0.0, mean_credit)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)
