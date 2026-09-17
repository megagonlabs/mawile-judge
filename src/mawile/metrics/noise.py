from __future__ import annotations

from statistics import pstdev
from typing import Any

from mawile.measurement import baselines_by_item
from mawile.schemas import JudgeConfig, JudgeResult, OutputType


def compute_noise_floor(
    results: list[JudgeResult],
    judge: JudgeConfig | None = None,
) -> dict[str, Any]:
    item_metrics: dict[str, dict[str, Any]] = {}
    for item_id, summary in baselines_by_item(results).items():
        labels = summary.valid_labels
        scores = summary.valid_scores
        if judge is not None and judge.output_type in {OutputType.SCALAR, OutputType.ORDINAL}:
            center = summary.score
            disagreements = (
                sum(abs(score - center) > judge.invariant_tolerance for score in scores)
                if center is not None
                else 0
            )
            analyzable = len(scores)
        else:
            majority = summary.label
            analyzable_labels = labels
            disagreements = (
                sum(label != majority for label in analyzable_labels)
                if majority is not None
                else 0
            )
            analyzable = len(analyzable_labels)
        missing = summary.result_count - analyzable
        # Missing or failed repeats do not establish a zero noise floor.
        noise_flip_rate = disagreements / analyzable if analyzable else None
        item_metrics[item_id] = {
            "noise_flip_rate": noise_flip_rate,
            "all_attempt_noise_risk": (
                (disagreements + missing) / summary.result_count
                if summary.result_count
                else 0.0
            ),
            "noise_score_sd": pstdev(scores) if scores else None,
            "attempted": summary.result_count,
            "analyzable": analyzable,
            "missing": missing,
        }

    mean_noise_flip_rate = (
        _mean(metric["noise_flip_rate"] for metric in item_metrics.values())
    )
    mean_noise_score_sd = (
        _mean(metric["noise_score_sd"] for metric in item_metrics.values())
    )

    return {
        "items": item_metrics,
        "mean_noise_flip_rate": mean_noise_flip_rate,
        "mean_noise_score_sd": mean_noise_score_sd,
    }


def _mean(values: Any) -> float | None:
    present = [float(value) for value in values if value is not None]
    return sum(present) / len(present) if present else None
