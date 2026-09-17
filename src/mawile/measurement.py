"""Canonical verdict semantics shared by metrics, reports, and artifacts."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from mawile.schemas import (
    Item,
    JudgeConfig,
    JudgeResult,
    OutputType,
    Perturbation,
    ScoreDirection,
)


ResultRecord = JudgeResult | Mapping[str, Any]
JudgeSemantics = JudgeConfig | Mapping[str, Any] | OutputType | str
PerturbationRecord = Perturbation | Mapping[str, Any]
_SCORE_NOT_PROVIDED = object()


@dataclass(frozen=True)
class VerdictSummary:
    """Aggregated verdicts for one item/variant, retaining analyzable inputs."""

    label: str | None
    score: float | None
    valid_labels: tuple[str, ...] = ()
    valid_scores: tuple[float, ...] = ()
    result_count: int = 0

    @property
    def label_count(self) -> int:
        return len(self.valid_labels)

    @property
    def score_count(self) -> int:
        return len(self.valid_scores)


def _field(
    record: ResultRecord | PerturbationRecord | JudgeSemantics,
    name: str,
    default: Any = None,
) -> Any:
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def _output_type(judge: JudgeSemantics | None) -> OutputType | None:
    if isinstance(judge, (OutputType, str)):
        try:
            return OutputType(judge)
        except ValueError:
            return None
    value = _field(judge, "output_type") if judge is not None else None
    try:
        return OutputType(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _invariant_tolerance(judge: JudgeSemantics | None) -> float:
    value = _field(judge, "invariant_tolerance", 0.0) if judge is not None else 0.0
    tolerance = coerce_float(value)
    return tolerance if tolerance is not None else 0.0


def _is_successful_result(result: ResultRecord) -> bool:
    """Saved pre-status records are legacy successes; explicit failures are not."""

    return _field(result, "call_status", "ok") == "ok" and _field(
        result, "parse_status", "ok"
    ) == "ok"


def validate_items_for_judge(items: list[Item], judge: JudgeConfig) -> None:
    """Reject shape mismatches before perturbations or model calls begin."""

    errors: list[str] = []
    for item in items:
        if judge.output_type == OutputType.PAIRWISE:
            if not item.is_pairwise:
                errors.append(f"{item.item_id}: pairwise judge requires candidates and position mapping")
        elif item.is_pairwise:
            errors.append(f"{item.item_id}: {judge.output_type.value} judge cannot grade pairwise candidates")
        elif item.output is None:
            errors.append(f"{item.item_id}: pointwise judge requires output")
    if errors:
        preview = "; ".join(errors[:5])
        suffix = f"; plus {len(errors) - 5} more" if len(errors) > 5 else ""
        raise ValueError(f"Item/judge shape mismatch: {preview}{suffix}")


def threshold_label(score: float, judge: JudgeConfig) -> str | None:
    if judge.threshold is None:
        return None
    if judge.score_direction == ScoreDirection.HIGHER_IS_BETTER:
        return "pass" if score >= judge.threshold else "fail"
    return "pass" if score <= judge.threshold else "fail"


def quality_delta(raw_delta: float, judge: JudgeConfig) -> float:
    """Convert a raw score delta to positive=better and negative=worse."""

    if judge.score_direction == ScoreDirection.HIGHER_IS_BETTER:
        return raw_delta
    return -raw_delta


def result_label(result: ResultRecord) -> str | None:
    """Return canonical semantic identity, never a stringified missing value."""

    if not _is_successful_result(result):
        return None
    verdict = _field(result, "parsed_verdict")
    if isinstance(verdict, Mapping):
        value = verdict.get(
            "candidate_id",
            verdict.get(
                "label",
                verdict.get("preference", verdict.get("winner", verdict.get("verdict"))),
            ),
        )
    else:
        value = verdict
    return None if value is None else str(value)


def result_position(result: ResultRecord) -> str | None:
    if not _is_successful_result(result):
        return None
    verdict = _field(result, "parsed_verdict")
    if not isinstance(verdict, Mapping):
        return None
    value = verdict.get("position")
    if value is None and verdict.get("candidate_id") is not None:
        value = verdict.get("label")
    return str(value) if value in {"A", "B"} else None


def result_score(result: ResultRecord) -> float | None:
    if not _is_successful_result(result):
        return None
    verdict = _field(result, "parsed_verdict")
    if not isinstance(verdict, Mapping):
        return None
    return coerce_float(verdict.get("score"))


def majority_label(labels: Sequence[str | None]) -> str | None:
    """First-observed majority label, ignoring unavailable verdicts."""

    analyzable = [label for label in labels if label is not None]
    return Counter(analyzable).most_common(1)[0][0] if analyzable else None


def mean_score(scores: Sequence[float]) -> float | None:
    """Mean of already-analyzable scores (empty means unavailable)."""

    return sum(scores) / len(scores) if scores else None


def _metadata(perturbation: PerturbationRecord) -> Mapping[str, Any]:
    metadata = _field(perturbation, "metadata", {})
    return metadata if isinstance(metadata, Mapping) else {}


def rescale_to_base(score: float, metadata: Mapping[str, Any]) -> float:
    """Invert the recorded 1..old_max -> 1..new_max affine rescale."""

    old_max = metadata.get("old_max")
    new_max = metadata.get("new_max")
    if not old_max or not new_max:
        return score
    old_max_value = coerce_float(old_max)
    new_max_value = coerce_float(new_max)
    if old_max_value is None or new_max_value is None or new_max_value <= 1:
        return score
    return 1 + (score - 1) * (old_max_value - 1) / (new_max_value - 1)


def normalized_score(
    result: ResultRecord, perturbation: PerturbationRecord | None = None
) -> float | None:
    """Return a finite parsed score on the original rubric scale."""

    score = result_score(result)
    if score is None or perturbation is None:
        return score
    normalized = rescale_to_base(score, _metadata(perturbation))
    return normalized if math.isfinite(normalized) else None


def summarize_verdicts(
    results: Sequence[ResultRecord],
    perturbation: PerturbationRecord | None = None,
) -> VerdictSummary:
    """Summarize labels and (optionally normalized) scores for a result group."""

    labels = tuple(
        label for result in results if (label := result_label(result)) is not None
    )
    scores = tuple(
        score
        for result in results
        if (score := normalized_score(result, perturbation)) is not None
    )
    return VerdictSummary(
        label=majority_label(labels),
        score=mean_score(scores),
        valid_labels=labels,
        valid_scores=scores,
        result_count=len(results),
    )


def baselines_by_item(results: Sequence[ResultRecord]) -> dict[str, VerdictSummary]:
    """Return original-variant verdict summaries keyed by stable item ID."""

    originals: dict[str, list[ResultRecord]] = defaultdict(list)
    for result in results:
        if _field(result, "variant_id") == "original":
            item_id = _field(result, "item_id")
            if item_id is not None:
                originals[str(item_id)].append(result)
    return {
        item_id: summarize_verdicts(item_results)
        for item_id, item_results in originals.items()
    }


def directional_baseline_eligibility(
    items: list[Item],
    results: Sequence[ResultRecord],
    judge: JudgeConfig,
) -> dict[str, Any]:
    """Resolve which items have room for a worse directional verdict.

    Eligibility is intentionally based on the judge's repeated original verdicts,
    not on gold labels. Binary items whose majority baseline is ``fail`` cannot
    move to a worse class. Ordinal items at the worst configured score boundary,
    within ``invariant_tolerance``, cannot move by a meaningful amount either.
    Other output shapes are left unfiltered.
    """

    baselines = baselines_by_item(results)

    ordinal_floor = _ordinal_worst_score(judge)
    filter_applied = judge.output_type == OutputType.BINARY or (
        judge.output_type == OutputType.ORDINAL and ordinal_floor is not None
    )
    decisions: list[dict[str, Any]] = []
    for item in items:
        baseline = baselines.get(item.item_id)
        baseline_label = baseline.label if baseline is not None else None
        baseline_score = baseline.score if baseline is not None else None
        eligible, reason = directional_baseline_has_room(
            baseline_label,
            baseline_score,
            judge,
        )

        decisions.append(
            {
                "item_id": item.item_id,
                "eligible": eligible,
                "reason": reason,
                "baseline_label": baseline_label,
                "baseline_score": baseline_score,
            }
        )

    eligible = sum(decision["eligible"] for decision in decisions)
    return {
        "policy": "judge_baseline",
        "output_type": judge.output_type.value,
        "filter_applied": filter_applied,
        "floor_score": ordinal_floor,
        "floor_tolerance": (
            judge.invariant_tolerance
            if judge.output_type == OutputType.ORDINAL and ordinal_floor is not None
            else None
        ),
        "eligible_item_count": eligible,
        "ineligible_item_count": len(decisions) - eligible,
        "items": decisions,
    }


def directional_baseline_has_room(
    baseline_label: str | None,
    baseline_score: float | None,
    judge: JudgeConfig,
) -> tuple[bool, str | None]:
    """Whether a baseline can move meaningfully in the worse direction."""

    if judge.output_type == OutputType.BINARY:
        if baseline_label is None:
            return False, "baseline_unavailable"
        if baseline_label.casefold() == "fail":
            return False, "baseline_fail"
    elif judge.output_type == OutputType.ORDINAL:
        ordinal_floor = _ordinal_worst_score(judge)
        if ordinal_floor is not None:
            if baseline_score is None:
                return False, "baseline_unavailable"
            if _within_ordinal_floor_tolerance(baseline_score, ordinal_floor, judge):
                return False, "baseline_within_floor_tolerance"
    return True, None


def _ordinal_worst_score(judge: JudgeConfig) -> float | None:
    if judge.output_type != OutputType.ORDINAL:
        return None
    if judge.score_direction == ScoreDirection.HIGHER_IS_BETTER:
        return judge.score_min
    return judge.score_max


def _within_ordinal_floor_tolerance(
    baseline_score: float,
    floor_score: float,
    judge: JudgeConfig,
) -> bool:
    if judge.score_direction == ScoreDirection.HIGHER_IS_BETTER:
        return baseline_score <= floor_score + judge.invariant_tolerance
    return baseline_score >= floor_score - judge.invariant_tolerance


def result_is_analyzable(
    result: ResultRecord,
    output_type: JudgeSemantics | None = None,
    *,
    judge: JudgeSemantics | None = None,
) -> bool:
    """Whether a successful result has the declared semantic value.

    A legacy saved record without ``output_type`` is only known to be a parsed
    verdict; it is deliberately not inferred to be a scalar from its shape.
    """

    if not _is_successful_result(result):
        return False
    semantics = judge if judge is not None else output_type
    declared_output_type = _output_type(semantics)
    if declared_output_type in {OutputType.SCALAR, OutputType.ORDINAL}:
        return result_score(result) is not None
    if declared_output_type in {OutputType.BINARY, OutputType.PAIRWISE}:
        return result_label(result) is not None
    return _field(result, "parsed_verdict") is not None


def invariant_disagrees(
    result: ResultRecord,
    *,
    baseline_label: str | None,
    baseline_score: float | None,
    judge: JudgeSemantics | None,
    score: float | None | object = _SCORE_NOT_PROVIDED,
) -> bool | None:
    """Evaluate one invariant result; ``None`` means it is not analyzable."""

    if not _is_successful_result(result):
        return None
    output_type = _output_type(judge)
    if output_type in {OutputType.SCALAR, OutputType.ORDINAL}:
        observed = (
            result_score(result)
            if score is _SCORE_NOT_PROVIDED
            else coerce_float(score)
        )
        if observed is None or baseline_score is None:
            return None
        return abs(observed - baseline_score) > _invariant_tolerance(judge)

    observed_label = result_label(result)
    if observed_label is None or baseline_label is None:
        return None
    return observed_label != baseline_label


def coerce_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    except (TypeError, ValueError):
        return None
