from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

from mawile.measurement import baselines_by_item
from mawile.metrics.directional import label_movement
from mawile.reporting.comparisons import compare_result
from mawile.reporting.traces import first_field_change, trace_by_result
from mawile.schemas import PerturbationExpectedEffect


@dataclass(frozen=True)
class ImpactExample:
    family: str
    operator: str
    dimension: str
    variant_id: str
    item_id: str
    expected_effect: str
    summary: str
    changed_field: str | None
    before: Any
    after: Any
    baseline_label: str | None
    perturbed_label: str | None
    baseline_score: float | None
    perturbed_score: float | None
    score_delta: float | None
    quality_delta: float | None
    flipped: bool | None
    run_id: str

    @property
    def impact_score(self) -> float:
        return (1.0 if self.flipped else 0.0) + abs(self.score_delta or 0.0)

    @property
    def label_movement(self) -> str:
        return label_movement(self.baseline_label, self.perturbed_label)

    @property
    def detected(self) -> bool | None:
        """Whether this example registered the degradation its operator declares."""

        if self.quality_delta is not None:
            return self.quality_delta < 0
        movement = self.label_movement
        return None if movement == "unknown" else movement == "worsened"

    @property
    def contradicted(self) -> bool | None:
        if self.quality_delta is not None:
            return self.quality_delta > 0
        movement = self.label_movement
        return None if movement == "unknown" else movement == "improved"


@dataclass(frozen=True)
class PerturbationTypeImpact:
    family: str
    operator: str
    dimension: str
    changed_fields: tuple[str, ...]
    num_results: int
    num_variants: int
    num_items: int
    flip_rate: float | None
    mean_shift: float | None
    mean_signed_shift: float | None
    excess_over_noise: float | None
    examples: tuple[ImpactExample, ...]


def summarize_impact_examples(
    artifact: dict[str, Any],
    judge_call_traces: list[dict[str, Any]] | None = None,
    *,
    max_examples_per_operator: int = 1,
) -> list[ImpactExample]:
    grouped: dict[tuple[str, str], list[ImpactExample]] = defaultdict(list)
    for example in _dedupe_impact_examples(
        _impact_examples_by_result(artifact, judge_call_traces)
    ).values():
        grouped[(example.family, example.operator)].append(example)

    selected: list[ImpactExample] = []
    for examples in grouped.values():
        selected.extend(
            sorted(examples, key=_impact_example_sort_key, reverse=True)[
                :max_examples_per_operator
            ]
        )
    return sorted(
        selected,
        key=lambda example: (example.family, example.operator, -example.impact_score),
    )


def summarize_perturbation_type_impacts(
    artifact: dict[str, Any],
    judge_call_traces: list[dict[str, Any]] | None = None,
) -> list[PerturbationTypeImpact]:
    examples = list(
        _dedupe_impact_examples(
            _impact_examples_by_result(artifact, judge_call_traces)
        ).values()
    )
    grouped: dict[tuple[str, str, str], list[ImpactExample]] = defaultdict(list)
    for example in examples:
        if not _is_invariant_expected_effect(example.expected_effect):
            continue
        grouped[(example.family, example.operator, example.dimension)].append(example)

    stats = _perturbation_type_stats(artifact)
    summaries: list[PerturbationTypeImpact] = []
    for key, type_examples in grouped.items():
        family, operator, dimension = key
        stat = stats.get(key)
        sorted_examples = tuple(sorted(type_examples, key=_impact_example_sort_key, reverse=True))
        changed_fields = (
            tuple(str(field) for field in stat.get("changed_fields", []))
            if stat is not None
            else tuple(
                sorted(
                    {
                        str(example.changed_field)
                        for example in sorted_examples
                        if example.changed_field
                    }
                )
            )
        )
        stat_flip_rate = stat.get("flip_rate") if stat is not None else None
        summaries.append(
            PerturbationTypeImpact(
                family=family,
                operator=operator,
                dimension=dimension,
                changed_fields=changed_fields,
                num_results=int(stat.get("num_results", 0)) if stat else len(sorted_examples),
                num_variants=int(stat.get("num_variants", 0)) if stat else len(sorted_examples),
                num_items=int(stat.get("num_items", 0))
                if stat
                else len({example.item_id for example in sorted_examples}),
                flip_rate=float(stat_flip_rate) if stat_flip_rate is not None else None,
                mean_shift=_optional_stat_float(stat, "mean_shift", None),
                mean_signed_shift=_optional_stat_float(stat, "mean_signed_shift", None),
                excess_over_noise=_optional_stat_float(stat, "excess_over_noise", None),
                examples=sorted_examples,
            )
        )
    return sorted(
        summaries,
        key=lambda summary: (
            summary.flip_rate if summary.flip_rate is not None else -1.0,
            abs(summary.mean_signed_shift or 0.0),
            summary.operator,
        ),
        reverse=True,
    )


def directional_examples_by_operator(
    artifact: dict[str, Any],
    judge_call_traces: list[dict[str, Any]] | None = None,
) -> dict[str, tuple[ImpactExample, ...]]:
    grouped: dict[str, list[ImpactExample]] = defaultdict(list)
    for example in _dedupe_impact_examples(
        _impact_examples_by_result(artifact, judge_call_traces)
    ).values():
        if _is_invariant_expected_effect(example.expected_effect):
            continue
        grouped[example.operator].append(example)
    return {
        operator: tuple(
            sorted(examples, key=_directional_example_sort_key, reverse=True)
        )
        for operator, examples in grouped.items()
    }


_MOVEMENT_SEVERITY = {"improved": 2, "unchanged": 1, "worsened": 0}


def _directional_example_sort_key(
    example: ImpactExample,
) -> tuple[bool, bool, float, int, str]:
    # Worst behaviour first: contradictions (positive delta or an improved
    # class), then missed degradations, then correctly detected cases.
    return (
        example.contradicted is True,
        example.detected is False,
        example.quality_delta if example.quality_delta is not None else 0.0,
        _MOVEMENT_SEVERITY.get(example.label_movement, 0),
        example.variant_id,
    )


def impact_trial_options(
    artifact: dict[str, Any],
    judge_call_traces: list[dict[str, Any]] | None,
    examples: Sequence[ImpactExample],
) -> list[tuple[ImpactExample, ...]]:
    keys = [(example.item_id, example.variant_id) for example in examples]
    requested = set(keys)
    grouped: dict[tuple[str, str], list[ImpactExample]] = defaultdict(list)
    for trial in _impact_examples_by_result(artifact, judge_call_traces):
        key = (trial.item_id, trial.variant_id)
        if key in requested:
            grouped[key].append(trial)

    return [
        tuple(sorted(grouped.get(key) or [example], key=_impact_trial_sort_key))
        for key, example in zip(keys, examples)
    ]


def _impact_examples_by_result(
    artifact: dict[str, Any],
    judge_call_traces: list[dict[str, Any]] | None = None,
) -> list[ImpactExample]:
    perturbations = {
        record["variant_id"]: record
        for record in artifact.get("perturbations", [])
    }
    baselines = baselines_by_item(artifact.get("judge_results", []))
    traces = trace_by_result(judge_call_traces or [])
    examples: list[ImpactExample] = []
    score_direction = str(
        (artifact.get("directional") or {}).get("score_direction")
        or (artifact.get("output_semantics") or {}).get("score_direction")
        or "higher_is_better"
    )

    for result in artifact.get("judge_results", []):
        variant_id = result.get("variant_id")
        if variant_id == "original" or variant_id not in perturbations:
            continue
        perturbation = perturbations[variant_id]
        baseline = baselines.get(str(result.get("item_id")))
        if baseline is None:
            continue
        trace = traces.get(
            (
                str(result.get("item_id")),
                str(variant_id),
                str(result.get("run_id")),
            )
        ) or traces.get((str(result.get("item_id")), str(variant_id), "*"))
        changed_field, before, after = first_field_change(trace)
        comparison = compare_result(
            result,
            baseline,
            perturbation=perturbation,
            output_semantics=artifact.get("output_semantics"),
            invariant=str(perturbation.get("expected_effect")) == "same_verdict",
        )
        score_delta = comparison["score_delta"]
        quality_delta = -score_delta if score_delta is not None and score_direction == "lower_is_better" else score_delta
        examples.append(
            ImpactExample(
                family=str(perturbation.get("family", "")),
                operator=str(perturbation.get("operator", "")),
                dimension=str(perturbation.get("metadata", {}).get("dimension", "")),
                variant_id=str(variant_id),
                item_id=str(result.get("item_id")),
                expected_effect=str(perturbation.get("expected_effect", "")),
                summary=str(perturbation.get("metadata", {}).get("summary", "")),
                changed_field=changed_field,
                before=before,
                after=after,
                baseline_label=comparison["baseline_label"],
                perturbed_label=comparison["perturbed_label"],
                baseline_score=comparison["baseline_score"],
                perturbed_score=comparison["perturbed_score"],
                score_delta=score_delta,
                quality_delta=quality_delta,
                flipped=comparison["flipped"],
                run_id=str(result.get("run_id")),
            )
        )
    return examples


def _dedupe_impact_examples(examples: list[ImpactExample]) -> dict[tuple[str, str], ImpactExample]:
    deduped: dict[tuple[str, str], ImpactExample] = {}
    for example in examples:
        key = (example.item_id, example.variant_id)
        existing = deduped.get(key)
        if existing is None or _impact_example_sort_key(example) > _impact_example_sort_key(existing):
            deduped[key] = example
    return deduped


def _impact_example_sort_key(example: ImpactExample) -> tuple[bool, float, str]:
    return (
        example.flipped is True,
        abs(example.score_delta or 0.0),
        example.variant_id,
    )


def _impact_trial_sort_key(example: ImpactExample) -> tuple[bool, int, str]:
    repeat = _run_id_repeat_index(example.run_id)
    return (repeat is None, repeat or 0, example.run_id)


def _is_invariant_expected_effect(value: Any) -> bool:
    effect = str(value or PerturbationExpectedEffect.SAME_VERDICT.value).strip().lower()
    return effect == PerturbationExpectedEffect.SAME_VERDICT.value


def _run_id_repeat_index(run_id: str) -> int | None:
    _, sep, suffix = str(run_id).rpartition("-")
    if sep and suffix.isdigit():
        return int(suffix)
    return None


def _perturbation_type_stats(
    artifact: dict[str, Any],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {
        (
            str(row.get("family", "")),
            str(row.get("operator", "")),
            str(row.get("dimension", "")),
        ): row
        for row in artifact.get("perturbation_attribution", [])
    }


def _optional_stat_float(
    stat: dict[str, Any] | None,
    key: str,
    default: float | None,
) -> float | None:
    if stat is None:
        return default
    value = stat.get(key)
    if value is None:
        return None if key in stat else default
    return float(value)
