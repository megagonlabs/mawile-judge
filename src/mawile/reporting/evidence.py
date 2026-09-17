from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from mawile.measurement import baselines_by_item
from mawile.operator_manifest import operator_manifest_from_artifact
from mawile.reporting.comparisons import compare_result
from mawile.reporting.traces import first_field_change, trace_by_result
from mawile.schemas import AuditRunConfig


MAX_EXAMPLE_TEXT = 700


def build_report_evidence(
    config: AuditRunConfig,
    artifact: dict[str, Any],
    judge_call_traces: list[dict[str, Any]] | None = None,
    *,
    max_examples: int = 5,
) -> dict[str, Any]:
    """Compact, factual evidence packet used by report writers.

    The run artifact can be quite large. This packet keeps the highest-signal
    facts and concrete examples small enough for an LLM to interpret and for the
    deterministic report to render.
    """

    risks = artifact.get("item_risks", [])
    results = artifact.get("judge_results", [])
    perturbations = artifact.get("perturbations", [])
    parse_errors_by_item = Counter(
        str(result.get("item_id"))
        for result in results
        if result.get("parse_status") != "ok"
    )

    return {
        "run_id": artifact.get("run_id"),
        "judge": {
            "provider": config.judge.provider,
            "model": config.judge.model,
            "output_type": config.judge.output_type.value,
            "threshold": config.judge.threshold,
        },
        "agents": {
            "perturbation": {
                "provider": config.perturbation_agent.provider,
                "model": config.perturbation_agent.model,
            },
            "validator": {
                "provider": config.resolved_validator_provider(),
                "model": config.resolved_validator_model(),
            },
            "planner": {
                "provider": config.resolved_planner_provider(),
                "model": config.resolved_planner_model(),
            },
            "summary": {
                "provider": config.resolved_summary_provider(),
                "model": config.resolved_summary_model(),
            },
        },
        "sample": {
            "items": artifact.get("item_count", 0),
            "perturbations": artifact.get("perturbation_count", 0),
            "repeats": config.audit.repeats,
            "judge_calls": len(results),
        },
        "headline_metrics": _headline_metrics(artifact),
        "correctness": artifact.get("correctness", {}),
        "directional": artifact.get("directional", {}),
        "replay": _replay_evidence(artifact),
        "equivariant": artifact.get("equivariant", {}),
        "coverage": artifact.get("coverage", {}),
        "operator_manifest": operator_manifest_from_artifact(artifact),
        "perturbation_ledger": artifact.get("perturbation_ledger", []),
        "top_attribution": _top_attribution(artifact.get("attribution", [])),
        "top_perturbation_types": _top_perturbation_types(
            artifact.get("perturbation_attribution", [])
        ),
        "top_routes": _top_routes(
            risks,
            top_k=config.audit.top_k_routes,
            parse_errors_by_item=parse_errors_by_item,
        ),
        "evidence_quality": _evidence_quality(
            artifact,
            perturbations=perturbations,
            results=results,
            parse_errors_by_item=parse_errors_by_item,
        ),
        "examples": summarize_report_examples(
            artifact,
            judge_call_traces or [],
            max_examples=max_examples,
        ),
    }


def summarize_report_examples(
    artifact: dict[str, Any],
    judge_call_traces: list[dict[str, Any]] | None = None,
    *,
    max_examples: int = 5,
) -> list[dict[str, Any]]:
    perturbations = {
        record["variant_id"]: record
        for record in artifact.get("perturbations", [])
    }
    type_metrics = _type_metrics_by_key(artifact)
    baselines = baselines_by_item(artifact.get("judge_results", []))
    traces = trace_by_result(judge_call_traces or [])
    examples_by_variant: dict[tuple[str, str], dict[str, Any]] = {}

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
        example = {
            "family": str(perturbation.get("family", "")),
            "operator": str(perturbation.get("operator", "")),
            "dimension": str(perturbation.get("metadata", {}).get("dimension", "")),
            "expected_effect": str(perturbation.get("expected_effect", "")),
            "variant_id": str(variant_id),
            "item_id": str(result.get("item_id")),
            "summary": str(perturbation.get("metadata", {}).get("summary", "")),
            "changed_field": changed_field,
            **comparison,
            "before": _truncate(before),
            "after": _truncate(after),
            "impact_score": (1.0 if comparison["flipped"] is True else 0.0)
            + abs(comparison["score_delta"] or 0.0),
        }
        key = (example["item_id"], example["variant_id"])
        existing = examples_by_variant.get(key)
        if existing is None or _example_sort_key(example) > _example_sort_key(existing):
            examples_by_variant[key] = example

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for example in examples_by_variant.values():
        grouped[(example["family"], example["operator"])].append(example)

    best_per_operator: list[dict[str, Any]] = []
    for (family, operator), operator_examples in grouped.items():
        selected = sorted(operator_examples, key=_example_sort_key, reverse=True)[0]
        metrics = type_metrics.get((family, operator, selected["dimension"]))
        if metrics is not None:
            selected = {
                **selected,
                "type_flip_rate": metrics.get("flip_rate"),
                "type_mean_shift": metrics.get("mean_shift"),
                "type_mean_signed_shift": metrics.get("mean_signed_shift"),
                "type_num_results": metrics.get("num_results", 0),
                "type_changed_fields": metrics.get("changed_fields", []),
            }
        best_per_operator.append(selected)
    return sorted(best_per_operator, key=_operator_example_sort_key, reverse=True)[:max_examples]


def route_reasons(risk: dict[str, Any], parse_error_count: int = 0) -> list[str]:
    reasons: list[str] = []
    if parse_error_count:
        reasons.append("parse failure")
    if risk.get("judge_error", risk.get("gold_error")) is True:
        reasons.append("gold mismatch")
    if int(risk.get("directional_contradictions") or 0) > 0:
        reasons.append("directional contradiction")
    elif int(risk.get("directional_missed") or 0) > 0:
        reasons.append("missed degradation")
    if int(risk.get("equivariant_inconsistent") or 0) > 0:
        reasons.append("position-swap inconsistency")
    if _invariant_flip_risk(risk) > 0:
        reasons.append("perturbation-sensitive")
    if float(risk.get("noise_risk") or 0.0) > 0:
        reasons.append("noisy baseline")
    if float(risk.get("boundary_proximity") or 0.0) > 0:
        reasons.append("near decision boundary")
    if reasons:
        return reasons
    if "flip_risk" in risk and risk.get("flip_risk") is None:
        return ["invariant evidence unavailable"]
    return ["low observed risk"]


def _headline_metrics(artifact: dict[str, Any]) -> dict[str, Any]:
    risks = artifact.get("item_risks", [])
    noise = artifact.get("noise_floor", {})
    directional = artifact.get("directional", {})
    return {
        "mean_noise_flip_rate": noise.get("mean_noise_flip_rate"),
        "mean_noise_score_sd": noise.get("mean_noise_score_sd"),
        "mean_flip_risk": _mean(
            [float(risk["flip_risk"]) for risk in risks if risk.get("flip_risk") is not None]
        ),
        "mean_excess_over_noise": _mean(
            [
                float(risk["excess_over_noise"])
                for risk in risks
                if risk.get("excess_over_noise") is not None
            ]
        ),
        "directional_degradation_detection_rate": directional.get(
            "degradation_detection_rate"
        ),
        "directional_contradiction_rate": directional.get("contradiction_rate"),
        "confidence_intervals": artifact.get("confidence_intervals", {}),
    }


def _replay_evidence(artifact: dict[str, Any]) -> dict[str, Any]:
    """Expose saved replay provenance without implying a new regeneration."""

    replay = artifact.get("replay") or {}
    if not replay:
        return {
            "mode": "source_run",
            "source_run_id": artifact.get("run_id"),
            "rendering_note": "This is saved source-run evidence; do not infer offline regeneration.",
        }
    return {
        "mode": "replay",
        "source_run_id": replay.get("source_run_id"),
        "suite_sha256": replay.get("suite_sha256"),
        "directional_cohort": replay.get("directional_cohort") or "frozen_source_baseline",
        "fresh_baseline_eligibility_present": replay.get("fresh_baseline_eligibility") is not None,
        "generator_validator_calls_are_historical": True,
        "directional_eligibility_caveat": (
            "Directional eligibility uses the frozen source baseline cohort; fresh replay "
            "eligibility is diagnostic only and does not replace it."
        ),
        "rendering_note": "Replay provenance only; do not infer that this report was regenerated offline.",
    }


def _top_attribution(attribution: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [row for row in attribution if row.get("family") != "noise"]
    return [
        {
            "family": row.get("family"),
            "num_results": row.get("num_results", 0),
            "flip_rate": row.get("family_flip_rate"),
            "mean_shift": row.get("family_mean_shift"),
            "signed_shift": row.get("family_mean_signed_shift"),
            "excess_over_noise": row.get("family_excess_over_noise"),
        }
        for row in sorted(
            rows,
            key=lambda row: (
                _sort_numeric(row.get("family_flip_rate")),
                abs(_sort_numeric(row.get("family_mean_signed_shift"))),
            ),
            reverse=True,
        )
    ]


def _top_perturbation_types(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "family": row.get("family"),
            "operator": row.get("operator"),
            "dimension": row.get("dimension"),
            "num_results": row.get("num_results", 0),
            "num_variants": row.get("num_variants", 0),
            "num_items": row.get("num_items", 0),
            "changed_fields": row.get("changed_fields", []),
            "flip_rate": row.get("flip_rate"),
            "mean_shift": row.get("mean_shift"),
            "signed_shift": row.get("mean_signed_shift"),
            "excess_over_noise": row.get("excess_over_noise"),
        }
        for row in sorted(
            rows,
            key=lambda row: (
                _sort_numeric(row.get("flip_rate")),
                abs(_sort_numeric(row.get("mean_signed_shift"))),
            ),
            reverse=True,
        )
    ]


def _top_routes(
    risks: list[dict[str, Any]],
    *,
    top_k: int,
    parse_errors_by_item: Counter[str],
) -> list[dict[str, Any]]:
    routes: list[dict[str, Any]] = []
    ranked_risks = sorted(
        risks,
        key=lambda risk: (
            -_invariant_flip_risk(risk),
            str(risk.get("item_id", "")),
        ),
    )
    for rank, risk in enumerate(ranked_risks[:top_k], start=1):
        item_id = str(risk.get("item_id"))
        reasons = route_reasons(risk, parse_errors_by_item[item_id])
        routes.append(
            {
                "rank": rank,
                "item_id": item_id,
                "original_verdict": risk.get("original_verdict"),
                "flip_risk": risk.get("flip_risk"),
                "excess_over_noise": risk.get("excess_over_noise"),
                "noise_risk": risk.get("noise_risk"),
                "boundary_proximity": risk.get("boundary_proximity", 0.0),
                "gold_label": risk.get("gold_label"),
                "judge_correct": risk.get("judge_correct"),
                "judge_error": risk.get("judge_error", risk.get("gold_error")),
                "reasons": reasons,
            }
        )
    return routes


def _invariant_flip_risk(risk: dict[str, Any]) -> float:
    return float(risk.get("flip_risk") or 0.0)


def _evidence_quality(
    artifact: dict[str, Any],
    *,
    perturbations: list[dict[str, Any]],
    results: list[dict[str, Any]],
    parse_errors_by_item: Counter[str],
) -> dict[str, Any]:
    validation_counts = Counter(
        str(record.get("status", "unknown"))
        for record in artifact.get("validation_report", [])
    )
    generation_records = artifact.get("generation_trace", [])
    # New artifacts may carry canonical application summaries.  Use them when
    # present; legacy artifacts fall back to explicit generation-trace statuses.
    application_summary = artifact.get("application_summary") or {}
    if application_summary:
        generation_count = int(application_summary["unique_variants"])
        generation_statuses = application_summary.get("generation_statuses") or {}
        application_statuses = application_summary.get("application_statuses") or {}
        generation_skips = int(generation_statuses.get("skipped", 0))
        generation_failures = int(generation_statuses.get("failed", 0))
        application_exclusions = int(application_statuses.get("baseline_ineligible", 0))
        application_generation_skips = int(application_statuses.get("generation_skipped", 0))
        application_generation_failures = int(application_statuses.get("generation_failed", 0))
    else:
        # Explicit legacy path only; raw trace length is not a generation count.
        generation_count = sum(
            1
            for record in generation_records
            if record.get("status") not in {"skipped", "failed"}
        )
        generation_skips = sum(
            1 for record in generation_records if record.get("status") == "skipped"
        )
        generation_failures = sum(
            1 for record in generation_records if record.get("status") == "failed"
        )
        application_exclusions = 0
        application_generation_skips = 0
        application_generation_failures = 0
    family_counts = Counter(str(record.get("family", "")) for record in perturbations)
    operator_counts = Counter(str(record.get("operator", "")) for record in perturbations)
    replay = artifact.get("replay") or {}
    judge_errors = [
        risk.get("item_id")
        for risk in artifact.get("item_risks", [])
        if risk.get("judge_error", risk.get("gold_error")) is True
    ]
    return {
        "transport_failures": int((artifact.get("coverage") or {}).get("transport_failures") or 0),
        "parse_errors": int((artifact.get("coverage") or {}).get("parse_failures") or 0),
        "analyzable_verdicts": int((artifact.get("coverage") or {}).get("analyzable_verdicts") or 0),
        "judge_calls_attempted": int((artifact.get("coverage") or {}).get("judge_calls_attempted") or len(results)),
        "completion_rate": (artifact.get("coverage") or {}).get("completion_rate"),
        "parse_rate": (artifact.get("coverage") or {}).get("parse_rate"),
        "parse_error_items": sorted(parse_errors_by_item),
        "gold_disagreements": len(judge_errors),
        "gold_disagreement_items": sorted(str(item_id) for item_id in judge_errors),
        "judge_errors": len(judge_errors),
        "judge_error_items": sorted(str(item_id) for item_id in judge_errors),
        "validation": dict(sorted(validation_counts.items())),
        "generated_perturbations": generation_count,
        "generation_skips": generation_skips,
        "generation_failures": generation_failures,
        "application_exclusions": application_exclusions,
        "application_generation_skips": application_generation_skips,
        "application_generation_failures": application_generation_failures,
        "replay_source_run_id": replay.get("source_run_id"),
        "replay_suite_sha256": replay.get("suite_sha256"),
        "replay_directional_eligibility_caveat": (
            "Directional eligibility uses the frozen source baseline cohort; fresh replay "
            "eligibility is diagnostic only and does not replace it."
            if replay
            else None
        ),
        "replay_generation_validation_historical": bool(replay),
        "perturbation_families": dict(sorted(family_counts.items())),
        "perturbation_operators": dict(sorted(operator_counts.items())),
    }


def _mean(values: list[Any]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    return sum(numeric) / len(numeric) if numeric else None


def _sort_numeric(value: Any) -> float:
    try:
        return float(value) if value is not None else -1.0
    except (TypeError, ValueError):
        return -1.0


def _example_sort_key(example: dict[str, Any]) -> tuple[bool, float, str, str, str]:
    return (
        bool(example.get("flipped")),
        abs(float(example.get("score_delta") or 0.0)),
        str(example.get("family", "")),
        str(example.get("operator", "")),
        str(example.get("variant_id", "")),
    )


def _operator_example_sort_key(example: dict[str, Any]) -> tuple[float, bool, float, str, str]:
    return (
        float(example.get("type_flip_rate") or 0.0),
        bool(example.get("flipped")),
        abs(float(example.get("score_delta") or 0.0)),
        str(example.get("family", "")),
        str(example.get("operator", "")),
    )


def _type_metrics_by_key(artifact: dict[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {
        (
            str(row.get("family", "")),
            str(row.get("operator", "")),
            str(row.get("dimension", "")),
        ): row
        for row in artifact.get("perturbation_attribution", [])
    }


def _truncate(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if len(text) <= MAX_EXAMPLE_TEXT:
        return text
    return f"{text[:MAX_EXAMPLE_TEXT].rstrip()}..."
