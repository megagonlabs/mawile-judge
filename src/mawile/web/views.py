from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Any

from mawile.config_payload import dump_config_yaml, validate_config_payload
from mawile.env import load_environment
from mawile.io import load_items
from mawile.measurement import (
    result_label as canonical_result_label,
    result_position as canonical_result_position,
    result_score as canonical_result_score,
    validate_items_for_judge,
)
from mawile.metrics.directional import case_contradicted, case_detected
from mawile.operator_catalog import (
    build_perturbation_previews,
    build_perturbation_previews_for_specs,
    catalog_operator_rows,
    dimension_rows,
    estimate_run_size,
    operator_status,
    validation_catalog_rows,
)
from mawile.operator_manifest import operator_manifest_from_artifact
from mawile.perturbations.registry import DIMENSIONS, DIRECTIONAL_DIMENSIONS
from mawile.project import PROJECT_ROOT, discover_config_files, discover_run_artifacts
from mawile.providers import provider_has_api_key
from mawile.reporting.evidence import route_reasons
from mawile.reporting.view_model import (
    directional_examples_by_operator,
    impact_trial_options,
    summarize_perturbation_type_impacts,
)
from mawile.run_artifacts import ARTIFACT_FILES, RunBundle
from mawile.schemas import OutputType, ScoreDirection
from mawile.web.forms import yaml_text
from mawile.web.state import AppState


def base_context(state: AppState, *, active: str) -> dict[str, Any]:
    return {
        "active": active,
        "config_files": [
            {"path": str(path), "label": relative_label(path)}
            for path in discover_config_files()
        ],
        "run_artifacts": [
            {
                "run_id": path.parent.name,
                "label": run_label(path.parent.name),
                "path_label": relative_label(path.parent),
            }
            for path in discover_run_artifacts()
        ],
        "current_source": (
            relative_label(state.config_source_path)
            if state.config_source_path
            else ("Uploaded config" if state.config_uploaded else "New config")
        ),
    }


def _raw_override_mapping_text(agent: dict[str, Any]) -> str:
    """Render an inherited decoding override without conflating {} and blank."""

    value = agent.get("decoding_params")
    return "{}" if value == {} else yaml_text(value)


def configure_context(
    state: AppState,
    *,
    errors: list[str] | None = None,
    notice: str | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    load_environment()
    payload = state.config_payload
    base_dir = state.config_base_dir
    config, validation_error = (
        validate_config_payload(payload, base_dir) if payload else (None, None)
    )
    items = []
    item_error = None
    rows = []
    previews = []
    preview_lookup = {}
    estimate = None
    labeled_count = 0
    has_api_key = bool(os.getenv("OPENAI_API_KEY"))
    if config is not None:
        try:
            has_api_key = provider_has_api_key(
                config,
                config.perturbation_agent.provider,
            )
            items = load_items(config.data)
            validate_items_for_judge(items, config.judge)
            labeled_count = sum(1 for item in items if item.gold_label is not None)
            rows = dimension_rows(config, has_api_key=has_api_key, include_not_runnable=True)
            previews = build_perturbation_previews(
                config,
                items,
                has_api_key=has_api_key,
                max_examples_per_dimension=1,
                include_not_runnable=True,
            )
            preview_lookup = {
                preview.operator: preview
                for preview in build_perturbation_previews_for_specs(
                    config,
                    items,
                    [*DIMENSIONS, *DIRECTIONAL_DIMENSIONS],
                    has_api_key=has_api_key,
                    max_examples_per_dimension=1,
                    include_not_runnable=True,
                )
            }
            preview_lookup.update({preview.operator: preview for preview in previews})
            estimate = estimate_run_size(config, items)
        except Exception as exc:  # noqa: BLE001 - displayed as validation context.
            item_error = f"{type(exc).__name__}: {exc}"

    audit = payload.get("audit", {}) if isinstance(payload.get("audit"), dict) else {}
    data = payload.get("data", {}) if isinstance(payload.get("data"), dict) else {}
    judge = payload.get("judge", {}) if isinstance(payload.get("judge"), dict) else {}
    output = payload.get("output", {}) if isinstance(payload.get("output"), dict) else {}
    perturbation_agent = (
        payload.get("perturbation_agent", {})
        if isinstance(payload.get("perturbation_agent"), dict)
        else {}
    )
    planner_agent = (
        payload.get("planner_agent", {})
        if isinstance(payload.get("planner_agent"), dict)
        else {}
    )
    summary_agent = (
        payload.get("summary_agent", {})
        if isinstance(payload.get("summary_agent"), dict)
        else {}
    )
    validator_agent = (
        payload.get("validator_agent", {})
        if isinstance(payload.get("validator_agent"), dict)
        else {}
    )

    invariant_selected = set(audit.get("perturbation_operators") or [])
    directional_selected = set(audit.get("directional_perturbation_operators") or [])
    operator_mode = "manual"
    if (
        state.last_planned_operators
        and invariant_selected == set(state.last_planned_operators.get("effective", []))
        and directional_selected
        == set(state.last_planned_operators.get("effective_directional", []))
    ):
        operator_mode = "suggest"
    # An explicit empty selection is baseline-only; never reinterpret it as all.
    invariant_selects_all = False
    invariant_selected_count = len(invariant_selected)
    directional_allowed = config is None or config.judge.output_type != OutputType.PAIRWISE

    perturbation_decoding = (
        perturbation_agent.get("decoding_params")
        if isinstance(perturbation_agent.get("decoding_params"), dict)
        else {}
    )
    saved_custom_rows = [
        {
            "name": str(row.get("name") or ""),
            "target": str(row.get("target") or "item.input"),
            "instruction": str(row.get("instruction") or ""),
            "expected_effect": str(row.get("expected_effect") or "same_verdict"),
            "enabled": bool(row.get("enabled", True)),
        }
        for row in audit.get("custom_perturbations", []) or []
        if isinstance(row, dict)
    ]
    custom_selected_count = sum(
        1
        for row in saved_custom_rows
        if row["enabled"] and (row["name"].strip() or row["instruction"].strip())
    )
    custom_rows = list(saved_custom_rows)
    custom_rows.append(
        {
            "name": "",
            "target": "item.input",
            "instruction": "",
            "expected_effect": "same_verdict",
            "enabled": True,
        }
    )

    context = {
        **base_context(state, active="configure"),
        "payload": payload,
        "base_dir": base_dir,
        "errors": errors or [],
        "notice": notice,
        "job_id": job_id,
        "validation_error": validation_error,
        "item_error": item_error,
        "config_valid": config is not None and item_error is None,
        "item_count": len(items),
        "labeled_count": labeled_count,
        "rows": rows,
        "previews": previews,
        "preview_lookup": preview_lookup,
        "estimate": estimate,
        "has_api_key": has_api_key,
        "output_options": [option.value for option in OutputType],
        "score_direction_options": [option.value for option in ScoreDirection],
        "judge": judge,
        "data": data,
        "audit": audit,
        "output": output,
        "perturbation_agent": perturbation_agent,
        "planner_agent": planner_agent,
        "summary_agent": summary_agent,
        "validator_agent": validator_agent,
        "invariant_specs": DIMENSIONS,
        "directional_specs": DIRECTIONAL_DIMENSIONS,
        "directional_allowed": directional_allowed,
        "operator_status": (
            {
                spec.operator: operator_status(spec, config, has_api_key=has_api_key)
                for spec in [*DIMENSIONS, *DIRECTIONAL_DIMENSIONS]
            }
            if config is not None
            else {}
        ),
        "invariant_selected": invariant_selected,
        "directional_selected": directional_selected,
        "operator_mode": operator_mode,
        "invariant_selects_all": invariant_selects_all,
        "invariant_selected_count": invariant_selected_count,
        "custom_selected_count": custom_selected_count,
        "custom_rows": custom_rows,
        "raw_yaml": dump_config_yaml(payload),
        "last_planned": state.last_planned_operators,
        "gold_enabled": data.get("gold_field", "gold_label") is not None,
        "gold_field_value": data.get("gold_field") or "gold_label",
        "judge_decoding_text": yaml_text(judge.get("decoding_params") or {}),
        "output_schema_text": yaml_text(judge.get("output_schema")),
        "perturbation_decoding_text": yaml_text(perturbation_decoding),
        "planner_provider_value": planner_agent.get("provider") or "",
        "planner_model_value": planner_agent.get("model") or "",
        "planner_decoding_text": _raw_override_mapping_text(planner_agent),
        "summary_provider_value": summary_agent.get("provider") or "",
        "summary_model_value": summary_agent.get("model") or "",
        "summary_decoding_text": _raw_override_mapping_text(summary_agent),
        "validator_provider_value": validator_agent.get("provider") or "",
        "validator_model_value": validator_agent.get("model") or "",
        "validator_decoding_text": _raw_override_mapping_text(validator_agent),
    }
    return context


def catalog_context(state: AppState) -> dict[str, Any]:
    return {
        **base_context(state, active="catalog"),
        "invariant_catalog": catalog_operator_rows(DIMENSIONS),
        "directional_catalog": catalog_operator_rows(DIRECTIONAL_DIMENSIONS),
        "validation_catalog": validation_catalog_rows(),
    }


def report_context(
    state: AppState,
    bundle: RunBundle,
    *,
    selected_operator: str | None = None,
    selected_item: str | None = None,
    selected_directional_operator: str | None = None,
) -> dict[str, Any]:
    artifact = bundle.artifact
    summaries = summarize_perturbation_type_impacts(artifact, bundle.traces)
    selected_type = _selected_summary(summaries, selected_operator)
    example = None
    example_trials = []
    if selected_type is not None and selected_type.examples:
        example = selected_type.examples[0]
        example_trials = list(impact_trial_options(artifact, bundle.traces, [example])[0])
        if example_trials:
            example = example_trials[0]

    directional = artifact.get("directional") or {}
    directional_examples = directional_examples_by_operator(artifact, bundle.traces)
    directional_row = _selected_directional_row(
        directional.get("operator_summaries") or [],
        directional_examples,
        selected_directional_operator,
    )
    directional_example = None
    directional_example_trials = []
    if directional_row is not None:
        operator_examples = directional_examples.get(str(directional_row.get("operator")), ())
        if operator_examples:
            directional_example = operator_examples[0]
            directional_example_trials = list(
                impact_trial_options(artifact, bundle.traces, [directional_example])[0]
            )
            if directional_example_trials:
                directional_example = directional_example_trials[0]

    item_risks = _ranked_review_risks(list(artifact.get("item_risks", [])), directional)
    selected_risk = _selected_risk(item_risks, selected_item)
    parse_errors = _parse_errors_by_item(artifact)
    cost_breakdown = build_cost_breakdown(artifact)

    return {
        **base_context(state, active="report"),
        "bundle": bundle,
        "bundle_run_label": run_label(bundle.run_id),
        "bundle_run_dir_label": relative_label(bundle.run_dir),
        "artifact": artifact,
        "report_action_panels": report_action_panels(artifact.get("report_interpretation") or {}),
        "generated_summary_notice": generated_summary_notice(
            artifact.get("report_interpretation") or {}
        ),
        "summaries": summaries,
        "selected_type": selected_type,
        "selected_example": example,
        "example_trials": example_trials,
        "selected_directional_row": directional_row,
        "selected_directional_example": directional_example,
        "directional_example_trials": directional_example_trials,
        "item_risks": item_risks,
        "selected_risk": selected_risk,
        "parse_errors": parse_errors,
        "overview_cards": overview_cards(artifact, summaries),
        "cost_category_rows": cost_breakdown["rows"],
        "cost_total_label": cost_breakdown["total_label"],
        "cost_pricing_complete": cost_breakdown["pricing_complete"],
        "cost_pricing_note": cost_breakdown["pricing_note"],
        "noise_flip_rate": (artifact.get("noise_floor") or {}).get("mean_noise_flip_rate"),
        "confusion": confusion_grid(artifact.get("correctness") or {}),
        "risk_cards": summaries[:4],
        "top_routes": [
            {
                **risk,
                "reasons": route_reasons(risk, parse_errors.get(str(risk.get("item_id")), 0)),
            }
            for risk in item_risks[:5]
        ],
        "quality": evidence_quality_stats(artifact),
        "correctness": artifact.get("correctness") or {},
        "directional": directional,
        "equivariant": artifact.get("equivariant") or {},
        "coverage": artifact.get("coverage") or {},
        "operator_manifest": operator_manifest_from_artifact(artifact),
        "ledger": artifact.get("perturbation_ledger") or [],
        "artifact_links": [
            {"key": key, "label": label}
            for key, label in [
                ("report", "Markdown report"),
                ("run-json", "Run JSON"),
                ("item-risks", "Item risks CSV"),
                ("config", "Resolved config"),
                ("suite", "Saved perturbation suite"),
                ("judge-calls", "Judge-call traces"),
                ("llm-calls", "LLM-call ledger"),
            ]
            if key in ARTIFACT_FILES
            and (key != "suite" or (bundle.run_dir / "suite.json").exists())
            and (
                key != "llm-calls"
                or bool(
                    bundle.paths.get("llm_call_trace")
                    and bundle.paths["llm_call_trace"].exists()
                )
            )
        ],
    }


def overview_cards(artifact: dict[str, Any], summaries: list[Any]) -> list[dict[str, Any]]:
    cost = artifact.get("cost", {}).get("experiment", {})
    noise = artifact.get("noise_floor", {})
    correctness = artifact.get("correctness") or {}
    directional = artifact.get("directional") or {}
    noise_rate = noise.get("mean_noise_flip_rate")
    mean_flip = _mean_numeric(
        risk.get("flip_risk")
        for risk in artifact.get("item_risks", [])
    )
    elevated = mean_flip is not None and float(mean_flip) > 0
    cards = [
        {
            "label": "Noise flip rate",
            "value": fmt_float(noise_rate),
            "caption": "Before perturbations",
            "icon": "i-activity",
            "bar": _clamp_unit(noise_rate),
            "tone": None,
        },
        {
            "label": "Invariant flip rate",
            "value": fmt_float(mean_flip),
            "caption": "After invariant perturbations",
            "icon": "i-activity",
            "bar": _clamp_unit(mean_flip),
            "tone": "warn" if elevated else None,
        }
    ]
    if correctness.get("has_gold_labels"):
        labeled = int(correctness.get("labeled_items") or 0)
        cards.insert(
            2,
            {
                "label": "Gold accuracy",
                "value": fmt_float(correctness.get("accuracy")),
                "caption": f"{labeled} labeled item{'' if labeled == 1 else 's'}",
                "icon": "i-target",
                "bar": _clamp_unit(correctness.get("accuracy")),
                "tone": None,
            },
        )
    if directional.get("has_directional"):
        detection = directional.get("degradation_detection_rate")
        contradiction = float(directional.get("contradiction_rate") or 0.0)
        detected = float(detection or 0.0)
        cards.insert(
            3 if correctness.get("has_gold_labels") else 2,
            {
                "label": "Directional detection",
                "value": fmt_float(detection),
                "caption": "Degraded item downgrade frequency",
                "icon": "i-trend-down",
                "bar": _clamp_unit(detection),
                "tone": "warn" if contradiction > 0 or detected < 1.0 else None,
            },
        )
    cards.append(
        {
            "label": "Cost",
            "value": (
                f"≥ {fmt_usd(cost.get('total_usd'))}"
                if cost.get("pricing_complete") is False
                else fmt_usd(cost.get("total_usd"))
            ),
            "caption": (
                f"{cost.get('total_calls', 0)} total calls"
                + (" · pricing incomplete" if cost.get("pricing_complete") is False else "")
            ),
            "icon": "i-file",
            "bar": None,
            "tone": None,
        }
    )
    return [card for card in cards if _card_value_available(card.get("value"))]


_COST_CATEGORY_LABELS = {
    "judge": "Judge",
    "planning": "Planning",
    "generation": "Perturbation generation",
    "perturbation_generation": "Perturbation generation",
    "validation": "Admissibility validation",
    "equivalence_validation": "Equivalence validation",
    "reporting": "Reporting",
}


def build_cost_breakdown(artifact: dict[str, Any]) -> dict[str, Any]:
    """Build compact cost rows while tolerating pre-breakdown run artifacts."""

    experiment = (artifact.get("cost") or {}).get("experiment") or {}
    raw_categories = experiment.get("by_category")
    categories = raw_categories if isinstance(raw_categories, dict) else {}
    if not categories and _cost_record_is_active(experiment):
        # Before phase-level accounting, aggregate token usage represented judge
        # calls. Preserve a useful row when an older run is opened in the UI.
        categories = {
            "judge": {
                "calls": experiment.get("judge_calls", experiment.get("total_calls", 0)),
                "input_tokens": experiment.get("input_tokens", 0),
                "cached_input_tokens": experiment.get("cached_input_tokens", 0),
                "cache_write_tokens": experiment.get("cache_write_tokens", 0),
                "output_tokens": experiment.get("output_tokens", 0),
                "total_usd": experiment.get("total_usd", 0),
                "pricing_complete": experiment.get(
                    "pricing_complete", not bool(experiment.get("unpriced_models"))
                ),
            }
        }

    ordered = sorted(
        categories.items(),
        key=lambda entry: _cost_category_sort_key(str(entry[0])),
    )
    rows = []
    for name, raw_row in ordered:
        row = raw_row if isinstance(raw_row, dict) else {}
        if not _cost_record_is_active(row):
            continue
        pricing_complete = bool(row.get("pricing_complete", True))
        rows.append(
            {
                "key": str(name),
                "label": _COST_CATEGORY_LABELS.get(
                    str(name), str(name).replace("_", " ").title()
                ),
                "calls": _cost_int(row.get("calls")),
                "input_tokens": _cost_int(row.get("input_tokens")),
                "cached_input_tokens": _cost_int(row.get("cached_input_tokens")),
                "cache_write_tokens": _cost_int(row.get("cache_write_tokens")),
                "output_tokens": _cost_int(row.get("output_tokens")),
                "total_usd": _format_precise_usd(row.get("total_usd")),
                "pricing_complete": pricing_complete,
            }
        )

    pricing_complete = bool(
        experiment.get("pricing_complete", not bool(experiment.get("unpriced_models")))
    )
    return {
        "rows": rows,
        "total_label": _format_precise_usd(experiment.get("total_usd")),
        "pricing_complete": pricing_complete,
        "pricing_note": _cost_pricing_note(experiment, pricing_complete),
    }


def _cost_record_is_active(record: dict[str, Any]) -> bool:
    return any(
        _cost_int(record.get(key)) > 0
        for key in (
            "calls",
            "total_calls",
            "input_tokens",
            "cached_input_tokens",
            "cache_write_tokens",
            "output_tokens",
        )
    )


def _cost_category_sort_key(name: str) -> tuple[int, str]:
    order = list(_COST_CATEGORY_LABELS)
    return (order.index(name) if name in order else len(order), name)


def _cost_pricing_note(experiment: dict[str, Any], pricing_complete: bool) -> str | None:
    if pricing_complete:
        return None
    reasons = []
    unpriced = [str(model) for model in experiment.get("unpriced_models") or []]
    if unpriced:
        reasons.append("no price entry for " + ", ".join(unpriced))
    missing_usage = _cost_int(experiment.get("missing_usage_calls"))
    if missing_usage:
        reasons.append(
            f"{missing_usage} call{'' if missing_usage == 1 else 's'} reported no token usage"
        )
    suffix = "; ".join(reasons) or "one or more calls could not be priced"
    return f"The displayed USD total is a lower bound: {suffix}."


def _cost_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _format_precise_usd(value: Any) -> str:
    try:
        amount = float(value or 0)
    except (TypeError, ValueError):
        return str(value)
    rendered = f"{amount:.6f}".rstrip("0").rstrip(".")
    if "." not in rendered:
        rendered += ".00"
    return f"${rendered}"


def _card_value_available(value: Any) -> bool:
    return str(value).strip().lower() not in {"", "n/a"}


def confusion_grid(correctness: dict[str, Any]) -> dict[str, Any]:
    matrix = correctness.get("confusion_matrix") or {}
    pred_labels: list[str] = []
    max_count = 0
    for preds in matrix.values():
        for pred, count in (preds or {}).items():
            if pred not in pred_labels:
                pred_labels.append(pred)
            try:
                max_count = max(max_count, int(count))
            except (TypeError, ValueError):
                continue
    rows = []
    for gold, preds in matrix.items():
        cells = []
        for pred in pred_labels:
            count = (preds or {}).get(pred, 0)
            try:
                intensity = round(float(count) / max_count, 3) if max_count else 0.0
            except (TypeError, ValueError):
                intensity = 0.0
            cells.append({"pred": pred, "count": count, "intensity": intensity})
        rows.append({"gold": gold, "cells": cells})
    return {"pred_labels": pred_labels, "rows": rows, "max": max_count}


def _clamp_unit(value: Any) -> float | None:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return None


def evidence_quality_stats(artifact: dict[str, Any]) -> dict[str, Any]:
    coverage = artifact.get("coverage") or {}
    judge_errors = sum(
        1
        for risk in artifact.get("item_risks", [])
        if risk.get("judge_error", risk.get("gold_error")) is True
    )
    validation_counts = Counter(
        str(record.get("status", "unknown"))
        for record in artifact.get("validation_report", [])
    )
    family_counts = Counter(
        str(record.get("family", ""))
        for record in artifact.get("perturbations", [])
    )
    validation_summary = (
        ", ".join(f"{status} {count}" for status, count in sorted(validation_counts.items()))
        if validation_counts
        else "No validation records"
    )
    return {
        "transport_failures": int(coverage.get("transport_failures") or 0),
        "parse_errors": int(coverage.get("parse_failures") or 0),
        "analyzable_verdicts": int(coverage.get("analyzable_verdicts") or 0),
        "judge_calls_attempted": int(
            coverage.get("judge_calls_attempted") or len(artifact.get("judge_results", []))
        ),
        "completion_rate": coverage.get("completion_rate"),
        "parse_rate": coverage.get("parse_rate"),
        "judge_errors": judge_errors,
        "generated_perturbations": _generated_perturbation_count(artifact),
        "validation_summary": validation_summary,
        "validation_counts": dict(validation_counts),
        "family_counts": dict(family_counts),
    }


def _generated_perturbation_count(artifact: dict[str, Any]) -> int:
    """Count generated variants, excluding generation skips/failures.

    New artifacts can provide the canonical summary; legacy artifacts use the
    generation trace with explicit status filtering rather than its raw length.
    """

    summary = artifact.get("application_summary") or {}
    if summary:
        return int(summary["unique_variants"])
    return sum(
        1
        for record in artifact.get("generation_trace", [])
        if record.get("status") not in {"skipped", "failed"}
    )


def selected_judge_results(artifact: dict[str, Any], item_id: str) -> list[dict[str, Any]]:
    effects = {
        str(perturbation.get("variant_id")): str(perturbation.get("expected_effect") or "")
        for perturbation in artifact.get("perturbations", [])
    }
    results = [
        result
        for result in artifact.get("judge_results", [])
        if str(result.get("item_id")) == str(item_id)
    ][:120]
    return [
        {
            **result,
            "check_label": _check_label(
                str(result.get("variant_id")), effects.get(str(result.get("variant_id")))
            ),
        }
        for result in results
    ]


def _check_label(variant_id: str, expected_effect: str | None) -> str:
    if variant_id == "original":
        return "Baseline"
    if expected_effect == "worse_verdict":
        return "Directional"
    if expected_effect == "same_candidate":
        return "Equivariant"
    if expected_effect:
        return "Invariant"
    return ""


def result_label(result: dict[str, Any]) -> str:
    value = canonical_result_label(result)
    if value is None:
        return "n/a"
    position = canonical_result_position(result)
    return f"{value} (position {position})" if position is not None else str(value)


def result_score(result: dict[str, Any]) -> str:
    score = canonical_result_score(result)
    if score is None:
        return "n/a"
    return fmt_float(score)


def relative_label(path: Path | str | None) -> str:
    if path is None:
        return ""
    parsed = Path(path)
    try:
        return str(parsed.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(parsed)


def run_label(run_id: str) -> str:
    value = str(run_id)
    if len(value) == 16 and value[8] == "T" and value.endswith("Z"):
        return (
            f"{value[0:4]}-{value[4:6]}-{value[6:8]} "
            f"{value[9:11]}:{value[11:13]} UTC"
        )
    return value


def family_label(value: str) -> str:
    return str(value).replace("_", " ").title()


_FAMILY_ACCENTS = {
    "judge_rubric": "rubric",
    "judge_prompt": "prompt",
    "agent_input": "dataset",
    "agent_output": "run",
    "agent_paired": "paired",
    "noise": "default",
}


def family_accent(value: str) -> str:
    return _FAMILY_ACCENTS.get(str(value).lower(), "default")


def title_label(value: str) -> str:
    return str(value).replace("_", " ").replace("-", " ").title()


def validation_label(value: Any) -> str:
    raw = "" if value is None else str(value).strip()
    normalized = raw.lower().replace("-", "_").replace(" ", "_")
    if not normalized:
        return "Not specified"
    if normalized in {"by_construction", "not_needed", "inherent"}:
        return "No LLM check recorded"
    if normalized == "meaning_preserve":
        return "LLM: Meaning preservation"
    if normalized == "directional_degradation":
        return "LLM: Directional Degradation"
    return title_label(raw)


def report_action_panels(interpretation: dict[str, Any]) -> list[dict[str, Any]]:
    sections = [
        ("Key findings", "key_findings", "i-check"),
        ("Recommended actions", "recommended_actions", "i-activity"),
        ("Limits of this run", "limitations", "i-alert"),
        ("Follow-up experiments", "follow_up_experiments", "i-zap"),
    ]
    return [
        {"title": title, "items": items, "icon": icon}
        for title, key, icon in sections
        if (items := _interpretation_items(interpretation, key))
    ]


def generated_summary_notice(interpretation: dict[str, Any]) -> str | None:
    source = str(interpretation.get("source") or "").strip().lower()
    if source == "llm" and not interpretation.get("llm_error"):
        return None
    if interpretation.get("llm_error"):
        return "LLM Summary Generation Failed. Showing deterministic fallback."
    return "LLM summary generation was not configured. Showing deterministic fallback."


def _interpretation_items(interpretation: dict[str, Any], key: str) -> list[str]:
    value = interpretation.get(key, [])
    if isinstance(value, str):
        value = [value]
    return [
        str(item).strip()
        for item in value or []
        if str(item).strip()
    ]


def fmt_float(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def fmt_signed(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):+.3f}"
    except (TypeError, ValueError):
        return str(value)


def fmt_usd(value: Any) -> str:
    if value is None:
        return "$0.00"
    try:
        return f"${float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def _selected_summary(summaries: list[Any], operator: str | None) -> Any | None:
    if not summaries:
        return None
    if operator:
        for summary in summaries:
            if summary.operator == operator:
                return summary
    return summaries[0]


def _selected_risk(risks: list[dict[str, Any]], item_id: str | None) -> dict[str, Any] | None:
    if not risks:
        return None
    if item_id:
        for risk in risks:
            if str(risk.get("item_id")) == str(item_id):
                return risk
    return risks[0]


def _ranked_review_risks(
    risks: list[dict[str, Any]],
    directional: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    stats = _directional_stats_by_item(directional or {})
    merged = [
        risk
        if "directional_cases" in risk
        else {**risk, **stats.get(str(risk.get("item_id")), _EMPTY_DIRECTIONAL_STATS)}
        for risk in risks
    ]
    ranked = sorted(
        merged,
        key=lambda risk: (
            -_review_priority(risk),
            -(_invariant_flip_risk(risk) or 0.0),
            str(risk.get("item_id", "")),
        ),
    )
    return [
        {
            **risk,
            "review_rank": rank,
            "review_priority": _review_priority(risk),
            "invariant_flip_risk": _invariant_flip_risk(risk),
        }
        for rank, risk in enumerate(ranked, start=1)
    ]


_EMPTY_DIRECTIONAL_STATS = {
    "directional_cases": 0,
    "directional_missed": 0,
    "directional_contradictions": 0,
    "directional_risk": None,
}


def _review_priority(risk: dict[str, Any]) -> float:
    return max(
        _invariant_flip_risk(risk) or 0.0,
        float(risk.get("directional_risk") or 0.0),
        float(risk.get("equivariant_risk") or 0.0),
    )


def _invariant_flip_risk(risk: dict[str, Any]) -> float | None:
    value = risk.get("flip_risk")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _directional_stats_by_item(directional: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cases_by_item: dict[str, list[dict[str, Any]]] = {}
    for case in directional.get("cases") or []:
        cases_by_item.setdefault(str(case.get("item_id")), []).append(case)
    if cases_by_item:
        stats: dict[str, dict[str, Any]] = {}
        for item_id, cases in cases_by_item.items():
            analyzable = [case for case in cases if case_detected(case) is not None]
            missed = sum(1 for case in analyzable if not case_detected(case))
            stats[item_id] = {
                "directional_cases": len(cases),
                "directional_missed": missed,
                "directional_contradictions": sum(
                    1 for case in analyzable if case_contradicted(case)
                ),
                "directional_risk": (
                    missed / len(analyzable) if analyzable else None
                ),
            }
        return stats

    # Older artifacts only persist the top missed cases; surface those as
    # reasons without inventing a per-item rate from a truncated list.
    stats = {}
    for row in directional.get("top_missed_degradations") or []:
        entry = stats.setdefault(
            str(row.get("item_id")), dict(_EMPTY_DIRECTIONAL_STATS)
        )
        entry["directional_cases"] += 1
        entry["directional_missed"] += 1
        if float(row.get("score_delta") or 0.0) > 0 or row.get("label_movement") == "improved":
            entry["directional_contradictions"] += 1
    return stats


def _selected_directional_row(
    operator_rows: list[dict[str, Any]],
    examples_by_operator: dict[str, tuple[Any, ...]],
    operator: str | None,
) -> dict[str, Any] | None:
    if not operator_rows:
        return None
    if operator:
        for row in operator_rows:
            if str(row.get("operator")) == operator:
                return row
    for row in operator_rows:
        if examples_by_operator.get(str(row.get("operator"))):
            return row
    return operator_rows[0]


def _parse_errors_by_item(artifact: dict[str, Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for result in artifact.get("judge_results", []):
        if result.get("parse_status") != "ok":
            counts[str(result.get("item_id"))] += 1
    return dict(counts)


def _mean_numeric(values: Any) -> float | None:
    parsed = []
    for value in values:
        try:
            parsed.append(float(value))
        except (TypeError, ValueError):
            continue
    if not parsed:
        return None
    return sum(parsed) / len(parsed)
