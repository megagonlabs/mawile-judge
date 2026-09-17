from __future__ import annotations

from typing import Any

from mawile.operator_manifest import operator_manifest_from_artifact
from mawile.reporting.evidence import build_report_evidence
from mawile.reporting.interpretation import deterministic_report_interpretation
from mawile.schemas import AuditRunConfig


def render_markdown_report(
    config: AuditRunConfig,
    artifact: dict[str, Any],
    judge_call_traces: list[dict[str, Any]] | None = None,
) -> str:
    evidence = build_report_evidence(config, artifact, judge_call_traces)
    interpretation = artifact.get("report_interpretation") or deterministic_report_interpretation(
        evidence
    )
    routes = evidence["top_routes"]
    attribution = artifact["attribution"]
    perturbation_attribution = artifact.get("perturbation_attribution", [])
    noise = artifact["noise_floor"]

    lines = [
        f"# MAWILE Report: {artifact['run_id']}",
        "",
        "## Judge Config Summary",
        "",
        f"- Judge: `{config.judge.provider}/{config.judge.model}`",
        "- Perturbation agent: "
        f"`{config.perturbation_agent.provider}/{config.perturbation_agent.model}`",
        "- Validator: "
        f"`{config.resolved_validator_provider()}/{config.resolved_validator_model()}`",
        f"- Output type: `{config.judge.output_type.value}`",
        f"- Items: `{artifact['item_count']}`",
        f"- Perturbations: `{artifact['perturbation_count']}`",
        f"- Repeats per original item: `{config.audit.repeats}`",
        "",
        *_provenance_lines(artifact),
        *_coverage_lines(artifact),
        *_planning_lines(artifact),
        *_operator_manifest_lines(artifact),
        "## Reliability Headline",
        "",
        _headline(artifact),
        "",
        "## Noise Floor",
        "",
        f"- Mean noise flip rate: `{_format_optional(noise.get('mean_noise_flip_rate'))}`",
        f"- Mean repeated-run score SD: `{_format_optional(noise.get('mean_noise_score_sd'))}`",
        "",
        *_correctness_lines(artifact),
        *_directional_lines(artifact),
        # New runs have no separate pairwise-equivalence lane. This helper
        # still renders legacy artifacts when they are opened and exported.
        *(_equivariant_lines(artifact) if artifact.get("equivariant") else []),
        "## Attribution",
        "",
        "| Family | Flip rate | Mean shift | Signed shift |",
        "|---|---:|---:|---:|",
    ]

    for row in attribution:
        lines.append(
            "| {family} | {flip} | {shift} | {signed} |".format(
                family=row["family"],
                flip=_format_optional(row.get("family_flip_rate")),
                shift=_format_optional(row.get("family_mean_shift")),
                signed=_format_signed_optional(row.get("family_mean_signed_shift")),
            )
        )

    lines.extend(
        [
            "",
            "## Perturbation Type Breakdown",
            "",
            "| Family | Perturbation type | Changed fields | Results | Variants | "
            "Items | Flip rate | Mean shift | Signed shift |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    if perturbation_attribution:
        for row in perturbation_attribution:
            lines.append(
                (
                    "| {family} | `{operator}` | {fields} | {results} | {variants} | "
                    "{items} | {flip} | {shift} | {signed} |"
                ).format(
                    family=row.get("family", ""),
                    operator=row.get("operator", ""),
                    fields=_format_changed_fields(row.get("changed_fields", [])),
                    results=row.get("num_results", 0),
                    variants=row.get("num_variants", 0),
                    items=row.get("num_items", 0),
                    flip=_format_optional(row.get("flip_rate")),
                    shift=_format_optional(row.get("mean_shift")),
                    signed=_format_signed_optional(row.get("mean_signed_shift")),
                )
            )
    else:
        lines.append("| none | `n/a` | n/a | 0 | 0 | 0 | not available | not available | not available |")

    lines.extend(
        [
            "",
            "## Confidence Intervals",
            "",
            "95% percentile-bootstrap intervals over items"
            f" (`{config.audit.bootstrap_samples}` resamples):",
            "",
            f"- Mean flip-risk: {_format_ci(artifact['confidence_intervals']['mean_flip_risk'])}",
            f"- Mean noise-risk: {_format_ci(artifact['confidence_intervals']['mean_noise_risk'])}",
            "",
            *_cost_lines(artifact["cost"]),
            "## Highest Flip Risk Items",
            "",
            "| Rank | Item | Verdict | Reason | Flip risk | Noise | Boundary |",
            "|---:|---|---|---|---:|---:|---:|",
        ]
    )

    for route in routes:
        lines.append(
            (
                "| {rank} | `{item}` | `{verdict}` | {reason} | {flip} | "
                "{noise} | {boundary} |"
            ).format(
                rank=route["rank"],
                item=route["item_id"],
                verdict=route["original_verdict"],
                reason=", ".join(route["reasons"]),
                flip=_format_optional(route["flip_risk"]),
                noise=_format_optional(route["noise_risk"]),
                boundary=_format_optional(route["boundary_proximity"]),
            )
        )

    lines.extend(
        [
            "",
            *_ledger_lines(artifact),
            *_interpretation_lines(interpretation),
            "",
            *_evidence_quality_lines(evidence),
            "",
            *_example_lines(evidence["examples"]),
        ]
    )
    return "\n".join(lines)


def _provenance_lines(artifact: dict[str, Any]) -> list[str]:
    replay = artifact.get("replay") or {}
    if replay:
        source_run_id = replay.get("source_run_id") or "not recorded"
        cohort = replay.get("directional_cohort") or "frozen source baseline"
        return [
            "## Report Provenance",
            "",
            f"- Report mode: `replay`; source run: `{source_run_id}`.",
            "- Generated perturbations and validation outcomes are reused from the "
            "saved frozen suite; generator and validator settings are historical and "
            "do not represent new replay measurements.",
            f"- Directional eligibility uses the `{cohort}` cohort. Any fresh replay "
            "baseline eligibility is diagnostic and does not replace the frozen source cohort.",
            "",
        ]
    return [
        "## Report Provenance",
        "",
        "- Report mode: `source run`; metrics, interpretation, and call provenance "
        "are rendered from the saved run artifact.",
        "",
    ]


def _planning_lines(artifact: dict[str, Any]) -> list[str]:
    """Render the suggester's selection when auto-routing produced the allow-list."""

    trace = artifact.get("planning_trace")
    if not trace:
        return []
    effective_total = len(trace.get("effective", [])) + len(
        trace.get("effective_directional", [])
    )
    selected_total = len(trace.get("selected", [])) + len(
        trace.get("selected_directional", [])
    )
    manual_total = len(trace.get("manual", [])) + len(trace.get("manual_directional", []))
    lines = [
        "## Perturbation Suggestions",
        "",
        f"- Auto-routed allow-list: `{effective_total}` operators "
        f"(`{selected_total}` suggester-selected, `{manual_total}` manual).",
    ]
    if trace.get("rationale"):
        lines.append(f"- Rationale: {trace['rationale']}")
    lines.append("")
    return lines


def _operator_manifest_lines(artifact: dict[str, Any]) -> list[str]:
    operators = operator_manifest_from_artifact(artifact).get("operators") or []
    if not operators:
        return []
    lines = [
        "## Operator Manifest",
        "",
        "| Perturbation | Changes | Relation | Expected effect | Validation |",
        "|---|---|---|---|---|",
    ]
    for row in operators:
        touches = ", ".join(str(value) for value in row.get("touches") or [])
        changes = str(row.get("target") or "")
        if touches:
            changes = f"{changes} ({touches})"
        lines.append(
            "| `{operator}` | `{changes}` | `{relation}` | `{effect}` | {validation} |".format(
                operator=row.get("operator", ""),
                changes=changes,
                effect=row.get("expected_effect", ""),
                relation=row.get("expected_relation", ""),
                validation=row.get("validation_kind") or "by construction",
            )
        )
    lines.append("")
    return lines


def _ledger_lines(artifact: dict[str, Any]) -> list[str]:
    ledger = artifact.get("perturbation_ledger") or []
    if not ledger:
        return []
    if any("generation_units" in row or "unique_variants" in row for row in ledger):
        lines = [
            "## Perturbation Ledger",
            "",
            "| Perturbation | Expected effect | Generation units | Generation status | Applications | Application status | Unique variants | Accepted variants | Rejected variants | Unavailable variants | Validation |",
            "|---|---|---:|---|---:|---|---:|---:|---:|---:|---|",
        ]
        for row in ledger:
            lines.append(
                "| `{operator}` | `{effect}` | {units} | `{generation}` | {applications} | `{application}` | {unique} | {accepted} | {rejected} | {unavailable} | {validation} |".format(
                    operator=row.get("operator", ""),
                    effect=row.get("expected_effect", ""),
                    units=row.get("generation_units", 0),
                    generation=row.get("generation_statuses", {}),
                    applications=row.get("applications", 0),
                    application=row.get("application_statuses", {}),
                    unique=row.get("unique_variants", 0),
                    accepted=row.get("accepted_variants", 0),
                    rejected=row.get("rejected_variants", 0),
                    unavailable=row.get("unavailable_variants", 0),
                    validation=row.get("validation_kind") or "by construction",
                )
            )
        lines.append("")
        return lines
    lines = [
        "## Perturbation Ledger",
        "",
        "| Perturbation | Expected effect | Attempted | Accepted | Rejected | Skipped | Validation |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for row in ledger:
        skipped = row.get("skipped_reason") or ""
        lines.append(
            "| `{operator}` | `{effect}` | {attempted} | {accepted} | {rejected} | {skipped} | {validation} |".format(
                operator=row.get("operator", ""),
                effect=row.get("expected_effect", ""),
                attempted=row.get("attempted", 0),
                accepted=row.get("accepted", 0),
                rejected=row.get("rejected", 0),
                skipped=skipped,
                validation=row.get("validation_kind") or "by construction",
            )
        )
    lines.append("")
    return lines


def _headline(artifact: dict[str, Any]) -> str:
    drift = _strongest_invariance_score_drift(artifact["attribution"])
    if drift is not None:
        return (
            f"This judge shows invariance score drift: `{drift['family']}` "
            f"perturbations move the score by "
            f"`{drift['family_mean_signed_shift']:+.2f}` on average despite "
            "preserving meaning."
        )
    attribution = [
        row
        for row in artifact["attribution"]
        if row["family"] != "noise"
        and row.get("family_flip_rate") is not None
        and row["family_flip_rate"] > 0
    ]
    if not attribution:
        if not any(
            row["family"] != "noise" and row.get("family_flip_rate") is not None
            for row in artifact["attribution"]
        ):
            return "Invariant stability is unavailable: no analyzable invariant comparisons were recorded."
        return "This judge is stable under the current invariance perturbation set."
    top = max(attribution, key=lambda row: row["family_flip_rate"])
    return (
        f"The highest observed instability comes from `{top['family']}` "
        f"perturbations with flip rate `{top['family_flip_rate']:.3f}`."
    )


def _directional_lines(artifact: dict[str, Any]) -> list[str]:
    directional = artifact.get("directional") or {}
    eligibility = directional.get("eligibility") or {}
    if not directional.get("has_directional"):
        if eligibility.get("directional_requested"):
            return [
                "## Directional Degradation",
                "",
                "- No directional cases were generated after baseline eligibility "
                "and operator applicability checks.",
                f"- Baseline-eligible items: `{eligibility.get('eligible_item_count', 0)}`; "
                f"skipped at the verdict floor: `{eligibility.get('ineligible_item_count', 0)}`.",
            ]
        return []
    lines = [
        "## Directional Degradation",
        "",
        f"- Directional cases: `{directional.get('num_cases', 0)}` "
        f"(`{directional.get('num_results', 0)}` judge calls)",
    ]
    if eligibility.get("filter_applied"):
        lines.append(
            f"- Baseline eligibility (`{eligibility.get('policy', 'judge_baseline')}`): "
            f"`{eligibility.get('eligible_item_count', 0)}` eligible; "
            f"`{eligibility.get('ineligible_item_count', 0)}` skipped at the verdict floor."
        )
    movement = directional.get("label_movement") or {}
    lines.extend(
        [
            f"- Detection basis: `{directional.get('detection_basis') or 'none'}` "
            f"(`{directional.get('num_analyzable_cases', 0)}` analyzable case(s))",
            f"- Degradation detection rate: "
            f"`{_format_optional(directional.get('degradation_detection_rate'))}`",
            f"- Contradiction rate: "
            f"`{_format_optional(directional.get('contradiction_rate'))}`",
            f"- Noise-adjusted detection rate: "
            f"`{_format_optional(directional.get('noise_adjusted_detection_rate'))}`",
        ]
    )
    if directional.get("score_supported"):
        lines.extend(
            [
                f"- Threshold worsening rate: "
                f"`{_format_optional(directional.get('threshold_worsening_rate'))}`",
                f"- Mean score delta: "
                f"`{_format_optional(directional.get('mean_score_delta'))}`",
            ]
        )
    else:
        lines.append(
            f"- Class movement: `{movement.get('worsened', 0)}` worsened, "
            f"`{movement.get('unchanged', 0)}` unchanged, "
            f"`{movement.get('improved', 0)}` improved, "
            f"`{movement.get('unknown', 0)}` unanalyzable"
        )
    lines.extend(
        [
            "",
            "| Perturbation type | Cases | Detection | Noise-adjusted | Threshold worsening | Contradiction | Mean delta |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in directional.get("operator_summaries", []):
        lines.append(
            "| `{operator}` | {cases} | {detect} | {noise} | {threshold} | {contradiction} | {delta} |".format(
                operator=row.get("operator", ""),
                cases=row.get("num_cases", 0),
                detect=_format_optional(row.get("degradation_detection_rate")),
                noise=_format_optional(row.get("noise_adjusted_detection_rate")),
                threshold=_format_optional(row.get("threshold_worsening_rate")),
                contradiction=_format_optional(row.get("contradiction_rate")),
                delta=_format_optional(row.get("mean_score_delta")),
            )
        )
    missed = directional.get("top_missed_degradations") or []
    if missed:
        lines.extend(
            [
                "",
                "Top missed or contradicted degradations:",
                "",
                "| Item | Perturbation | Movement | Baseline | Perturbed | Delta | Noise floor |",
                "|---|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in missed:
            lines.append(
                "| `{item}` | `{operator}` | {movement} | {baseline} | {perturbed} | {delta} | {noise} |".format(
                    item=row.get("item_id", ""),
                    operator=row.get("operator", ""),
                    movement=row.get("label_movement") or "n/a",
                    baseline=_verdict_cell(row.get("baseline_score"), row.get("baseline_label")),
                    perturbed=_verdict_cell(row.get("perturbed_score"), row.get("perturbed_label")),
                    delta=_format_optional(row.get("score_delta")),
                    noise=_directional_noise_cell(row),
                )
            )
    lines.append("")
    return lines


def _correctness_lines(artifact: dict[str, Any]) -> list[str]:
    correctness = artifact.get("correctness") or {}
    if not correctness.get("has_gold_labels"):
        return []
    lines = [
        "## Validity Against Gold Labels",
        "",
        f"- Labeled items: `{correctness.get('labeled_items', 0)}`",
        f"- Accuracy: `{_format_optional(correctness.get('accuracy'))}`",
        f"- Error rate: `{_format_optional(correctness.get('error_rate'))}`",
        f"- Correct / errors: `{correctness.get('correct', 0)}` / `{correctness.get('errors', 0)}`",
    ]
    if correctness.get("mean_absolute_score_error") is not None:
        lines.append(
            "- Mean absolute score error: "
            f"`{_format_optional(correctness.get('mean_absolute_score_error'))}`"
        )
    if correctness.get("rank_correlation") is not None:
        lines.append(
            "- Spearman rank correlation: "
            f"`{_format_optional(correctness.get('rank_correlation'))}`"
        )
    if correctness.get("tolerance") is not None:
        lines.append(f"- Scalar gold tolerance: `{correctness.get('tolerance')}`")
    review = correctness.get("review_utility") or {}
    if review:
        lines.extend(
            [
                f"- Top-{review.get('top_k', 0)} routed precision: "
                f"`{_format_optional(review.get('routed_precision'))}`",
                f"- Top-{review.get('top_k', 0)} error capture rate: "
                f"`{_format_optional(review.get('error_capture_rate'))}`",
            ]
        )
    matrix = correctness.get("confusion_matrix") or {}
    if matrix:
        lines.extend(["", "Confusion matrix (`gold` rows, `judge` columns):", ""])
        columns = sorted({pred for preds in matrix.values() for pred in preds})
        lines.append("| Gold | " + " | ".join(columns) + " |")
        lines.append("|---|" + "|".join("---:" for _ in columns) + "|")
        for gold, preds in sorted(matrix.items()):
            values = [str(preds.get(column, 0)) for column in columns]
            lines.append(f"| `{gold}` | " + " | ".join(values) + " |")
    lines.append("")
    return lines


def _coverage_lines(artifact: dict[str, Any]) -> list[str]:
    coverage = artifact.get("coverage") or {}
    if not coverage:
        return []
    return [
        "## Coverage and Missingness",
        "",
        f"- Source items: `{coverage.get('source_items_sampled', 0)}`",
        *(
            [f"- Operator applications planned: `{coverage['operator_applications_planned']}`"]
            if "operator_applications_planned" in coverage
            else (
                [f"- Operator applications attempted: `{coverage['operator_applications_attempted']}`"]
                if "operator_applications_attempted" in coverage
                else []
            )
        ),
        *(
            [f"- Generation units: `{coverage.get('generation_units', 0)}` ({coverage.get('generation_statuses', {})})"]
            if "generation_units" in coverage
            else []
        ),
        *(
            [f"- Application statuses: `{coverage.get('application_statuses', {})}`"]
            if "application_statuses" in coverage
            else []
        ),
        f"- Variants generated / rejected / accepted: "
        f"`{coverage.get('variants_generated', 0)}` / "
        f"`{coverage.get('variants_rejected_by_validation', 0)}` / "
        f"`{coverage.get('variants_accepted', 0)}`",
        f"- Variants with validation unavailable: "
        f"`{coverage.get('variants_validation_unavailable', 0)}`",
        f"- Judge calls attempted / analyzable: "
        f"`{coverage.get('judge_calls_attempted', 0)}` / "
        f"`{coverage.get('analyzable_verdicts', 0)}`",
        f"- Transport / parse failures: `{coverage.get('transport_failures', 0)}` / "
        f"`{coverage.get('parse_failures', 0)}`",
        f"- Completion / parse / analyzable rates: "
        f"`{_format_optional(coverage.get('completion_rate'))}` / "
        f"`{_format_optional(coverage.get('parse_rate'))}` / "
        f"`{_format_optional(coverage.get('analyzable_rate'))}`",
        f"- Semantic / conservative all-attempt invariant risk: "
        f"`{_format_optional(coverage.get('semantic_mean_invariant_risk'))}` / "
        f"`{_format_optional(coverage.get('conservative_all_attempt_mean_invariant_risk'))}`",
        "",
    ]


def _equivariant_lines(artifact: dict[str, Any]) -> list[str]:
    metrics = artifact.get("equivariant") or {}
    if not metrics.get("has_equivariant"):
        return []
    return [
        "## Pairwise Position Equivariance",
        "",
        f"- Position-swap cases / analyzable: `{metrics.get('num_cases', 0)}` / "
        f"`{metrics.get('analyzable_cases', 0)}`",
        f"- Candidate-identity consistency: "
        f"`{_format_optional(metrics.get('candidate_identity_consistency_rate'))}`",
        f"- Conservative all-attempt consistency: "
        f"`{_format_optional(metrics.get('all_attempt_consistency_rate'))}`",
        f"- Raw A/B inversion rate: "
        f"`{_format_optional(metrics.get('position_inversion_rate'))}`",
        f"- First-position selection rate: "
        f"`{_format_optional(metrics.get('first_position_selection_rate'))}`",
        "",
    ]


_INVARIANCE_FAMILIES = {
    "judge_prompt",
    "judge_rubric",
    "agent_input",
    "agent_output",
    "agent_paired",
}


def _strongest_invariance_score_drift(attribution: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Meaning-preserving probes can expose a signed score drift without being
    directional perturbation tests."""

    candidates = [
        row
        for row in attribution
        if row["family"] in _INVARIANCE_FAMILIES
        and row.get("family_mean_signed_shift") is not None
        and abs(row["family_mean_signed_shift"]) >= 0.5
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda row: abs(row["family_mean_signed_shift"]))


def _cost_lines(cost: dict[str, Any]) -> list[str]:
    experiment = cost["experiment"]
    categories = experiment.get("by_category", {})
    active_categories = [
        (name, row)
        for name, row in categories.items()
        if any(
            int(row.get(field) or 0) > 0
            for field in ("calls", "input_tokens", "cached_input_tokens", "output_tokens")
        )
    ]
    active_categories.sort(key=lambda entry: _cost_category_sort_key(entry[0]))
    show_cache_writes = any(
        int(row.get("cache_write_tokens") or 0) > 0
        for _, row in active_categories
    )

    total_usd = _format_usd(experiment.get("total_usd"))
    pricing_complete = bool(experiment.get("pricing_complete", True))
    cost_prefix = "`$" if pricing_complete else "at least `$"
    lines = [
        "## Cost",
        "",
        f"- All recorded LLM activity cost {cost_prefix}{total_usd}` across "
        f"`{experiment.get('total_calls', 0)}` calls "
        f"(`{experiment.get('input_tokens', 0)}` input tokens, including "
        f"`{experiment.get('cached_input_tokens', 0)}` cached reads and "
        f"`{experiment.get('cache_write_tokens', 0)}` cache writes, plus "
        f"`{experiment.get('output_tokens', 0)}` output tokens).",
    ]

    if active_categories:
        if show_cache_writes:
            header = (
                "| LLM phase | Calls | Input tokens | Cached input | Cache writes | "
                "Output tokens | Cost (USD) |"
            )
            separator = "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"
        else:
            header = (
                "| LLM phase | Calls | Input tokens | Cached input | Output tokens | "
                "Cost (USD) |"
            )
            separator = "| --- | ---: | ---: | ---: | ---: | ---: |"
        lines.extend(
            [
                "",
                header,
                separator,
                *[
                    _cost_category_row(name, row, show_cache_writes=show_cache_writes)
                    for name, row in active_categories
                ],
                "",
            ]
        )

    if not pricing_complete:
        unpriced = experiment.get("unpriced_models", [])
        missing_usage_calls = int(experiment.get("missing_usage_calls") or 0)
        reasons = []
        if unpriced:
            joined = ", ".join(f"`{model}`" for model in unpriced)
            reasons.append(f"no price entry was available for {joined}")
        if missing_usage_calls:
            reasons.append(
                f"{missing_usage_calls} call{'' if missing_usage_calls == 1 else 's'} "
                "reported no token usage"
            )
        reason = " and ".join(reasons) or "one or more calls could not be priced"
        lines.append(
            f"- **Pricing is incomplete:** the reported total is a lower bound because "
            f"{reason}."
        )
    elif not experiment.get("input_tokens") and not experiment.get("output_tokens"):
        lines.append("- No LLM token usage was reported (for example, by the mock judge).")

    # Historical artifacts may carry this retired projection.  Keep their
    # rendered report legible without emitting it for new v2 runs.
    production = cost.get("production_sampling")
    if isinstance(production, dict):
        projected_usd = production.get("usd_per_100_items")
        projected_cost = (
            f"`${_format_usd(projected_usd)}` per 100 items"
            if projected_usd is not None
            else "not available"
        )
        lines.append(
            "- Legacy Judge-only production projection: mean "
            f"`{_format_optional(production.get('mean_repeats_to_stable'))}` repeats to a stable "
            f"verdict (max `{production.get('max_repeats_to_stable')}`), estimated judge "
            f"inference cost {projected_cost}. This excludes planning, perturbation generation, "
            "validation, and reporting."
        )
    lines.append("")
    return lines


_COST_CATEGORY_LABELS = {
    "judge": "Judge",
    "planning": "Planning",
    "generation": "Perturbation generation",
    "perturbation_generation": "Perturbation generation",
    "validation": "Equivalence validation",
    "equivalence_validation": "Equivalence validation",
    "reporting": "Reporting",
}


def _cost_category_sort_key(name: str) -> tuple[int, str]:
    order = list(_COST_CATEGORY_LABELS)
    return (order.index(name) if name in order else len(order), name)


def _cost_category_row(
    name: str,
    row: dict[str, Any],
    *,
    show_cache_writes: bool,
) -> str:
    label = _COST_CATEGORY_LABELS.get(name, name.replace("_", " ").title())
    total_usd = _format_usd(row.get("total_usd"))
    cost = f"${total_usd}" if row.get("pricing_complete", True) else f"≥ ${total_usd}"
    cache_write_cell = (
        f" {row.get('cache_write_tokens', 0)} |" if show_cache_writes else ""
    )
    return (
        f"| {label} | {row.get('calls', 0)} | {row.get('input_tokens', 0)} | "
        f"{row.get('cached_input_tokens', 0)} |{cache_write_cell} "
        f"{row.get('output_tokens', 0)} | {cost} |"
    )


def _interpretation_lines(interpretation: dict[str, Any]) -> list[str]:
    source = interpretation.get("source", "deterministic")
    model = interpretation.get("model")
    label = f"{source}" if model is None else f"{source}: `{model}`"
    lines = [
        "## Interpretation",
        "",
        f"_Source: {label}_",
        "",
        str(interpretation.get("executive_summary") or ""),
        "",
        "### Key Findings",
        "",
        *_bullets(interpretation.get("key_findings", [])),
        "",
        "### Recommended Actions",
        "",
        *_bullets(interpretation.get("recommended_actions", [])),
    ]
    if interpretation.get("limitations"):
        lines.extend(["", "### Limitations", "", *_bullets(interpretation["limitations"])])
    if interpretation.get("follow_up_experiments"):
        lines.extend(
            [
                "",
                "### Follow-Up Experiments",
                "",
                *_bullets(interpretation["follow_up_experiments"]),
            ]
        )
    if interpretation.get("llm_error"):
        lines.extend(["", f"- LLM report fallback: `{interpretation['llm_error']}`"])
    return lines


def _evidence_quality_lines(evidence: dict[str, Any]) -> list[str]:
    quality = evidence["evidence_quality"]
    validation = quality.get("validation", {})
    families = quality.get("perturbation_families", {})
    family_summary = ", ".join(
        f"`{family}` {count}" for family, count in families.items() if family
    ) or "none"
    validation_summary = ", ".join(
        f"`{status}` {count}" for status, count in validation.items()
    ) or "none"
    lines = [
        "## Evidence Quality",
        "",
        f"- Parse errors: `{quality.get('parse_errors', 0)}`",
        f"- Judge errors against gold labels: `{quality.get('judge_errors', quality.get('gold_disagreements', 0))}`",
        f"- Generated perturbations: `{quality.get('generated_perturbations', 0)}`",
        f"- Generation skips: `{quality.get('generation_skips', 0)}`",
        f"- Generation failures: `{quality.get('generation_failures', 0)}`",
        f"- Baseline exclusions / application skips / failures: "
        f"`{quality.get('application_exclusions', 0)}` / "
        f"`{quality.get('application_generation_skips', 0)}` / "
        f"`{quality.get('application_generation_failures', 0)}`",
        f"- Validation results: {validation_summary}",
        f"- Perturbation coverage by family: {family_summary}",
    ]
    if quality.get("replay_directional_eligibility_caveat"):
        lines.extend(
            [
                "",
                f"- Replay provenance: source run `{quality.get('replay_source_run_id') or 'not recorded'}`; "
                "generation/validation calls are historical, not new replay cost.",
                f"- Frozen eligibility caveat: {quality['replay_directional_eligibility_caveat']}",
            ]
        )
    lines.append("")
    return lines


def _example_lines(examples: list[dict[str, Any]]) -> list[str]:
    if not examples:
        return [
            "## Representative Perturbation Type Examples",
            "",
            "No perturbation examples were recorded.",
            "",
        ]
    lines = ["## Representative Perturbation Type Examples", ""]
    for example in examples:
        score_delta = (
            "n/a"
            if example.get("score_delta") is None
            else f"{example['score_delta']:+.3f}"
        )
        type_flip_rate = example.get("type_flip_rate")
        type_results = example.get("type_num_results")
        lines.extend(
            [
                f"### `{example['family']}` / `{example['operator']}`",
                "",
                f"- Selected example item: `{example['item_id']}`",
                f"- Type flip rate: `{_format_optional(type_flip_rate)}` over "
                f"`{type_results or 0}` judge calls",
                f"- Expected effect: `{example.get('expected_effect') or 'n/a'}`",
                f"- Changed field: `{example.get('changed_field') or 'n/a'}`",
                f"- Verdict: `{example.get('baseline_label')}` -> `{example.get('perturbed_label')}`",
                f"- Score delta: `{score_delta}`",
            ]
        )
        if example.get("summary"):
            lines.append(f"- Perturbation: {example['summary']}")
        if example.get("before") is not None or example.get("after") is not None:
            lines.extend(
                [
                    "",
                    "Before:",
                    "",
                    "```text",
                    _format_block(example.get("before")),
                    "```",
                    "",
                    "After:",
                    "",
                    "```text",
                    _format_block(example.get("after")),
                    "```",
                    "",
                ]
            )
    return lines


def _bullets(values: list[Any]) -> list[str]:
    return [f"- {value}" for value in values] or ["- No specific findings."]


def _format_block(value: Any) -> str:
    if value is None:
        return "n/a"
    return str(value)


def _format_optional(value: float | None) -> str:
    if value is None:
        return "not available"
    return f"{value:.3f}"


def _format_signed_optional(value: float | None) -> str:
    if value is None:
        return "not available"
    return f"{value:+.3f}"


def _verdict_cell(score: float | None, label: str | None) -> str:
    if score is not None:
        return f"{score:.3f}"
    return f"`{label}`" if label else "not available"


def _directional_noise_cell(row: dict[str, Any]) -> str:
    if row.get("detection_basis") == "label":
        return _format_optional(row.get("noise_flip_rate"))
    return _format_optional(row.get("noise_score_sd"))


def _format_usd(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value}"


def _format_changed_fields(fields: Any) -> str:
    if not fields:
        return "n/a"
    if isinstance(fields, list):
        return ", ".join(f"`{field}`" for field in fields)
    return f"`{fields}`"


def _format_ci(ci: dict[str, Any] | None) -> str:
    if ci is None or any(ci.get(key) is None for key in ("mean", "lo", "hi", "n")):
        return "`not available`"
    return f"`{ci['mean']:.3f}` [`{ci['lo']:.3f}`, `{ci['hi']:.3f}`] (n=`{ci['n']}`)"
