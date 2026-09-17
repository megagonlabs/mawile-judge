from __future__ import annotations

import json
from typing import Any, Protocol

from mawile.llm import (
    chat_message_text,
    create_chat_completion,
    make_llm_call_record,
)
from mawile.perturbations.base import parse_json_object
from mawile.providers import build_provider_client, provider_has_api_key
from mawile.reporting.evidence import build_report_evidence
from mawile.schemas import AuditRunConfig


class ReportInterpreter(Protocol):
    def interpret(self, evidence: dict[str, Any]) -> dict[str, Any]:
        ...


class OpenAIReportInterpreter:
    """Writes an evidence-grounded narrative for a MAWILE report.

    The model is an interpreter only: metrics are computed before this class runs,
    and the instructions forbid inventing findings that are absent from evidence.
    """

    def __init__(
        self,
        client: Any,
        model: str,
        decoding_params: dict[str, Any] | None = None,
        provider: str = "openai",
    ) -> None:
        self._client = client
        self._model = model
        self._provider = provider
        self._decoding = decoding_params or {}
        self.last_call: dict[str, Any] | None = None

    def interpret(self, evidence: dict[str, Any]) -> dict[str, Any]:
        call_record = make_llm_call_record(
            "reporting",
            self._provider,
            self._model,
        )
        self.last_call = call_record
        try:
            response = create_chat_completion(
                self._client,
                model=self._model,
                instructions=_instructions(),
                input_text=json.dumps(evidence, ensure_ascii=False, sort_keys=True),
                decoding_params=self._decoding,
            )
        except Exception as exc:
            call_record.update(
                make_llm_call_record(
                    "reporting",
                    self._provider,
                    self._model,
                    status="error",
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            raise
        call_record.update(
            make_llm_call_record(
                "reporting",
                self._provider,
                self._model,
                response,
                status="ok",
            )
        )
        payload = parse_json_object(_message_text(response))
        if payload is None:
            call_record.update(
                status="parse_error",
                error="report interpreter returned no parseable JSON object",
            )
            raise ValueError("report interpreter returned no parseable JSON object")
        return _normalize_interpretation(
            payload,
            source="llm",
            model=self._model,
            provider=self._provider,
            llm_call=call_record,
        )


def build_report_interpretation(
    config: AuditRunConfig,
    artifact: dict[str, Any],
    judge_call_traces: list[dict[str, Any]] | None = None,
    *,
    interpreter: ReportInterpreter | None = None,
) -> dict[str, Any]:
    evidence = build_report_evidence(config, artifact, judge_call_traces)
    fallback = deterministic_report_interpretation(evidence)
    selected = interpreter or _default_interpreter(config)
    if selected is None:
        return fallback
    try:
        return selected.interpret(evidence)
    except Exception as exc:
        fallback["llm_error"] = f"{type(exc).__name__}: {exc}"
        if (last_call := getattr(selected, "last_call", None)) is not None:
            fallback["llm_call"] = last_call
        return fallback


def deterministic_report_interpretation(evidence: dict[str, Any]) -> dict[str, Any]:
    metrics = evidence.get("headline_metrics", {})
    quality = evidence.get("evidence_quality", {})
    top_routes = evidence.get("top_routes", [])
    attribution = evidence.get("top_attribution", [])
    perturbation_types = evidence.get("top_perturbation_types", [])
    sample = evidence.get("sample", {})
    judge = evidence.get("judge", {})
    directional_metrics = evidence.get("directional") or {}
    equivariant_metrics = evidence.get("equivariant") or {}
    coverage = evidence.get("coverage") or {}

    raw_mean_flip = metrics.get("mean_flip_risk")
    mean_flip = float(raw_mean_flip or 0.0)
    raw_mean_noise = metrics.get("mean_noise_flip_rate")
    mean_noise = float(raw_mean_noise or 0.0)
    drift = _strongest_invariance_score_drift(attribution)
    sensitive_routes = [
        route for route in top_routes if "perturbation-sensitive" in route.get("reasons", [])
    ]
    noisy_routes = [
        route for route in top_routes if "noisy baseline" in route.get("reasons", [])
    ]
    boundary_routes = [
        route for route in top_routes if "near decision boundary" in route.get("reasons", [])
    ]
    parse_errors = int(quality.get("parse_errors") or 0)
    judge_errors = int(quality.get("judge_errors", quality.get("gold_disagreements")) or 0)

    if raw_mean_flip is None:
        summary = (
            "Invariant stability is unavailable because this run produced no "
            "analyzable invariant comparisons."
        )
    elif drift is not None:
        summary = (
            f"The strongest signal is invariance score drift in {drift['family']}: "
            f"meaning-preserving changes moved scores by "
            f"{drift['signed_shift']:+.2f} on average."
        )
    elif sensitive_routes or mean_flip > 0:
        summary = (
            "The run found perturbation sensitivity under invariant changes; "
            f"mean flip risk is {mean_flip:.3f}."
        )
    elif mean_noise > 0:
        summary = (
            "The main reliability issue is baseline repeat noise rather than a "
            f"specific perturbation family; mean noise flip rate is {mean_noise:.3f}."
        )
    else:
        summary = (
            "The judge was stable under the tested invariance perturbations. "
            "Any review signal is coming from calibration checks such as boundary "
            "proximity or judge errors against gold labels, not observed perturbation flips."
        )

    findings = [
        "Mean invariant flip risk: "
        + (f"{mean_flip:.3f}" if raw_mean_flip is not None else "not available")
        + "; mean noise flip rate: "
        + (f"{mean_noise:.3f}" if raw_mean_noise is not None else "not available")
        + ".",
        f"Coverage: {sample.get('items', 0)} items, {sample.get('perturbations', 0)} perturbations, "
        f"{sample.get('repeats', 0)} repeats.",
    ]
    if attribution and attribution[0].get("flip_rate") is not None:
        top = attribution[0]
        findings.append(
            f"Top attribution family: {top['family']} with flip rate "
            f"{float(top['flip_rate']):.3f} and signed shift "
            + (
                f"{float(top['signed_shift']):+.3f}."
                if top.get("signed_shift") is not None
                else "not available."
            )
        )
    if perturbation_types and perturbation_types[0].get("flip_rate") is not None:
        top_type = perturbation_types[0]
        findings.append(
            f"Top perturbation type: {top_type['operator']} with flip rate "
            f"{float(top_type['flip_rate']):.3f} across "
            f"{int(top_type.get('num_results') or 0)} judge calls."
        )
    if directional_metrics.get("has_directional"):
        if directional_metrics.get("degradation_detection_rate") is not None:
            basis = directional_metrics.get("detection_basis")
            findings.append(
                "Directional degradation detection rate: "
                f"{float(directional_metrics.get('degradation_detection_rate') or 0.0):.3f}; "
                "contradiction rate: "
                f"{float(directional_metrics.get('contradiction_rate') or 0.0):.3f} "
                f"({_basis_phrase(basis)})."
            )
        else:
            findings.append(
                "Directional degradations were run, but no case produced an analyzable "
                "baseline and perturbed verdict."
            )
    if equivariant_metrics.get("has_equivariant"):
        findings.append(
            "Pairwise candidate-identity consistency after position swap: "
            f"{float(equivariant_metrics.get('candidate_identity_consistency_rate') or 0.0):.3f}; "
            "first-position selection rate: "
            f"{float(equivariant_metrics.get('first_position_selection_rate') or 0.0):.3f}."
        )
    if parse_errors:
        findings.append(f"{parse_errors} judge calls failed to parse.")
    correctness = evidence.get("correctness") or {}
    if correctness.get("has_gold_labels"):
        findings.append(
            f"Gold-label accuracy: {float(correctness.get('accuracy') or 0.0):.3f} "
            f"over {int(correctness.get('labeled_items') or 0)} labeled item(s)."
        )
    if judge_errors:
        items = ", ".join(quality.get("judge_error_items", quality.get("gold_disagreement_items", [])))
        findings.append(f"{judge_errors} judge error(s) against gold labels: {items}.")
    if boundary_routes and not sensitive_routes and not noisy_routes:
        findings.append(
            "The top review items are near the decision boundary, but did not show "
            "perturbation flips."
        )

    actions: list[str] = []
    if drift is not None:
        actions.append(
            f"Calibrate {drift['family']} behavior with anchor examples before "
            "depending on affected verdicts."
        )
    if (
        directional_metrics.get("has_directional")
        and directional_metrics.get("degradation_detection_rate") is not None
    ):
        contradiction_rate = float(directional_metrics.get("contradiction_rate") or 0.0)
        detection_rate = float(directional_metrics.get("degradation_detection_rate") or 0.0)
        if contradiction_rate > 0 or detection_rate < 1.0:
            actions.append(
                "Inspect missed directional degradations before trusting the judge's "
                "ability to discriminate worse outputs."
            )
    if sensitive_routes:
        actions.append(
            "Inspect the perturbation-sensitive review items and the top attribution "
            "families before changing the rubric."
        )
    if noisy_routes or mean_noise > 0:
        actions.append(
            "Increase repeats or stabilize decoding before interpreting perturbation "
            "differences as judge bias."
        )
    if parse_errors:
        actions.append("Fix parser or output-schema failures before trusting aggregate rates.")
    if equivariant_metrics.get("has_equivariant") and float(
        equivariant_metrics.get("candidate_identity_consistency_rate") or 0.0
    ) < 1.0:
        actions.append(
            "Inspect position-swap inconsistencies before trusting pairwise preferences."
        )
    if judge_errors:
        actions.append(
            "Review judge-error cases for rubric calibration or targeted repair."
        )
    if boundary_routes and not sensitive_routes:
        actions.append(
            "Use near-boundary items for spot review or threshold calibration; do not "
            "treat them as perturbation failures unless flips appear."
        )
    if not actions:
        actions.append(
            "Keep this configuration as a stable baseline and expand coverage with "
            "more items or LLM-backed perturbations."
        )

    limitations = []
    if judge.get("model") == "mock":
        limitations.append("This run used the mock judge, so cost and model behavior are local test signals.")
    if int(sample.get("items") or 0) < 30:
        limitations.append("The item sample is small, so confidence intervals and family rankings are exploratory.")
    if int(quality.get("generated_perturbations") or 0) == 0:
        limitations.append("No LLM-backed perturbations were generated in this run.")
    if not quality.get("validation"):
        limitations.append("No generated perturbations required validation.")
    if directional_metrics.get("has_directional") and not directional_metrics.get("score_supported"):
        limitations.append(
            "Directional detection compares canonical classes, so signed score shift "
            "and threshold-crossing metrics are unavailable for this judge."
        )
    if float(coverage.get("analyzable_rate") or 1.0) < 0.95:
        limitations.append(
            "More than 5% of judge attempts were not analyzable, so headline comparisons are blocked."
        )

    follow_ups = [
        "Run the same configuration on a larger item sample.",
        "Enable LLM-backed perturbations with an independent perturbation model.",
    ]
    if boundary_routes:
        follow_ups.append("Add targeted boundary cases around the judge threshold.")
    if judge_errors:
        follow_ups.append("Inspect judge-error cases alongside their perturbation traces.")

    return {
        "source": "deterministic",
        "model": None,
        "provider": None,
        "executive_summary": summary,
        "key_findings": findings,
        "recommended_actions": actions,
        "limitations": limitations,
        "follow_up_experiments": follow_ups,
    }


def _default_interpreter(config: AuditRunConfig) -> ReportInterpreter | None:
    summary_model = config.resolved_summary_model()
    summary_provider = config.resolved_summary_provider()
    if summary_model == "mock" or not provider_has_api_key(config, summary_provider):
        return None
    return OpenAIReportInterpreter(
        build_provider_client(config, summary_provider),
        summary_model,
        config.resolved_summary_decoding_params(),
        provider=summary_provider,
    )


def _instructions() -> str:
    return (
        "You are writing the interpretation section for a MAWILE audit report. "
        "MAWILE audits an LLM judge by repeating original judge calls, applying "
        "meaning-preserving perturbations, applying optional directional degradations, "
        "and measuring label flips, score shifts, noise, invariant flip risk, "
        "gold-label correctness when labels are present, directional degradation "
        "detection, validation quality, and concrete examples.\n\n"
        "Base every claim on the supplied JSON evidence. Do not recompute metrics, "
        "invent families, or overstate causality. If an item is listed only because "
        "it is near the decision boundary, describe it as a calibration or spot-review "
        "candidate, not as perturbation instability. If no invariant flips are "
        "present, do not recommend changing the rubric solely because calibration "
        "signals are present.\n\n"
        "Return one JSON object with these exact keys: "
        "\"executive_summary\" (short paragraph), \"key_findings\" (array of concise "
        "strings), \"recommended_actions\" (array of concrete strings), \"limitations\" "
        "(array of strings), and \"follow_up_experiments\" (array of strings)."
    )


def _normalize_interpretation(
    payload: dict[str, Any],
    *,
    source: str,
    model: str | None,
    provider: str | None,
    llm_call: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "source": source,
        "model": model,
        "provider": provider,
        "llm_call": llm_call,
        "executive_summary": str(payload.get("executive_summary") or "").strip(),
        "key_findings": _string_list(payload.get("key_findings")),
        "recommended_actions": _string_list(payload.get("recommended_actions")),
        "limitations": _string_list(payload.get("limitations")),
        "follow_up_experiments": _string_list(payload.get("follow_up_experiments")),
    }


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _message_text(response: Any) -> str:
    return chat_message_text(response)


def _basis_phrase(basis: str | None) -> str:
    if basis == "score":
        return "score-based"
    if basis == "label":
        return "class-based"
    if basis == "mixed":
        return "score- and class-based"
    return "no analyzable basis"


def _strongest_invariance_score_drift(attribution: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        row
        for row in attribution
        if abs(float(row.get("signed_shift") or 0.0)) >= 0.5
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda row: abs(float(row.get("signed_shift") or 0.0)))
