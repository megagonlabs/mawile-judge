from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from mawile.llm import (
    chat_message_text,
    create_chat_completion,
    make_llm_call_record,
)
from mawile.perturbations.base import (
    DimensionSpec,
    parse_json_object,
)
from mawile.perturbations.planning.context import (
    SuggestionContext,
    build_audit_context,
    build_suggestion_context,
)
from mawile.perturbations.registry import all_dimensions
from mawile.providers import provider_has_api_key
from mawile.schemas import AuditRunConfig


class PlannerError(RuntimeError):
    """Raised when the planner cannot produce a usable operator selection.

    The planner replaces hand-routing, so a silent fallback (e.g. to the whole
    catalog) would hide a misrouted audit. Any unusable result aborts the run.
    """


@dataclass(frozen=True)
class PlanResult:
    operators: list[str]
    rationale: str
    provider: str | None = None
    model: str | None = None
    call_record: dict[str, Any] | None = None
    messages: list[dict[str, str]] = field(default_factory=list)
    context_metadata: dict[str, Any] = field(default_factory=dict)


class PerturbationPlanner(Protocol):
    """Selects the operator allow-list from the judge config with one LLM call.

    Like the perturbation agent, the planner must be independent of the judge under
    test -- routing a judge's probes with its own model is circular.
    """

    def plan(self, config: AuditRunConfig) -> PlanResult:
        ...


class OpenAIPlannerAgent:
    """Asks an independent model which operators suit the judge under test.

    Invariant operators hold meaning fixed, so they are appropriate only when
    their touched aspect is orthogonal to what the judge grades. Directional
    operators intentionally degrade the item, so they are appropriate when they
    exercise a failure mode the judge should catch.
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

    def plan(self, config: AuditRunConfig) -> PlanResult:
        candidates = _candidate_specs(config)
        catalog = {spec.operator: spec for spec in candidates}
        context = build_suggestion_context(config)
        instructions = _build_instructions(candidates, config)
        input_text = _build_input(config, context)
        messages = _prompt_messages(instructions, input_text)
        output, call_record = self._complete(instructions, input_text)
        payload = parse_json_object(output)
        if payload is None:
            call_record.update(
                status="parse_error",
                error="planner returned no parseable JSON object",
            )
            raise PlannerError("planner returned no parseable JSON object")
        raw = payload.get("operators")
        if not isinstance(raw, list):
            call_record.update(
                status="parse_error",
                error="planner response had no 'operators' list",
            )
            raise PlannerError("planner response had no 'operators' list")
        non_strings = [name for name in raw if not isinstance(name, str)]
        unknown = [name for name in raw if isinstance(name, str) and name not in catalog]
        if non_strings or unknown:
            detail = []
            if unknown:
                detail.append("unknown operators: " + ", ".join(sorted(set(unknown))))
            if non_strings:
                detail.append("operator names must be strings")
            message = "planner selected invalid operators (" + "; ".join(detail) + ")"
            call_record.update(status="parse_error", error=message)
            raise PlannerError(message)
        # An intentional empty plan is valid: it requests a baseline-only audit.
        selected = list(dict.fromkeys(raw))
        return PlanResult(
            operators=selected,
            rationale=str(payload.get("rationale") or ""),
            provider=self._provider,
            model=self._model,
            call_record=call_record,
            messages=messages,
            context_metadata=context.metadata,
        )

    def _complete(self, instructions: str, input_text: str) -> tuple[str, dict[str, Any]]:
        call_record = make_llm_call_record(
            "planning",
            self._provider,
            self._model,
        )
        try:
            response = create_chat_completion(
                self._client,
                model=self._model,
                instructions=instructions,
                input_text=input_text,
                decoding_params=self._decoding,
            )
        except Exception as exc:  # A failed planner call aborts -- never guess a list.
            call_record.update(
                make_llm_call_record(
                    "planning",
                    self._provider,
                    self._model,
                    status="error",
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            raise PlannerError(f"planner model call failed: {exc}") from exc
        call_record.update(
            make_llm_call_record(
                "planning",
                self._provider,
                self._model,
                response,
                status="ok",
            )
        )
        output = _message_text(response)
        if not output:
            call_record.update(
                status="invalid_response",
                error="planner model returned empty output",
            )
            raise PlannerError(f"planner model returned empty output{_empty_output_detail(response)}")
        return output, call_record


def _candidate_specs(config: AuditRunConfig) -> list[DimensionSpec]:
    """Every operator that fits the judge -- applicability gates the rest, 
    so the model is never offered an operator the pipeline would silently drop."""

    return all_dimensions(config)


def _build_instructions(candidates: list[DimensionSpec], config: AuditRunConfig) -> str:
    catalog = "\n".join(operator_catalog_line(spec, config) for spec in candidates)
    return (
        "You are routing perturbation probes for a MAWILE audit of an LLM judge.\n\n"
        "Invariant operators have expected_effect=same_verdict: they rewrite part "
        "of the judge prompt or transcript while holding meaning fixed, so a reliable "
        "judge's verdict should not move. Keep an invariant operator only when the "
        "aspect it changes (its `touches` tag) is orthogonal to what this judge "
        "evaluates. Drop invariant operators whose touched aspect overlaps the "
        "judge's grading target.\n\n"
        + (
            "Directional operators have expected_effect=worse_verdict: they intentionally "
            "make the item worse, so a reliable judge should produce a worse verdict. "
            "Keep directional operators when the degradation represents a failure mode "
            "this judge should detect.\n\n"
            if config.judge.output_type.value != "pairwise"
            else "Pairwise audits use invariant probes only; do not select directional operators.\n\n"
        )
        + "Selection budget: choose only a targeted set of relevant probes. An empty "
        "list is valid when baseline-only measurement is appropriate.\n\n"
        "Candidate operators (one JSON object per line):\n"
        f"{catalog}\n\n"
        "Select the operators appropriate for the judge described below. Return a JSON "
        "object with two keys: \"operators\" (the selected operator names, drawn only from "
        "the candidates above) and \"rationale\" (a brief explanation of what the judge "
        "grades and why you kept or dropped the borderline operators)."
    )


def _build_input(config: AuditRunConfig, context: SuggestionContext) -> str:
    return (
        "JUDGE AND AUDIT CONFIGURATION:\n"
        f"{build_audit_context(config)}\n\n"
        "DATASET CONTEXT:\n"
        f"{context.text}"
    )


def operator_catalog_line(spec: DimensionSpec, config: AuditRunConfig) -> str:
    runnable, availability = _operator_availability(spec, config)
    payload = {
        "operator": spec.operator,
        "dimension": spec.dimension,
        "family": spec.family.value,
        "target": spec.target,
        "applies_to": spec.applies_to,
        "applicability": (
            f"applicable to the current {config.judge.output_type.value} judge configuration"
        ),
        "kind": spec.kind,
        "generation_kind": "deterministic" if spec.kind == "det" else "llm",
        "expected_effect": spec.default_effect.value,
        "expected_relation": spec.expected_relation.value,
        "touches": list(spec.touches),
        "description": " ".join((spec.description or "").split()),
        "operation": _operation_description(spec),
        "validation_kind": spec.validation_kind,
        "validation_method": spec.validation_kind or "deterministic construction",
        "limitations": list(spec.limitations),
        "runnable": runnable,
        "availability": availability,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _operation_description(spec: DimensionSpec) -> str:
    if spec.instruction:
        return " ".join(spec.instruction.split())
    if spec.builder is not None:
        return f"{spec.builder.__module__}.{spec.builder.__qualname__}"
    return " ".join((spec.description or "").split())


def _operator_availability(
    spec: DimensionSpec,
    config: AuditRunConfig,
) -> tuple[bool, str]:
    if spec.kind == "det":
        return True, "deterministic operator"
    if config.perturbation_agent.model == "mock":
        return False, "perturbation_agent.model is mock"
    if not provider_has_api_key(config, config.perturbation_agent.provider):
        return False, "perturbation-agent provider key is not configured"
    return True, "LLM perturbation generation and validation are configured"


def _prompt_messages(instructions: str, input_text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": instructions},
        {"role": "user", "content": input_text},
    ]


def _message_text(response: Any) -> str:
    """The assistant's message text, excluding provider reasoning fields."""

    return chat_message_text(response)


def _empty_output_detail(response: Any) -> str:
    """Finish/usage context for an empty response, so exhaustion is legible.

    A reasoning model counts its reasoning tokens against ``max_completion_tokens``; if
    it spends the whole budget thinking, the call completes with no message text.
    """

    bits: list[str] = []
    choices = getattr(response, "choices", None) or []
    finish_reason = getattr(choices[0], "finish_reason", None) if choices else None
    if finish_reason:
        bits.append(f"finish_reason={finish_reason}")
    usage = getattr(response, "usage", None)
    if usage is not None:
        completion_tokens = getattr(usage, "completion_tokens", None)
        if completion_tokens is not None:
            bits.append(f"completion_tokens={completion_tokens}")
        details = getattr(usage, "completion_tokens_details", None)
        reasoning_tokens = getattr(details, "reasoning_tokens", None) if details is not None else None
        if reasoning_tokens is not None:
            bits.append(f"reasoning_tokens={reasoning_tokens}")
    suffix = f" ({', '.join(bits)})" if bits else ""
    if finish_reason == "length":
        suffix += (
            " -- the completion token limit was reached before any message text; "
            "raise planner_agent.decoding_params.max_completion_tokens"
        )
    return suffix
