from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

from mawile.config import resolve_planned_operator_selection
from mawile.config_payload import build_config_from_payload
from mawile.llm import chat_message_text, create_chat_completion
from mawile.perturbations.base import parse_json_object
from mawile.perturbations.planning import (
    OpenAIPlannerAgent,
    PerturbationPlanner,
    build_suggestion_context,
    operator_catalog_line,
)
from mawile.perturbations.planning.context import build_audit_context
from mawile.perturbations.registry import DIMENSIONS, DIRECTIONAL_DIMENSIONS, all_dimensions
from mawile.providers import build_provider_client
from mawile.schemas import AuditRunConfig, PerturbationExpectedEffect


@dataclass(frozen=True)
class PlannedOperatorSelection:
    selected: list[str]
    manual: list[str]
    effective: list[str]
    selected_directional: list[str]
    manual_directional: list[str]
    effective_directional: list[str]
    rationale: str
    messages: list[dict[str, str]] = field(default_factory=list)
    context_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CustomPerturbationSuggestion:
    name: str
    target: str
    instruction: str
    expected_effect: str
    rationale: str = ""
    messages: list[dict[str, str]] = field(default_factory=list)
    context_metadata: dict[str, Any] = field(default_factory=dict)


class CustomPerturbationSuggester(Protocol):
    def suggest(
        self,
        config: AuditRunConfig,
        *,
        target: str,
        expected_effect: str,
    ) -> CustomPerturbationSuggestion:
        ...


class OpenAICustomPerturbationSuggester:
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

    def suggest(
        self,
        config: AuditRunConfig,
        *,
        target: str,
        expected_effect: str,
    ) -> CustomPerturbationSuggestion:
        context = build_suggestion_context(config)
        instructions = _custom_suggestion_instructions()
        input_text = _custom_suggestion_input(
            config,
            target=target,
            expected_effect=expected_effect,
            dataset_context=context.text,
        )
        output = self._complete(instructions, input_text)
        payload = parse_json_object(output)
        if payload is None:
            raise ValueError("custom perturbation suggester returned no parseable JSON object")
        suggestion = _custom_suggestion_from_payload(
            payload,
            requested_target=target,
            requested_effect=expected_effect,
        )
        return replace(
            suggestion,
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": input_text},
            ],
            context_metadata=context.metadata,
        )

    def _complete(self, instructions: str, input_text: str) -> str:
        try:
            response = create_chat_completion(
                self._client,
                model=self._model,
                instructions=instructions,
                input_text=input_text,
                decoding_params=self._decoding,
            )
        except Exception as exc:
            raise ValueError(f"custom perturbation suggester model call failed: {exc}") from exc
        output = _message_text(response)
        if not output:
            raise ValueError("custom perturbation suggester model returned empty output")
        return output


def plan_perturbation_operators(
    payload: dict[str, Any],
    base_dir: Path,
    *,
    planner: PerturbationPlanner | None = None,
) -> tuple[dict[str, Any], PlannedOperatorSelection]:
    prepared = copy.deepcopy(payload)
    config = build_config_from_payload(prepared, base_dir)
    planner = planner or _default_planner(config)

    result = planner.plan(config)
    known = {spec.operator for spec in DIMENSIONS}
    known_directional = {spec.operator for spec in DIRECTIONAL_DIMENSIONS}
    manual = [
        operator
        for operator in config.audit.perturbation_operators
        if operator in known
    ]
    manual_directional = [
        operator
        for operator in config.audit.directional_perturbation_operators
        if operator in known_directional
    ]
    effective, effective_directional = resolve_planned_operator_selection(
        config, result.operators
    )
    selected = list(effective)
    selected_directional = list(effective_directional)

    audit_payload = prepared.setdefault("audit", {})
    audit_payload["perturbation_operators"] = effective
    audit_payload["directional_perturbation_operators"] = effective_directional
    audit_payload["plan_operators"] = False

    return prepared, PlannedOperatorSelection(
        selected=selected,
        manual=manual,
        effective=effective,
        selected_directional=selected_directional,
        manual_directional=manual_directional,
        effective_directional=effective_directional,
        rationale=result.rationale,
        messages=result.messages,
        context_metadata=result.context_metadata,
    )


def suggest_custom_perturbation(
    payload: dict[str, Any],
    base_dir: Path,
    *,
    target: str,
    expected_effect: str,
    suggester: CustomPerturbationSuggester | None = None,
) -> CustomPerturbationSuggestion:
    if target not in _CUSTOM_SUGGESTION_TARGETS:
        raise ValueError("custom perturbation target is not supported")
    if expected_effect not in _CUSTOM_SUGGESTION_EFFECTS:
        raise ValueError("custom perturbation relationship is not supported")
    config = build_config_from_payload(payload, base_dir)
    suggester = suggester or _default_custom_suggester(config)
    return suggester.suggest(config, target=target, expected_effect=expected_effect)


def _default_planner(config: AuditRunConfig) -> PerturbationPlanner:
    planner_model = config.resolved_planner_model()
    planner_provider = config.resolved_planner_provider()
    if planner_model == "mock":
        raise ValueError(
            "Set planner_agent.model or perturbation_agent.model to a non-mock model "
            "before generating perturbation suggestions."
        )
    return OpenAIPlannerAgent(
        build_provider_client(config, planner_provider),
        planner_model,
        config.resolved_planner_decoding_params(),
        provider=planner_provider,
    )


def _default_custom_suggester(config: AuditRunConfig) -> CustomPerturbationSuggester:
    suggester_model = config.resolved_planner_model()
    suggester_provider = config.resolved_planner_provider()
    if suggester_model == "mock":
        raise ValueError(
            "Set planner_agent.model or perturbation_agent.model to a non-mock model "
            "before suggesting a custom perturbation."
        )
    return OpenAICustomPerturbationSuggester(
        build_provider_client(config, suggester_provider),
        suggester_model,
        config.resolved_planner_decoding_params(),
        provider=suggester_provider,
    )


_CUSTOM_SUGGESTION_TARGETS = {
    "judge.prompt_template",
    "judge.rubric",
    "item.input",
    "item.output",
}
_CUSTOM_SUGGESTION_EFFECTS = {
    PerturbationExpectedEffect.SAME_VERDICT.value,
    PerturbationExpectedEffect.WORSE_VERDICT.value,
}


def _custom_suggestion_instructions() -> str:
    return (
        "You suggest one custom perturbation for a MAWILE audit of an LLM judge. "
        "The suggestion must complement the built-in perturbation catalog and be "
        "specific enough for a separate perturbation agent to execute. Return only "
        "one JSON object with keys: name, target, expected_effect, instruction, and "
        "rationale. The target and expected_effect must exactly match the requested "
        "values. For expected_effect=same_verdict, the instruction must preserve the "
        "judge-relevant meaning so a reliable judge should keep the same verdict. "
        "For expected_effect=worse_verdict, the instruction must intentionally make "
        "the item or judge instrument worse in a way this judge should penalize."
    )


def _custom_suggestion_input(
    config: AuditRunConfig,
    *,
    target: str,
    expected_effect: str,
    dataset_context: str,
) -> str:
    built_ins = "\n".join(
        operator_catalog_line(spec, config)
        for spec in all_dimensions(config)
    )
    relationship = (
        "meaning-preserving invariant"
        if expected_effect == PerturbationExpectedEffect.SAME_VERDICT.value
        else "directional degradation"
    )
    return (
        f"Requested custom perturbation:\n"
        f"- target: {target}\n"
        f"- expected_effect: {expected_effect} ({relationship})\n\n"
        "JUDGE AND AUDIT CONFIGURATION:\n"
        f"{build_audit_context(config)}\n\n"
        f"Built-in operators to avoid duplicating:\n{built_ins}\n\n"
        "DATASET CONTEXT:\n"
        f"{dataset_context}"
    )


def _custom_suggestion_from_payload(
    payload: dict[str, Any],
    *,
    requested_target: str,
    requested_effect: str,
) -> CustomPerturbationSuggestion:
    target = str(payload.get("target") or requested_target).strip()
    expected_effect = str(payload.get("expected_effect") or requested_effect).strip()
    if requested_target not in _CUSTOM_SUGGESTION_TARGETS:
        raise ValueError("custom perturbation target is not supported")
    if requested_effect not in _CUSTOM_SUGGESTION_EFFECTS:
        raise ValueError("custom perturbation relationship is not supported")
    if target != requested_target:
        raise ValueError("custom perturbation suggester changed the requested target")
    if expected_effect != requested_effect:
        raise ValueError("custom perturbation suggester changed the requested relationship")
    name = str(payload.get("name") or "").strip()
    instruction = str(payload.get("instruction") or "").strip()
    if not name:
        raise ValueError("custom perturbation suggester returned no name")
    if not instruction:
        raise ValueError("custom perturbation suggester returned no instruction")
    return CustomPerturbationSuggestion(
        name=name,
        target=target,
        instruction=instruction,
        expected_effect=expected_effect,
        rationale=str(payload.get("rationale") or "").strip(),
    )


def _message_text(response: Any) -> str:
    return chat_message_text(response)


__all__ = [
    "CustomPerturbationSuggestion",
    "CustomPerturbationSuggester",
    "OpenAICustomPerturbationSuggester",
    "PlannedOperatorSelection",
    "plan_perturbation_operators",
    "suggest_custom_perturbation",
]
