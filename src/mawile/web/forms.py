from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import yaml

from mawile.perturbations.registry import DIRECTIONAL_DIMENSIONS, DIMENSIONS
from mawile.schemas import PerturbationExpectedEffect
from mawile.config_payload import parse_mapping_text

CUSTOM_TARGETS = {
    "judge.prompt_template",
    "judge.rubric",
    "item.input",
    "item.output",
}
CUSTOM_EXPECTED_EFFECTS = {
    PerturbationExpectedEffect.SAME_VERDICT.value,
    PerturbationExpectedEffect.WORSE_VERDICT.value,
}


class FormErrors(ValueError):
    def __init__(self, errors: list[str]) -> None:
        super().__init__("; ".join(errors))
        self.errors = errors


def payload_from_form(
    current_payload: dict[str, Any],
    form: Any,
) -> dict[str, Any]:
    data = _FormReader(form)
    payload = copy.deepcopy(current_payload)
    errors: list[str] = []

    judge = payload.setdefault("judge", {})
    data_payload = payload.setdefault("data", {})
    audit = payload.setdefault("audit", {})
    output = payload.setdefault("output", {})
    perturbation_agent = payload.setdefault("perturbation_agent", {})

    _set_text(
        judge,
        "provider",
        data.get("judge_provider", str(judge.get("provider") or "openai")),
    )
    _set_text(judge, "model", data.get("judge_model", ""))
    _set_text(judge, "output_type", data.get("output_type", "binary"))
    _set_text(judge, "prompt_template", data.get("prompt_template", ""))
    _set_text(judge, "rubric", data.get("rubric", ""))

    if data.checked("threshold_enabled"):
        judge["threshold"] = _parse_float(data.get("threshold", "3"), "Threshold", errors)
    else:
        judge["threshold"] = None
    _set_text(
        judge,
        "score_direction",
        data.get("score_direction", "higher_is_better"),
    )
    for key, label in (
        ("score_min", "Score minimum"),
        ("score_max", "Score maximum"),
        ("gold_tolerance", "Gold tolerance"),
    ):
        raw = data.get(key, "").strip()
        judge[key] = _parse_float(raw, label, errors) if raw else None
    judge["invariant_tolerance"] = _parse_float(
        data.get("invariant_tolerance", "0"),
        "Invariant tolerance",
        errors,
        minimum=0,
    )

    _parse_mapping_field(
        judge,
        "decoding_params",
        data.get("judge_decoding_params", ""),
        "Judge decoding parameters",
        errors,
        empty_value={},
    )
    _parse_mapping_field(
        judge,
        "output_schema",
        data.get("output_schema", ""),
        "Output schema",
        errors,
        empty_value=None,
    )

    _set_text(data_payload, "items_path", data.get("items_path", ""))
    context_path = data.get("context_path", "").strip()
    if context_path:
        data_payload["context_path"] = context_path
    else:
        data_payload.pop("context_path", None)
    _set_text(data_payload, "input_field", data.get("input_field", "input"))
    _set_text(data_payload, "output_field", data.get("output_field", "output"))
    if data.checked("gold_enabled"):
        gold_field = data.get("gold_field", "gold_label").strip() or "gold_label"
        data_payload["gold_field"] = gold_field
    else:
        data_payload["gold_field"] = None

    audit["repeats"] = _parse_int(data.get("repeats", "10"), "Repeats", errors, minimum=1)
    audit["num_workers"] = _parse_int(
        data.get("num_workers", "1"),
        "Parallel judge workers",
        errors,
        minimum=1,
    )
    audit["scalar_delta"] = _parse_float(
        data.get("scalar_delta", "1.0"),
        "Boundary width",
        errors,
        minimum=0.000001,
    )
    audit["top_k_routes"] = _parse_int(
        data.get("top_k_routes", "10"),
        "Review list size",
        errors,
        minimum=1,
    )
    _set_text(output, "runs_dir", data.get("runs_dir", "runs"))

    known_invariant = {spec.operator for spec in DIMENSIONS}
    selected_invariant = _known_values(
        data.getlist("perturbation_operators"),
        known_invariant,
    )
    # Empty is deliberately meaningful: it requests a baseline-only audit.
    audit["perturbation_operators"] = selected_invariant
    selected_directional = _known_values(
        data.getlist("directional_operators"),
        {spec.operator for spec in DIRECTIONAL_DIMENSIONS},
    )
    audit["directional_perturbation_operators"] = selected_directional
    if judge.get("output_type") == "pairwise" and selected_directional:
        errors.append("Pairwise audits do not support directional perturbations.")

    custom_rows = _custom_perturbations_from_form(
        data,
        errors,
        directional_allowed=judge.get("output_type") != "pairwise",
    )
    if custom_rows:
        audit["custom_perturbations"] = custom_rows
    else:
        audit.pop("custom_perturbations", None)

    _set_text(
        perturbation_agent,
        "provider",
        data.get(
            "perturbation_agent_provider",
            str(perturbation_agent.get("provider") or "openai"),
        ),
    )
    _set_text(perturbation_agent, "model", data.get("perturbation_agent_model", ""))
    _parse_mapping_field(
        perturbation_agent,
        "decoding_params",
        data.get("perturbation_decoding_params", ""),
        "Perturbation agent decoding parameters",
        errors,
        empty_value={},
    )
    _agent_override_from_form(
        payload,
        data,
        key="planner_agent",
        provider_field="planner_provider",
        model_field="planner_model",
        params_field="planner_decoding_params",
        params_label="Perturbation Suggester decoding parameters",
        errors=errors,
    )
    planner_context_max_chars = _parse_int(
        data.get("planner_dataset_context_max_chars", "40000"),
        "Perturbation Suggester dataset context limit",
        errors,
        minimum=1000,
    )
    planner_payload = payload.get("planner_agent")
    if planner_context_max_chars != 40000:
        if not isinstance(planner_payload, dict):
            planner_payload = {}
            payload["planner_agent"] = planner_payload
        planner_payload["dataset_context_max_chars"] = planner_context_max_chars
    elif isinstance(planner_payload, dict):
        planner_payload.pop("dataset_context_max_chars", None)
        _drop_empty_role_payload(payload, "planner_agent")
    audit["plan_operators"] = False

    _agent_override_from_form(
        payload,
        data,
        key="summary_agent",
        provider_field="summary_provider",
        model_field="summary_model",
        params_field="summary_decoding_params",
        params_label="Summary agent decoding parameters",
        errors=errors,
    )

    _agent_override_from_form(
        payload,
        data,
        key="validator_agent",
        provider_field="validator_provider",
        model_field="validator_model",
        params_field="validator_decoding_params",
        params_label="Validator decoding parameters",
        errors=errors,
    )

    if errors:
        raise FormErrors(errors)
    return payload


def _custom_perturbations_from_form(
    data: "_FormReader",
    errors: list[str],
    *,
    directional_allowed: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    count = _parse_int(data.get("custom_count", "0"), "Custom perturbation row count", errors, minimum=0)
    for index in range(count):
        name = data.get(f"custom_{index}_name", "").strip()
        target = data.get(f"custom_{index}_target", "item.input").strip()
        instruction = data.get(f"custom_{index}_instruction", "").strip()
        expected_effect = data.get(
            f"custom_{index}_expected_effect",
            PerturbationExpectedEffect.SAME_VERDICT.value,
        ).strip() or PerturbationExpectedEffect.SAME_VERDICT.value
        enabled = data.checked(f"custom_{index}_enabled")
        if not any([name, instruction]):
            continue
        if target not in CUSTOM_TARGETS:
            errors.append(f"Custom perturbation {index + 1}: unsupported target.")
            target = "item.input"
        if expected_effect not in CUSTOM_EXPECTED_EFFECTS:
            errors.append(f"Custom perturbation {index + 1}: unsupported relationship.")
            expected_effect = PerturbationExpectedEffect.SAME_VERDICT.value
        if (
            not directional_allowed
            and expected_effect == PerturbationExpectedEffect.WORSE_VERDICT.value
        ):
            errors.append(
                f"Custom perturbation {index + 1}: pairwise audits do not support directional perturbations."
            )
        rows.append(
            {
                "name": name,
                "target": target,
                "instruction": instruction,
                "expected_effect": expected_effect,
                "enabled": enabled,
            }
        )
    return rows


def _agent_override_from_form(
    payload: dict[str, Any],
    data: "_FormReader",
    *,
    key: str,
    provider_field: str,
    model_field: str,
    params_field: str,
    params_label: str,
    errors: list[str],
) -> None:
    existing = payload.get(key)
    role_payload = copy.deepcopy(existing) if isinstance(existing, dict) else {}
    provider_override = data.get(provider_field, "").strip() or None
    model_override = data.get(model_field, "").strip() or None
    params = _parse_agent_params(
        data.get(params_field, ""),
        params_label,
        errors,
    )
    # Blank means inherit; a literal YAML {} is an intentional empty override.
    params_override = params
    connection = {
        "provider": provider_override,
        "model": model_override,
        "decoding_params": params_override,
    }
    if not role_payload and all(value is None for value in connection.values()):
        payload.pop(key, None)
        return
    for field, value in connection.items():
        if value is not None or field in role_payload:
            role_payload[field] = value
    payload[key] = role_payload
    _drop_empty_role_payload(payload, key)


def _drop_empty_role_payload(payload: dict[str, Any], key: str) -> None:
    role_payload = payload.get(key)
    if not isinstance(role_payload, dict):
        return
    connection_fields = {"provider", "model", "decoding_params"}
    has_connection = any(role_payload.get(field) is not None for field in connection_fields)
    has_extra = any(field not in connection_fields for field in role_payload)
    if not has_connection and not has_extra:
        payload.pop(key, None)


def _parse_agent_params(raw: str, label: str, errors: list[str]) -> dict[str, Any] | None:
    try:
        parsed = parse_mapping_text(raw, empty_value=None)
    except Exception as exc:  # noqa: BLE001 - collected for form display.
        errors.append(f"{label}: {exc}")
        return None
    if parsed is None:
        return None
    if not isinstance(parsed, dict):
        errors.append(f"{label}: expected a YAML mapping.")
        return None
    return parsed


def _parse_mapping_field(
    payload: dict[str, Any],
    key: str,
    raw: str,
    label: str,
    errors: list[str],
    *,
    empty_value: Any,
) -> None:
    try:
        parsed = parse_mapping_text(raw, empty_value=empty_value)
    except Exception as exc:  # noqa: BLE001 - collected for form display.
        errors.append(f"{label}: {exc}")
        return
    if parsed is None or isinstance(parsed, str):
        payload[key] = parsed
        return
    if not isinstance(parsed, dict):
        errors.append(f"{label}: expected a YAML mapping.")
        return
    payload[key] = parsed


def _parse_int(raw: str, label: str, errors: list[str], *, minimum: int = 0) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        errors.append(f"{label}: expected an integer.")
        return minimum
    if value < minimum:
        errors.append(f"{label}: must be at least {minimum}.")
        return minimum
    return value


def _parse_float(
    raw: str,
    label: str,
    errors: list[str],
    *,
    minimum: float | None = None,
) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        errors.append(f"{label}: expected a number.")
        return minimum or 0.0
    if minimum is not None and value < minimum:
        errors.append(f"{label}: must be at least {minimum}.")
        return minimum
    return value


def _set_text(payload: dict[str, Any], key: str, value: str) -> None:
    payload[key] = str(value)


def _known_values(values: list[str], known: set[str]) -> list[str]:
    return [value for value in values if value in known]


class _FormReader:
    def __init__(self, form: Any) -> None:
        self.form = form

    def get(self, key: str, default: str = "") -> str:
        if hasattr(self.form, "get"):
            value = self.form.get(key, default)
        elif isinstance(self.form, Mapping):
            value = self.form.get(key, default)
        else:
            value = default
        if isinstance(value, list):
            value = value[0] if value else default
        return "" if value is None else str(value)

    def getlist(self, key: str) -> list[str]:
        if hasattr(self.form, "getlist"):
            return [str(value) for value in self.form.getlist(key)]
        value = self.form.get(key, []) if isinstance(self.form, Mapping) else []
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item) for item in value]
        return [str(value)]

    def checked(self, key: str) -> bool:
        value = self.get(key, "")
        return value.lower() in {"1", "true", "yes", "on", "checked"}

def yaml_text(value: Any) -> str:
    if value is None or value == {}:
        return ""
    return yaml.safe_dump(value, sort_keys=False).strip()
