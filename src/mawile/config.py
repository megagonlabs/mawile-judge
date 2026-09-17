from __future__ import annotations

from pathlib import Path
from typing import Any
import yaml

from mawile.env import load_environment
from mawile.schemas import AuditRunConfig


def _resolve_input_path(path: Path, base_dir: Path) -> Path:
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _resolve_output_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def load_config(path: str | Path) -> AuditRunConfig:
    load_environment()
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = expand_config_environment(yaml.safe_load(handle) or {})

    config = AuditRunConfig.model_validate(raw)
    resolve_config_paths(config, config_path.parent)
    validate_config(config)
    return config


def resolve_config_paths(config: AuditRunConfig, base_dir: Path) -> None:
    """Resolve config-file inputs relative to their source and outputs to cwd.

    Payload and file-based configuration share this exact policy.  ``runs_dir``
    has historically been cwd-relative, while data/context paths belong beside
    the YAML or payload source.
    """

    config.data.items_path = _resolve_input_path(config.data.items_path, base_dir)
    if config.data.context_path is not None:
        config.data.context_path = _resolve_input_path(config.data.context_path, base_dir)
    config.output.runs_dir = _resolve_output_path(config.output.runs_dir)


def _validate_operator_names(config: AuditRunConfig) -> None:
    """Reject typo'd operator allow-lists instead of silently dropping them."""

    from mawile.perturbations.registry import DIRECTIONAL_DIMENSIONS, DIMENSIONS

    known = {spec.operator for spec in DIMENSIONS}
    unknown = sorted(set(config.audit.perturbation_operators) - known)
    if unknown:
        raise ValueError(
            "Unknown perturbation_operators: "
            + ", ".join(unknown)
            + ". Use an explicit list of registry operator names; an empty list runs no probes."
        )
    known_directional = {spec.operator for spec in DIRECTIONAL_DIMENSIONS}
    unknown_directional = sorted(set(config.audit.directional_perturbation_operators) - known_directional)
    if unknown_directional:
        raise ValueError(
            "Unknown directional_perturbation_operators: "
            + ", ".join(unknown_directional)
            + ". Use directional registry operator names."
        )


def validate_operator_selection(config: AuditRunConfig) -> None:
    """Validate every explicitly selected operator before any model calls.

    This is intentionally separate from Pydantic schema validation: callers that
    construct or mutate an ``AuditRunConfig`` in Python can use the same gate as
    YAML and UI payloads.  A selected name must be both registered and usable for
    the current judge; silently dropping an invalid choice hides audit intent.
    """

    from mawile.perturbations.registry import (
        DIRECTIONAL_DIMENSIONS,
        DIMENSIONS,
        supports_output_type,
    )
    from mawile.schemas import OutputType, PerturbationExpectedEffect

    _validate_duplicate_operator_names(config)
    _validate_operator_names(config)
    invariant = {spec.operator: spec for spec in DIMENSIONS}
    directional = {spec.operator: spec for spec in DIRECTIONAL_DIMENSIONS}

    unsupported = [
        name
        for name in config.audit.perturbation_operators
        if not supports_output_type(invariant[name], config)
        or (
            invariant[name].applicability is not None
            and not invariant[name].applicability(config)
        )
    ]
    if unsupported:
        raise ValueError(
            "Selected perturbation_operators are unsupported for this judge: "
            + ", ".join(sorted(set(unsupported)))
        )
    if config.judge.output_type == OutputType.PAIRWISE:
        unsupported_custom = [
            custom.name
            for custom in config.audit.custom_perturbations
            if custom.enabled and custom.target == "item.output"
        ]
        if unsupported_custom:
            raise ValueError(
                "Selected custom perturbations are unsupported for a pairwise judge: "
                + ", ".join(unsupported_custom)
            )
        directional_custom = [
            custom.name
            for custom in config.audit.custom_perturbations
            if custom.enabled
            and custom.expected_effect == PerturbationExpectedEffect.WORSE_VERDICT
        ]
        if config.audit.directional_perturbation_operators or directional_custom:
            selected = [*config.audit.directional_perturbation_operators, *directional_custom]
            raise ValueError(
                "Pairwise audits do not support directional perturbations; remove: "
                + ", ".join(selected)
            )
        return

    unsupported_directional = [
        name
        for name in config.audit.directional_perturbation_operators
        if not supports_output_type(directional[name], config)
        or (
            directional[name].applicability is not None
            and not directional[name].applicability(config)
        )
    ]
    if unsupported_directional:
        raise ValueError(
            "Selected directional_perturbation_operators are unsupported for this judge: "
            + ", ".join(sorted(set(unsupported_directional)))
        )


def _validate_duplicate_operator_names(config: AuditRunConfig) -> None:
    """Reject repeated explicit selections rather than changing probe weight."""

    for field_name, names in (
        ("perturbation_operators", config.audit.perturbation_operators),
        ("directional_perturbation_operators", config.audit.directional_perturbation_operators),
    ):
        seen: set[str] = set()
        duplicates: list[str] = []
        for name in names:
            if name in seen and name not in duplicates:
                duplicates.append(name)
            seen.add(name)
        if duplicates:
            raise ValueError(f"Duplicate {field_name}: " + ", ".join(duplicates))


def _validate_planner_config(config: AuditRunConfig) -> None:
    """The suggester defaults to the perturbation-agent model, so it needs a real one."""

    if config.audit.plan_operators and config.resolved_planner_model() == "mock":
        raise ValueError(
            "audit.plan_operators requires a non-mock Perturbation Suggester model "
            "(set planner_agent.model or perturbation_agent.model, independent of the judge under test)."
        )


def validate_config(config: AuditRunConfig) -> None:
    """Shared runtime and payload validation for a resolved audit configuration."""

    validate_operator_selection(config)
    _validate_planner_config(config)


def resolve_planned_operator_selection(
    config: AuditRunConfig,
    operators: list[str],
) -> tuple[list[str], list[str]]:
    """Split and validate a planner result without merging manual selections.

    The planner owns the resulting lists.  This small helper is shared by the
    CLI, UI, and runtime planning paths so an unknown or unsupported model
    response can never be quietly discarded.
    """

    from mawile.perturbations.registry import DIRECTIONAL_DIMENSIONS, DIMENSIONS

    known_invariant = {spec.operator for spec in DIMENSIONS}
    known_directional = {spec.operator for spec in DIRECTIONAL_DIMENSIONS}
    non_strings = [name for name in operators if not isinstance(name, str)]
    if non_strings:
        raise ValueError("Planner operator names must be strings")
    unknown = [name for name in operators if name not in known_invariant | known_directional]
    if unknown:
        raise ValueError("Planner selected unknown operators: " + ", ".join(sorted(set(unknown))))

    # Preserve the returned order while removing duplicates, which makes the
    # exact resolved selection inspectable and reproducible.
    unique = list(dict.fromkeys(operators))
    invariant = [name for name in unique if name in known_invariant]
    directional = [name for name in unique if name in known_directional]
    resolved = config.model_copy(deep=True)
    resolved.audit.perturbation_operators = invariant
    resolved.audit.directional_perturbation_operators = directional
    resolved.audit.plan_operators = False
    validate_operator_selection(resolved)
    return invariant, directional


def expand_config_environment(value: Any, *, key: str | None = None) -> Any:
    """Expand config values without ever expanding an ``api_key_env`` name.

    Provider profiles store names such as ``GEMINI_API_KEY``. If a user writes
    ``${GEMINI_API_KEY}`` there by mistake, expanding it would place the secret
    itself into validation errors or resolved config artifacts.
    """

    import os

    if isinstance(value, dict):
        return {
            item_key: expand_config_environment(item_value, key=str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, list):
        return [expand_config_environment(item, key=key) for item in value]
    if isinstance(value, str) and key != "api_key_env":
        return os.path.expandvars(value)
    return value


# Private compatibility for callers that imported the old helper.
_expand_env = expand_config_environment
