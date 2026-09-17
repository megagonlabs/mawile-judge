from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from mawile.config import (
    expand_config_environment,
    load_config,
    resolve_config_paths,
    validate_config,
)
from mawile.io import load_items
from mawile.env import load_environment
from mawile.schemas import AuditRunConfig, Item


@dataclass(frozen=True)
class LoadedConfigPayload:
    payload: dict[str, Any]
    base_dir: Path
    source_path: Path | None = None


def load_config_payload(path: Path) -> LoadedConfigPayload:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return LoadedConfigPayload(
        payload=payload,
        base_dir=path.parent.resolve(),
        source_path=path.resolve(),
    )


def build_config_from_payload(payload: dict[str, Any], base_dir: Path) -> AuditRunConfig:
    load_environment()
    config = AuditRunConfig.model_validate(expand_config_environment(payload))
    resolve_config_paths(config, base_dir)
    validate_config(config)
    return config


def validate_config_payload(
    payload: dict[str, Any],
    base_dir: Path,
) -> tuple[AuditRunConfig | None, str | None]:
    try:
        return build_config_from_payload(payload, base_dir), None
    except ValidationError as exc:
        return None, str(exc)
    except (OSError, TypeError, ValueError) as exc:
        return None, str(exc)


def dump_config_yaml(payload: dict[str, Any]) -> str:
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=False)


def parse_mapping_text(raw: str, *, empty_value: Any = None) -> Any:
    if not raw.strip():
        return empty_value
    parsed = yaml.safe_load(raw)
    if parsed is None:
        return empty_value
    if not isinstance(parsed, (dict, str)):
        raise ValueError("Expected a mapping, string, or empty value.")
    return parsed


def load_config_and_items(path: Path) -> tuple[AuditRunConfig, list[Item]]:
    config = load_config(path)
    return config, load_items(config.data)

__all__ = [
    "LoadedConfigPayload",
    "build_config_from_payload",
    "dump_config_yaml",
    "load_config_and_items",
    "load_config_payload",
    "parse_mapping_text",
    "validate_config_payload",
]
