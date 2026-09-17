from __future__ import annotations

import os
from types import MappingProxyType
from typing import Any, Mapping

from mawile.env import load_environment
from mawile.schemas import AuditRunConfig, ProviderConfig


BUILTIN_PROVIDERS: Mapping[str, ProviderConfig] = MappingProxyType(
    {
        "openai": ProviderConfig(
            api_key_env="OPENAI_API_KEY",
            base_url="https://api.openai.com/v1",
        ),
        "fireworks": ProviderConfig(
            api_key_env="FIREWORKS_API_KEY",
            base_url="https://api.fireworks.ai/inference/v1",
        ),
        "gemini": ProviderConfig(
            api_key_env="GEMINI_API_KEY",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        ),
    }
)


def resolve_provider(config: AuditRunConfig, name: str) -> ProviderConfig:
    """Resolve a named connection profile, preferring config-defined profiles.

    A config entry may replace a built-in with the same name. Returning a copy
    keeps callers from mutating either the frozen built-in registry or the
    resolved run configuration by accident.
    """

    configured = config.providers.get(name)
    if configured is not None:
        return configured.model_copy(deep=True)
    builtin = BUILTIN_PROVIDERS.get(name)
    if builtin is not None:
        return builtin.model_copy(deep=True)
    known = sorted(set(BUILTIN_PROVIDERS) | set(config.providers))
    detail = f" Known providers: {', '.join(known)}." if known else ""
    raise ValueError(f"Unknown provider profile {name!r}.{detail}")


def provider_has_api_key(config: AuditRunConfig, name: str) -> bool:
    """Whether the resolved provider's credential is present in the environment."""

    load_environment()
    provider = resolve_provider(config, name)
    return bool(os.getenv(provider.api_key_env))


def build_provider_client(config: AuditRunConfig, name: str) -> Any:
    """Build an OpenAI client for a configured compatible endpoint.

    Credentials are looked up only at runtime from the environment-variable
    name stored in the profile.
    """

    load_environment()
    provider = resolve_provider(config, name)
    api_key = os.getenv(provider.api_key_env)
    if not api_key:
        raise ValueError(
            f"Provider {name!r} requires API key environment variable "
            f"{provider.api_key_env}."
        )

    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=provider.base_url)
