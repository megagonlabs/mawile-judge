"""Central per-provider/model token pricing.

The figures below are USD per 1,000,000 tokens and are starting-point estimates;
verify them against current provider pricing and edit as needed. Models with no
entry are surfaced as unpriced; reported dollar totals are then lower bounds.
"""

from __future__ import annotations

from dataclasses import dataclass

_PER_MILLION = 1_000_000


@dataclass(frozen=True)
class ModelPrice:
    """USD per 1,000,000 tokens for a single model."""

    input_per_mtok: float
    output_per_mtok: float
    cached_input_per_mtok: float | None = None
    cache_write_per_mtok: float | None = None
    long_context_threshold: int | None = None
    long_context_input_multiplier: float = 1.0
    long_context_output_multiplier: float = 1.0


# USD per 1M tokens.
_OPENAI_MODEL_PRICES: dict[str, ModelPrice] = {
    # --- GPT-5 ---
    "gpt-5.6-luna": ModelPrice(0.20, 1.20, 0.02),
    "gpt-5.6-sol": ModelPrice(4.00, 20.00, 0.40),
    "gpt-5.5": ModelPrice(5.00, 30.00),
    "gpt-5.5-pro": ModelPrice(30.00, 180.00),
    "gpt-5.4": ModelPrice(2.50, 15.00, 0.25),
    "gpt-5.4-mini": ModelPrice(0.75, 4.50, 0.075),
    "gpt-5.4-nano": ModelPrice(0.20, 1.25, 0.02),
    "gpt-5.4-pro": ModelPrice(30.00, 180.00),
    "gpt-5.2": ModelPrice(1.75, 14.00, 0.175),
    "gpt-5.2-pro": ModelPrice(21.00, 168.00),
    "gpt-5.1": ModelPrice(1.25, 10.00),
    "gpt-5": ModelPrice(1.25, 10.00, 0.125),
    "gpt-5-mini": ModelPrice(0.25, 2.00, 0.025),
    "gpt-5-nano": ModelPrice(0.05, 0.40),
    "gpt-5-pro": ModelPrice(15.00, 120.00),
    # --- GPT-4.x ---
    "gpt-4.1": ModelPrice(2.00, 8.00, 0.50),
    "gpt-4.1-mini": ModelPrice(0.40, 1.60, 0.10),
    "gpt-4.1-nano": ModelPrice(0.10, 0.40, 0.025),
    "gpt-4o": ModelPrice(2.50, 10.00),
    "gpt-4o-2024-05-13": ModelPrice(5.00, 15.00),
    "gpt-4o-mini": ModelPrice(0.15, 0.60),
    # --- o-series ---
    "o1": ModelPrice(15.00, 60.00),
    "o1-pro": ModelPrice(150.00, 600.00),
    "o1-mini": ModelPrice(1.10, 4.40),
    "o3": ModelPrice(2.00, 8.00),
    "o3-pro": ModelPrice(20.00, 80.00),
    "o3-mini": ModelPrice(1.10, 4.40),
    "o4-mini": ModelPrice(1.10, 4.40),
    # --- Legacy GPT-4 ---
    "gpt-4-turbo-2024-04-09": ModelPrice(10.00, 30.00),
    "gpt-4-0125-preview": ModelPrice(10.00, 30.00),
    "gpt-4-1106-preview": ModelPrice(10.00, 30.00),
    "gpt-4-1106-vision-preview": ModelPrice(10.00, 30.00),
    "gpt-4-0613": ModelPrice(30.00, 60.00),
    "gpt-4-0314": ModelPrice(30.00, 60.00),
    "gpt-4-32k": ModelPrice(60.00, 120.00),
    # --- GPT-3.5 ---
    "gpt-3.5-turbo": ModelPrice(0.50, 1.50),
    "gpt-3.5-turbo-0125": ModelPrice(0.50, 1.50),
    "gpt-3.5-turbo-1106": ModelPrice(1.00, 2.00),
    "gpt-3.5-turbo-0613": ModelPrice(1.50, 2.00),
    "gpt-3.5-0301": ModelPrice(1.50, 2.00),
    "gpt-3.5-turbo-instruct": ModelPrice(1.50, 2.00),
    "gpt-3.5-turbo-16k-0613": ModelPrice(3.00, 4.00),
}

_FIREWORKS_MODEL_PRICES: dict[str, ModelPrice] = {
    # Qwen3.8 Max's model-library URL uses the shorter alias, while the API model
    # page exposes qwen3p8-2p4t-a95b as the canonical model path.
    "accounts/fireworks/models/qwen3p8-max": ModelPrice(2.00, 6.00, 0.25),
    "accounts/fireworks/models/qwen3p8-2p4t-a95b": ModelPrice(2.00, 6.00, 0.25),
    "accounts/fireworks/models/deepseek-v4-flash": ModelPrice(0.14, 0.28, 0.028),
    "accounts/fireworks/models/deepseek-v4-flash-0731": ModelPrice(0.22, 0.66, 0.007),
    "accounts/fireworks/models/kimi-k3": ModelPrice(3.00, 15.00, 0.30),
}


# Provider is part of the key because the same model name can be sold at
# different rates by different providers.
MODEL_PRICES: dict[tuple[str, str], ModelPrice] = {
    **{("openai", model): price for model, price in _OPENAI_MODEL_PRICES.items()},
    **{("fireworks", model): price for model, price in _FIREWORKS_MODEL_PRICES.items()},
}


def lookup_price(provider: str, model: str) -> ModelPrice | None:
    """Return the provider-specific price entry, or None if none applies.

    An exact key wins; otherwise the longest model key for the same provider
    whose name is followed by a ``-`` suffix wins. Thus dated snapshots map to
    their base entry without crossing provider boundaries.
    """

    if (exact := MODEL_PRICES.get((provider, model))) is not None:
        return exact

    best_key: str | None = None
    for price_provider, price_model in MODEL_PRICES:
        if price_provider != provider:
            continue
        if model.startswith(f"{price_model}-") and (
            best_key is None or len(price_model) > len(best_key)
        ):
            best_key = price_model
    return MODEL_PRICES[(provider, best_key)] if best_key is not None else None


def call_cost_usd(
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    *,
    cached_input_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float | None:
    """USD for a single call, or None if it has no provider/model price.

    Cached reads and cache writes are subsets of input tokens. When the table
    lacks an applicable cache rate, this returns the known-cost lower bound;
    :func:`call_price_complete` lets the caller mark that subtotal incomplete.
    """

    price = lookup_price(provider, model)
    if price is None:
        return None

    input_tokens = max(0, input_tokens)
    output_tokens = max(0, output_tokens)
    cached_input_tokens = min(max(0, cached_input_tokens), input_tokens)
    cache_write_tokens = min(
        max(0, cache_write_tokens), input_tokens - cached_input_tokens
    )
    uncached_input_tokens = input_tokens - cached_input_tokens - cache_write_tokens
    cached_rate = price.cached_input_per_mtok
    if cached_rate is None:
        cached_rate = 0.0
    cache_write_rate = price.cache_write_per_mtok
    if cache_write_rate is None:
        cache_write_rate = 0.0
    input_multiplier = 1.0
    output_multiplier = 1.0
    if (
        price.long_context_threshold is not None
        and input_tokens > price.long_context_threshold
    ):
        input_multiplier = price.long_context_input_multiplier
        output_multiplier = price.long_context_output_multiplier
    return (
        uncached_input_tokens * price.input_per_mtok * input_multiplier
        + cached_input_tokens * cached_rate * input_multiplier
        + cache_write_tokens * cache_write_rate * input_multiplier
        + output_tokens * price.output_per_mtok * output_multiplier
    ) / _PER_MILLION


def call_price_complete(
    provider: str,
    model: str,
    *,
    cached_input_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> bool:
    """Whether every token class present in a call has an explicit rate."""

    price = lookup_price(provider, model)
    if price is None:
        return False
    if cached_input_tokens > 0 and price.cached_input_per_mtok is None:
        return False
    if cache_write_tokens > 0 and price.cache_write_per_mtok is None:
        return False
    return True
