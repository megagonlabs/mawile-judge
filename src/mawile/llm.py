from __future__ import annotations

from typing import Any


def create_chat_completion(
    client: Any,
    *,
    model: str,
    instructions: str,
    input_text: str,
    decoding_params: dict[str, Any] | None = None,
    response_format: dict[str, Any] | None = None,
) -> Any:
    """Call the common OpenAI-compatible Chat Completions surface.

    MAWILE uses single-turn system + user exchanges for every LLM role. Keeping
    that translation here makes provider routing independent of judge,
    generation, planning, validation, and reporting code.
    """

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": instructions},
            {"role": "user", "content": input_text},
        ],
        **{
            key: value
            for key, value in (decoding_params or {}).items()
            if value is not None
        },
    }
    if response_format is not None:
        kwargs["response_format"] = response_format
    return client.chat.completions.create(**kwargs)


def chat_message_text(response: Any) -> str:
    """Extract assistant text from a Chat Completion response."""

    choices = getattr(response, "choices", None) or []
    if choices:
        content = getattr(getattr(choices[0], "message", None), "content", None)
        if isinstance(content, str):
            return content.strip()
    return ""


def usage_dict(response: Any) -> dict[str, Any]:
    """Return token usage with MAWILE's provider-neutral input/output names."""

    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        payload = usage.model_dump(mode="json")
    elif isinstance(usage, dict):
        payload = dict(usage)
    else:
        payload = {
            name: getattr(usage, name)
            for name in (
                "input_tokens",
                "output_tokens",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "cached_input_tokens",
                "cache_write_tokens",
            )
            if getattr(usage, name, None) is not None
        }
        for name in ("prompt_tokens_details", "input_tokens_details"):
            if (details := getattr(usage, name, None)) is not None:
                payload[name] = _object_dict(details)
    if payload.get("input_tokens") is None and payload.get("prompt_tokens") is not None:
        payload["input_tokens"] = payload["prompt_tokens"]
    if payload.get("output_tokens") is None and payload.get("completion_tokens") is not None:
        payload["output_tokens"] = payload["completion_tokens"]
    if payload.get("cached_input_tokens") is None:
        for details_name in ("prompt_tokens_details", "input_tokens_details"):
            details = payload.get(details_name)
            cached_tokens = (
                details.get("cached_tokens") if isinstance(details, dict) else None
            )
            if cached_tokens is not None:
                payload["cached_input_tokens"] = cached_tokens
                break
    if payload.get("cache_write_tokens") is None:
        for details_name in ("prompt_tokens_details", "input_tokens_details"):
            details = payload.get(details_name)
            write_tokens = (
                details.get("cache_write_tokens") if isinstance(details, dict) else None
            )
            if write_tokens is not None:
                payload["cache_write_tokens"] = write_tokens
                break
    return payload


def _object_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return dict(value)
    return {
        name: item
        for name, item in vars(value).items()
        if not name.startswith("_")
    }


def make_llm_call_record(
    category: str,
    provider: str,
    model: str,
    response: Any | None = None,
    *,
    status: str = "attempted",
    error: str | None = None,
) -> dict[str, Any]:
    """Build the small provider-neutral record used for token accounting."""

    record: dict[str, Any] = {
        "category": category,
        "provider": provider,
        "model": model,
        "status": status,
        "usage": usage_dict(response) if response is not None else {},
    }
    if response is not None and (response_id := getattr(response, "id", None)) is not None:
        record["response_id"] = response_id
    if error is not None:
        record["error"] = error
    return record
