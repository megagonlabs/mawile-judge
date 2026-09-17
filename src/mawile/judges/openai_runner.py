from __future__ import annotations

import json
import threading
from typing import Any

from mawile.judges.base import JudgeRunner
from mawile.judges.parser import parse_judge_output
from mawile.judges.prompting import (
    build_judge_input,
    build_judge_instructions,
    build_response_format,
)
from mawile.env import load_environment
from mawile.llm import (
    chat_message_text,
    create_chat_completion,
    usage_dict,
)
from mawile.providers import build_provider_client
from mawile.schemas import AuditRunConfig, Item, JudgeConfig, JudgeResult, Perturbation


class OpenAIJudgeRunner(JudgeRunner):
    def __init__(
        self,
        client: Any | None = None,
        *,
        config: AuditRunConfig | None = None,
    ) -> None:
        self._client = client
        self._config = config
        self._client_lock = threading.Lock()

    def run(
        self,
        item: Item,
        judge_config: JudgeConfig,
        perturbation: Perturbation | None,
        run_id: str,
    ) -> JudgeResult:
        provider = judge_config.provider
        model = judge_config.model
        metadata: dict[str, Any] = {}
        try:
            response = create_chat_completion(
                self._get_client(provider),
                model=model,
                instructions=build_judge_instructions(judge_config),
                input_text=build_judge_input(item),
                decoding_params=judge_config.decoding_params,
                response_format=build_response_format(judge_config),
            )
            # Capture billable usage before any response extraction or parsing;
            # a later local failure must not erase a completed provider call.
            metadata = _response_metadata(response, provider)
            raw_output = _extract_response_text(response)
            parsed_verdict, parse_status = parse_judge_output(
                raw_output,
                output_type=judge_config.output_type,
                threshold=judge_config.threshold,
                score_direction=judge_config.score_direction,
                score_min=judge_config.score_min,
                score_max=judge_config.score_max,
                item=item,
            )
            call_status = "ok"
        except Exception as exc:  # The pipeline should preserve failures as artifacts.
            raw_output = f"{type(exc).__name__}: {exc}"
            parsed_verdict = None
            call_status = "transport_error"
            parse_status = "not_attempted"
            metadata["error_type"] = type(exc).__name__

        metadata.update(
            {
                "provider": provider,
                "model": model,
                "family": perturbation.family.value if perturbation else "noise",
                "operator": perturbation.operator if perturbation else "repeat",
            }
        )
        return JudgeResult(
            item_id=item.item_id,
            variant_id=perturbation.variant_id if perturbation else "original",
            run_id=run_id,
            raw_judge_output=raw_output,
            parsed_verdict=parsed_verdict,
            call_status=call_status,
            parse_status=parse_status,
            metadata=metadata,
        )

    def _get_client(self, provider: str) -> Any:
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    if self._config is None:
                        if provider != "openai":
                            raise ValueError(
                                "A full run config is required for a non-OpenAI judge provider."
                            )
                        self._client = build_openai_client()
                    else:
                        self._client = build_provider_client(self._config, provider)
        return self._client


def build_openai_client() -> Any:
    load_environment()
    from openai import OpenAI

    return OpenAI()


def _extract_response_text(response: Any) -> str:
    if text := chat_message_text(response):
        return text

    if hasattr(response, "model_dump_json"):
        return response.model_dump_json()
    return json.dumps(response, default=str)


def _response_metadata(response: Any, provider: str = "openai") -> dict[str, Any]:
    metadata: dict[str, Any] = {"provider": provider}
    for attr in ("id", "model"):
        value = getattr(response, attr, None)
        if value is not None:
            metadata[f"provider_{attr}"] = value
    if usage := usage_dict(response):
        metadata["usage"] = usage
    return metadata
