from __future__ import annotations

import json
from typing import Any

from mawile.schemas import Item, JudgeConfig, OutputType


def build_judge_instructions(judge_config: JudgeConfig) -> str:
    comparison = (
        "greater than or equal to"
        if judge_config.score_direction.value == "higher_is_better"
        else "less than or equal to"
    )
    failing_side = (
        "lower scores are fail"
        if judge_config.score_direction.value == "higher_is_better"
        else "higher scores are fail"
    )
    threshold_text = (
        f"\nDecision threshold: scores {comparison} "
        f"{judge_config.threshold} are pass; {failing_side}."
        if judge_config.threshold is not None
        else ""
    )
    blocks = {
        "prompt": judge_config.prompt_template.strip(),
        "rubric": f"Rubric:\n{judge_config.rubric.strip()}{threshold_text}",
        "output": _output_instruction(judge_config.output_type),
    }
    seen: set[str] = set()
    ordered: list[str] = []
    for key in judge_config.prompt_block_order:
        if key in blocks and key not in seen:
            ordered.append(blocks[key])
            seen.add(key)
    ordered.extend(block for key, block in blocks.items() if key not in seen)
    return "\n\n".join(ordered)


def build_grading_context(judge: JudgeConfig) -> str:
    """Read-only instrument context shared by generation and validation.

    Include each instruction once and use the actual response format supplied to
    the judge, so the two research roles cannot drift or duplicate large rubrics.
    """

    return (
        f"<judge_instructions>\n{build_judge_instructions(judge)}\n</judge_instructions>\n"
        "<judge_output_semantics>\n"
        f"output_type={judge.output_type.value}; score_direction={judge.score_direction.value}; "
        f"threshold={judge.threshold!r}; score_range=({judge.score_min!r}, {judge.score_max!r})\n"
        f"response_format={json.dumps(build_response_format(judge), ensure_ascii=False, sort_keys=True)}\n"
        "</judge_output_semantics>"
    )


def build_judge_input(item: Item) -> str:
    # Metadata is audit-side provenance and may contain gold or subgroup labels;
    # it must never become evidence available to the judge under test.
    payload: dict[str, Any] = {"input": item.input}
    if item.is_pairwise:
        payload["output"] = {
            "candidates_by_position": {
                position: {
                    "candidate_id": candidate_id,
                    "value": item.candidates[candidate_id],
                }
                for position, candidate_id in item.position_to_candidate.items()
            }
        }
    else:
        payload["output"] = item.output
    return (
        "Evaluate this item and return JSON only.\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)}"
    )


def _output_instruction(output_type: OutputType) -> str:
    if output_type == OutputType.PAIRWISE:
        return (
            "Compare the two candidates shown at positions A and B. Return only a "
            "JSON object. Set `label` to `A` when the candidate at position A is "
            "better, or `B` when the candidate at position B is better. Include "
            "`reason` with a brief explanation."
        )
    return (
        "Return only a JSON object. Include `score` when the judge uses a score, "
        "`label` when the judge makes a pass/fail decision, and `reason` with a "
        "brief explanation."
    )


def build_response_format(judge_config: JudgeConfig) -> dict[str, Any]:
    """Response-format payload for OpenAI-compatible Chat Completions.

    Non-structured verdicts get a strict JSON schema so the model returns exactly
    the verdict fields and cannot echo the prompt's schema description back as its
    answer. Structured judges fall back to a free-form JSON object unless the
    config supplies its own object schema.
    """

    schema = _verdict_schema(judge_config)
    if schema is None:
        return {"type": "json_object"}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "judge_verdict",
            "strict": True,
            "schema": schema,
        },
    }


def _verdict_schema(judge_config: JudgeConfig) -> dict[str, Any] | None:
    if judge_config.output_type == OutputType.STRUCTURED:
        schema = judge_config.output_schema
        if isinstance(schema, dict) and schema.get("type") == "object" and "properties" in schema:
            return schema
        return None

    properties: dict[str, Any] = {}
    for field in _verdict_fields(judge_config):
        if field == "label":
            enum = ["A", "B"] if judge_config.output_type == OutputType.PAIRWISE else ["pass", "fail"]
            properties["label"] = {"type": "string", "enum": enum}
        elif field == "score":
            score_schema: dict[str, Any] = {"type": "number"}
            if judge_config.score_min is not None:
                score_schema["minimum"] = judge_config.score_min
            if judge_config.score_max is not None:
                score_schema["maximum"] = judge_config.score_max
            properties["score"] = score_schema
        else:
            properties[field] = {"type": "string"}
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _verdict_fields(judge_config: JudgeConfig) -> list[str]:
    """The verdict keys to enforce, honoring `output_schema.fields` when given.

    Falls back to type-appropriate defaults, then guarantees the field the parser
    treats as the verdict for that type: a `score` for scalar/ordinal judges, a
    `label` for binary/pairwise ones.
    """

    schema = judge_config.output_schema
    if isinstance(schema, dict) and isinstance(schema.get("fields"), list):
        fields = [str(field) for field in schema["fields"] if str(field)]
    else:
        fields = []

    if not fields:
        if judge_config.output_type in {OutputType.ORDINAL, OutputType.SCALAR}:
            fields = ["score", "reason"]
        else:
            fields = ["label", "reason"]

    if judge_config.output_type in {OutputType.ORDINAL, OutputType.SCALAR} and "score" not in fields:
        fields = ["score", *fields]
    if judge_config.output_type in {OutputType.BINARY, OutputType.PAIRWISE} and "label" not in fields:
        fields = ["label", *fields]
    return fields
