from __future__ import annotations

from mawile.perturbations.base import DimensionSpec, build_variant
from mawile.schemas import AuditRunConfig, Item, Perturbation


def build_instruction_order(
    spec: DimensionSpec, config: AuditRunConfig, items: list[Item]
) -> list[Perturbation]:
    if config.judge.prompt_block_order == ["output", "prompt", "rubric"]:
        return []
    return [
        build_variant(
            spec,
            variant_id=spec.operator,
            summary="Move the output-format instruction before the main prompt without rewriting any block.",
            changed_fields=["judge.prompt_block_order"],
            judge_config_overrides={"prompt_block_order": ["output", "prompt", "rubric"]},
        )
    ]


def build_reasoning_style(
    spec: DimensionSpec, config: AuditRunConfig, items: list[Item]
) -> list[Perturbation]:
    base = config.judge.prompt_template.strip()
    styles = {
        "direct": "Decide the score directly without writing out step-by-step reasoning.",
        "step_by_step": (
            "Think step by step about how the output meets the rubric before deciding the score."
        ),
        "criterion_by_criterion": (
            "Evaluate the output against each rubric criterion in turn before deciding the score."
        ),
    }
    return [
        build_variant(
            spec,
            variant_id=f"{spec.operator}__{key}",
            summary=f"Ask the judge to reason {key.replace('_', ' ')}.",
            judge_config_overrides={"prompt_template": f"{base}\n\n{text}"},
            # The directive is not a purely structural permutation: a reasoning
            # instruction can accidentally alter the evaluation task.  Gate it
            # with the same independent semantic check as generated rewrites.
            extra_metadata={
                "style": key,
                "requires_validation": True,
                "validation_kind": "judge_prompt_preserve",
                "validation_field": "prompt_template",
                "validation_target": "judge_config",
            },
        )
        for key, text in styles.items()
    ]
