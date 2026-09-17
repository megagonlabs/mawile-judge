"""Declarative table of perturbation operators.

Each :class:`DimensionSpec` names an operator under the stable
``<component>_<operator>`` convention. Deterministic operators carry a
``builder``; LLM operators carry an ``instruction`` the perturbation agent runs
plus the ``validation_kind`` that gates the result. Every operator declares the
aspect(s) it ``touches`` so the (currently manual) router can pick the probes
appropriate for the judge under test.

``DIMENSIONS`` contains invariant probes. ``DIRECTIONAL_DIMENSIONS`` contains
optional degradation probes for pointwise audits only; pairwise audits never
offer or run directional probes.
"Reference anchoring" is absent:
this harness does not surface reference answers in the judge prompt, so there is
nothing to anchor. Score-scale transformations are not part of the catalog.
"""

from __future__ import annotations

import re

from mawile.perturbations.base import DimensionSpec
from mawile.perturbations.deterministic.judge_ops import (
    build_judge_rubric_criterion_reorder,
)
from mawile.perturbations.deterministic.prompt import (
    build_instruction_order,
    build_reasoning_style,
)
from mawile.perturbations.deterministic.directional import (
    build_output_partial_completion,
)
from mawile.perturbations.deterministic.pairwise import build_position_swap
from mawile.schemas import (
    AuditRunConfig,
    ExpectedRelation,
    OutputType,
    PerturbationExpectedEffect,
    PerturbationFamily,
)

F = PerturbationFamily

_JUDGE_WORDING = ("judge_wording",)



def _is_pairwise(config: AuditRunConfig) -> bool:
    return config.judge.output_type == OutputType.PAIRWISE


def supports_output_type(spec: DimensionSpec, config: AuditRunConfig) -> bool:
    if config.judge.output_type == OutputType.PAIRWISE:
        # Pointwise output rewrites do not define which candidate is edited and
        # therefore cannot enter pairwise headline results. Pairwise support is
        # deliberately limited to judge/input probes plus the typed position swap.
        return spec.family not in {F.AGENT_OUTPUT, F.AGENT_PAIRED} or (
            spec.operator == "agent_paired_position_swap"
        )
    return spec.operator != "agent_paired_position_swap"


_CUSTOM_TARGET_FAMILY = {
    "judge.prompt_template": F.JUDGE_PROMPT,
    "judge.rubric": F.JUDGE_RUBRIC,
    "item.input": F.AGENT_INPUT,
    "item.output": F.AGENT_OUTPUT,
}

_CUSTOM_TARGET_VALIDATION = {
    "judge.prompt_template": "judge_prompt_preserve",
    "judge.rubric": "judge_rubric_preserve",
    "item.input": "meaning_preserve",
    "item.output": "same_answer",
}

_CUSTOM_DIRECTIONAL_VALIDATION = {
    "judge.prompt_template": "directional_degradation",
    "judge.rubric": "directional_degradation",
    "item.input": "directional_degradation",
    "item.output": "directional_degradation",
}


DIMENSIONS: list[DimensionSpec] = [
    DimensionSpec(
        operator="agent_paired_position_swap",
        dimension="Pairwise position swap",
        family=F.AGENT_PAIRED,
        kind="det",
        target="item.output.position_to_candidate",
        applies_to="item",
        touches=("candidate_position",),
        default_effect=PerturbationExpectedEffect.SAME_VERDICT,
        expected_relation=ExpectedRelation.INVARIANT,
        builder=build_position_swap,
        applicability=_is_pairwise,
        description=(
            "Swaps candidate positions A and B while preserving stable candidate "
            "identities. After candidate-ID canonicalization, the verdict should "
            "remain the same."
        ),
    ),
    # -- Judge prompt (judge-side) -------------------------------------------
    DimensionSpec(
        operator="judge_prompt_instruction_order",
        dimension="Instruction ordering",
        family=F.JUDGE_PROMPT,
        kind="det",
        target="judge.prompt_block_order",
        applies_to="judge",
        touches=_JUDGE_WORDING,
        builder=build_instruction_order,
        description=(
            "Moves the output-format instruction ahead of the main prompt and "
            "rubric without rewriting any judge text."
        ),
    ),
    DimensionSpec(
        operator="judge_prompt_reasoning_style",
        dimension="Reasoning style",
        family=F.JUDGE_PROMPT,
        kind="det",
        target="judge.prompt_template",
        applies_to="judge",
        touches=_JUDGE_WORDING,
        builder=build_reasoning_style,
        validation_kind="judge_prompt_preserve",
        description=(
            "Adds one of three explicit reasoning instructions to the judge prompt: "
            "decide directly, think step by step, or evaluate criterion by criterion."
        ),
    ),
    DimensionSpec(
        operator="judge_prompt_role_framing",
        dimension="Role framing",
        family=F.JUDGE_PROMPT,
        kind="llm",
        target="judge.prompt_template",
        applies_to="judge",
        touches=_JUDGE_WORDING,
        instruction=(
            "Rewrite the judge prompt to change only the evaluator persona (for "
            "example from a strict evaluator to a careful, fair reviewer). Keep the "
            "evaluation task, rubric references, and output requirements identical."
        ),
        validation_kind="judge_prompt_preserve",
        description=(
            "Changes only the evaluator persona or role framing while preserving the "
            "evaluation task, rubric references, and output requirements."
        ),
    ),
    DimensionSpec(
        operator="judge_prompt_paraphrase",
        dimension="Prompt paraphrase",
        family=F.JUDGE_PROMPT,
        kind="llm",
        target="judge.prompt_template",
        applies_to="judge",
        touches=_JUDGE_WORDING,
        instruction=(
            "Rewrite the judge prompt, preserving the evaluation task, rubric "
            "references, and output requirements exactly. Change wording and sentence "
            "structure only."
        ),
        validation_kind="judge_prompt_preserve",
        description=(
            "Paraphrases the judge prompt's wording and sentence structure while "
            "preserving the task, rubric references, and output requirements."
        ),
    ),
    # -- Judge rubric (judge-side) -------------------------------------------
    DimensionSpec(
        operator="judge_rubric_criterion_reorder",
        dimension="Rubric criterion reorder",
        family=F.JUDGE_RUBRIC,
        kind="det",
        target="judge.rubric",
        applies_to="judge",
        touches=_JUDGE_WORDING,
        builder=build_judge_rubric_criterion_reorder,
        validation_kind="judge_rubric_preserve",
        description=(
            "Reorders explicitly delimited independent rubric blocks; ambiguous or "
            "free-form rubrics are skipped, and non-structural blocks are semantically "
            "validated."
        ),
    ),
    DimensionSpec(
        operator="judge_rubric_paraphrase",
        dimension="Rubric paraphrase",
        family=F.JUDGE_RUBRIC,
        kind="llm",
        target="judge.rubric",
        applies_to="judge",
        touches=_JUDGE_WORDING,
        instruction=(
            "Reword each rubric criterion, preserving its meaning, ordering, and "
            "thresholds exactly."
        ),
        validation_kind="judge_rubric_preserve",
        description=(
            "Paraphrases rubric criteria while preserving their meaning, ordering, "
            "and thresholds."
        ),
    ),
    # -- Agent input (item-side, edits input) --------------------------------
    DimensionSpec(
        operator="agent_input_minor_typo_noise",
        dimension="Minor typo or noise",
        family=F.AGENT_INPUT,
        kind="llm",
        target="item.input",
        applies_to="item",
        touches=("surface",),
        instruction=(
            "Introduce a small number of natural typographical errors in ordinary "
            "prose only. Preserve the exact meaning, content, requirements, line "
            "breaks, and formatting. Do not alter code blocks, inline code, "
            "identifiers, numbers, URLs, quoted literals, JSON, Markdown delimiters, "
            "or any text whose exact spelling could affect the task."
        ),
        validation_kind="typo_preserve",
        description=(
            "Uses the perturbation agent to introduce a few natural prose typos, "
            "then validates that meaning and exact-syntax content are preserved."
        ),
    ),
    DimensionSpec(
        operator="agent_input_paraphrase",
        dimension="Input paraphrase",
        family=F.AGENT_INPUT,
        kind="llm",
        target="item.input",
        applies_to="item",
        touches=("lexical",),
        instruction=(
            "Rewrite the task input so it preserves the exact same meaning and "
            "requirements with no added, removed, or contradicted information. "
            "Change wording and sentence structure only."
        ),
        validation_kind="meaning_preserve",
        description=(
            "Paraphrases the task input while preserving the same meaning, "
            "requirements, and information."
        ),
    ),
    DimensionSpec(
        operator="agent_input_distractor_insertion",
        dimension="Irrelevant distractor insertion",
        family=F.AGENT_INPUT,
        kind="llm",
        target="item.input",
        applies_to="item",
        touches=("lexical",),
        instruction=(
            "Add one or two plausible but irrelevant sentences to the task input that "
            "do not change what is being asked or the information needed to answer it."
        ),
        validation_kind="sample_plausibility",
        description=(
            "Adds plausible but irrelevant extra context to the task input without "
            "changing what is being asked."
        ),
    ),
    DimensionSpec(
        operator="agent_input_context_compression",
        dimension="Context compression",
        family=F.AGENT_INPUT,
        kind="llm",
        target="item.input",
        applies_to="item",
        touches=("facts",),
        instruction=(
            "Remove redundant or repeated context from the task input while keeping "
            "every requirement and all information needed to complete the task."
        ),
        validation_kind="facts_preserve",
        description=(
            "Compresses the task input by removing redundant context while preserving "
            "all requirements and needed information."
        ),
    ),
    # -- Agent output (item-side, edits output) ------------------------------
    DimensionSpec(
        operator="agent_output_format_conversion",
        dimension="Output format conversion",
        family=F.AGENT_OUTPUT,
        kind="llm",
        target="item.output",
        applies_to="item",
        touches=("format",),
        instruction=(
            "Convert the target output between formats (prose, bullet list, numbered "
            "list, or table) without changing the answer or its correctness."
        ),
        validation_kind="same_answer",
        description=(
            "Converts the target output between formats without changing the answer "
            "or its correctness."
        ),
    ),
    DimensionSpec(
        operator="agent_output_politeness_tone_shift",
        dimension="Politeness or tone shift",
        family=F.AGENT_OUTPUT,
        kind="llm",
        target="item.output",
        applies_to="item",
        touches=("tone",),
        instruction=(
            "Change the politeness or tone of the target output (e.g. warmer or more "
            "neutral) without changing its substance or answer."
        ),
        validation_kind="same_answer",
        description=(
            "Changes the target output's politeness or tone while preserving its "
            "substance and answer."
        ),
    ),
    DimensionSpec(
        operator="agent_output_uncertainty_calibration",
        dimension="Uncertainty calibration",
        family=F.AGENT_OUTPUT,
        kind="llm",
        target="item.output",
        applies_to="item",
        touches=("confidence",),
        instruction=(
            "Make the same answer sound more or less certain in the target output, "
            "changing only the expressed confidence and not the answer itself."
        ),
        validation_kind="same_answer",
        description=(
            "Adjusts expressed confidence in the target output without changing the "
            "answer itself."
        ),
    ),
    DimensionSpec(
        operator="agent_output_verbosity_shift",
        dimension="Verbosity shift",
        family=F.AGENT_OUTPUT,
        kind="llm",
        target="item.output",
        applies_to="item",
        touches=("length",),
        instruction=(
            "Rewrite the target output to be noticeably shorter or longer while "
            "preserving the same answer and the same core correctness. Do not add or "
            "remove substantive claims."
        ),
        validation_kind="same_answer",
        description=(
            "Makes the target output noticeably shorter or longer while preserving "
            "the same answer and substantive claims."
        ),
    ),
    # -- Cross-field paired edits --------------------------------------------
    DimensionSpec(
        operator="agent_paired_entity_rename",
        dimension="Entity renaming",
        family=F.AGENT_PAIRED,
        kind="llm",
        target="item.input",
        applies_to="item",
        touches=("identity",),
        paired=True,
        instruction=(
            "Rename the named entities (people, places, organizations) consistently "
            "across every section of the transcript, preserving every relationship "
            "and the task's meaning. Change nothing else."
        ),
        validation_kind="pair_coherence",
        description=(
            "Renames entities consistently across the transcript while preserving "
            "relationships and task meaning."
        ),
    ),
]


_WORSE = PerturbationExpectedEffect.WORSE_VERDICT

DIRECTIONAL_DIMENSIONS: list[DimensionSpec] = [
    DimensionSpec(
        operator="agent_output_partial_completion",
        dimension="Partial completion",
        family=F.AGENT_OUTPUT,
        kind="det",
        target="item.output",
        applies_to="item",
        touches=("completeness",),
        default_effect=_WORSE,
        expected_relation=ExpectedRelation.DIRECTIONAL,
        builder=build_output_partial_completion,
        description=(
            "Removes the final independent bullet, paragraph, or sentence from the "
            "candidate output so the answer becomes plausibly incomplete."
        ),
        limitations=("Only builds when the output has multiple separable chunks.",),
        validation_kind="directional_degradation",
    ),
    DimensionSpec(
        operator="agent_output_format_violation",
        dimension="Format violation",
        family=F.AGENT_OUTPUT,
        kind="llm",
        target="item.output",
        applies_to="item",
        touches=("format",),
        default_effect=_WORSE,
        expected_relation=ExpectedRelation.DIRECTIONAL,
        instruction=(
            "Identify an explicit output-format requirement in the <input> that the "
            "current <output> satisfies, then minimally rewrite the <output> so it "
            "violates exactly one such requirement. Preserve the response's "
            "substantive content, claims, correctness, and all non-format "
            "requirements. A format requirement may use any natural-language "
            "constraint; do not limit applicability to named formats or fixed "
            "patterns."
        ),
        validation_kind="format_degradation",
        description=(
            "Uses the perturbation agent to break exactly one explicit output-format "
            "requirement that the current output satisfies."
        ),
        limitations=(
            "May skip items with no explicit format requirement already satisfied by "
            "the output.",
        ),
    ),
    DimensionSpec(
        operator="agent_input_unmet_requirement_injection",
        dimension="Unmet requirement injection",
        family=F.AGENT_INPUT,
        kind="llm",
        target="item.input",
        applies_to="item",
        touches=("requirements",),
        default_effect=_WORSE,
        expected_relation=ExpectedRelation.DIRECTIONAL,
        instruction=(
            "Add exactly one plausible, task-relevant requirement to the <input> "
            "section that the existing <output> does not satisfy. Keep the rest of "
            "the input unchanged and do not rewrite the output."
        ),
        validation_kind="unmet_requirement",
        description=(
            "Adds one plausible task-relevant requirement to the input that the "
            "existing output does not satisfy."
        ),
        limitations=("May skip items where no natural unmet requirement can be added.",),
    ),
    DimensionSpec(
        operator="agent_output_unsupported_claim_insertion",
        dimension="Unsupported claim insertion",
        family=F.AGENT_OUTPUT,
        kind="llm",
        target="item.output",
        applies_to="item",
        touches=("support",),
        default_effect=_WORSE,
        expected_relation=ExpectedRelation.DIRECTIONAL,
        instruction=(
            "Add exactly one plausible but unsupported substantive claim to the "
            "<output> section. Keep the original style and all other content as "
            "unchanged as possible."
        ),
        validation_kind="directional_degradation",
        description=(
            "Adds one plausible but unsupported substantive claim to the output."
        ),
    ),
    DimensionSpec(
        operator="agent_output_factual_inconsistency_insertion",
        dimension="Factual inconsistency insertion",
        family=F.AGENT_OUTPUT,
        kind="llm",
        target="item.output",
        applies_to="item",
        touches=("facts",),
        default_effect=_WORSE,
        expected_relation=ExpectedRelation.DIRECTIONAL,
        instruction=(
            "Introduce exactly one small factual or reasoning inconsistency into the "
            "<output> section while keeping the rest of the response plausible and "
            "close to the original."
        ),
        validation_kind="directional_degradation",
        description=(
            "Introduces one small factual or reasoning inconsistency into the output."
        ),
    ),
    DimensionSpec(
        operator="agent_output_requirement_omission",
        dimension="Requirement omission",
        family=F.AGENT_OUTPUT,
        kind="llm",
        target="item.output",
        applies_to="item",
        touches=("requirements",),
        default_effect=_WORSE,
        expected_relation=ExpectedRelation.DIRECTIONAL,
        instruction=(
            "Remove or weaken exactly one part of the <output> section that satisfies "
            "a requirement in the task. Keep the output otherwise natural and close "
            "to the original."
        ),
        validation_kind="directional_degradation",
        description=(
            "Removes or weakens one output segment that satisfies a task requirement."
        ),
        limitations=("May skip items where the satisfied requirements are not clear.",),
    ),
    DimensionSpec(
        operator="agent_output_over_refusal",
        dimension="Over-refusal",
        family=F.AGENT_OUTPUT,
        kind="llm",
        target="item.output",
        applies_to="item",
        touches=("refusal",),
        default_effect=_WORSE,
        expected_relation=ExpectedRelation.DIRECTIONAL,
        instruction=(
            "If the task is answerable and the current <output> gives an answer, "
            "rewrite the <output> into an unnecessary refusal or abstention. Keep it "
            "plausible for a model response. If this item is not answerable or is "
            "already a refusal, mark it inapplicable."
        ),
        validation_kind="refusal_degradation",
        description=(
            "Turns an answerable output into an unnecessary refusal or abstention "
            "when the item is applicable."
        ),
        limitations=("Skips items that are already refusals or not clearly answerable.",),
    ),
]


def all_dimensions(config: AuditRunConfig | None = None) -> list[DimensionSpec]:
    """Every built-in catalog spec, including directional operators."""

    if config is None:
        return [*DIMENSIONS, *DIRECTIONAL_DIMENSIONS]
    candidates = DIMENSIONS if config.judge.output_type == OutputType.PAIRWISE else [*DIMENSIONS, *DIRECTIONAL_DIMENSIONS]
    return [
        spec
        for spec in candidates
        if supports_output_type(spec, config)
        and (spec.applicability is None or spec.applicability(config))
    ]


def selected_dimensions(config: AuditRunConfig) -> list[DimensionSpec]:
    """Resolve the explicit selection once, in configured order.

    Empty built-in lists mean no probes.  Validation happens here too so callers
    that construct a config in Python cannot silently lose an invalid selection.
    """

    from mawile.config import validate_operator_selection

    validate_operator_selection(config)
    invariant = {spec.operator: spec for spec in DIMENSIONS}
    directional = {spec.operator: spec for spec in DIRECTIONAL_DIMENSIONS}
    selected = [invariant[name] for name in config.audit.perturbation_operators]
    selected.extend(
        spec for spec in custom_dimensions(config) if supports_output_type(spec, config)
    )
    selected.extend(
        directional[name] for name in config.audit.directional_perturbation_operators
    )
    return selected


def selected_directional_dimensions(config: AuditRunConfig) -> list[DimensionSpec]:
    if config.judge.output_type == OutputType.PAIRWISE:
        return []
    allow = set(config.audit.directional_perturbation_operators)
    if not allow:
        return []
    selected: list[DimensionSpec] = []
    for spec in DIRECTIONAL_DIMENSIONS:
        if spec.operator not in allow:
            continue
        if not supports_output_type(spec, config):
            continue
        if spec.applicability is not None and not spec.applicability(config):
            continue
        selected.append(spec)
    return selected


def enabled_dimensions(config: AuditRunConfig) -> list[DimensionSpec]:
    return selected_dimensions(config)


def deterministic_dimensions(config: AuditRunConfig) -> list[DimensionSpec]:
    return [spec for spec in enabled_dimensions(config) if spec.kind == "det"]


def llm_dimensions(config: AuditRunConfig) -> list[DimensionSpec]:
    return [spec for spec in enabled_dimensions(config) if spec.kind == "llm"]


def custom_dimensions(config: AuditRunConfig) -> list[DimensionSpec]:
    seen = {spec.operator for spec in [*DIMENSIONS, *DIRECTIONAL_DIMENSIONS]}
    custom_specs: list[DimensionSpec] = []
    for custom in config.audit.custom_perturbations:
        if not custom.enabled:
            continue
        family = _CUSTOM_TARGET_FAMILY[custom.target]
        directional = custom.expected_effect == PerturbationExpectedEffect.WORSE_VERDICT
        operator = _unique_operator(
            f"{family.value}_custom_{_slugify(custom.name)}",
            seen,
        )
        seen.add(operator)
        custom_specs.append(
            DimensionSpec(
                operator=operator,
                dimension=custom.name,
                family=family,
                kind="llm",
                target=custom.target,
                applies_to="judge" if custom.target.startswith("judge.") else "item",
                touches=("custom",),
                default_effect=custom.expected_effect,
                expected_relation=(
                    ExpectedRelation.DIRECTIONAL
                    if directional
                    else ExpectedRelation.INVARIANT
                ),
                instruction=custom.instruction,
                description=custom.instruction,
                validation_kind=(
                    _CUSTOM_DIRECTIONAL_VALIDATION[custom.target]
                    if directional
                    else _CUSTOM_TARGET_VALIDATION[custom.target]
                ),
            )
        )
    return custom_specs


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    return slug or "perturbation"


def _unique_operator(base: str, seen: set[str]) -> str:
    if base not in seen:
        return base
    suffix = 2
    while f"{base}_{suffix}" in seen:
        suffix += 1
    return f"{base}_{suffix}"
