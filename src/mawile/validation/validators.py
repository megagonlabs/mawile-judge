from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import re
import json
import logging
from typing import Any, Protocol

from openai import APIConnectionError, APIStatusError, OpenAI
from pydantic import ValidationError

from mawile.llm import (
    chat_message_text,
    create_chat_completion,
    make_llm_call_record,
)
from mawile.perturbations.base import (
    apply_item_overrides,
    apply_judge_overrides,
    labeled_item,
    stringify,
)
from mawile.judges.prompting import build_grading_context
from mawile.schemas import AuditRunConfig, Item, Perturbation

logger = logging.getLogger(__name__)


class PerturbationValidator(Protocol):
    """Checks whether a generated perturbation is admissible.

    A validator MUST be independent of the judge under test. Using the judge to
    vet its own probes is circular: "is this a valid paraphrase?" and "does the
    judge treat the two as equivalent?" are different questions, and the second
    is exactly the signal we are trying to measure.

    The question asked depends on the dimension's ``validation_kind``:
    ``meaning_preserve`` for input rewrites that must mean the same,
    ``typo_preserve`` for typo-only input rewrites that must also preserve exact
    syntax such as code and identifiers,
    ``sample_plausibility`` for input edits that must stay realistic same-task
    samples, ``same_answer`` for output edits that must keep the same answer,
    ``facts_preserve`` for edits that may shorten but must keep every
    task-relevant fact (e.g. context compression), ``pair_coherence`` for paired
    edits that rewrite several fields and must stay mutually consistent, and
    directional checks for edits that should clearly make an item worse, including
    ``format_degradation`` for a single explicit output-format violation.
    Judge-side rewrites use prompt/rubric-specific preservation checks.
    """

    def validate(
        self, original: Any, perturbed: Any, perturbation: Perturbation
    ) -> "ValidationResult":
        ...


@dataclass(frozen=True)
class ValidationResult:
    """Small typed result shared by rule and independent semantic validators.

    ``__iter__`` keeps the historical ``accepted, details = ...`` interface
    usable by callers while exposing a named object for durable reports/tests.
    """

    accepted: bool
    details: dict[str, Any]

    def __iter__(self):
        yield self.accepted
        yield self.details


def _validation_field(perturbation: Perturbation) -> str:
    if field := perturbation.metadata.get("validation_field"):
        return str(field)
    for changed_field in perturbation.changed_fields:
        root, _, path = changed_field.partition(".")
        if root in {"item", "judge"} and path:
            return path
    return "output"


def _validation_kind(perturbation: Perturbation) -> str:
    return perturbation.metadata.get("validation_kind") or "meaning_preserve"


def _validation_target(perturbation: Perturbation) -> str:
    if target := perturbation.metadata.get("validation_target"):
        return str(target)
    if any(field.startswith("judge.") for field in perturbation.changed_fields):
        return "judge_config"
    if _validation_kind(perturbation) in {
        "same_answer",
        "pair_coherence",
        "directional_degradation",
        "format_degradation",
        "unmet_requirement",
        "refusal_degradation",
    }:
        return "transcript"
    return "item_field"


class RuleEquivalenceValidator:
    """Cheap, deterministic guardrails for generated variants.

    Catches only mechanical failure modes with no model call.  It deliberately
    never accepts a semantic rewrite based on length or token overlap: doing so
    would turn an implementation convenience into scientific validity.
    """

    def __init__(self, min_length_ratio: float = 0.5, max_length_ratio: float = 2.0) -> None:
        self.min_length_ratio = min_length_ratio
        self.max_length_ratio = max_length_ratio

    def validate(
        self, original: Any, perturbed: Any, perturbation: Perturbation
    ) -> ValidationResult:
        field = _validation_field(perturbation)
        kind = _validation_kind(perturbation)
        field_value = getattr(perturbed, field, None)
        if field_value is None or not stringify(field_value).strip():
            return ValidationResult(False, {"reason": "empty_output", "validator": "rule", "field": field})
        before, after = _validation_texts(original, perturbed, perturbation)
        before = before.strip()
        after = after.strip()
        if not after:
            return ValidationResult(False, {"reason": "empty_output", "validator": "rule", "field": field})
        if before == after:
            return ValidationResult(False, {
                "reason": "unchanged_output",
                "validator": "rule",
                "field": field,
            })
        return ValidationResult(False, {
            "reason": "semantic_validation_required",
            "kind": kind,
            "validator": "rule",
            "field": field,
        })


_PROMPTS = {
    "meaning_preserve": (
        "You check whether two texts convey the same meaning. Accept only if the "
        "second text preserves the meaning of the first with no added, removed, or "
        "contradicted information."
    ),
    "typo_preserve": (
        "You check a typo-only perturbation of a task input. Accept only if the "
        "second text differs solely by a small number of "
        "natural typographical errors in ordinary prose, preserves every word's "
        "intended meaning and every requirement, and leaves formatting, code blocks, "
        "inline code, identifiers, numbers, URLs, quoted literals, JSON, and Markdown "
        "delimiters unchanged."
    ),
    "sample_plausibility": (
        "You check whether two task inputs are interchangeable samples. Accept only "
        "if the second input is a realistic variant from the same task distribution "
        "that asks for the same thing."
    ),
    "same_answer": (
        "You check whether an output rewrite remains a valid answer to its task. The "
        "original and rewritten transcripts are given as tagged sections. Accept only "
        "if the second output answers the unchanged input and gives "
        "the same answer with the same correctness as the first, preserving any explicit "
        "format/style requirements and preserving rather than silently fixing errors "
        "already present in the original."
    ),
    "facts_preserve": (
        "You check whether a rewritten text preserves every task-relevant fact. Accept "
        "only if the second text keeps every requirement, "
        "constraint, and piece of information needed to do the task (it may drop only "
        "redundant restatement)."
    ),
    "pair_coherence": (
        "You check whether a rewritten transcript stays coherent and equivalent. The "
        "two transcripts are given as tagged sections. Accept only if the second keeps "
        "the same task and answer as the first, with entities and "
        "terms updated consistently across every section and no contradictions between "
        "sections."
    ),
    "judge_prompt_preserve": (
        "You check whether two judge prompts define the same evaluation instrument. "
        "Accept only if the second keeps the same evaluation task, "
        "rubric references, thresholds, and output requirements while changing only "
        "wording/persona/register."
    ),
    "judge_rubric_preserve": (
        "You check whether two judge rubrics define the same scoring criteria. Accept "
        "only if the second preserves every criterion, threshold, "
        "score meaning, and ordering semantics while changing only wording or format, "
        "without changing substantive scoring behavior."
    ),
    "judge_rubric_rescale": (
        "You check whether a rubric was affinely rescaled without changing what it "
        "rewards. Accept only if the second rubric preserves the same ordering and "
        "criterion meanings on the new numeric range."
    ),
    "directional_degradation": (
        "You check whether an edit is a valid directional degradation. Accept only if "
        "the second text changes only the intended target, "
        "remains plausible for the audit, and clearly makes the item, output, or "
        "evaluation instrument worse in the intended way. A mere deletion, shorter "
        "answer, or changed wording is not evidence of degradation; assess the task, "
        "rubric, output requirements, and whether the original's errors remain; "
        "otherwise reject it."
    ),
    "format_degradation": (
        "You check whether an output edit creates one valid format degradation. The "
        "two transcripts are given as tagged sections. Accept only if the unchanged "
        "input contains an explicit output-format "
        "requirement, the first output satisfies that requirement, the second output "
        "violates exactly one such requirement, and the edit preserves substantive "
        "content, claims, correctness, and all non-format requirements."
    ),
    "unmet_requirement": (
        "You check whether a transcript edit adds one valid unmet requirement. The "
        "two transcripts are given as tagged sections. Accept only if "
        "the second transcript adds exactly one plausible requirement to the input "
        "and the unchanged output does not satisfy it, with no unrelated changes."
    ),
    "refusal_degradation": (
        "You check whether a transcript edit creates an over-refusal. The two "
        "transcripts are given as tagged sections. Accept only if "
        "the first output answered an answerable task and the second output changes "
        "only into an unnecessary refusal or abstention."
    ),
}

_JSON_CONTRACT = (
    'Reply with exactly one JSON object: {"accepted": true, "reason": "short reason"}. '
    "`accepted` must be a boolean and `reason` must be non-empty and at most 240 characters."
)


def _criteria_prompt(kind: str) -> str:
    """Attach the one shared machine-readable output contract."""

    text = _PROMPTS.get(kind, _PROMPTS["meaning_preserve"])
    return f"{text}\n\n{_JSON_CONTRACT}"


def validation_prompt(kind: str | None) -> str | None:
    if kind is None:
        return None
    return _criteria_prompt(kind)


def validation_prompts() -> dict[str, str]:
    """Return the built-in validation prompts keyed by validation kind."""

    return {kind: _criteria_prompt(kind) for kind in _PROMPTS}


class LLMEquivalenceValidator:
    """Independent LLM check, deliberately *not* the judge under test.

    Uses a neutral prompt chosen by the dimension's ``validation_kind`` and a
    model/role separate from the judge, so admissibility does not depend on the
    instrument being audited.
    """

    def __init__(
        self,
        client: Any,
        model: str,
        decoding_params: dict[str, Any] | None = None,
        provider: str = "openai",
    ) -> None:
        # Let the SDK own exponential backoff, jitter, and Retry-After handling.
        # Four retries means at most five attempts, without nested retry loops.
        # Copy the client so other roles retain their existing retry policy.
        self._client = client.with_options(max_retries=4) if isinstance(client, OpenAI) else client
        self._model = model
        self._provider = provider
        self._decoding = decoding_params or {}
        self.call_records: list[dict[str, Any]] = []
        self._grading_context = ""

    def reset_call_records(self) -> None:
        self.call_records = []

    def set_grading_context(self, context: str) -> None:
        """Set read-only judge context for the next validation batch."""

        self._grading_context = context

    def validate(
        self, original: Any, perturbed: Any, perturbation: Perturbation
    ) -> ValidationResult:
        kind = _validation_kind(perturbation)
        before, after = _validation_texts(original, perturbed, perturbation)
        call_record = make_llm_call_record(
            "equivalence_validation",
            self._provider,
            self._model,
        )
        self.call_records.append(call_record)
        try:
            instructions = _criteria_prompt(kind)
            instructions += (
                f"\n\nIntended operator: {perturbation.metadata.get('intended_operator', perturbation.operator)}; "
                f"declared relation: {perturbation.metadata.get('declared_relation', perturbation.expected_relation.value)}."
            )
            if self._grading_context:
                instructions = (
                    f"{instructions}\n\nThe following is read-only grading context. "
                    "Use it to preserve the judged task, output semantics, explicit "
                    "format/style constraints, and any original errors; do not use "
                    "the judge under test as the source of the verdict.\n"
                    f"{self._grading_context}"
                )
            response = create_chat_completion(
                self._client,
                model=self._model,
                instructions=instructions,
                input_text=(
                    "Original transcript/field:\n"
                    f"{before}\n\nPerturbed transcript/field:\n{after}"
                ),
                decoding_params=self._decoding,
            )
        except Exception as exc:
            call_record.update(
                make_llm_call_record(
                    "equivalence_validation",
                    self._provider,
                    self._model,
                    status="error",
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            if isinstance(exc, APIConnectionError) or (
                isinstance(exc, APIStatusError)
                and (exc.status_code in {408, 409, 429} or exc.status_code >= 500)
            ):
                logger.warning(
                    "Validation unavailable for %s (%s/%s): %s; continuing",
                    perturbation.variant_id, self._provider, self._model, exc,
                )
                return ValidationResult(False, {
                    "status": "unavailable",
                    "reason": "validation_unavailable",
                    "error": call_record["error"],
                    "status_code": getattr(exc, "status_code", None),
                    "kind": kind,
                    "validator": "llm",
                    "validator_provider": self._provider,
                    "validator_model": self._model,
                })
            raise
        call_record.update(
            make_llm_call_record(
                "equivalence_validation",
                self._provider,
                self._model,
                response,
                status="ok",
            )
        )
        raw_verdict = chat_message_text(response).strip()
        try:
            verdict = json.loads(raw_verdict)
        except json.JSONDecodeError:
            verdict = None
        if (
            not isinstance(verdict, dict)
            or type(verdict.get("accepted")) is not bool
            or not isinstance(verdict.get("reason"), str)
            or not verdict["reason"].strip()
            or len(verdict["reason"].strip()) > 240
        ):
            return ValidationResult(False, {
                "reason": "invalid_validator_response",
                "validator_reason": "Validator must return JSON {accepted: boolean, reason: short string}.",
                "kind": kind,
                "validator": "llm",
                "validator_provider": self._provider,
                "validator_model": self._model,
            })
        return ValidationResult(verdict["accepted"], {
            "reason": verdict["reason"].strip(),
            "kind": kind,
            "validator_reason": verdict["reason"].strip(),
            "verdict": verdict["accepted"],
            "validator": "llm",
            "validator_provider": self._provider,
            "validator_model": self._model,
        })


def _validation_texts(original: Any, perturbed: Any, perturbation: Perturbation) -> tuple[str, str]:
    target = _validation_target(perturbation)
    if target == "transcript" or isinstance(original, Item):
        return labeled_item(original), labeled_item(perturbed)
    field = _validation_field(perturbation)
    return stringify(getattr(original, field, None)), stringify(getattr(perturbed, field, None))


def requires_validation(perturbation: Perturbation) -> bool:
    """Whether the declared transform needs semantic admissibility validation.

    Some deterministic edits (partial completion and reasoning directives) are
    only candidate constructions and therefore carry the same gate as generated
    rewrites.  Typed structural transforms are checked separately.
    """

    return bool(perturbation.metadata.get("requires_validation"))


def _grading_context(config: AuditRunConfig | None) -> str:
    return build_grading_context(config.judge) if config is not None else ""


def _structural_check(
    perturbation: Perturbation, original: Any, perturbed: Any
) -> tuple[bool, str] | None:
    """Validate transforms whose relation is provable without semantics.

    ``None`` means this is a semantic transform and must go through the declared
    validator; callers must not treat it as an automatic pass.
    """

    operator = perturbation.operator
    if operator == "agent_paired_position_swap":
        if not isinstance(original, Item) or not isinstance(perturbed, Item):
            return False, "position_swap_requires_pairwise_items"
        if not original.is_pairwise or not perturbed.is_pairwise:
            return False, "position_swap_requires_pairwise_items"
        before = original.output
        after = perturbed.output
        original_fields = original.model_dump()
        perturbed_fields = perturbed.model_dump()
        original_fields.pop("output", None)
        perturbed_fields.pop("output", None)
        if original_fields != perturbed_fields:
            return False, "position_swap_changed_unrelated_item_fields"
        if not isinstance(before, dict) or not isinstance(after, dict):
            return False, "position_swap_malformed_output"
        if set(before.get("candidates", {})) != set(after.get("candidates", {})):
            return False, "position_swap_changed_candidate_set"
        if before.get("candidates") != after.get("candidates"):
            return False, "position_swap_changed_candidate_values"
        original_positions = before.get("position_to_candidate")
        if not isinstance(original_positions, dict) or set(original_positions) != {"A", "B"}:
            return False, "position_swap_malformed_mapping"
        expected = {"A": original_positions["B"], "B": original_positions["A"]}
        actual = after.get("position_to_candidate")
        if actual != expected:
            return False, "position_swap_not_exact_two_cycle"
        if actual == before.get("position_to_candidate"):
            return False, "position_swap_noop"
        return True, "structural_position_swap"

    if operator == "judge_prompt_instruction_order":
        original_fields = original.model_dump()
        perturbed_fields = perturbed.model_dump()
        original_fields.pop("prompt_block_order", None)
        perturbed_fields.pop("prompt_block_order", None)
        if original_fields != perturbed_fields:
            return False, "prompt_block_order_changed_unrelated_judge_fields"
        before = getattr(original, "prompt_block_order", None)
        after = getattr(perturbed, "prompt_block_order", None)
        expected = {"prompt", "rubric", "output"}
        if not isinstance(after, list) or set(after) != expected or len(after) != 3:
            return False, "prompt_block_order_not_permutation"
        if before == after:
            return False, "prompt_block_order_noop"
        return True, "structural_prompt_block_permutation"

    if perturbation.metadata.get("structural_invariant") == "rubric_block_permutation":
        before = getattr(original, "rubric", None)
        after = getattr(perturbed, "rubric", None)
        if not isinstance(before, str) or not isinstance(after, str):
            return False, "rubric_reorder_malformed"
        original_fields = original.model_dump()
        perturbed_fields = perturbed.model_dump()
        original_fields.pop("rubric", None)
        perturbed_fields.pop("rubric", None)
        if original_fields != perturbed_fields:
            return False, "rubric_reorder_changed_unrelated_judge_fields"
        if before == after:
            return False, "rubric_reorder_noop"
        # Recheck the builder's grammar, including unique score labels and
        # order-dependent references. Metadata alone cannot establish validity.
        from mawile.perturbations.deterministic.judge_ops import _independent_blocks

        original_blocks = _independent_blocks(before)
        perturbed_blocks = _independent_blocks(after)
        if not original_blocks or not perturbed_blocks:
            return False, "rubric_reorder_not_flat_numeric_rows"
        prefix, rows, flat = original_blocks
        new_prefix, new_rows, new_flat = perturbed_blocks
        if not flat or not new_flat or prefix != new_prefix:
            return False, "rubric_reorder_not_flat_numeric_rows"
        if sorted(rows) != sorted(new_rows):
            return False, "rubric_reorder_changed_block_content"
        return True, "structural_rubric_block_permutation"

    return None


def validate_perturbations(
    perturbations: list[Perturbation],
    items: list[Item],
    validator: PerturbationValidator | None = None,
    config: AuditRunConfig | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[list[Perturbation], list[dict[str, Any]]]:
    """Stamp each perturbation's ``validity_status`` and return a validation log.

    Returns all perturbations (with updated status); callers drop the
    ``rejected`` and ``unavailable`` ones before judging. Typed structural
    variants are checked offline; semantic variants (including deterministic candidates) use the
    declared validator. ``progress_callback`` receives
    ``(completed, total)`` after each variant that needs validation is
    resolved, including ones rejected before reaching the validator.
    """

    validator = validator or RuleEquivalenceValidator()
    reset_call_records = getattr(validator, "reset_call_records", None)
    if callable(reset_call_records):
        reset_call_records()
    item_by_id = {item.item_id: item for item in items}
    report: list[dict[str, Any]] = []
    validated: list[Perturbation] = []
    structural_operators = {
        "agent_paired_position_swap",
        "judge_prompt_instruction_order",
        "judge_rubric_criterion_reorder",
    }
    needs_resolution = lambda p: requires_validation(p) or p.operator in structural_operators
    total = sum(1 for perturbation in perturbations if needs_resolution(perturbation))
    completed = 0

    set_context = getattr(validator, "set_grading_context", None)
    if callable(set_context):
        set_context(_grading_context(config))

    def unit_done() -> None:
        nonlocal completed
        completed += 1
        if progress_callback is not None:
            progress_callback(completed, total)

    for perturbation in perturbations:
        if not needs_resolution(perturbation):
            validated.append(perturbation)
            continue

        base_record = _validation_record_base(perturbation)
        target = _validation_target(perturbation)
        if target == "judge_config":
            if config is None:
                validated.append(perturbation.model_copy(update={"validity_status": "rejected"}))
                report.append({**base_record, "status": "rejected", "reason": "missing_config"})
                unit_done()
                continue
            original: Any = config.judge
            try:
                perturbed = apply_judge_overrides(config.judge, perturbation.judge_config_overrides)
            except ValidationError as exc:
                validated.append(perturbation.model_copy(update={"validity_status": "rejected"}))
                report.append(
                    {
                        **base_record,
                        "status": "rejected",
                        "reason": "invalid_override",
                        "details": exc.errors(),
                    }
                )
                unit_done()
                continue
        else:
            original = item_by_id.get(perturbation.item_id)
            if original is None:
                validated.append(perturbation.model_copy(update={"validity_status": "rejected"}))
                report.append({**base_record, "status": "rejected", "reason": "missing_item"})
                unit_done()
                continue
            try:
                perturbed = apply_item_overrides(original, perturbation.item_overrides)
            except ValidationError as exc:
                validated.append(perturbation.model_copy(update={"validity_status": "rejected"}))
                report.append(
                    {
                        **base_record,
                        "status": "rejected",
                        "reason": "invalid_override",
                        "details": exc.errors(),
                    }
                )
                unit_done()
                continue
        structural = _structural_check(perturbation, original, perturbed)
        if structural is not None:
            structurally_valid, reason = structural
            if not structurally_valid:
                validated.append(perturbation.model_copy(update={"validity_status": "rejected"}))
                report.append({**base_record, "status": "rejected", "reason": reason, "validator": "structural"})
                unit_done()
                continue
            if not requires_validation(perturbation):
                # This is a genuinely structural invariant.  Its acceptance is
                # justified by the exact typed relation, not by a rule heuristic.
                validated.append(perturbation)
                unit_done()
                continue
        if _validation_kind(perturbation) == "typo_preserve":
            before, after = _validation_texts(original, perturbed, perturbation)
            if _protected_typo_content(before) != _protected_typo_content(after):
                validated.append(
                    perturbation.model_copy(update={"validity_status": "rejected"})
                )
                report.append(
                    {
                        **base_record,
                        "status": "rejected",
                        "reason": "protected_content_changed",
                    }
                )
                unit_done()
                continue
        before, after = _validation_texts(original, perturbed, perturbation)
        if before.strip() == after.strip():
            validated.append(perturbation.model_copy(update={"validity_status": "rejected"}))
            report.append({**base_record, "status": "rejected", "reason": "unchanged_rewrite"})
            unit_done()
            continue
        accepted, details = validator.validate(original, perturbed, perturbation)
        status = "accepted" if accepted else "rejected"
        if details.get("status") == "unavailable":
            status = "unavailable"
        validated.append(perturbation.model_copy(update={"validity_status": status}))
        report.append({**base_record, "status": status, **details})
        unit_done()

    return validated, report


_TYPO_PROTECTED_PATTERNS = (
    re.compile(r"```.*?```", re.DOTALL),
    re.compile(r"`[^`\n]*`"),
    re.compile(r"https?://[^\s)>\]]+"),
    re.compile(r"(?<![\w.])[+-]?\d+(?:\.\d+)?(?![\w.])"),
    re.compile(r"\"[^\"\n]*\"|'[^'\n]*'"),
)


def _protected_typo_content(text: str) -> tuple[tuple[str, ...], ...]:
    """Return exact-syntax spans that a prose-only typo edit may not change."""

    return tuple(tuple(pattern.findall(text)) for pattern in _TYPO_PROTECTED_PATTERNS)


def _validation_record_base(perturbation: Perturbation) -> dict[str, Any]:
    kind = _validation_kind(perturbation)
    return {
        "variant_id": perturbation.variant_id,
        "operator": perturbation.operator,
        "intended_operator": perturbation.metadata.get("intended_operator", perturbation.operator),
        "declared_relation": perturbation.metadata.get(
            "declared_relation", perturbation.expected_relation.value
        ),
        "expected_effect": perturbation.expected_effect.value,
        "validation_kind": kind,
        "validation_prompt": validation_prompt(kind),
    }
