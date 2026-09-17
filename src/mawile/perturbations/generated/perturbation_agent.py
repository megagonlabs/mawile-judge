from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Protocol

from mawile.applications import GenerationOutcome, GenerationUnit, resolve_generation_units
from mawile.judges.prompting import build_grading_context
from mawile.llm import (
    chat_message_text,
    create_chat_completion,
    make_llm_call_record,
)
from mawile.perturbations.base import (
    DimensionSpec,
    ITEM_FIELDS,
    build_variant,
    labeled_item,
    parse_json_object,
)
from mawile.perturbations.registry import llm_dimensions
from mawile.schemas import (
    AuditRunConfig,
    Item,
    JudgeConfig,
    Perturbation,
    PerturbationExpectedEffect,
)


class PerturbationGenerator(Protocol):
    """Produces the LLM-backed perturbation variants for a run.

    The agent must be independent of the judge under test (see
    :class:`mawile.schemas.PerturbationAgentConfig`). Item-side variants
    carry ``requires_validation`` so the validation step gates them with an
    independent check before they count toward flip-risk.
    """

    def generate(
        self,
        config: AuditRunConfig,
        items: list[Item],
        *,
        directional_items: list[Item] | None = None,
    ) -> list[Perturbation]:
        ...


class OpenAIPerturbationAgent:
    """Runs each enabled LLM dimension through a compatible model provider."""

    def __init__(
        self,
        client: Any,
        model: str,
        decoding_params: dict[str, Any] | None = None,
        provider: str = "openai",
    ) -> None:
        self._client = client
        self._model = model
        self._provider = provider
        self._decoding = decoding_params or {}
        self.generation_skips: list[dict[str, Any]] = []
        self.generation_outcomes: list[GenerationOutcome] = []
        self.call_records: list[dict[str, Any]] = []
        self._last_generation_outcome = ""
        self._grading_context = ""
        # Receives (completed, total) after each generation unit; one unit is a
        # judge-side dimension or one item under an item-side dimension.
        self.progress_callback: Callable[[int, int], None] | None = None

    def generate(
        self,
        config: AuditRunConfig,
        items: list[Item],
        *,
        directional_items: list[Item] | None = None,
    ) -> list[Perturbation]:
        specs = list(llm_dimensions(config))
        directional_items = items if directional_items is None else directional_items
        units = resolve_generation_units(
            specs,
            items,
            directional_eligible_item_ids={item.item_id for item in directional_items},
        )
        variants, _outcomes = self.generate_units(config, units)
        return variants

    def generate_units(
        self,
        config: AuditRunConfig,
        units: list[GenerationUnit],
    ) -> tuple[list[Perturbation], list[GenerationOutcome]]:
        """Generate exactly the canonical LLM scopes supplied by the pipeline."""

        self.generation_skips = []
        self.generation_outcomes = []
        self.call_records = []
        total_units = len(units)
        completed_units = 0

        def unit_done() -> None:
            nonlocal completed_units
            completed_units += 1
            if self.progress_callback is not None:
                self.progress_callback(completed_units, total_units)

        def execute(unit: GenerationUnit) -> tuple[
            Perturbation | None,
            list[dict[str, Any]],
            list[dict[str, Any]],
        ]:
            # Keep per-unit mutable bookkeeping isolated. Results are merged in
            # planned order below, so concurrency does not make artifacts flaky.
            worker = OpenAIPerturbationAgent(
                self._client,
                self._model,
                self._decoding,
                provider=self._provider,
            )
            worker._grading_context = build_grading_context(config.judge)
            try:
                if unit.item is None:
                    variant = worker._judge_variant(unit.spec, config.judge)
                else:
                    variant = (
                        worker._directional_item_variant(unit.spec, unit.item)
                        if unit.spec.default_effect == PerturbationExpectedEffect.WORSE_VERDICT
                        else worker._item_variant(unit.spec, unit.item)
                    )
            except Exception as exc:
                worker._record_skip(unit.spec, unit.item, "failed", "generator_exception")
                worker.generation_skips[-1]["error"] = f"{type(exc).__name__}: {exc}"
                variant = None
            return variant, worker.call_records, worker.generation_skips

        results: list[
            tuple[Perturbation | None, list[dict[str, Any]], list[dict[str, Any]]]
            | None
        ] = [None] * total_units
        workers = (
            min(max(1, config.audit.num_workers), total_units) if total_units else 1
        )
        if workers == 1:
            for index, unit in enumerate(units):
                results[index] = execute(unit)
                unit_done()
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(execute, unit): index
                    for index, unit in enumerate(units)
                }
                for future in as_completed(futures):
                    results[futures[future]] = future.result()
                    unit_done()

        variants: list[Perturbation] = []
        for unit, result in zip(units, results, strict=True):
            if result is None:  # Defensive: every planned unit must fill one slot.
                self.generation_outcomes.append(
                    GenerationOutcome(
                        unit.spec.operator,
                        unit.unit_id,
                        unit.item_id,
                        "failed",
                        "generator_returned_no_outcome",
                    )
                )
                continue
            variant, call_records, generation_skips = result
            if variant is not None:
                variants.append(variant)
                self.generation_outcomes.append(
                    GenerationOutcome(
                        unit.spec.operator,
                        unit.unit_id,
                        unit.item_id,
                        "generated",
                        variant_ids=[variant.variant_id],
                    )
                )
            else:
                record = generation_skips[-1] if generation_skips else {}
                status = record.get("status")
                if status not in {"skipped", "failed"}:
                    status = "failed"
                self.generation_outcomes.append(
                    GenerationOutcome(
                        unit.spec.operator,
                        unit.unit_id,
                        unit.item_id,
                        status,
                        str(record.get("reason") or "generator_returned_no_outcome"),
                    )
                )
            self.call_records.extend(call_records)
            self.generation_skips.extend(generation_skips)
        return variants, self.generation_outcomes

    # -- judge-side dimensions ------------------------------------------------
    def _judge_variant(self, spec, judge: JudgeConfig) -> Perturbation | None:
        field = spec.target.split(".")[-1]  # prompt_template | rubric
        noun = "prompt" if field == "prompt_template" else "rubric"
        source = getattr(judge, field, None)
        if not isinstance(source, str) or not source.strip():
            self._record_skip(spec, None, "skipped", "missing_field")
            return None
        rewritten = self._complete(
            f"{spec.instruction} Return only the rewritten {noun} and nothing else.\n\n"
            "Use the complete read-only grading context below; preserve every other "
            "judge behavior and output semantic.\n"
            f"{self._grading_context}",
            source,
        )
        if rewritten is None:
            self._record_skip(spec, None, "failed", self._last_generation_outcome or "generation_failed")
            return None
        if rewritten.strip() == str(getattr(judge, field)).strip():
            self._record_skip(spec, None, "skipped", "unchanged_rewrite")
            return None
        return build_variant(
            spec,
            variant_id=spec.operator,
            summary=f"Perturbation agent: {spec.dimension}.",
            judge_config_overrides={field: rewritten},
            extra_metadata=self._validation_metadata(spec, field, target="judge_config"),
        )

    # -- item-side dimensions -------------------------------------------------
    def _item_variant(self, spec, item: Item) -> Perturbation | None:
        if spec.paired:
            return self._paired_variant(spec, item)
        field = spec.target.split(".")[-1]
        value = getattr(item, field, None)
        if value is None:
            self._record_skip(spec, item, "skipped", "missing_field")
            return None
        # Pass the whole transcript as read-only context so a rewrite of one field
        # (e.g. the output) stays coherent with the others (e.g. the input it answers).
        instructions = (
            f"{spec.instruction} You are rewriting ONLY the <{field}> section of the "
            "transcript below; every other section is read-only context you must not "
            f"change or echo. Return only the rewritten {field} text and nothing else."
        )
        rewritten = self._complete(
            instructions + "\n\nComplete read-only grading context:\n" + self._grading_context,
            _generation_input(item),
        )
        if rewritten is None:
            self._record_skip(spec, item, "failed", self._last_generation_outcome or "generation_failed")
            return None
        if rewritten.strip() == str(value).strip():
            self._record_skip(spec, item, "skipped", "unchanged_rewrite")
            return None
        return build_variant(
            spec,
            variant_id=f"{item.item_id}__{spec.operator}",
            item_id=item.item_id,
            summary=f"Perturbation agent: {spec.dimension}.",
            item_overrides={field: rewritten},
            extra_metadata=self._validation_metadata(
                spec,
                field,
                target="transcript" if field == "output" else "item_field",
            ),
        )

    def _directional_item_variant(self, spec, item: Item) -> Perturbation | None:
        field = spec.target.split(".")[-1]
        value = getattr(item, field, None)
        if value is None:
            self._record_skip(spec, item, "skipped", "missing_field")
            return None
        instructions = (
            f"{spec.instruction} You are creating a directional degradation for a "
            "judge audit. Read the whole transcript, but rewrite ONLY the "
            f"<{field}> section. If this degradation is not naturally applicable, "
            "return JSON with applicable=false and a short skip_reason. If it is "
            "applicable, return JSON with applicable=true, rewritten=<the rewritten "
            f"{field} text>, and summary=<short description>. The edit must be "
            "minimal, plausible, and intentionally worse for the judge's expected "
            "verdict. Return only the JSON object."
        )
        payload = self._complete_json(
            instructions + "\n\nComplete read-only grading context:\n" + self._grading_context,
            _generation_input(item),
        )
        if payload is None:
            self._record_skip(spec, item, "failed", self._last_generation_outcome or "invalid_json")
            return None
        if payload.get("applicable") is False:
            self._record_skip(spec, item, "skipped", str(payload.get("skip_reason") or "inapplicable"))
            return None
        rewritten = payload.get("rewritten", payload.get(field))
        if not isinstance(rewritten, str) or not rewritten.strip():
            self._record_skip(spec, item, "failed", "empty_rewrite")
            return None
        if rewritten.strip() == str(value).strip():
            self._record_skip(spec, item, "skipped", "unchanged_rewrite")
            return None
        summary = str(payload.get("summary") or f"Perturbation agent: {spec.dimension}.")
        return build_variant(
            spec,
            variant_id=f"{item.item_id}__{spec.operator}",
            item_id=item.item_id,
            summary=summary,
            item_overrides={field: rewritten.strip()},
            extra_metadata={
                **self._validation_metadata(spec, field, target="transcript"),
                "directional": True,
            },
        )

    def _paired_variant(self, spec, item: Item) -> Perturbation | None:
        present = [name for name in ITEM_FIELDS if getattr(item, name) is not None]
        if not present:
            self._record_skip(spec, item, "skipped", "missing_field")
            return None
        instructions = (
            f"{spec.instruction} The sections below are one transcript. Return a JSON "
            f"object whose keys are exactly {present}, giving the rewritten value for "
            "each; change nothing else."
        )
        payload = self._complete_json(
            instructions + "\n\nComplete read-only grading context:\n" + self._grading_context,
            _generation_input(item),
        )
        if payload is None:
            self._record_skip(spec, item, "failed", self._last_generation_outcome or "invalid_json")
            return None
        if any(name not in payload for name in present):
            self._record_skip(spec, item, "failed", "missing_field")
            return None
        if all(payload[name] == getattr(item, name) for name in present):
            self._record_skip(spec, item, "skipped", "unchanged_rewrite")
            return None
        return build_variant(
            spec,
            variant_id=f"{item.item_id}__{spec.operator}",
            item_id=item.item_id,
            summary=f"Perturbation agent: {spec.dimension}.",
            changed_fields=[f"item.{name}" for name in present],
            item_overrides={name: payload[name] for name in present},
            extra_metadata={
                "requires_validation": True,
                "validation_kind": spec.validation_kind,
                "validation_target": "transcript",
                "generator_model": self._model,
                "generator_provider": self._provider,
            },
        )

    def _validation_metadata(self, spec, field: str, *, target: str) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "requires_validation": True,
            "validation_kind": spec.validation_kind,
            "validation_field": field,
            "validation_target": target,
            "generator_model": self._model,
            "generator_provider": self._provider,
        }
        if "custom" in spec.touches:
            metadata.update(
                {
                    "custom": True,
                    "custom_name": spec.dimension,
                    "custom_instruction": spec.instruction,
                }
            )
        return metadata

    def _record_skip(
        self,
        spec: DimensionSpec,
        item: Item | None,
        status: str,
        reason: str,
    ) -> None:
        self.generation_skips.append(
            {
                "variant_id": None,
                "item_id": item.item_id if item is not None else "*",
                "family": spec.family.value,
                "operator": spec.operator,
                "intended_operator": spec.operator,
                "declared_relation": spec.expected_relation.value,
                "expected_effect": spec.default_effect.value,
                "changed_fields": [spec.target],
                "status": status,
                "reason": reason,
                "metadata": {
                    "dimension": spec.dimension,
                    "generator_model": self._model,
                    "generator_provider": self._provider,
                },
            }
        )

    # -- model plumbing -------------------------------------------------------
    def _complete(self, instructions: str, input_text: str) -> str | None:
        call_record = make_llm_call_record(
            "perturbation_generation",
            self._provider,
            self._model,
        )
        self.call_records.append(call_record)
        self._last_generation_outcome = ""
        try:
            response = create_chat_completion(
                self._client,
                model=self._model,
                instructions=instructions,
                input_text=input_text,
                decoding_params=self._decoding,
            )
        except Exception as exc:  # Generation failures should not abort the audit.
            self._last_generation_outcome = "api_failure"
            call_record.update(
                make_llm_call_record(
                    "perturbation_generation",
                    self._provider,
                    self._model,
                    status="error",
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            return None
        call_record.update(
            make_llm_call_record(
                "perturbation_generation",
                self._provider,
                self._model,
                response,
                status="ok",
            )
        )
        text = chat_message_text(response)
        if not text or not text.strip():
            self._last_generation_outcome = "empty_response"
            return None
        self._last_generation_outcome = "ok"
        return text

    def _complete_json(self, instructions: str, input_text: str) -> dict[str, Any] | None:
        raw = self._complete(instructions, input_text)
        if raw is None:
            return None
        parsed = parse_json_object(raw)
        if parsed is None:
            self._last_generation_outcome = "invalid_json"
        return parsed


def _generation_input(item: Item) -> str:
    return f"<original_transcript>\n{labeled_item(item)}\n</original_transcript>"
