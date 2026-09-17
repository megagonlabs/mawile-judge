from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from mawile.io import load_items
from mawile.applications import (
    ApplicationOutcome,
    GenerationOutcome,
    GenerationUnit,
    record_judge_outcomes,
    resolve_applications,
    resolve_generation_units,
    summarize_application_outcomes,
)
from mawile.operator_manifest import (
    build_operator_manifest,
    build_perturbation_ledger,
)
from mawile.judges import JudgeRunner, get_judge_runner
from mawile.judges.prompting import build_judge_input, build_judge_instructions
from mawile.metrics import (
    apply_directional_risk,
    compute_confidence_intervals,
    compute_cost,
    compute_coverage,
    compute_directional_metrics,
    compute_item_risks,
    compute_judge_correctness,
    compute_noise_floor,
    summarize_attribution,
    summarize_perturbation_attribution,
)
from mawile.measurement import (
    directional_baseline_eligibility,
    validate_items_for_judge,
)
from mawile.config import resolve_planned_operator_selection, validate_config
from mawile.perturbations import build_perturbations
from mawile.perturbations.base import (
    DimensionSpec,
    apply_item_overrides,
    apply_judge_overrides,
)
from mawile.perturbations.generated import OpenAIPerturbationAgent, PerturbationGenerator
from mawile.perturbations.planning import (
    OpenAIPlannerAgent,
    PerturbationPlanner,
    PlanResult,
)
from mawile.perturbations.registry import (
    llm_dimensions,
    selected_dimensions,
)
from mawile.providers import build_provider_client, provider_has_api_key
from mawile.reporting.artifacts import append_stage_record, write_artifacts, write_stage_snapshot
from mawile.reporting.interpretation import build_report_interpretation
from mawile.reporting.markdown import render_markdown_report
from mawile.suites import (
    AuditSuite,
    build_suite,
    load_suite,
    reset_application_counts,
    validate_replay_config,
    write_suite,
)
from mawile.schemas import (
    AuditRunConfig,
    Item,
    ItemRisk,
    JudgeConfig,
    JudgeResult,
    Perturbation,
    PerturbationExpectedEffect,
)
from mawile.validation import (
    LLMEquivalenceValidator,
    PerturbationValidator,
    RuleEquivalenceValidator,
    requires_validation,
    validate_perturbations,
)


@dataclass(frozen=True)
class AuditRunResult:
    run_id: str
    artifact: dict[str, Any]
    paths: dict[str, Path]


ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class _PlannedCall:
    """One judge call to issue, with everything needed to trace it."""

    call_type: str
    original_item: Item
    submitted_item: Item
    submitted_config: JudgeConfig
    perturbation: Perturbation | None
    run_id: str


@dataclass(frozen=True)
class _LoadedAuditInputs:
    items: list[Item]
    selected_specs: list[DimensionSpec]


@dataclass(frozen=True)
class _PerturbationPhaseResult:
    accepted: list[Perturbation]
    attempted: list[Perturbation]
    generated: list[Perturbation]
    generation_trace: list[dict[str, Any]]
    validation_report: list[dict[str, Any]]
    validation_calls: int
    llm_calls: list[dict[str, Any]]
    operator_manifest: dict[str, Any]
    ledger: list[dict[str, Any]]
    generation_outcomes: list[GenerationOutcome]
    application_outcomes: list[ApplicationOutcome]


class AuditPipeline:
    def __init__(
        self,
        config: AuditRunConfig,
        runner: JudgeRunner | None = None,
        generator: PerturbationGenerator | None = None,
        validator: PerturbationValidator | None = None,
        planner: PerturbationPlanner | None = None,
        progress_callback: ProgressCallback | None = None,
        suite_path: Path | None = None,
    ) -> None:
        # Planning resolves a run-local suite.  Never mutate the caller's
        # configuration object while doing so; it may be reused for another run.
        self.config = config.model_copy(deep=True)
        self.progress_callback = progress_callback
        # Construction must be inert: callers may construct a pipeline merely to
        # inspect it, and planning mutates an effective allow-list.  Defaults and
        # all paid work are therefore resolved by ``run`` after local validation.
        self.planner = planner
        self.runner = runner
        self.generator = generator
        self.validator = validator
        self.suite_path = suite_path
        self.plan_result: PlanResult | None = None
        self._planning_manual: list[str] = []
        self._planning_manual_directional: list[str] = []
        self._planning_selected: list[str] = []
        self._planning_selected_directional: list[str] = []

    def _apply_operator_plan(self) -> PlanResult | None:
        """Resolve planning explicitly, after all local failures are ruled out."""

        if not self.config.audit.plan_operators:
            return None
        self.planner = self.planner or _default_planner(self.config)
        result = self.planner.plan(self.config)
        manual = list(self.config.audit.perturbation_operators)
        manual_directional = list(self.config.audit.directional_perturbation_operators)
        selected, selected_directional = resolve_planned_operator_selection(
            self.config, result.operators
        )
        # A planned suite is authoritative, rather than a second, implicit
        # selection mode.  Manual values are trace evidence, not a union rule.
        self.config.audit.perturbation_operators = selected
        self.config.audit.directional_perturbation_operators = selected_directional
        self.config.audit.plan_operators = False
        self._planning_manual = manual
        self._planning_manual_directional = manual_directional
        self._planning_selected = selected
        self._planning_selected_directional = selected_directional
        return result

    def _planning_trace(self) -> dict[str, Any] | None:
        if self.plan_result is None:
            return None
        return {
            "selected": self._planning_selected,
            "manual": self._planning_manual,
            "effective": list(self.config.audit.perturbation_operators),
            "selected_directional": self._planning_selected_directional,
            "manual_directional": self._planning_manual_directional,
            "effective_directional": list(self.config.audit.directional_perturbation_operators),
            "rationale": self.plan_result.rationale,
            "provider": self.plan_result.provider,
            "model": self.plan_result.model,
            "llm_call": self.plan_result.call_record,
            "messages": self.plan_result.messages,
            "context_metadata": self.plan_result.context_metadata,
        }

    def _progress(self, stage: str, **details: Any) -> None:
        if self.progress_callback is None:
            return
        self.progress_callback({"stage": stage, **details})

    def run(self) -> AuditRunResult:
        self._progress("started", message="Starting audit")
        if self.suite_path is not None:
            return self._run_suite_replay()
        # Validate all inexpensive, deterministic prerequisites before planner,
        # generator, validator, or judge calls can spend money.
        validate_config(self.config)
        run_id = _new_run_id()
        events: list[dict[str, Any]] = [_event("run_started", run_id=run_id)]
        loaded = self._load_audit_inputs(events)
        self._checkpoint(run_id, "items", loaded.items)
        self.plan_result = self._apply_operator_plan()
        # Planning may have produced a new effective suite, so validate it too.
        validate_config(self.config)
        loaded = _LoadedAuditInputs(
            items=loaded.items,
            selected_specs=selected_dimensions(self.config),
        )
        self._checkpoint(
            run_id,
            "operator_selection",
            [
                {
                    "operators": [spec.operator for spec in loaded.selected_specs],
                    "planning_trace": self._planning_trace(),
                }
            ],
        )
        self._ensure_semantic_prerequisites(loaded.selected_specs)
        self.runner = self.runner or get_judge_runner(self.config.judge.model, config=self.config)
        self.generator = self.generator if self.generator is not None else _default_generator(self.config)
        self.validator = self.validator or _default_validator(self.config)
        planning_trace = self._planning_trace()
        if planning_trace is not None:
            self._progress(
                "planning_complete",
                message="Perturbation Suggester selected operators",
            )
            events.append(_event("operators_planned", **planning_trace))

        baseline_results, baseline_traces = self._run_baseline_judge_phase(
            run_id,
            loaded.items,
            events,
        )
        self._checkpoint(run_id, "baseline_judge_results", baseline_results)
        directional_eligibility = directional_baseline_eligibility(
            loaded.items,
            baseline_results,
            self.config.judge,
        )
        directional_eligibility["directional_requested"] = any(
            spec.default_effect == PerturbationExpectedEffect.WORSE_VERDICT
            for spec in loaded.selected_specs
        )
        eligible_item_ids = {
            decision["item_id"]
            for decision in directional_eligibility["items"]
            if decision["eligible"]
        }
        directional_items = [
            item for item in loaded.items if item.item_id in eligible_item_ids
        ]
        events.append(
            _event(
                "directional_baseline_eligibility_resolved",
                policy=directional_eligibility["policy"],
                eligible_items=directional_eligibility["eligible_item_count"],
                ineligible_items=directional_eligibility["ineligible_item_count"],
            )
        )
        perturbation_phase = self._build_and_validate_perturbations(
            run_id,
            loaded,
            directional_items,
            directional_eligibility,
            events,
        )
        self._checkpoint(
            run_id,
            "generated_perturbations",
            perturbation_phase.generated,
        )
        self._checkpoint(
            run_id,
            "validation_report",
            perturbation_phase.validation_report,
        )
        write_suite(
            self.config.output.runs_dir / run_id,
            build_suite(
                run_id=run_id,
                config=self.config,
                items=loaded.items,
                baseline_results=baseline_results,
                generated=perturbation_phase.generated,
                attempted=perturbation_phase.attempted,
                accepted=perturbation_phase.accepted,
                generation_outcomes=perturbation_phase.generation_outcomes,
                application_outcomes=perturbation_phase.application_outcomes,
                validation_report=perturbation_phase.validation_report,
                operator_manifest=perturbation_phase.operator_manifest,
                planning_trace=planning_trace,
                directional_eligibility=directional_eligibility,
                generation_trace=perturbation_phase.generation_trace,
            ),
        )
        return self._finish_after_perturbations(
            run_id=run_id,
            loaded=loaded,
            perturbation_phase=perturbation_phase,
            baseline_results=baseline_results,
            baseline_traces=baseline_traces,
            planning_trace=planning_trace,
            directional_eligibility=directional_eligibility,
            events=events,
        )

    def _run_suite_replay(self) -> AuditRunResult:
        """Rerun judge measurements from frozen source evidence only."""

        suite = load_suite(self.suite_path)
        validate_replay_config(suite, self.config)
        validate_config(self.config)
        run_id = _new_run_id()
        events: list[dict[str, Any]] = [
            _event("run_started", run_id=run_id),
            _event(
                "suite_replay_loaded",
                source_run_id=suite.provenance.get("source_run_id"),
                suite_sha256=suite.content_sha256,
            ),
        ]
        selected_specs = selected_dimensions(self.config)
        loaded = _LoadedAuditInputs(items=suite.items, selected_specs=selected_specs)
        self._checkpoint(run_id, "items", loaded.items)
        self._checkpoint(
            run_id,
            "operator_selection",
            [{"operators": [spec.operator for spec in selected_specs], "planning_trace": suite.planning_trace}],
        )
        # Replay never plans, generates, or validates; source evidence has
        # already been checked by load_suite and remains an immutable cohort.
        source_eligibility = suite.directional_eligibility
        source_eligibility["directional_requested"] = any(
            spec.default_effect == PerturbationExpectedEffect.WORSE_VERDICT
            for spec in selected_specs
        )
        applications = reset_application_counts(
            suite.application_outcomes, self.config.audit.repeats
        )
        phase = _PerturbationPhaseResult(
            accepted=suite.accepted,
            attempted=suite.attempted,
            generated=suite.generated,
            generation_trace=suite.generation_trace,
            validation_report=suite.validation_report,
            validation_calls=0,
            llm_calls=[],
            operator_manifest=suite.operator_manifest,
            ledger=[],
            generation_outcomes=suite.generation_outcomes,
            application_outcomes=applications,
        )
        self._checkpoint(run_id, "generation_outcomes", phase.generation_outcomes)
        self._checkpoint(run_id, "proposed_perturbations", phase.attempted)
        self._checkpoint(run_id, "generated_perturbations", phase.generated)
        self._checkpoint(run_id, "validation_report", phase.validation_report)
        self._checkpoint(run_id, "application_outcomes", applications)
        # The replay bundle is a byte-stable copy of the verified source suite.
        write_suite(self.config.output.runs_dir / run_id, suite.payload)
        self.runner = self.runner or get_judge_runner(self.config.judge.model, config=self.config)
        baseline_results, baseline_traces = self._run_baseline_judge_phase(
            run_id, loaded.items, events
        )
        self._checkpoint(run_id, "baseline_judge_results", baseline_results)
        fresh_eligibility = directional_baseline_eligibility(
            loaded.items, baseline_results, self.config.judge
        )
        source_eligibility["replay_source_baseline_results"] = len(suite.baseline_results)
        replay_provenance = {
            "suite_sha256": suite.content_sha256,
            "source_run_id": suite.provenance["source_run_id"],
            "directional_cohort": "frozen_source_baseline",
            "fresh_baseline_eligibility": fresh_eligibility,
            "historical_llm_calls_included_in_replay_cost": False,
        }
        return self._finish_after_perturbations(
            run_id=run_id,
            loaded=loaded,
            perturbation_phase=phase,
            baseline_results=baseline_results,
            baseline_traces=baseline_traces,
            planning_trace=suite.planning_trace,
            directional_eligibility=source_eligibility,
            events=events,
            charge_planning_call=False,
            replay_provenance=replay_provenance,
        )

    def _checkpoint(self, run_id: str, stage: str, records: list[Any]) -> None:
        """Best-effort durable phase output; a checkpoint failure must be visible."""

        serialized = [
            (
                record.model_dump(mode="json")
                if hasattr(record, "model_dump")
                else asdict(record)
                if is_dataclass(record)
                else record
            )
            for record in records
        ]
        write_stage_snapshot(
            run_id=run_id,
            runs_dir=self.config.output.runs_dir,
            config=self.config,
            stage=stage,
            records=serialized,
        )

    def _load_audit_inputs(self, events: list[dict[str, Any]]) -> _LoadedAuditInputs:
        self._progress("loading_items", message="Loading items")
        items = load_items(self.config.data)
        validate_items_for_judge(items, self.config.judge)
        selected_specs = selected_dimensions(self.config)
        events.append(
            _event(
                "items_loaded",
                item_count=len(items),
                items_path=str(self.config.data.items_path),
            )
        )
        return _LoadedAuditInputs(
            items=items,
            selected_specs=selected_specs,
        )

    def _ensure_semantic_prerequisites(
        self, selected_specs: list[DimensionSpec]
    ) -> None:
        """Fail before baseline billing when selected semantic work cannot run.

        A mock perturbation agent is valid only when no LLM probe is selected.
        Any selected semantic work must have its explicitly configured generator
        and independent validator available rather than degrading to a local
        heuristic.
        """

        generated_specs = [spec for spec in selected_specs if spec.kind == "llm"]
        # A deterministic rewrite can still need semantic admissibility (for
        # example, a reasoning-style instruction).  Structural permutations are
        # the only deterministic probes admitted without an independent model.
        semantic_validation_specs = [
            spec
            for spec in selected_specs
            if spec.kind == "llm" or spec.validation_kind is not None
        ]
        if not semantic_validation_specs:
            return
        if generated_specs and self.generator is None:
            provider = self.config.perturbation_agent.provider
            model = self.config.perturbation_agent.model
            if model == "mock":
                names = ", ".join(spec.operator for spec in generated_specs)
                raise ValueError(
                    "Selected LLM perturbations require a non-mock perturbation_agent.model "
                    f"or an injected generator: {names}."
                )
            if not provider_has_api_key(self.config, provider):
                # Use the provider resolver for its precise, safe env-var-name
                # diagnostic (never a credential value).
                build_provider_client(self.config, provider)
        # An injected validator is an intentional dependency (common for local
        # tests and controlled studies).  Otherwise semantic generated probes
        # cannot be admitted by the deterministic rule guard.
        validator_needed = bool(semantic_validation_specs)
        if self.validator is None and validator_needed:
            provider = self.config.resolved_validator_provider()
            model = self.config.resolved_validator_model()
            if model == "mock":
                raise ValueError(
                    "Selected semantic perturbations require a non-mock independent validator model."
                )
            if not provider_has_api_key(self.config, provider):
                build_provider_client(self.config, provider)

    def _build_and_validate_perturbations(
        self,
        run_id: str,
        loaded: _LoadedAuditInputs,
        directional_items: list[Item],
        directional_eligibility: dict[str, Any],
        events: list[dict[str, Any]],
    ) -> _PerturbationPhaseResult:
        eligible_item_ids = {item.item_id for item in directional_items}
        ineligible_reasons = {
            str(decision["item_id"]): str(decision.get("reason") or "baseline_ineligible")
            for decision in directional_eligibility.get("items", [])
            if not decision.get("eligible")
        }
        generation_units = resolve_generation_units(
            loaded.selected_specs,
            loaded.items,
            directional_eligible_item_ids=eligible_item_ids,
        )
        deterministic_units = [unit for unit in generation_units if unit.spec.kind == "det"]
        llm_units = [unit for unit in generation_units if unit.spec.kind == "llm"]
        self._progress("building_perturbations", message="Building deterministic perturbations")
        deterministic, deterministic_outcomes = _build_deterministic_units(
            self.config,
            loaded.items,
            deterministic_units,
        )
        events.append(_event("deterministic_perturbations_built", count=len(deterministic)))
        self._progress(
            "generating_perturbations",
            message="Generating LLM-backed perturbations",
        )
        generated, generated_outcomes = self._generate_llm_perturbations(
            loaded.items,
            directional_items,
            llm_units,
        )
        generation_outcomes = [*deterministic_outcomes, *generated_outcomes]
        perturbations = [*deterministic, *generated]
        events.append(
            _event(
                "generated_perturbations_built",
                count=len(generated),
                skipped=sum(outcome.status == "skipped" for outcome in generation_outcomes),
                failed=sum(outcome.status == "failed" for outcome in generation_outcomes),
            )
        )
        # Keep proposed variants even when the validator or a later judge phase
        # fails.  The final public artifact still contains accepted variants only.
        self._checkpoint(
            run_id,
            "generation_units",
            [_generation_unit_record(unit) for unit in generation_units],
        )
        self._checkpoint(run_id, "generation_outcomes", generation_outcomes)
        self._checkpoint(run_id, "proposed_perturbations", perturbations)
        generation_trace = [
            _generation_trace_record(perturbation)
            for perturbation in perturbations
            if perturbation.metadata.get("generator_model")
        ]
        generation_trace.extend(_generation_skip_records(self.generator))

        # Reject an injected generator's out-of-scope, duplicate, or unselected
        # output before any semantic validator call can spend money.  Variants
        # are provisionally non-rejected here; final statuses are reconciled
        # after validation below.
        resolve_applications(
            loaded.selected_specs,
            loaded.items,
            generation_units,
            generation_outcomes,
            perturbations,
            [],
            directional_eligible_item_ids=eligible_item_ids,
            directional_ineligible_reasons=ineligible_reasons,
            repeats=self.config.audit.repeats,
        )
        self._progress("validating_perturbations", message="Validating perturbations")
        validated_perturbations, validation_report = validate_perturbations(
            perturbations,
            loaded.items,
            self.validator,
            config=self.config,
            progress_callback=lambda current, total: self._progress(
                "validation_call_completed",
                message="Validating perturbations",
                current=current,
                total=total,
            ),
        )
        validation_calls = sum(1 for p in validated_perturbations if requires_validation(p))
        accepted = [
            p for p in validated_perturbations
            if p.validity_status in {"accepted", "not_needed"}
        ]
        application_outcomes = resolve_applications(
            loaded.selected_specs,
            loaded.items,
            generation_units,
            generation_outcomes,
            validated_perturbations,
            validation_report,
            directional_eligible_item_ids=eligible_item_ids,
            directional_ineligible_reasons=ineligible_reasons,
            repeats=self.config.audit.repeats,
        )
        accepted_ids = {
            variant_id
            for application in application_outcomes
            for variant_id in application.accepted_variant_ids
        }
        accepted = [
            perturbation for perturbation in accepted
            if perturbation.variant_id in accepted_ids
        ]
        operator_manifest = build_operator_manifest(loaded.selected_specs)
        self._checkpoint(run_id, "application_outcomes", application_outcomes)
        events.append(
            _event(
                "perturbations_validated",
                validation_calls=validation_calls,
                accepted_count=len(accepted),
                rejected_count=sum(
                    1 for record in validation_report if record.get("status") == "rejected"
                ),
                unavailable_count=sum(
                    1 for record in validation_report if record.get("status") == "unavailable"
                ),
            )
        )
        llm_calls = [
            *_component_llm_calls(self.generator),
            *_component_llm_calls(self.validator),
        ]
        return _PerturbationPhaseResult(
            accepted=accepted,
            attempted=validated_perturbations,
            generated=generated,
            generation_trace=generation_trace,
            validation_report=validation_report,
            validation_calls=validation_calls,
            llm_calls=llm_calls,
            operator_manifest=operator_manifest,
            ledger=[],
            generation_outcomes=generation_outcomes,
            application_outcomes=application_outcomes,
        )

    def _run_baseline_judge_phase(
        self,
        run_id: str,
        items: list[Item],
        events: list[dict[str, Any]],
    ) -> tuple[list[JudgeResult], list[dict[str, Any]]]:
        results, judge_call_traces = self._run_judge_calls(
            items,
            [],
            include_originals=True,
            progress_prefix="baseline_",
            checkpoint_stage=(run_id, "baseline_judge_results"),
        )
        events.append(_event("baseline_judge_calls_completed", count=len(results)))
        return results, judge_call_traces

    def _run_judge_phase(
        self,
        run_id: str,
        items: list[Item],
        perturbations: list[Perturbation],
        applications: list[ApplicationOutcome],
        events: list[dict[str, Any]],
    ) -> tuple[list[JudgeResult], list[dict[str, Any]]]:
        accepted_ids = {
            variant_id
            for application in applications
            for variant_id in application.accepted_variant_ids
        }
        selected = [
            perturbation for perturbation in perturbations
            if perturbation.variant_id in accepted_ids
        ]
        if len(selected) != len(accepted_ids):
            raise ValueError("Application records reference an unknown accepted variant")
        application_pairs = [
            (application.item_id, variant_id)
            for application in applications
            for variant_id in application.accepted_variant_ids
        ]
        results, judge_call_traces = self._run_judge_calls(
            items,
            selected,
            include_originals=False,
            application_pairs=application_pairs,
            checkpoint_stage=(run_id, "perturbation_judge_results"),
        )
        events.append(_event("perturbation_judge_calls_completed", count=len(results)))
        return results, judge_call_traces

    def _finish_after_perturbations(
        self,
        *,
        run_id: str,
        loaded: _LoadedAuditInputs,
        perturbation_phase: _PerturbationPhaseResult,
        baseline_results: list[JudgeResult],
        baseline_traces: list[dict[str, Any]],
        planning_trace: dict[str, Any] | None,
        directional_eligibility: dict[str, Any],
        events: list[dict[str, Any]],
        charge_planning_call: bool = True,
        replay_provenance: dict[str, Any] | None = None,
    ) -> AuditRunResult:
        """Account for frozen applications after their fresh judge measurements."""

        perturbation_results, perturbation_traces = self._run_judge_phase(
            run_id,
            loaded.items,
            perturbation_phase.accepted,
            perturbation_phase.application_outcomes,
            events,
        )
        applications = record_judge_outcomes(
            perturbation_phase.application_outcomes,
            perturbation_results,
            self.config.judge,
        )
        perturbation_phase = replace(
            perturbation_phase,
            application_outcomes=applications,
            ledger=build_perturbation_ledger(
                loaded.selected_specs,
                perturbation_phase.attempted,
                perturbation_phase.accepted,
                perturbation_phase.validation_report,
                application_outcomes=applications,
                generation_outcomes=perturbation_phase.generation_outcomes,
            ),
        )
        self._checkpoint(run_id, "application_outcomes", applications)
        self._checkpoint(run_id, "perturbation_judge_results", perturbation_results)
        judge_results = [*baseline_results, *perturbation_results]
        judge_traces = [*baseline_traces, *perturbation_traces]
        events.append(_event("judge_calls_completed", count=len(judge_results)))
        artifact, item_risks = self._compute_metrics_phase(
            run_id,
            loaded,
            perturbation_phase,
            judge_results,
            judge_traces,
            planning_trace,
            directional_eligibility,
            events,
            charge_planning_call=charge_planning_call,
            replay_provenance=replay_provenance,
        )
        paths = self._write_artifact_phase(run_id, artifact, item_risks, judge_traces, events)
        self._progress(
            "complete",
            message="Audit replay complete" if replay_provenance is not None else "Audit complete",
        )
        return AuditRunResult(run_id=run_id, artifact=artifact, paths=paths)

    def _compute_metrics_phase(
        self,
        run_id: str,
        loaded: _LoadedAuditInputs,
        perturbation_phase: _PerturbationPhaseResult,
        judge_results: list[JudgeResult],
        judge_call_traces: list[dict[str, Any]],
        planning_trace: dict[str, Any] | None,
        directional_eligibility: dict[str, Any],
        events: list[dict[str, Any]],
        *,
        charge_planning_call: bool = True,
        replay_provenance: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], list[ItemRisk]]:
        self._progress("computing_metrics", message="Computing metrics")
        noise_floor = compute_noise_floor(judge_results, self.config.judge)
        item_risks = compute_item_risks(
            items=loaded.items,
            results=judge_results,
            perturbations=perturbation_phase.accepted,
            config=self.config,
            noise_floor=noise_floor,
        )
        attribution = summarize_attribution(
            judge_results,
            perturbation_phase.accepted,
            noise_floor,
            self.config.judge,
        )
        perturbation_attribution = summarize_perturbation_attribution(
            judge_results,
            perturbation_phase.accepted,
            noise_floor,
            self.config.judge,
        )
        directional = compute_directional_metrics(
            results=judge_results,
            perturbations=perturbation_phase.accepted,
            noise_floor=noise_floor,
            config=self.config,
        )
        directional["eligibility"] = directional_eligibility
        item_risks = apply_directional_risk(item_risks, directional)
        confidence_intervals = compute_confidence_intervals(
            item_risks,
            n_samples=self.config.audit.bootstrap_samples,
            seed=self.config.audit.bootstrap_seed,
        )
        correctness = compute_judge_correctness(
            item_risks,
            top_k=self.config.audit.top_k_routes,
            judge=self.config.judge,
        )
        application_summary = summarize_application_outcomes(
            perturbation_phase.application_outcomes,
            perturbation_phase.generation_outcomes,
        )
        coverage = compute_coverage(
            config=self.config,
            source_item_count=len(loaded.items),
            attempted_perturbations=perturbation_phase.attempted,
            accepted_perturbations=perturbation_phase.accepted,
            validation_report=perturbation_phase.validation_report,
            judge_results=judge_results,
            item_risks=item_risks,
            application_summary=application_summary,
        )
        artifact = {
            "run_id": run_id,
            "measurement_spec_version": "2.0",
            "output_semantics": {
                "output_type": self.config.judge.output_type.value,
                "score_direction": self.config.judge.score_direction.value,
                "score_min": self.config.judge.score_min,
                "score_max": self.config.judge.score_max,
                "threshold": self.config.judge.threshold,
                "invariant_tolerance": self.config.judge.invariant_tolerance,
                "gold_tolerance": self.config.judge.gold_tolerance,
            },
            "item_count": len(loaded.items),
            "perturbation_count": len(perturbation_phase.accepted),
            "judge_results": [result.model_dump(mode="json") for result in judge_results],
            "attempted_perturbations": [
                perturbation.model_dump(mode="json")
                for perturbation in perturbation_phase.attempted
            ],
            "perturbations": [
                perturbation.model_dump(mode="json")
                for perturbation in perturbation_phase.accepted
            ],
            "operator_manifest": perturbation_phase.operator_manifest,
            "perturbation_ledger": perturbation_phase.ledger,
            "validation_report": perturbation_phase.validation_report,
            "planning_trace": planning_trace,
            "generation_trace": perturbation_phase.generation_trace,
            "generation_outcomes": [
                asdict(outcome) for outcome in perturbation_phase.generation_outcomes
            ],
            "application_outcomes": [
                asdict(outcome) for outcome in perturbation_phase.application_outcomes
            ],
            "application_summary": application_summary,
            "noise_floor": noise_floor,
            "item_risks": [risk.model_dump(mode="json") for risk in item_risks],
            "attribution": attribution,
            "perturbation_attribution": perturbation_attribution,
            "confidence_intervals": confidence_intervals,
            "correctness": correctness,
            "directional": directional,
            "coverage": coverage,
            **({"replay": replay_provenance} if replay_provenance is not None else {}),
        }
        artifact["report_interpretation"] = build_report_interpretation(
            self.config,
            artifact,
            judge_call_traces,
        )
        llm_calls: list[dict[str, Any]] = []
        if (
            charge_planning_call
            and planning_trace is not None
            and planning_trace.get("llm_call") is not None
        ):
            llm_calls.append(_copy_llm_call(planning_trace["llm_call"]))
        llm_calls.extend(perturbation_phase.llm_calls)
        llm_calls.extend(_judge_llm_calls(judge_results, self.config.judge))
        reporting_call = artifact["report_interpretation"].get("llm_call")
        if reporting_call is not None:
            llm_calls.append(_copy_llm_call(reporting_call))
        artifact["llm_calls"] = llm_calls
        artifact["cost"] = compute_cost(
            item_risks=item_risks,
            call_records=llm_calls,
        )
        events.append(_event("metrics_computed"))
        return artifact, item_risks

    def _write_artifact_phase(
        self,
        run_id: str,
        artifact: dict[str, Any],
        item_risks: list[ItemRisk],
        judge_call_traces: list[dict[str, Any]],
        events: list[dict[str, Any]],
    ) -> dict[str, Path]:
        self._progress("writing_artifacts", message="Writing artifacts")
        report_markdown = render_markdown_report(
            self.config,
            artifact,
            judge_call_traces=judge_call_traces,
        )
        return write_artifacts(
            run_id=run_id,
            runs_dir=self.config.output.runs_dir,
            artifact=artifact,
            report_markdown=report_markdown,
            risks=item_risks,
            config=self.config,
            events=events,
            judge_call_traces=judge_call_traces,
        )

    def _generate_llm_perturbations(
        self,
        items: list[Item],
        directional_items: list[Item],
        units: list[GenerationUnit],
    ) -> tuple[list[Perturbation], list[GenerationOutcome]]:
        if not units:
            return [], []
        if self.generator is None:
            return [], [
                GenerationOutcome(
                    unit.spec.operator,
                    unit.unit_id,
                    unit.item_id,
                    "failed",
                    "generator_unavailable",
                )
                for unit in units
            ]
        # The progress hook is duck-typed so custom generators that don't expose
        # it still satisfy the PerturbationGenerator protocol.
        if hasattr(self.generator, "progress_callback"):
            self.generator.progress_callback = lambda current, total: self._progress(
                "generation_unit_completed",
                message="Generating LLM-backed perturbations",
                current=current,
                total=total,
            )
        generate_units = getattr(self.generator, "generate_units", None)
        if callable(generate_units):
            variants, outcomes = generate_units(self.config, units)
            return list(variants), list(outcomes)
        variants = list(
            self.generator.generate(
                self.config,
                items,
                directional_items=directional_items,
            )
        )
        return variants, _legacy_generator_outcomes(units, variants, self.generator)

    def _plan_judge_calls(
        self,
        items: list[Item],
        perturbations: list[Perturbation],
        *,
        include_originals: bool = True,
        application_pairs: list[tuple[str, str]] | None = None,
    ) -> list[_PlannedCall]:
        """Enumerate a phase's judge calls in deterministic order.

        Each original item is judged ``repeats`` times to set the noise floor;
        each perturbation variant is judged the same number of times so its flip
        rate is comparable to that floor rather than a single noisy draw.
        """

        plans: list[_PlannedCall] = []
        if include_originals:
            for item in items:
                for repeat in range(self.config.audit.repeats):
                    plans.append(
                        _PlannedCall(
                            call_type="original_repeat",
                            original_item=item,
                            submitted_item=item,
                            submitted_config=self.config.judge,
                            perturbation=None,
                            run_id=f"original-{repeat + 1}",
                        )
                    )

        item_by_id = {item.item_id: item for item in items}
        perturbation_by_id = {perturbation.variant_id: perturbation for perturbation in perturbations}
        for item_id, variant_id in application_pairs or []:
            item = item_by_id.get(item_id)
            perturbation = perturbation_by_id.get(variant_id)
            if item is None or perturbation is None:
                raise ValueError("Application judge plan references an unknown item or variant")
            if perturbation.item_id not in {"*", item_id}:
                raise ValueError("Application judge plan expands a variant outside its item scope")
            perturbed_config = apply_judge_overrides(
                self.config.judge,
                perturbation.judge_config_overrides,
            )
            perturbed_item = apply_item_overrides(item, perturbation.item_overrides)
            for repeat in range(self.config.audit.repeats):
                plans.append(
                    _PlannedCall(
                        call_type="perturbation",
                        original_item=item,
                        submitted_item=perturbed_item,
                        submitted_config=perturbed_config,
                        perturbation=perturbation,
                        run_id=f"{perturbation.variant_id}-{repeat + 1}",
                    )
                )
        return plans

    def _run_judge_calls(
        self,
        items: list[Item],
        perturbations: list[Perturbation],
        *,
        include_originals: bool = True,
        application_pairs: list[tuple[str, str]] | None = None,
        progress_prefix: str = "",
        checkpoint_stage: tuple[str, str] | None = None,
    ) -> tuple[list[JudgeResult], list[dict]]:
        plans = self._plan_judge_calls(
            items,
            perturbations,
            include_originals=include_originals,
            application_pairs=application_pairs,
        )
        total_calls = len(plans)
        completed_calls = 0
        message = "Running baseline judge calls" if progress_prefix else "Running judge calls"
        self._progress(
            f"{progress_prefix}judge_calls_started",
            message=message,
            current=0,
            total=total_calls,
        )
        # Pre-sized so each call writes its own slot: order stays identical to the
        # plan regardless of how workers interleave, keeping artifacts stable.
        results: list[JudgeResult | None] = [None] * len(plans)
        traces: list[dict[str, Any] | None] = [None] * len(plans)
        checkpoint_lock = Lock()
        workers = max(1, self.config.audit.num_workers)

        def execute(index: int) -> None:
            plan = plans[index]
            result = self.runner.run(
                item=plan.submitted_item,
                judge_config=plan.submitted_config,
                perturbation=plan.perturbation,
                run_id=plan.run_id,
            )
            results[index] = result
            traces[index] = _judge_call_trace(
                call_type=plan.call_type,
                original_item=plan.original_item,
                submitted_item=plan.submitted_item,
                base_config=self.config.judge,
                submitted_config=plan.submitted_config,
                perturbation=plan.perturbation,
                result=result,
            )
            if checkpoint_stage is not None:
                # A run can fail after any remote call.  Serialize snapshots so
                # concurrent workers never interleave partial JSONL writes.
                with checkpoint_lock:
                    append_stage_record(
                        run_id=checkpoint_stage[0],
                        runs_dir=self.config.output.runs_dir,
                        config=self.config,
                        stage=checkpoint_stage[1],
                        record=result.model_dump(mode="json"),
                    )

        def mark_complete() -> None:
            nonlocal completed_calls
            completed_calls += 1
            self._progress(
                f"{progress_prefix}judge_call_completed",
                message=message,
                current=completed_calls,
                total=total_calls,
            )

        if workers == 1:
            for index in range(len(plans)):
                execute(index)
                mark_complete()
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(execute, index) for index in range(len(plans))]
                for future in as_completed(futures):
                    future.result()
                    mark_complete()

        self._progress(
            f"{progress_prefix}judge_calls_complete",
            message=(
                "Baseline judge calls complete"
                if progress_prefix
                else "Judge calls complete"
            ),
            current=completed_calls,
            total=total_calls,
        )

        return results, traces  # type: ignore[return-value]


def _llm_perturbations_enabled(config: AuditRunConfig) -> bool:
    """True when an independent agent model is configured and some enabled family
    contains an LLM-backed dimension to generate."""

    return config.perturbation_agent.model != "mock" and bool(llm_dimensions(config))


def _default_planner(config: AuditRunConfig) -> PerturbationPlanner | None:
    """Build the suggester only when auto-routing is enabled.

    The suggester uses its own agent config when present, otherwise it inherits the
    perturbation-agent model (independent of the judge). A missing model or key is
    a hard error rather than a skip: silently leaving the allow-list unrouted
    would defeat the point of asking for independent suggestions.
    """

    if not config.audit.plan_operators:
        return None
    planner_model = config.resolved_planner_model()
    planner_provider = config.resolved_planner_provider()
    if planner_model == "mock":
        raise ValueError(
            "audit.plan_operators requires a non-mock Perturbation Suggester model "
            "(set planner_agent.model or perturbation_agent.model)."
        )
    return OpenAIPlannerAgent(
        build_provider_client(config, planner_provider),
        planner_model,
        config.resolved_planner_decoding_params(),
        provider=planner_provider,
    )


def _default_generator(config: AuditRunConfig) -> PerturbationGenerator | None:
    """Build the perturbation agent only when LLM dimensions are enabled, an agent
    model is configured, and a key is present. Otherwise only deterministic
    dimensions run."""

    provider = config.perturbation_agent.provider
    if not _llm_perturbations_enabled(config):
        return None
    if not provider_has_api_key(config, provider):
        build_provider_client(config, provider)
        raise AssertionError("provider client unexpectedly resolved without credentials")
    return OpenAIPerturbationAgent(
        build_provider_client(config, provider),
        config.perturbation_agent.model,
        config.perturbation_agent.decoding_params,
        provider=provider,
    )


def _default_validator(config: AuditRunConfig) -> PerturbationValidator:
    """Independent validator -- the perturbation agent's model, never the judge.

    Uses an LLM check whenever a selected operator requires semantic
    admissibility. The deterministic rule guardrails are reserved for selected
    structural operators, whose validity they can actually establish.
    """

    validator_model = config.resolved_validator_model()
    validator_provider = config.resolved_validator_provider()
    semantic_validation_selected = any(
        spec.kind == "llm" or spec.validation_kind is not None
        for spec in selected_dimensions(config)
    )
    if semantic_validation_selected:
        if validator_model == "mock":
            raise ValueError(
                "Selected semantic perturbations require a non-mock independent validator model."
            )
        if provider_has_api_key(config, validator_provider):
            return LLMEquivalenceValidator(
                build_provider_client(config, validator_provider),
                validator_model,
                config.resolved_validator_decoding_params(),
                provider=validator_provider,
            )
        build_provider_client(config, validator_provider)
        raise AssertionError("provider client unexpectedly resolved without credentials")
    return RuleEquivalenceValidator()


def _event(event: str, **details: Any) -> dict[str, Any]:
    return {"timestamp": _utc_timestamp(), "event": event, **details}


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _new_run_id() -> str:
    """Human-sortable ID with entropy for concurrent UI/CLI runs."""

    return f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}-{uuid4().hex[:8]}"


def _build_deterministic_units(
    config: AuditRunConfig,
    items: list[Item],
    units: list[GenerationUnit],
) -> tuple[list[Perturbation], list[GenerationOutcome]]:
    """Build selected rule transforms one canonical unit at a time."""

    variants: list[Perturbation] = []
    outcomes: list[GenerationOutcome] = []
    for unit in units:
        try:
            built = build_perturbations(
                config,
                items,
                generation_units=[unit],
            )
        except Exception as exc:
            outcomes.append(
                GenerationOutcome(
                    unit.spec.operator,
                    unit.unit_id,
                    unit.item_id,
                    "failed",
                    f"deterministic_builder_error:{type(exc).__name__}",
                )
            )
            continue
        variants.extend(built)
        outcomes.append(
            GenerationOutcome(
                unit.spec.operator,
                unit.unit_id,
                unit.item_id,
                "generated" if built else "skipped",
                None if built else "no_applicable_transform",
                [variant.variant_id for variant in built],
            )
        )
    return variants, outcomes


def _generation_unit_record(unit: GenerationUnit) -> dict[str, Any]:
    """Persist scope evidence without serializing a builder function."""

    return {
        "unit_id": unit.unit_id,
        "operator": unit.spec.operator,
        "item_id": unit.item_id,
        "target_item_ids": unit.target_item_ids,
        "generation_method": "rule" if unit.spec.kind == "det" else "llm",
    }


def _legacy_generator_outcomes(
    units: list[GenerationUnit],
    variants: list[Perturbation],
    generator: PerturbationGenerator,
) -> list[GenerationOutcome]:
    """Adapt the original batch generator protocol without hiding omissions."""

    variants_by_scope: dict[tuple[str, str], list[Perturbation]] = {}
    for variant in variants:
        variants_by_scope.setdefault((variant.operator, variant.item_id), []).append(variant)
    skips_by_scope: dict[tuple[str, str], dict[str, Any]] = {}
    raw_skips = getattr(generator, "generation_skips", [])
    if isinstance(raw_skips, list):
        for record in raw_skips:
            if not isinstance(record, dict):
                continue
            operator = record.get("operator")
            item_id = record.get("item_id")
            status = record.get("status")
            if (
                isinstance(operator, str)
                and isinstance(item_id, str)
                and status in {"skipped", "failed"}
            ):
                skips_by_scope[(operator, item_id)] = record
    outcomes: list[GenerationOutcome] = []
    for unit in units:
        scope = (unit.spec.operator, unit.item_id)
        generated = variants_by_scope.get(scope, [])
        if generated:
            outcomes.append(
                GenerationOutcome(
                    unit.spec.operator,
                    unit.unit_id,
                    unit.item_id,
                    "generated",
                    variant_ids=[variant.variant_id for variant in generated],
                )
            )
            continue
        skipped = skips_by_scope.get(scope)
        if skipped is not None:
            outcomes.append(
                GenerationOutcome(
                    unit.spec.operator,
                    unit.unit_id,
                    unit.item_id,
                    skipped["status"],
                    str(skipped.get("reason") or skipped["status"]),
                )
            )
            continue
        outcomes.append(
            GenerationOutcome(
                unit.spec.operator,
                unit.unit_id,
                unit.item_id,
                "failed",
                "generator_returned_no_outcome",
            )
        )
    return outcomes


def _generation_trace_record(perturbation) -> dict[str, Any]:
    return {
        "timestamp": _utc_timestamp(),
        "variant_id": perturbation.variant_id,
        "item_id": perturbation.item_id,
        "family": perturbation.family.value,
        "operator": perturbation.operator,
        "expected_effect": perturbation.expected_effect.value,
        "expected_relation": perturbation.expected_relation.value,
        "changed_fields": perturbation.changed_fields,
        "metadata": perturbation.metadata,
        "judge_config_overrides": perturbation.judge_config_overrides,
        "item_overrides": perturbation.item_overrides,
    }


def _generation_skip_records(generator: PerturbationGenerator | None) -> list[dict[str, Any]]:
    if generator is None:
        return []
    records = getattr(generator, "generation_skips", None)
    if not records:
        return []
    return [{"timestamp": _utc_timestamp(), **record} for record in records]


def _component_llm_calls(component: Any) -> list[dict[str, Any]]:
    """Snapshot actual API attempts exposed by an LLM-backed pipeline role."""

    records = getattr(component, "call_records", None)
    if not isinstance(records, list):
        return []
    return [_copy_llm_call(record) for record in records if isinstance(record, dict)]


def _judge_llm_calls(
    results: list[JudgeResult],
    judge: JudgeConfig,
) -> list[dict[str, Any]]:
    """Project judge results into the same provider-neutral call ledger.

    The local mock runner is deliberately excluded: it produces judge results,
    but does not make an LLM API call and therefore has no token cost.
    """

    calls: list[dict[str, Any]] = []
    for result in results:
        metadata = result.metadata or {}
        model = str(metadata.get("model") or judge.model)
        if model == "mock":
            continue
        usage = metadata.get("usage")
        calls.append(
            {
                "category": "judge",
                "provider": str(metadata.get("provider") or judge.provider),
                "model": model,
                "status": result.call_status,
                "usage": dict(usage) if isinstance(usage, dict) else {},
                "item_id": result.item_id,
                "variant_id": result.variant_id,
                "run_id": result.run_id,
            }
        )
    return calls


def _copy_llm_call(record: dict[str, Any]) -> dict[str, Any]:
    """Copy a ledger record, including its nested usage map, for the artifact."""

    copied = dict(record)
    if isinstance(record.get("usage"), dict):
        copied["usage"] = dict(record["usage"])
    return copied


def _judge_call_trace(
    *,
    call_type: str,
    original_item,
    submitted_item,
    base_config,
    submitted_config,
    perturbation,
    result: JudgeResult,
) -> dict[str, Any]:
    return {
        "timestamp": _utc_timestamp(),
        "call_type": call_type,
        "item_id": original_item.item_id,
        "submitted_item_id": submitted_item.item_id,
        "variant_id": perturbation.variant_id if perturbation else "original",
        "run_id": result.run_id,
        "family": perturbation.family.value if perturbation else "noise",
        "operator": perturbation.operator if perturbation else "repeat",
        "expected_effect": (
            perturbation.expected_effect.value if perturbation else "same_verdict"
        ),
        "expected_relation": (
            perturbation.expected_relation.value if perturbation else "invariant"
        ),
        "validity_status": perturbation.validity_status if perturbation else "not_needed",
        "changed_fields": perturbation.changed_fields if perturbation else [],
        "field_changes": (
            _field_changes(original_item, submitted_item, base_config, submitted_config, perturbation)
            if perturbation
            else []
        ),
        "judge_config_overrides": perturbation.judge_config_overrides if perturbation else {},
        "item_overrides": perturbation.item_overrides if perturbation else {},
        "rendered_judge": {
            "instructions": build_judge_instructions(submitted_config),
            "input": build_judge_input(submitted_item),
        },
        "result": result.model_dump(mode="json"),
    }


def _field_changes(
    original_item,
    submitted_item,
    base_config,
    submitted_config,
    perturbation,
) -> list[dict[str, Any]]:
    before_payloads = {
        "item": original_item.model_dump(mode="json"),
        "judge": base_config.model_dump(mode="json"),
    }
    after_payloads = {
        "item": submitted_item.model_dump(mode="json"),
        "judge": submitted_config.model_dump(mode="json"),
    }
    changes: list[dict[str, Any]] = []
    for field in perturbation.changed_fields:
        root, _, path = field.partition(".")
        changes.append(
            {
                "field": field,
                "before": _nested_lookup(before_payloads.get(root, {}), path),
                "after": _nested_lookup(after_payloads.get(root, {}), path),
            }
        )
    return changes


def _nested_lookup(payload: Any, dotted_path: str) -> Any:
    current = payload
    for part in dotted_path.split("."):
        if not part:
            continue
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current
