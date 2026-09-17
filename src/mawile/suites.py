"""Self-contained, integrity-checked perturbation suites for replay."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

from mawile.applications import (
    ApplicationOutcome,
    GenerationOutcome,
    resolve_applications,
    resolve_generation_units,
)
from mawile.measurement import directional_baseline_eligibility, validate_items_for_judge
from mawile.perturbations.registry import selected_dimensions
from mawile.schemas import AuditRunConfig, Item, JudgeResult, Perturbation


SUITE_VERSION = "1.0"


@dataclass(frozen=True)
class AuditSuite:
    """Validated immutable source evidence used to replay judge measurements."""

    payload: dict[str, Any]
    source_config: AuditRunConfig
    items: list[Item]
    baseline_results: list[JudgeResult]
    generated: list[Perturbation]
    attempted: list[Perturbation]
    accepted: list[Perturbation]
    generation_outcomes: list[GenerationOutcome]
    application_outcomes: list[ApplicationOutcome]
    validation_report: list[dict[str, Any]]
    operator_manifest: dict[str, Any]

    @property
    def content_sha256(self) -> str:
        return str(self.payload["content_sha256"])

    @property
    def planning_trace(self) -> dict[str, Any] | None:
        value = self.payload.get("planning_trace")
        return value if isinstance(value, dict) else None

    @property
    def directional_eligibility(self) -> dict[str, Any]:
        value = self.payload.get("directional_eligibility")
        return dict(value) if isinstance(value, Mapping) else {}

    @property
    def generation_trace(self) -> list[dict[str, Any]]:
        value = self.payload.get("generation_trace")
        return [dict(row) for row in value] if isinstance(value, list) else []

    @property
    def provenance(self) -> dict[str, Any]:
        value = self.payload.get("provenance")
        return dict(value) if isinstance(value, Mapping) else {}


def build_suite(
    *,
    run_id: str,
    config: AuditRunConfig,
    items: list[Item],
    baseline_results: list[JudgeResult],
    generated: list[Perturbation],
    attempted: list[Perturbation],
    accepted: list[Perturbation],
    generation_outcomes: list[GenerationOutcome],
    application_outcomes: list[ApplicationOutcome],
    validation_report: list[dict[str, Any]],
    operator_manifest: dict[str, Any],
    planning_trace: dict[str, Any] | None,
    directional_eligibility: dict[str, Any],
    generation_trace: list[dict[str, Any]],
) -> dict[str, Any]:
    """Capture all source evidence needed to rerun only judge measurements."""

    payload: dict[str, Any] = {
        "suite_version": SUITE_VERSION,
        "provenance": {
            "source_run_id": run_id,
            "measurement_spec_version": "2.0",
            "historical_llm_calls_included_in_replay_cost": False,
        },
        "source_config": config.model_dump(mode="json"),
        "items": [item.model_dump(mode="json") for item in items],
        "selected_operator_ids": [spec.operator for spec in selected_dimensions(config)],
        "planning_trace": planning_trace,
        "directional_eligibility": directional_eligibility,
        "source_baseline_results": [result.model_dump(mode="json") for result in baseline_results],
        "generated_perturbations": [row.model_dump(mode="json") for row in generated],
        "attempted_perturbations": [row.model_dump(mode="json") for row in attempted],
        "accepted_perturbations": [row.model_dump(mode="json") for row in accepted],
        "generation_outcomes": [asdict(row) for row in generation_outcomes],
        "application_outcomes": [asdict(row) for row in application_outcomes],
        "validation_report": validation_report,
        # This is source evidence, not a live view of the registry.  A future
        # registry description must not rewrite what the saved study claimed.
        "operator_manifest": operator_manifest,
        "generation_trace": generation_trace,
    }
    return _with_hash(payload)


def write_suite(run_dir: Path, suite: Mapping[str, Any]) -> Path:
    """Write a canonical immutable suite before perturbed judge calls begin."""

    verified = _verify_hash(dict(suite))
    path = run_dir / "suite.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(_canonical_json(verified))
    return path


def load_suite(path: Path) -> AuditSuite:
    """Load and strictly validate an existing suite without accessing its dataset."""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot load suite {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("Suite must contain a JSON object")
    raw = _verify_hash(raw)
    if raw.get("suite_version") != SUITE_VERSION:
        raise ValueError(f"Unsupported suite_version: {raw.get('suite_version')!r}")
    try:
        source_config = AuditRunConfig.model_validate(raw["source_config"])
        items = [Item.model_validate(row) for row in _rows(raw, "items")]
        validate_items_for_judge(items, source_config.judge)
        baseline = [JudgeResult.model_validate(row) for row in _rows(raw, "source_baseline_results")]
        generated = [Perturbation.model_validate(row) for row in _rows(raw, "generated_perturbations")]
        attempted = [Perturbation.model_validate(row) for row in _rows(raw, "attempted_perturbations")]
        accepted = [Perturbation.model_validate(row) for row in _rows(raw, "accepted_perturbations")]
        generation = [GenerationOutcome(**row) for row in _rows(raw, "generation_outcomes")]
        applications = [ApplicationOutcome(**row) for row in _rows(raw, "application_outcomes")]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid suite records: {exc}") from exc
    validation = _rows(raw, "validation_report")
    eligibility = raw.get("directional_eligibility")
    if not isinstance(eligibility, Mapping):
        raise ValueError("Suite directional_eligibility must be an object")
    provenance = raw.get("provenance")
    _validate_provenance(provenance)
    operator_manifest = raw.get("operator_manifest")
    _validate_operator_manifest(operator_manifest, raw.get("selected_operator_ids"))
    item_ids = {item.item_id for item in items}
    if any(result.variant_id != "original" or result.item_id not in item_ids for result in baseline):
        raise ValueError("Suite source_baseline_results must contain only original results for saved items")
    _validate_eligibility(eligibility, items, baseline, source_config)
    selected = selected_dimensions(source_config)
    if raw.get("selected_operator_ids") != [spec.operator for spec in selected]:
        raise ValueError("Suite selected operators do not match its source config")
    eligible = {
        str(row["item_id"])
        for row in eligibility.get("items", [])
        if isinstance(row, Mapping) and row.get("eligible")
    }
    reasons = {
        str(row["item_id"]): str(row.get("reason") or "baseline_ineligible")
        for row in eligibility.get("items", [])
        if isinstance(row, Mapping) and not row.get("eligible")
    }
    units = resolve_generation_units(selected, items, directional_eligible_item_ids=eligible)
    resolved = resolve_applications(
        selected,
        items,
        units,
        generation,
        attempted,
        validation,
        directional_eligible_item_ids=eligible,
        directional_ineligible_reasons=reasons,
        repeats=source_config.audit.repeats,
    )
    if _application_static_rows(resolved) != _application_static_rows(applications):
        raise ValueError("Suite application outcomes do not match saved generation/validation evidence")
    _validate_variant_membership(generated, attempted, accepted, applications)
    return AuditSuite(
        raw, source_config, items, baseline, generated, attempted, accepted,
        generation, applications, validation, operator_manifest,
    )


def validate_replay_config(suite: AuditSuite, config: AuditRunConfig) -> None:
    """Allow operational replay settings while guarding frozen study semantics."""

    source = _frozen_config(suite.source_config)
    requested = _frozen_config(config)
    changed_sections = [key for key in source if source[key] != requested[key]]
    if changed_sections:
        raise ValueError(
            "Replay config changes frozen suite semantics in "
            + ", ".join(changed_sections)
            + "; only judge model/provider/decoding, "
            "repeats/workers/bootstrap/reporting/output/providers may differ."
        )


def reset_application_counts(
    applications: list[ApplicationOutcome], repeats: int
) -> list[ApplicationOutcome]:
    return [
        replace(
            row,
            expected_judge_calls=len(row.accepted_variant_ids) * repeats,
            attempted_judge_calls=0,
            analyzable_judge_calls=0,
        )
        for row in applications
    ]


def _frozen_config(config: AuditRunConfig) -> dict[str, Any]:
    raw = config.model_dump(mode="json")
    judge = raw["judge"]
    for key in ("model", "provider", "decoding_params"):
        judge.pop(key, None)
    audit = raw["audit"]
    for key in ("repeats", "num_workers", "bootstrap_samples", "bootstrap_seed", "top_k_routes"):
        audit.pop(key, None)
    raw.pop("output", None)
    raw.pop("providers", None)
    raw.pop("summary_agent", None)
    return raw


def _application_static_rows(rows: list[ApplicationOutcome]) -> list[dict[str, Any]]:
    return [
        {
            **asdict(row),
            "attempted_judge_calls": 0,
            "analyzable_judge_calls": 0,
        }
        for row in rows
    ]


def _validate_variant_membership(
    generated: list[Perturbation],
    attempted: list[Perturbation],
    accepted: list[Perturbation],
    applications: list[ApplicationOutcome],
) -> None:
    attempted_by_id = {row.variant_id: row for row in attempted}
    if len(attempted_by_id) != len(attempted):
        raise ValueError("Suite attempted perturbations contain duplicate IDs")
    generated_ids = [row.variant_id for row in generated]
    if len(generated_ids) != len(set(generated_ids)):
        raise ValueError("Suite generated perturbations contain duplicate IDs")
    for row in generated:
        attempted_row = attempted_by_id.get(row.variant_id)
        if attempted_row is None or _without_validity(attempted_row) != _without_validity(row):
            raise ValueError("Suite generated perturbations do not match attempted evidence")
    accepted_by_id = {row.variant_id: row for row in accepted}
    if len(accepted_by_id) != len(accepted):
        raise ValueError("Suite accepted perturbations contain duplicate IDs")
    expected = {
        row.variant_id: row
        for row in attempted
        if row.validity_status in {"accepted", "not_needed"}
    }
    if accepted_by_id != expected:
        raise ValueError("Suite accepted perturbations do not exactly match accepted attempted evidence")
    application_ids = [
        variant_id for row in applications for variant_id in row.accepted_variant_ids
    ]
    if set(application_ids) != set(accepted_by_id):
        raise ValueError("Suite accepted perturbations do not match application outcomes")


def _validate_provenance(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("Suite provenance must be an object")
    source_run_id = value.get("source_run_id")
    if not isinstance(source_run_id, str) or not source_run_id:
        raise ValueError("Suite provenance must include a non-empty source_run_id")
    if value.get("measurement_spec_version") != "2.0":
        raise ValueError("Suite provenance must declare measurement_spec_version 2.0")
    if value.get("historical_llm_calls_included_in_replay_cost") is not False:
        raise ValueError("Suite provenance must exclude historical LLM costs")


def _validate_operator_manifest(manifest: Any, selected_ids: Any) -> None:
    if not isinstance(manifest, dict) or not isinstance(manifest.get("operators"), list):
        raise ValueError("Suite operator_manifest must contain an operators list")
    rows = manifest["operators"]
    if not all(isinstance(row, Mapping) and isinstance(row.get("operator"), str) for row in rows):
        raise ValueError("Suite operator_manifest rows must identify their operator")
    operators = [row["operator"] for row in rows]
    if len(operators) != len(set(operators)) or operators != selected_ids:
        raise ValueError("Suite operator_manifest does not match selected operators")


def _without_validity(perturbation: Perturbation) -> dict[str, Any]:
    raw = perturbation.model_dump(mode="json")
    raw.pop("validity_status", None)
    return raw


def _validate_eligibility(
    saved: Mapping[str, Any],
    items: list[Item],
    baseline: list[JudgeResult],
    config: AuditRunConfig,
) -> None:
    rows = saved.get("items")
    if not isinstance(rows, list) or len(rows) != len(items):
        raise ValueError("Suite directional eligibility must contain exactly one decision per item")
    if any(not isinstance(row, Mapping) or not isinstance(row.get("eligible"), bool) for row in rows):
        raise ValueError("Suite directional eligibility decisions must have boolean eligible values")
    if {str(row.get("item_id")) for row in rows} != {item.item_id for item in items}:
        raise ValueError("Suite directional eligibility decisions do not match saved items")
    derived = directional_baseline_eligibility(items, baseline, config.judge)
    saved_core = {key: value for key, value in saved.items() if key != "directional_requested"}
    if saved_core != derived:
        raise ValueError("Suite directional eligibility does not match its source baseline")


def _rows(raw: Mapping[str, Any], name: str) -> list[dict[str, Any]]:
    value = raw.get(name)
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"Suite {name} must be a list of objects")
    return value


def _with_hash(payload: dict[str, Any]) -> dict[str, Any]:
    payload = dict(payload)
    payload.pop("content_sha256", None)
    payload["content_sha256"] = sha256(_canonical_json(payload)).hexdigest()
    return payload


def _verify_hash(payload: dict[str, Any]) -> dict[str, Any]:
    expected = payload.get("content_sha256")
    if not isinstance(expected, str):
        raise ValueError("Suite is missing content_sha256")
    unsigned = dict(payload)
    unsigned.pop("content_sha256", None)
    actual = sha256(_canonical_json(unsigned)).hexdigest()
    if actual != expected:
        raise ValueError("Suite content_sha256 does not match its contents")
    return payload


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


__all__ = [
    "AuditSuite",
    "SUITE_VERSION",
    "build_suite",
    "load_suite",
    "reset_application_counts",
    "validate_replay_config",
    "write_suite",
]
