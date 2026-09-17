from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError
import yaml

from mawile.io import load_json, load_jsonl
from mawile.project import PROJECT_ROOT, discover_run_artifacts
from mawile.schemas import AuditRunConfig

ARTIFACT_FILES = {
    "run-json": ("run.json", "application/json"),
    "report": ("report.md", "text/markdown"),
    "item-risks": ("item_risks.csv", "text/csv"),
    "config": ("config.resolved.yaml", "application/x-yaml"),
    "suite": ("suite.json", "application/json"),
    "perturbations": ("perturbations.jsonl", "application/jsonl"),
    "judge-results": ("judge_results.jsonl", "application/jsonl"),
    "judge-calls": ("traces/judge_calls.jsonl", "application/jsonl"),
    "llm-calls": ("traces/llm_calls.jsonl", "application/jsonl"),
    "generation-calls": ("traces/generation_calls.jsonl", "application/jsonl"),
    "validation-calls": ("traces/validation_calls.jsonl", "application/jsonl"),
    "events": ("logs/events.jsonl", "application/jsonl"),
}


@dataclass
class RunBundle:
    run_id: str
    run_dir: Path
    artifact: dict[str, Any]
    report_text: str
    traces: list[dict[str, Any]]
    paths: dict[str, Path]
    config: AuditRunConfig | None


def load_run_bundle(run_dir: Path) -> RunBundle:
    run_dir = run_dir.resolve()
    artifact = load_json(run_dir / "run.json")
    if not isinstance(artifact, dict):
        raise ValueError("Saved run.json must contain a JSON object")
    report_path = run_dir / "report.md"
    trace_path = run_dir / "traces/judge_calls.jsonl"
    config_path = run_dir / "config.resolved.yaml"
    paths = {
        "run_dir": run_dir,
        "json": run_dir / "run.json",
        "report": report_path,
        "item_risks_csv": run_dir / "item_risks.csv",
        "resolved_config": config_path,
        "suite": run_dir / "suite.json",
        "perturbations_jsonl": run_dir / "perturbations.jsonl",
        "judge_results_jsonl": run_dir / "judge_results.jsonl",
        "judge_call_trace": trace_path,
        "llm_call_trace": run_dir / "traces/llm_calls.jsonl",
    }
    config = None
    if config_path.exists():
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        config = _load_saved_config(raw)
    return RunBundle(
        run_id=str(artifact.get("run_id") or run_dir.name),
        run_dir=run_dir,
        artifact=artifact,
        report_text=report_path.read_text(encoding="utf-8") if report_path.exists() else "",
        traces=load_jsonl(trace_path),
        paths=paths,
        config=config,
    )


def _load_saved_config(raw: Any) -> AuditRunConfig:
    """Validate known saved-config fields while ignoring fields from other versions.

    New configs remain strict through :func:`mawile.config.load_config`. Completed
    runs are historical artifacts, though, and must stay viewable when a newer or
    older MAWILE version has fields that the current schema does not recognize.
    """

    while True:
        try:
            return AuditRunConfig.model_validate(raw)
        except ValidationError as exc:
            extra_locations = [
                error["loc"]
                for error in exc.errors()
                if error.get("type") == "extra_forbidden"
            ]
            if not extra_locations:
                raise
            removed = False
            for location in sorted(extra_locations, key=len, reverse=True):
                removed = _remove_value_at_location(raw, location) or removed
            if not removed:
                raise


def _remove_value_at_location(value: Any, location: tuple[Any, ...]) -> bool:
    if not location:
        return False
    parent = value
    for part in location[:-1]:
        if isinstance(parent, dict) and part in parent:
            parent = parent[part]
        elif isinstance(parent, list) and isinstance(part, int) and part < len(parent):
            parent = parent[part]
        else:
            return False
    final = location[-1]
    if isinstance(parent, dict) and final in parent:
        del parent[final]
        return True
    if isinstance(parent, list) and isinstance(final, int) and final < len(parent):
        del parent[final]
        return True
    return False


def discover_run_dir(run_id: str) -> Path | None:
    for run_json in discover_run_artifacts():
        if run_json.parent.name == run_id:
            return run_json.parent
    candidate = PROJECT_ROOT / "runs" / run_id
    if (candidate / "run.json").exists():
        return candidate
    return None


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True
