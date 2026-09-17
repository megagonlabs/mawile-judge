from __future__ import annotations

import json
from pathlib import Path

import yaml

from mawile.io import write_json, write_jsonl, write_risk_csv
from mawile.schemas import AuditRunConfig, ItemRisk


def write_stage_snapshot(
    *,
    run_id: str,
    runs_dir: Path,
    config: AuditRunConfig,
    stage: str,
    records: list[dict],
) -> Path:
    """Persist completed phase output before a later paid phase can fail.

    These intentionally small, append-free snapshots are not a resume system:
    they are a durable forensic record for an incomplete run.  The final
    artifact writer remains the sole producer of the public run bundle.
    """

    run_dir = runs_dir / run_id
    stages_dir = run_dir / "stages"
    stages_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.resolved.yaml"
    if not config_path.exists():
        config_path.write_text(
            yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False),
            encoding="utf-8",
        )
    path = stages_dir / f"{stage}.jsonl"
    write_jsonl(path, records)
    return path


def append_stage_record(
    *,
    run_id: str,
    runs_dir: Path,
    config: AuditRunConfig,
    stage: str,
    record: dict,
) -> Path:
    """Append one completed remote-call result under the caller's write lock."""

    run_dir = runs_dir / run_id
    stages_dir = run_dir / "stages"
    stages_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.resolved.yaml"
    if not config_path.exists():
        config_path.write_text(
            yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False),
            encoding="utf-8",
        )
    path = stages_dir / f"{stage}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    return path


def write_artifacts(
    run_id: str,
    runs_dir: Path,
    artifact: dict,
    report_markdown: str,
    risks: list[ItemRisk],
    config: AuditRunConfig,
    events: list[dict],
    judge_call_traces: list[dict],
) -> dict[str, Path]:
    run_dir = runs_dir / run_id
    traces_dir = run_dir / "traces"
    logs_dir = run_dir / "logs"
    run_dir.mkdir(parents=True, exist_ok=True)
    traces_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    json_path = run_dir / "run.json"
    csv_path = run_dir / "item_risks.csv"
    report_path = run_dir / "report.md"
    resolved_config_path = run_dir / "config.resolved.yaml"
    perturbations_path = run_dir / "perturbations.jsonl"
    judge_results_path = run_dir / "judge_results.jsonl"
    validation_path = traces_dir / "validation_calls.jsonl"
    generation_path = traces_dir / "generation_calls.jsonl"
    judge_calls_path = traces_dir / "judge_calls.jsonl"
    llm_calls_path = traces_dir / "llm_calls.jsonl"
    events_path = logs_dir / "events.jsonl"
    suite_path = run_dir / "suite.json"

    write_json(json_path, artifact)
    write_risk_csv(csv_path, risks)
    report_path.write_text(report_markdown, encoding="utf-8")
    resolved_config_path.write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    write_jsonl(perturbations_path, artifact["perturbations"])
    write_jsonl(judge_results_path, artifact["judge_results"])
    write_jsonl(validation_path, artifact["validation_report"])
    write_jsonl(generation_path, artifact["generation_trace"])
    write_jsonl(judge_calls_path, judge_call_traces)
    write_jsonl(llm_calls_path, artifact.get("llm_calls", []))
    write_jsonl(events_path, events)

    return {
        "run_dir": run_dir,
        "json": json_path,
        "item_risks_csv": csv_path,
        "report": report_path,
        "resolved_config": resolved_config_path,
        "perturbations_jsonl": perturbations_path,
        "judge_results_jsonl": judge_results_path,
        "validation_trace": validation_path,
        "generation_trace": generation_path,
        "judge_call_trace": judge_calls_path,
        "llm_call_trace": llm_calls_path,
        "events": events_path,
        **({"suite": suite_path} if suite_path.exists() else {}),
    }
