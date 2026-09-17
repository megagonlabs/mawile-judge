from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, TextIO

from tqdm import tqdm

from mawile.config import load_config
from mawile.env import load_environment
from mawile.config_payload import (
    dump_config_yaml,
    load_config_and_items,
    load_config_payload,
)
from mawile.operator_catalog import (
    build_perturbation_previews,
    catalog_operator_rows,
    estimate_run_size,
    validation_catalog_rows,
)
from mawile.pipeline import AuditPipeline
from mawile.project import discover_config_files, discover_run_artifacts
from mawile.providers import provider_has_api_key
from mawile.run_artifacts import load_run_bundle
from mawile.reporting.regenerate import regenerate_report
from mawile.suites import load_suite
from mawile.perturbations.registry import DIMENSIONS, DIRECTIONAL_DIMENSIONS
from mawile.suggestions import (
    plan_perturbation_operators,
    suggest_custom_perturbation,
)


def validate_command(args: argparse.Namespace) -> int:
    try:
        config, items = load_config_and_items(args.config)
    except (ValueError, OSError) as exc:
        print(f"Config is invalid: {exc}", file=sys.stderr)
        return 1
    print(f"Config is valid: {config.data.items_path} ({len(items)} items)")
    return 0


def run_command(args: argparse.Namespace) -> int:
    return _execute_run(args)


def replay_command(args: argparse.Namespace) -> int:
    return _execute_run(args, suite_path=args.suite)


def _execute_run(args: argparse.Namespace, *, suite_path: Path | None = None) -> int:
    progress = _TqdmProgress() if args.progress else None
    try:
        if suite_path is not None and args.config is None:
            load_environment()
            config = load_suite(suite_path).source_config.model_copy(deep=True)
        else:
            config = load_config(args.config)
        if suite_path is not None and args.runs_dir is not None:
            config.output.runs_dir = args.runs_dir.expanduser().resolve()
        result = AuditPipeline(
            config, progress_callback=progress, suite_path=suite_path,
        ).run()
    except (ValueError, OSError) as exc:
        print(f"Audit run failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if progress is not None:
            progress.close()
    print(f"Audit run complete: {result.run_id}")
    for label, path in result.paths.items():
        print(f"{label}: {path}")
    return 0


def configs_command(args: argparse.Namespace) -> int:
    paths = discover_config_files()
    if args.format == "json":
        _print_json([str(path) for path in paths])
        return 0

    if not paths:
        print("No checked-in configs found.")
        return 0
    print("Audit configs:")
    for path in paths:
        print(f"- {path}")
    return 0


def catalog_command(args: argparse.Namespace) -> int:
    invariant = catalog_operator_rows(DIMENSIONS)
    directional = catalog_operator_rows(DIRECTIONAL_DIMENSIONS)
    validations = validation_catalog_rows() if args.validation else []

    if args.format == "json":
        payload: dict[str, Any] = {
            "invariant": [_to_jsonable(row) for row in invariant],
            "directional": [_to_jsonable(row) for row in directional],
        }
        if args.validation:
            payload["validation"] = [_to_jsonable(row) for row in validations]
        _print_json(payload)
        return 0

    _print_operator_section("Invariant operators", invariant)
    print()
    _print_operator_section("Directional operators", directional)
    if args.validation:
        print()
        print("Validation checks:")
        for row in validations:
            operators = ", ".join(row.operators) if row.operators else "none"
            print(f"- {row.kind}: {operators}")
    return 0


def inspect_command(args: argparse.Namespace) -> int:
    try:
        config, items = load_config_and_items(args.config)
    except (ValueError, OSError) as exc:
        print(f"Config is invalid: {exc}", file=sys.stderr)
        return 1

    has_api_key = provider_has_api_key(
        config,
        config.perturbation_agent.provider,
    )
    previews = build_perturbation_previews(
        config,
        items,
        has_api_key=has_api_key,
        include_not_runnable=args.include_not_runnable,
    )
    rows = previews if args.previews else []
    size = estimate_run_size(config, items)

    enabled_rows = [_to_jsonable(row) for row in previews]

    if args.format == "json":
        _print_json(
            {
                "config": str(Path(args.config).expanduser().resolve()),
                "items": len(items),
                "judge_model": config.judge.model,
                "judge_provider": config.judge.provider,
                "perturbation_agent_model": config.perturbation_agent.model,
                "perturbation_agent_provider": config.perturbation_agent.provider,
                "estimate": size,
                "operators": enabled_rows,
                "previews": [_to_jsonable(row) for row in rows],
            }
        )
        return 0

    print(f"Config: {Path(args.config).expanduser().resolve()}")
    print(f"Judge model: {config.judge.provider}/{config.judge.model}")
    print(
        "Perturbation agent: "
        f"{config.perturbation_agent.provider}/{config.perturbation_agent.model}"
    )
    print(f"Items: {len(items)}")
    print(
        "Estimated judge calls: "
        f"{size['known_judge_calls']} known, up to {size['max_judge_calls']} with LLM operators"
    )
    print(
        "Estimated upper-bound generation units: "
        f"{size['generation_units_upper_bound']} total "
        f"({size['llm_generation_units']} agent, "
        f"{size['deterministic_perturbations']} deterministic variants)"
    )
    print()
    print("Enabled operators:")
    if not enabled_rows:
        print("- none")
    for row in enabled_rows:
        print(
            f"- {row['operator']} [{row['kind']}] "
            f"{row['target']} -> {row['expected_effect']} ({row['status']})"
        )
    if args.previews:
        print()
        print("Previews:")
        if not rows:
            print("- none")
        for preview in rows:
            print(f"- {preview.operator}: {preview.summary}")
            if preview.variant_id:
                print(f"  variant: {preview.variant_id} item: {preview.item_id}")
            print(f"  field: {preview.changed_field}")
            print(f"  before: {_one_line(preview.before)}")
            if preview.after is not None:
                print(f"  after: {_one_line(preview.after)}")
    return 0


def plan_command(args: argparse.Namespace) -> int:
    try:
        loaded = load_config_payload(args.config)
        planned_payload, selection = plan_perturbation_operators(
            loaded.payload,
            loaded.base_dir,
        )
    except (ValueError, OSError) as exc:
        print(f"Perturbation Suggester failed: {exc}", file=sys.stderr)
        return 1

    planned_yaml = dump_config_yaml(planned_payload)
    output_path: Path | None = None
    trace_path: Path | None = None
    if args.output:
        output_path = args.output.expanduser().resolve()
        output_path.write_text(planned_yaml, encoding="utf-8")

    payload = {
        "selected": selection.selected,
        "manual": selection.manual,
        "effective": selection.effective,
        "selected_directional": selection.selected_directional,
        "manual_directional": selection.manual_directional,
        "effective_directional": selection.effective_directional,
        "rationale": selection.rationale,
        "messages": list(getattr(selection, "messages", [])),
        "context_metadata": dict(getattr(selection, "context_metadata", {})),
        "output": str(output_path) if output_path else None,
    }
    if output_path is not None:
        trace_path = output_path.with_suffix(".planning.json")
        trace_payload = {**payload, "planning_trace_output": str(trace_path)}
        trace_path.write_text(
            json.dumps(trace_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        payload["planning_trace_output"] = str(trace_path)

    if args.format == "json":
        _print_json(payload)
        return 0
    if args.format == "yaml":
        print(planned_yaml, end="")
        _print_prompt_trace(payload, stream=sys.stderr)
        return 0

    print("Perturbation suggestions:")
    print(f"- invariant selected: {_comma(selection.selected)}")
    print(f"- invariant effective: {_comma(selection.effective)}")
    print(f"- directional selected: {_comma(selection.selected_directional)}")
    print(f"- directional effective: {_comma(selection.effective_directional)}")
    if selection.rationale:
        print(f"- rationale: {selection.rationale}")
    if output_path:
        print(f"Planned config written: {output_path}")
        print(f"Planning trace written: {trace_path}")
    _print_prompt_trace(payload)
    return 0


def suggest_custom_command(args: argparse.Namespace) -> int:
    try:
        loaded = load_config_payload(args.config)
        suggestion = suggest_custom_perturbation(
            loaded.payload,
            loaded.base_dir,
            target=args.target,
            expected_effect=args.expected_effect,
        )
    except (ValueError, OSError) as exc:
        print(f"Custom perturbation suggestion failed: {exc}", file=sys.stderr)
        return 1

    payload = _to_jsonable(suggestion)
    config_payload = {
        "name": suggestion.name,
        "target": suggestion.target,
        "instruction": suggestion.instruction,
        "expected_effect": suggestion.expected_effect,
        "enabled": True,
    }
    if args.format == "json":
        _print_json(payload)
        return 0
    if args.format == "yaml":
        print(dump_config_yaml({"custom_perturbations": [config_payload]}), end="")
        _print_prompt_trace(payload, stream=sys.stderr)
        return 0

    print("Custom perturbation suggestion:")
    print(f"- name: {suggestion.name}")
    print(f"- target: {suggestion.target}")
    print(f"- expected_effect: {suggestion.expected_effect}")
    print(f"- instruction: {suggestion.instruction}")
    if suggestion.rationale:
        print(f"- rationale: {suggestion.rationale}")
    _print_prompt_trace(payload)
    return 0


def _print_prompt_trace(payload: dict[str, Any], *, stream: Any | None = None) -> None:
    if stream is None:
        stream = sys.stdout
    trace = {
        "messages": payload.get("messages", []),
        "context_metadata": payload.get("context_metadata", {}),
    }
    print("Planner prompt trace:", file=stream)
    print(json.dumps(trace, ensure_ascii=False, indent=2, sort_keys=True), file=stream)


def runs_command(args: argparse.Namespace) -> int:
    run_json_paths = (
        discover_run_artifacts(args.runs_dir)
        if args.runs_dir
        else discover_run_artifacts()
    )
    if args.limit >= 0:
        run_json_paths = run_json_paths[: args.limit]

    rows: list[dict[str, Any]] = []
    for path in run_json_paths:
        try:
            bundle = load_run_bundle(path.parent)
        except (ValueError, OSError, json.JSONDecodeError):
            continue
        rows.append(
            {
                "run_id": bundle.run_id,
                "run_dir": str(bundle.run_dir),
                "item_count": bundle.artifact.get("item_count"),
                "perturbation_count": bundle.artifact.get("perturbation_count"),
                "report": str(bundle.paths["report"]),
            }
        )

    if args.format == "json":
        _print_json(rows)
        return 0
    if not rows:
        print("No completed runs found.")
        return 0
    print("Completed runs:")
    for row in rows:
        print(
            f"- {row['run_id']} "
            f"items={row['item_count']} perturbations={row['perturbation_count']} "
            f"report={row['report']}"
        )
    return 0


def report_command(args: argparse.Namespace) -> int:
    if args.output is not None and not args.regenerate:
        print("--output requires --regenerate.", file=sys.stderr)
        return 1
    if args.regenerate and (args.path or args.format != "text"):
        print("--regenerate cannot be combined with --path or --format json.", file=sys.stderr)
        return 1
    run_dir = _run_dir_for_id(args.run_id, args.runs_dir)
    if run_dir is None:
        print("Run not found.", file=sys.stderr)
        return 1

    if args.regenerate:
        try:
            report = regenerate_report(run_dir, args.output)
        except (ValueError, OSError) as exc:
            print(f"Could not regenerate report: {exc}", file=sys.stderr)
            return 1
        if args.output is None:
            print(report, end="" if report.endswith("\n") else "\n")
        else:
            print(f"Report written: {args.output.resolve()}")
        return 0

    try:
        bundle = load_run_bundle(run_dir)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"Could not load run: {exc}", file=sys.stderr)
        return 1

    if args.format == "json":
        _print_json(
            {
                "run_id": bundle.run_id,
                "run_dir": str(bundle.run_dir),
                "report": str(bundle.paths["report"]),
                "item_count": bundle.artifact.get("item_count"),
                "perturbation_count": bundle.artifact.get("perturbation_count"),
            }
        )
        return 0
    if args.path:
        print(bundle.paths["report"])
        return 0
    print(bundle.report_text, end="" if bundle.report_text.endswith("\n") else "\n")
    return 0


def _run_dir_for_id(run_id: str | None, runs_dir: Path | None = None) -> Path | None:
    run_json_paths = (
        discover_run_artifacts(runs_dir) if runs_dir else discover_run_artifacts()
    )
    if not run_json_paths:
        return None
    if run_id is None:
        return run_json_paths[0].parent
    for path in run_json_paths:
        if path.parent.name == run_id:
            return path.parent
    candidate = (runs_dir or Path("runs")) / run_id
    if (candidate / "run.json").exists():
        return candidate.resolve()
    return None


class _TqdmProgress:
    """Render pipeline events with at most one active terminal progress bar."""

    _BAR_STAGES = {
        "generation_unit_completed": ("generation", "rewrite"),
        "validation_call_completed": ("validation", "check"),
        "baseline_judge_calls_started": ("baseline_judge_calls", "call"),
        "baseline_judge_call_completed": ("baseline_judge_calls", "call"),
        "baseline_judge_calls_complete": ("baseline_judge_calls", "call"),
        "judge_calls_started": ("judge_calls", "call"),
        "judge_call_completed": ("judge_calls", "call"),
        "judge_calls_complete": ("judge_calls", "call"),
    }
    _COMPLETION_STAGES = {
        "baseline_judge_calls_complete",
        "judge_calls_complete",
    }

    def __init__(
        self,
        *,
        stream: TextIO | None = None,
        tqdm_factory: Any | None = None,
    ) -> None:
        self.stream = stream or sys.stderr
        self._tqdm_factory = tqdm_factory or tqdm
        self._bar: Any | None = None
        self._bar_key: str | None = None

    def __call__(self, event: dict[str, Any]) -> None:
        stage = str(event.get("stage") or "progress")
        message = str(event.get("message") or stage)
        current = event.get("current")
        total = event.get("total")
        bar_spec = self._BAR_STAGES.get(stage)

        if bar_spec is not None and current is not None and total is not None:
            key, unit = bar_spec
            if self._bar is None or self._bar_key != key:
                self.close()
                self._bar = self._tqdm_factory(
                    total=total,
                    desc=message,
                    unit=unit,
                    dynamic_ncols=True,
                    disable=None,
                    file=self.stream,
                )
                self._bar_key = key
            delta = max(0, int(current) - int(self._bar.n))
            if delta:
                self._bar.update(delta)
            if stage in self._COMPLETION_STAGES:
                self.close()
            return

        self.close()
        print(message, file=self.stream)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None
            self._bar_key = None


def _print_operator_section(title: str, rows: list[Any]) -> None:
    print(f"{title} ({len(rows)}):")
    for row in rows:
        description = row.description or row.instruction or ""
        validation = f", validation={row.validation_kind}" if row.validation_kind else ""
        print(
            f"- {row.operator} [{row.kind}] "
            f"target={row.target}, effect={row.expected_effect}{validation}"
        )
        if description:
            print(f"  {_one_line(description, limit=140)}")


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    return value


def _print_json(payload: Any) -> None:
    print(json.dumps(_to_jsonable(payload), indent=2, sort_keys=True))


def _comma(values: list[str]) -> str:
    return ", ".join(values) if values else "none"


def _one_line(value: Any, *, limit: int = 160) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."
