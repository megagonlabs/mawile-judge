from __future__ import annotations

import argparse
from pathlib import Path

from mawile.cli_commands import (
    catalog_command,
    configs_command,
    inspect_command,
    plan_command,
    replay_command,
    report_command,
    run_command,
    runs_command,
    suggest_custom_command,
    validate_command,
)
from mawile.env import load_environment
from mawile.schemas import PerturbationExpectedEffect


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mawile")
    subparsers = parser.add_subparsers(dest="command", required=True)

    configs_parser = subparsers.add_parser("configs", help="List checked-in audit configs")
    configs_parser.set_defaults(handler=configs_command)
    configs_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format",
    )

    catalog_parser = subparsers.add_parser("catalog", help="Show perturbation catalog")
    catalog_parser.set_defaults(handler=catalog_command)
    catalog_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format",
    )
    catalog_parser.add_argument(
        "--validation",
        action="store_true",
        help="Include validation prompt catalog",
    )

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Validate a config and summarize enabled operators and run size",
    )
    inspect_parser.set_defaults(handler=inspect_command)
    inspect_parser.add_argument("config", type=Path, help="Path to an audit YAML config")
    inspect_parser.add_argument(
        "--previews",
        action="store_true",
        help="Show one preview example per enabled operator",
    )
    inspect_parser.add_argument(
        "--include-not-runnable",
        action="store_true",
        help="Include LLM-backed operators whose prerequisites are unavailable",
    )
    inspect_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format",
    )

    plan_parser = subparsers.add_parser(
        "plan",
        help="Run the Perturbation Suggester and show the resolved operator allow-list",
    )
    plan_parser.set_defaults(handler=plan_command)
    plan_parser.add_argument("config", type=Path, help="Path to an audit YAML config")
    plan_parser.add_argument(
        "--output",
        type=Path,
        help="Write the planned config YAML to this path",
    )
    plan_parser.add_argument(
        "--format",
        choices=["text", "json", "yaml"],
        default="text",
        help="Output format",
    )

    custom_parser = subparsers.add_parser(
        "suggest-custom",
        help="Suggest one custom perturbation for a config",
    )
    custom_parser.set_defaults(handler=suggest_custom_command)
    custom_parser.add_argument("config", type=Path, help="Path to an audit YAML config")
    custom_parser.add_argument(
        "--target",
        required=True,
        choices=["judge.prompt_template", "judge.rubric", "item.input", "item.output"],
        help="Field the custom perturbation should rewrite",
    )
    custom_parser.add_argument(
        "--expected-effect",
        choices=[
            PerturbationExpectedEffect.SAME_VERDICT.value,
            PerturbationExpectedEffect.WORSE_VERDICT.value,
        ],
        default=PerturbationExpectedEffect.SAME_VERDICT.value,
        help="Expected verdict relationship",
    )
    custom_parser.add_argument(
        "--format",
        choices=["text", "json", "yaml"],
        default="text",
        help="Output format",
    )

    run_parser = subparsers.add_parser("run", help="Run a MAWILE audit")
    run_parser.set_defaults(handler=run_command)
    run_parser.add_argument("config", type=Path, help="Path to an audit YAML config")
    run_parser.add_argument(
        "--progress",
        action="store_true",
        help="Show stage progress with updating terminal bars",
    )

    replay_parser = subparsers.add_parser(
        "replay",
        help="Run new judge measurements on an exact saved perturbation suite",
    )
    replay_parser.set_defaults(handler=replay_command)
    replay_parser.add_argument("suite", type=Path, help="Path to a saved suite.json")
    replay_parser.add_argument(
        "--config", type=Path,
        help="Optional config overriding judge execution settings; suite semantics must match",
    )
    replay_parser.add_argument(
        "--runs-dir", type=Path, help="Directory for the new run (source suite is never modified)",
    )
    replay_parser.add_argument(
        "--progress", action="store_true", help="Show stage progress",
    )

    validate_parser = subparsers.add_parser("validate", help="Validate an audit config")
    validate_parser.set_defaults(handler=validate_command)
    validate_parser.add_argument("config", type=Path, help="Path to an audit YAML config")

    runs_parser = subparsers.add_parser("runs", help="List completed audit runs")
    runs_parser.set_defaults(handler=runs_command)
    runs_parser.add_argument(
        "--runs-dir",
        type=Path,
        help="Directory containing run subdirectories",
    )
    runs_parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum runs to show",
    )
    runs_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format",
    )

    report_parser = subparsers.add_parser("report", help="Show a completed run report")
    report_parser.set_defaults(handler=report_command)
    report_parser.add_argument(
        "run_id",
        nargs="?",
        help="Run id to show; defaults to the latest run",
    )
    report_parser.add_argument(
        "--runs-dir",
        type=Path,
        help="Directory containing run subdirectories",
    )
    report_parser.add_argument(
        "--path",
        action="store_true",
        help="Print only the report path",
    )
    report_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format",
    )
    report_parser.add_argument(
        "--regenerate", action="store_true",
        help="Render Markdown from saved evidence offline, without recalculating metrics",
    )
    report_parser.add_argument(
        "--output", type=Path,
        help="With --regenerate, create a new Markdown file; existing files are never overwritten",
    )

    ui_parser = subparsers.add_parser("ui", help="Start the local HTML UI")
    ui_parser.set_defaults(handler=ui_command)
    ui_parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind")
    ui_parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    ui_parser.add_argument(
        "--reload",
        action="store_true",
        help="Restart the UI server when source files change",
    )

    return parser


def ui_command(args: argparse.Namespace) -> int:
    import uvicorn

    load_environment()
    uvicorn.run(
        "mawile.web.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)




if __name__ == "__main__":
    raise SystemExit(main())
