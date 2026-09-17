"""Offline rendering of an already completed MAWILE run."""

from __future__ import annotations

from pathlib import Path

import yaml

from mawile.reporting.markdown import render_markdown_report
from mawile.run_artifacts import load_run_bundle


def regenerate_report(run_dir: Path, output: Path | None = None) -> str:
    """Render a saved run without executing metrics, providers, or model calls.

    The source run directory is read-only from this function's perspective.
    When ``output`` is supplied, it must name a new file; existing files are
    rejected rather than overwritten.
    """

    run_dir = Path(run_dir)
    config_path = run_dir / "config.resolved.yaml"
    try:
        bundle = load_run_bundle(run_dir)
    except (OSError, TypeError, ValueError, yaml.YAMLError, AttributeError) as exc:
        raise ValueError(f"Cannot load saved run for offline regeneration: {exc}") from exc
    if bundle.config is None:
        config_path = bundle.paths.get("resolved_config", config_path)
        raise ValueError(
            "Cannot regenerate report because the saved config is missing or unusable: "
            f"{config_path}"
        )

    # render_markdown_report reads stored artifact metrics, interpretation, and
    # persisted judge traces. It does not invoke the report interpreter. Legacy
    # artifacts without a saved interpretation get an explicit unavailable
    # placeholder rather than a newly recomputed narrative.
    artifact = bundle.artifact
    if not isinstance(artifact.get("report_interpretation"), dict) or not artifact.get(
        "report_interpretation"
    ):
        artifact = dict(artifact)
        artifact["report_interpretation"] = {
            "source": "offline_regeneration",
            "executive_summary": "No saved report interpretation is available for this run.",
            "key_findings": [],
            "recommended_actions": [],
            "limitations": ["The saved run did not include report interpretation provenance."],
            "follow_up_experiments": [],
        }
    try:
        report = render_markdown_report(bundle.config, artifact, bundle.traces)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Cannot regenerate report because the saved artifact is missing or malformed: "
            f"{exc}"
        ) from exc
    if output is not None:
        output = Path(output)
        source_dir = run_dir.resolve()
        output_path = output.resolve(strict=False)
        try:
            output_path.relative_to(source_dir)
        except ValueError:
            pass
        else:
            raise ValueError(
                "Regenerated report output must be outside the source run directory"
            )
        with output_path.open("x", encoding="utf-8") as handle:
            handle.write(report)
    return report


__all__ = ["regenerate_report"]
