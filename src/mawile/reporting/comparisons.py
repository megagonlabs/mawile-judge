"""Shared report-side comparisons built on canonical measurement semantics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mawile.measurement import VerdictSummary, invariant_disagrees, normalized_score, result_label


def compare_result(
    result: Mapping[str, Any],
    baseline: VerdictSummary | None,
    *,
    perturbation: Mapping[str, Any] | None = None,
    output_semantics: Mapping[str, Any] | None = None,
    invariant: bool = False,
) -> dict[str, Any]:
    """Compare one result with its baseline, preserving unavailable values.

    Canonical accessors reject transport/parse failures even when a stale
    ``parsed_verdict`` was persisted.  For legacy artifacts without
    ``output_semantics``, invariant comparisons conservatively use class labels
    only; numeric missingness is not treated as stability.
    """

    baseline_label = baseline.label if baseline is not None else None
    baseline_score = baseline.score if baseline is not None else None
    perturbed_label = result_label(result)
    perturbed_score = normalized_score(result, perturbation)
    score_delta = (
        perturbed_score - baseline_score
        if perturbed_score is not None and baseline_score is not None
        else None
    )
    # Missing semantics and directional examples use class movement here;
    # directional detection/contradiction is reported separately.
    flipped = invariant_disagrees(
        result,
        baseline_label=baseline_label,
        baseline_score=baseline_score,
        judge=output_semantics if invariant else None,
        score=perturbed_score,
    )
    return {
        "baseline_label": baseline_label,
        "perturbed_label": perturbed_label,
        "baseline_score": baseline_score,
        "perturbed_score": perturbed_score,
        "score_delta": score_delta,
        "flipped": flipped,
    }


__all__ = ["compare_result"]
