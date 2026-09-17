"""Compatibility re-exports for legacy ``mawile.metrics.common`` imports."""

from mawile.measurement import (
    majority_label,
    mean_score,
    normalized_score,
    rescale_to_base,
    result_label,
    result_score,
)

__all__ = [
    "majority_label",
    "mean_score",
    "normalized_score",
    "rescale_to_base",
    "result_label",
    "result_score",
]
