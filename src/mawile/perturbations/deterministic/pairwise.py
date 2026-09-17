from __future__ import annotations

from mawile.perturbations.base import DimensionSpec, build_variant
from mawile.schemas import AuditRunConfig, Item, Perturbation


def build_position_swap(
    spec: DimensionSpec,
    _config: AuditRunConfig,
    items: list[Item],
) -> list[Perturbation]:
    """Swap display positions while keeping stable candidate identities fixed."""

    variants: list[Perturbation] = []
    for item in items:
        if not item.is_pairwise:
            continue
        swapped = {
            "A": item.position_to_candidate["B"],
            "B": item.position_to_candidate["A"],
        }
        swapped_output = {
            "candidates": dict(item.candidates),
            "position_to_candidate": swapped,
        }
        variants.append(
            build_variant(
                spec,
                variant_id=f"{item.item_id}__{spec.operator}",
                item_id=item.item_id,
                summary="Swapped candidate display positions A and B.",
                changed_fields=["item.output.position_to_candidate"],
                item_overrides={"output": swapped_output},
                extra_metadata={
                    "original_position_to_candidate": dict(item.position_to_candidate),
                    "swapped_position_to_candidate": swapped,
                },
            )
        )
    return variants
