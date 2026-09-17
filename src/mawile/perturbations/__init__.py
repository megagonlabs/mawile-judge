from __future__ import annotations

from collections.abc import Sequence

from mawile.applications import GenerationUnit, resolve_generation_units
from mawile.perturbations.registry import deterministic_dimensions
from mawile.schemas import (
    AuditRunConfig,
    Item,
    Perturbation,
)


def build_perturbations(
    config: AuditRunConfig,
    items: list[Item],
    *,
    directional_items: list[Item] | None = None,
    generation_units: Sequence[GenerationUnit] | None = None,
) -> list[Perturbation]:
    """Build every enabled deterministic perturbation variant.

    LLM-backed dimensions are produced separately by the perturbation agent (see
    :mod:`mawile.perturbations.generated`), since they require a model call.
    """

    if generation_units is None:
        directional_items = items if directional_items is None else directional_items
        generation_units = resolve_generation_units(
            list(deterministic_dimensions(config)),
            items,
            directional_eligible_item_ids={item.item_id for item in directional_items},
        )
    variants: list[Perturbation] = []
    for unit in generation_units:
        spec = unit.spec
        if spec.kind != "det":
            continue
        assert spec.builder is not None  # deterministic specs always carry a builder
        builder_items = (
            [item for item in items if item.item_id in unit.target_item_ids]
            if unit.item is None
            else [unit.item]
        )
        variants.extend(
            spec.builder(
                spec,
                config,
                builder_items,
            )
        )
    return variants
