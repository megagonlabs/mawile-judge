from __future__ import annotations

import re

from mawile.perturbations.base import DimensionSpec, build_variant
from mawile.schemas import AuditRunConfig, Item, Perturbation


def build_output_partial_completion(
    spec: DimensionSpec, config: AuditRunConfig, items: list[Item]
) -> list[Perturbation]:
    """Remove a final independent chunk so the output is plausibly incomplete."""

    variants: list[Perturbation] = []
    for item in items:
        if not isinstance(item.output, str) or not item.output.strip():
            continue
        degraded = _remove_final_chunk(item.output)
        if degraded is None or degraded == item.output:
            continue
        variants.append(
            build_variant(
                spec,
                variant_id=f"{item.item_id}__{spec.operator}",
                item_id=item.item_id,
                summary="Remove the final independent part of the output.",
                item_overrides={"output": degraded},
                extra_metadata={
                    "degradation_kind": "partial_completion",
                    # Removing text is only a candidate construction.  Whether
                    # the removed chunk was an independent requirement must be
                    # established semantically against the task and rubric.
                    "requires_validation": True,
                    "validation_kind": "directional_degradation",
                    "validation_target": "transcript",
                    "validation_field": "output",
                    "directional": True,
                },
            )
        )
    return variants


def _remove_final_chunk(text: str) -> str | None:
    bullet = _remove_final_bullet(text)
    if bullet is not None:
        return bullet

    paragraph = _remove_final_paragraph(text)
    if paragraph is not None:
        return paragraph

    sentence = _remove_final_sentence(text)
    if sentence is not None:
        return sentence
    return None


def _remove_final_bullet(text: str) -> str | None:
    lines = text.splitlines()
    bullet_indices = [index for index, line in enumerate(lines) if _is_list_item(line)]
    if len(bullet_indices) < 2:
        return None
    remove_at = bullet_indices[-1]
    kept = [line for index, line in enumerate(lines) if index != remove_at]
    degraded = "\n".join(kept).strip()
    return degraded or None


def _remove_final_paragraph(text: str) -> str | None:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if len(paragraphs) < 2:
        return None
    return "\n\n".join(paragraphs[:-1]).strip() or None


def _remove_final_sentence(text: str) -> str | None:
    matches = list(re.finditer(r"[^.!?]+[.!?](?:\s+|$)", text.strip()))
    if len(matches) < 2:
        return None
    degraded = text[: matches[-1].start()].strip()
    return degraded or None


def _is_list_item(line: str) -> bool:
    return _is_bullet_item(line) or _is_numbered_item(line)


def _is_bullet_item(line: str) -> bool:
    return bool(re.match(r"^\s*[-*]\s+\S", line))


def _is_numbered_item(line: str) -> bool:
    return bool(re.match(r"^\s*\d+[.)]\s+\S", line))
