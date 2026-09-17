"""Judge-side deterministic builders.

Each emits a single variant that rewrites one judge field (``prompt_template`` or
``rubric``) with a pure text transform. Judge-side edits apply once per run, not
per item.
"""

from __future__ import annotations

import re

from mawile.perturbations.base import DimensionSpec, build_variant
from mawile.schemas import AuditRunConfig, Item, Perturbation


_BLOCK_START = re.compile(
    r"^\s*(?:#{1,6}\s+\S.*|(?:criterion|criteria|score|level|rating)\s*[:#\-]?\s*\S.*|\d+\s*:\s*\S.*)$",
    re.IGNORECASE,
)


def _independent_blocks(text: str) -> tuple[str, list[str], bool] | None:
    """Extract explicitly labelled rubric blocks.

    A rubric is not necessarily line-oriented: blindly reversing lines can move
    continuation text, examples, or a threshold into a different criterion.  We
    therefore require each block to begin with a recognizable heading/score row
    and at least one blank-line or heading boundary between independent blocks.
    Free-form prose returns ``None`` and is intentionally skipped.
    """

    lines = text.splitlines()
    starts = [index for index, line in enumerate(lines) if _BLOCK_START.match(line)]
    if len(starts) < 2:
        return None
    prefix_lines = lines[: starts[0]]
    # Preserve a conventional rubric preamble, but reject arbitrary prose
    # before the first block rather than silently dropping it.
    nonblank_prefix = [line.strip() for line in prefix_lines if line.strip()]
    if nonblank_prefix and any(not re.match(r"^(?:rubric|criteria)\s*:?$", line, re.I) for line in nonblank_prefix):
        return None
    # Every detected block must be explicitly separated from its predecessor;
    # otherwise a pair of ordinary prose lines was mistaken for score rows.
    blocks: list[str] = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(lines)
        block = "\n".join(lines[start:end]).strip()
        if block:
            blocks.append(block)
    if len(blocks) < 2:
        return None
    # A reference to position/order makes a block permutation semantic rather
    # than a provable invariance (and is therefore rejected by this offline
    # builder).  Continuation text remains attached to its own block.
    if re.search(r"\b(?:first|second|third|previous|next|above|below|in order)\b", text, re.I):
        return None
    numeric_rows = all(
        len([line for line in block.splitlines() if line.strip()]) == 1
        and re.match(r"^\s*\d+\s*:\s*\S", block)
        for block in blocks
    )
    if numeric_rows:
        labels = [re.match(r"^\s*(\d+)\s*:", block).group(1) for block in blocks]
        numeric_rows = len(labels) == len(set(labels))
    prefix = "\n".join(prefix_lines).rstrip()
    return prefix, blocks, numeric_rows


def build_judge_rubric_criterion_reorder(
    spec: DimensionSpec, config: AuditRunConfig, items: list[Item]
) -> list[Perturbation]:
    current = config.judge.rubric
    if not isinstance(current, str) or not current.strip():
        return []
    parsed = _independent_blocks(current)
    if parsed is None:
        return []
    prefix, blocks, numeric_rows = parsed
    reordered = "\n\n".join(reversed(blocks))
    new_value = f"{prefix}\n\n{reordered}" if prefix else reordered
    if not new_value or new_value == current:
        return []
    metadata: dict[str, object] = {}
    if numeric_rows:
        metadata["structural_invariant"] = "rubric_block_permutation"
    else:
        metadata.update(
            {
                "requires_validation": True,
                "validation_kind": "judge_rubric_preserve",
                "validation_target": "judge_config",
                "validation_field": "rubric",
            }
        )
    return [
        build_variant(
            spec,
            variant_id=spec.operator,
            summary="Reorder explicitly delimited independent rubric blocks.",
            judge_config_overrides={"rubric": new_value},
            extra_metadata=metadata,
        )
    ]
