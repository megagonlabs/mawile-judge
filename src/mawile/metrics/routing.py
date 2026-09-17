from __future__ import annotations

from collections import defaultdict
from typing import Any

from mawile.metrics.directional import case_contradicted, case_detected
from mawile.schemas import ItemRisk


def review_priority(risk: ItemRisk) -> float:
    """Combined routing signal: an item is review-worthy if either its
    invariant stability or its directional detection looks bad."""
    return max(
        risk.flip_risk or 0.0,
        risk.directional_risk or 0.0,
    )


def rank_item_risks(risks: list[ItemRisk]) -> list[ItemRisk]:
    ranked = sorted(
        risks,
        key=lambda risk: (
            -review_priority(risk),
            -(risk.flip_risk or 0.0),
            risk.item_id,
        ),
    )
    return [
        risk.model_copy(update={"routing_rank": rank})
        for rank, risk in enumerate(ranked, start=1)
    ]


def apply_directional_risk(
    risks: list[ItemRisk],
    directional_summary: dict[str, Any],
) -> list[ItemRisk]:
    """Fold per-item directional outcomes into item risks and re-rank."""
    cases_by_item: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in directional_summary.get("cases", []):
        cases_by_item[str(case.get("item_id"))].append(case)
    if not cases_by_item:
        return risks

    updated = []
    for risk in risks:
        cases = cases_by_item.get(risk.item_id, [])
        analyzable = [case for case in cases if case_detected(case) is not None]
        missed = sum(1 for case in analyzable if not case_detected(case))
        contradictions = sum(1 for case in analyzable if case_contradicted(case))
        updated.append(
            risk.model_copy(
                update={
                    "directional_cases": len(cases),
                    "directional_missed": missed,
                    "directional_contradictions": contradictions,
                    "directional_risk": (
                        missed / len(analyzable) if analyzable else None
                    ),
                }
            )
        )
    return rank_item_risks(updated)
