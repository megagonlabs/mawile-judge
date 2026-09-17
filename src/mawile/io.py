from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from mawile.schemas import DataConfig, Item, ItemRisk


def load_items(data_config: DataConfig) -> list[Item]:
    items: list[Item] = []
    seen_ids: set[str] = set()
    with data_config.items_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            item_id = str(row.get("item_id", index)).strip()
            if not item_id or item_id == "*":
                raise ValueError(
                    f"Invalid reserved item_id on line {index}: {item_id!r}; "
                    "item IDs must be non-empty and cannot be '*'."
                )
            if item_id in seen_ids:
                raise ValueError(f"Duplicate item_id {item_id!r} on line {index}")
            seen_ids.add(item_id)
            gold_label = (
                row.get(data_config.gold_field)
                if data_config.gold_field is not None
                else None
            )
            items.append(
                Item(
                    item_id=item_id,
                    input=row.get(data_config.input_field),
                    output=row.get(data_config.output_field),
                    metadata=row.get("metadata", {}),
                    gold_label=gold_label,
                    human_label_distribution=row.get("human_label_distribution"),
                )
            )
    return items


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def write_risk_csv(path: Path, risks: Iterable[ItemRisk]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "item_id": risk.item_id,
            "original_verdict": risk.original_verdict,
            "original_score": risk.original_score,
            "gold_label": risk.gold_label,
            "judge_correct": risk.judge_correct,
            "judge_error": risk.judge_error,
            "noise_risk": risk.noise_risk,
            "flip_risk": risk.flip_risk,
            "all_attempt_flip_risk": risk.all_attempt_flip_risk,
            "excess_over_noise": risk.excess_over_noise,
            "invariant_attempts": risk.invariant_attempts,
            "invariant_analyzable": risk.invariant_analyzable,
            "boundary_proximity": risk.boundary_proximity,
            "routing_rank": risk.routing_rank,
            "gold_error": risk.gold_error,
            "directional_cases": risk.directional_cases,
            "directional_missed": risk.directional_missed,
            "directional_contradictions": risk.directional_contradictions,
            "directional_risk": risk.directional_risk,
        }
        for risk in risks
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
