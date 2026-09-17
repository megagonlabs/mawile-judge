from __future__ import annotations

import json
from typing import Any

from mawile.judges.base import JudgeRunner
from mawile.measurement import threshold_label
from mawile.schemas import Item, JudgeConfig, JudgeResult, OutputType, Perturbation


class MockJudgeRunner(JudgeRunner):
    """A deterministic local judge for pipeline development. Testing only.

    It is intentionally simple: it penalizes obvious unsupported claims,
    missing-format cases, and known degradation markers. Real judge runners
    should preserve the same return schema.
    """

    def run(
        self,
        item: Item,
        judge_config: JudgeConfig,
        perturbation: Perturbation | None,
        run_id: str,
    ) -> JudgeResult:
        verdict = self._verdict(item, judge_config)
        return JudgeResult(
            item_id=item.item_id,
            variant_id=perturbation.variant_id if perturbation else "original",
            run_id=run_id,
            raw_judge_output=json.dumps(verdict, sort_keys=True),
            parsed_verdict=verdict,
            parse_status="ok",
            metadata={
                "model": judge_config.model,
                "family": perturbation.family.value if perturbation else "noise",
                "operator": perturbation.operator if perturbation else "repeat",
            },
        )

    def _verdict(self, item: Item, judge_config: JudgeConfig) -> dict[str, Any]:
        if judge_config.output_type == OutputType.PAIRWISE:
            scored: dict[str, tuple[float, list[str]]] = {}
            for position in ("A", "B"):
                candidate_item = item.model_copy(
                    update={
                        "output": item.candidate_at(position),
                        "gold_label": None,
                    }
                )
                scored[position] = self._score(candidate_item, judge_config)
            position = max(("A", "B"), key=lambda value: scored[value][0])
            candidate_id = item.candidate_id_at(position)
            return {
                "label": position,
                "position": position,
                "candidate_id": candidate_id,
                "reason": f"Candidate {position} received the higher mock score.",
            }

        score, reasons = self._score(item, judge_config)
        verdict: dict[str, Any] = {
            "score": score,
            "reason": "; ".join(reasons) if reasons else "No major issues found.",
        }
        label = threshold_label(score, judge_config)
        if judge_config.output_type == OutputType.BINARY:
            # Binary mock configs historically use a 3.0 cutoff when omitted.
            label = label or ("pass" if score >= 3.0 else "fail")
        if label is not None:
            verdict["label"] = label
        return verdict

    def _score(self, item: Item, judge_config: JudgeConfig) -> tuple[float, list[str]]:
        task_text = _stringify(item.input).lower()
        output_text = _stringify(item.output).lower()
        prompt_text = judge_config.prompt_template.lower()
        score = 5.0
        reasons: list[str] = []

        penalties = {
            "unsupported claim": 2.0,
            "not mention": 1.5,
            "does not mention": 1.5,
            "factual inconsistency": 2.0,
            "format violation": 1.5,
            "missing requirement": 2.0,
            "incomplete": 1.5,
            "incorrect": 1.5,
            "error": 1.0,
        }
        for marker, penalty in penalties.items():
            if marker in output_text:
                score -= penalty
                reasons.append(f"Detected {marker}.")

        if "exactly three bullet" in task_text:
            bullet_count = sum(
                1 for line in _stringify(item.output).splitlines() if line.strip().startswith("-")
            )
            if bullet_count != 3:
                score -= 2.0
                reasons.append("Expected exactly three bullet points.")

        if "strict" in prompt_text and len(output_text) < 120:
            score -= 0.25
            reasons.append("Strict prompt applied a small completeness penalty.")

        return max(1.0, min(5.0, score)), reasons


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True)
