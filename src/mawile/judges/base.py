from __future__ import annotations

from abc import ABC, abstractmethod

from mawile.schemas import Item, JudgeConfig, JudgeResult, Perturbation


class JudgeRunner(ABC):
    @abstractmethod
    def run(
        self,
        item: Item,
        judge_config: JudgeConfig,
        perturbation: Perturbation | None,
        run_id: str,
    ) -> JudgeResult:
        """Run the configured judge once and return a normalized result."""

