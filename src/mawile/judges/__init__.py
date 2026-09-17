from mawile.judges.base import JudgeRunner
from mawile.judges.mock_runner import MockJudgeRunner
from typing import TYPE_CHECKING

from mawile.judges.openai_runner import OpenAIJudgeRunner

if TYPE_CHECKING:
    from mawile.schemas import AuditRunConfig

__all__ = ["JudgeRunner", "MockJudgeRunner", "OpenAIJudgeRunner", "get_judge_runner"]


def get_judge_runner(
    model: str,
    *,
    config: "AuditRunConfig | None" = None,
) -> JudgeRunner:
    if model == "mock":
        return MockJudgeRunner()
    return OpenAIJudgeRunner(config=config)
