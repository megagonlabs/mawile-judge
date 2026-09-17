from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from mawile.schemas import ItemRisk, JudgeConfig, OutputType


def compute_judge_correctness(
    item_risks: list[ItemRisk],
    *,
    top_k: int,
    judge: JudgeConfig | None = None,
) -> dict[str, Any]:
    """Summarize original judge correctness against trusted gold labels."""

    scalar = judge is not None and judge.output_type in {
        OutputType.SCALAR,
        OutputType.ORDINAL,
    }
    labeled = [
        risk
        for risk in item_risks
        if risk.gold_label is not None
        and (risk.original_score is not None if scalar else risk.original_verdict is not None)
    ]
    if not labeled:
        return {
            "has_gold_labels": False,
            "labeled_items": 0,
            "correct": 0,
            "errors": 0,
            "accuracy": None,
            "error_rate": None,
            "confusion_matrix": {},
            "score_items": 0,
            "mean_absolute_score_error": None,
            "review_utility": None,
            "metric_type": None,
            "rank_correlation": None,
            "tolerance": judge.gold_tolerance if scalar and judge is not None else None,
        }

    if scalar:
        return _scalar_correctness(labeled, top_k=top_k, judge=judge)

    errors = [risk for risk in labeled if _judge_error(risk) is True]
    correct = [risk for risk in labeled if _judge_error(risk) is False]
    score_errors = [
        abs(float(risk.original_score) - float(risk.gold_label))
        for risk in labeled
        if risk.original_score is not None and _is_number(risk.gold_label)
    ]

    return {
        "has_gold_labels": True,
        "labeled_items": len(labeled),
        "correct": len(correct),
        "errors": len(errors),
        "accuracy": len(correct) / len(labeled),
        "error_rate": len(errors) / len(labeled),
        "confusion_matrix": _confusion_matrix(labeled),
        "score_items": len(score_errors),
        "mean_absolute_score_error": (
            sum(score_errors) / len(score_errors) if score_errors else None
        ),
        "review_utility": _review_utility(labeled, top_k=top_k),
        "metric_type": "pairwise_accuracy" if judge and judge.output_type == OutputType.PAIRWISE else "class_accuracy",
        "rank_correlation": None,
        "tolerance": None,
    }


def _scalar_correctness(
    risks: list[ItemRisk],
    *,
    top_k: int,
    judge: JudgeConfig,
) -> dict[str, Any]:
    pairs = [
        (float(risk.original_score), float(risk.gold_label), risk)
        for risk in risks
        if risk.original_score is not None and _is_number(risk.gold_label)
    ]
    errors = [abs(predicted - gold) for predicted, gold, _ in pairs]
    tolerance = judge.gold_tolerance
    tolerance_errors = (
        [error > tolerance for error in errors] if tolerance is not None else []
    )
    evaluable = [risk for _, _, risk in pairs]
    return {
        "has_gold_labels": bool(pairs),
        "labeled_items": len(pairs),
        "correct": sum(not error for error in tolerance_errors) if tolerance is not None else 0,
        "errors": sum(tolerance_errors) if tolerance is not None else 0,
        "accuracy": (
            sum(not error for error in tolerance_errors) / len(pairs)
            if tolerance is not None and pairs
            else None
        ),
        "error_rate": (
            sum(tolerance_errors) / len(pairs)
            if tolerance is not None and pairs
            else None
        ),
        "confusion_matrix": {},
        "score_items": len(pairs),
        "mean_absolute_score_error": sum(errors) / len(errors) if errors else None,
        "rank_correlation": _spearman(
            [predicted for predicted, _, _ in pairs],
            [gold for _, gold, _ in pairs],
        ),
        "review_utility": (
            _review_utility(evaluable, top_k=top_k) if tolerance is not None else None
        ),
        "metric_type": "scalar",
        "tolerance": tolerance,
    }


def _confusion_matrix(risks: list[ItemRisk]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for risk in risks:
        gold = _label_key(risk.gold_label)
        predicted = _label_key(risk.original_verdict)
        counts[gold][predicted] += 1
    return {
        gold: dict(sorted(predicted_counts.items()))
        for gold, predicted_counts in sorted(counts.items())
    }


def _review_utility(risks: list[ItemRisk], *, top_k: int) -> dict[str, Any]:
    ranked = sorted(
        risks,
        key=lambda risk: (
            risk.routing_rank is None,
            risk.routing_rank or 0,
            str(risk.item_id),
        ),
    )
    routed = ranked[:top_k]
    total_errors = sum(1 for risk in risks if _judge_error(risk) is True)
    routed_errors = sum(1 for risk in routed if _judge_error(risk) is True)
    return {
        "top_k": top_k,
        "routed_labeled_items": len(routed),
        "routed_errors": routed_errors,
        "routed_precision": routed_errors / len(routed) if routed else None,
        "error_capture_rate": (
            routed_errors / total_errors if total_errors else None
        ),
    }


def _label_key(value: Any) -> str:
    return "null" if value is None else str(value).strip().lower()


def _judge_error(risk: ItemRisk) -> bool | None:
    if risk.judge_error is not None:
        return risk.judge_error
    if risk.original_verdict is None or risk.gold_label is None:
        return None
    return _label_key(risk.original_verdict) != _label_key(risk.gold_label)


def _is_number(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _spearman(predicted: list[float], gold: list[float]) -> float | None:
    if len(predicted) < 2 or len(predicted) != len(gold):
        return None
    x = _ranks(predicted)
    y = _ranks(gold)
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y))
    x_ss = sum((a - x_mean) ** 2 for a in x)
    y_ss = sum((b - y_mean) ** 2 for b in y)
    if x_ss == 0 or y_ss == 0:
        return None
    return numerator / (x_ss * y_ss) ** 0.5


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        average_rank = (index + 1 + end) / 2
        for ordered_index in order[index:end]:
            ranks[ordered_index] = average_rank
        index = end
    return ranks
