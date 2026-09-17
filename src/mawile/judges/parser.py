from __future__ import annotations

import json
import math
import re
from typing import Any, Literal

from mawile.schemas import Item, OutputType, ScoreDirection


ParseStatus = Literal["ok", "error"]


def parse_judge_output(
    raw_output: str,
    output_type: OutputType,
    threshold: float | None = None,
    *,
    score_direction: ScoreDirection = ScoreDirection.HIGHER_IS_BETTER,
    score_min: float | None = None,
    score_max: float | None = None,
    item: Item | None = None,
) -> tuple[dict[str, Any] | str | int | float | bool | None, ParseStatus]:
    parsed_json = _parse_json_object(raw_output)
    if isinstance(parsed_json, dict):
        verdict = _normalize_dict(
            parsed_json, output_type, threshold, score_direction, item
        )
        return verdict, _status_for(verdict, output_type, score_min, score_max, item)

    verdict = _parse_from_text(
        raw_output, output_type, threshold, score_direction, item
    )
    return verdict, _status_for(verdict, output_type, score_min, score_max, item)


def _normalize_dict(
    payload: dict[str, Any],
    output_type: OutputType,
    threshold: float | None,
    score_direction: ScoreDirection,
    item: Item | None,
) -> dict[str, Any]:
    normalized = dict(payload)
    candidate = _nested_verdict_candidate(payload)

    if "score" not in normalized:
        for source in (candidate, payload):
            if not isinstance(source, dict):
                continue
            for key in ("score", "rating", "value", "grade"):
                if key in source:
                    normalized["score"] = source[key]
                    break
            if "score" in normalized:
                break

    score = _coerce_float(normalized.get("score"))
    if score is not None:
        normalized["score"] = score

    label = _first_present(
        normalized,
        candidate,
        keys=("label", "verdict", "decision", "preference", "winner"),
    )
    normalized_label = (
        _normalize_pairwise_label(label)
        if output_type == OutputType.PAIRWISE
        else _normalize_label(label)
    )
    if normalized_label is None and score is not None and threshold is not None:
        normalized_label = _threshold_label(score, threshold, score_direction)
    if normalized_label is not None:
        normalized["label"] = normalized_label
        if output_type == OutputType.PAIRWISE:
            normalized["position"] = normalized_label
            if item is not None and item.is_pairwise:
                normalized["candidate_id"] = item.candidate_id_at(normalized_label)
    elif output_type in {OutputType.BINARY, OutputType.PAIRWISE}:
        # Do not let an unrecognized natural-language label survive as a
        # syntactically present verdict field (e.g. "incorrect; it does not
        # pass").  The caller should receive a parse error instead.
        normalized.pop("label", None)

    reason = _first_present(normalized, candidate, keys=("reason", "explanation", "rationale"))
    if reason is not None and "reason" not in normalized:
        normalized["reason"] = reason

    return normalized


def _parse_from_text(
    raw_output: str,
    output_type: OutputType,
    threshold: float | None,
    score_direction: ScoreDirection,
    item: Item | None,
) -> dict[str, Any] | None:
    score = _extract_score(raw_output)
    label = (
        _normalize_pairwise_label(raw_output)
        if output_type == OutputType.PAIRWISE
        else _normalize_label(raw_output)
    )
    if label is None and score is not None and threshold is not None:
        label = _threshold_label(score, threshold, score_direction)

    verdict: dict[str, Any] = {}
    if score is not None:
        verdict["score"] = score
    if label is not None:
        verdict["label"] = label
        if output_type == OutputType.PAIRWISE:
            verdict["position"] = label
            if item is not None and item.is_pairwise:
                verdict["candidate_id"] = item.candidate_id_at(label)
    if raw_output.strip():
        verdict["reason"] = raw_output.strip()

    if output_type == OutputType.BINARY and "label" not in verdict:
        return None
    if output_type == OutputType.PAIRWISE and "label" not in verdict:
        return None
    if output_type in {OutputType.SCALAR, OutputType.ORDINAL} and "score" not in verdict:
        return None
    return verdict or None


def _parse_json_object(raw_output: str) -> Any | None:
    text = raw_output.strip()
    # Verdicts are an interface, not prose to be mined for a plausible object.
    # Requiring the complete response to be JSON prevents a rationale containing
    # words such as "pass" from becoming a binary verdict by accident.
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[0].strip().casefold() in {"```", "```json"}:
            text = "\n".join(lines[1:-1]).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _nested_verdict_candidate(payload: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("judgment", "verdict", "result", "decision"):
        value = payload.get(key)
        if isinstance(value, dict):
            return value

    fields = payload.get("fields")
    if isinstance(fields, list):
        dict_fields = [field for field in fields if isinstance(field, dict)]
        if len(dict_fields) == 1:
            return dict_fields[0]
    if isinstance(fields, dict):
        return fields
    return None


def _first_present(
    primary: dict[str, Any],
    secondary: dict[str, Any] | None,
    keys: tuple[str, ...],
) -> Any | None:
    for source in (primary, secondary):
        if not isinstance(source, dict):
            continue
        for key in keys:
            if key in source:
                return source[key]
    return None


def _extract_score(text: str) -> float | None:
    patterns = [
        r"\bscore\s*[:=]\s*(-?\d+(?:\.\d+)?)\b",
        r"\brating\s*[:=]\s*(-?\d+(?:\.\d+)?)\b",
        r"\b(-?\d+(?:\.\d+)?)\s*/\s*\d+(?:\.\d+)?\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return _coerce_float(match.group(1))
    return None


def _normalize_label(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().casefold().strip(" \t\r\n.,:;!?()[]{}")
    positive = {"pass", "passed", "true", "yes", "acceptable", "success", "correct"}
    negative = {"fail", "failed", "false", "no", "unacceptable", "failure", "incorrect"}
    if text in positive:
        return "pass"
    if text in negative:
        return "fail"
    return None


def _normalize_pairwise_label(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    a_values = {
        "a",
        "candidate_a",
        "candidate a",
        "output_a",
        "output 1",
        "response_a",
        "first",
        "left",
        "1",
    }
    b_values = {
        "b",
        "candidate_b",
        "candidate b",
        "output_b",
        "output 2",
        "response_b",
        "second",
        "right",
        "2",
    }
    if text in a_values:
        return "A"
    if text in b_values:
        return "B"
    return None


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    except (TypeError, ValueError):
        return None


def _status_for(
    verdict: Any,
    output_type: OutputType,
    score_min: float | None,
    score_max: float | None,
    item: Item | None,
) -> ParseStatus:
    if output_type == OutputType.STRUCTURED:
        return "ok" if isinstance(verdict, dict) else "error"
    if output_type == OutputType.BINARY:
        return "ok" if isinstance(verdict, dict) and verdict.get("label") else "error"
    if output_type == OutputType.PAIRWISE:
        if not isinstance(verdict, dict) or not verdict.get("label"):
            return "error"
        if item is not None and item.is_pairwise and not verdict.get("candidate_id"):
            return "error"
        return "ok"
    if output_type in {OutputType.SCALAR, OutputType.ORDINAL}:
        if not isinstance(verdict, dict) or verdict.get("score") is None:
            return "error"
        score = _coerce_float(verdict.get("score"))
        if score is None:
            return "error"
        if score_min is not None and score < score_min:
            return "error"
        if score_max is not None and score > score_max:
            return "error"
        return "ok"
    return "ok" if verdict is not None else "error"


def _threshold_label(
    score: float,
    threshold: float,
    direction: ScoreDirection,
) -> str:
    if direction == ScoreDirection.HIGHER_IS_BETTER:
        return "pass" if score >= threshold else "fail"
    return "pass" if score <= threshold else "fail"
