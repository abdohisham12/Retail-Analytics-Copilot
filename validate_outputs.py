"""Validate agent JSONL outputs against the expected contract.

Usage:
    python validate_outputs.py --pred outputs_hybrid.jsonl --questions sample_questions_hybrid_eval.jsonl
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click


@dataclass
class QuestionSpec:
    id: str
    format_hint: str


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_float(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    return isinstance(value, (int, float))


def _match_struct(value: Any, spec: str) -> bool:
    if not isinstance(value, dict):
        return False
    spec = spec.strip("{} ")
    if not spec:
        return True
    for part in spec.split(","):
        name, _, type_name = part.strip().partition(":")
        name = name.strip()
        type_name = type_name.strip()
        if not name or not type_name:
            continue
        if name not in value:
            return False
        current = value[name]
        if type_name == "str" and not isinstance(current, str):
            return False
        if type_name == "int" and not _is_int(current):
            return False
        if type_name == "float" and not _is_float(current):
            return False
    return True


def _format_is_valid(format_hint: str, answer: Any) -> bool:
    if not format_hint:
        return True
    if format_hint == "int":
        return _is_int(answer)
    if format_hint == "float":
        return _is_float(answer)
    if format_hint.startswith("{") and format_hint.endswith("}"):
        return _match_struct(answer, format_hint)
    if format_hint.startswith("list["):
        if not isinstance(answer, list):
            return False
        inner = format_hint[len("list[") : -1]
        if inner.startswith("{"):
            return all(_match_struct(item, inner) for item in answer)
        if inner == "int":
            return all(_is_int(item) for item in answer)
        if inner == "float":
            return all(_is_float(item) for item in answer)
        if inner == "str":
            return all(isinstance(item, str) for item in answer)
    return True


def _check_explanation(explanation: str) -> List[str]:
    from agent.graph_hybrid import _normalize_explanation  # reuse logic

    issues: List[str] = []
    norm = _normalize_explanation(explanation or "")
    if not norm:
        issues.append("missing_explanation")
    if norm != explanation:
        issues.append("explanation_too_long_or_untrimmed")
    return issues


def _check_citations(citations: List[str]) -> List[str]:
    issues: List[str] = []
    if not citations:
        issues.append("missing_citations")
    for c in citations:
        if not isinstance(c, str) or len(c.strip()) < 3:
            issues.append("invalid_citation_token")
            break
    return issues


def validate_prediction(
    question: QuestionSpec, pred: Dict[str, Any]
) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    if pred.get("id") != question.id:
        errors.append("id_mismatch")

    answer = pred.get("final_answer", None)
    if answer is None:
        errors.append("missing_final_answer")
    elif not _format_is_valid(question.format_hint, answer):
        errors.append("format_mismatch")

    explanation = pred.get("explanation", "") or ""
    errors.extend(_check_explanation(explanation))

    citations = pred.get("citations", []) or []
    errors.extend(_check_citations(citations))

    ok = not errors
    return ok, errors


@click.command()
@click.option(
    "--pred",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="JSONL file with agent predictions.",
)
@click.option(
    "--questions",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="JSONL file with questions and format_hint.",
)
def main(pred: Path, questions: Path) -> None:
    q_rows = _load_jsonl(questions)
    p_rows = _load_jsonl(pred)
    by_id = {row["id"]: row for row in p_rows}

    total = len(q_rows)
    ok_count = 0

    for q in q_rows:
        spec = QuestionSpec(id=q["id"], format_hint=q.get("format_hint", ""))
        pred_row = by_id.get(spec.id)
        if not pred_row:
            print(f"[MISS] {spec.id}: no prediction row")
            continue
        ok, errors = validate_prediction(spec, pred_row)
        if ok:
            ok_count += 1
            print(f"[OK]   {spec.id}")
        else:
            print(f"[FAIL] {spec.id}: {', '.join(errors)}")

    print(f"\nSummary: {ok_count}/{total} predictions pass the contract checks.")


if __name__ == "__main__":
    main()


