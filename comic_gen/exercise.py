from __future__ import annotations

import re
from typing import Any

from .trace import add_trace


def _pick_keyword(document: dict[str, Any], fallback: str) -> str:
    keywords = document.get("simplified", {}).get("keywords", [])
    for word in keywords:
        if len(word) >= 4:
            return word
    return fallback


def _build_blank_prompt(
    line: str,
    keyword: str,
) -> tuple[str, list[str], list[str]]:
    if keyword.lower() in line.lower():
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        masked = pattern.sub("____", line, count=1)
        return masked, ["____"], [keyword]

    words = line.split()
    if not words:
        return "____", ["____"], [keyword]

    index = min(len(words) - 1, 2)
    answer = re.sub(r"[^A-Za-z]", "", words[index]) or keyword
    words[index] = "____"
    return " ".join(words), ["____"], [answer]


def generate_exercises(document: dict[str, Any]) -> None:
    exercises = []
    for panel in document.get("panels", []):
        panel_id = panel["panel_id"]
        first_line = (
            panel["dialogue"][0]["text"] if panel.get("dialogue") else ""
        )
        keyword = _pick_keyword(document, "learning")
        prompt_text, blanks, answers = _build_blank_prompt(first_line, keyword)
        exercises.append(
            {
                "exercise_id": f"ex_{panel_id}",
                "panel_id": panel_id,
                "prompt": prompt_text,
                "blanks": blanks,
                "answer_key": answers,
                "feedback_rules": {
                    "case_sensitive": False,
                    "allow_trim_spaces": True,
                },
            }
        )

    document["exercises"] = exercises
    add_trace(
        document,
        "step6_exercises",
        "ok",
        f"Generated {len(exercises)} exercises",
    )


def evaluate_answer(
    document: dict[str, Any], panel_id: str, user_answer: str
) -> tuple[bool, str]:
    exercise = next(
        (
            e
            for e in document.get("exercises", [])
            if e["panel_id"] == panel_id
        ),
        None,
    )
    if not exercise:
        return False, "Exercise not found for the selected panel."

    expected = exercise["answer_key"][0]
    normalized_user = user_answer.strip().lower()
    normalized_expected = expected.strip().lower()
    ok = normalized_user == normalized_expected
    if ok:
        return True, "Correct. Great writing practice."

    return (
        False,
        (
            "Not yet. Try again with a central word from the panel "
            f"(expected: {expected})."
        ),
    )
