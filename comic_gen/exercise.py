from __future__ import annotations

import re
from typing import Any

from .trace import add_trace

FEEDBACK_TRANSLATIONS = {
    "en": {
        "missing": "Exercise not found for the selected panel.",
        "correct": "Correct. Great writing practice.",
        "retry": "Not yet. Try again with a central word from the panel (expected: {expected}).",
    },
    "pt": {
        "missing": "Exercicio nao encontrado para o quadro selecionado.",
        "correct": "Correto. Otima pratica de escrita.",
        "retry": "Ainda nao. Tente outra vez com uma palavra central do quadro (esperado: {expected}).",
    },
    "es": {
        "missing": "No se encontro ejercicio para la vineta seleccionada.",
        "correct": "Correcto. Buena practica de escritura.",
        "retry": "Todavia no. Intenta otra vez con una palabra central de la vineta (esperado: {expected}).",
    },
    "fr": {
        "missing": "Aucun exercice trouve pour la case choisie.",
        "correct": "Correct. Bonne pratique d'ecriture.",
        "retry": "Pas encore. Reessayez avec un mot central de la case (attendu: {expected}).",
    },
    "de": {
        "missing": "Keine Uebung fuer das ausgewaehlte Bild gefunden.",
        "correct": "Richtig. Gute Schreibuebung.",
        "retry": "Noch nicht. Versuche es mit einem wichtigen Wort aus dem Bild (erwartet: {expected}).",
    },
}


def _feedback_text(language: str, key: str, **kwargs: str) -> str:
    """Return localized exercise feedback text."""
    template = FEEDBACK_TRANSLATIONS.get(language, FEEDBACK_TRANSLATIONS["en"]).get(
        key,
        FEEDBACK_TRANSLATIONS["en"][key],
    )
    return template.format(**kwargs)


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
        return False, _feedback_text(
            str(document.get("language", "en")),
            "missing",
        )

    expected = exercise["answer_key"][0]
    normalized_user = user_answer.strip().lower()
    normalized_expected = expected.strip().lower()
    ok = normalized_user == normalized_expected
    language = str(document.get("language", "en"))
    if ok:
        return True, _feedback_text(language, "correct")

    return (
        False,
        _feedback_text(language, "retry", expected=expected),
    )
