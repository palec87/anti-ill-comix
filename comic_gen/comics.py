from __future__ import annotations

import re
from typing import Any

from .trace import add_trace

STYLE_SCENE_HINTS = {
    "newspaper": "clean lines, documentary tone",
    "watercolor": "soft colors and gentle brush texture",
    "minimal": "simple shapes and high readability",
    "retro": "vintage ink and halftone texture",
}


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def simplify_article(document: dict[str, Any]) -> None:
    text = document["article"]["fulltext"]
    sentences = _split_sentences(text)
    top = sentences[:3] if sentences else [text]
    summary = " ".join(top)

    words = re.findall(r"[A-Za-z]{4,}", summary.lower())
    keywords = []
    for w in words:
        if w not in keywords:
            keywords.append(w)
        if len(keywords) == 6:
            break

    document["simplified"] = {
        "summary": summary,
        "level": "A2",
        "keywords": keywords,
    }
    add_trace(document, "step2_simplify", "ok", "Simplified summary generated")


def generate_characters(document: dict[str, Any]) -> None:
    language = document.get("language", "en")
    if language == "es":
        characters = [
            {
                "id": "char_guide",
                "name": "Guia",
                "description": (
                    "Persona que explica la noticia con palabras simples."
                ),
            },
            {
                "id": "char_learner",
                "name": "Estudiante",
                "description": "Adulto que practica lectura y escritura.",
            },
        ]
    else:
        characters = [
            {
                "id": "char_guide",
                "name": "Guide",
                "description": (
                    "Supportive mentor who explains the story in plain words."
                ),
            },
            {
                "id": "char_learner",
                "name": "Learner",
                "description": (
                    "Adult student practicing reading and writing skills."
                ),
            },
        ]

    document["characters"] = characters
    add_trace(document, "step3_characters", "ok", "Characters generated")


def _short_line(text: str, limit: int = 72) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    trimmed = text[: limit - 3].rstrip()
    return f"{trimmed}..."


def generate_panels(document: dict[str, Any], panel_count: int = 3) -> None:
    panel_count = max(3, min(5, panel_count))
    summary_sentences = _split_sentences(document["simplified"]["summary"])
    style_id = document.get("style_id", "minimal")
    style_hint = STYLE_SCENE_HINTS.get(style_id, STYLE_SCENE_HINTS["minimal"])

    if not summary_sentences:
        summary_sentences = [document["simplified"]["summary"]]

    panels = []
    for idx in range(panel_count):
        sentence = summary_sentences[idx % len(summary_sentences)]
        guide_line = _short_line(sentence)
        learner_line = _short_line(
            "I understand. I will write one key idea from this panel."
        )

        panel_id = f"panel_{idx + 1}"
        panels.append(
            {
                "panel_id": panel_id,
                "frame_index": idx + 1,
                "scene_description": (
                    f"Comic panel {idx + 1} in {style_id} style, "
                    f"{style_hint}. "
                    "Two adults discuss the news in a learning workshop."
                ),
                "dialogue": [
                    {"character_id": "char_guide", "text": guide_line},
                    {"character_id": "char_learner", "text": learner_line},
                ],
                "bubbles": [
                    {"bbox_px": [30, 30, 300, 90]},
                    {"bbox_px": [30, 140, 300, 90]},
                ],
                "render": {
                    "image_path": f"assets/panel_{idx + 1}.png",
                    "overlay_applied": True,
                },
            }
        )

    document["panels"] = panels
    add_trace(
        document,
        "step4_panels",
        "ok",
        f"Generated {len(panels)} comic panels",
    )


def apply_overlay(document: dict[str, Any]) -> None:
    # Deterministic MVP keeps overlay data in JSON for predictable replay.
    add_trace(
        document,
        "step5_overlay",
        "ok",
        "Overlay metadata created for all panels",
    )
