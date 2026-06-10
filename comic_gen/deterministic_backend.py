from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from . import exercise
from .text_utils import split_sentences
from .trace import add_trace

STYLE_SCENE_HINTS = {
    "newspaper": "clean lines, documentary tone",
    "watercolor": "soft colors and gentle brush texture",
    "minimal": "simple shapes and high readability",
    "retro": "vintage ink and halftone texture",
}


def _example_characters(language: str) -> list[dict[str, str]] | None:
    repo_root = Path(__file__).resolve().parents[1]
    example_path = repo_root / "examples" / f"{language}_demo.json"
    if not example_path.exists():
        return None
    try:
        payload = json.loads(example_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    characters = payload.get("characters")
    if not isinstance(characters, list) or not characters:
        return None
    normalized = _normalize_characters(characters)
    if not normalized:
        return None
    return normalized


def _normalize_characters(raw: Any) -> list[dict[str, str]]:
    if not isinstance(raw, list):
        return []

    normalized: list[dict[str, str]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        description = str(item.get("description", "")).strip()
        char_id = str(item.get("id", "")).strip()
        if not name or not description:
            continue
        if not char_id:
            slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
            char_id = f"char_{slug or index + 1}"
        normalized.append(
            {
                "id": char_id,
                "name": name,
                "description": description,
            }
        )
        if len(normalized) == 3:
            break

    return normalized if len(normalized) >= 2 else []


def _generate_characters(document: dict[str, Any]) -> None:
    """Load deterministic example characters for the selected language."""
    language = str(document.get("language", "en"))
    document["characters"] = _example_characters(language)
    add_trace(
        document,
        "step3_characters",
        "ok",
        "Characters loaded from deterministic examples",
    )


def _short_line(text: str, limit: int = 72) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    trimmed = text[: limit - 3].rstrip()
    return f"{trimmed}..."


def _generate_panels(
    document: dict[str, Any],
    panel_count: int = 3,
) -> None:
    """Create deterministic panel scripts and image references."""
    panel_count = max(3, min(5, panel_count))
    summary_sentences = split_sentences(document["simplified"]["summary"])
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
        image_path = f"assets/panel_{idx + 1}.png"
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
                    "image_path": image_path,
                    "image_source": "deterministic",
                    "overlay_applied": False,
                },
            }
        )

    document["panels"] = panels
    add_trace(
        document,
        "step4_panels",
        "ok",
        f"Generated {len(panels)} comic panels (deterministic={len(panels)})",
    )


def apply_overlay(document: dict[str, Any]) -> None:
    """Mark overlay metadata as applied for all generated panels."""
    for panel in document.get("panels", []):
        panel.setdefault("render", {})["overlay_applied"] = True
    add_trace(
        document,
        "step5_overlay",
        "ok",
        "Overlay metadata created for all panels",
    )


def deterministic_pipeline(
    document: dict[str, Any],
    panel_count: int,
) -> None:
    """Run deterministic end-to-end generation without model inference."""
    _simplify_article(document)
    _generate_characters(document)
    _generate_panels(document, panel_count=panel_count)
    apply_overlay(document)
    exercise.generate_exercises(document)


def _simplify_article(document: dict[str, Any]) -> None:
    """Build a short A2 summary and keyword list from article fulltext."""
    text = document["article"]["fulltext"]
    sentences = split_sentences(text)
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
