from __future__ import annotations

import re
from time import perf_counter
from typing import Any

from .image_backend import ImageGenerationError, MAX_SEED, generate_panel_image
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


def _default_image_options() -> dict[str, Any]:
    return {
        "enable_live_images": False,
        "model_repo_id": "stabilityai/sdxl-turbo",
        "negative_prompt": "",
        "seed": 0,
        "randomize_seed": True,
        "width": 256,
        "height": 256,
        "guidance_scale": 0.0,
        "num_inference_steps": 2,
    }


def _clamp_dimension(value: Any) -> int:
    try:
        v = int(value)
    except (TypeError, ValueError):
        v = 256
    v = max(256, min(512, v))
    return v - (v % 32)


def _normalized_image_options(
    image_options: dict[str, Any] | None,
) -> dict[str, Any]:
    options = _default_image_options()
    if image_options:
        options.update(image_options)

    options["width"] = _clamp_dimension(options.get("width", 256))
    options["height"] = _clamp_dimension(options.get("height", 256))
    options["seed"] = max(0, min(MAX_SEED, int(options.get("seed", 0))))
    options["num_inference_steps"] = max(
        1,
        min(50, int(options.get("num_inference_steps", 2))),
    )
    options["guidance_scale"] = float(options.get("guidance_scale", 0.0))
    options["enable_live_images"] = bool(
        options.get("enable_live_images", False)
    )
    options["randomize_seed"] = bool(options.get("randomize_seed", True))
    options["model_repo_id"] = str(
        options.get("model_repo_id", "stabilityai/sdxl-turbo")
    )
    options["negative_prompt"] = str(options.get("negative_prompt", ""))
    return options


def generate_panels(
    document: dict[str, Any],
    panel_count: int = 3,
    image_options: dict[str, Any] | None = None,
) -> None:
    panel_count = max(3, min(5, panel_count))
    options = _normalized_image_options(image_options)
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
        fallback_path = f"assets/panel_{idx + 1}.png"
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
                    "image_path": fallback_path,
                    "overlay_applied": False,
                },
            }
        )

    if options["enable_live_images"]:
        for panel in panels:
            panel_id = panel["panel_id"]
            fallback_path = panel["render"]["image_path"]
            panel_seed = options["seed"] + (panel["frame_index"] - 1)
            started = perf_counter()
            add_trace(
                document,
                "step4_image_generate",
                "start",
                f"{panel_id} generation started",
            )
            try:
                image_path, used_seed, device = generate_panel_image(
                    prompt=panel["scene_description"],
                    negative_prompt=options["negative_prompt"],
                    session_id=document["session_id"],
                    panel_id=panel_id,
                    model_repo_id=options["model_repo_id"],
                    seed=panel_seed,
                    randomize_seed=options["randomize_seed"],
                    width=options["width"],
                    height=options["height"],
                    guidance_scale=options["guidance_scale"],
                    num_inference_steps=options["num_inference_steps"],
                )
                elapsed_ms = int((perf_counter() - started) * 1000)
                panel["render"]["image_path"] = image_path
                panel["render"]["image_source"] = "live"
                panel["render"]["seed"] = used_seed
                panel["render"]["device"] = device
                add_trace(
                    document,
                    "step4_image_generate",
                    "ok",
                    f"{panel_id} live image generated in {elapsed_ms}ms",
                )
            except ImageGenerationError as exc:
                elapsed_ms = int((perf_counter() - started) * 1000)
                panel["render"]["image_path"] = fallback_path
                panel["render"]["image_source"] = "fallback"
                panel["render"]["seed"] = panel_seed
                add_trace(
                    document,
                    "step4_image_generate",
                    "fallback",
                    f"{panel_id} fallback after {elapsed_ms}ms: {exc}",
                )
    else:
        for panel in panels:
            panel["render"]["image_source"] = "deterministic"

    document["panels"] = panels
    add_trace(
        document,
        "step4_panels",
        "ok",
        f"Generated {len(panels)} comic panels",
    )


def apply_overlay(document: dict[str, Any]) -> None:
    # Phase 1 keeps overlay in the UI layer; preserve schema flag consistency.
    for panel in document.get("panels", []):
        panel.setdefault("render", {})["overlay_applied"] = True
    add_trace(
        document,
        "step5_overlay",
        "ok",
        "Overlay metadata created for all panels",
    )
