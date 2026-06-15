from __future__ import annotations

from typing import Any
import logging

from .image_backend import MAX_SEED, generate_image_panels
from .text_backend import (
    TextGenerationError,
    UnifiedGenerationError,
    generate_text_content_from_article,
)
from .text_utils import _normalize_model_fields
from .backends import deterministic_pipeline
from .translation_backend import translate_session_content
from .trace import add_trace
from .errors import ModelPipelineError

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def _default_image_options() -> dict[str, Any]:
    return {
        "enable_live_images": False,
        "use_serverless_image_api": False,
        "model_repo_id": "black-forest-labs/FLUX.1-schnell",
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
    options["use_serverless_image_api"] = bool(
        options.get("use_serverless_image_api", False)
    )
    options["randomize_seed"] = bool(options.get("randomize_seed", True))
    model_repo_id = options.get(
        "model_repo_id",
        "black-forest-labs/FLUX.1-schnell",
    )
    if isinstance(model_repo_id, tuple) and len(model_repo_id) == 1:
        model_repo_id = model_repo_id[0]
    options["model_repo_id"] = model_repo_id
    options["negative_prompt"] = str(options.get("negative_prompt", ""))
    return options


def _translate_or_fallback_to_english(
    document: dict[str, Any],
    target_language: str,
    source_language: str = "en",
) -> None:
    """Translate learner content or mark English fallback."""
    ui = document.setdefault("ui", {})
    if target_language == "en":
        ui["content_language"] = "en"
        return

    try:
        translated = translate_session_content(
            document,
            target_language,
            source_language=source_language,
        )
        ui["content_language"] = target_language if translated else source_language
    except Exception as exc:
        ui["content_language"] = "en"
        add_trace(
            document,
            "translation",
            "fallback",
            f"Translation failed, keeping English content: {exc}",
        )


def _sync_overlay_text_from_dialogue(document: dict[str, Any]) -> None:
    """Copy canonical dialogue text into bubble render metadata."""
    synced = 0
    for panel in document.get("panels", []):
        if not isinstance(panel, dict):
            continue
        dialogue = panel.get("dialogue", [])
        bubbles = panel.get("bubbles", [])
        if not isinstance(dialogue, list) or not isinstance(bubbles, list):
            continue
        for index, bubble in enumerate(bubbles):
            if not isinstance(bubble, dict) or index >= len(dialogue):
                continue
            line = dialogue[index]
            if not isinstance(line, dict):
                continue
            text = str(line.get("text", "")).strip()
            if not text:
                continue
            if bubble.get("text") != text:
                bubble["text"] = text
                synced += 1

    if synced:
        add_trace(
            document,
            "overlay_text",
            "synced",
            f"Synced {synced} bubble texts from dialogue",
        )


def generate_story_pipeline(
    document: dict[str, Any],
    panel_count: int,
    text_model_repo_id: str,
    reading_level: str = "A2",
    image_options: dict[str, Any] | None = None,
) -> None:
    options = _normalized_image_options(image_options)
    options["enable_live_images"] = True
    target_language = str(document.get("language", "en"))
    logger.info("Running unified model pipeline (text + image)")

    try:
        generated = generate_text_content_from_article(
            language="en",
            style_id=str(document.get("style_id", "minimal")),
            reading_level=reading_level,
            article=document.get("article", {}),
            panel_count=panel_count,
            model_repo_id=text_model_repo_id,
        )
        repairs = generated.get("_normalization_repairs", [])
        if isinstance(repairs, list) and repairs:
            add_trace(
                document,
                "model_normalization",
                "repaired",
                f"Repaired {', '.join(str(item) for item in repairs)}",
            )
        logger.info(f"Simplified output in generate_story_pipeline: {generated.get('simplified', {})}")
        (
            simplified,
            characters,
            panels,
            exercises_data,
        ) = _normalize_model_fields(generated)
        document["simplified"] = simplified
        document["characters"] = characters
        document["panels"] = panels
        document["exercises"] = exercises_data
        _translate_or_fallback_to_english(
            document,
            target_language,
            source_language="en",
        )
        _sync_overlay_text_from_dialogue(document)
        add_trace(
            document,
            "model_pipeline",
            "ok",
            f"Model text fields generated using {text_model_repo_id}",
        )
        generate_image_panels(
            document,
            document["panels"],
            options,
            strict_mode=False,
        )
        add_trace(
            document,
            "step6_exercises",
            "ok",
            f"Loaded {len(document.get('exercises', []))} model exercises",
        )
        add_trace(
            document,
            "model_pipeline",
            "ok",
            "Unified model pipeline completed",
        )
    except (
        UnifiedGenerationError,
        TextGenerationError,
        ModelPipelineError,
    ) as exc:
        add_trace(
            document,
            "model_pipeline",
            "fallback",
            f"Model pipeline failed, switching to deterministic: {exc}",
        )
        deterministic_pipeline(
            document,
        )
        document.setdefault("simplified", {})["level"] = reading_level
        content_language = str(document.get("language", target_language))
        if content_language == target_language:
            document.setdefault("ui", {})["content_language"] = target_language
        else:
            _translate_or_fallback_to_english(
                document,
                target_language,
                source_language=content_language or "en",
            )
        _sync_overlay_text_from_dialogue(document)
