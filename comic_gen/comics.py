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


def _normalize_model_repo_id(value: Any) -> str:
    """Return a valid model repo id string from UI/config values."""
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            value = value[0]
        else:
            value = value[0] if value else ""
    model_repo_id = str(value or "black-forest-labs/FLUX.1-schnell").strip()
    return model_repo_id or "black-forest-labs/FLUX.1-schnell"


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
    options["model_repo_id"] = _normalize_model_repo_id(
        options.get("model_repo_id", "black-forest-labs/FLUX.1-schnell")
    )
    options["negative_prompt"] = str(options.get("negative_prompt", ""))
    return options


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
        try:
            translate_session_content(
                document,
                target_language,
                source_language="en",
            )
        except Exception as exc:
            add_trace(
                document,
                "translation",
                "fallback",
                f"Translation failed, keeping canonical content: {exc}",
            )
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
        try:
            translate_session_content(
                document,
                target_language,
                source_language=str(document.get("language", target_language)),
            )
        except Exception as translate_exc:
            add_trace(
                document,
                "translation",
                "fallback",
                f"Translation failed after deterministic fallback: {translate_exc}",
            )
