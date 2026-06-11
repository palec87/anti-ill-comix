from __future__ import annotations

from typing import Any
import logging

from .image_backend import MAX_SEED, generate_image_panels
from .text_backend import (
    TextGenerationError,
    UnifiedGenerationError,
    generate_text_content_from_article,
    _normalize_model_fields,
)
from .backends import deterministic_pipeline
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
    options["use_serverless_image_api"] = bool(
        options.get("use_serverless_image_api", False)
    )
    options["randomize_seed"] = bool(options.get("randomize_seed", True))
    options["model_repo_id"] = str(
        options.get("model_repo_id", "stabilityai/sdxl-turbo")
    )
    options["negative_prompt"] = str(options.get("negative_prompt", ""))
    return options


def generate_story_pipeline(
    document: dict[str, Any],
    panel_count: int,
    enable_model_generation: bool,
    text_model_repo_id: str,
    image_options: dict[str, Any] | None = None,
) -> None:
    if not enable_model_generation:
        logger.info("Running deterministic pipeline")
        deterministic_pipeline(document)
        return

    options = _normalized_image_options(image_options)
    options["enable_live_images"] = True
    logger.info("Running unified model pipeline (text + image)")

    try:
        generated = generate_text_content_from_article(
            language=str(document.get("language", "en")),
            style_id=str(document.get("style_id", "minimal")),
            article=document.get("article", {}),
            panel_count=panel_count,
            model_repo_id=text_model_repo_id,
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
