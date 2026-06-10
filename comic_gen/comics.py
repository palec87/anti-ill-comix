from __future__ import annotations

from time import perf_counter
from typing import Any
import logging

from .image_backend import ImageGenerationError, MAX_SEED, generate_panel_image
from .text_backend import (
    TextGenerationError,
    UnifiedGenerationError,
    generate_session_fields_from_article,
    _normalize_model_fields,
)
from .deterministic_backend import apply_overlay, deterministic_pipeline
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


def _apply_image_generation_to_panels(
    document: dict[str, Any],
    panels: list[dict[str, Any]],
    options: dict[str, Any],
    strict_mode: bool = False,
) -> dict[str, int]:
    image_source_counts: dict[str, int] = {}
    if options["enable_live_images"]:
        for panel in panels:
            panel_id = panel.get("panel_id", "")
            frame_index = int(panel.get("frame_index", 1))
            render = panel.setdefault("render", {})
            fallback_path = str(
                render.get("image_path", f"assets/panel_{frame_index}.png")
            )
            panel_seed = options["seed"] + (frame_index - 1)
            started = perf_counter()
            add_trace(
                document,
                "step4_image_generate",
                "start",
                f"{panel_id} generation started",
            )
            try:
                image_path, used_seed, device = generate_panel_image(
                    document=document,
                    prompt=str(panel.get("scene_description", "")),
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
                render["image_path"] = image_path
                render["image_source"] = "live"
                image_source_counts["live"] = (
                    image_source_counts.get("live", 0) + 1
                )
                render["seed"] = used_seed
                render["device"] = device
                add_trace(
                    document,
                    "step4_image_generate",
                    "ok",
                    f"{panel_id} live image generated in {elapsed_ms}ms",
                )
            except ImageGenerationError as exc:
                elapsed_ms = int((perf_counter() - started) * 1000)
                if strict_mode:
                    raise ModelPipelineError(
                        f"image generation failed for {panel_id}: {exc}"
                    ) from exc
                render["image_path"] = fallback_path
                render["image_source"] = "fallback"
                image_source_counts["fallback"] = (
                    image_source_counts.get("fallback", 0) + 1
                )
                render["seed"] = panel_seed
                add_trace(
                    document,
                    "step4_image_generate",
                    "fallback",
                    f"{panel_id} fallback after {elapsed_ms}ms: {exc}",
                )
    else:
        for panel in panels:
            render = panel.setdefault("render", {})
            frame_index = int(panel.get("frame_index", 1))
            render.setdefault("image_path", f"assets/panel_{frame_index}.png")
            render["image_source"] = "deterministic"
            image_source_counts["deterministic"] = (
                image_source_counts.get("deterministic", 0) + 1
            )

    summary_bits = [
        f"{key}={value}"
        for key, value in image_source_counts.items()
    ]
    add_trace(
        document,
        "step4_panels",
        "ok",
        (
            f"Generated {len(panels)} comic panels"
            f" ({', '.join(summary_bits)})"
        ),
    )
    return image_source_counts


def generate_story_pipeline(
    document: dict[str, Any],
    panel_count: int,
    enable_model_generation: bool,
    text_model_repo_id: str,
    image_options: dict[str, Any] | None = None,
) -> None:
    if not enable_model_generation:
        add_trace(
            document,
            "model_pipeline",
            "ok",
            "Model mode disabled, using deterministic pipeline",
        )
        deterministic_pipeline(
            document,
            panel_count=panel_count,
        )
        return

    options = _normalized_image_options(image_options)
    options["enable_live_images"] = True
    add_trace(
        document,
        "model_pipeline",
        "start",
        "Unified model pipeline started (text + image)",
    )
    try:
        generated = generate_session_fields_from_article(
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
        logger.info(f"Document after normalization in generate_story_pipeline: {document['characters']}")
        add_trace(
            document,
            "model_pipeline",
            "ok",
            f"Model text fields generated using {text_model_repo_id}",
        )
        _apply_image_generation_to_panels(
            document,
            document["panels"],
            options,
            strict_mode=True,
        )
        apply_overlay(document)
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
            panel_count=panel_count,
        )
