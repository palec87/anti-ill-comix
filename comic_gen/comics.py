from __future__ import annotations

import json
import re
from pathlib import Path
from time import perf_counter
from typing import Any

from . import exercise
from .image_backend import ImageGenerationError, MAX_SEED, generate_panel_image
from .text_backend import (
    TEXT_BACKEND_PIPELINE,
    TextGenerationError,
    UnifiedGenerationError,
    generate_characters_from_text,
    generate_session_fields_from_article,
)
from .trace import add_trace

STYLE_SCENE_HINTS = {
    "newspaper": "clean lines, documentary tone",
    "watercolor": "soft colors and gentle brush texture",
    "minimal": "simple shapes and high readability",
    "retro": "vintage ink and halftone texture",
}


class ModelPipelineError(RuntimeError):
    pass


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


def _extract_json_array(raw_text: str) -> list[dict[str, Any]] | None:
    start = raw_text.find("[")
    end = raw_text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = raw_text[start:end + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, list):
        return parsed
    return None


def generate_characters(
    document: dict[str, Any],
    enable_model_generation: bool = False,
    model_repo_id: str = "openbmb/MiniCPM5-1B",
    text_backend_mode: str = TEXT_BACKEND_PIPELINE,
) -> None:
    language = document.get("language", "en")
    fulltext = str(document.get("article", {}).get("fulltext", "")).strip()
    summary_for_prompt = " ".join(_split_sentences(fulltext)[:3])
    characters: list[dict[str, str]] = []

    if enable_model_generation and fulltext:
        try:
            prompt_text = fulltext
            if summary_for_prompt:
                prompt_text = (
                    f"Article summary seed: {summary_for_prompt}\n"
                    f"Article full text: {fulltext}"
                )
            raw = generate_characters_from_text(
                fulltext=prompt_text,
                language=language,
                model_repo_id=model_repo_id,
                backend_mode=text_backend_mode,
            )
            extracted = _extract_json_array(raw)
            if extracted is None:
                raise TextGenerationError("model output is not a JSON array")
            characters = _normalize_characters(extracted)
            if not characters:
                raise TextGenerationError(
                    "model output missing 2-3 characters"
                )
            add_trace(
                document,
                "step3_characters",
                "ok",
                f"Characters generated from full text using {model_repo_id}",
            )
        except TextGenerationError as exc:
            fallback = _example_characters(language)
            characters = fallback
            add_trace(
                document,
                "step3_characters",
                "fallback",
                f"Model character generation failed: {exc}",
            )
    else:
        characters = _example_characters(language)
        add_trace(
            document,
            "step3_characters",
            "ok",
            "Characters loaded from deterministic examples",
        )

    document["characters"] = characters


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

    document["panels"] = panels
    _apply_image_generation_to_panels(document, panels, options)


def _deterministic_pipeline(
    document: dict[str, Any],
    panel_count: int,
) -> None:
    simplify_article(document)
    generate_characters(document, enable_model_generation=False)
    generate_panels(
        document,
        panel_count=panel_count,
        image_options={"enable_live_images": False},
    )
    apply_overlay(document)
    exercise.generate_exercises(document)


def _normalize_model_fields(
    generated: dict[str, Any],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    simplified = generated.get("simplified")
    characters = generated.get("characters")
    panels = generated.get("panels")
    exercises_data = generated.get("exercises")
    if not isinstance(simplified, dict):
        raise ModelPipelineError("model payload missing simplified")
    if not isinstance(characters, list):
        raise ModelPipelineError("model payload missing characters")
    if not isinstance(panels, list):
        raise ModelPipelineError("model payload missing panels")
    if not isinstance(exercises_data, list):
        raise ModelPipelineError("model payload missing exercises")
    return simplified, characters, panels, exercises_data


def generate_story_pipeline(
    document: dict[str, Any],
    panel_count: int,
    enable_model_generation: bool,
    text_model_repo_id: str,
    image_options: dict[str, Any] | None = None,
    text_backend_mode: str = TEXT_BACKEND_PIPELINE,
) -> None:
    if not enable_model_generation:
        add_trace(
            document,
            "model_pipeline",
            "ok",
            "Model mode disabled, using deterministic pipeline",
        )
        _deterministic_pipeline(document, panel_count=panel_count)
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
            backend_mode=text_backend_mode,
        )
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
        _deterministic_pipeline(document, panel_count=panel_count)


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
