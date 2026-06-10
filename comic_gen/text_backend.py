from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from .errors import ModelPipelineError
from .prompts import UNIFIED_SESSION_PROMPT

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

_GENERATOR: Any | None = None
_GENERATOR_MODEL_ID = ""
_INFERENCE_CLIENT: Any | None = None


class TextGenerationError(RuntimeError):
    pass


class UnifiedGenerationError(RuntimeError):
    pass


def _is_serverless_enabled() -> bool:
    value = os.environ.get("HF_USE_SERVERLESS", "0").strip().lower()
    logger.debug("HF_USE_SERVERLESS=%s", value)
    return value in {"1", "true", "yes", "on"}


def _get_inference_client() -> Any:
    global _INFERENCE_CLIENT
    if _INFERENCE_CLIENT is not None:
        return _INFERENCE_CLIENT

    token = os.environ.get("HF_TOKEN", "").strip()
    logger.debug("HF_TOKEN=%s", "****" if token else "(not set)")
    if not token:
        raise TextGenerationError(
            "HF_TOKEN is required for serverless API mode"
        )

    try:
        from huggingface_hub import InferenceClient
    except Exception as exc:
        raise TextGenerationError(
            "huggingface_hub import failed for serverless API mode"
        ) from exc

    _INFERENCE_CLIENT = InferenceClient(token=token)
    return _INFERENCE_CLIENT


def _generate_with_serverless_api(
    prompt: str,
    model_repo_id: str,
    max_new_tokens: int,
) -> str:
    client = _get_inference_client()
    try:
        response = client.chat_completion(
            model=model_repo_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
        )
        generated = response.choices[0].message.content
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        raise TextGenerationError(
            f"text generation failed (serverless_api), {model_repo_id}, {detail}"
        ) from exc

    if not isinstance(generated, str) or not generated.strip():
        raise TextGenerationError("invalid text generation output")
    return generated


def _get_generator(model_repo_id: str) -> Any:
    global _GENERATOR, _GENERATOR_MODEL_ID

    if _GENERATOR is not None and _GENERATOR_MODEL_ID == model_repo_id:
        return _GENERATOR

    try:
        import torch
        from transformers import pipeline
    except Exception as exc:
        raise TextGenerationError(
            "transformers/torch import failed"
        ) from exc

    device = 0 if torch.cuda.is_available() else -1
    try:
        generator = pipeline(
            "text-generation",
            model=model_repo_id,
            trust_remote_code=True,
            device=device,
        )
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        raise TextGenerationError(
            f"failed to load text model '{model_repo_id}' ({detail})"
        ) from exc

    _GENERATOR = generator
    _GENERATOR_MODEL_ID = model_repo_id
    return generator


def _generate_with_pipeline(
    prompt: str,
    model_repo_id: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
) -> str:
    if _is_serverless_enabled():
        try:
            generated = _generate_with_serverless_api(
                prompt=prompt,
                model_repo_id=model_repo_id,
                max_new_tokens=max_new_tokens,
            )
            return generated
        except TextGenerationError as exc:
            logger.warning(
                "Serverless API failed; falling back to local pipeline: %s",
                exc,
            )

    generator = _get_generator(model_repo_id)
    try:
        outputs = generator(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            return_full_text=False,
        )
    except Exception as exc:
        raise TextGenerationError("text generation failed (pipeline)") from exc

    if not outputs:
        raise TextGenerationError("empty text generation output")

    generated = outputs[0].get("generated_text", "")
    if not isinstance(generated, str) or not generated.strip():
        raise TextGenerationError("invalid text generation output")
    return generated


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise UnifiedGenerationError("model output missing JSON object")
    try:
        parsed = json.loads(raw_text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise UnifiedGenerationError("model output is not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise UnifiedGenerationError("model output root must be object")
    return parsed


def _normalize_simplified(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise UnifiedGenerationError("simplified must be object")
    summary = str(raw.get("summary", "")).strip()
    level = str(raw.get("level", "A2")).strip() or "A2"
    keywords_raw = raw.get("keywords", [])
    if not isinstance(keywords_raw, list):
        keywords_raw = []
    keywords: list[str] = []
    for item in keywords_raw:
        value = str(item).strip().lower()
        value = re.sub(r"[^a-zA-Z0-9_-]", "", value)
        if value and value not in keywords:
            keywords.append(value)
        if len(keywords) == 6:
            break
    if not summary:
        raise UnifiedGenerationError("simplified.summary is required")
    return {
        "summary": summary,
        "level": level,
        "keywords": keywords,
    }


def _normalize_characters(raw: Any) -> list[dict[str, str]]:
    if not isinstance(raw, list):
        raise UnifiedGenerationError("characters must be array")
    normalized: list[dict[str, str]] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        char_id = str(item.get("id", "")).strip()
        name = str(item.get("name", "")).strip()
        description = str(item.get("description", "")).strip()
        if not name or not description:
            continue
        if not char_id:
            slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
            char_id = f"char_{slug or idx + 1}"
        normalized.append(
            {
                "id": char_id,
                "name": name,
                "description": description,
            }
        )
        if len(normalized) == 3:
            break
    if len(normalized) < 2:
        raise UnifiedGenerationError(
            "characters must include at least 2 entries"
        )
    return normalized


def _normalize_bbox(raw: Any) -> list[int]:
    if not isinstance(raw, list) or len(raw) != 4:
        return [30, 30, 300, 90]
    vals: list[int] = []
    for value in raw:
        try:
            vals.append(int(value))
        except (TypeError, ValueError):
            vals.append(0)
    x = max(0, min(511, vals[0]))
    y = max(0, min(511, vals[1]))
    w = max(30, min(512 - x, vals[2]))
    h = max(30, min(512 - y, vals[3]))
    return [x, y, w, h]


def _normalize_panels(raw: Any, panel_count: int) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        raise UnifiedGenerationError("panels must be array")
    normalized: list[dict[str, Any]] = []
    target_count = max(3, min(5, panel_count))
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        panel_id = str(item.get("panel_id", f"panel_{idx + 1}")).strip()
        frame_index = idx + 1
        scene = str(item.get("scene_description", "")).strip()
        if not scene:
            scene = f"Comic panel {frame_index} scene"

        dialogue_raw = item.get("dialogue", [])
        dialogue: list[dict[str, str]] = []
        if isinstance(dialogue_raw, list):
            for line in dialogue_raw[:3]:
                if not isinstance(line, dict):
                    continue
                character_id = str(line.get("character_id", "")).strip()
                text = str(line.get("text", "")).strip()
                if character_id and text:
                    dialogue.append(
                        {"character_id": character_id, "text": text}
                    )
        if len(dialogue) < 1:
            raise UnifiedGenerationError(
                "each panel needs at least 1 dialogue line"
            )

        bubbles_raw = item.get("bubbles", [])
        bubbles: list[dict[str, list[int]]] = []
        if isinstance(bubbles_raw, list):
            for bubble in bubbles_raw[: len(dialogue)]:
                if not isinstance(bubble, dict):
                    continue
                bubbles.append(
                    {"bbox_px": _normalize_bbox(bubble.get("bbox_px"))}
                )
        while len(bubbles) < len(dialogue):
            y = 30 + (len(bubbles) * 110)
            bubbles.append({"bbox_px": [30, y, 300, 90]})

        render_raw = item.get("render", {})
        if not isinstance(render_raw, dict):
            render_raw = {}
        render = {
            "image_path": str(
                render_raw.get("image_path", f"assets/panel_{frame_index}.png")
            ),
            "overlay_applied": bool(render_raw.get("overlay_applied", False)),
        }

        normalized.append(
            {
                "panel_id": panel_id,
                "frame_index": frame_index,
                "scene_description": scene,
                "dialogue": dialogue,
                "bubbles": bubbles,
                "render": render,
            }
        )
        if len(normalized) == target_count:
            break

    if len(normalized) < 3:
        raise UnifiedGenerationError("panels must include 3-5 entries")
    return normalized


def _normalize_exercises(
    raw: Any,
    panels: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        raise UnifiedGenerationError("exercises must be array")
    panel_ids = [p["panel_id"] for p in panels]
    target = len(panel_ids)
    normalized: list[dict[str, Any]] = []
    by_panel: dict[str, dict[str, Any]] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        panel_id = str(item.get("panel_id", "")).strip()
        if panel_id:
            by_panel[panel_id] = item

    for panel_id in panel_ids:
        item = by_panel.get(panel_id)
        if not isinstance(item, dict):
            raise UnifiedGenerationError("missing exercise for panel")
        prompt = str(item.get("prompt", "")).strip()
        blanks = item.get("blanks", [])
        answer_key = item.get("answer_key", [])
        if not isinstance(blanks, list) or not blanks:
            blanks = ["____"]
        if not isinstance(answer_key, list) or not answer_key:
            raise UnifiedGenerationError("exercise answer_key missing")
        feedback_rules = item.get("feedback_rules", {})
        if not isinstance(feedback_rules, dict):
            feedback_rules = {}
        normalized.append(
            {
                "exercise_id": str(
                    item.get("exercise_id", f"ex_{panel_id}")
                ).strip()
                or f"ex_{panel_id}",
                "panel_id": panel_id,
                "prompt": prompt or "____",
                "blanks": [str(x) for x in blanks],
                "answer_key": [str(x) for x in answer_key],
                "feedback_rules": {
                    "case_sensitive": bool(
                        feedback_rules.get("case_sensitive", False)
                    ),
                    "allow_trim_spaces": bool(
                        feedback_rules.get("allow_trim_spaces", True)
                    ),
                },
            }
        )
        if len(normalized) == target:
            break

    if len(normalized) != target:
        raise UnifiedGenerationError("exercise count must match panel count")
    return normalized


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

# def generate_characters_from_text(
#     fulltext: str,
#     language: str,
#     model_repo_id: str,
# ) -> str:
#     prompt = (
#         "Create 2 or 3 comic characters from the article below. "
#         "First summarize the article in one sentence mentally, "
#         "then output only "
#         "a JSON array with objects using keys id, name, description. "
#         "Use language code "
#         f"'{language}'. Keep text age-appropriate and simple. "
#         "Do not include markdown. Article:\n"
#         f"{fulltext}"
#     )

#     try:
#         generated = _generate_with_pipeline(
#             prompt,
#             model_repo_id=model_repo_id,
#             max_new_tokens=220,
#             do_sample=False,
#             temperature=0.1,
#         )
#         logger.info(
#             "Generated characters text: %s",
#             generated,
#         )
#     except Exception as exc:
#         raise TextGenerationError("text generation failed") from exc
#     return generated


def generate_session_fields_from_article(
    language: str,
    style_id: str,
    article: dict[str, Any],
    panel_count: int,
    model_repo_id: str,
) -> dict[str, Any]:
    fulltext = str(article.get("fulltext", "")).strip()
    title = str(article.get("title", "")).strip()
    if not fulltext:
        raise UnifiedGenerationError("article.fulltext is required")

    target_count = max(3, min(5, panel_count))
    prompt = (
        f"{UNIFIED_SESSION_PROMPT}\n"
        f"language={language}\n"
        f"style_id={style_id}\n"
        f"panel_count={target_count}\n"
        f"article_title={title}\n"
        f"article_fulltext={fulltext}\n"
    )
    try:
        raw_text = _generate_with_pipeline(
            prompt,
            model_repo_id=model_repo_id,
            max_new_tokens=1600,
            do_sample=False,
            temperature=0.1,
        )
    except Exception as exc:
        raise UnifiedGenerationError("model generation call failed") from exc

    if not isinstance(raw_text, str) or not raw_text.strip():
        raise UnifiedGenerationError("invalid model output text")

    logger.info(
        "Generated unified session text: %s",
        raw_text,
    )
    payload = _extract_json_object(raw_text)
    logger.info("Extracted JSON payload for unified session: %s", payload)

    try:
        simplified = _normalize_simplified(payload.get("simplified"))
    except UnifiedGenerationError as exc:
        raise UnifiedGenerationError("simplified normalization failed") from exc
    logger.info("Normalized simplified: %s", simplified)

    try:
        characters = _normalize_characters(payload.get("characters"))
    except UnifiedGenerationError as exc:
        raise UnifiedGenerationError("character normalization failed") from exc
    logger.info("Normalized characters: %s", characters)

    try:
        panels = _normalize_panels(payload.get("panels"), target_count)
    except UnifiedGenerationError as exc:
        raise UnifiedGenerationError("panel normalization failed") from exc
    logger.info("Normalized panels: %s", panels)

    try:
        exercises = _normalize_exercises(payload.get("exercises"), panels)
    except UnifiedGenerationError as exc:
        raise UnifiedGenerationError("exercise normalization failed") from exc
    logger.info("Normalized exercises: %s", exercises)

    return {
        "simplified": simplified,
        "characters": characters,
        "panels": panels,
        "exercises": exercises,
    }
