from __future__ import annotations

from outlines import Generator, from_transformers
import logging
import os
from typing import Any

from .text_utils import (
    _normalize_simplified,
    _normalize_characters,
    _normalize_panels,
    _normalize_exercises,
    extract_json_object,
    ComicResponse,
)
from .errors import (
    TextGenerationError,
    UnifiedGenerationError,
)
from .prompts import UNIFIED_SESSION_PROMPT

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    force=True,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

IS_LOCAL = os.environ.get("LOCAL_DEV", "False") == "True"
_GENERATOR: Any | None = None
_GENERATOR_MODEL_ID = ""
_INFERENCE_CLIENT: Any | None = None


def _detect_normalization_repairs(payload: dict[str, Any]) -> list[str]:
    """Describe shape repairs that normalization will apply."""
    repairs: list[str] = []
    panels = payload.get("panels", [])
    if isinstance(panels, list):
        for panel in panels:
            if not isinstance(panel, dict):
                continue
            dialogue = panel.get("dialogue", [])
            bubbles = panel.get("bubbles", [])
            if (
                isinstance(dialogue, list)
                and dialogue
                and isinstance(bubbles, list)
                and len(bubbles) < len(dialogue)
            ):
                repairs.append("panel bubbles")
                break

    exercises = payload.get("exercises", [])
    if isinstance(exercises, list):
        for item in exercises:
            if not isinstance(item, dict):
                continue
            prompt = str(item.get("prompt", ""))
            blanks = item.get("blanks", [])
            if (
                ("____" in prompt or "_______" in prompt)
                and isinstance(blanks, list)
                and any(str(blank).strip("_") for blank in blanks)
            ):
                repairs.append("exercise blanks")
                break
    return repairs


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
            "text generation failed (serverless_api), "
            f"{model_repo_id}, {detail}"
        ) from exc

    if not isinstance(generated, str) or not generated.strip():
        raise TextGenerationError("invalid text generation output")
    return generated


def conditional_gpu_decorator(func):
    if not IS_LOCAL:
        import spaces
        return spaces.GPU(func)  # Wraps with ZeroGPU in production.
    return func  # Passes through for local development.


@conditional_gpu_decorator
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
    import transformers
    import torch
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_repo_id,
        dtype=torch.float16,
        device_map="cuda"
    )
    hf_tokenizer = transformers.AutoTokenizer.from_pretrained(model_repo_id)

    model = from_transformers(hf_model, hf_tokenizer)
    structured_generator = Generator(model, ComicResponse)
    generated = structured_generator(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
    )
    validated = ComicResponse.model_validate_json(generated)
    json_string = validated.model_dump_json()
    logger.info("Generated unified session text (structured): %s", json_string)
    return json_string


def generate_text_content_from_article(
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
        UNIFIED_SESSION_PROMPT
        .replace("{language}", language)
        .replace("{style_id}", style_id)
        .replace("{panel_count}", str(target_count))
        .replace(
            "{user_input_source_material}",
            f"article_title={title}\narticle_fulltext={fulltext}",
        )
    )
    try:
        raw_text = _generate_with_pipeline(
            prompt,
            model_repo_id=model_repo_id,
            max_new_tokens=2048,
            do_sample=False,
            temperature=0.1,
        )
    except Exception as exc:
        raise UnifiedGenerationError("Text generation call failed") from exc

    if not isinstance(raw_text, str) or not raw_text.strip():
        raise UnifiedGenerationError(
            "invalid model output, Not a non-empty string"
        )

    payload = extract_json_object(raw_text)
    logger.info("Extracted JSON payload for unified session: %s", payload)
    repairs = _detect_normalization_repairs(payload)

    try:
        simplified = _normalize_simplified(payload.get("simplified"))
    except UnifiedGenerationError as exc:
        raise UnifiedGenerationError(
            "simplified normalization failed"
        ) from exc
    # logger.info("Normalized simplified: %s", simplified)

    try:
        characters = _normalize_characters(payload.get("characters"))
    except UnifiedGenerationError as exc:
        raise UnifiedGenerationError("character normalization failed") from exc
    # logger.info("Normalized characters: %s", characters)

    try:
        panels = _normalize_panels(payload.get("panels"), target_count)
    except UnifiedGenerationError as exc:
        raise UnifiedGenerationError("panel normalization failed") from exc
    # logger.info("Normalized panels: %s", panels)

    try:
        exercises = _normalize_exercises(payload.get("exercises"), panels)
    except UnifiedGenerationError as exc:
        raise UnifiedGenerationError("exercise normalization failed") from exc
    # logger.info("Normalized exercises: %s", exercises)

    return {
        "simplified": simplified,
        "characters": characters,
        "panels": panels,
        "exercises": exercises,
        "_normalization_repairs": repairs,
    }
