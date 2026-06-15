from __future__ import annotations

import contextlib
import io
import logging
import os
import sys
from typing import Any

from .trace import add_trace

logger = logging.getLogger(__name__)

DEFAULT_TRANSLATION_MODEL_ID = "facebook/nllb-200-distilled-600M"
NLLB_LANGUAGE_CODES = {
    "en": "eng_Latn",
    "pt": "por_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
}
PLACEHOLDER_TOKEN = "__BLANK_PLACEHOLDER__"
_PIPELINES: dict[tuple[str, str, str], Any] = {}


def _reconfigure_stream(stream: Any) -> None:
    """Best-effort stream reconfiguration for Windows code pages."""
    reconfigure = getattr(stream, "reconfigure", None)
    if not callable(reconfigure):
        return
    try:
        reconfigure(encoding="utf-8", errors="backslashreplace")
    except (OSError, ValueError):
        return


def _ensure_utf8_stdio() -> None:
    """Keep model logging from failing on non-ASCII translation text."""
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for stream in (sys.stdout, sys.stderr):
        _reconfigure_stream(stream)
    for handler in logging.getLogger().handlers + logger.handlers:
        _reconfigure_stream(getattr(handler, "stream", None))


@contextlib.contextmanager
def _quiet_model_output():
    """Suppress model chatter that can fail on non-UTF-8 consoles."""
    _ensure_utf8_stdio()
    with contextlib.redirect_stdout(io.StringIO()):
        with contextlib.redirect_stderr(io.StringIO()):
            yield


def _get_translation_pipeline(
    model_id: str,
    source_language: str,
    target_language: str,
) -> Any:
    """Load and cache a translation pipeline.

    Args:
        model_id: Hugging Face model id.
        source_language: NLLB source language code.
        target_language: NLLB target language code.

    Returns:
        A callable Transformers translation pipeline.
    """
    cache_key = (model_id, source_language, target_language)
    if cache_key in _PIPELINES:
        return _PIPELINES[cache_key]

    _ensure_utf8_stdio()
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    with _quiet_model_output():
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            src_lang=source_language,
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(target_language)

    def translator(text: str) -> list[dict[str, str]]:
        tokenizer.src_lang = source_language
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        generated = model.generate(
            **encoded,
            forced_bos_token_id=forced_bos_token_id,
            max_new_tokens=256,
        )
        translated = tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
        )[0]
        return [{"translation_text": translated}]

    _PIPELINES[cache_key] = translator
    return translator


def _translation_result_text(result: Any) -> str:
    """Extract text from a Transformers translation result."""
    if isinstance(result, list) and result:
        first = result[0]
        if isinstance(first, dict):
            return str(first.get("translation_text", "")).strip()
    if isinstance(result, dict):
        return str(result.get("translation_text", "")).strip()
    return str(result or "").strip()


def _run_translator(translator: Any, text: str) -> Any:
    """Call a translator while suppressing unsafe console output."""
    with _quiet_model_output():
        return translator(text)


def translate_text(
    text: Any,
    target_language: str,
    *,
    source_language: str = "en",
    model_id: str = DEFAULT_TRANSLATION_MODEL_ID,
    preserve_blanks: bool = False,
) -> str:
    """Translate text to the target language.

    Args:
        text: Source text.
        target_language: App language code.
        source_language: App source language code.
        model_id: Hugging Face translation model id.
        preserve_blanks: Whether to protect blank placeholders.

    Returns:
        Translated text, or the source text for English/no-op cases.
    """
    logger.info("Translating text to %s", target_language)
    value = str(text or "").strip()
    if not value:
        return value

    if source_language not in NLLB_LANGUAGE_CODES:
        raise ValueError(f"Unsupported source language: {source_language}")
    if target_language not in NLLB_LANGUAGE_CODES:
        raise ValueError(f"Unsupported target language: {target_language}")

    source_code = NLLB_LANGUAGE_CODES[source_language]
    target_code = NLLB_LANGUAGE_CODES[target_language]
    if source_code == target_code:
        return value

    protected = value
    if preserve_blanks:
        protected = protected.replace("_______", PLACEHOLDER_TOKEN)
        protected = protected.replace("____", PLACEHOLDER_TOKEN)

    translator = _get_translation_pipeline(model_id, source_code, target_code)
    _ensure_utf8_stdio()
    translated = _translation_result_text(_run_translator(translator, protected))
    translated = translated or value
    if preserve_blanks:
        translated = translated.replace(PLACEHOLDER_TOKEN, "____")
        if "____" not in translated and "____" in value:
            translated = f"{translated} ____"
    return translated


def translate_session_content(
    document: dict[str, Any],
    target_language: str,
    *,
    source_language: str = "en",
    model_id: str = DEFAULT_TRANSLATION_MODEL_ID,
) -> bool:
    """Translate learner-facing session fields in place.

    Args:
        document: Session document to mutate.
        target_language: App target language code.
        source_language: App source language code.
        model_id: Hugging Face translation model id.

    Returns:
        True when translation was applied, False for no-op English.
    """
    if NLLB_LANGUAGE_CODES.get(source_language) == NLLB_LANGUAGE_CODES.get(
        target_language
    ):
        return False

    add_trace(
        document,
        "translation",
        "start",
        f"Translating learner content to {target_language}",
    )

    simplified = document.get("simplified", {})
    if isinstance(simplified, dict):
        simplified["summary"] = translate_text(
            simplified.get("summary", ""),
            target_language,
            source_language=source_language,
            model_id=model_id,
        )
        keywords = simplified.get("keywords", [])
        if isinstance(keywords, list):
            simplified["keywords"] = [
                translate_text(
                    keyword,
                    target_language,
                    source_language=source_language,
                    model_id=model_id,
                )
                for keyword in keywords
            ]

    for character in document.get("characters", []):
        if not isinstance(character, dict):
            continue
        character["description"] = translate_text(
            character.get("description", ""),
            target_language,
            source_language=source_language,
            model_id=model_id,
        )

    for panel in document.get("panels", []):
        if not isinstance(panel, dict):
            continue
        panel["scene_description"] = translate_text(
            panel.get("scene_description", ""),
            target_language,
            source_language=source_language,
            model_id=model_id,
        )
        for line in panel.get("dialogue", []):
            if not isinstance(line, dict):
                continue
            line["text"] = translate_text(
                line.get("text", ""),
                target_language,
                source_language=source_language,
                model_id=model_id,
            )

    for item in document.get("exercises", []):
        if not isinstance(item, dict):
            continue
        item["prompt"] = translate_text(
            item.get("prompt", ""),
            target_language,
            source_language=source_language,
            model_id=model_id,
            preserve_blanks=True,
        )
        answer_key = item.get("answer_key", [])
        if isinstance(answer_key, list):
            item["answer_key"] = [
                translate_text(
                    answer,
                    target_language,
                    source_language=source_language,
                    model_id=model_id,
                )
                for answer in answer_key
            ]

    add_trace(
        document,
        "translation",
        "ok",
        f"Translated learner content to {target_language}",
    )
    return True
