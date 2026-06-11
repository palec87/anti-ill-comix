from __future__ import annotations

import re
import json
import logging
from pathlib import Path
from typing import Any

from .errors import UnifiedGenerationError


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def load_json_article(language: str) -> dict[str, str]:
    repo_root = Path(__file__).resolve().parents[1]
    example_path = repo_root / "examples" / f"{language}_demo.json"
    if not example_path.exists():
        raise FileNotFoundError(f"Example file not found for language '{language}'")
    try:
        payload = json.loads(example_path.read_text(encoding="utf-8"))
        return payload
    except (OSError, json.JSONDecodeError) as e:
        logger.error(f"Error loading JSON article for '{language}': {e}")
        raise


def clean_text(value: str) -> str:
    value = re.sub(r"\s+", " ", value or "").strip()
    return value


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


def split_sentences(text: str) -> list[str]:
    """Split text into sentence-like chunks."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]
