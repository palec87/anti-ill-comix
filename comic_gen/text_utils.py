from __future__ import annotations

import re
from typing import Any

from .errors import UnifiedGenerationError


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
