from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import new_session_id, utc_now_iso, validate_session_document
from .trace import add_trace

SCHEMA_VERSION = "1.0.0"


def build_base_session(
    language: str,
    style_id: str,
    source_payload: dict[str, str],
) -> dict[str, Any]:
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "session_id": new_session_id(),
        "language": language,
        "style_id": style_id,
        "source": {
            "link": source_payload.get("source_link", ""),
            "publisher": source_payload.get("publisher", "Unknown Publisher"),
            "published_at": source_payload.get("published_at", utc_now_iso()),
        },
        "article": {
            "title": source_payload.get("title", "Untitled article"),
            "fulltext": source_payload.get("fulltext", ""),
        },
        "simplified": {"summary": "", "level": "A2", "keywords": []},
        "characters": [],
        "panels": [],
        "exercises": [],
        "ui": {
            "language_label": language,
            "style_label": style_id,
            "selector_state": {"language": language, "style": style_id},
        },
        "trace": [],
    }
    add_trace(document, "session", "ok", "Session initialized")
    return document


def validate_or_raise(document: dict[str, Any]) -> dict[str, Any]:
    validate_session_document(document)
    return document


def save_session(document: dict[str, Any], output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    target = output_path / f"{document['session_id']}.json"
    target.write_text(
        json.dumps(document, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return target


def load_session(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_or_raise(document)
    return document
