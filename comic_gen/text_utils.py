from __future__ import annotations

import re
import json
import logging
from pathlib import Path
from typing import Any
from datetime import datetime, timezone
import xml.etree.ElementTree as ET
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


def _parse_rss(xml_text: str) -> dict[str, str] | None:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return None

    # Try RSS item first.
    item = root.find("./channel/item")
    if item is None:
        # Try Atom entry.
        item = root.find("{http://www.w3.org/2005/Atom}entry")
        if item is None:
            return None
        title = clean_text(
            item.findtext("{http://www.w3.org/2005/Atom}title", "")
        )
        summary = clean_text(
            item.findtext("{http://www.w3.org/2005/Atom}summary", "")
        )
        link_elem = item.find("{http://www.w3.org/2005/Atom}link")
        link = ""
        if link_elem is not None:
            link = link_elem.attrib.get("href", "")
        published = clean_text(
            item.findtext("{http://www.w3.org/2005/Atom}updated", "")
        )
        return {
            "publisher": "Atom Feed",
            "source_link": link,
            "published_at": (
                published or datetime.now(timezone.utc).isoformat()
            ),
            "title": title or "Untitled",
            "fulltext": summary or title,
        }

    title = clean_text(item.findtext("title", ""))
    description = clean_text(item.findtext("description", ""))
    link = clean_text(item.findtext("link", ""))
    pub_date = clean_text(item.findtext("pubDate", ""))
    return {
        "publisher": "RSS Feed",
        "source_link": link,
        "published_at": pub_date or datetime.now(timezone.utc).isoformat(),
        "title": title or "Untitled",
        "fulltext": description or title,
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


def split_sentences(text: str) -> list[str]:
    """Split text into sentence-like chunks."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def extract_json_object(raw_text: str) -> dict[str, Any]:
    def _balanced_json_objects(text: str) -> list[str]:
        objects: list[str] = []
        start_idx = -1
        depth = 0
        in_string = False
        escaped = False

        for idx, char in enumerate(text):
            if in_string:
                if escaped:
                    escaped = False
                    continue
                if char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char == "{":
                if depth == 0:
                    start_idx = idx
                depth += 1
            elif char == "}" and depth > 0:
                depth -= 1
                if depth == 0 and start_idx != -1:
                    objects.append(text[start_idx: idx + 1])
                    start_idx = -1

        return objects

    candidates: list[str] = []
    fenced_matches = re.findall(r"```(?:json)?\\s*([\\s\\S]*?)```", raw_text)
    candidates.extend(
        match.strip()
        for match in fenced_matches
        if match.strip()
    )
    candidates.append(raw_text)

    decoder = json.JSONDecoder()
    parse_errors: list[str] = []

    for candidate in candidates:
        for blob in _balanced_json_objects(candidate):
            try:
                parsed, end_idx = decoder.raw_decode(blob)
                if blob[end_idx:].strip():
                    parse_errors.append("trailing_non_json_content")
                    continue
                if not isinstance(parsed, dict):
                    parse_errors.append("root_not_object")
                    continue
                return parsed
            except json.JSONDecodeError as exc:
                context_start = max(0, exc.pos - 40)
                context_end = min(len(blob), exc.pos + 40)
                context = blob[context_start:context_end]
                parse_errors.append(
                    f"{exc.msg} at line {exc.lineno}, col {exc.colno}, "
                    f"pos {exc.pos}; context={context!r}"
                )

    if "{" not in raw_text or "}" not in raw_text:
        raise UnifiedGenerationError("model output missing JSON object")

    logger.info("Failed JSON parse details: %s", " | ".join(parse_errors))
    raise UnifiedGenerationError("model output is not valid JSON")