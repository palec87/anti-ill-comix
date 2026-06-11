from __future__ import annotations

import json
import urllib.error
import urllib.request
import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any

from . import exercise
from .text_utils import load_json_article, clean_text

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

LANGUAGE_FEEDS = {
    "en": "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
    "es": (
        "https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/section/"
        "inenglish/portada"
    ),
    "fr": "https://www.lemonde.fr/en/international/rss_full.xml",
    "de": "https://www.spiegel.de/international/index.rss",
}


def deterministic_pipeline(
    document: dict[str, Any],
) -> None:
    """Run deterministic end-to-end generation without model inference."""
    payload = load_json_article(document.get("language", "en"))
    document["characters"] = payload.get("characters", [])
    document["panels"] = payload.get("panels", [])
    exercise.generate_exercises(document)


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


def fetch_article(
    language: str,
    use_live_feed: bool = False,
    model_generation: bool = False,
) -> dict[str, str]:
    normalized = language.lower()
    if use_live_feed:
        logger.info(f"Fetching live article for language '{normalized}'")

        feed_url = LANGUAGE_FEEDS.get(normalized)
        if not feed_url:
            return load_json_article(language)

        try:
            with urllib.request.urlopen(feed_url, timeout=10) as response:
                payload = response.read().decode("utf-8", errors="ignore")
            parsed = _parse_rss(payload)
            if parsed and parsed.get("source_link") and parsed.get("fulltext"):
                return parsed
        except (urllib.error.URLError, TimeoutError, ValueError):
            return load_json_article(language)

    else:
        payload = load_json_article(language)

        if model_generation:
            try:
                parsed = {
                    "publisher": payload.get("source").get(
                        "publisher"
                    ),
                    "source_link": payload.get("source").get("link"),
                    "published_at": payload.get("source").get(
                        "published_at"
                    ),
                    "title": payload.get("article").get("title"),
                    "fulltext": payload.get("article").get("fulltext"),
                }
                if parsed and parsed.get("source_link") and parsed.get(
                    "fulltext"
                ):
                    return parsed
            except (OSError, json.JSONDecodeError):
                logger.warning(
                    f"Error reading example file for '{language}',"
                    f" falling back to deterministic article."
                )
            except Exception as e:
                logger.warning(
                    f"Unexpected error loading example file for '{language}':"
                    f" {e}. Falling back to deterministic article."
                )
        return payload

    return load_json_article(language)
