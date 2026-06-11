from __future__ import annotations

import json
import urllib.error
import urllib.request
import logging
from typing import Any
import html
from urllib.parse import quote

from . import exercise
from .text_utils import load_json_article, _parse_rss

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


def fallback_image_src(panel: dict[str, Any]) -> str:
    frame_index = panel.get("frame_index", "?")
    scene = html.escape(panel.get("scene_description", "Comic panel"))
    scene = scene[:110]
    svg_lines = [
        (
            '<svg xmlns="http://www.w3.org/2000/svg" '
            'width="512" height="512" viewBox="0 0 512 512">'
        ),
        '  <defs>',
        '    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">',
        '      <stop offset="0%" stop-color="#fff7e8"/>',
        '      <stop offset="100%" stop-color="#eef4ff"/>',
        '    </linearGradient>',
        '  </defs>',
        '  <rect width="512" height="512" rx="28" fill="url(#bg)"/>',
        (
            '  <rect x="24" y="24" width="464" height="464" '
            'rx="24" fill="none" stroke="#c5cfdb" stroke-width="3"/>'
        ),
        (
            '  <text x="40" y="72" font-size="22" '
            'font-family="Arial, sans-serif" fill="#415166">'
            f'Panel {frame_index}</text>'
        ),
        (
            '  <text x="40" y="120" font-size="16" '
            'font-family="Arial, sans-serif" fill="#5f6f82">'
            f'{scene}</text>'
        ),
        '  <circle cx="150" cy="270" r="54" fill="#d8e4f2"/>',
        '  <circle cx="344" cy="270" r="54" fill="#f2dec7"/>',
        (
            '  <rect x="106" y="334" width="88" height="108" '
            'rx="22" fill="#d8e4f2"/>'
        ),
        (
            '  <rect x="300" y="334" width="88" height="108" '
            'rx="22" fill="#f2dec7"/>'
        ),
        (
            '  <path d="M88 180h160a18 18 0 0 1 18 18v48a18 18 '
            '0 0 1-18 18h-90l-34 24 10-24H88a18 18 0 0 1-18-18v-48'
            'a18 18 0 0 1 18-18z" fill="#ffffff" stroke="#9ba6b2" '
            'stroke-width="3"/>'
        ),
        (
            '  <path d="M264 120h160a18 18 0 0 1 18 18v48a18 18 '
            '0 0 1-18 18h-84l-34 24 10-24h-52a18 18 0 0 1-18-18v-48'
            'a18 18 0 0 1 18-18z" fill="#ffffff" stroke="#9ba6b2" '
            'stroke-width="3"/>'
        ),
        '</svg>',
    ]
    svg = "\n".join(svg_lines)
    return "data:image/svg+xml;utf8," + quote(svg)


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
