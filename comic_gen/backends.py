from __future__ import annotations

import re
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

LANGUAGE_FEEDS = {
    "en": "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
    "es": (
        "https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/section/"
        "inenglish/portada"
    ),
    "fr": "https://www.lemonde.fr/en/international/rss_full.xml",
    "de": "https://www.spiegel.de/international/index.rss",
}

DETERMINISTIC_ARTICLES = {
    "en": {
        "publisher": "Example Daily",
        "source_link": "https://example.org/en/health-community-garden",
        "published_at": datetime.now(timezone.utc).isoformat(),
        "title": "Community Gardens Help Adults Learn Together",
        "fulltext": (
            "A city neighborhood opened a new community garden program. "
            "Adults meet each week to grow vegetables, read short guides, "
            "and write notes "
            "about what they learn. Organizers say the activity improves "
            "reading confidence "
            "because people use clear instructions in a real situation."
        ),
    },
    "es": {
        "publisher": "Ejemplo Noticias",
        "source_link": "https://example.org/es/biblioteca-barrio",
        "published_at": datetime.now(timezone.utc).isoformat(),
        "title": "Una biblioteca de barrio impulsa la lectura adulta",
        "fulltext": (
            "Una biblioteca local inicio talleres semanales para "
            "personas adultas. "
            "Los participantes leen noticias breves, conversan en grupo "
            "y escriben resumenes "
            "con ayuda de un tutor. El programa mejora la comprension "
            "y la confianza para escribir."
        ),
    },
    "fr": {
        "publisher": "Nouvelles Exemple",
        "source_link": "https://example.org/fr/atelier-langue",
        "published_at": datetime.now(timezone.utc).isoformat(),
        "title": "Un atelier local renforce la lecture des adultes",
        "fulltext": (
            "Un centre communautaire propose des ateliers de lecture simple. "
            "Les participants lisent des articles courts et ecrivent "
            "de petites syntheses. "
            "Les formateurs observent une meilleure comprehension "
            "semaine apres semaine."
        ),
    },
    "de": {
        "publisher": "Beispiel Nachrichten",
        "source_link": "https://example.org/de/lernkreis",
        "published_at": datetime.now(timezone.utc).isoformat(),
        "title": "Lernkreis verbessert Lesen im Alltag",
        "fulltext": (
            "Ein lokaler Lernkreis trifft sich jede Woche fuer kurze "
            "Nachrichten und Schreibuebungen. "
            "Die Teilnehmenden lesen gemeinsam und notieren wichtige "
            "Begriffe. "
            "Lehrkraefte berichten ueber bessere Lesesicherheit im Alltag."
        ),
    },
}


def _clean_text(value: str) -> str:
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
        title = _clean_text(
            item.findtext("{http://www.w3.org/2005/Atom}title", "")
        )
        summary = _clean_text(
            item.findtext("{http://www.w3.org/2005/Atom}summary", "")
        )
        link_elem = item.find("{http://www.w3.org/2005/Atom}link")
        link = ""
        if link_elem is not None:
            link = link_elem.attrib.get("href", "")
        published = _clean_text(
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

    title = _clean_text(item.findtext("title", ""))
    description = _clean_text(item.findtext("description", ""))
    link = _clean_text(item.findtext("link", ""))
    pub_date = _clean_text(item.findtext("pubDate", ""))
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
) -> dict[str, str]:
    normalized = language.lower()
    if not use_live_feed:
        return DETERMINISTIC_ARTICLES.get(
            normalized,
            DETERMINISTIC_ARTICLES["en"],
        )

    feed_url = LANGUAGE_FEEDS.get(normalized)
    if not feed_url:
        return DETERMINISTIC_ARTICLES.get(
            normalized,
            DETERMINISTIC_ARTICLES["en"],
        )

    try:
        with urllib.request.urlopen(feed_url, timeout=10) as response:
            payload = response.read().decode("utf-8", errors="ignore")
        parsed = _parse_rss(payload)
        if parsed and parsed.get("source_link") and parsed.get("fulltext"):
            return parsed
    except (urllib.error.URLError, TimeoutError, ValueError):
        pass

    return DETERMINISTIC_ARTICLES.get(
        normalized,
        DETERMINISTIC_ARTICLES["en"],
    )
