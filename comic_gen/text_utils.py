from __future__ import annotations

import re
import json
import logging
from pathlib import Path
from typing import Any
from datetime import datetime, timezone
import xml.etree.ElementTree as ET
from .errors import UnifiedGenerationError, ModelPipelineError

from pydantic import BaseModel, Field
from typing import List


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# 1. Define your exact structure using Pydantic
class Character(BaseModel):
    id: str
    name: str
    description: str


class Bubble(BaseModel):
    bbox_px: List[int] = Field(..., description="[x, y, w, h] elements")


class Dialogue(BaseModel):
    character_id: str
    text: str


class Render(BaseModel):
    image_path: str
    overlay_applied: bool


class Panel(BaseModel):
    panel_id: str
    frame_index: int
    scene_description: str
    dialogue: List[Dialogue]
    bubbles: List[Bubble]
    render: Render


class ExerciseRules(BaseModel):
    case_sensitive: bool
    allow_trim_spaces: bool


class Exercise(BaseModel):
    exercise_id: str
    panel_id: str
    prompt: str
    blanks: List[str]
    answer_key: List[str]
    feedback_rules: ExerciseRules


class SimplifiedData(BaseModel):
    summary: str
    level: str
    keywords: List[str]


# The master schema matching your prompt
class ComicResponse(BaseModel):
    simplified: SimplifiedData
    characters: List[Character]
    panels: List[Panel]
    exercises: List[Exercise]


UI_TRANSLATIONS = {
    "en": {
        "app_subtitle": "Turn international news into simple comic practice for adult reading and writing.",
        "language": "Language",
        "art_style": "Art style",
        "reading_level": "Reading level",
        "panel_count": "Panel count",
        "live_feed": "Use live RSS article",
        "advanced_options": "Advanced Options",
        "serverless": "Use HF serverless API for text + image generation",
        "debug": "Debug panel rendering",
        "negative_prompt": "Negative prompt",
        "negative_prompt_placeholder": "Optional quality or style exclusions",
        "seed": "Seed",
        "randomize_seed": "Randomize seed",
        "width": "Width",
        "height": "Height",
        "guidance_scale": "Guidance scale",
        "inference_steps": "Inference steps",
        "generate": "Generate Comic Strip",
        "source": "Source",
        "publisher": "Publisher",
        "title": "Title",
        "link": "Link",
        "published": "Published",
        "summary": "Simplified Summary",
        "keywords": "Keywords",
        "transcript": "Transcript",
        "panel": "Panel",
        "exercises_heading": "### Writing Exercises",
        "select_panel": "Select panel",
        "exercise": "Exercise",
        "generate_first": "Generate a strip first.",
        "unlock_exercises": "Generate a strip to unlock exercises.",
        "no_exercise": "No exercise available for this panel.",
        "answer_label": "Your answer",
        "answer_placeholder": "Type the missing word",
        "submit": "Submit Answer",
        "trace": "Trace / Debug",
    },
    "pt": {
        "app_subtitle": "Transforme noticias internacionais em pratica simples de leitura e escrita com quadrinhos.",
        "language": "Idioma",
        "art_style": "Estilo de arte",
        "reading_level": "Nivel de leitura",
        "panel_count": "Numero de quadros",
        "live_feed": "Usar noticia RSS ao vivo",
        "advanced_options": "Opcoes avancadas",
        "serverless": "Usar API serverless HF para texto + imagem",
        "debug": "Depurar quadros",
        "negative_prompt": "Prompt negativo",
        "negative_prompt_placeholder": "Exclusoes opcionais de qualidade ou estilo",
        "seed": "Semente",
        "randomize_seed": "Semente aleatoria",
        "width": "Largura",
        "height": "Altura",
        "guidance_scale": "Escala de orientacao",
        "inference_steps": "Passos de inferencia",
        "generate": "Gerar quadrinhos",
        "source": "Fonte",
        "publisher": "Publicador",
        "title": "Titulo",
        "link": "Link",
        "published": "Publicado",
        "summary": "Resumo simplificado",
        "keywords": "Palavras-chave",
        "transcript": "Transcricao",
        "panel": "Quadro",
        "exercises_heading": "### Exercicios de escrita",
        "select_panel": "Selecionar quadro",
        "exercise": "Exercicio",
        "generate_first": "Gere uma tira primeiro.",
        "unlock_exercises": "Gere uma tira para liberar os exercicios.",
        "no_exercise": "Nao ha exercicio para este quadro.",
        "answer_label": "Sua resposta",
        "answer_placeholder": "Digite a palavra que falta",
        "submit": "Enviar resposta",
        "trace": "Rastro / Debug",
    },
    "es": {
        "app_subtitle": "Convierte noticias internacionales en practica simple de lectura y escritura con comics.",
        "language": "Idioma",
        "art_style": "Estilo artistico",
        "reading_level": "Nivel de lectura",
        "panel_count": "Numero de vinetas",
        "live_feed": "Usar noticia RSS en vivo",
        "advanced_options": "Opciones avanzadas",
        "serverless": "Usar API serverless HF para texto + imagen",
        "debug": "Depurar vinetas",
        "negative_prompt": "Prompt negativo",
        "negative_prompt_placeholder": "Exclusiones opcionales de calidad o estilo",
        "seed": "Semilla",
        "randomize_seed": "Semilla aleatoria",
        "width": "Ancho",
        "height": "Alto",
        "guidance_scale": "Escala de guia",
        "inference_steps": "Pasos de inferencia",
        "generate": "Generar comic",
        "source": "Fuente",
        "publisher": "Editor",
        "title": "Titulo",
        "link": "Enlace",
        "published": "Publicado",
        "summary": "Resumen simplificado",
        "keywords": "Palabras clave",
        "transcript": "Transcripcion",
        "panel": "Vineta",
        "exercises_heading": "### Ejercicios de escritura",
        "select_panel": "Seleccionar vineta",
        "exercise": "Ejercicio",
        "generate_first": "Primero genera una tira.",
        "unlock_exercises": "Genera una tira para desbloquear ejercicios.",
        "no_exercise": "No hay ejercicio para esta vineta.",
        "answer_label": "Tu respuesta",
        "answer_placeholder": "Escribe la palabra que falta",
        "submit": "Enviar respuesta",
        "trace": "Traza / Debug",
    },
    "fr": {
        "app_subtitle": "Transformez des nouvelles internationales en pratique simple de lecture et d'ecriture avec BD.",
        "language": "Langue",
        "art_style": "Style artistique",
        "reading_level": "Niveau de lecture",
        "panel_count": "Nombre de cases",
        "live_feed": "Utiliser une nouvelle RSS en direct",
        "advanced_options": "Options avancees",
        "serverless": "Utiliser l'API HF serverless pour texte + image",
        "debug": "Debug des cases",
        "negative_prompt": "Prompt negatif",
        "negative_prompt_placeholder": "Exclusions optionnelles de qualite ou de style",
        "seed": "Graine",
        "randomize_seed": "Graine aleatoire",
        "width": "Largeur",
        "height": "Hauteur",
        "guidance_scale": "Echelle de guidage",
        "inference_steps": "Etapes d'inference",
        "generate": "Generer la BD",
        "source": "Source",
        "publisher": "Editeur",
        "title": "Titre",
        "link": "Lien",
        "published": "Publie",
        "summary": "Resume simplifie",
        "keywords": "Mots cles",
        "transcript": "Transcription",
        "panel": "Case",
        "exercises_heading": "### Exercices d'ecriture",
        "select_panel": "Choisir une case",
        "exercise": "Exercice",
        "generate_first": "Generez d'abord une BD.",
        "unlock_exercises": "Generez une BD pour ouvrir les exercices.",
        "no_exercise": "Aucun exercice pour cette case.",
        "answer_label": "Votre reponse",
        "answer_placeholder": "Tapez le mot manquant",
        "submit": "Envoyer la reponse",
        "trace": "Trace / Debug",
    },
    "de": {
        "app_subtitle": "Verwandle internationale Nachrichten in einfache Lese- und Schreibuebung mit Comics.",
        "language": "Sprache",
        "art_style": "Kunststil",
        "reading_level": "Leseniveau",
        "panel_count": "Anzahl der Bilder",
        "live_feed": "Live-RSS-Nachricht verwenden",
        "advanced_options": "Erweiterte Optionen",
        "serverless": "HF Serverless API fuer Text + Bild verwenden",
        "debug": "Bilder debuggen",
        "negative_prompt": "Negativer Prompt",
        "negative_prompt_placeholder": "Optionale Qualitaets- oder Stil-Ausschluesse",
        "seed": "Seed",
        "randomize_seed": "Zufaelliger Seed",
        "width": "Breite",
        "height": "Hoehe",
        "guidance_scale": "Fuehrungsstaerke",
        "inference_steps": "Inferenzschritte",
        "generate": "Comic erzeugen",
        "source": "Quelle",
        "publisher": "Herausgeber",
        "title": "Titel",
        "link": "Link",
        "published": "Veroeffentlicht",
        "summary": "Vereinfachte Zusammenfassung",
        "keywords": "Schluesselwoerter",
        "transcript": "Transkript",
        "panel": "Bild",
        "exercises_heading": "### Schreibuebungen",
        "select_panel": "Bild auswaehlen",
        "exercise": "Uebung",
        "generate_first": "Erzeuge zuerst einen Comic.",
        "unlock_exercises": "Erzeuge einen Comic, um Uebungen zu oeffnen.",
        "no_exercise": "Keine Uebung fuer dieses Bild.",
        "answer_label": "Deine Antwort",
        "answer_placeholder": "Gib das fehlende Wort ein",
        "submit": "Antwort senden",
        "trace": "Trace / Debug",
    },
}


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
    if len(normalized) < 1:
        raise UnifiedGenerationError(
            "characters must include at least 1 entry"
        )
    return normalized


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

        bubbles = _repair_bubbles(item.get("bubbles", []), len(dialogue))

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
        if "____" in prompt or "_______" in prompt:
            blanks = ["____" for _ in answer_key]
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


def _default_bboxes(count: int) -> list[list[int]]:
    """Return stable default bubble boxes for a 256px square panel."""
    layouts = {
        1: [[10, 10, 118, 30]],
        2: [[10, 10, 108, 30], [138, 10, 108, 30]],
        3: [[10, 10, 108, 28], [138, 10, 108, 28], [10, 206, 108, 28]],
    }
    return layouts.get(count, layouts[3])[:count]


def _box_overlap_ratio(first: list[int], second: list[int]) -> float:
    """Return overlap area compared to the smaller bubble area."""
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    x_overlap = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    y_overlap = max(0, min(ay + ah, by + bh) - max(ay, by))
    overlap = x_overlap * y_overlap
    smaller_area = max(1, min(aw * ah, bw * bh))
    return overlap / smaller_area


def _normalize_bbox(
    raw: Any,
    min_width: int = 82,
    min_height: int = 24,
) -> tuple[list[int], bool]:
    """Normalize one bubble box and report whether it needed repair."""
    if not isinstance(raw, list) or len(raw) != 4:
        return [10, 10, 108, 30], True
    vals: list[int] = []
    for value in raw:
        try:
            vals.append(int(value))
        except (TypeError, ValueError):
            vals.append(0)
    repaired = vals != raw
    x, y, w, h = vals
    if x < 0 or y < 0 or x > 255 or y > 255 or w < min_width or h < min_height:
        repaired = True
    w = max(min_width, min(256, w))
    h = max(min_height, min(256, h))
    x = max(0, min(256 - w, x))
    y = max(0, min(256 - h, y))
    return [x, y, w, h], repaired


def _repair_bubbles(
    raw: Any,
    dialogue_count: int,
) -> list[dict[str, list[int]]]:
    """Repair model bubble boxes so every dialogue line has readable space."""
    target = max(0, min(3, dialogue_count))
    if target == 0:
        return []
    defaults = _default_bboxes(target)
    if not isinstance(raw, list) or len(raw) < target:
        return [{"bbox_px": box} for box in defaults]

    boxes: list[list[int]] = []
    repaired = False
    for bubble in raw[:target]:
        if not isinstance(bubble, dict):
            repaired = True
            break
        box, box_repaired = _normalize_bbox(bubble.get("bbox_px"))
        boxes.append(box)
        repaired = repaired or box_repaired

    if len(boxes) != target:
        return [{"bbox_px": box} for box in defaults]

    for index, box in enumerate(boxes):
        for other in boxes[index + 1:]:
            if _box_overlap_ratio(box, other) > 0.2:
                repaired = True
                break
        if repaired:
            break

    if repaired:
        boxes = defaults
    return [{"bbox_px": box} for box in boxes]


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


def split_sentences(text: str) -> list[str]:
    """Split text into sentence-like chunks."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def extract_json_object(raw_text: str) -> dict[str, Any]:
    # If it's already a valid json object string, return it directly
    try:
        parsed = json.loads(raw_text.strip())
        return parsed
    except json.JSONDecodeError:
        pass
        
    # If it used '=' instead of wrapping everything in a single object:
    # We will capture each assignment block and wrap them inside a valid json container
    sections = re.findall(r'(\w+)\s*=\s*([\[{].*?)(?=\n\w+\s*=|$)', raw_text, re.DOTALL)
    
    if sections:
        json_dict = {}
        for key, value in sections:
            try:
                # Parse the individual Pythonic array/object string blocks safely
                json_dict[key] = json.loads(value.strip())
            except Exception:
                # If individual parsing fails, fall back to string cleaning
                continue
        return json_dict
        
    raise UnifiedGenerationError("model output is not valid JSON")
