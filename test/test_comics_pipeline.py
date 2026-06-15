from __future__ import annotations

import pytest

from comic_gen import comics
from comic_gen.errors import UnifiedGenerationError


def _document() -> dict:
    return {
        "language": "en",
        "style_id": "minimal",
        "article": {
            "title": "Garden news",
            "fulltext": "Adults read a short article.",
        },
        "simplified": {"summary": "", "level": "A2", "keywords": []},
        "characters": [],
        "panels": [],
        "exercises": [],
        "trace": [],
    }


def _generated(level: str = "A2") -> dict:
    panels = []
    exercises = []
    for index in range(1, 4):
        panel_id = f"P{index}"
        panels.append(
            {
                "panel_id": panel_id,
                "frame_index": index,
                "scene_description": f"Scene {index}",
                "dialogue": [{"character_id": "A1", "text": "Line"}],
                "bubbles": [{"bbox_px": [10, 10, 108, 30]}],
                "render": {
                    "image_path": f"assets/panel_{index}.png",
                    "overlay_applied": True,
                },
            }
        )
        exercises.append(
            {
                "exercise_id": f"E{index}",
                "panel_id": panel_id,
                "prompt": "Line ____",
                "blanks": ["____"],
                "answer_key": ["line"],
                "feedback_rules": {
                    "case_sensitive": False,
                    "allow_trim_spaces": True,
                },
            }
        )
    return {
        "simplified": {
            "summary": "Short summary",
            "level": level,
            "keywords": ["reading"],
        },
        "characters": [
            {
                "id": "A1",
                "name": "Guide",
                "description": "Plain language guide.",
            }
        ],
        "panels": panels,
        "exercises": exercises,
    }


def test_generate_story_pipeline_passes_reading_level(monkeypatch):
    captured = {}

    def _fake_text(**kwargs):
        captured["reading_level"] = kwargs["reading_level"]
        captured["language"] = kwargs["language"]
        return _generated(level=kwargs["reading_level"])

    monkeypatch.setattr(
        "comic_gen.comics.generate_text_content_from_article",
        _fake_text,
    )
    monkeypatch.setattr(
        "comic_gen.comics.generate_image_panels",
        lambda *args, **kwargs: {"deterministic": 3},
    )

    document = _document()
    comics.generate_story_pipeline(
        document,
        panel_count=3,
        text_model_repo_id="test-model",
        reading_level="B2",
        image_options={"enable_live_images": False},
    )

    assert captured["reading_level"] == "B2"
    assert captured["language"] == "en"
    assert document["simplified"]["level"] == "B2"


def test_generate_story_pipeline_translates_before_images(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        "comic_gen.comics.generate_text_content_from_article",
        lambda **kwargs: _generated(level=kwargs["reading_level"]),
    )

    def _fake_translate(document, target_language, **kwargs):
        captured["target_language"] = target_language
        document["panels"][0]["dialogue"][0]["text"] = "ES:Linea"
        document["exercises"][0]["prompt"] = "ES:Linea ____"
        return True

    def _fake_images(document, panels, options, strict_mode=False):
        captured["dialogue_for_image"] = panels[0]["dialogue"][0]["text"]
        captured["exercise_for_image"] = document["exercises"][0]["prompt"]
        return {"deterministic": 3}

    monkeypatch.setattr(
        "comic_gen.comics.translate_session_content",
        _fake_translate,
    )
    monkeypatch.setattr(
        "comic_gen.comics.generate_image_panels",
        _fake_images,
    )

    document = _document()
    document["language"] = "es"
    comics.generate_story_pipeline(
        document,
        panel_count=3,
        text_model_repo_id="test-model",
        reading_level="A2",
        image_options={"enable_live_images": False},
    )

    assert captured["target_language"] == "es"
    assert captured["dialogue_for_image"] == "ES:Linea"
    assert captured["exercise_for_image"] == "ES:Linea ____"


def test_generate_story_pipeline_translation_failure_keeps_content(monkeypatch):
    monkeypatch.setattr(
        "comic_gen.comics.generate_text_content_from_article",
        lambda **kwargs: _generated(level=kwargs["reading_level"]),
    )
    monkeypatch.setattr(
        "comic_gen.comics.translate_session_content",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(
        "comic_gen.comics.generate_image_panels",
        lambda *args, **kwargs: {"deterministic": 3},
    )

    document = _document()
    document["language"] = "fr"
    comics.generate_story_pipeline(
        document,
        panel_count=3,
        text_model_repo_id="test-model",
        reading_level="A2",
        image_options={"enable_live_images": False},
    )

    assert document["panels"][0]["dialogue"][0]["text"] == "Line"
    assert any(
        item["step"] == "translation" and item["status"] == "fallback"
        for item in document["trace"]
    )


def test_generate_story_pipeline_fallback_preserves_reading_level(monkeypatch):
    def _fake_text(**kwargs):
        raise UnifiedGenerationError("text failed")

    def _fake_deterministic(document):
        document["simplified"] = {
            "summary": "Fallback summary",
            "level": "A2",
            "keywords": ["fallback"],
        }

    monkeypatch.setattr(
        "comic_gen.comics.generate_text_content_from_article",
        _fake_text,
    )
    monkeypatch.setattr(
        "comic_gen.comics.deterministic_pipeline",
        _fake_deterministic,
    )

    document = _document()
    comics.generate_story_pipeline(
        document,
        panel_count=3,
        text_model_repo_id="test-model",
        reading_level="B2",
        image_options={"enable_live_images": False},
    )

    assert document["simplified"]["level"] == "B2"
