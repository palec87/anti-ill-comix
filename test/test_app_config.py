from __future__ import annotations

import os

from app import (
    SERVERLESS_IMAGE_MODEL_ID,
    SPACES_IMAGE_MODEL_ID,
    _select_image_model,
    generate_strip,
)


def test_select_image_model_uses_flux_for_serverless():
    model_id = _select_image_model(use_serverless_api=True)

    assert model_id == SERVERLESS_IMAGE_MODEL_ID
    assert model_id == "black-forest-labs/FLUX.1-schnell"


def test_select_image_model_uses_flux_for_spaces():
    model_id = _select_image_model(use_serverless_api=False)

    assert model_id == SPACES_IMAGE_MODEL_ID
    assert model_id == "black-forest-labs/FLUX.1-schnell"


def _patch_generate_strip_dependencies(monkeypatch):
    captured = {}

    def _fake_fetch_article(language, use_live_feed=False):
        captured["language"] = language
        captured["use_live_feed"] = use_live_feed
        return {"title": "News", "fulltext": "Short article"}

    def _fake_build_base_session(language, style_id, payload):
        return {
            "language": language,
            "style_id": style_id,
            "source": {
                "publisher": "Example",
                "link": "https://example.test",
                "published_at": "2026-06-15",
            },
            "article": {
                "title": payload["title"],
                "fulltext": payload["fulltext"],
            },
            "simplified": {
                "summary": "Short summary",
                "level": "A2",
                "keywords": ["news"],
            },
            "characters": [],
            "panels": [],
            "exercises": [],
            "trace": [],
            "ui": {},
        }

    def _fake_generate_story_pipeline(
        document,
        panel_count,
        reading_level,
        text_model_repo_id,
        image_options,
    ):
        captured["panel_count"] = panel_count
        captured["reading_level"] = reading_level
        captured["text_model_repo_id"] = text_model_repo_id
        captured["image_options"] = image_options

    monkeypatch.setattr("app.backends.fetch_article", _fake_fetch_article)
    monkeypatch.setattr("app.session.build_base_session", _fake_build_base_session)
    monkeypatch.setattr("app.comics.generate_story_pipeline", _fake_generate_story_pipeline)
    monkeypatch.setattr("app.session.validate_or_raise", lambda document: None)
    return captured


def test_generate_strip_defaults_to_serverless_when_local_unchecked(monkeypatch):
    captured = _patch_generate_strip_dependencies(monkeypatch)

    generate_strip(
        "English",
        "minimal",
        "A2",
        False,
        3,
        False,
        "",
        0,
        True,
        256,
        256,
        0.0,
        2,
    )

    assert os.environ["HF_USE_SERVERLESS"] == "1"
    assert os.environ["HF_USE_SERVERLESS_IMAGE"] == "1"
    assert captured["image_options"]["use_serverless_image_api"] is True
    assert captured["image_options"]["model_repo_id"] == SERVERLESS_IMAGE_MODEL_ID


def test_generate_strip_uses_local_when_advanced_option_checked(monkeypatch):
    captured = _patch_generate_strip_dependencies(monkeypatch)

    generate_strip(
        "English",
        "minimal",
        "A2",
        False,
        3,
        True,
        "",
        0,
        True,
        256,
        256,
        0.0,
        2,
    )

    assert os.environ["HF_USE_SERVERLESS"] == "0"
    assert os.environ["HF_USE_SERVERLESS_IMAGE"] == "0"
    assert captured["image_options"]["use_serverless_image_api"] is False
    assert captured["image_options"]["model_repo_id"] == SPACES_IMAGE_MODEL_ID
