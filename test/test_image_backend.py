from __future__ import annotations

import pytest

from comic_gen.errors import ModelPipelineError
from comic_gen.image_backend import (
    IMAGE_TEXT_NEGATIVE_PROMPT,
    ImageGenerationError,
    _generate_panel_image,
    build_image_prompt,
    generate_image_panels,
)


def _document() -> dict:
    return {
        "session_id": "abc",
        "language": "en",
        "style_id": "watercolor",
        "simplified": {
            "summary": "Adults read a short article about a garden project.",
            "level": "A2",
            "keywords": ["garden", "reading", "confidence"],
        },
        "characters": [
            {
                "id": "char_guide",
                "name": "Guide",
                "description": "A calm mentor using plain language.",
            },
            {
                "id": "char_learner",
                "name": "Learner",
                "description": "An adult practicing reading.",
            },
        ],
        "trace": [],
    }


def _panel() -> dict:
    return {
        "panel_id": "panel_1",
        "frame_index": 1,
        "scene_description": "Two adults read together in a garden.",
        "dialogue": [
            {
                "character_id": "char_guide",
                "text": "We read one short instruction together.",
            },
            {
                "character_id": "char_learner",
                "text": "Now I understand the main idea.",
            },
        ],
        "render": {},
    }


def _options(enable_live_images: bool = True) -> dict:
    return {
        "enable_live_images": enable_live_images,
        "use_serverless_image_api": True,
        "model_repo_id": "stabilityai/sdxl-turbo",
        "negative_prompt": "",
        "seed": 11,
        "randomize_seed": False,
        "width": 256,
        "height": 256,
        "guidance_scale": 0.0,
        "num_inference_steps": 2,
    }


def test_build_image_prompt_includes_session_and_panel_text():
    prompt = build_image_prompt(_document(), _panel())

    assert "watercolor" in prompt
    assert "Language context: en" in prompt
    assert "Adults read a short article" in prompt
    assert "garden, reading, confidence" in prompt
    assert "Guide: A calm mentor" in prompt
    assert "Two adults read together" in prompt
    assert "We read one short instruction" in prompt
    assert "Do not draw readable letters" in prompt
    assert "Do not draw speech bubbles" in prompt
    assert "readable words" in prompt


def test_generate_panel_image_uses_serverless_without_local_fallback(
    monkeypatch,
):
    def _fake_serverless(**kwargs):
        assert kwargs["seed"] == 5
        return "C:/tmp/panel_1.png", "serverless"

    monkeypatch.setattr(
        "comic_gen.image_backend._generate_panel_image_serverless",
        _fake_serverless,
    )

    out_path, used_seed, image_source = _generate_panel_image(
        document={"trace": []},
        prompt="A panel scene",
        negative_prompt="",
        session_id="session-1",
        panel_id="panel_1",
        model_repo_id="stabilityai/sdxl-turbo",
        seed=5,
        randomize_seed=False,
        width=256,
        height=256,
        guidance_scale=0.0,
        num_inference_steps=2,
        use_serverless_api=True,
    )

    assert out_path.endswith("panel_1.png")
    assert used_seed == 5
    assert image_source == "serverless"


def test_generate_image_panels_passes_stored_image_prompt(monkeypatch):
    captured_prompt = ""
    captured_negative_prompt = ""

    def _fake_generate_panel_image(**kwargs):
        nonlocal captured_prompt, captured_negative_prompt
        captured_prompt = kwargs["prompt"]
        captured_negative_prompt = kwargs["negative_prompt"]
        assert kwargs["use_serverless_api"] is True
        return "C:/tmp/panel_1.png", 11, "serverless"

    monkeypatch.setattr(
        "comic_gen.image_backend._generate_panel_image",
        _fake_generate_panel_image,
    )

    document = _document()
    panels = [_panel()]

    counts = generate_image_panels(
        document,
        panels,
        _options(enable_live_images=True),
        strict_mode=True,
    )

    assert counts == {"serverless": 1}
    assert panels[0]["render"]["image_source"] == "serverless"
    assert panels[0]["render"]["image_prompt"] == captured_prompt
    assert "Adults read a short article" in captured_prompt
    assert "We read one short instruction" in captured_prompt
    assert IMAGE_TEXT_NEGATIVE_PROMPT in captured_negative_prompt
    assert panels[0]["render"]["overlay_applied"] is True


def test_generate_image_panels_records_prompt_for_deterministic_path(
    monkeypatch,
):
    monkeypatch.delenv("HF_USE_SERVERLESS_IMAGE", raising=False)
    document = _document()
    panels = [_panel()]

    counts = generate_image_panels(
        document,
        panels,
        _options(enable_live_images=False),
        strict_mode=True,
    )

    assert counts == {"deterministic": 1}
    assert panels[0]["render"]["image_source"] == "deterministic"
    assert "Adults read a short article" in panels[0]["render"]["image_prompt"]
    assert panels[0]["render"]["overlay_applied"] is True


def test_generate_image_panels_raises_model_error_on_serverless_failure(
    monkeypatch,
):
    def _failing_generate_panel_image(**kwargs):
        raise ImageGenerationError("serverless failed")

    monkeypatch.setattr(
        "comic_gen.image_backend._generate_panel_image",
        _failing_generate_panel_image,
    )

    with pytest.raises(ModelPipelineError, match="image generation failed"):
        generate_image_panels(
            _document(),
            [_panel()],
            _options(enable_live_images=True),
            strict_mode=True,
        )
