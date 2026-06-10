from __future__ import annotations

import pytest

from comic_gen.errors import ModelPipelineError
from comic_gen.image_backend import (
    ImageGenerationError,
    _generate_panel_image,
    apply_image_generation_to_panels,
)


def test_generate_panel_image_uses_serverless_without_local_fallback(monkeypatch):
    def _fake_serverless(**kwargs):
        return "C:/tmp/panel_1.png", "serverless"

    def _should_not_call_local(_model_repo_id: str):
        raise AssertionError("Local pipeline should not be called")

    monkeypatch.setattr(
        "comic_gen.image_backend._generate_panel_image_serverless",
        _fake_serverless,
    )
    monkeypatch.setattr(
        "comic_gen.image_backend._get_pipeline",
        _should_not_call_local,
    )

    out_path, used_seed, provider = _generate_panel_image(
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
    assert provider == "serverless"


def test_apply_image_generation_marks_serverless_source(monkeypatch):
    def _fake_generate_panel_image(**kwargs):
        assert kwargs["use_serverless_api"] is True
        return "C:/tmp/panel_1.png", 11, "serverless"

    monkeypatch.setattr(
        "comic_gen.image_backend._generate_panel_image",
        _fake_generate_panel_image,
    )

    document = {"session_id": "abc", "trace": []}
    panels = [
        {
            "panel_id": "panel_1",
            "frame_index": 1,
            "scene_description": "Scene",
            "render": {},
        }
    ]
    options = {
        "enable_live_images": True,
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

    counts = apply_image_generation_to_panels(
        document,
        panels,
        options,
        strict_mode=True,
    )

    assert counts == {"serverless": 1}
    assert panels[0]["render"]["image_source"] == "serverless"


def test_apply_image_generation_raises_model_error_on_serverless_failure(monkeypatch):
    def _failing_generate_panel_image(**kwargs):
        raise ImageGenerationError("serverless failed")

    monkeypatch.setattr(
        "comic_gen.image_backend._generate_panel_image",
        _failing_generate_panel_image,
    )

    document = {"session_id": "abc", "trace": []}
    panels = [
        {
            "panel_id": "panel_1",
            "frame_index": 1,
            "scene_description": "Scene",
            "render": {},
        }
    ]
    options = {
        "enable_live_images": True,
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

    with pytest.raises(ModelPipelineError, match="image generation failed"):
        apply_image_generation_to_panels(
            document,
            panels,
            options,
            strict_mode=True,
        )
