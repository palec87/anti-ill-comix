from __future__ import annotations

import pytest

from comic_gen.errors import ModelPipelineError
from comic_gen.comics import _normalized_image_options
from comic_gen.image_backend import (
    IMAGE_TEXT_NEGATIVE_PROMPT,
    ImageGenerationError,
    _generate_panel_image,
    _get_diffusion_pipeline,
    _pipeline_call_kwargs,
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
        "model_repo_id": "black-forest-labs/FLUX.1-schnell",
        "negative_prompt": "",
        "seed": 11,
        "randomize_seed": False,
        "width": 256,
        "height": 256,
        "guidance_scale": 0.0,
        "num_inference_steps": 2,
    }


def test_normalized_image_options_unwraps_singleton_model_tuple():
    options = _normalized_image_options(
        {"model_repo_id": ("black-forest-labs/FLUX.1-schnell",)}
    )

    assert options["model_repo_id"] == "black-forest-labs/FLUX.1-schnell"


def test_build_image_prompt_includes_session_and_panel_text():
    prompt = build_image_prompt(_document(), _panel())

    assert "watercolor" in prompt
    assert "Two adults read together" in prompt
    assert prompt.startswith("Plain comic scene only.")
    assert "Characters and background only." in prompt
    assert "speech" not in prompt.lower()
    assert "bubble" not in prompt.lower()
    assert "text" not in prompt.lower()
    assert "We read one short instruction" not in prompt
    assert "Adults read a short article" not in prompt
    assert len(prompt.split()) <= 35
    assert "speech bubble" in IMAGE_TEXT_NEGATIVE_PROMPT
    assert "dialogue balloon" in IMAGE_TEXT_NEGATIVE_PROMPT
    assert "words" in IMAGE_TEXT_NEGATIVE_PROMPT


def test_generate_panel_image_uses_serverless_without_local_fallback(
    monkeypatch,
):
    captured = {}

    def _fake_serverless(**kwargs):
        captured["model_repo_id"] = kwargs["model_repo_id"]
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
        model_repo_id=("black-forest-labs/FLUX.1-schnell",),
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
    assert captured["model_repo_id"] == "black-forest-labs/FLUX.1-schnell"


def test_pipeline_call_kwargs_filters_unsupported_model_kwarg():
    class _Pipe:
        def __call__(self, prompt, width, max_sequence_length=None):
            return None

    kwargs = _pipeline_call_kwargs(
        _Pipe(),
        {
            "prompt": "Prompt",
            "width": 256,
            "height": 256,
            "model": "bad",
            "max_sequence_length": 256,
        },
    )

    assert kwargs == {
        "prompt": "Prompt",
        "width": 256,
        "max_sequence_length": 256,
    }


def test_local_pipeline_is_cached(monkeypatch):
    import comic_gen.image_backend as image_backend

    calls = {"from_pretrained": 0, "to": 0}

    class _Pipe:
        def to(self, **kwargs):
            calls["to"] += 1
            return self

    class _FakePipeline:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            calls["from_pretrained"] += 1
            return _Pipe()

    monkeypatch.setenv("HF_TOKEN", "hf_test")
    monkeypatch.setattr(
        "diffusers.DiffusionPipeline",
        _FakePipeline,
    )
    image_backend._reset_diffusion_pipeline()

    first = _get_diffusion_pipeline("black-forest-labs/FLUX.1-schnell")
    second = _get_diffusion_pipeline("black-forest-labs/FLUX.1-schnell")

    assert first is second
    assert calls == {"from_pretrained": 1, "to": 1}


def test_local_generation_retries_cuda_errors_and_resets(monkeypatch):
    calls = {"attempts": 0, "resets": 0}

    def _fake_local_once(**kwargs):
        calls["attempts"] += 1
        if calls["attempts"] < 3:
            raise RuntimeError("NVML CUDACachingAllocator INTERNAL ASSERT")
        return "C:/tmp/panel_1.png", "cuda"

    def _fake_reset():
        calls["resets"] += 1

    monkeypatch.setattr(
        "comic_gen.image_backend._generate_panel_image_local_once",
        _fake_local_once,
    )
    monkeypatch.setattr(
        "comic_gen.image_backend._reset_diffusion_pipeline",
        _fake_reset,
    )

    out_path, used_seed, image_source = _generate_panel_image(
        document={"trace": []},
        prompt="Prompt",
        negative_prompt="",
        session_id="session-1",
        panel_id="panel_1",
        model_repo_id="black-forest-labs/FLUX.1-schnell",
        seed=5,
        randomize_seed=False,
        width=256,
        height=256,
        guidance_scale=0.0,
        num_inference_steps=2,
        use_serverless_api=False,
    )

    assert out_path.endswith("panel_1.png")
    assert used_seed == 5
    assert image_source == "cuda"
    assert calls == {"attempts": 3, "resets": 2}


def test_local_generation_raises_after_three_failed_attempts(monkeypatch):
    calls = {"attempts": 0}

    def _fake_local_once(**kwargs):
        calls["attempts"] += 1
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(
        "comic_gen.image_backend._generate_panel_image_local_once",
        _fake_local_once,
    )
    monkeypatch.setattr(
        "comic_gen.image_backend._reset_diffusion_pipeline",
        lambda: None,
    )

    with pytest.raises(ImageGenerationError, match="after 3 attempts"):
        _generate_panel_image(
            document={"trace": []},
            prompt="Prompt",
            negative_prompt="",
            session_id="session-1",
            panel_id="panel_1",
            model_repo_id="black-forest-labs/FLUX.1-schnell",
            seed=5,
            randomize_seed=False,
            width=256,
            height=256,
            guidance_scale=0.0,
            num_inference_steps=2,
            use_serverless_api=False,
        )

    assert calls["attempts"] == 3


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
    assert captured_prompt.startswith("Plain comic scene only.")
    assert "Two adults read together" in captured_prompt
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
    assert panels[0]["render"]["image_prompt"].startswith(
        "Plain comic scene only."
    )
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
