from __future__ import annotations

from app import (
    SERVERLESS_IMAGE_MODEL_ID,
    SERVERLESS_IMAGE_PROVIDER,
    SPACES_IMAGE_MODEL_ID,
    SPACES_IMAGE_PROVIDER,
    _select_image_model,
)


def test_select_image_model_uses_flux_for_serverless():
    model_id, provider = _select_image_model(use_serverless_api=True)

    assert model_id == SERVERLESS_IMAGE_MODEL_ID
    assert model_id == "black-forest-labs/FLUX.1-schnell"
    assert provider == SERVERLESS_IMAGE_PROVIDER
    assert provider == "black-forest-labs"


def test_select_image_model_uses_sdxl_turbo_for_spaces():
    model_id, provider = _select_image_model(use_serverless_api=False)

    assert model_id == SPACES_IMAGE_MODEL_ID
    assert model_id == "stabilityai/sdxl-turbo"
    assert provider == SPACES_IMAGE_PROVIDER
    assert provider == "hf-inference"
