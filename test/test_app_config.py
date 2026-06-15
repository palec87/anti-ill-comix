from __future__ import annotations

from app import (
    SERVERLESS_IMAGE_MODEL_ID,
    SPACES_IMAGE_MODEL_ID,
    _select_image_model,
)


def test_select_image_model_uses_flux_for_serverless():
    model_id = _select_image_model(use_serverless_api=True)

    assert model_id == SERVERLESS_IMAGE_MODEL_ID
    assert model_id == "black-forest-labs/FLUX.1-schnell"


def test_select_image_model_uses_flux_for_spaces():
    model_id = _select_image_model(use_serverless_api=False)

    assert model_id == SPACES_IMAGE_MODEL_ID
    assert model_id == "black-forest-labs/FLUX.1-schnell"
