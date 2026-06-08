from __future__ import annotations

import random
import tempfile
from pathlib import Path
from typing import Any

from .trace import add_trace

MAX_SEED = 2**31 - 1

_PIPELINE: Any | None = None
_PIPELINE_MODEL_ID: str | None = None
_DEVICE: str | None = None


class ImageGenerationError(Exception):
    """Raised when live image generation fails."""


def _get_pipeline(model_repo_id: str) -> tuple[Any, Any, str]:
    global _PIPELINE
    global _PIPELINE_MODEL_ID
    global _DEVICE

    if _PIPELINE is not None and _PIPELINE_MODEL_ID == model_repo_id:
        return _PIPELINE, __import__("torch"), _DEVICE or "cpu"

    try:
        import torch
        from diffusers import DiffusionPipeline
    except Exception as exc:
        raise ImageGenerationError(
            f"Diffusers runtime unavailable: {exc}"
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    try:
        pipe = DiffusionPipeline.from_pretrained(
            model_repo_id,
            torch_dtype=torch_dtype,
        )
        pipe = pipe.to(device)
    except Exception as exc:
        raise ImageGenerationError(
            f"Failed to load model {model_repo_id}: {exc}"
        ) from exc

    _PIPELINE = pipe
    _PIPELINE_MODEL_ID = model_repo_id
    _DEVICE = device
    return _PIPELINE, torch, device


def generate_panel_image(
    *,
    document: dict[str, Any],
    prompt: str,
    negative_prompt: str,
    session_id: str,
    panel_id: str,
    model_repo_id: str,
    seed: int,
    randomize_seed: bool,
    width: int,
    height: int,
    guidance_scale: float,
    num_inference_steps: int,
) -> tuple[str, int, str]:
    pipe, torch, device = _get_pipeline(model_repo_id)

    chosen_seed = random.randint(0, MAX_SEED) if randomize_seed else seed
    generator = torch.Generator(device="cpu").manual_seed(chosen_seed)

    try:
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
            generator=generator,
        ).images[0]

    except Exception as exc:
        add_trace(
            document,
            "step4_image_generate",
            "error",
            f"Inference failed for {panel_id}: {exc}",
        )
        raise ImageGenerationError(
            f"Inference failed for {panel_id}: {exc}"
        ) from exc

    out_dir = Path(tempfile.gettempdir()) / "anti_ill_comix" / session_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{panel_id}.png"
    image.save(out_path)
    add_trace(
        document,
        "step4: generate_panel_image()",
        "ok",
        f"{panel_id} image saved to {out_path}",
    )
    return str(out_path), chosen_seed, device
