from __future__ import annotations

import logging
import os
import random
import tempfile
from pathlib import Path
from typing import Any
from time import perf_counter

from comic_gen.errors import ModelPipelineError, ImageGenerationError

from .trace import add_trace

MAX_SEED = 2**31 - 1
logger = logging.getLogger(__name__)

IS_LOCAL = os.environ.get("LOCAL_DEV", "False") == "True"
_GENERATOR: Any | None = None
_GENERATOR_MODEL_ID = ""
_INFERENCE_CLIENT: Any | None = None


def _is_serverless_image_enabled(options: dict[str, Any]) -> bool:
    option_value = bool(options.get("use_serverless_image_api", False))
    env_value = os.environ.get("HF_USE_SERVERLESS_IMAGE", "0").strip().lower()
    env_enabled = env_value in {"1", "true", "yes", "on"}
    enabled = option_value or env_enabled
    logger.debug(
        "HF_USE_SERVERLESS_IMAGE=%s use_serverless_image_api=%s enabled=%s",
        env_value,
        option_value,
        enabled,
    )
    return enabled


def _get_inference_client() -> Any:
    global _INFERENCE_CLIENT
    if _INFERENCE_CLIENT is not None:
        return _INFERENCE_CLIENT

    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise ImageGenerationError(
            "HF_TOKEN is required for serverless image API mode"
        )

    try:
        from huggingface_hub import InferenceClient
    except Exception as exc:
        raise ImageGenerationError(
            "huggingface_hub import failed for serverless image API mode"
        ) from exc

    _INFERENCE_CLIENT = InferenceClient(token=token)
    return _INFERENCE_CLIENT


def _build_output_path(session_id: str, panel_id: str) -> Path:
    out_dir = Path(tempfile.gettempdir()) / "anti_ill_comix" / session_id
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{panel_id}.png"


def _generate_panel_image_serverless(
    *,
    prompt: str,
    negative_prompt: str,
    session_id: str,
    panel_id: str,
    model_repo_id: str,
    width: int,
    height: int,
    guidance_scale: float,
    num_inference_steps: int,
) -> tuple[str, str]:
    client = _get_inference_client()
    try:
        image = client.text_to_image(
            prompt=prompt,
            model=model_repo_id,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        )
    except Exception as exc:
        raise ImageGenerationError(
            f"Serverless inference failed for {panel_id}: {exc}"
        ) from exc

    out_path = _build_output_path(session_id, panel_id)
    try:
        image.save(out_path)
    except Exception as exc:
        raise ImageGenerationError(
            f"Failed to save serverless image for {panel_id}: {exc}"
        ) from exc
    return str(out_path), "serverless"


def conditional_gpu_decorator(func):
    if not IS_LOCAL:
        import spaces
        return spaces.GPU(func)  # Wraps with ZeroGPU in production.
    return func  # Passes through for local development.


@conditional_gpu_decorator
def _generate_panel_image(
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
    use_serverless_api: bool,
) -> tuple[str, int, str]:
    chosen_seed = random.randint(0, MAX_SEED) if randomize_seed else seed

    if use_serverless_api:
        out_path, provider = _generate_panel_image_serverless(
            prompt=prompt,
            negative_prompt=negative_prompt,
            session_id=session_id,
            panel_id=panel_id,
            model_repo_id=model_repo_id,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        )
        add_trace(
            document,
            "step4: generate_panel_image()",
            "ok",
            f"{panel_id} image saved to {out_path}",
        )
        return out_path, chosen_seed, provider

    import torch
    from diffusers import DiffusionPipeline

    # switch to "mps" for apple devices
    pipe = DiffusionPipeline.from_pretrained(
        model_repo_id,
        # "black-forest-labs/FLUX.1-schnell",
        dtype=torch.float16,
        token=os.environ.get("HF_TOKEN"),
        device="cuda",
    )
    pipe.to(device="cuda", dtype=torch.float16)
    # image = pipe(prompt=prompt).images[0]

    image = pipe(
        prompt=prompt,
        model=model_repo_id,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
    ).images[0]

    out_path = _build_output_path(session_id, panel_id)
    image.save(out_path)
    add_trace(
        document,
        "step4: generate_panel_image()",
        "ok",
        f"{panel_id} image saved to {out_path}",
    )
    return str(out_path), chosen_seed, 'cuda'


def generate_image_panels(
    document: dict[str, Any],
    panels: list[dict[str, Any]],
    options: dict[str, Any],
    strict_mode: bool = False,
) -> dict[str, int]:
    image_source_counts: dict[str, int] = {}
    use_serverless_api = _is_serverless_image_enabled(options)
    if options["enable_live_images"]:
        for panel in panels:
            panel_id = panel.get("panel_id", "")
            frame_index = int(panel.get("frame_index", 1))
            render = panel.setdefault("render", {})
            fallback_path = str(
                render.get("image_path", f"assets/panel_{frame_index}.png")
            )
            panel_seed = options["seed"] + (frame_index - 1)
            started = perf_counter()
            add_trace(
                document,
                "step4_image_generate",
                "start",
                f"{panel_id} generation started",
            )
            try:
                image_path, used_seed, device = _generate_panel_image(
                    document=document,
                    prompt=str(panel.get("scene_description", "")),
                    negative_prompt=options["negative_prompt"],
                    session_id=document["session_id"],
                    panel_id=panel_id,
                    model_repo_id=options["model_repo_id"],
                    seed=panel_seed,
                    randomize_seed=options["randomize_seed"],
                    width=options["width"],
                    height=options["height"],
                    guidance_scale=options["guidance_scale"],
                    num_inference_steps=options["num_inference_steps"],
                    use_serverless_api=use_serverless_api,
                )
                elapsed_ms = int((perf_counter() - started) * 1000)
                render["image_path"] = image_path
                image_source = "serverless" if device == "serverless" else "live"
                render["image_source"] = image_source
                image_source_counts[image_source] = (
                    image_source_counts.get(image_source, 0) + 1
                )
                render["seed"] = used_seed
                render["device"] = device
                add_trace(
                    document,
                    "step4_image_generate",
                    "ok",
                    f"{panel_id} {image_source} image generated in {elapsed_ms}ms",
                )
            except ImageGenerationError as exc:
                elapsed_ms = int((perf_counter() - started) * 1000)
                if strict_mode:
                    raise ModelPipelineError(
                        f"image generation failed for {panel_id}: {exc}"
                    ) from exc
                render["image_path"] = fallback_path
                render["image_source"] = "fallback"
                image_source_counts["fallback"] = (
                    image_source_counts.get("fallback", 0) + 1
                )
                render["seed"] = panel_seed
                add_trace(
                    document,
                    "step4_image_generate",
                    "fallback",
                    f"{panel_id} fallback after {elapsed_ms}ms: {exc}",
                )
    else:
        for panel in panels:
            render = panel.setdefault("render", {})
            frame_index = int(panel.get("frame_index", 1))
            render.setdefault("image_path", f"assets/panel_{frame_index}.png")
            render["image_source"] = "deterministic"
            image_source_counts["deterministic"] = (
                image_source_counts.get("deterministic", 0) + 1
            )

    summary_bits = [
        f"{key}={value}"
        for key, value in image_source_counts.items()
    ]
    add_trace(
        document,
        "step4_panels",
        "ok",
        (
            f"Generated {len(panels)} comic panels"
            f" ({', '.join(summary_bits)})"
        ),
    )
    return image_source_counts
