from __future__ import annotations

import os
import base64
import html
import logging
from pathlib import Path
from typing import Any

import gradio as gr

from comic_gen import backends, comics, exercise, session, trace
from comic_gen.models import ValidationError

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Silence noisy HTTP client internals
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

try:
    from dotenv import load_dotenv

    # Load environment variables from .env file
    load_dotenv()
except ImportError:
    logger.warning("python-dotenv not installed, skipping .env loading")


LANGUAGE_OPTIONS = {
    "English": "en",
    "Espanol": "es",
    "Francais": "fr",
    "Deutsch": "de",
}

STYLE_OPTIONS = ["minimal", "newspaper", "watercolor", "retro"]
MAX_SEED = 2**31 - 1
MAX_IMAGE_SIZE = 512
DEFAULT_IMAGE_MODEL_ID = "black-forest-labs/FLUX.1-schnell"#"stabilityai/sdxl-turbo"
DEFAULT_OPENBMB_TEXT_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct" #"openbmb/MiniCPM5-1B"


def _render_source(document: dict[str, Any]) -> str:
    source = document["source"]
    article = document["article"]
    return (
        f"### Source\n"
        f"- Publisher: {source['publisher']}\n"
        f"- Title: {article['title']}\n"
        f"- Link: {source['link']}\n"
        f"- Published: {source['published_at']}"
    )


def _render_summary(document: dict[str, Any]) -> str:
    simplified = document["simplified"]
    keywords = ", ".join(simplified.get("keywords", []))
    return (
        f"### Simplified Summary ({simplified['level']})\n"
        f"{simplified['summary']}\n\n"
        f"Keywords: {keywords}"
    )


def _panel_image_src(panel: dict[str, Any]) -> tuple[str, str]:
    render = panel.get("render", {})
    image_path = render.get("image_path")
    source_label = html.escape(render.get("image_source", "deterministic"))
    if image_path:
        path = Path(image_path)
        if path.exists() and path.is_file():
            try:
                encoded = base64.b64encode(path.read_bytes()).decode("ascii")
                return f"data:image/png;base64,{encoded}", source_label
            except OSError:
                pass

    return backends.fallback_image_src(panel), "placeholder"


def _panel_image_html(panel: dict[str, Any]) -> str:
    image_src, _ = _panel_image_src(panel)
    image_tag = (
        "<img src='"
        f"{image_src}' alt='Panel {panel['frame_index']}' "
        "class='panel-image'/>"
    )
    return (
        "<div class='panel-media'>"
        f"{image_tag}"
        "</div>"
    )


def _overlay_debug_html(panel: dict[str, Any]) -> str:
    width = 512
    height = 512
    parts = [
        "<div class='overlay-debug'>",
        "<div class='overlay-canvas'>",
    ]
    dialogue = panel.get("dialogue", [])
    for index, bubble in enumerate(panel.get("bubbles", [])):
        bbox = bubble.get("bbox_px", [0, 0, 120, 60])
        x, y, box_w, box_h = bbox
        line = dialogue[index] if index < len(dialogue) else {}
        char_id = html.escape(line.get("character_id", "narrator"))
        text = html.escape(line.get("text", ""))
        left = max(0.0, min(100.0, (x / width) * 100))
        top = max(0.0, min(100.0, (y / height) * 100))
        bubble_w = max(8.0, min(100.0 - left, (box_w / width) * 100))
        bubble_h = max(8.0, min(100.0 - top, (box_h / height) * 100))
        parts.append(
            "<div class='overlay-bubble' "
            f"style='left:{left:.2f}%;top:{top:.2f}%;"
            f"width:{bubble_w:.2f}%;height:{bubble_h:.2f}%;'>"
            f"<div class='overlay-speaker'>{char_id}</div>"
            f"<div class='overlay-line'>{text}</div>"
            "</div>"
        )
    parts.append("</div>")
    parts.append("</div>")
    return "".join(parts)


def _render_panels_html(
    document: dict[str, Any],
    debug_mode: bool = False,
) -> str:
    cards = []
    for panel in document.get("panels", []):
        image_html = _panel_image_html(panel)
        if debug_mode:
            cards.append(
                (
                    "<div class='panel-card debug-card'>"
                    f"<h4>Panel {panel['frame_index']}</h4>"
                    "<div class='debug-row'>"
                    "<div class='debug-label'>Raw image</div>"
                    f"{image_html}"
                    "</div>"
                    "<div class='debug-row'>"
                    "<div class='debug-label'>Overlay preview</div>"
                    f"{_overlay_debug_html(panel)}"
                    "</div>"
                    "</div>"
                )
            )
            continue

        cards.append(
            (
                "<div class='panel-card'>"
                f"{image_html}"
                "</div>"
            )
        )

    return "<div class='panel-grid'>" + "".join(cards) + "</div>"


def _render_transcript(document: dict[str, Any]) -> str:
    lines: list[str] = ["### Transcript"]

    # Use enumerate to track if we are on the first panel or a later one
    for i, panel in enumerate(document.get("panels", [])):
        # If it's the 2nd panel or later, inject an empty string for a blank line gap
        if i > 0:
            lines.append("")

        lines.append(f"PANEL {panel['frame_index']}")
        for line in panel.get("dialogue", []):
            lines.append(f"- {line['character_id']}: {line['text']}")

    return "\n".join(lines)


def _panel_choices(document: dict[str, Any]) -> list[str]:
    return [f"panel_{p['frame_index']}" for p in document.get("panels", [])]


def generate_strip(
    language_label: str,
    style_id: str,
    use_live_feed: bool,
    panel_count: int,
    enable_model_generation: bool,
    use_serverless_api: bool,
    negative_prompt: str,
    seed: int,
    randomize_seed: bool,
    width: int,
    height: int,
    guidance_scale: float,
    num_inference_steps: int,
    debug_mode: bool,
) -> tuple[dict[str, Any], str, str, str, str, list[str], str, dict[str, Any]]:
    language = LANGUAGE_OPTIONS.get(language_label, "en")
    payload = backends.fetch_article(
        language,
        use_live_feed=use_live_feed,
        model_generation=enable_model_generation,
        )
    document = session.build_base_session(language, style_id, payload)
    document.setdefault("ui", {})["debug_mode"] = debug_mode

    # Toggle optional HF serverless generation path used for text and image.
    os.environ["HF_USE_SERVERLESS"] = "1" if use_serverless_api else "0"
    os.environ["HF_USE_SERVERLESS_IMAGE"] = "1" if use_serverless_api else "0"

    try:
        comics.generate_story_pipeline(
            document,
            panel_count=panel_count,
            enable_model_generation=enable_model_generation,
            text_model_repo_id=DEFAULT_OPENBMB_TEXT_MODEL_ID,
            image_options={
                "model_repo_id": DEFAULT_IMAGE_MODEL_ID,
                "use_serverless_image_api": use_serverless_api,
                "negative_prompt": negative_prompt,
                "seed": seed,
                "randomize_seed": randomize_seed,
                "width": width,
                "height": height,
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps,
            },
        )
        trace.add_trace(
            document,
            "step7_ui",
            "ok",
            "Selectors and panel routes prepared",
        )
        trace.add_trace(document, "text_generation_output_test", 'ok',
                        f'Text generation output trace test: {language}')
        session.validate_or_raise(document)
    except ValidationError as exc:
        trace.add_trace(document, "validation", "error", str(exc))
        raise gr.Error(f"Schema validation failed: {exc}")

    choices = _panel_choices(document)
    first_panel = choices[0] if choices else ""

    return (
        document,
        _render_source(document),
        _render_summary(document),
        _render_panels_html(document, debug_mode=debug_mode),
        _render_transcript(document),
        choices,
        first_panel,
        {"trace": document.get("trace", [])},
    )


def generate_strip_ui(
    language_label: str,
    style_id: str,
    use_live_feed: bool,
    panel_count: int,
    enable_model_generation: bool,
    use_serverless_api: bool,
    negative_prompt: str,
    seed: int,
    randomize_seed: bool,
    width: int,
    height: int,
    guidance_scale: float,
    num_inference_steps: int,
    debug_mode: bool,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> tuple[dict[str, Any], str, str, str, str, gr.Dropdown, dict[str, Any]]:
    progress(0.05, desc="Fetching article")
    (
        document,
        source_md,
        summary_md,
        panel_html,
        transcript_md,
        choices,
        first_panel,
        trace_payload,
    ) = generate_strip(
        language_label,
        style_id,
        use_live_feed,
        panel_count,
        enable_model_generation,
        use_serverless_api,
        negative_prompt,
        seed,
        randomize_seed,
        width,
        height,
        guidance_scale,
        num_inference_steps,
        debug_mode,
    )
    progress(1.0, desc="Comic strip ready")

    selector_update = gr.Dropdown(
        choices=choices,
        value=first_panel if first_panel else None,
    )

    return (
        document,
        source_md,
        summary_md,
        panel_html,
        transcript_md,
        selector_update,
        trace_payload,
    )


def load_exercise(
    selected_panel: str,
    document: dict[str, Any],
) -> tuple[str, str]:
    if not document:
        return "Generate a strip first.", ""

    exercise_item = next(
        (
            item
            for item in document.get("exercises", [])
            if item["panel_id"] == selected_panel
        ),
        None,
    )
    if not exercise_item:
        return "No exercise available for this panel.", ""

    return f"### Exercise\n{exercise_item['prompt']}", ""


def submit_answer(
    selected_panel: str,
    answer: str,
    document: dict[str, Any],
) -> str:
    if not document:
        return "Generate a strip first."
    ok, feedback = exercise.evaluate_answer(document, selected_panel, answer)
    status = "ok" if ok else "retry"
    trace.add_trace(document, "exercise_submit", status, feedback)
    return feedback


with gr.Blocks() as demo:
    session_state = gr.State({})

    with gr.Column(elem_id="app"):
        gr.Markdown("# Anti-Ill Comix")
        gr.Markdown(
            (
                "Turn international news into simple comic practice "
                "for adult reading and writing."
            )
        )

        with gr.Row():
            language_input = gr.Dropdown(
                choices=list(LANGUAGE_OPTIONS.keys()),
                value="English",
                label="Language",
            )
            style_input = gr.Dropdown(
                choices=STYLE_OPTIONS,
                value="minimal",
                label="Art style",
            )
            panel_count = gr.Slider(
                minimum=3,
                maximum=5,
                step=1,
                value=3,
                label="Panel count",
            )
            live_feed_input = gr.Checkbox(
                label="Use live RSS article",
                value=False,
            )

        with gr.Accordion("Model Generation (Optional)", open=False):
            enable_model_generation_input = gr.Checkbox(
                label="Enable model generation (text + images)",
                value=False,
            )
            use_serverless_api_input = gr.Checkbox(
                label="Use HF serverless API for text + image generation",
                value=False,
            )
            debug_mode_input = gr.Checkbox(
                label="Debug panel rendering",
                value=False,
            )
            negative_prompt_input = gr.Textbox(
                label="Negative prompt",
                placeholder="Optional quality or style exclusions",
                value="",
            )
            seed_input = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=0,
            )
            randomize_seed_input = gr.Checkbox(
                label="Randomize seed",
                value=True,
            )
            with gr.Row():
                width_input = gr.Slider(
                    label="Width",
                    minimum=256,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=256,
                )
                height_input = gr.Slider(
                    label="Height",
                    minimum=256,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=256,
                )
            with gr.Row():
                guidance_scale_input = gr.Slider(
                    label="Guidance scale",
                    minimum=0.0,
                    maximum=10.0,
                    step=0.1,
                    value=0.0,
                )
                num_steps_input = gr.Slider(
                    label="Inference steps",
                    minimum=1,
                    maximum=20,
                    step=1,
                    value=2,
                )

        generate_button = gr.Button("Generate Comic Strip", variant="primary")

        source_md = gr.Markdown()
        summary_md = gr.Markdown()
        panel_html = gr.HTML()
        transcript_md = gr.Markdown(elem_id="transcript_md")

        gr.Markdown("### Writing Exercises")
        panel_selector = gr.Dropdown(
            choices=[],
            label="Select panel",
            value=None,
        )
        exercise_md = gr.Markdown(
            value="Generate a strip to unlock exercises."
        )
        answer_input = gr.Textbox(
            label="Your answer",
            placeholder="Type the missing word",
        )
        submit_button = gr.Button("Submit Answer")
        feedback_md = gr.Markdown()

        trace_json = gr.JSON(label="Trace")

    generate_button.click(
        fn=generate_strip_ui,
        inputs=[
            language_input,
            style_input,
            live_feed_input,
            panel_count,
            enable_model_generation_input,
            use_serverless_api_input,
            negative_prompt_input,
            seed_input,
            randomize_seed_input,
            width_input,
            height_input,
            guidance_scale_input,
            num_steps_input,
            debug_mode_input,
        ],
        outputs=[
            session_state,
            source_md,
            summary_md,
            panel_html,
            transcript_md,
            panel_selector,
            trace_json,
        ],
    )

    panel_selector.change(
        fn=load_exercise,
        inputs=[panel_selector, session_state],
        outputs=[exercise_md, answer_input],
    )

    submit_button.click(
        fn=submit_answer,
        inputs=[panel_selector, answer_input, session_state],
        outputs=feedback_md,
    )


if __name__ == "__main__":
    css_path = Path(__file__).parent / "assets" / "style.css"
    css = ""
    if css_path.exists():
        css = css_path.read_text()

    demo.launch(css=css)
