from __future__ import annotations

import base64
import html
from pathlib import Path
from typing import Any
from urllib.parse import quote

import gradio as gr

from comic_gen import backends, comics, exercise, session, trace
from comic_gen.models import ValidationError

LANGUAGE_OPTIONS = {
    "English": "en",
    "Espanol": "es",
    "Francais": "fr",
    "Deutsch": "de",
}

STYLE_OPTIONS = ["minimal", "newspaper", "watercolor", "retro"]
MAX_SEED = 2**31 - 1
MAX_IMAGE_SIZE = 512
DEFAULT_IMAGE_MODEL_ID = "stabilityai/sdxl-turbo"
DEFAULT_OPENBMB_TEXT_MODEL_ID = "openbmb/MiniCPM5-1B"


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


def _bubble_html(line: dict[str, str]) -> str:
    char_id = html.escape(line["character_id"])
    text = html.escape(line["text"])
    return f"<div class='bubble'><b>{char_id}</b>: {text}</div>"


def _fallback_image_src(panel: dict[str, Any]) -> str:
    frame_index = panel.get("frame_index", "?")
    scene = html.escape(panel.get("scene_description", "Comic panel"))
    scene = scene[:110]
    svg_lines = [
        (
            '<svg xmlns="http://www.w3.org/2000/svg" '
            'width="512" height="512" viewBox="0 0 512 512">'
        ),
        '  <defs>',
        '    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">',
        '      <stop offset="0%" stop-color="#fff7e8"/>',
        '      <stop offset="100%" stop-color="#eef4ff"/>',
        '    </linearGradient>',
        '  </defs>',
        '  <rect width="512" height="512" rx="28" fill="url(#bg)"/>',
        (
            '  <rect x="24" y="24" width="464" height="464" '
            'rx="24" fill="none" stroke="#c5cfdb" stroke-width="3"/>'
        ),
        (
            '  <text x="40" y="72" font-size="22" '
            'font-family="Arial, sans-serif" fill="#415166">'
            f'Panel {frame_index}</text>'
        ),
        (
            '  <text x="40" y="120" font-size="16" '
            'font-family="Arial, sans-serif" fill="#5f6f82">'
            f'{scene}</text>'
        ),
        '  <circle cx="150" cy="270" r="54" fill="#d8e4f2"/>',
        '  <circle cx="344" cy="270" r="54" fill="#f2dec7"/>',
        (
            '  <rect x="106" y="334" width="88" height="108" '
            'rx="22" fill="#d8e4f2"/>'
        ),
        (
            '  <rect x="300" y="334" width="88" height="108" '
            'rx="22" fill="#f2dec7"/>'
        ),
        (
            '  <path d="M88 180h160a18 18 0 0 1 18 18v48a18 18 '
            '0 0 1-18 18h-90l-34 24 10-24H88a18 18 0 0 1-18-18v-48'
            'a18 18 0 0 1 18-18z" fill="#ffffff" stroke="#9ba6b2" '
            'stroke-width="3"/>'
        ),
        (
            '  <path d="M264 120h160a18 18 0 0 1 18 18v48a18 18 '
            '0 0 1-18 18h-84l-34 24 10-24h-52a18 18 0 0 1-18-18v-48'
            'a18 18 0 0 1 18-18z" fill="#ffffff" stroke="#9ba6b2" '
            'stroke-width="3"/>'
        ),
        '</svg>',
    ]
    svg = "\n".join(svg_lines)
    return "data:image/svg+xml;utf8," + quote(svg)


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

    return _fallback_image_src(panel), "placeholder"


def _panel_image_html(panel: dict[str, Any]) -> str:
    image_src, source_label = _panel_image_src(panel)
    image_tag = (
        "<img src='"
        f"{image_src}' alt='Panel {panel['frame_index']}' "
        "class='panel-image'/>"
    )
    return (
        "<div class='panel-media'>"
        f"{image_tag}"
        # f"<div class='image-meta'>image: {source_label}</div>"
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
        dialogue_html = "".join(
            [
                _bubble_html(line)
                for line in panel.get("dialogue", [])
            ]
        )
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
                    # f"<p>{html.escape(panel['scene_description'])}</p>"
                    # f"{dialogue_html}"
                    "</div>"
                )
            )
            continue

        cards.append(
            (
                "<div class='panel-card'>"
                # f"<h4>Panel {panel['frame_index']}</h4>"
                f"{image_html}"
                # f"<p>{html.escape(panel['scene_description'])}</p>"
                # f"{dialogue_html}"
                "</div>"
            )
        )

    return "<div class='panel-grid'>" + "".join(cards) + "</div>"


def _render_transcript(document: dict[str, Any]) -> str:
    lines: list[str] = ["### Transcript"]
    for panel in document.get("panels", []):
        lines.append(f"Panel {panel['frame_index']}")
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
    enable_live_images: bool,
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
    payload = backends.fetch_article(language, use_live_feed=use_live_feed)
    document = session.build_base_session(language, style_id, payload)
    document.setdefault("ui", {})["debug_mode"] = debug_mode

    try:
        comics.simplify_article(document)
        comics.generate_characters(document)
        comics.generate_panels(
            document,
            panel_count=panel_count,
            image_options={
                "enable_live_images": enable_live_images,
                "model_repo_id": DEFAULT_IMAGE_MODEL_ID,
                "negative_prompt": negative_prompt,
                "seed": seed,
                "randomize_seed": randomize_seed,
                "width": width,
                "height": height,
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps,
            },
        )
        comics.apply_overlay(document)
        exercise.generate_exercises(document)
        trace.add_trace(
            document,
            "step7_ui",
            "ok",
            "Selectors and panel routes prepared",
        )
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
    enable_live_images: bool,
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
        enable_live_images,
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

        with gr.Accordion("Image Generation (Optional)", open=False):
            enable_live_images = gr.Checkbox(
                label="Enable live panel image generation",
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
        transcript_md = gr.Markdown()

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
            enable_live_images,
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
