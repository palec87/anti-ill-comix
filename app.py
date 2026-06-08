from __future__ import annotations

import html
from typing import Any

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


def _render_panels_html(document: dict[str, Any]) -> str:
    cards = []
    for panel in document.get("panels", []):
        dialogue_html = "".join(
            [
                _bubble_html(line)
                for line in panel.get("dialogue", [])
            ]
        )
        cards.append(
            (
                "<div class='panel-card'>"
                f"<h4>Panel {panel['frame_index']}</h4>"
                f"<p>{html.escape(panel['scene_description'])}</p>"
                f"{dialogue_html}"
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
) -> tuple[dict[str, Any], str, str, str, str, list[str], str, dict[str, Any]]:
    language = LANGUAGE_OPTIONS.get(language_label, "en")
    payload = backends.fetch_article(language, use_live_feed=use_live_feed)
    document = session.build_base_session(language, style_id, payload)

    try:
        comics.simplify_article(document)
        comics.generate_characters(document)
        comics.generate_panels(document, panel_count=panel_count)
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
        _render_panels_html(document),
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
) -> tuple[dict[str, Any], str, str, str, str, gr.Dropdown, dict[str, Any]]:
    (
        document,
        source_md,
        summary_md,
        panel_html,
        transcript_md,
        choices,
        first_panel,
        trace_payload,
    ) = generate_strip(language_label, style_id, use_live_feed, panel_count)

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


css = """
#app {
  max-width: 1100px;
  margin: 0 auto;
}
.panel-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 12px;
}
.panel-card {
  border: 1px solid #d5d7de;
  border-radius: 12px;
  padding: 12px;
  background: linear-gradient(180deg, #fffaf1, #f2f7ff);
}
.bubble {
  margin: 8px 0;
  padding: 8px;
  border-radius: 10px;
  background: #ffffff;
  border: 1px dashed #9ba6b2;
}
"""

with gr.Blocks(css=css) as demo:
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
        inputs=[language_input, style_input, live_feed_input, panel_count],
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
    demo.launch()
