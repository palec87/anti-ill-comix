from __future__ import annotations

from app import (
    READING_LEVEL_OPTIONS,
    _overlay_bubbles_html,
    _panel_choices,
    _panel_image_html,
    load_exercise,
)


def _document() -> dict:
    return {
        "panels": [
            {
                "panel_id": "P1",
                "frame_index": 1,
                "dialogue": [],
                "bubbles": [],
                "render": {"overlay_applied": True, "image_path": "x.png"},
            },
            {
                "panel_id": "P2",
                "frame_index": 2,
                "dialogue": [],
                "bubbles": [],
                "render": {"overlay_applied": True, "image_path": "y.png"},
            },
        ],
        "exercises": [
            {
                "exercise_id": "E1",
                "panel_id": "P1",
                "prompt": "Rob Wrubel says AI can make ads in ____ weeks.",
                "blanks": ["____"],
                "answer_key": ["two"],
                "feedback_rules": {
                    "case_sensitive": False,
                    "allow_trim_spaces": True,
                },
            }
        ],
    }


def test_panel_choices_use_canonical_panel_ids():
    assert _panel_choices(_document()) == [
        ("Panel 1", "P1"),
        ("Panel 2", "P2"),
    ]


def test_reading_level_options_are_a1_to_b2():
    assert READING_LEVEL_OPTIONS == ["A1", "A2", "B1", "B2"]


def test_load_exercise_uses_canonical_panel_id():
    prompt, answer = load_exercise("P1", _document())

    assert "Rob Wrubel says" in prompt
    assert answer == ""


def test_load_exercise_supports_legacy_frame_panel_id():
    prompt, answer = load_exercise("panel_1", _document())

    assert "Rob Wrubel says" in prompt
    assert answer == ""


def test_overlay_bubbles_html_uses_inline_styles_and_escapes_text():
    panel = {
        "panel_id": "P1",
        "frame_index": 1,
        "dialogue": [
            {
                "character_id": "A1",
                "text": "Use <simple> words & clear steps.",
            }
        ],
        "bubbles": [{"bbox_px": [10, 10, 108, 30]}],
        "render": {"overlay_applied": True, "image_path": "missing.png"},
    }

    html = _overlay_bubbles_html(panel)

    assert "position:absolute;inset:0;z-index:4" in html
    assert "pointer-events:none" in html
    assert "left:3.91%;top:3.91%" in html
    assert "max-width:42.19%;max-height:11.72%" in html
    assert "width:max-content;height:auto" in html
    assert "padding:2px 5px" in html
    assert "font-size:clamp(7px,1.8vw,11px)" in html
    assert "min-height" not in html
    assert "Use &lt;simple&gt; words &amp; clear steps." in html


def test_panel_image_html_contains_overlay_inside_media_container():
    overlay = "<div class='overlay-layer'>Overlay</div>"
    panel = {
        "panel_id": "P1",
        "frame_index": 1,
        "dialogue": [],
        "bubbles": [],
        "render": {"overlay_applied": True, "image_path": "missing.png"},
    }

    html = _panel_image_html(panel, overlay)

    assert "class='panel-media'" in html
    assert "style='position:relative" in html
    assert html.index("<img") < html.index(overlay) < html.index("</div>")
