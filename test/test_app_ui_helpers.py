from __future__ import annotations

from app import _panel_choices, load_exercise


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


def test_load_exercise_uses_canonical_panel_id():
    prompt, answer = load_exercise("P1", _document())

    assert "Rob Wrubel says" in prompt
    assert answer == ""


def test_load_exercise_supports_legacy_frame_panel_id():
    prompt, answer = load_exercise("panel_1", _document())

    assert "Rob Wrubel says" in prompt
    assert answer == ""
