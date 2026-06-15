import copy
import json

import pytest

from comic_gen.text_backend import (
    UnifiedGenerationError,
    _normalize_exercises,
    generate_text_content_from_article,
)
from comic_gen.text_utils import _normalize_panels


PAYLOAD = {
    "simplified": {
        "summary": "A city neighborhood opened a community garden where adults meet to grow vegetables, read guides, and write notes. This helps improve their reading confidence.",
        "level": "Beginner",
        "keywords": ["community garden", "adults", "vegetables", "reading", "confidence"],
    },
    "characters": [
        {
            "id": "001",
            "name": "Sarah",
            "description": "A friendly adult participating in the community garden program.",
        },
        {
            "id": "002",
            "name": "Tom",
            "description": "Another adult who enjoys learning new things at the community garden.",
        },
    ],
    "panels": [
        {
            "panel_id": "001",
            "frame_index": 1,
            "scene_description": "Sarah and Tom are planting seeds in a garden bed.",
            "dialogue": [
                {"character_id": "001", "text": "Let's plant some seeds!"},
                {"character_id": "002", "text": "Great idea, Sarah!"},
            ],
            "bubbles": [
                {"bbox_px": [100, 100, 200, 50]},
                {"bbox_px": [200, 100, 200, 50]},
            ],
            "render": {"image_path": "path/to/image1.png", "overlay_applied": True},
        },
        {
            "panel_id": "002",
            "frame_index": 2,
            "scene_description": "Sarah and Tom are reading a guide about planting vegetables.",
            "dialogue": [
                {"character_id": "001", "text": "Look at this guide!"},
                {"character_id": "002", "text": "It's very clear and easy to understand."},
            ],
            "bubbles": [
                {"bbox_px": [150, 200, 200, 50]},
                {"bbox_px": [350, 200, 200, 50]},
            ],
            "render": {"image_path": "path/to/image2.png", "overlay_applied": True},
        },
        {
            "panel_id": "003",
            "frame_index": 3,
            "scene_description": "Sarah and Tom are writing notes about what they learned.",
            "dialogue": [
                {"character_id": "001", "text": "I'm writing down what I learned."},
                {"character_id": "002", "text": "Me too! It helps me remember better."},
            ],
            "bubbles": [
                {"bbox_px": [100, 300, 200, 50]},
                {"bbox_px": [300, 300, 200, 50]},
            ],
            "render": {"image_path": "path/to/image3.png", "overlay_applied": True},
        },
    ],
    "exercises": [
        {
            "exercise_id": "ex_panel_1",
            "panel_id": "001",
            "prompt": "Sarah and Tom are planting seeds in a community garden plot. They are going to plant some ____ today.",
            "blanks": ["____"],
            "answer_key": ["seeds"],
            "feedback_rules": {
                "case_sensitive": False,
                "allow_trim_spaces": True,
            },
        },
        {
            "exercise_id": "ex_panel_2",
            "panel_id": "002",
            "prompt": "Sarah and Tom are reading a short guide about planting vegetables. They are going to read it ____.",
            "blanks": ["____"],
            "answer_key": ["together"],
            "feedback_rules": {
                "case_sensitive": False,
                "allow_trim_spaces": True,
            },
        },
        {
            "exercise_id": "ex_panel_3",
            "panel_id": "003",
            "prompt": "Sarah and Tom are writing notes about what they learned from the guide. They are going to write down what they ____.",
            "blanks": ["____"],
            "answer_key": ["learned"],
            "feedback_rules": {
                "case_sensitive": False,
                "allow_trim_spaces": True,
            },
        },
    ],
}


def test_normalize_exercises_rejects_scalar_fields_in_payload():
    raw_exercises = copy.deepcopy(PAYLOAD["exercises"])
    for item in raw_exercises:
        item["answer_key"] = item["answer_key"][0]
        item["feedback_rules"] = "incorrect-shape"

    with pytest.raises(UnifiedGenerationError, match="exercise answer_key missing"):
        _normalize_exercises(raw_exercises, PAYLOAD["panels"])


def test_normalize_exercises_accepts_normalized_exercise_shape():
    raw_exercises = copy.deepcopy(PAYLOAD["exercises"])

    normalized = _normalize_exercises(raw_exercises, PAYLOAD["panels"])

    assert normalized == [
        {
            "exercise_id": "ex_panel_1",
            "panel_id": "001",
            "prompt": "Sarah and Tom are planting seeds in a community garden plot. They are going to plant some ____ today.",
            "blanks": ["____"],
            "answer_key": ["seeds"],
            "feedback_rules": {
                "case_sensitive": False,
                "allow_trim_spaces": True,
            },
        },
        {
            "exercise_id": "ex_panel_2",
            "panel_id": "002",
            "prompt": "Sarah and Tom are reading a short guide about planting vegetables. They are going to read it ____.",
            "blanks": ["____"],
            "answer_key": ["together"],
            "feedback_rules": {
                "case_sensitive": False,
                "allow_trim_spaces": True,
            },
        },
        {
            "exercise_id": "ex_panel_3",
            "panel_id": "003",
            "prompt": "Sarah and Tom are writing notes about what they learned from the guide. They are going to write down what they ____.",
            "blanks": ["____"],
            "answer_key": ["learned"],
            "feedback_rules": {
                "case_sensitive": False,
                "allow_trim_spaces": True,
            },
        },
    ]


def test_generate_text_prompt_honors_panel_count_and_article(monkeypatch):
    captured = {}

    def _fake_generate_with_pipeline(
        prompt,
        model_repo_id,
        max_new_tokens,
        do_sample,
        temperature,
    ):
        captured["prompt"] = prompt
        panels = []
        exercises = []
        for index in range(1, 5):
            panel_id = f"panel_{index}"
            panels.append(
                {
                    "panel_id": panel_id,
                    "frame_index": index,
                    "scene_description": f"Scene {index}",
                    "dialogue": [
                        {
                            "character_id": "char_guide",
                            "text": f"Line {index}",
                        }
                    ],
                    "bubbles": [{"bbox_px": [30, 30, 300, 90]}],
                    "render": {
                        "image_path": f"assets/panel_{index}.png",
                        "overlay_applied": True,
                    },
                }
            )
            exercises.append(
                {
                    "exercise_id": f"ex_{panel_id}",
                    "panel_id": panel_id,
                    "prompt": "Line ____",
                    "blanks": ["____"],
                    "answer_key": [str(index)],
                    "feedback_rules": {
                        "case_sensitive": False,
                        "allow_trim_spaces": True,
                    },
                }
            )
        return json.dumps(
            {
                "simplified": {
                    "summary": "Short summary",
                    "level": "A2",
                    "keywords": ["reading"],
                },
                "characters": [
                    {
                        "id": "char_guide",
                        "name": "Guide",
                        "description": "Plain language mentor",
                    }
                ],
                "panels": panels,
                "exercises": exercises,
            }
        )

    monkeypatch.setattr(
        "comic_gen.text_backend._generate_with_pipeline",
        _fake_generate_with_pipeline,
    )

    generated = generate_text_content_from_article(
        language="en",
        style_id="retro",
        reading_level="B1",
        article={
            "title": "Garden news",
            "fulltext": "Adults read a local garden article together.",
        },
        panel_count=4,
        model_repo_id="test-model",
    )

    assert "- Generate exactly 4 panels." in captured["prompt"]
    assert "Target Language Code: en" in captured["prompt"]
    assert "Comic Style ID: retro" in captured["prompt"]
    assert "Reading Level: B1" in captured["prompt"]
    assert "Set simplified.level exactly to B1." in captured["prompt"]
    assert "article_title=Garden news" in captured["prompt"]
    assert "Adults read a local garden article" in captured["prompt"]
    assert generated["simplified"]["level"] == "B1"
    assert len(generated["panels"]) == 4
    assert len(generated["exercises"]) == 4


def test_model_payload_with_empty_bubbles_and_answer_blanks_is_repaired(
    monkeypatch,
):
    payload = {
        "simplified": {
            "summary": (
                "Artists are creating imperfect art to oppose AI's perfect "
                "images."
            ),
            "level": "Intermediate",
            "keywords": ["Artists", "AI", "Imperfect"],
        },
        "characters": [
            {
                "id": "A1",
                "name": "Rob Wrubel",
                "description": "Co-founder and managing partner at Silverside.",
            }
        ],
        "panels": [
            {
                "panel_id": "P1",
                "frame_index": 1,
                "scene_description": "Rob Wrubel speaking at an event.",
                "dialogue": [
                    {
                        "character_id": "A1",
                        "text": (
                            "What's incredible about AI is that you can go "
                            "from script to production in just two weeks!"
                        ),
                    }
                ],
                "bubbles": [],
                "render": {
                    "image_path": "event_speech.jpg",
                    "overlay_applied": True,
                },
            },
            {
                "panel_id": "P2",
                "frame_index": 2,
                "scene_description": (
                    "An AI-generated ad with computerized polar bears and "
                    "fake-looking trucks."
                ),
                "dialogue": [
                    {"character_id": "A1", "text": "The ad was widely despised."}
                ],
                "bubbles": [],
                "render": {"image_path": "ai_ad.jpg", "overlay_applied": True},
            },
            {
                "panel_id": "P3",
                "frame_index": 3,
                "scene_description": (
                    "Rob Wrubel admitting the backlash importance."
                ),
                "dialogue": [
                    {
                        "character_id": "A1",
                        "text": (
                            "The conversation around the ad became almost as "
                            "important as the ad itself."
                        ),
                    }
                ],
                "bubbles": [],
                "render": {
                    "image_path": "ad_backlash.jpg",
                    "overlay_applied": True,
                },
            },
        ],
        "exercises": [
            {
                "exercise_id": "E1",
                "panel_id": "P1",
                "prompt": (
                    "Rob Wrubel says that AI can make ads in just _______ "
                    "weeks."
                ),
                "blanks": ["two"],
                "answer_key": ["two"],
                "feedback_rules": {
                    "case_sensitive": False,
                    "allow_trim_spaces": True,
                },
            },
            {
                "exercise_id": "E2",
                "panel_id": "P2",
                "prompt": "The AI-generated ad was _______ by people.",
                "blanks": ["despised"],
                "answer_key": ["despised"],
                "feedback_rules": {
                    "case_sensitive": False,
                    "allow_trim_spaces": True,
                },
            },
            {
                "exercise_id": "E3",
                "panel_id": "P3",
                "prompt": (
                    "Rob Wrubel admitted that the _______ around the ad was "
                    "important."
                ),
                "blanks": ["conversation"],
                "answer_key": ["conversation"],
                "feedback_rules": {
                    "case_sensitive": False,
                    "allow_trim_spaces": True,
                },
            },
        ],
    }

    monkeypatch.setattr(
        "comic_gen.text_backend._generate_with_pipeline",
        lambda *args, **kwargs: json.dumps(payload),
    )

    generated = generate_text_content_from_article(
        language="en",
        style_id="minimal",
        reading_level="A1",
        article={"title": "AI art", "fulltext": "Article text"},
        panel_count=3,
        model_repo_id="test-model",
    )

    assert [panel["panel_id"] for panel in generated["panels"]] == [
        "P1",
        "P2",
        "P3",
    ]
    assert all(panel["bubbles"] for panel in generated["panels"])
    assert generated["simplified"]["level"] == "A1"
    assert generated["exercises"][0]["panel_id"] == "P1"
    assert generated["exercises"][0]["blanks"] == ["____"]
    assert generated["exercises"][0]["answer_key"] == ["two"]
    assert set(generated["_normalization_repairs"]) == {
        "panel bubbles",
        "exercise blanks",
    }


def test_normalize_panels_creates_compact_upper_corner_bubbles():
    panels = _normalize_panels(
        [
            {
                "panel_id": "P1",
                "frame_index": 1,
                "scene_description": "Two adults read a note.",
                "dialogue": [
                    {"character_id": "A1", "text": "Read this first."},
                    {"character_id": "A2", "text": "I understand now."},
                ],
                "bubbles": [],
            },
            {
                "panel_id": "P2",
                "frame_index": 2,
                "scene_description": "A worker points to a form.",
                "dialogue": [{"character_id": "A1", "text": "Write here."}],
                "bubbles": [],
            },
            {
                "panel_id": "P3",
                "frame_index": 3,
                "scene_description": "People smile together.",
                "dialogue": [{"character_id": "A1", "text": "Good work."}],
                "bubbles": [],
            },
        ],
        3,
    )

    assert panels[0]["bubbles"] == [
        {"bbox_px": [10, 10, 108, 30]},
        {"bbox_px": [138, 10, 108, 30]},
    ]
    assert panels[1]["bubbles"] == [{"bbox_px": [10, 10, 118, 30]}]
    for bubble in panels[0]["bubbles"]:
        x, y, width, height = bubble["bbox_px"]
        assert x >= 0
        assert y >= 0
        assert x + width <= 256
        assert y + height <= 256


def test_normalize_panels_repairs_bad_and_overlapping_boxes():
    panels = _normalize_panels(
        [
            {
                "panel_id": "P1",
                "frame_index": 1,
                "scene_description": "A reader asks a question.",
                "dialogue": [
                    {"character_id": "A1", "text": "Where is the address?"},
                    {"character_id": "A2", "text": "It is at the top."},
                    {"character_id": "A1", "text": "Thank you."},
                ],
                "bubbles": [
                    {"bbox_px": [250, -20, 10, 10]},
                    {"bbox_px": [252, -10, 8, 8]},
                    {"bbox_px": [254, -5, 6, 6]},
                ],
            },
            {
                "panel_id": "P2",
                "frame_index": 2,
                "scene_description": "People review a paper.",
                "dialogue": [{"character_id": "A1", "text": "Check line one."}],
                "bubbles": [],
            },
            {
                "panel_id": "P3",
                "frame_index": 3,
                "scene_description": "A person writes slowly.",
                "dialogue": [{"character_id": "A1", "text": "I can do it."}],
                "bubbles": [],
            },
        ],
        3,
    )

    assert panels[0]["bubbles"] == [
        {"bbox_px": [10, 10, 108, 28]},
        {"bbox_px": [138, 10, 108, 28]},
        {"bbox_px": [10, 206, 108, 28]},
    ]


def test_normalize_panels_preserves_small_valid_model_box():
    panels = _normalize_panels(
        [
            {
                "panel_id": "P1",
                "frame_index": 1,
                "scene_description": "A reader studies a short note.",
                "dialogue": [{"character_id": "A1", "text": "Short note."}],
                "bubbles": [{"bbox_px": [24, 18, 88, 26]}],
            },
            {
                "panel_id": "P2",
                "frame_index": 2,
                "scene_description": "A learner checks a form.",
                "dialogue": [{"character_id": "A1", "text": "Looks clear."}],
                "bubbles": [],
            },
            {
                "panel_id": "P3",
                "frame_index": 3,
                "scene_description": "The group completes the task.",
                "dialogue": [{"character_id": "A1", "text": "Done."}],
                "bubbles": [],
            },
        ],
        3,
    )

    assert panels[0]["bubbles"] == [{"bbox_px": [24, 18, 88, 26]}]
