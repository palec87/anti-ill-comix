import copy
import json

import pytest

from comic_gen.text_backend import (
    UnifiedGenerationError,
    _normalize_exercises,
    generate_text_content_from_article,
)


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
    assert "article_title=Garden news" in captured["prompt"]
    assert "Adults read a local garden article" in captured["prompt"]
    assert len(generated["panels"]) == 4
    assert len(generated["exercises"]) == 4
