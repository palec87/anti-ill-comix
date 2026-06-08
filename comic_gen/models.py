from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


class ValidationError(Exception):
    """Raised when the session JSON does not match the strict contract."""


def new_session_id() -> str:
    return f"sess-{uuid4().hex[:12]}"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_field(obj: dict[str, Any], key: str, expected_type: type) -> None:
    if key not in obj:
        raise ValidationError(f"Missing required field: {key}")
    if not isinstance(obj[key], expected_type):
        raise ValidationError(f"Field {key} must be {expected_type.__name__}")


def _validate_top_level(document: dict[str, Any]) -> None:
    required_fields: list[tuple[str, type]] = [
        ("schema_version", str),
        ("session_id", str),
        ("language", str),
        ("style_id", str),
        ("source", dict),
        ("article", dict),
        ("simplified", dict),
        ("characters", list),
        ("panels", list),
        ("exercises", list),
        ("trace", list),
    ]
    for key, typ in required_fields:
        _require_field(document, key, typ)


def _validate_step_contracts(document: dict[str, Any]) -> None:
    source = document["source"]
    article = document["article"]
    simplified = document["simplified"]
    characters = document["characters"]
    panels = document["panels"]
    exercises = document["exercises"]

    for key in ("link", "publisher", "published_at"):
        _require_field(source, key, str)
    for key in ("title", "fulltext"):
        _require_field(article, key, str)
    for key in ("summary", "level", "keywords"):
        if key == "keywords":
            _require_field(simplified, key, list)
        else:
            _require_field(simplified, key, str)

    if not isinstance(characters, list):
        raise ValidationError("characters must be an array")
    for idx, c in enumerate(characters):
        if not isinstance(c, dict):
            raise ValidationError(f"characters[{idx}] must be an object")
        for key in ("id", "name", "description"):
            _require_field(c, key, str)

    if not (3 <= len(panels) <= 5):
        raise ValidationError("panels must have length between 3 and 5")

    seen_panel_ids: set[str] = set()
    for idx, p in enumerate(panels):
        if not isinstance(p, dict):
            raise ValidationError(f"panels[{idx}] must be an object")
        for key, typ in (
            ("panel_id", str),
            ("frame_index", int),
            ("scene_description", str),
            ("dialogue", list),
            ("bubbles", list),
            ("render", dict),
        ):
            _require_field(p, key, typ)

        if p["panel_id"] in seen_panel_ids:
            raise ValidationError(f"Duplicate panel_id: {p['panel_id']}")
        seen_panel_ids.add(p["panel_id"])

        for d_idx, d in enumerate(p["dialogue"]):
            if not isinstance(d, dict):
                raise ValidationError(
                    f"panels[{idx}].dialogue[{d_idx}] must be object"
                )
            for key in ("character_id", "text"):
                _require_field(d, key, str)

        for b_idx, b in enumerate(p["bubbles"]):
            if not isinstance(b, dict):
                raise ValidationError(
                    f"panels[{idx}].bubbles[{b_idx}] must be object"
                )
            _require_field(b, "bbox_px", list)
            bbox = b["bbox_px"]
            if len(bbox) != 4 or any(
                (not isinstance(v, int) or v < 0) for v in bbox
            ):
                raise ValidationError(
                    (
                        f"panels[{idx}].bubbles[{b_idx}].bbox_px "
                        "must be [x,y,w,h] ints >= 0"
                    )
                )

        render = p["render"]
        _require_field(render, "overlay_applied", bool)
        if "image_path" not in render and "image_bytes_ref" not in render:
            raise ValidationError(
                f"panels[{idx}].render needs image_path or image_bytes_ref"
            )

    if len(exercises) != len(panels):
        raise ValidationError("exercises length must match panels length")

    seen_exercise_ids: set[str] = set()
    for idx, ex in enumerate(exercises):
        if not isinstance(ex, dict):
            raise ValidationError(f"exercises[{idx}] must be an object")
        for key, typ in (
            ("exercise_id", str),
            ("panel_id", str),
            ("prompt", str),
            ("blanks", list),
            ("answer_key", list),
            ("feedback_rules", dict),
        ):
            _require_field(ex, key, typ)

        if ex["exercise_id"] in seen_exercise_ids:
            raise ValidationError(
                f"Duplicate exercise_id: {ex['exercise_id']}"
            )
        seen_exercise_ids.add(ex["exercise_id"])

        if len(ex["blanks"]) != len(ex["answer_key"]):
            raise ValidationError(
                f"exercises[{idx}] blanks length must match answer_key length"
            )

        if ex["panel_id"] not in seen_panel_ids:
            raise ValidationError(
                (
                    f"exercises[{idx}] references unknown panel_id "
                    f"{ex['panel_id']}"
                )
            )


def validate_session_document(document: dict[str, Any]) -> None:
    _validate_top_level(document)
    _validate_step_contracts(document)
