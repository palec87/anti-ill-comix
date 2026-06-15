from __future__ import annotations

import pytest

from comic_gen.translation_backend import (
    NLLB_LANGUAGE_CODES,
    _get_translation_pipeline,
    translate_session_content,
    translate_text,
)


def test_translate_text_noops_for_english():
    assert translate_text("Hello", "en") == "Hello"


def test_translate_text_uses_nllb_codes(monkeypatch):
    captured = {}

    def _fake_get_pipeline(model_id, source_language, target_language):
        captured["model_id"] = model_id
        captured["source_language"] = source_language
        captured["target_language"] = target_language
        return lambda text: [{"translation_text": f"ES:{text}"}]

    monkeypatch.setattr(
        "comic_gen.translation_backend._get_translation_pipeline",
        _fake_get_pipeline,
    )

    translated = translate_text("Hello", "es")

    assert translated == "ES:Hello"
    assert captured["source_language"] == NLLB_LANGUAGE_CODES["en"]
    assert captured["target_language"] == NLLB_LANGUAGE_CODES["es"]


def test_translate_text_handles_non_ascii_source_without_logging_failure(
    monkeypatch,
):
    def _fake_get_pipeline(model_id, source_language, target_language):
        return lambda text: [{"translation_text": f"OK:{text}"}]

    monkeypatch.setattr(
        "comic_gen.translation_backend._get_translation_pipeline",
        _fake_get_pipeline,
    )

    assert translate_text("Zażółć gęślą jaźń 世界", "de").startswith("OK:")


def test_get_translation_pipeline_uses_seq2seq_model(monkeypatch):
    calls = {}

    class FakeTokenizer:
        src_lang = ""

        def __call__(self, text, **kwargs):
            calls["text"] = text
            calls["tokenizer_kwargs"] = kwargs
            return {"input_ids": [[1, 2, 3]]}

        def convert_tokens_to_ids(self, token):
            calls["target_language"] = token
            return 42

        def batch_decode(self, generated, **kwargs):
            calls["decode_kwargs"] = kwargs
            return ["Hola"]

    class FakeModel:
        def generate(self, **kwargs):
            calls["generate_kwargs"] = kwargs
            return [[4, 5, 6]]

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls["tokenizer_model_id"] = model_id
            calls["tokenizer_load_kwargs"] = kwargs
            return FakeTokenizer()

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(model_id):
            calls["model_id"] = model_id
            return FakeModel()

    monkeypatch.setattr(
        "transformers.AutoTokenizer",
        FakeAutoTokenizer,
    )
    monkeypatch.setattr(
        "transformers.AutoModelForSeq2SeqLM",
        FakeAutoModel,
    )
    monkeypatch.setattr("comic_gen.translation_backend._PIPELINES", {})

    translator = _get_translation_pipeline("test-model", "eng_Latn", "spa_Latn")

    assert translator("Hello") == [{"translation_text": "Hola"}]
    assert calls["tokenizer_load_kwargs"] == {"src_lang": "eng_Latn"}
    assert calls["target_language"] == "spa_Latn"
    assert calls["generate_kwargs"]["forced_bos_token_id"] == 42
    assert calls["decode_kwargs"] == {"skip_special_tokens": True}


def test_translate_text_preserves_blank_placeholder(monkeypatch):
    monkeypatch.setattr(
        "comic_gen.translation_backend._get_translation_pipeline",
        lambda *args, **kwargs: (
            lambda text: [{"translation_text": text.replace("Read", "Leer")}]
        ),
    )

    assert translate_text("Read ____ today", "es", preserve_blanks=True) == (
        "Leer ____ today"
    )


def test_translate_text_raises_on_failed_translation(monkeypatch):
    def _broken_pipeline(*args, **kwargs):
        return lambda text: (_ for _ in ()).throw(RuntimeError("boom"))

    monkeypatch.setattr(
        "comic_gen.translation_backend._get_translation_pipeline",
        _broken_pipeline,
    )

    with pytest.raises(RuntimeError, match="boom"):
        translate_text("Hello", "es")


def test_translate_text_raises_on_unsupported_language():
    with pytest.raises(ValueError, match="Unsupported target language"):
        translate_text("Hello", "it")


def test_translate_session_content_translates_learner_fields(monkeypatch):
    monkeypatch.setattr(
        "comic_gen.translation_backend.translate_text",
        lambda text, target_language, **kwargs: f"{target_language}:{text}",
    )
    document = {
        "simplified": {"summary": "Summary", "keywords": ["word"]},
        "characters": [{"description": "A guide"}],
        "panels": [
            {
                "scene_description": "A scene",
                "dialogue": [{"text": "Hello"}],
            }
        ],
        "exercises": [{"prompt": "Hello ____"}],
        "trace": [],
    }

    applied = translate_session_content(document, "fr")

    assert applied is True
    assert document["simplified"]["summary"] == "fr:Summary"
    assert document["simplified"]["keywords"] == ["fr:word"]
    assert document["characters"][0]["description"] == "fr:A guide"
    assert document["panels"][0]["scene_description"] == "fr:A scene"
    assert document["panels"][0]["dialogue"][0]["text"] == "fr:Hello"
    assert document["exercises"][0]["prompt"] == "fr:Hello ____"
    assert document["trace"][-1]["status"] == "ok"
