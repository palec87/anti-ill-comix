---
title: Anti Ill Comix
emoji: 📰
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 6.5.1
app_file: app.py
pinned: false
license: mit
short_description: Multilingual news-to-comic literacy practice for adults
---

# Anti Ill Comix

**Social post:** [LinkedIn](https://www.linkedin.com/posts/david-palecek-he-him-49478b21b_buildsmall-spaces-share-7472407166625386496-TVOq/?utm_source=share&utm_medium=member_desktop&rcm=ACoAADd3WPABP9izUbshdFS4XZ132_eodF2hpBU)

**Demo video**


Anti Ill Comix is a Gradio application that turns international news into simple comic-based reading and writing practice for adult learners. To make AI life-changing technology, technology-poor people should be given a chance. With global average literacy rate of 86% among adults, there is 756 million people getting further from modern life with every digital advance. Even in EU with the average literacy rate of 99%, that still means 3.5 million people.

I happen to know one such community in the south of Portugal, and wanted to test automatic learning content generation with the following objectives:

- To be relevant to adults by content based on current news via RSS feed (not implemented yet).
- Language adjusted by sources from national news.
- To be concise in a form of comics.
- Article summaries complexity skill level adjustable.
- Accompanied with fill in writing exercises.


The app fetches or loads a news article, simplifies it, turns it into 3-5 comic panels, generates panel images, overlays dialogue bubbles in the UI, and attaches one fill-in-the-blank exercise for each of the panels.

## What does it contain

- Select an output language: English, Portuguese, Spanish, French, or German.
- Select a difficulty level from A1 to B2.
- Select a visual style such as minimal, newspaper, watercolor, or retro.
- Generate a short comic strip from a live RSS article or deterministic demo content.
- Translate learner-facing generated content with a translation model.
- Overlay speech bubbles in HTML instead of baking text into images.
- Route each panel to a writing exercise with answer feedback.
- Keep a structured session JSON for replay, debugging, and validation.

## Pipeline

1. Fetch a source article from RSS or load a deterministic example from `examples/`.
2. Build a canonical session JSON with source, article, language, style, and UI state.
3. Generate simplified content with exercises in canonical English.
4. Normalize characters, panels, dialogue, bubbles, and exercises.
5. Translate learner-facing fields into the selected language.
6. Generate or fall back to comic panel images.
7. Render the comic strip, transcript, exercises, and trace/debug output.

## Models

Default text model:

```text
Qwen/Qwen2.5-7B-Instruct
```

Default image model for local/Spaces and serverless image generation:

```text
black-forest-labs/FLUX.1-schnell
```

Default translation model:

```text
facebook/nllb-200-distilled-600M
```

The deterministic example path remains available when model generation fails.

## Supported Languages

The UI has static dictionary translations for:

- `en` English
- `pt` Portuguese
- `es` Spanish
- `fr` French
- `de` German

Example session files are stored at:

```text
examples/en_demo.json
examples/pt_demo.json
examples/es_demo.json
examples/fr_demo.json
examples/de_demo.json
```

## Runtime Notes

For Hugging Face model/serverless paths, set:

```text
HF_TOKEN=your_hugging_face_token
```

Serverless API text and image generation is set to default, local or Spaces generation can be toggled from the app's advanced options. If live model generation fails, the app records the error in trace/debug output and falls back where possible.

The image prompt intentionally asks for scene art only. Speech text is rendered separately by the UI overlay so generated images should not contain readable words or speech bubbles.

## Local Development

Install dependencies:

```bash
uv sync
```

Run the app:

```bash
gradio app.py
```

Run tests:

```bash
uv run pytest
```


This repository currently ships a plain `requirements.txt` for Spaces.

## Repository Layout

```text
app.py                 Gradio UI and app wiring
assets/style.css       Newspaper-style UI styling
comic_gen/             Generation, validation, translation, and rendering helpers
examples/              Deterministic multilingual demo sessions
test/                  Pytest coverage
requirements.txt       Hugging Face Spaces dependencies
AGENTS.md              Agent/project guidance
```

Key modules:

- `comic_gen/comics.py`: Orchestrates text generation, translation, image generation, and fallback.
- `comic_gen/text_backend.py`: Builds the structured text-generation prompt and normalizes model output.
- `comic_gen/translation_backend.py`: Translates learner-facing content with NLLB.
- `comic_gen/image_backend.py`: Builds image prompts and handles FLUX image generation/fallback.
- `comic_gen/exercise.py`: Creates and evaluates fill-in-the-blank exercises.
- `comic_gen/models.py`: Validates the strict session JSON contract.

## Data Contract

Each session is one JSON document with stable IDs and replay-friendly fields:

- `source`
- `article`
- `simplified`
- `characters`
- `panels`
- `exercises`
- `ui`
- `trace`

The schema version is currently:

```text
1.0.0
```

## Known issues

1. Translation issues with special characters
2. Appearance of empty text bubbles too often in the image generation step, even with negative prompts.
3. Inconsistent translation fallback
4. Missing RSS due to uncertainty licensing resolution.
