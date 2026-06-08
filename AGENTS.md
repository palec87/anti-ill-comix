# Agent instruction for anti-ill-comics repository

This repository builds a multilingual literacy assistant that turns international news into simple comic-based learning and writing practice for adults. It is deployed on hugging face spaces using a Gradio framework.

## Product Goals
Increase reading comprehension, reduce text complexity, improve guided writing skills, and keep content culturally and linguistically relevant. The application should be as simple as possible with clean visual look and focus on the comic strip 

## End-to-End Workflow

Step 1: Fetch one recent article from an international source based on selected language via RSS feed or an API call. The output must be a json structure with mandatory fields of 'source_link', 'fulltext', 'summary', 'language', 'session_id' etc.. These can be expanded if necessary. Provide json file template for hardcoded examples.
Step 2: Produce a simplified summary in learner-friendly language which expands the news json file which is tagged by the chosen artistic style. In addition, convert the summary into per person dialogues (plus narrator if necessary) tagged by frame number and character id. The input is the json from previous step, output is enriched json.
Step 3: based on the summarized text and the dialogue, create character descriptions as a prompt for the comic script, again added to the json under 'characters' field.
Step 4: From the json, generate 3-5 frames of a comic script with dialogues and bounding boxes in pixels per frame of the comic strip for bubbles based on the characters lines.
Step 5: Create an overlay layer from the image and text fields filling the bubbles.
Step 6: Attach a clickable writing exercise to each panel using fill-in-the-blank on central vocabulary.
Step 7: Let users choose output language and comic art style at the top of the application. Use flags and emojis instead of words.

## Strict Data Contract (Define Now)
Define and enforce schema now to avoid integration failures between agents.

Why now:
- Multiple agents pass data to each other. Without a strict contract, each step can silently break the next step.
- UI features (clickable panels and exercise routing) need stable IDs and panel indexes from the start.
- Re-render and replay require deterministic, complete JSON records per session.
- Early schema validation reduces debugging cost and makes test coverage meaningful.

Rules:
- Use one canonical JSON document per session.
- Additive changes only: new fields are optional by default, existing required fields must stay backward compatible.
- Every step must validate input schema and output schema before handoff.
- If validation fails, return deterministic fallback content and an error entry in trace.

Minimum required top-level fields:
- session_id (string)
- language (string, BCP-47 or simple language code)
- style_id (string)
- source (object)
- article (object)
- simplified (object)
- characters (array)
- panels (array, length 3-5)
- exercises (array, same length as panels)
- trace (array)

Step output contracts:
- Step 1 output must include: source.link, source.publisher, source.published_at, article.title, article.fulltext.
- Step 2 output must include: simplified.summary, simplified.level, simplified.keywords.
- Step 3 output must include: characters[].id, characters[].name, characters[].description.
- Step 4 output must include per panel: panel_id, frame_index, scene_description, dialogue[].character_id, dialogue[].text, bubbles[].bbox_px.
- Step 5 output must include per panel: render.image_path or render.image_bytes_ref, render.overlay_applied.
- Step 6 output must include per panel: exercise_id, prompt, blanks[], answer_key[], feedback_rules.
- Step 7 output must include: ui.language_label, ui.style_label, ui.selector_state.

Validation gates:
- Gate A (after Step 1): reject empty fulltext and missing source link.
- Gate B (after Step 2): summary length and readability target must be met.
- Gate C (after Step 4): 3-5 panels required and all bubble boxes must be inside panel bounds.
- Gate D (after Step 6): one exercise per panel and all blanks map to answer keys.
- Gate E (pre-UI): panel count equals exercise count and all IDs are unique.

Versioning:
- Include schema_version at top level.
- Increment minor version for additive changes, major version for breaking changes.
- Keep one deterministic example JSON per supported language in repository examples.

## Agent Responsibilities
Content Intake Agent: source selection, article fetch, metadata, attribution.
Simplification Agent: summarize, simplify, and enforce readability target.
Comic Script Agent: create panel-by-panel dialogue and scene prompts.
Comic generation Agent: generate 3-5 images for a comic with bounding boxes for text fields
Exercise Agent: generate gap-fill tasks and answer keys with difficulty control according to A1, A2, B1 etc..
UI Agent: render comic panels, Merge image layer with text, click-through exercises, language/style selectors.
QA/Evaluation Agent: validate factual consistency, readability, position of the text in respect to the images, safety, and exercise quality.

## Quality and Safety Rules
Use minimum amount of text to make the application accessible to low literacy audience.

Use age-appropriate neutral language, avoid harmful content, keep summaries factual, preserve key context, and include source attribution with a html link under the comic strip.

Check carefully overlay between the images and the texts in the comics.

## Engineering priorities
Work in small steps. Each commit should leave the app runnable.

Prefer the smallest change that makes the current milestone work.

Use uv for local development. Keep requirements.txt generated for Hugging Face Spaces.

The deterministic example based on pregenerated images, texts, and exercises should always work in all the languages, one example per language. Any model backend added later should be optional and should fall back gracefully if loading or generation fails.

Keep resolution of images small, adjust character styles for low details.

Prefer a simple Gradio Blocks app until a task explicitly asks for a custom frontend or gr.Server.

Do not introduce Docker, fine-tuning, authentication, persistent storage, or heavy runtime changes unless the current task explicitly asks for them.

### Non-Functional Requirements
Support multilingual pipeline, accessibility-first UI, and a low-resource mode for CPU-only environments.

### Definition of Done
A user can select language and style, generate article-to-comic output, click each panel into an exercise, submit answers, and receive feedback in the same language.

## Advanced priorities
Add option to insert own text to convert to the educational comic strip

Store generated content including the transcript and exercise for full replication of the comic strip later, language and styles need to be stored as well. Keep as much information in a json format.

This re-rendering should be added to the top of the application as a drop-down menu

## Out of Scope
Real-time newsroom coverage guarantees, unrestricted scraping, and model-specific hard dependencies in policy text.

## Suggested repo structure

app.py
pyproject.toml
uv.lock
requirements.txt
README.md
AGENTS.md

comic_gen/
  __init__.py
  models.py
  session.py
  comics.py
  exercise.py
  trace.py
  backends.py
  prompts.py

assets/
  stage.css

This structure is a suggestion, not a strict rule. Keep the repo easy to understand.

## Local commands
activate venv
```
.venv/Scripts/activate
```

install dependencies:
```
uv sync
```

Check syntax:
```
uv run python -m py_compile app.py comic_gen/*.py
```

Generate Hugging Face Space requirements:
```
uv pip compile pyproject.toml -o requirements.txt
```

If tests are added, run:
```
uv run pytest
```


## Coding style

Keep modules small and readable.

Use dataclasses or Pydantic models for core state. Avoid passing around loose dictionaries once the main app structure is in place.

Avoid secrets in code. Do not hard-code tokens, private URLs, or machine-specific paths.

Do not hide important behavior in global side effects. Lazy-loaded model backends are okay when needed, but the deterministic path should stay simple.

Do not introduce login or authentication unless explicitly asked.

Do not add heavy model dependencies unless the task explicitly asks for them.


## UI guidance

Use Gradio Blocks as the default UI framework for now.

The page should roughly flow like this:

    Title and short description
    language and style selectors
    Strip Generation button
    Frame where the comics will be generated
    Exercise links per image
    Transcript
    Trace/debug information
    Backend/model settings, if present

The selectors, comics and exercise buttons should fit a normal laptop screen without lots of scrolling.

Check that:

    The comics part is visually dominant
    Buttons are easy to find
    The transcript of the comics is updated
    Exercises update
    Minimum text is used

## Logging

Log events sequentially. In debug mode, always point to the method which is being called
- application loaded
- log buttons triggered actions
- log every action completion
- Log completion of a task by the agent
- log json with the transcript and settings
- log overall user score in the exercises for the comic strip

## Commit expectations

Each commit should represent a working milestone and leave the app runnable.

Use conventional, readable commit messages, for example:

    fix: add uv setup
    feat: add new artistic style
    feat: implemented exercises by difficulty.
    docs: update space readme

Before finishing a task, summarize what changed and which validation command passed.