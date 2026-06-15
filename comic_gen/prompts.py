SIMPLIFICATION_PROMPT = (
    "Rewrite this article in plain language for adult learners. "
    "Keep key facts, short sentences, and practical vocabulary."
)

DIALOGUE_PROMPT = (
    "Convert summary into short character dialogue for 3 comic panels. "
    "Each speech line should fit a small bubble."
)

UNIFIED_SESSION_PROMPT = (
    "### Role and Objective\n"
    "Generate educational comic book content based on the 'article_fulltext' from user_input_source_material"
    "tailored for adults with low literacy skills. Keep the vocabulary accessible but "
    "age-appropriate (avoid childish topics).\n\n"
    
    "### Constraints\n"
    "- Return ONLY a valid JSON object. Do not wrap it in markdown code blocks or ```json fences.\n"
    "- Do NOT output any conversational text before or after the JSON structure.\n"
    "- Use strict double quotes (\") for all keys and string values.\n"
    "- Generate exactly {panel_count} panels.\n"
    "- Set simplified.level exactly to {reading_level}.\n"
    "- Length of the 'panels' list must match the length of the 'exercises' list.\n"
    "- For each panel_id, there needs to be exactly one exercise in the 'exercises' list.\n"
    "- There can be 1-3 characters.\n\n"
    
    "### Context Parameters\n"
    "- Target Language Code: {language}\n"
    "- Comic Style ID: {style_id}\n\n"
    "- Reading Level: {reading_level}\n\n"

    "### Reading Level Guidance\n"
    "- A1: very short sentences, everyday words, one idea per sentence.\n"
    "- A2: short sentences, common vocabulary, simple connectors.\n"
    "- B1: clear plain language with some detail and cause/effect.\n"
    "- B2: adult plain language with fuller context, still concise.\n\n"
    
    "### Target JSON Schema Definition\n"
    "Your output must follow this JSON structure:\n"
    "{\n"
    '  "simplified": {\n'
    '    "summary": "String",\n'
    '    "level": "String",\n'
    '    "keywords": ["String", "String"]\n'
    "  },\n"
    '  "characters": [\n'
    '    {"id": "String", "name": "String", "description": "String"}\n'
    "  ],\n"
    '  "panels": [\n'
    "    {\n"
    '      "panel_id": "String",\n'
    '      "frame_index": Integer,\n'
    '      "scene_description": "String",\n'
    '      "dialogue": [{"character_id": "String", "text": "String"}],\n'
    '      "bubbles": [{"bbox_px": [X, Y, W, H]}],\n'
    '      "render": {"image_path": "String", "overlay_applied": Boolean}\n'
    "    }\n"
    "  ],\n"
    '  "exercises": [\n'
    "    {\n"
    '      "exercise_id": "String",\n'
    '      "panel_id": "String",\n'
    '      "prompt": "String using ____ placeholder syntax",\n'
    '      "blanks": ["String"],\n'
    '      "answer_key": ["String"],\n'
    '      "feedback_rules": {"case_sensitive": false, "allow_trim_spaces": true}\n'
    "    }\n"
    "  ]\n"
    "}\n\n"

    "### Input Content Source\n"
    "Generate the JSON data based on the following input material:\n"
    "{user_input_source_material}"
)
