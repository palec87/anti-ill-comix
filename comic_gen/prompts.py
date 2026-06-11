SIMPLIFICATION_PROMPT = (
    "Rewrite this article in plain language for adult learners. "
    "Keep key facts, short sentences, and practical vocabulary."
)

DIALOGUE_PROMPT = (
    "Convert summary into short character dialogue for 3-5 comic panels. "
    "Each speech line should fit a small bubble."
)

UNIFIED_SESSION_PROMPT = (
    "### Role and Objective\n"
    "You are an expert curriculum developer. Generate educational comic book content "
    "tailored for adults with low literacy skills. Keep the vocabulary accessible but "
    "age-appropriate (avoid childish topics).\n\n"
    
    "### Constraints\n"
    "- Return ONLY a valid JSON object. Do not wrap it in markdown code blocks or ```json fences.\n"
    "- Do NOT output any conversational text before or after the JSON structure.\n"
    "- Use strict double quotes (\") for all keys and string values.\n"
    "- Maintain exact consistency: the 'exercises' list length must exactly match the 'panels' list length.\n\n"
    
    "### Context Parameters\n"
    "- Target Language Code: {language_code}\n"
    "- Comic Style ID: {style_id}\n\n"
    
    "### Target JSON Schema Definition\n"
    "Your output must follow this JSON structure exactly:\n"
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

# UNIFIED_SESSION_PROMPT = (
#     "You are generating educational comic JSON content for adults with low "
#     "literacy. Return ONLY valid JSON object with keys: simplified, "
#     "characters, panels, exercises. Do not output markdown, code fences, "
#     "or any text before/after the JSON. Use double quotes for all keys and "
#     "string values. "
#     "Use input language code and style_id. "
#     "Schema: simplified={summary:string, level:string, keywords:string[]}; "
#     "characters=[{id,name,description}] with 2-3 entries; "
#     "panels=[{panel_id, frame_index, scene_description, dialogue, bubbles, "
#     "render}] with 3-5 entries where dialogue items are {character_id,text}, "
#     "bubbles items are {bbox_px:[x,y,w,h]} integers inside 512x512, and "
#     "render includes image_path and overlay_applied boolean; "
#     "exercises=[{exercise_id,panel_id,prompt,blanks,answer_key,"
#     "feedback_rules}] "
#     "example of the exercise output format: "
#     '{"exercise_id":"ex_panel_1","panel_id":"panel_1",'
#     '"prompt":"We read one short ____ together.",'
#     '"blanks":["____"],"answer_key":["instruction"],'
#     '"feedback_rules":{"case_sensitive":false,'
#     '"allow_trim_spaces":true}}'
#     "same length as panels. Keep text simple and adult age-appropriate."
# )
