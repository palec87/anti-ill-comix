SIMPLIFICATION_PROMPT = (
    "Rewrite this article in plain language for adult learners. "
    "Keep key facts, short sentences, and practical vocabulary."
)

DIALOGUE_PROMPT = (
    "Convert summary into short character dialogue for 3-5 comic panels. "
    "Each speech line should fit a small bubble."
)

UNIFIED_SESSION_PROMPT = (
    "You are generating educational comic JSON content for adults with low "
    "literacy. Return ONLY valid JSON object with keys: simplified, "
    "characters, panels, exercises. Do not output markdown, code fences, "
    "or any text before/after the JSON. Use double quotes for all keys and "
    "string values. "
    "Use input language code and style_id. "
    "Schema: simplified={summary:string, level:string, keywords:string[]}; "
    "characters=[{id,name,description}] with 2-3 entries; "
    "panels=[{panel_id, frame_index, scene_description, dialogue, bubbles, "
    "render}] with 3-5 entries where dialogue items are {character_id,text}, "
    "bubbles items are {bbox_px:[x,y,w,h]} integers inside 512x512, and "
    "render includes image_path and overlay_applied boolean; "
    "exercises=[{exercise_id,panel_id,prompt,blanks,answer_key,"
    "feedback_rules}] "
    "example of the exercise output format: "
    '{"exercise_id":"ex_panel_1","panel_id":"panel_1",'
    '"prompt":"We read one short ____ together.",'
    '"blanks":["____"],"answer_key":["instruction"],'
    '"feedback_rules":{"case_sensitive":false,'
    '"allow_trim_spaces":true}}'
    "same length as panels. Keep text simple and adult age-appropriate."
)
