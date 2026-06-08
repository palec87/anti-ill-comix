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
    "characters, panels, exercises. Use input language code and style_id. "
    "Schema: simplified={summary:string, level:string, keywords:string[]}; "
    "characters=[{id,name,description}] with 2-3 entries; "
    "panels=[{panel_id, frame_index, scene_description, dialogue, bubbles, "
    "render}] with 3-5 entries where dialogue items are {character_id,text}, "
    "bubbles items are {bbox_px:[x,y,w,h]} integers inside 512x512, and "
    "render includes image_path and overlay_applied boolean; "
    "exercises=[{exercise_id,panel_id,prompt,blanks,answer_key,"
    "feedback_rules}] "
    "same length as panels. Keep text simple and age-appropriate."
)
