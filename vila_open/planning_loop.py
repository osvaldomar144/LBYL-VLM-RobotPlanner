# vila_open/planning_loop.py

import json
from PIL import Image
from .schema import Plan
from .prompts import build_system_prompt, build_user_prompt
from .vlm_client import VLMClient

def extract_json_substring(text: str) -> str:
    first = text.find("{")
    last = text.rfind("}")
    if first == -1 or last == -1 or last <= first:
        # Fallback: a volte i modelli piccoli non mettono parentesi se il prompt è forte
        return text 
    return text[first:last + 1]

def plan_next_step(
    image: Image.Image,
    goal_instruction: str,
    current_state: dict,
    vlm_client: VLMClient,
) -> Plan:
    """
    Chiede alla VLM la prossima mossa basandosi sull'immagine e stato corrente.
    """
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(goal_instruction, current_state)

    # Chiamata VLM (qui usiamo il client esistente)
    raw_output = vlm_client.generate_plan_text(
        image=image,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    try:
        json_str = extract_json_substring(raw_output)
        # Pulizia extra per output sporchi (es. markdown ```json ... ```)
        json_str = json_str.replace("```json", "").replace("```", "").strip()
        
        json_data = json.loads(json_str)
        return Plan.from_dict(json_data)
    except json.JSONDecodeError:
        print(f"[Planner ERROR] Invalid JSON from VLM: {raw_output}")
        return Plan(plan=[]) # Ritorna piano vuoto in caso di errore