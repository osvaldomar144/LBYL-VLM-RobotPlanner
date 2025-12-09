# vila_open/planning_loop.py

import json
from typing import List, Dict, Any
from PIL import Image

from .schema import Plan
from .prompts import build_system_prompt, build_user_prompt
from .vlm_client import VLMClient


def extract_json_substring(text: str) -> str:
    """
    Alcuni modelli possono aggiungere testo extra.
    Qui estraiamo la prima sottostringa che sembra un JSON.
    """
    first = text.find("{")
    last = text.rfind("}")
    if first == -1 or last == -1 or last <= first:
        raise ValueError(f"No JSON object found in model output: {text}")
    return text[first:last + 1]


def plan_once(
    image: Image.Image,
    goal_instruction: str,
    history: List[Dict[str, Any]],
    available_primitives: List[str],
    vlm_client: VLMClient,
) -> Plan:
    """
    Una singola chiamata al planner (stile ViLa):
    immagine + goal + history  -> piano completo (oggetto Plan).
    """
    system_prompt = build_system_prompt(available_primitives)
    user_prompt = build_user_prompt(goal_instruction, history)

    raw_output = vlm_client.generate_plan_text(
        image=image,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    json_str = extract_json_substring(raw_output)
    json_data = json.loads(json_str)
    plan = Plan.from_dict(json_data)
    return plan