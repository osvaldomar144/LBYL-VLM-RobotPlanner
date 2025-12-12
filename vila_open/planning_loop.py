# vila_open/planning_loop.py

import json
import re
from PIL import Image
from .schema import Plan
from .prompts import build_system_prompt, build_user_prompt
from .vlm_client import VLMClient

def extract_json_substring(text: str) -> str:
    """
    Estrae il JSON in modo robusto usando regex per trovare il blocco
    tra la prima graffa aperta e l'ultima chiusa.
    """
    # 1. Cerca pattern markdown ```json ... ```
    pattern = r"```json(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # 2. Se non trova markdown, cerca dalla prima { all'ultima }
    first = text.find("{")
    last = text.rfind("}")
    
    if first != -1 and last != -1 and last > first:
        return text[first:last + 1]
    
    # 3. Fallback: ritorna tutto e spera
    return text 

def plan_next_step(
    image: Image.Image,
    goal_instruction: str,
    current_state: dict,
    vlm_client: VLMClient,
    last_action_report: str = None 
) -> Plan:
    
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(goal_instruction, current_state, last_action_report)

    # Chiamata VLM
    raw_output = vlm_client.generate_plan_text(
        image=image,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    try:
        json_str = extract_json_substring(raw_output)
        # Pulizia caratteri sporchi
        json_str = json_str.strip()
        
        json_data = json.loads(json_str)
        return Plan.from_dict(json_data)
    except json.JSONDecodeError as e:
        print(f"[Planner ERROR] JSON non valido. Raw output:\n{raw_output}\nError: {e}")
        return Plan(plan=[])