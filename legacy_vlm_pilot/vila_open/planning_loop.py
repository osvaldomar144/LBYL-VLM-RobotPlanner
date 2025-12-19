# vila_open/planning_loop.py

import json
import re
from typing import Sequence, Union
from PIL import Image

from .schema import Plan
from .prompts import build_system_prompt, build_user_prompt
from .vlm_client import VLMClient


def extract_json_substring(text: str) -> str:
    """
    Estrae il JSON in modo robusto usando regex per trovare il blocco
    tra la prima graffa aperta e l'ultima chiusa.
    """
    # 1) Cerca pattern markdown ```json ... ```
    pattern = r"```json(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # 2) Cerca dalla prima { all'ultima }
    first = text.find("{")
    last = text.rfind("}")

    if first != -1 and last != -1 and last > first:
        return text[first:last + 1]

    # 3) Fallback
    return text


def plan_next_step(
    image: Union[Image.Image, Sequence[Image.Image]],
    goal_instruction: str,
    current_state: dict,
    vlm_client: VLMClient,
    last_action_report: str = None,
) -> Plan:
    """
    Pianifica il prossimo step usando la VLM.

    image può essere:
      - PIL.Image.Image (single-image, compatibile col vecchio flow)
      - Sequence[PIL.Image.Image] (multi-image, es. [global_view, hand_view])
    """

    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(goal_instruction, current_state, last_action_report)

    # (Consigliato) se multi-image, aggiungi una nota esplicita nel testo utente
    # per rendere inequivocabile l'ordine delle immagini.
    if isinstance(image, (list, tuple)) and len(image) >= 2:
        user_prompt = (
            "IMAGES ORDER:\n"
            "- Image 1 = GLOBAL view (robot from behind)\n"
            "- Image 2 = HAND view (camera on gripper)\n"
            "These are two views of the SAME scene at the SAME time.\n\n"
            + user_prompt
        )

    raw_output = vlm_client.generate_plan_text(
        image=image,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    try:
        json_str = extract_json_substring(raw_output).strip()
        json_data = json.loads(json_str)
        return Plan.from_dict(json_data)
    except json.JSONDecodeError as e:
        print(f"[Planner ERROR] JSON non valido. Raw output:\n{raw_output}\nError: {e}")
        return Plan(plan=[])
    except Exception as e:
        print(f"[Planner ERROR] Errore inatteso. Raw output:\n{raw_output}\nError: {e}")
        return Plan(plan=[])
