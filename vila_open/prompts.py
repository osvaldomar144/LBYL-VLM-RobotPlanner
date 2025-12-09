# vila_open/prompts.py

from typing import List, Dict, Any


PLAN_SCHEMA_STR = r"""
You must output a JSON object with the following structure:

{
  "plan": [
    {
      "step_id": 1,
      "primitive": "pick",
      "object": "mug",
      "from_location": "sink"
    },
    {
      "step_id": 2,
      "primitive": "place",
      "object": "mug",
      "to_location": "microwave_interior"
    }
  ]
}

- "primitive" MUST be one of: "pick", "place", "open", "close",
  "press_button", "turn_knob", "navigate".
- step_id MUST start from 1 and increase by 1 for each step.
- Only include fields that are necessary for the action.
- Do NOT include any explanation outside the JSON.
"""


def build_system_prompt(available_primitives: List[str]) -> str:
    primitives_str = ", ".join(f'"{p}"' for p in available_primitives)
    return (
        "You are a robotic task planner operating in a kitchen environment. "
        "You see an image of the current scene. Your job is to produce a "
        "sequence of high-level actions (a plan) that a robot can execute.\n\n"
        f"The robot can ONLY use the following primitives: {primitives_str}.\n\n"
        + PLAN_SCHEMA_STR
    )


def history_to_text(history: List[Dict[str, Any]]) -> str:
    """
    history: lista di dict del tipo:
      { "step_id": int, "primitive": "...", "object": "...", "result": "success/failed" }
    """
    if not history:
        return "No actions have been executed yet."
    lines = []
    for h in history:
        desc = f"step {h['step_id']}: {h['primitive']}"
        if h.get("object"):
            desc += f" (object={h['object']})"
        if h.get("result"):
            desc += f" -> result={h['result']}"
        lines.append(desc)
    return "\n".join(lines)


def build_user_prompt(goal_instruction: str,
                      history: List[Dict[str, Any]]) -> str:
    history_str = history_to_text(history)
    return (
        "High-level instruction from the user:\n"
        f"{goal_instruction}\n\n"
        "Actions already executed so far:\n"
        f"{history_str}\n\n"
        "Look carefully at the provided image of the current scene and reason step by step "
        "about what the robot should do next to achieve the goal from the current state.\n\n"
        "Then, output ONLY a JSON object describing a COMPLETE plan from now until the goal "
        "is accomplished, following exactly the required JSON schema."
    )