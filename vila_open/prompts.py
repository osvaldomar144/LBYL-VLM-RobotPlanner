# vila_open/prompts.py

from typing import List, Dict, Any

# Definiamo lo schema JSON che la VLM deve rispettare
PLAN_SCHEMA_STR = r"""
You must output a JSON object with the following structure:

{
  "plan": [
    {
      "primitive": "base",
      "params": [0.8, 0, 0],
      "reasoning": "Moving closer to the counter"
    }
  ]
}

COMMAND RULES:
1. "base": params [vx, vy, w]. Range -1.0 to 1.0.
   - Use [1, 0, 0] to move FORWARD.
   - Use [-1, 0, 0] to move BACKWARD.
   - Use [0, 0, 1] to ROTATE left.
2. "arm": params [x, y, z]. Range -1.0 to 1.0. 
   - These are DELTA movements relative to current hand position.
   - [1, 0, 0] is Forward (away from robot).
   - [0, 0, 1] is Up.
   - [0, 0, -1] is Down.
3. "torso": params [velocity]. Range -1.0 to 1.0.
   - [1.0] to go UP, [-1.0] to go DOWN.
4. "gripper": params [1.0] (CLOSE) or [-1.0] (OPEN).

Output ONLY the JSON. No preamble.
"""

def build_system_prompt() -> str:
    return (
        "You are a Robot Pilot operating a mobile manipulator (Panda arm on wheels).\n"
        "You receive an image of what the robot sees.\n"
        "Your goal is to guide the robot continuously towards the objective.\n\n"
        "CONTROL STRATEGY:\n"
        "- If the target is far away or not reachable: use 'base' to navigate or 'torso' to adjust height.\n"
        "- If the target is within reach: use 'arm' to align the hand and approach it.\n"
        "- Use small steps. You will be called again in the next frame to correct the trajectory.\n"
        "- If you need to grasp: align the gripper above/around the object, lower the arm, then close gripper.\n\n"
        + PLAN_SCHEMA_STR
    )

def build_user_prompt(goal_instruction: str, current_state: dict) -> str:
    """
    Costruisce il prompt utente con lo stato attuale del robot.
    """
    # Formattiamo lo stato per aiutare la VLM (soprattutto per la depth estimation)
    pos_str = f"[{current_state['eef_xyz'][0]}, {current_state['eef_xyz'][1]}, {current_state['eef_xyz'][2]}]"
    
    return (
        f"GOAL: {goal_instruction}\n\n"
        f"CURRENT ROBOT STATE:\n"
        f"- EEF Position (XYZ): {pos_str}\n"
        f"- Gripper State: {current_state['gripper']}\n"
        f"- Safety Warning: {current_state['safety_warning']}\n\n"
        "Look at the image. Determine the immediate NEXT MOVE to progress towards the goal.\n"
        "Output the JSON plan."
    )