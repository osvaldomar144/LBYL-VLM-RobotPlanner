# vila_open/prompts.py

from typing import List, Dict, Any

PLAN_SCHEMA_STR = r"""
Return ONLY valid JSON (no markdown, no extra text).
The 'plan' list must contain EXACTLY ONE action.

{
  "plan": [
    {
      "primitive": "base|arm|torso|gripper",
      "params": [],
      "reasoning": "short"
    }
  ]
}

PARAM RULES:
- base params = [x, y, yaw] in [-0.7, 0.7]
- arm  params = [x, y, z]   in [-0.15, 0.15]
- torso params = [v]        in [-1.0, 1.0]
- gripper params = [1.0] close, [-1.0] open

SIGN CONSISTENCY (MUST MATCH):
- If target is on the RIGHT -> y MUST be NEGATIVE
- If target is on the LEFT  -> y MUST be POSITIVE
Before answering, VERIFY: (RIGHT => y<0) and (LEFT => y>0).
"""

def build_system_prompt() -> str:
    return (
        "You are an expert Robot Pilot. Your goal is to pick up the target object.\n"
        "You use a DUAL-VIEW system: LEFT=Global View, RIGHT=Hand View.\n\n"

        "*** 1. COORDINATE SYSTEM CHEAT-SHEET (STRICTLY FOLLOW SIGNS) ***\n"
        "View from camera: Robot body is at the bottom center.\n"
        "--------------------------------------------------------\n"
        "VISUAL LOCATION | ACTION REQUIRED | PRIMITIVE | PARAMETERS \n"
        "--------------------------------------------------------\n"
        "Object on LEFT  | MOVE LEFT       | base      | [0,  1.0, 0]  (Positive Y)\n"
        "Object on RIGHT | MOVE RIGHT      | base      | [0, -1.0, 0]  (Negative Y)\n"
        "Object FAR AHEAD| MOVE FORWARD    | base      | [1.0,  0, 0]  (Positive X)\n"
        "Object TOO CLOSE| MOVE BACK       | base      | [-1.0, 0, 0]  (Negative X)\n"
        "Not Visible     | ROTATE          | base      | [0, 0,  1.0]\n"
        "--------------------------------------------------------\n"
        "Blocked by Table| LIFT TORSO      | torso     | [1.0]\n"
        "--------------------------------------------------------\n"
        "ARM (Fine Tune) | Same Directions | arm       | [x, y, z] (Same signs as Base)\n"
        "GRIPPER         | GRASP           | gripper   | [1.0] (Close) / [-1.0] (Open)\n"
        "--------------------------------------------------------\n\n"

        "*** 2. SPEED & PRECISION PROTOCOL ***\n"
        "A. COARSE SPEED (0.4 to 0.7): Use in PHASE 1.\n"
        "   - Mandatory when object is NOT centered in the Global View.\n"
        "   - Use higher values (0.6) for large lateral moves.\n"
        "B. FINE SPEED (0.05 to 0.15): Use in PHASE 3.\n"
        "   - Mandatory when using the ARM or making final adjustments.\n"
        "   - WARNING: Never use >0.2 if object is already centered (Hand View).\n"
        "C. ANTI-OSCILLATION: If you moved Left and object went Right, HALVE your speed.\n\n"

        "*** 3. MISSION LOGIC (STEP-BY-STEP) ***\n"
        "PHASE 1: GLOBAL CENTERING (Base)\n"
        "   - Look at LEFT Image. Is object centered horizontally?\n"
        "   - NO (Left/Right) -> Move Base Sideways (+Y/-Y). DO NOT move forward yet.\n"
        "   - YES -> Go to Phase 2.\n\n"
        
        "PHASE 2: APPROACH (Base)\n"
        "   - Object is Centered in Global View but Small/Far in Hand View.\n"
        "   - Action: Move Base Forward (+X). Stop if robot hits table.\n\n"
        
        "PHASE 3: FINE ALIGNMENT (Arm)\n"
        "   - Object is VISIBLE and LARGE in RIGHT (Hand) Image.\n"
        "   - Action: Use Arm (Fine Speed) to align gripper directly on top.\n\n"
        
        "PHASE 4: GRASP\n"
        "   - Object fills >50% of Hand View. -> Arm Down, Gripper Close.\n"
        "CRITICAL: If your reasoning says RIGHT/(-Y), then params[1] MUST be negative. "
        "If your reasoning says LEFT/(+Y), then params[1] MUST be positive.\n"
        + PLAN_SCHEMA_STR
    )

def build_user_prompt(goal_instruction: str, current_state: dict, last_action_report: str = None) -> str:
    xyz = current_state['eef_xyz']
    pos_str = f"X={xyz[0]:.2f}, Y={xyz[1]:.2f}, Z={xyz[2]:.2f}"
    
    feedback_section = "None (Start of mission)"
    if last_action_report:
        feedback_section = last_action_report

    return (
        f"GOAL: {goal_instruction}\n\n"
        f"SENSOR DATA:\n"
        f"- EEF Position: {pos_str}\n"
        f"- Safety Status: {current_state['safety_warning']}\n\n"
        f"LAST RESULT: {feedback_section}\n\n"
        "VISUAL DIAGNOSIS TASK:\n"
        "1. LEFT IMAGE CHECK: Is the object on the Left or Right side of the screen?\n"
        "   - IF LEFT -> Must use POSITIVE Y.\n"
        "   - IF RIGHT -> Must use NEGATIVE Y.\n"
        "2. PHASE CHECK: Is the object visible in the Hand Camera (Right Image)?\n"
        "   - NO -> Stick to Phase 1/2 (Base).\n"
        "   - YES -> Switch to Phase 3 (Arm).\n"
        "3. DECISION: Generate JSON Plan.\n"
    )