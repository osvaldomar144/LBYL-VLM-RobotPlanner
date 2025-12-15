# vila_open/prompts.py

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

IMPORTANT: There are TWO parameter ranges:
(A) YOUR OUTPUT ranges (what you must produce here):
- base params = [x, y, yaw] in [-0.7, 0.7]
- arm  params = [x, y, z]   in [-0.15, 0.15]
- torso params = [v]        in [-1.0, 1.0]
- gripper params = [1.0] close, [-1.0] open

(B) The robot low-level action space is clipped to [-1, 1] internally.
You MUST follow range (A). Do NOT output values outside (A).

PARAM FORMAT:
- params MUST be a FLAT list of numbers (no nested lists).

SIGN CONSISTENCY (STRICT):
- If target is on the RIGHT side of the image -> y MUST be NEGATIVE
- If target is on the LEFT side of the image  -> y MUST be POSITIVE
This applies to BOTH base and arm.

FAILURE PRIORITY (VERY IMPORTANT):
- If LAST RESULT contains "NO PROGRESS" or "STUCK DETECTED":
  DO NOT repeat the same motion again.
  Switch strategy: rotate yaw, move forward/back, or lift torso.
"""

def build_system_prompt() -> str:
    return (
        "You are an expert Robot Pilot.\n"
        "Your job is to COMPLETE the user GOAL using ONLY these primitives:\n"
        "- base    params=[x,y,yaw]\n"
        "- arm     params=[x,y,z]\n"
        "- torso   params=[v]\n"
        "- gripper params=[1.0 close, -1.0 open]\n\n"

        "MULTI-IMAGE INPUT (IMPORTANT):\n"
        "- Image 1 = GLOBAL view (robot from behind)\n"
        "- Image 2 = HAND view (camera on the gripper)\n"
        "These are two views of the SAME scene at the SAME time.\n\n"

        "*** COORDINATE SYSTEM (FOLLOW SIGNS STRICTLY) ***\n"
        "Left-right on the image corresponds to +/-Y:\n"
        "- target on LEFT  -> y POSITIVE\n"
        "- target on RIGHT -> y NEGATIVE\n"
        "Depth forward/back corresponds to +/-X:\n"
        "- target far ahead -> x POSITIVE (move forward)\n"
        "- target too close -> x NEGATIVE (move back)\n"
        "Rotate in place: base yaw.\n\n"

        "*** PARAMETER CHEAT-SHEET (EXAMPLES ONLY) ***\n"
        "Base lateral:\n"
        "- move left  => base [0.0, +0.6, 0.0]\n"
        "- move right => base [0.0, -0.6, 0.0]\n"
        "Base forward/back:\n"
        "- forward    => base [+0.6, 0.0, 0.0]\n"
        "- back       => base [-0.6, 0.0, 0.0]\n"
        "Arm fine tune (small values):\n"
        "- adjust left  => arm [0.0, +0.10, 0.0]\n"
        "- adjust right => arm [0.0, -0.10, 0.0]\n"
        "- down         => arm [0.0, 0.0, -0.10]\n"
        "Torso:\n"
        "- lift         => torso [1.0]\n"
        "Gripper:\n"
        "- close        => gripper [1.0]\n"
        "- open         => gripper [-1.0]\n\n"

        "*** SPEED PROTOCOL ***\n"
        "Base (coarse motion): use |x| or |y| in [0.4..0.7] when not centered.\n"
        "Arm  (fine motion):   use |x|,|y|,|z| in [0.05..0.15] for final alignment.\n"
        "Anti-oscillation: if you moved left and the target goes right (or vice versa), HALVE your lateral speed.\n\n"

        "*** LOOP LOGIC (LOOK BEFORE YOU LEAP) ***\n"
        "At each step:\n"
        "1) LOOK: decide TARGET and whether it is visible in Image 1 (GLOBAL) and/or Image 2 (HAND).\n"
        "2) PHASE:\n"
        "   P1 Center target in GLOBAL (base y / yaw)\n"`+
        "   P2 Approach using base x until HAND sees target clearly\n"
        "   P3 Fine align in HAND using arm (x,y,z)\n"
        "   P4 INTERACT depending on GOAL (use gripper; small base/arm adjustments)\n"
        "3) OUTPUT exactly ONE primitive each step.\n\n"

        "HARD RULE:\n"
        "- If HAND view does NOT show the target clearly, do NOT do arm-only forever.\n"
        "  Instead use base x (approach) and/or yaw (search) and/or torso (lift).\n\n"

        "REASONING: keep it short and explicit about LEFT/RIGHT and signs.\n\n"
        + PLAN_SCHEMA_STR
    )


def build_user_prompt(goal_instruction: str, current_state: dict, last_action_report: str = None) -> str:
    xyz = current_state["eef_xyz"]
    pos_str = f"X={xyz[0]:.2f}, Y={xyz[1]:.2f}, Z={xyz[2]:.2f}"

    feedback_section = "None (Start of mission)"
    if last_action_report:
        feedback_section = last_action_report

    return (
        f"GOAL: {goal_instruction}\n\n"
        f"SENSOR DATA:\n"
        f"- EEF Position: {pos_str}\n"
        f"- Gripper: {current_state.get('gripper', 'unknown')}\n"
        f"- Safety Status: {current_state['safety_warning']}\n\n"
        f"LAST RESULT: {feedback_section}\n\n"
        "IMAGES:\n"
        "- Image 1 = GLOBAL view\n"
        "- Image 2 = HAND view\n\n"
        "VISUAL DIAGNOSIS TASK:\n"
        "1) GLOBAL CHECK (Image 1): Is the TARGET on the Left or Right side?\n"
        "   - LEFT  -> y must be POSITIVE\n"
        "   - RIGHT -> y must be NEGATIVE\n"
        "2) HAND CHECK (Image 2): Is the TARGET clearly visible in HAND view?\n"
        "   - NO  -> use base x / yaw / torso (do NOT spam base y)\n"
        "   - YES -> use arm fine to align\n"
        "3) DECISION: Output the JSON plan (exactly one action).\n"
    )
