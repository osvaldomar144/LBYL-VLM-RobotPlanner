# scripts/run_vlm_pilot.py

import argparse
import cv2
import numpy as np
import sys
import os
import re
from PIL import Image
from collections import deque

# opzionale: aiuta un po' contro OOM/fragmentation
try:
    import torch
    _HAS_TORCH = True
except Exception:
    torch = None
    _HAS_TORCH = False

# ... (Gestione Path invariata) ...
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from robot_vlm_lib import VLMRobotInterface
except ImportError:
    from scripts.robot_vlm_lib import VLMRobotInterface

try:
    from vila_open.vlm_client import VLMClient, VLMConfig
    from vila_open.planning_loop import plan_next_step
except ImportError:
    print("ERRORE: Impossibile importare vila_open.")
    sys.exit(1)


# -----------------------------
# Utils immagini
# -----------------------------
def numpy_img_to_pil(np_img: np.ndarray) -> Image.Image:
    """
    np_img atteso in RGB uint8 (H,W,3).
    """
    if np_img.dtype != np.uint8:
        np_img = np_img.astype(np.uint8)
    return Image.fromarray(np_img)


def update_visual_monitor(img_np, step_name="Init"):
    """
    Visualizza e salva un frame composito SOLO per debug umano.
    """
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    cv2.imshow("VLM Dual-Eye Monitor", img_bgr)
    cv2.waitKey(1)
    filename = f"live_view_{step_name}.png"
    cv2.imwrite(filename, img_bgr)
    return filename


def compose_dual(img_global, img_hand):
    """
    Crea immagine composita + overlay.
    NOTA: serve SOLO per monitor (la VLM riceve le due immagini separate).
    """
    dual = np.hstack((img_global, img_hand)).copy()
    h, w, _ = dual.shape
    mid = w // 2
    cv2.line(dual, (mid, 0), (mid, h - 1), (255, 255, 255), 2)
    cv2.putText(dual, "GLOBAL (LEFT)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(dual, "HAND (RIGHT)", (mid + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return dual


# -----------------------------
# Guardrail (LBYL-style)
# -----------------------------
BASE_LIM = 0.7
ARM_LIM = 0.15


def _safe_float(x):
    try:
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    except Exception:
        return None


def _clip(v, lim):
    v = _safe_float(v)
    if v is None:
        return None
    return float(max(-lim, min(lim, v)))


def infer_y_sign_from_reasoning(reasoning: str):
    """
    Se il reasoning dichiara esplicitamente y positive/negative, usalo come vincolo.
    """
    if not reasoning:
        return None
    r = reasoning.upper()
    pos = ("Y POSITIVE" in r) or ("POSITIVE Y" in r) or ("+Y" in r)
    neg = ("Y NEGATIVE" in r) or ("NEGATIVE Y" in r) or ("-Y" in r)
    if pos and not neg:
        return +1
    if neg and not pos:
        return -1
    return None


def infer_lr_from_reasoning(reasoning: str):
    """
    Inferisce LEFT/RIGHT da frasi sulla POSIZIONE del target.
    Evita falsi positivi tipo "HAND (RIGHT)" / "Image 2 (RIGHT)".
    """
    if not reasoning:
        return None
    r = reasoning.upper()

    # supporto formato esplicito (se lo userai nel prompt)
    m = re.search(r"TARGET_SIDE\s*=\s*(LEFT|RIGHT|CENTER|UNKNOWN)", r)
    if m:
        side = m.group(1)
        if side == "LEFT":
            return "LEFT"
        if side == "RIGHT":
            return "RIGHT"
        return None

    # pattern "target ... left/right"
    if re.search(r"\bTARGET\b.*\bRIGHT\b", r):
        return "RIGHT"
    if re.search(r"\bTARGET\b.*\bLEFT\b", r):
        return "LEFT"

    right_signals = [
        "ON THE RIGHT",
        "TO THE RIGHT",
        "RIGHT SIDE OF THE IMAGE",
        "RIGHT SIDE",
        "MOVE RIGHT",
    ]
    left_signals = [
        "ON THE LEFT",
        "TO THE LEFT",
        "LEFT SIDE OF THE IMAGE",
        "LEFT SIDE",
        "MOVE LEFT",
    ]

    def safe_contains(sig: str) -> bool:
        # evita match banali "RIGHT image"/"LEFT image"
        if sig in ("RIGHT", "LEFT"):
            return False
        return sig in r

    if any(safe_contains(s) for s in right_signals):
        return "RIGHT"
    if any(safe_contains(s) for s in left_signals):
        return "LEFT"

    return None


def _flatten_numeric_params(obj, out, depth=0, max_depth=3):
    if depth > max_depth:
        return
    if isinstance(obj, (list, tuple)):
        for x in obj:
            _flatten_numeric_params(x, out, depth + 1, max_depth)
        return
    v = _safe_float(obj)
    if v is not None:
        out.append(v)


def _fix_len_params(params, n_expected):
    nums = []
    _flatten_numeric_params(params, nums)
    if len(nums) > n_expected:
        nums = nums[:n_expected]
    elif len(nums) < n_expected:
        nums = nums + [0.0] * (n_expected - len(nums))
    return nums


def validate_and_compile(action, recent_actions: deque):
    """
    Returns: (ok:bool, action, msg:str)
    - ok False => do not execute, feed msg back to VLM
    """
    prim = getattr(action, "primitive", None)
    raw_params = getattr(action, "params", []) or []
    reasoning = getattr(action, "reasoning", "") or ""

    if prim not in ("base", "arm", "torso", "gripper"):
        return False, action, f"PLAN REJECTED: unknown primitive '{prim}'. Use only base|arm|torso|gripper."

    # AUTO-FIX lunghezze (evita reject loop)
    auto_fixed = False
    if prim == "base":
        params = _fix_len_params(raw_params, 3)
        auto_fixed = (not isinstance(raw_params, (list, tuple))) or (len(raw_params) != 3)
    elif prim == "arm":
        params = _fix_len_params(raw_params, 3)
        auto_fixed = (not isinstance(raw_params, (list, tuple))) or (len(raw_params) != 3)
    elif prim == "torso":
        params = _fix_len_params(raw_params, 1)
        auto_fixed = (not isinstance(raw_params, (list, tuple))) or (len(raw_params) != 1)
    elif prim == "gripper":
        params = _fix_len_params(raw_params, 1)
        auto_fixed = (not isinstance(raw_params, (list, tuple))) or (len(raw_params) != 1)

    # clamp + sign repair
    if prim == "base":
        x = _clip(params[0], BASE_LIM)
        y = _clip(params[1], BASE_LIM)
        yaw = _clip(params[2], BASE_LIM)
        if x is None or y is None or yaw is None:
            return False, action, "PLAN REJECTED: base params contain non-numeric/NaN/Inf."

        desired = None
        lr = infer_lr_from_reasoning(reasoning)
        if lr == "RIGHT":
            desired = -1  # RIGHT => y negative
        elif lr == "LEFT":
            desired = +1  # LEFT  => y positive
        else:
            ys = infer_y_sign_from_reasoning(reasoning)
            if ys is not None:
                desired = ys

        if desired == +1 and y < 0:
            y = abs(y)
        elif desired == -1 and y > 0:
            y = -abs(y)

        # anti-oscillation laterale
        if recent_actions:
            last = recent_actions[-1]
            if last["primitive"] == "base":
                y0 = last["params"][1]
                if abs(y0) > 1e-3 and abs(y) > 1e-3 and (y0 * y < 0):
                    y *= 0.5

        params = [x, y, yaw]

    elif prim == "arm":
        x = _clip(params[0], ARM_LIM)
        y = _clip(params[1], ARM_LIM)
        z = _clip(params[2], ARM_LIM)
        if x is None or y is None or z is None:
            return False, action, "PLAN REJECTED: arm params contain non-numeric/NaN/Inf."

        # FIX IMPORTANTISSIMO: arm deve seguire la stessa regola di base
        desired = None
        lr = infer_lr_from_reasoning(reasoning)
        if lr == "RIGHT":
            desired = -1  # RIGHT => y negative
        elif lr == "LEFT":
            desired = +1  # LEFT  => y positive
        else:
            ys = infer_y_sign_from_reasoning(reasoning)
            if ys is not None:
                desired = ys

        if desired == +1 and y < 0:
            y = abs(y)
        elif desired == -1 and y > 0:
            y = -abs(y)

        params = [x, y, z]

    elif prim == "torso":
        v = _clip(params[0], 1.0)
        if v is None:
            return False, action, "PLAN REJECTED: torso param is non-numeric/NaN/Inf."
        params = [v]

    elif prim == "gripper":
        v = _safe_float(params[0])
        if v is None:
            return False, action, "PLAN REJECTED: gripper param is non-numeric/NaN/Inf."
        params = [1.0 if v > 0 else -1.0]

    action.params = params
    if auto_fixed:
        return True, action, "PLAN OK (auto-fixed params length)"
    return True, action, "PLAN OK"


# -----------------------------
# Progress signal (EEF delta)
# -----------------------------
def assess_progress(primitive: str, params: list, result_info: dict):
    """
    Decide se l'azione ha avuto effetto (in modo robusto) usando eef_delta_norm.
    """
    delta = result_info.get("eef_delta_xyz", [0.0, 0.0, 0.0])
    dn = float(result_info.get("eef_delta_norm", 0.0))
    dx, dy, dz = (float(delta[0]), float(delta[1]), float(delta[2]))

    # soglie conservative (da tarare, ma già utili per “bloccato contro ostacolo”)
    if primitive == "base":
        x, y, yaw = float(params[0]), float(params[1]), float(params[2])
        trans_cmd = abs(x) + abs(y)

        # se stai ruotando, l'EEF può muoversi poco: non usare dn come “fail”
        if abs(yaw) >= 0.2:
            return True, f"motion=rotate(yaw={yaw:+.2f}), eef_delta={dn:.4f}m"

        # se il comando traslazionale è “grande” e dn è quasi zero => bloccato
        if trans_cmd >= 0.25 and dn < 0.006:
            return False, f"NO PROGRESS: base cmd x/y={x:+.2f},{y:+.2f} but eef_delta={dn:.4f}m"
        return True, f"eef_delta={dn:.4f}m (dx={dx:+.4f}, dy={dy:+.4f}, dz={dz:+.4f})"

    if primitive == "arm":
        x, y, z = float(params[0]), float(params[1]), float(params[2])
        cmd = abs(x) + abs(y) + abs(z)
        if cmd >= 0.12 and dn < 0.002:
            return False, f"NO PROGRESS: arm cmd {x:+.2f},{y:+.2f},{z:+.2f} but eef_delta={dn:.4f}m"
        return True, f"eef_delta={dn:.4f}m (dx={dx:+.4f}, dy={dy:+.4f}, dz={dz:+.4f})"

    if primitive == "torso":
        v = float(params[0])
        if abs(v) >= 0.5 and abs(dz) < 0.003:
            return False, f"NO PROGRESS: torso cmd v={v:+.2f} but eef_dz={dz:+.4f}m"
        return True, f"eef_delta={dn:.4f}m (dz={dz:+.4f})"

    # gripper: progresso non misurabile con EEF delta
    return True, f"eef_delta={dn:.4f}m"


def stuck_hint_from_history(recent_actions: deque, recent_progress_ok: deque):
    """
    Se ripete base y e spesso NO PROGRESS, inietta un hint LBYL nel contesto.
    """
    if len(recent_actions) < 4 or len(recent_progress_ok) < 4:
        return None

    seq_actions = list(recent_actions)[-4:]
    seq_ok = list(recent_progress_ok)[-4:]

    # pattern: base laterale ripetuta (x~0,y!=0,yaw~0)
    if all(a["primitive"] == "base" for a in seq_actions):
        lateral = []
        for a in seq_actions:
            x, y, yaw = a["params"]
            lateral.append((abs(x) < 0.05) and (abs(y) >= 0.10) and (abs(yaw) < 0.05))
        if all(lateral) and (seq_ok.count(False) >= 2):
            return ("STUCK DETECTED: repeated base-y lateral moves with NO PROGRESS. "
                    "NEXT action MUST be: base x (approach) OR base yaw rotate OR torso lift. "
                    "Avoid base y now.")
    return None


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="PnPCounterToCab")
    parser.add_argument("--max_steps", type=int, default=100)
    args = parser.parse_args()

    print("--- VLM ROBOT PILOT: MULTI-IMAGE (GLOBAL + HAND) ---")

    # IMPORTANT: riduci token per evitare OOM (JSON 1-step non ne ha bisogno)
    vlm_config = VLMConfig(device="cuda", max_new_tokens=256, verbose=False)
    vlm_client = VLMClient(vlm_config)

    print(f"[System] Loading Robot ({args.env_name})...")
    robot = VLMRobotInterface(env_name=args.env_name, render=True)

    recent_actions = deque(maxlen=8)
    recent_progress_ok = deque(maxlen=8)

    # =========================================================================
    # INIT
    # =========================================================================
    print("\n[Phase 1] Initializing Dual Views...")
    text_info, visual_dict = robot.get_context()

    img_global = visual_dict.get("robot0_agentview_center")
    img_hand = visual_dict.get("robot0_eye_in_hand")

    if img_global is None or img_hand is None:
        print("[Error] Una delle due camere manca! Verifica robot_vlm_lib.py")
        robot.close()
        return

    dual_view_img = compose_dual(img_global, img_hand)
    update_visual_monitor(dual_view_img, step_name="00_START")

    print("---------------------------------------------------------------")
    print(" INPUT VLM: 2 immagini separate (1=GLOBAL, 2=HAND).")
    print(" MONITOR: finestra mostra composita con label.")
    print("---------------------------------------------------------------")

    try:
        user_goal = input("INSERT GOAL > ").strip()
        if not user_goal:
            user_goal = "Look around"
    except EOFError:
        return

    print(f'\n[Mission Start] GOAL: "{user_goal}"')

    last_action_report = "None (Start of mission)"

    try:
        for step in range(args.max_steps):
            print(f"\n--- STEP {step+1}/{args.max_steps} ---")

            text_info, visual_dict = robot.get_context()
            img_global = visual_dict.get("robot0_agentview_center")
            img_hand = visual_dict.get("robot0_eye_in_hand")

            if img_global is None or img_hand is None:
                print("[System] Missing camera frame. Stopping.")
                break

            # Monitor debug
            dual_view_img = compose_dual(img_global, img_hand)
            update_visual_monitor(dual_view_img, step_name=f"{step+1:02d}")

            # === MULTI-IMAGE INPUT per la VLM ===
            pil_global = numpy_img_to_pil(img_global)
            pil_hand = numpy_img_to_pil(img_hand)
            vlm_images = [pil_global, pil_hand]

            # Inject stuck hint (prima di chiamare la VLM)
            hint = stuck_hint_from_history(recent_actions, recent_progress_ok)
            brain_context = last_action_report
            if hint:
                print(hint)
                brain_context = f"{last_action_report}\n{hint}"

            # best-effort cache clear
            if _HAS_TORCH and torch.cuda.is_available():
                torch.cuda.empty_cache()

            # PLAN
            print(f"[Brain Input] Context: {brain_context}")
            print("[Brain] Thinking...", end="", flush=True)

            plan = plan_next_step(
                image=vlm_images,
                goal_instruction=user_goal,
                current_state=text_info,
                vlm_client=vlm_client,
                last_action_report=brain_context,
            )
            print(" Done.")

            if not plan.plan:
                print("[Brain] ??? Confusion. Skip.")
                last_action_report = "PLAN FAILED: empty plan. Output a valid JSON plan with exactly one action."
                continue

            action = plan.plan[0]

            ok, action, msg = validate_and_compile(action, recent_actions)
            if not ok:
                print(f"[Leap] {msg}")
                last_action_report = msg + " Output a corrected plan."
                continue
            elif "auto-fixed" in msg.lower():
                print(f"[Leap] {msg}")

            print(f"[Pilot] Action: \033[94m{action.primitive.upper()} {action.params}\033[0m")
            if getattr(action, "reasoning", None):
                print(f"[Pilot] Reason: {action.reasoning}")

            # LEAP
            result_info = robot.execute_action(action.primitive, action.params)

            # history (azioni)
            recent_actions.append({"primitive": action.primitive, "params": action.params})

            # FEEDBACK (safety)
            safety_status = result_info.get("safety_warning", "nominal")
            clamped = result_info.get("movement_clamped", False)

            # PROGRESS (eef delta)
            progress_ok, progress_note = assess_progress(action.primitive, action.params, result_info)
            recent_progress_ok.append(bool(progress_ok))

            if clamped:
                print(f"\033[91m[Result] BLOCKED (SAFETY CLAMP)! Safety: {safety_status}\033[0m")
                last_action_report = (
                    f"CRITICAL FAILURE: Last action ({action.primitive} {action.params}) hit safety limits. "
                    f"{progress_note}. Robot is hitting something. Try base yaw / base x / torso."
                )
            elif safety_status != "nominal":
                last_action_report = f"WARNING: Safety alert ({safety_status}). {progress_note}. Be careful."
            elif not progress_ok:
                print(f"\033[91m[Result] NO PROGRESS\033[0m -> {progress_note}")
                last_action_report = (
                    f"NO PROGRESS: Last action ({action.primitive} {action.params}) produced almost no movement. "
                    f"{progress_note}. "
                    "Do NOT repeat the same action. Try: base yaw rotate OR base x approach OR torso lift. "
                    "If you were moving left, try rotating or moving forward instead."
                )
            else:
                last_action_report = f"Last action ({action.primitive} {action.params}) was SUCCESSFUL. {progress_note}."

    except KeyboardInterrupt:
        print("\n[System] Interrupted.")
    except Exception as e:
        print(f"\n[System] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()
        robot.close()


if __name__ == "__main__":
    main()
