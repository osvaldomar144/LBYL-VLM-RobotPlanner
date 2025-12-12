# scripts/run_vlm_pilot.py

import argparse
import time
import cv2
import numpy as np
import sys
import os
from PIL import Image

# ... (Gestione Path invariata) ...
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path: sys.path.append(project_root)
if current_dir not in sys.path: sys.path.append(current_dir)

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

def numpy_img_to_pil(np_img):
    return Image.fromarray(np_img)

def update_visual_monitor(img_np, step_name="Init"):
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    cv2.imshow("VLM Dual-Eye Monitor", img_bgr)
    cv2.waitKey(1)
    filename = f"live_view_{step_name}.png"
    cv2.imwrite(filename, img_bgr)
    return filename

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="PnPCounterToCab")
    parser.add_argument("--max_steps", type=int, default=100)
    args = parser.parse_args()

    print("--- VLM ROBOT PILOT: DUAL CAMERA VERSION ---")
    
    # Token alti per sicurezza
    vlm_config = VLMConfig(device="cuda", max_new_tokens=1024, verbose=False)
    vlm_client = VLMClient(vlm_config)

    print(f"[System] Loading Robot ({args.env_name})...")
    robot = VLMRobotInterface(env_name=args.env_name, render=True)

    # =========================================================================
    # FASE 1: DUAL VIEW SETUP
    # =========================================================================
    print("\n[Phase 1] Initializing Dual Views...")
    text_info, visual_dict = robot.get_context()
    
    # Recuperiamo entrambe le camere
    img_global = visual_dict.get('robot0_agentview_center')
    img_local = visual_dict.get('robot0_eye_in_hand')
    
    if img_global is None or img_local is None:
        print("[Error] Una delle due camere manca! Verifica robot_vlm_lib.py")
        robot.close()
        return

    # Uniamo le immagini affiancate (Stitching orizzontale)
    # Sinistra: Globale | Destra: Mano
    dual_view_img = np.hstack((img_global, img_local))

    update_visual_monitor(dual_view_img, step_name="00_START")
    
    print("---------------------------------------------------------------")
    print(" VEDI DUE IMMAGINI: SX=Globale, DX=Mano.")
    print("---------------------------------------------------------------")
    
    try:
        user_goal = input("INSERT GOAL > ").strip()
        if not user_goal: user_goal = "Look around"
    except EOFError: return

    print(f"\n[Mission Start] GOAL: \"{user_goal}\"")
    
    last_action_report = "None (Start of mission)"

    try:
        for step in range(args.max_steps):
            print(f"\n--- STEP {step+1}/{args.max_steps} ---")
            
            text_info, visual_dict = robot.get_context()
            img_global = visual_dict.get('robot0_agentview_center')
            img_local = visual_dict.get('robot0_eye_in_hand')
            
            if img_global is None: break
            
            # Creazione immagine composita per la VLM
            dual_view_img = np.hstack((img_global, img_local))
            
            update_visual_monitor(dual_view_img, step_name=f"{step+1:02d}")
            pil_image = numpy_img_to_pil(dual_view_img)

            # PLAN
            print(f"[Brain Input] Context: {last_action_report}")
            print("[Brain] Thinking...", end="", flush=True)
            
            plan = plan_next_step(
                image=pil_image,
                goal_instruction=user_goal,
                current_state=text_info,
                vlm_client=vlm_client,
                last_action_report=last_action_report
            )
            print(" Done.")

            if not plan.plan:
                print("[Brain] ??? Confusion. Skip.")
                continue

            action = plan.plan[0]
            print(f"[Pilot] Action: \033[94m{action.primitive.upper()} {action.params}\033[0m")
            if action.reasoning:
                print(f"[Pilot] Reason: {action.reasoning}")

            # LEAP
            result_info = robot.execute_action(action.primitive, action.params)

            # FEEDBACK
            safety_status = result_info.get('safety_warning', 'nominal')
            clamped = result_info.get('movement_clamped', False)
            
            if clamped:
                print(f"\033[91m[Result] BLOCKED! Safety: {safety_status}\033[0m")
                last_action_report = (
                    f"CRITICAL FAILURE: Last action ({action.primitive} {action.params}) hit safety limits. "
                    "Robot is hitting something. Try moving BASE or LIFTING TORSO."
                )
            elif safety_status != 'nominal':
                last_action_report = f"WARNING: Last action caused safety alert ({safety_status}). Be careful."
            else:
                last_action_report = f"Last action ({action.primitive}) was SUCCESSFUL."

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