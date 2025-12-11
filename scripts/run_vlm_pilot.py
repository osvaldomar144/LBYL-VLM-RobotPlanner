# scripts/run_vlm_pilot.py

import argparse
import time
import cv2
import numpy as np
import sys
import os
from PIL import Image

# --- GESTIONE DEI PATH (IMPORTANTE) ---
# Otteniamo il percorso assoluto della cartella corrente (scripts/)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Otteniamo il percorso della cartella padre (LBYL-VLM-RobotPlanner/)
project_root = os.path.dirname(current_dir)

# Aggiungiamo la root al sys.path così Python può trovare "vila_open"
if project_root not in sys.path:
    sys.path.append(project_root)

# Aggiungiamo anche la cartella corrente per sicurezza (per robot_vlm_lib)
if current_dir not in sys.path:
    sys.path.append(current_dir)

# --- ORA POSSIAMO IMPORTARE TUTTO ---

# 1. Importiamo la libreria Robot (che è nella stessa cartella 'scripts')
try:
    from robot_vlm_lib import VLMRobotInterface
except ImportError:
    # Fallback nel caso Python faccia i capricci coi path relativi
    from scripts.robot_vlm_lib import VLMRobotInterface

# 2. Importiamo i moduli VLM (che sono nella cartella 'vila_open')
try:
    from vila_open.vlm_client import VLMClient, VLMConfig
    from vila_open.planning_loop import plan_next_step
except ImportError as e:
    print(f"\n[ERRORE IMPORT] Non riesco a trovare 'vila_open'.")
    print(f"Assicurati di lanciare lo script dalla root del progetto oppure che la struttura sia corretta.")
    print(f"Path corrente analizzato: {sys.path}\n")
    raise e

def numpy_img_to_pil(np_img):
    """Converte immagine (H,W,3) Numpy -> PIL Image"""
    return Image.fromarray(np_img)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="PnPCounterToCab")
    parser.add_argument("--goal", type=str, default="Pick the apple and place it in the shelf")
    parser.add_argument("--max_steps", type=int, default=100)
    args = parser.parse_args()

    print("--- VLM ROBOT PILOT: INITIALIZING ---")

    # 1. Inizializzazione VLM Client
    print("[System] Loading VLM (LLaVA-OneVision)...")
    # Nota: Assicurati di avere memoria GPU a sufficienza. 
    # Se crasha per OOM, riduci max_new_tokens o usa device="cpu" (lento) per debug.
    vlm_config = VLMConfig(device="cuda", max_new_tokens=256, verbose=False)
    vlm_client = VLMClient(vlm_config)

    # 2. Inizializzazione Robot (Libreria Nostra)
    print(f"[System] Loading Robot Environment ({args.env_name})...")
    # render=True apre la finestra MuJoCo
    robot = VLMRobotInterface(env_name=args.env_name, render=True)

    print(f"\n[Mission] GOAL: \"{args.goal}\"")
    print("[Mission] Start Pilot Loop...")

    try:
        for step in range(args.max_steps):
            print(f"\n--- STEP {step+1}/{args.max_steps} ---")
            
            # A. LOOK (Prendi contesto dalla libreria)
            text_info, visual_dict = robot.get_context()
            
            # Selezione Camera:
            # - 'robot0_agentview_center' vede tutto il robot e il tavolo (buono per navigazione)
            # - 'robot0_eye_in_hand' vede solo davanti alla mano (buono per grasp finale)
            # Per ora usiamo quella globale che è più sicura per iniziare
            main_camera = 'robot0_agentview_center' 
            np_image = visual_dict.get(main_camera)
            
            if np_image is None:
                print(f"[Error] Immagine da {main_camera} non trovata!")
                break
                
            pil_image = numpy_img_to_pil(np_image)

            # B. PLAN (Chiedi alla VLM)
            print("[Brain] Thinking...")
            plan = plan_next_step(
                image=pil_image,
                goal_instruction=args.goal,
                current_state=text_info,
                vlm_client=vlm_client
            )

            # C. LEAP (Esegui azione)
            if not plan.plan:
                print("[Brain] ??? Nessun piano generato (Confusion). Riprovo...")
                continue

            action = plan.plan[0] # Eseguiamo solo la prima azione immediata
            
            # Log visivo carino
            print(f"[Pilot] Action: \033[94m{action.primitive.upper()} {action.params}\033[0m")
            if action.reasoning:
                print(f"[Pilot] Reason: {action.reasoning}")

            # Esecuzione fisica tramite la tua libreria
            result_info = robot.execute_action(action.primitive, action.params)

            # D. FEEDBACK CHECK
            if result_info.get('safety_warning') != 'nominal':
                print(f"\033[91m[Safety Alert] {result_info['safety_warning']}\033[0m")
            
            if result_info.get('movement_clamped'):
                print("\033[93m[Feedback] L'azione è stata rallentata perché il robot sbatteva!\033[0m")

    except KeyboardInterrupt:
        print("\n[System] Interrupted by user.")
    except Exception as e:
        print(f"\n[System] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        robot.close()
        print("[System] Shutdown.")

if __name__ == "__main__":
    main()