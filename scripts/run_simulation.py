import sys
import os
import time
import numpy as np
import robocasa
import robosuite
from PIL import Image

# Aggiungi src al path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from planner import VLMPlanner
from policy import VLAPolicy

# --- CONFIGURAZIONE CAMERA ---
CAMERA_NAME = "robot0_agentview_right" 

# --- CONFIGURAZIONE FISICA ---
# NUOVO FLAG: SWAP_XY
# Se True: Scambia X e Y. Necessario quando la camera è laterale (90 gradi).
# VLA X (Profondità) -> Robot Y (Lato)
# VLA Y (Lato) -> Robot X (Avanti)
SWAP_XY = True  

# Dopo lo scambio, applichiamo i segni.
# Con camera a destra:
# - VLA "Destra" (Immagine) corrisponde a Robot "Avanti" (+X) -> INVERT_X = False
# - VLA "Profondità" (Immagine) corrisponde a Robot "Sinistra" (+Y) -> INVERT_Y = False (di solito)
INVERT_X = False  
INVERT_Y = False  # Proviamo False dopo lo swap. Se va a destra invece che sinistra, metti True.
INVERT_Z = False  

# GUADAGNI
XY_GAIN = 100.0   
Z_GAIN = 30.0     
ROT_GAIN = 30.0   

def adapt_action(vla_action, env_action_dim):
    final_action = np.zeros(env_action_dim)
    
    # Raw Output [x, y, z, ...]
    raw_pos = vla_action[:3]
    raw_rot = vla_action[3:6]
    
    target_pos = np.zeros(3)
    
    # --- 1. SCAMBIO ASSI (ROTATION 90°) ---
    if SWAP_XY:
        # Mappatura per camera laterale DESTRA
        # L'asse Y dell'immagine (destra/sinistra) diventa l'asse X del robot (avanti/indietro)
        # L'asse X dell'immagine (profondità) diventa l'asse Y del robot (laterale)
        target_pos[0] = raw_pos[1]  # Robot X = VLA Y
        target_pos[1] = raw_pos[0]  # Robot Y = VLA X
        target_pos[2] = raw_pos[2]  # Z resta Z
    else:
        target_pos = raw_pos.copy()

    # --- 2. CORREZIONE SEGNI ---
    if INVERT_X: target_pos[0] = -target_pos[0]
    if INVERT_Y: target_pos[1] = -target_pos[1]
    if INVERT_Z: target_pos[2] = -target_pos[2]
    
    # --- 3. APPLICAZIONE GAIN ---
    amplified_pos = np.array([
        target_pos[0] * XY_GAIN,
        target_pos[1] * XY_GAIN,
        target_pos[2] * Z_GAIN
    ])
    
    # --- 4. NOISE ANTI-STUCK ---
    if np.linalg.norm(amplified_pos) < 0.05:
        amplified_pos += np.random.normal(0, 0.01, 3)

    # Clip
    final_pos = np.clip(amplified_pos, -1.0, 1.0)
    final_rot = np.clip(raw_rot * ROT_GAIN, -1.0, 1.0)
    
    # Assegnazione
    final_action[0:3] = final_pos
    final_action[3:6] = final_rot 
    
    # Gripper
    vla_gripper_prob = vla_action[6]
    final_action[6] = -1.0 if vla_gripper_prob > 0.5 else 1.0
    
    # Base/Torso Fermi
    final_action[7:10] = [0.0, 0.0, 0.0]
    if env_action_dim > 10:
        final_action[10:] = 0.0
        
    return final_action, target_pos

def main():
    ENV_NAME = "PnPCounterToCab" 
    ROBOT_NAME = "PandaMobile"
    
    print(f"=== AVVIO SIMULAZIONE (Camera: {CAMERA_NAME}) ===")
    print(f"SwapXY: {SWAP_XY} | InvX: {INVERT_X} | InvY: {INVERT_Y}")

    # 1. Caricamento Modelli
    planner = VLMPlanner()
    policy = VLAPolicy()

    # 2. Setup Ambiente
    controller_config = {
        "type": "OSC_POSE",
        "input_max": 1,
        "input_min": -1,
        "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
        "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
        "kp": 150,
        "damping_ratio": 1,
        "impedance_mode": "fixed",
        "kp_limits": [0, 300],
        "damping_ratio_limits": [0, 10],
        "uncouple_pos_ori": True,
        "control_delta": True,
        "interpolation": None,
        "ramp_ratio": 0.2
    }
    
    try:
        env = robosuite.make(
            env_name=ENV_NAME,
            robots=[ROBOT_NAME],
            controller_configs=controller_config,
            has_renderer=True,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names=[CAMERA_NAME],
            camera_heights=256,
            camera_widths=256,
            render_camera=CAMERA_NAME 
        )
    except Exception as e:
        print(f"[ERRORE] {e}")
        return

    obs = env.reset()
    action_dim = env.action_dim
    
    # --- FASE PREVIEW ---
    print("\n[INFO] Rendering anteprima...")
    for _ in range(30):
        env.render()
        env.step(np.zeros(action_dim)) 
    
    # 3. Input Utente
    print("\n[SCENA PRONTA]")
    user_instruction = input(">> Inserisci comando (es. 'pick the sponge'): ")
    if not user_instruction: user_instruction = "pick the object"

    # 4. Planner
    print("\n[Planner] Generazione piano...")
    img_array = obs[f'{CAMERA_NAME}_image']
    img_array = np.flipud(img_array)
    current_img = Image.fromarray(img_array)
    
    current_img.save("debug_planner_view.png")
    
    plan_steps = planner.plan(current_img, user_instruction)
    print(f"[Planner] Tasks: {plan_steps}")

    # 5. Policy Loop
    for i, step_desc in enumerate(plan_steps):
        print(f"\n---> TASK {i+1}: '{step_desc}'")
        
        for t in range(120):
            # Get Image
            img_array = obs[f'{CAMERA_NAME}_image']
            img_array = np.flipud(img_array)
            img = Image.fromarray(img_array)

            # VLA Inference
            raw_action = policy.get_action(img, step_desc)
            
            # Adapter
            robocasa_action, corrected_pos = adapt_action(raw_action, action_dim)
            
            # --- DEBUG LOGGING DIREZIONALE ---
            if t % 5 == 0:
                # Decodifica basata sui comandi finali al robot
                dx, dy, dz = robocasa_action[0:3]
                
                # Robot Frame: X=Avanti, Y=Sinistra
                dir_list = []
                if dx > 0.1: dir_list.append("AVANTI (Robot)")
                elif dx < -0.1: dir_list.append("INDIETRO (Robot)")
                
                if dy > 0.1: dir_list.append("SINISTRA (Robot)")
                elif dy < -0.1: dir_list.append("DESTRA (Robot)")

                if dz > 0.1: dir_list.append("SU")
                elif dz < -0.1: dir_list.append("GIU")
                
                dir_str = " + ".join(dir_list) if dir_list else "FERMO"
                print(f"[Step {t:03}] Movimento: {dir_str}")

            # Step Env
            obs, reward, done, info = env.step(robocasa_action)
            env.render()
            
            if done: break
        
        if done: break

    print("Simulazione terminata.")
    env.close()

if __name__ == "__main__":
    main()