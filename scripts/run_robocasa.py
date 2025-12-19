# scripts/run_robocasa.py
import sys
import os
import numpy as np
import robocasa
import robosuite
from robosuite.controllers import load_controller_config
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from planner import VLMPlanner
from policy import VLAPolicy

def main():
    # --- CONFIGURAZIONE ---
    # Scegli un environment valido di RoboCasa. Esempio: "PandaMobileKitchen"
    # Assicurati che 'robots' sia compatibile (es. PandaMobile)
    ENV_NAME = "PnPCounterToCab" # Esempio standard RoboCasa (Pick and Place)
    ROBOT_NAME = "PandaMobile" 
    
    # --- 1. CARICAMENTO MODELLI ---
    planner = VLMPlanner()
    policy = VLAPolicy()

    # --- 2. SETUP AMBIENTE ---
    print(f"Avvio ambiente {ENV_NAME}...")
    controller_config = load_controller_config(default_controller="OSC_POSE")
    
    env = robosuite.make(
        env_name=ENV_NAME,
        robots=[ROBOT_NAME],
        controller_configs=controller_config,
        has_renderer=True,
        has_offscreen_renderer=True, # Necessario per catturare immagini per la VLA
        use_camera_obs=True,
        camera_names=["agentview", "robot0_eye_in_hand"], # agentview è standard per OpenVLA
        camera_heights=256, # Risoluzione per OpenVLA (224x224 resize avviene nel modello)
        camera_widths=256,
        render_camera="agentview"
    )
    
    obs = env.reset()
    
    # Istruzione fittizia o presa dall'utente
    user_instruction = "Put the vegetable in the pot" 
    print(f"\nGOAL: {user_instruction}")

    # --- 3. PIANIFICAZIONE (VLM) ---
    # Converti osservazione in PIL Image
    # Nota: robosuite restituisce immagini capovolte verticalmente spesso, verifichiamo
    img_array = obs['agentview_image']
    # Flip vertical se necessario (spesso robosuite ha l'asse Y invertito nell'array numpy)
    img_array = np.flipud(img_array) 
    current_img = Image.fromarray(img_array)
    
    plan_steps = planner.plan(current_img, user_instruction)
    print(f"PIANO GENERATO: {plan_steps}")

    # --- 4. ESECUZIONE (VLA) ---
    for step_idx, task_desc in enumerate(plan_steps):
        print(f"--> Esecuzione Step {step_idx+1}: {task_desc}")
        
        # Loop di controllo per ogni sub-task (max 60 step per azione per evitare blocchi)
        for _ in range(60): 
            # Cattura immagine corrente
            img_array = obs['agentview_image']
            img_array = np.flipud(img_array)
            img = Image.fromarray(img_array)

            # Inferenza VLA
            raw_action = policy.get_action(img, task_desc)
            
            # --- ADAPTER (Cruciale) ---
            # OpenVLA output (7): [x, y, z, roll, pitch, yaw, gripper]
            # RoboCasa PandaMobile input: Dipende dalla config, ma spesso include la base.
            # Se il robot ha 7 DOF braccio + 2/3 Base + 1 Gripper = ~10-11
            
            # ESEMPIO DI ADAPTER SEMPLICE (SOLO BRACCIO, BASE FERMA)
            # Creiamo un vettore di zeri della dimensione richiesta dall'env
            action_dim = env.action_dim
            final_action = np.zeros(action_dim)
            
            # Riempiamo la parte del braccio (assumendo i primi 6 siano pos/ori) e ultimo gripper
            # Attenzione: verifica la mappatura esatta di PandaMobile in RoboCasa docs
            # Solitamente: [x,y,z, ax,ay,az, gripper] (7) per il braccio.
            
            # Scaliamo l'azione se necessario (OpenVLA è aggressivo)
            final_action[:6] = raw_action[:6] * 1.0 
            # Gripper: OpenVLA [0,1], RoboCasa spesso [-1, 1]
            final_action[6] = 1 if raw_action[6] > 0.5 else -1 
            
            # Se la base è controllata dagli ultimi indici, lasciamoli a 0 per ora
            
            # Step simulazione
            obs, reward, done, info = env.step(final_action)
            env.render()
            
            if done: 
                print("Task completato dall'ambiente!")
                break
        
        if done: break

    env.close()

if __name__ == "__main__":
    main()