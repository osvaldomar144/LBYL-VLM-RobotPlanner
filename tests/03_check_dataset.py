import h5py
import cv2
import numpy as np
import os
import mani_skill2.envs
import gymnasium as gym
import json

def check_demonstrations():
    # Percorso del dataset
    dataset_path = "demos/v0/rigid_body/PickCube-v0/trajectory.h5"
    output_dir = "dataset_preview"
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(dataset_path):
        print(f"ERRORE: Non trovo il file {dataset_path}.")
        return

    print(f">>> Apertura dataset: {dataset_path}")
    h5_file = h5py.File(dataset_path, "r")
    
    # --- 1. RILEVAMENTO AUTOMATICO CONFIGURAZIONE ---
    traj_ids = list(h5_file.keys())
    print(f">>> Trovate {len(traj_ids)} dimostrazioni.")
    
    # Analizziamo la prima traiettoria per capire il formato
    traj_id = "traj_0"
    grp = h5_file[traj_id]
    actions = grp["actions"][:]
    action_dim = actions.shape[-1]
    
    print(f">>> Dimensione Azioni rilevata: {action_dim}")
    
    # LOGICA DI FALLBACK
    # Se non c'è env_info, impostiamo noi i valori sapendo cosa abbiamo scaricato.
    env_id = "PickCube-v0"
    
    # Se le azioni sono 8 (7 giunti + 1 pinza), è Joint Control.
    # Se sono 7 (3 pos + 3 rot + 1 pinza), è End-Effector Control.
    if action_dim == 8:
        print(">>> Rilevato controllo GIUNTI (Joint Control).")
        control_mode = "pd_joint_delta_pos"
    else:
        print(">>> Rilevato controllo END-EFFECTOR.")
        control_mode = "pd_ee_delta_pose"

    # --- 2. CREAZIONE AMBIENTE ---
    print(f">>> Creazione ambiente {env_id} con mode={control_mode}...")
    env = gym.make(
        env_id, 
        obs_mode="rgbd", 
        control_mode=control_mode, 
        render_mode="rgb_array"
    )
    
    # Reset
    obs, _ = env.reset()
    
    # Caricamento Stato Esatto (CRUCIALE per replay fedele)
    if "env_states" in grp:
        print(">>> Caricamento stato iniziale esatto...")
        env_states = grp["env_states"][:]
        env.set_state(env_states[0])
    
    print(f">>> Generazione video di preview ({len(actions)} steps)...")
    
    frames = []
    
    for i, action in enumerate(actions):
        # Eseguiamo l'azione
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Sincronizzazione fisica precisa per evitare errori di accumulo
        if "env_states" in grp and i + 1 < len(env_states):
             env.set_state(env_states[i+1])

        rgb = env.render()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        frames.append(bgr)

    h5_file.close()
    
    # Salvataggio Video
    if frames:
        h, w, _ = frames[0].shape
        out_path = f'{output_dir}/demo_preview.mp4'
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (w, h))
        for f in frames:
            out.write(f)
        out.release()
        env.close()
        print(f">>> SUCCESSO! Video salvato in: {out_path}")
        print(">>> Controlla il video: deve mostrare un FRANKA che prende il cubo.")
    else:
        print(">>> Errore: Nessun frame generato.")

if __name__ == "__main__":
    check_demonstrations()