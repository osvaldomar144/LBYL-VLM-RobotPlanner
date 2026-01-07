import h5py
import numpy as np
import os
import gymnasium as gym
import mani_skill2.envs
import json
import cv2
from tqdm import tqdm
import shutil
import transforms3d
import glob

# --- CONFIGURAZIONE TASK ---
# Se il path è un file .h5, lo processa direttamente.
# Se è una cartella (come YCB), cerca tutti i .h5 dentro.
TASKS_TO_PROCESS = [
    # (Nome Task, Path, Istruzione Template)
    ("PickCube-v0", "demos/v0/rigid_body/PickCube-v0/trajectory.h5", "Pick up the red cube"),
    ("StackCube-v0", "demos/v0/rigid_body/StackCube-v0/trajectory.h5", "Stack the red cube on the green cube"),
    
    # Per YCB passiamo la cartella. Lo script troverà i file .h5 dentro.
    ("PickSingleYCB-v0", "demos/v0/rigid_body/PickSingleYCB-v0", "Pick up the {obj_name}") 
]

def clean_ycb_name(filename):
    # Esempio: "003_cracker_box.h5" -> "cracker box"
    name_no_ext = os.path.splitext(os.path.basename(filename))[0]
    # Rimuovi numeri iniziali (es. 003_)
    parts = name_no_ext.split("_")
    if parts[0].isdigit() or (len(parts[0]) == 3 and parts[0][0].isdigit()): 
        name = " ".join(parts[1:])
    else:
        name = " ".join(parts)
    return name

def calculate_delta_pose(current_pose, next_pose):
    delta_pos = next_pose.p - current_pose.p
    mat_curr = transforms3d.quaternions.quat2mat(current_pose.q)
    mat_next = transforms3d.quaternions.quat2mat(next_pose.q)
    mat_delta = mat_curr.T @ mat_next
    d_roll, d_pitch, d_yaw = transforms3d.euler.mat2euler(mat_delta, axes='sxyz')
    return np.array([delta_pos[0], delta_pos[1], delta_pos[2], d_roll, d_pitch, d_yaw])

def process_h5_file(env_id, filepath, instruction, output_folder, json_list, max_episodes=50):
    print(f"   -> Processing: {os.path.basename(filepath)} | Istruzione: '{instruction}'")
    
    # Setup Ambiente
    env = gym.make(env_id, obs_mode="rgbd", control_mode="pd_joint_delta_pos", render_mode="rgb_array")
    
    h5 = h5py.File(filepath, "r")
    keys = list(h5.keys())[:max_episodes]
    
    local_cnt = 0
    
    for ep_idx, traj_id in enumerate(keys):
        grp = h5[traj_id]
        
        # Reset & Load State
        env.reset()
        if "env_states" in grp:
            env_states = grp["env_states"][:]
            env.set_state(env_states[0])
            
        actions_joint = grp["actions"][:]
        
        for step_idx, joint_action in enumerate(actions_joint):
            # Trova TCP (Mano)
            ee_link = None
            for link in env.agent.robot.get_links():
                if link.name in ["panda_hand_tcp", "link_tcp", "panda_hand"]:
                    ee_link = link
                    break
            if ee_link is None: ee_link = env.agent.robot.get_links()[-1]

            # Posa Corrente & Render
            ee_pose_current = ee_link.get_pose()
            rgb = env.render()
            
            # Step Fisico
            env.step(joint_action)
            if "env_states" in grp and step_idx + 1 < len(env_states):
                env.set_state(env_states[step_idx+1])
            
            # Posa Next & Delta
            ee_pose_next = ee_link.get_pose()
            delta_6d = calculate_delta_pose(ee_pose_current, ee_pose_next)
            gripper = joint_action[-1]
            final_action = np.concatenate([delta_6d, [gripper]])
            
            # Salvataggio
            # Usiamo un nome univoco basato sull'hash del percorso per evitare collisioni
            safe_name = os.path.basename(filepath).replace(".h5", "")
            img_name = f"{env_id}_{safe_name}_ep{ep_idx:03d}_step{step_idx:04d}.png"
            img_path = os.path.join(output_folder, "images", img_name)
            
            cv2.imwrite(img_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            
            json_list.append({
                "image_path": f"images/{img_name}",
                "instruction": instruction,
                "action": final_action.tolist(),
                "dataset_source": env_id
            })
            local_cnt += 1
            
    h5.close()
    env.close()
    return local_cnt

def convert_dataset_v3():
    output_folder = "dataset_finetuning_v2" # Manteniamo la stessa cartella per unire tutto
    
    # Se vuoi ricominciare da capo, scommenta queste righe. 
    # Altrimenti AGGIUNGERA' i nuovi dati a quelli esistenti (utile se hai già fatto PickCube/Stack)
    if os.path.exists(output_folder):
       shutil.rmtree(output_folder)
    os.makedirs(f"{output_folder}/images", exist_ok=True)
    
    json_data = []
    total_frames = 0
    
    print(f">>> Inizio Creazione Dataset V3 in: {output_folder}")

    for env_id, path, template in TASKS_TO_PROCESS:
        
        # CASO 1: È un file singolo (PickCube, StackCube)
        if os.path.isfile(path):
            print(f">>> [FILE SINGOLO] {env_id}")
            cnt = process_h5_file(env_id, path, template, output_folder, json_data, max_episodes=50)
            total_frames += cnt
            
        # CASO 2: È una cartella (PickSingleYCB)
        elif os.path.isdir(path):
            print(f">>> [CARTELLA YCB] {env_id} - Scansione file...")
            h5_files = glob.glob(os.path.join(path, "*.h5"))
            h5_files.sort()
            
            # Limitiamo a 5 oggetti diversi per non far esplodere il tempo di conversione (puoi aumentarli)
            print(f"    Trovati {len(h5_files)} oggetti. Li processo TUTTI per la massima robustezza.")
            # h5_files = h5_files[:5] 
            
            for h5_path in h5_files:
                # Estrai nome oggetto dal file (es. 003_cracker_box -> cracker box)
                obj_name = clean_ycb_name(h5_path)
                dynamic_instruction = template.format(obj_name=obj_name)
                
                # Limitiamo a 20 episodi per oggetto YCB (così abbiamo varietà senza troppi dati uguali)
                cnt = process_h5_file(env_id, h5_path, dynamic_instruction, output_folder, json_data, max_episodes=20)
                total_frames += cnt
        
        else:
            print(f"!!! ERRORE: Path non trovato: {path}")

    # Salva JSON Finale
    with open(f"{output_folder}/dataset.json", "w") as f:
        json.dump(json_data, f, indent=2)
        
    print(f"\n>>> MISSION COMPLETE. Totale frames: {total_frames}")
    print(f">>> Dataset pronto per il Fine-Tuning!")

if __name__ == "__main__":
    convert_dataset_v3()