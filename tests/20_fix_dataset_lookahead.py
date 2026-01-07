import json
import numpy as np
import os
from tqdm import tqdm

# --- CONFIGURAZIONE ---
INPUT_JSON = "dataset_finetuning_v2/dataset.json"
OUTPUT_JSON = "dataset_finetuning_v3/dataset.json"
OUTPUT_DIR = "dataset_finetuning_v3"

# QUANTO GUARDARE AVANTI?
# 10 significa: "Prevedi dove sarà la mano tra 10 frame"
# Con frame rate alti, questo è fondamentale per vedere movimento.
LOOKAHEAD_STEPS = 10 

# SCALING
# Aiuta a riempire i bin 0-255. 
ACTION_SCALE = 5.0 

def main():
    print(f">>> Caricamento {INPUT_JSON}...")
    with open(INPUT_JSON, 'r') as f:
        data = json.load(f)
    
    # 1. RAGGRUPPARE PER EPISODIO
    episodes = {}
    print(">>> Raggruppamento episodi...")
    
    for item in tqdm(data):
        # Estrarre l'ID episodio dal path immagine
        # Esempio path tipico: "data/PickCube-v1/trajectory_ep_0_step_0.png"
        # O: "data/PickCube-v1/trajectory_ep000_step0000.png"
        
        path_parts = item['image_path'].replace('.png', '').replace('.jpg', '').split('_')
        
        ep_id = "unknown"
        step_id = 0
        
        # Parsing robusto del nome file
        for i, part in enumerate(path_parts):
            if part.startswith("ep") and len(part) > 2 and part[2:].isdigit():
                ep_id = part
            elif part == "ep" and i+1 < len(path_parts) and path_parts[i+1].isdigit():
                ep_id = f"ep{path_parts[i+1]}"
            
            if part.startswith("step") and len(part) > 4 and part[4:].isdigit():
                step_id = int(part[4:])
            elif part == "step" and i+1 < len(path_parts) and path_parts[i+1].isdigit():
                step_id = int(path_parts[i+1])
        
        # Chiave univoca
        task_name = item['image_path'].split('/')[0] 
        full_ep_id = f"{task_name}_{ep_id}"
        
        if full_ep_id not in episodes:
            episodes[full_ep_id] = []
        
        item['_step_id'] = step_id
        episodes[full_ep_id].append(item)

    print(f"Trovati {len(episodes)} episodi distinti.")

    # 2. RICALCOLO AZIONI (Lookahead)
    new_dataset = []
    skipped_frames = 0
    
    print(f">>> Applicazione Lookahead ({LOOKAHEAD_STEPS} step) e Scaling (x{ACTION_SCALE})...")
    
    for ep_id, frames in tqdm(episodes.items()):
        # Ordina per tempo
        frames.sort(key=lambda x: x['_step_id'])
        
        num_frames = len(frames)
        
        for i in range(num_frames):
            # Non possiamo calcolare l'azione per gli ultimi N frame
            if i + LOOKAHEAD_STEPS >= num_frames:
                skipped_frames += 1
                continue
                
            current_frame = frames[i]
            
            # Calcolo Azione Accumulata
            accumulated_action = np.zeros(7) 
            
            # Pinza futura
            gripper_final = frames[i + LOOKAHEAD_STEPS]['action'][6]
            
            # Somma vettoriale dei delta per i prossimi N step
            for k in range(LOOKAHEAD_STEPS):
                next_act = np.array(frames[i+k]['action'])
                accumulated_action[:6] += next_act[:6]
            
            # Applica Scala
            accumulated_action[:6] = accumulated_action[:6] * ACTION_SCALE
            
            # Imposta stato pinza
            accumulated_action[6] = gripper_final

            # Clipa per sicurezza
            accumulated_action = np.clip(accumulated_action, -1.0, 1.0)
            
            # FIX ID: Generiamo un ID univoco se manca
            frame_id = current_frame.get('id', f"{ep_id}_step{current_frame['_step_id']}")

            new_item = {
                "id": frame_id,
                "image_path": current_frame['image_path'],
                "instruction": current_frame['instruction'],
                "action": accumulated_action.tolist()
            }
            new_dataset.append(new_item)

    # 3. SALVATAGGIO
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(new_dataset, f, indent=2)
        
    print("-" * 50)
    print(f"DATASET ORIGINALE: {len(data)} frame")
    print(f"NUOVO DATASET V3:  {len(new_dataset)} frame")
    print(f"Salvato in: {OUTPUT_JSON}")
    
    # Statistiche veloci
    acts = np.array([x['action'] for x in new_dataset])
    vels = np.linalg.norm(acts[:, :3], axis=1)
    print(f"Nuova Velocità Media: {np.mean(vels):.4f}")
    print(f"Nuova Velocità Max:   {np.max(vels):.4f}")
    print(f"Frame 'BUONI' (> 0.05): {np.sum(vels > 0.05)} ({np.sum(vels > 0.05)/len(new_dataset)*100:.1f}%)")

if __name__ == "__main__":
    main()