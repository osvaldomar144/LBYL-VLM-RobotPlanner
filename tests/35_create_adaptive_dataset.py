import json
import numpy as np
import os
from tqdm import tqdm

# --- CONFIGURAZIONE V4 ---
INPUT_JSON = "dataset_finetuning_v2/dataset.json" # Partiamo sempre dai dati Raw originali
OUTPUT_DIR = "dataset_finetuning_v4"
OUTPUT_JSON = f"{OUTPUT_DIR}/dataset.json"

# LOGICA ADATTIVA
LOOKAHEAD_MAX = 10      # Per movimenti ampi (Approach)
LOOKAHEAD_MIN = 1       # Per movimenti fini (Grasp)
ACTION_SCALE = 5.0      # Manteniamo lo scaling per aiutare il tokenizer

def main():
    print(f">>> Generazione Dataset V4 (ADAPTIVE)...")
    with open(INPUT_JSON, 'r') as f:
        data = json.load(f)
    
    # 1. Raggruppa per Episodi
    episodes = {}
    for item in tqdm(data, desc="Indexing"):
        path_parts = item['image_path'].replace('.png', '').replace('.jpg', '').split('_')
        ep_id = "unknown"
        step_id = 0
        
        # Parsing robusto
        for i, part in enumerate(path_parts):
            if part.startswith("ep") and len(part) > 2 and part[2:].isdigit():
                ep_id = part
            elif part == "ep" and i+1 < len(path_parts) and path_parts[i+1].isdigit():
                ep_id = f"ep{path_parts[i+1]}"
            if part.startswith("step") and len(part) > 4 and part[4:].isdigit():
                step_id = int(part[4:])
            elif part == "step" and i+1 < len(path_parts) and path_parts[i+1].isdigit():
                step_id = int(path_parts[i+1])
        
        task_name = item['image_path'].split('/')[0]
        full_ep_id = f"{task_name}_{ep_id}"
        
        if full_ep_id not in episodes: episodes[full_ep_id] = []
        item['_step_id'] = step_id
        episodes[full_ep_id].append(item)

    # 2. Elaborazione Intelligente
    new_dataset = []
    
    print(f">>> Elaborazione con Lookahead Dinamico...")
    for ep_id, frames in tqdm(episodes.items()):
        frames.sort(key=lambda x: x['_step_id'])
        num_frames = len(frames)
        
        for i in range(num_frames):
            current_frame = frames[i]
            
            # --- ANALISI DEL FUTURO ---
            # Determiniamo se siamo in una fase "delicata"
            is_critical_phase = False
            
            # Controllo 1: La pinza cambia stato nei prossimi step?
            current_grip = current_frame['action'][6]
            for k in range(1, LOOKAHEAD_MAX + 1):
                if i + k < num_frames:
                    future_grip = frames[i+k]['action'][6]
                    # Se la pinza cambia stato (apre->chiude o viceversa), siamo in fase critica
                    if np.sign(future_grip) != np.sign(current_grip):
                        is_critical_phase = True
                        break
            
            # Controllo 2: Velocità intrinseca molto bassa (Siamo vicini al target)
            raw_vel = np.linalg.norm(np.array(current_frame['action'][:3]))
            if raw_vel < 0.002: # 2mm
                is_critical_phase = True

            # --- DECISIONE LOOKAHEAD ---
            if is_critical_phase:
                steps_to_accumulate = LOOKAHEAD_MIN # 1 step (Massima fedeltà)
            else:
                steps_to_accumulate = LOOKAHEAD_MAX # 10 step (Massima velocità)

            # Check limiti array
            if i + steps_to_accumulate >= num_frames:
                continue # Saltiamo gli ultimissimi frame
            
            # --- CALCOLO AZIONE ---
            accumulated_action = np.zeros(7)
            
            # Somma i delta per il numero di step decisi
            for k in range(steps_to_accumulate):
                next_act = np.array(frames[i+k]['action'])
                accumulated_action[:6] += next_act[:6]
            
            # Se stiamo usando Lookahead 1, NON scaliamo x5, altrimenti diventa un salto enorme.
            # Se usiamo Lookahead 10, scaliamo.
            # IDEA MIGLIORE: Scaliamo SEMPRE per riempire il range -1,1, ma proporzionalmente.
            # Se Lookahead è basso, il movimento è piccolo, lo scale aiuta a renderlo visibile al tokenizer.
            accumulated_action[:6] = accumulated_action[:6] * ACTION_SCALE
            
            # La pinza prende lo stato finale del lookahead
            accumulated_action[6] = frames[i + steps_to_accumulate]['action'][6]
            
            # Clip
            accumulated_action = np.clip(accumulated_action, -1.0, 1.0)
            
            # Fix path immagine (per puntare alla cartella v2 originale)
            # Assumiamo che dataset_finetuning_v4 sia allo stesso livello di v2
            orig_path = current_frame['image_path']
            # Costruiamo ../dataset_finetuning_v2/images/...
            if not orig_path.startswith("../"):
                 fixed_path = os.path.join("..", "dataset_finetuning_v2", orig_path)
            else:
                 fixed_path = orig_path

            # ID Univoco
            frame_id = current_frame.get('id', f"{ep_id}_step{current_frame['_step_id']}")

            new_item = {
                "id": frame_id,
                "image_path": fixed_path,
                "instruction": current_frame['instruction'],
                "action": accumulated_action.tolist()
            }
            new_dataset.append(new_item)

    # Salvataggio
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(new_dataset, f, indent=2)
        
    print(f"Dataset V4 Creato: {len(new_dataset)} frame.")
    print(f"Salvato in: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()