import os
import json
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor

# --- CONFIGURAZIONE ---
DATASET_PATH = "dataset_finetuning_v2/dataset.json"
ACTION_MIN = -0.05
ACTION_MAX = 0.05
MAX_LENGTH = 512

def discretize_actions(actions, n_bins=256, min_val=ACTION_MIN, max_val=ACTION_MAX):
    actions = np.clip(actions, min_val, max_val)
    bins = ((actions - min_val) / (max_val - min_val) * (n_bins - 1)).astype(int)
    return bins

def debug_labels():
    print(f">>> DEBUG LABELS & MASKING")
    
    model_id = "openvla/openvla-7b"
    print(">>> Caricamento Tokenizer...")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    tokenizer = processor.tokenizer
    
    with open(DATASET_PATH, 'r') as f:
        data = json.load(f)
    
    # Prendiamo un esempio a caso (es. il numero 50)
    idx = 50 
    item = data[idx]
    
    print(f"\n--- ESEMPIO #{idx} ---")
    print(f"Istruzione: {item['instruction']}")
    print(f"Azione Raw: {item['action']}")
    
    # 1. Preparazione Testo
    prompt_template = "In: What action should the robot take to {}?\nOut:"
    prompt_text = prompt_template.format(item['instruction'])
    
    action_bins = discretize_actions(np.array(item['action']))
    action_text = " ".join([str(b) for b in action_bins])
    
    full_text = prompt_text + " " + action_text
    print(f"Testo Completo: '{full_text}'")
    
    # 2. Tokenization
    # Creiamo un'immagine nera dummy per far contento il processor
    image = Image.new('RGB', (224, 224), color='black')
    
    inputs = processor(
        text=full_text,
        images=image,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    
    input_ids = inputs["input_ids"][0]
    
    # 3. Logica di Mascheramento (La stessa dello script di training)
    labels = input_ids.clone()
    
    action_token_ids = tokenizer(action_text, add_special_tokens=False).input_ids
    num_action_tokens = len(action_token_ids)
    
    # Cerchiamo il padding
    try:
        pad_start = (input_ids == tokenizer.pad_token_id).nonzero(as_tuple=True)[0][0].item()
    except IndexError:
        pad_start = len(input_ids)
    
    # Calcolo inizio azione
    action_start = max(0, pad_start - num_action_tokens - 1) # <--- PUNTO CRITICO
    
    # Applichiamo la maschera
    labels[:action_start] = -100 
    labels[pad_start:] = -100    

    # 4. DECODIFICA PER VERIFICA
    print("\n--- COSA VEDE IL MODELLO ---")
    
    # Decodifichiamo i token che NON sono mascherati (-100)
    visible_tokens = labels.clone()
    visible_tokens[visible_tokens == -100] = tokenizer.pad_token_id # Sostituisco -100 con pad per decodificare
    
    decoded_target = tokenizer.decode(visible_tokens, skip_special_tokens=True)
    
    print(f"Tokens Totali Azione: {num_action_tokens}")
    print(f"Start Index calcolato: {action_start}")
    print(f"End Index (Pad Start): {pad_start}")
    
    print(f"\n>>> TARGET REALE (Quello su cui si calcola la Loss):")
    print(f"'{decoded_target}'")
    
    print("\n---------------------------------------------------")
    print("ANALISI:")
    if "Out:" in decoded_target:
        print("❌ ERRORE GRAVE: Il modello sta imparando a predire 'Out:'. Ecco perché la loss è bassa!")
    elif len(decoded_target.strip()) == 0:
        print("❌ ERRORE GRAVE: Il target è vuoto!")
    else:
        print("✅ MASCHERA OK: Il modello vede solo i numeri.")
        print(f"Controlla se i numeri sono tutti uguali (es. 127 127 127).")
        print(f"Bin generati: {action_bins}")

if __name__ == "__main__":
    debug_labels()