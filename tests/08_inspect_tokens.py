import os
import json
import torch
import numpy as np
from transformers import AutoProcessor

# CONFIGURAZIONE (Stessa del Platinum)
DATASET_PATH = "dataset_finetuning_v2/dataset.json"
MODEL_ID = "openvla/openvla-7b"
ACTION_MIN = -0.1
ACTION_MAX = 0.1

def discretize_actions(actions, n_bins=256, min_val=ACTION_MIN, max_val=ACTION_MAX):
    actions = np.clip(actions, min_val, max_val)
    bins = ((actions - min_val) / (max_val - min_val) * (n_bins - 1)).astype(int)
    return bins

def inspect_tokens():
    print(f">>> Caricamento Processor: {MODEL_ID}")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer = processor.tokenizer
    
    with open(DATASET_PATH, 'r') as f:
        data = json.load(f)
    
    # Prendiamo un esempio a caso (es. index 100)
    item = data[100]
    
    # Simuliamo ESATTAMENTE la costruzione del Platinum Script
    prompt_text = "In: What action should the robot take to {}?\nOut:".format(item['instruction'])
    action_bins = discretize_actions(np.array(item['action']))
    action_text = " ".join([str(b) for b in action_bins])
    full_text = prompt_text + " " + action_text
    
    print(f"\n--- TESTO COMPLETO ---")
    print(f"'{full_text}'")
    
    # Tokenizzazione
    # Nota: OpenVLA aggiunge token immagine all'inizio, noi simuliamo solo la parte testo qui
    # per vedere se la maschera è allineata.
    inputs = tokenizer(full_text, return_tensors="pt")
    input_ids = inputs["input_ids"][0]
    
    # Ricostruzione logica di mascheramento Platinum
    labels = input_ids.clone()
    action_token_ids = tokenizer(action_text, add_special_tokens=False).input_ids
    
    # Calcolo indice
    pad_start = len(input_ids) 
    action_start = max(0, pad_start - len(action_token_ids))
    
    labels[:action_start] = -100
    
    print(f"\n--- ANALISI TOKEN-BY-TOKEN ---")
    print(f"{'IDX':<5} | {'ID':<8} | {'TOKEN':<15} | {'LABEL':<8} | {'STATUS'}")
    print("-" * 65)
    
    # Mostriamo solo gli ultimi 30 token (dove avviene la magia)
    start_view = max(0, len(input_ids) - 30)
    
    for i in range(start_view, len(input_ids)):
        token_id = input_ids[i].item()
        token_str = tokenizer.decode([token_id])
        label_val = labels[i].item()
        
        status = "✅ TRAIN" if label_val != -100 else "❌ MASK"
        token_str_safe = token_str.replace('\n', '\\n').replace(' ', '_')
        
        print(f"{i:<5} | {token_id:<8} | {token_str_safe:<15} | {label_val:<8} | {status}")

if __name__ == "__main__":
    inspect_tokens()