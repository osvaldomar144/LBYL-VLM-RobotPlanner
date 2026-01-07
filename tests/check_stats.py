import json
import numpy as np

def check_stats():
    path = "dataset_finetuning_v2/dataset.json"
    print(f">>> Analisi statistiche azioni in {path}...")
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    all_actions = []
    for item in data:
        all_actions.append(item['action'])
    
    actions = np.array(all_actions)
    
    print(f"--- REPORT ---")
    print(f"Numero campioni: {len(actions)}")
    print(f"Minimo assoluto nel dataset: {actions.min():.6f}")
    print(f"Massimo assoluto nel dataset: {actions.max():.6f}")
    print(f"Media: {actions.mean():.6f}")
    print(f"Std Dev: {actions.std():.6f}")
    
    # Simuliamo la discretizzazione attuale
    print("\n--- SIMULAZIONE DISCRETIZZAZIONE ATTUALE [-1, 1] ---")
    bins = ((actions - (-1)) / (1 - (-1)) * 255).astype(int)
    unique_bins = np.unique(bins)
    print(f"Bin unici usati (su 256 disponibili): {len(unique_bins)}")
    print(f"I 10 bin più frequenti: {unique_bins[:10]} ...")
    
    if len(unique_bins) < 20:
        print("\n>>> DIAGNOSI: IL PROBLEMA È QUI! Usiamo troppi pochi bin.")
        print(">>> SOLUZIONE: Dobbiamo restringere il range (min_val, max_val).")

if __name__ == "__main__":
    check_stats()