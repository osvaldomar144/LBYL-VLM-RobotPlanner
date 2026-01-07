import json
import numpy as np

# CONFIGURAZIONE UGUALE AL TRAINING
ACTION_MIN = -0.1
ACTION_MAX = 0.1

def discretize_actions(actions, n_bins=256, min_val=ACTION_MIN, max_val=ACTION_MAX):
    actions = np.clip(actions, min_val, max_val)
    bins = ((actions - min_val) / (max_val - min_val) * (n_bins - 1)).astype(int)
    return bins

def check_histogram():
    print(">>> Analisi Bilanciamento Dataset...")
    with open("dataset_finetuning_v2/dataset.json", 'r') as f:
        data = json.load(f)
    
    all_bins = []
    kept_count = 0
    
    print(">>> Applicazione Filtro & Discretizzazione...")
    for item in data:
        action = np.array(item['action'])
        
        # Stesso filtro del training
        if np.max(np.abs(action[:6])) > 1e-3 or abs(action[6]) > 0.8:
            bins = discretize_actions(action)
            all_bins.extend(bins[:6]) # Contiamo solo i giunti del braccio (non la pinza)
            kept_count += 1
            
    all_bins = np.array(all_bins)
    
    # Conteggio
    total_samples = len(all_bins)
    zeros_127 = np.sum(all_bins == 127)
    zeros_128 = np.sum(all_bins == 128)
    zeros = zeros_127 + zeros_128
    
    print(f"\n--- RISULTATO ---")
    print(f"Frame Totali analizzati: {kept_count}")
    print(f"Numeri Totali (Giunti):  {total_samples}")
    print(f"Quantità di '127/128' (Fermo): {zeros} ({zeros/total_samples*100:.2f}%)")
    print(f"Quantità di Movimento Reale:   {total_samples - zeros} ({(total_samples-zeros)/total_samples*100:.2f}%)")
    
    print("\n>>> INTERPRETAZIONE:")
    if (zeros/total_samples) > 0.8:
        print("❌ SQUILIBRIO CRITICO: Più dell'80% dei dati è 'Stai Fermo'.")
        print("   Il modello imparerà a non muoversi mai per abbassare la loss.")
        print("   SOLUZIONE: Dobbiamo usare il 'Balanced Downsampling'.")
    else:
        print("✅ BILANCIAMENTO OK: Il dataset è sano.")

if __name__ == "__main__":
    check_histogram()