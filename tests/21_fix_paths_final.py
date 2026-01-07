import json
import os
from tqdm import tqdm

# --- CONFIGURAZIONE ---
# Il dataset con le azioni corrette (V3)
INPUT_JSON_V3 = "dataset_finetuning_v3/dataset.json"
# Dove lo risalviamo (sovrascriviamo lo stesso file per comodità)
OUTPUT_JSON_FINAL = "dataset_finetuning_v3/dataset.json"

# La cartella dove SONO FISICAMENTE le immagini (relativa alla root del progetto)
IMAGE_SOURCE_DIR = "dataset_finetuning_v2"

def main():
    print(f">>> Caricamento {INPUT_JSON_V3}...")
    with open(INPUT_JSON_V3, 'r') as f:
        data = json.load(f)
    
    print(f"Trovati {len(data)} frame. Correzione percorsi immagini...")
    
    fixed_data = []
    for item in tqdm(data):
        # Il percorso originale nel JSON v2 era relativo alla cartella v2
        # Es: "images/ep0_step0.png" o "data/PickCube/..."
        orig_path = item['image_path']
        
        # Costruiamo il percorso reale dalla root del progetto
        # Es: "dataset_finetuning_v2/images/ep0_step0.png"
        full_real_path = os.path.join(IMAGE_SOURCE_DIR, orig_path)
        
        # Ora calcoliamo il percorso relativo DALLA cartella v3 ALLA cartella v2
        # Dobbiamo salire di un livello (..)
        new_relative_path = os.path.join("..", full_real_path)
        
        # Aggiorniamo l'item
        item['image_path'] = new_relative_path
        fixed_data.append(item)
        
    print(f">>> Salvataggio dataset FINALE in {OUTPUT_JSON_FINAL}...")
    with open(OUTPUT_JSON_FINAL, 'w') as f:
        json.dump(fixed_data, f, indent=2)

    # VERIFICA
    print("\n--- VERIFICA ESEMPIO ---")
    print("Vecchio path (esempio):", data[0]['image_path'] if data else "N/A")
    print("Nuovo path (esempio):  ", fixed_data[0]['image_path'] if fixed_data else "N/A")
    print("-" * 30)
    print("Se il nuovo path inizia con '../dataset_finetuning_v2/', è CORRETTO.")
    print("Ora sei pronto per il training veloce! 🚀")

if __name__ == "__main__":
    main()