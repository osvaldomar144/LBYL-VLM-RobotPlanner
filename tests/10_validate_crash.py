import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from PIL import Image
import numpy as np
import json
import os
import random

# --- CONFIGURAZIONE ---
BASE_MODEL_ID = "openvla/openvla-7b"
ADAPTER_PATH = "checkpoints_franka_platinum/ckpt-9500" 
DATASET_PATH = "dataset_finetuning_v2/dataset.json"

# Range usato nel training
ACTION_MIN = -0.1
ACTION_MAX = 0.1

def discretize_actions(actions, n_bins=256, min_val=ACTION_MIN, max_val=ACTION_MAX):
    actions = np.clip(actions, min_val, max_val)
    bins = ((actions - min_val) / (max_val - min_val) * (n_bins - 1)).astype(int)
    return bins

def main():
    print(f">>> Caricamento Modello...")
    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForVision2Seq.from_pretrained(BASE_MODEL_ID, quantization_config=bnb_config, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.to("cuda")
    model.eval()

    print(">>> Caricamento Dataset e Ricerca Movimento...")
    with open(DATASET_PATH, 'r') as f:
        data = json.load(f)

    # FILTRO: Prendiamo solo esempi dove il braccio si muove TANTO (> 5cm)
    moving_examples = []
    print("Scansione per frame ad alta velocità...")
    for item in data:
        action = np.array(item['action'])
        # Se un asse si muove più di 0.02 (2cm), è un buon test
        if np.max(np.abs(action[:6])) > 0.02: 
            moving_examples.append(item)
            
    print(f"Trovati {len(moving_examples)} frame movimentati su {len(data)}.")
    
    # Ne prendiamo 5 a caso tra quelli movimentati
    test_samples = random.sample(moving_examples, 5)

    print("\n>>> INIZIO TEST COMPARATIVO (Verità vs Robot) <<<\n")
    
    for item in test_samples:
        instruction = item['instruction']
        image_path = os.path.join(os.path.dirname(DATASET_PATH), item['image_path'])
        
        try:
            image = Image.open(image_path).convert("RGB")
        except:
            continue

        # Calcoliamo la VERITÀ (Target)
        real_action_bins = discretize_actions(np.array(item['action']))
        target_str = " ".join([str(b) for b in real_action_bins])

        # Chiediamo al ROBOT
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"
        inputs = processor(prompt, images=image, return_tensors="pt").to("cuda", dtype=torch.bfloat16)
        if inputs["pixel_values"].shape[1] == 3:
             inputs["pixel_values"] = torch.cat([inputs["pixel_values"], inputs["pixel_values"]], dim=1)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, max_new_tokens=35, do_sample=False, pad_token_id=processor.tokenizer.pad_token_id
            )
        prediction = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].replace(prompt, "").strip()

        print(f"📝 Task: {instruction}")
        print(f"🎯 TARGET (Vero): {target_str}")
        print(f"🤖 ROBOT (Pred):  {prediction}")
        
        # Analisi Rapida
        if "127 127 127" in prediction:
            print("❌ ERRORE: Il robot è rimasto fermo mentre doveva muoversi.")
        else:
            print("✅ SUCCESSO: Il robot ha reagito!")
        print("-" * 50)

if __name__ == "__main__":
    main()