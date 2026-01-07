import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm

# --- CONFIGURAZIONE DIAMOND 💎 ---
DATASET_PATH = "dataset_finetuning_v3/dataset.json" 
OUTPUT_DIR = "checkpoints_franka_diamond"

# Parametri Iper-Ottimizzati
BATCH_SIZE = 4          
GRAD_ACCUMULATION = 4   
LEARNING_RATE = 5e-5    
MAX_LENGTH = 512
EPOCHS = 1              
SAVE_STEPS = 500        
DEBUG_EVERY = 50        # Ogni 50 step vediamo cosa pensa il robot

# Range Azione ESTESO per Dataset V3 (Scaling x5 applicato)
ACTION_MIN = -1.0 
ACTION_MAX = 1.0

def discretize_actions(actions, n_bins=256, min_val=ACTION_MIN, max_val=ACTION_MAX):
    actions = np.clip(actions, min_val, max_val)
    bins = ((actions - min_val) / (max_val - min_val) * (n_bins - 1)).astype(int)
    return bins

class DiamondDataset(Dataset):
    def __init__(self, json_path, processor):
        print(f">>> Caricamento Dataset DIAMOND: {json_path}")
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        self.root_dir = os.path.dirname(json_path) 
        self.processor = processor
        self.prompt_template = "In: What action should the robot take to {}?\nOut:"
        
        print(f"--- DATASET PRONTO ---")
        print(f"Totale Frame: {len(self.data)}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Gestione path relativi (../v2/...)
        img_path = os.path.join(self.root_dir, item['image_path'])
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new('RGB', (224, 224), color='black')

        prompt_text = self.prompt_template.format(item['instruction'])
        
        action_bins = discretize_actions(np.array(item['action']))
        action_text = " ".join([str(b) for b in action_bins])
        full_text = prompt_text + " " + action_text
        
        inputs = self.processor(
            text=full_text,
            images=image,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        
        if "pixel_values" in inputs:
            pv = inputs["pixel_values"]
            if pv.dim() == 4: pv = pv.squeeze(0)
            if pv.shape[0] == 3: pv = torch.cat([pv, pv], dim=0) 
            inputs["pixel_values"] = pv

        input_ids = inputs["input_ids"][0]
        labels = input_ids.clone()
        
        # Masking
        try:
            pad_start = (input_ids == self.processor.tokenizer.pad_token_id).nonzero(as_tuple=True)[0][0].item()
        except IndexError:
            pad_start = len(input_ids)
            
        action_token_ids = self.processor.tokenizer(action_text, add_special_tokens=False).input_ids
        action_len = len(action_token_ids)
        action_start = max(0, pad_start - action_len)
        
        labels[:action_start] = -100 
        labels[pad_start:] = -100 
        
        inputs["input_ids"] = input_ids
        inputs["labels"] = labels
        if "attention_mask" in inputs: inputs["attention_mask"] = inputs["attention_mask"].squeeze(0)
            
        return inputs

def safe_debug_print(processor, logits, labels, step):
    print(f"\n⚡ DIAMOND DEBUG (Step {step}) ⚡")
    
    # Argmax sui logits per ottenere i token predetti
    pred_tokens_full = torch.argmax(logits[0], dim=-1)
    true_labels = labels[0]
    
    # --- FIX CRASH: Allineamento Lunghezze ---
    # OpenVLA aggiunge token immagine all'inizio. Labels ha solo testo.
    # Dobbiamo prendere solo la parte finale della predizione che corrisponde al testo.
    len_labels = true_labels.shape[0]
    if pred_tokens_full.shape[0] > len_labels:
        # Prendi solo gli ultimi N token
        pred_tokens = pred_tokens_full[-len_labels:]
    else:
        pred_tokens = pred_tokens_full
    # -----------------------------------------

    # Decodifica Target (Vero)
    valid_indices = true_labels != -100
    if valid_indices.sum() == 0:
        print("⚠️ Nessuna label valida in questo batch sample.")
        return

    target_tokens = true_labels[valid_indices]
    target_str = processor.decode(target_tokens, skip_special_tokens=True)
    
    # Decodifica Predizione (Robot)
    # Usiamo gli stessi indici validi per vedere cosa ha predetto il robot IN QUEI PUNTI
    pred_relevant = pred_tokens[valid_indices]
    pred_str = processor.decode(pred_relevant, skip_special_tokens=True)

    print(f"🎯 TARGET (Vero):  {target_str}")
    print(f"🤖 ROBOT  (Pred):  {pred_str}")
    
    # Analisi Numerica Rapida
    try:
        t_nums = [int(x) for x in target_str.split() if x.isdigit()]
        p_nums = [int(x) for x in pred_str.split() if x.isdigit()]
        
        if len(t_nums) >= 3 and len(p_nums) >= 3:
            diff = np.abs(np.array(t_nums[:3]) - np.array(p_nums[:3]))
            print(f"📊 Delta (primi 3 giunti): {diff}")
            if np.mean(diff) < 5:
                print("🏆 PRECISIONE ALTISSIMA!")
            elif np.mean(diff) < 20:
                print("✅ Buona direzione.")
            else:
                print("❌ Ancora lontano.")
    except:
        pass

    # Check "Sindrome del 127"
    if "127 127 127" in pred_str:
        print("⚠️  Warning: Il robot vuole stare FERMO (127).")
    else:
        print("🚀  Ottimo: Il robot predice MOVIMENTO.")
    
    print("-" * 50)

def train():
    print(f">>> AVVIO TRAINING DIAMOND 💎 (Lookahead Dataset)")
    
    model_id = "openvla/openvla-7b"
    print(">>> Caricamento Processor...")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    
    print(">>> Caricamento Modello (4-bit)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForVision2Seq.from_pretrained(
        model_id, quantization_config=bnb_config, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True
    )
    
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    
    peft_config = LoraConfig(
        r=32, lora_alpha=64, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    dataset = DiamondDataset(DATASET_PATH, processor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    print(f">>> Inizio Training Loop ({len(dataloader)} step per epoca)...")
    model.train()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    global_step = 0
    total_loss = 0
    
    for epoch in range(EPOCHS):
        progress = tqdm(dataloader, desc=f"Epoca {epoch+1}")
        
        for step, batch in enumerate(progress):
            batch = {k: v.to("cuda") for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs.loss / GRAD_ACCUMULATION
            loss.backward()
            
            total_loss += loss.item()
            
            if (step + 1) % GRAD_ACCUMULATION == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                
                if global_step % DEBUG_EVERY == 0:
                    safe_debug_print(processor, outputs.logits, batch["labels"], global_step)
                
                progress.set_postfix({"Loss": f"{total_loss:.4f}", "Step": global_step})
                total_loss = 0
                
                if global_step % SAVE_STEPS == 0:
                    print(f"\n>>> Salvataggio checkpoint {global_step}...")
                    model.save_pretrained(f"{OUTPUT_DIR}/ckpt-{global_step}")

    print(">>> Salvataggio Finale...")
    model.save_pretrained(f"{OUTPUT_DIR}/final")
    print(">>> Training Completato! 💎")

if __name__ == "__main__":
    train()