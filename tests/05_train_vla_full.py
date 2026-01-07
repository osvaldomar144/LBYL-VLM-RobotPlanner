import os
import json
import torch
import random
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import numpy as np
from tqdm import tqdm
import time

# --- CONFIGURAZIONE PLATINUM ---
DATASET_PATH = "dataset_finetuning_v2/dataset.json"
OUTPUT_DIR = "checkpoints_franka_platinum"

BATCH_SIZE = 8
GRAD_ACCUMULATION = 2
LEARNING_RATE = 4e-5  # Manteniamo il LR basso
MAX_LENGTH = 512
EPOCHS = 1
SAVE_STEPS = 500      # Salviamo più spesso dato che il dataset sarà più piccolo
DEBUG_EVERY = 20

# Range Azione
ACTION_MIN = -0.1
ACTION_MAX = 0.1

def discretize_actions(actions, n_bins=256, min_val=ACTION_MIN, max_val=ACTION_MAX):
    actions = np.clip(actions, min_val, max_val)
    bins = ((actions - min_val) / (max_val - min_val) * (n_bins - 1)).astype(int)
    return bins

class RobotDataset(Dataset):
    def __init__(self, json_path, processor):
        print(f">>> Caricamento JSON...")
        with open(json_path, 'r') as f:
            raw_data = json.load(f)
        
        self.root_dir = os.path.dirname(json_path)
        self.processor = processor
        self.prompt_template = "In: What action should the robot take to {}?\nOut:"
        
        self.data = []
        
        print(">>> APPLICAZIONE BILANCIAMENTO AGGRESSIVO...")
        kept_fast = 0
        kept_slow = 0
        discarded = 0
        
        for item in raw_data:
            action = np.array(item['action'])
            
            # Calcoliamo quanto si muove il braccio
            arm_movement = np.max(np.abs(action[:6]))
            gripper_change = abs(action[6]) > 0.8
            
            is_moving = arm_movement > 0.005 # Soglia 5mm (non più 1mm!)
            
            if is_moving or gripper_change:
                # Se si muove o usa la pinza, lo teniamo SEMPRE
                self.data.append(item)
                kept_fast += 1
            else:
                # Se è lento/fermo, lo teniamo solo il 10% delle volte
                # Questo riduce drasticamente i "127 127 127"
                if random.random() < 0.10: 
                    self.data.append(item)
                    kept_slow += 1
                else:
                    discarded += 1
                
        print(f"--- STATISTICHE DATASET ---")
        print(f"Totale Originale: {len(raw_data)}")
        print(f"Frame 'Veloci' (Tenuti): {kept_fast}")
        print(f"Frame 'Lenti'  (Tenuti): {kept_slow} (Downsampled 10%)")
        print(f"Frame Scartati: {discarded}")
        print(f"DATASET FINALE: {len(self.data)}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        try:
            image = Image.open(os.path.join(self.root_dir, item['image_path'])).convert("RGB")
        except:
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
            elif pv.shape[0] == 1: pv = pv.repeat(6, 1, 1)
            inputs["pixel_values"] = pv

        input_ids = inputs["input_ids"][0]
        labels = input_ids.clone()
        
        try:
            pad_start = (input_ids == self.processor.tokenizer.pad_token_id).nonzero(as_tuple=True)[0][0].item()
        except IndexError:
            pad_start = len(input_ids)
            
        action_token_ids = self.processor.tokenizer(action_text, add_special_tokens=False).input_ids
        action_start = max(0, pad_start - len(action_token_ids))
        labels[:action_start] = -100 
        labels[pad_start:] = -100    
        
        inputs["input_ids"] = input_ids
        inputs["labels"] = labels
        if "attention_mask" in inputs: inputs["attention_mask"] = inputs["attention_mask"].squeeze(0)
            
        return inputs

def safe_debug_print(processor, logits, labels, step):
    print(f"\n⚡ LIVE DEBUG (Step {step}) ⚡")
    pred_tokens_full = torch.argmax(logits[0], dim=-1)
    true_labels = labels[0]
    
    seq_len = true_labels.shape[0]
    if pred_tokens_full.shape[0] > seq_len:
        pred_tokens = pred_tokens_full[-seq_len:]
    else:
        pred_tokens = pred_tokens_full

    target_tokens_safe = true_labels.clone()
    target_tokens_safe[target_tokens_safe == -100] = processor.tokenizer.pad_token_id
    target_str = processor.decode(target_tokens_safe, skip_special_tokens=True)
    
    action_mask = (true_labels != -100)
    if action_mask.sum() > 0:
        pred_action_tokens = pred_tokens[action_mask]
        pred_str = processor.decode(pred_action_tokens, skip_special_tokens=True)
    else:
        pred_str = "[NULL]"

    print(f"🎯 TARGET:  {target_str}")
    print(f"🤖 ROBOT:   {pred_str}")
    
    if target_str.strip() == pred_str.strip():
        print("🏆 EXACT MATCH")
    elif "127 127 127" in pred_str:
        print("⚠️  STATICO (Questo deve succedere MENO spesso ora)")
    else:
        print("✅ MOVIMENTO")
    print("---------------------------------------------------------------")

def train():
    print(f">>> AVVIO TRAINING PLATINUM (Downsampling Attivo)")
    
    model_id = "openvla/openvla-7b"
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    
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
    
    dataset = RobotDataset(DATASET_PATH, processor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    print(f">>> Inizio Loop...")
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
                    model.save_pretrained(f"{OUTPUT_DIR}/ckpt-{global_step}")

    model.save_pretrained(f"{OUTPUT_DIR}/final")

if __name__ == "__main__":
    train()