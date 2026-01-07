import torch
import numpy as np
import gymnasium as gym
import mani_skill.envs
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from PIL import Image
import time
import os
import cv2
from collections import deque

# --- CONFIGURAZIONE ---
BASE_MODEL_ID = "openvla/openvla-7b"
ADAPTER_PATH = "checkpoints_franka_diamond/ckpt-4000" 

TASK_ID = "PickCube-v1"
obs_mode = "rgbd"
control_mode = "pd_ee_delta_pose"

ACTION_MIN = -1.0 
ACTION_MAX = 1.0

# --- PARAMETRI DINAMICI ---
BASE_SCALE = 2.5        # Velocità di crociera (avvicinamento)
FINE_SCALE = 0.5        # Velocità di precisione (quando è vicino)
GRIP_THRESHOLD = -0.7   # Deve essere molto convinto per chiudere (-0.7 è più severo di -0.5)
SMOOTHING_WINDOW = 2    
ACTION_REPEAT = 1       

def de_discretize_actions(bins, min_val=ACTION_MIN, max_val=ACTION_MAX):
    bins = np.array(bins)
    actions = min_val + (bins / 255.0) * (max_val - min_val)
    return actions

def get_action_from_model(model, processor, image, instruction):
    prompt = f"In: What action should the robot take to {instruction}?\nOut:"
    inputs = processor(prompt, images=image, return_tensors="pt").to("cuda", dtype=torch.bfloat16)
    if inputs["pixel_values"].shape[1] == 3:
         inputs["pixel_values"] = torch.cat([inputs["pixel_values"], inputs["pixel_values"]], dim=1)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, max_new_tokens=45, do_sample=False, pad_token_id=processor.tokenizer.pad_token_id
        )
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    prediction = output_text.replace(prompt, "").strip()
    
    try:
        bin_list = [int(x) for x in prediction.split() if x.isdigit()]
        if len(bin_list) >= 7: return bin_list[:7], prediction
        else: return None, prediction
    except: return None, prediction

def main():
    print(f">>> Caricamento Modello Precisione...")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(BASE_MODEL_ID, quantization_config=bnb_config, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.to("cuda")
    model.eval()
    
    print(f">>> Avvio ManiSkill Multi-View...")
    # 'rgb_array' ci dà la vista principale. Per altre viste dobbiamo accedere alle camere interne.
    env = gym.make(TASK_ID, obs_mode=obs_mode, control_mode=control_mode, render_mode="rgb_array", max_episode_steps=1000)
    
    print(">>> RESET...")
    obs, _ = env.reset()
    INSTRUCTION = "pick up the red cube"
    
    action_history = deque(maxlen=SMOOTHING_WINDOW)
    
    # Stato interno per il controllo adattivo
    is_grasping = False
    
    print("\n>>> INIZIO DEMO (Premi 'q' per uscire)...")
    
    step = 0
    try:
        while True:
            # 1. CATTURA MULTI-VIEW (Trucco per debug umano)
            # Vista Principale (Laterale - Quella che vede il robot)
            render_main = env.render() 
            
            # Vista Secondaria (Dall'alto - Base Camera)
            # Proviamo a estrarla dall'osservazione se disponibile
            render_top = None
            if 'sensor_data' in obs and 'base_camera' in obs['sensor_data']:
                render_top = obs['sensor_data']['base_camera']['rgb']
            elif 'image' in obs and 'base_camera' in obs['image']:
                render_top = obs['image']['base_camera']['rgb']
                
            # Sanitizzazione Main View (per il modello)
            if isinstance(render_main, torch.Tensor): render_main = render_main.cpu().numpy()
            if render_main.ndim == 4: render_main = render_main[0]
            if render_main.max() <= 1.5: render_main = (render_main * 255).astype(np.uint8)
            else: render_main = render_main.astype(np.uint8)
            
            # Sanitizzazione Top View (per l'umano)
            if render_top is not None:
                if isinstance(render_top, torch.Tensor): render_top = render_top.cpu().numpy()
                if render_top.ndim == 4: render_top = render_top[0]
                if render_top.max() <= 1.5: render_top = (render_top * 255).astype(np.uint8)
                else: render_top = render_top.astype(np.uint8)
            else:
                render_top = np.zeros_like(render_main) # Nero se non disponibile

            # Composizione Video (Affiancati)
            img_pil = Image.fromarray(render_main)
            
            cv2_main = cv2.cvtColor(render_main, cv2.COLOR_RGB2BGR)
            cv2_top = cv2.cvtColor(render_top, cv2.COLOR_RGB2BGR)
            
            # Ridimensiona per affiancare bene
            h, w = cv2_main.shape[:2]
            cv2_top_resized = cv2.resize(cv2_top, (w, h))
            
            combined_view = np.hstack((cv2_main, cv2_top_resized))
            cv2.imshow("Robot Eye (Left) | Top View (Right)", combined_view)
            
            if cv2.waitKey(1) & 0xFF == ord('q'): break

            # 2. CERVELLO
            action_bins, raw_text = get_action_from_model(model, processor, img_pil, INSTRUCTION)
            
            if action_bins:
                raw_physics = de_discretize_actions(action_bins)
                action_history.append(raw_physics)
                avg_action = np.mean(np.array(action_history), axis=0)
                
                final_action = avg_action.copy()
                
                # --- LOGICA ADATTIVA (SENZA TRUCCHI DI STATO) ---
                # Usiamo solo l'output del modello per decidere la velocità
                
                # Se il modello vuole chiudere la pinza (è vicino all'oggetto) -> RALLENTA
                if avg_action[6] < -0.2: 
                    current_scale = FINE_SCALE
                    mode = "PRECISION"
                else:
                    current_scale = BASE_SCALE
                    mode = "FAST"
                
                final_action[:6] = final_action[:6] * current_scale
                
                # Logica Pinza Binaria (Più severa)
                if avg_action[6] < GRIP_THRESHOLD:
                    final_action[6] = -1.0 # Chiudi forte
                    grip_st = "✊"
                else:
                    final_action[6] = 1.0  # Apri forte
                    grip_st = "✋"
                
                # Logghiamo Z per vedere se risale
                z_arrow = "⬆️" if final_action[2] > 0.01 else ("⬇️" if final_action[2] < -0.01 else "➖")
                
                print(f"Step {step} | [{mode}] Act: {np.round(final_action[:3], 3)} {z_arrow} | {grip_st}")
                
                # 3. Esecuzione
                obs, reward, terminated, truncated, info = env.step(final_action)
                
                success = False
                if isinstance(info, dict) and "success" in info:
                    if isinstance(info["success"], torch.Tensor): success = info["success"].item()
                    else: success = info["success"]
                
                if success:
                    print(f"\n🏆🏆🏆 MISSIONE COMPIUTA! 🏆🏆🏆")
                    time.sleep(2)
                    obs, _ = env.reset()
                    action_history.clear()
                    
                if terminated or truncated:
                    print(f">>> Reset.")
                    obs, _ = env.reset()
                    action_history.clear()
            else:
                print(f"⚠️ {raw_text}")
                obs, _, _, _, _ = env.step(np.zeros(7))
            
            step += 1

    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()