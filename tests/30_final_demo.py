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

# --- PARAMETRI GENERALIZZABILI ---
# Nessun hack specifico per il task. Solo scaling fisico.
ACTION_SCALE = 2.5      # Moltiplicatore per rendere le intenzioni del modello visibili
GRIP_SCALE = 2.0        # Aiuta a binarizzare la pinza (aperto/chiuso)
SMOOTHING_WINDOW = 2    # Minimo smoothing per stabilità
ACTION_REPEAT = 1       # 1:1 con il modello per massimo controllo

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
    print(f">>> Caricamento Modello Piro (No Bias)...")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(BASE_MODEL_ID, quantization_config=bnb_config, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.to("cuda")
    model.eval()
    
    print(f">>> Avvio ManiSkill...")
    env = gym.make(TASK_ID, obs_mode=obs_mode, control_mode=control_mode, render_mode="rgb_array", max_episode_steps=1000)
    
    print(">>> RESET...")
    obs, _ = env.reset()
    INSTRUCTION = "pick up the red cube"
    
    action_history = deque(maxlen=SMOOTHING_WINDOW)
    
    print("\n>>> INIZIO TEST PURO (Premi 'q' per uscire)...")
    
    step = 0
    try:
        while True:
            # 1. Visione
            render_frame = env.render() 
            if isinstance(render_frame, torch.Tensor): render_frame = render_frame.cpu().numpy()
            if render_frame.ndim == 4: render_frame = render_frame[0]
            if render_frame.max() <= 1.5: render_frame = (render_frame * 255).astype(np.uint8)
            else: render_frame = render_frame.astype(np.uint8)

            img_pil = Image.fromarray(render_frame)
            cv2_img = cv2.cvtColor(render_frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("VLA Vision", cv2_img)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

            # 2. Cervello
            action_bins, raw_text = get_action_from_model(model, processor, img_pil, INSTRUCTION)
            
            if action_bins:
                raw_physics = de_discretize_actions(action_bins)
                action_history.append(raw_physics)
                avg_action = np.mean(np.array(action_history), axis=0)
                
                final_action = avg_action.copy()
                
                # NESSUN BIAS DI TASK QUI. SOLO SCALING FISICO.
                final_action[:6] = final_action[:6] * ACTION_SCALE
                final_action[6] = final_action[6] * GRIP_SCALE
                
                grip_st = "✊ CHIUSO" if final_action[6] < -0.5 else "✋ APERTO"
                
                # Logghiamo Z specificamente per vedere se prova a salire
                z_val = final_action[2]
                z_arrow = "⬆️" if z_val > 0.01 else ("⬇️" if z_val < -0.01 else "➖")
                
                print(f"Step {step} | Act: {np.round(final_action[:3], 3)} ({z_arrow}) | {grip_st}")
                
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