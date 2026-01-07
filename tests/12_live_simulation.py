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
ADAPTER_PATH = "checkpoints_franka_platinum/ckpt-9500" 

TASK_ID = "PickCube-v1"
obs_mode = "rgbd"
# PROVIAMO A CAMBIARE IL CONTROLLO IN WORLD FRAME SE POSSIBILE
# Se non funziona, torna a pd_ee_delta_pose
control_mode = "pd_ee_delta_pose" 

ACTION_MIN = -0.1
ACTION_MAX = 0.1

# --- PARAMETRI ESTREMI PER DEBUG ---
# Se il robot è timido, dobbiamo urlargli i comandi.
ACTION_SCALE = 25.0      # Moltiplicatore GIGANTE. Se dice 0.01, diventa 0.25 (25cm!)
SMOOTHING_WINDOW = 1     # DISABILITATO smoothing per vedere la reazione pura
ACTION_REPEAT = 1        # DISABILITATO repeat per vedere ogni singolo frame

def de_discretize_actions(bins, min_val=ACTION_MIN, max_val=ACTION_MAX):
    bins = np.array(bins)
    # Conversione standard
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
    print(f">>> Caricamento Modello...")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(BASE_MODEL_ID, quantization_config=bnb_config, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.to("cuda")
    model.eval()
    
    print(f">>> Avvio ManiSkill (Render RGB)...")
    env = gym.make(TASK_ID, obs_mode=obs_mode, control_mode=control_mode, render_mode="rgb_array", max_episode_steps=500)
    
    print(">>> RESET...")
    obs, _ = env.reset()
    
    INSTRUCTION = "pick up the red cube"
    os.makedirs("debug_final_test", exist_ok=True)
    
    print("\n>>> INIZIO DEBUG ESTREMO <<<")
    
    step = 0
    try:
        while True:
            # 1. CATTURA
            render_frame = env.render() 
            if isinstance(render_frame, torch.Tensor): render_frame = render_frame.cpu().numpy()
            if render_frame.ndim == 4: render_frame = render_frame[0]
            if render_frame.max() <= 1.5: render_frame = (render_frame * 255).astype(np.uint8)
            else: render_frame = render_frame.astype(np.uint8)

            img_pil = Image.fromarray(render_frame)
            
            # Mostra
            cv2_img = cv2.cvtColor(render_frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("Robot View", cv2_img)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

            # Salva OGNI frame per capire se cambia qualcosa
            img_pil.save(f"debug_final_test/step_{step}.png")

            # 2. INFERENZA
            action_bins, raw_text = get_action_from_model(model, processor, img_pil, INSTRUCTION)
            
            if action_bins:
                raw_physics = de_discretize_actions(action_bins)
                
                # Calcolo deviazione da "FERMO" (127)
                bins_array = np.array(action_bins[:6])
                diff_from_still = bins_array - 127
                is_trying_to_move = np.any(np.abs(diff_from_still) > 0)
                
                final_action = raw_physics.copy()
                
                # LOGICA DI BOOST DINAMICA
                # Se il modello prova a muoversi anche di poco (es: 128 o 126), SPINGIAMOLO!
                final_action[:6] = final_action[:6] * ACTION_SCALE
                
                # Pinza
                final_action[6] = final_action[6] * 2.0 

                print(f"Step {step} | Bins: {action_bins[:3]} | Diff: {diff_from_still[:3]} | Act: {np.round(final_action[:3],3)}")
                
                if not is_trying_to_move:
                    print("   ⚠️ IL MODELLO È BLOCCATO SU 127 (FERMO)")
                
                obs, reward, terminated, truncated, info = env.step(final_action)
                
                if terminated or truncated:
                    print(f">>> RESET (Success: {info.get('success', False)})")
                    obs, _ = env.reset()
                    time.sleep(1)
            else:
                print(f"⚠️ Confuso")
                obs, _, _, _, _ = env.step(np.zeros(7))
            
            step += 1

    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()