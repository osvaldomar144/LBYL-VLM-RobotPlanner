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
# PROVA CON IL CHECKPOINT 4000 o 2000 SE IL 6500 È TROPPO PIGRO
ADAPTER_PATH = "checkpoints_franka_adaptive/ckpt-2000" 

TASK_ID = "PickCube-v1"
obs_mode = "rgbd"
control_mode = "pd_ee_delta_pose"

ACTION_MIN = -1.0 
ACTION_MAX = 1.0

# --- PARAMETRI TUNING ---
# Invece di uno scale fisso, usiamo una curva.
# I piccoli movimenti hanno bisogno di una spinta enorme per vincere l'inerzia.
SMALL_ACTION_BOOST = 20.0  # Se il modello dice "muoviti poco", noi diciamo "MUOVITI!"
LARGE_ACTION_SCALE = 3.0   # Se il modello dice "corri", noi usiamo questo scale standard
GRIP_SCALE = 2.0
SMOOTHING_WINDOW = 2
ACTION_REPEAT = 1

def smart_de_discretize(bins):
    """
    Converte i bin in azioni fisiche con una logica intelligente.
    Bin 127 -> 0.0 (Zero assoluto)
    Bin 126/128 -> Boost enorme
    Bin distanti -> Scale normale
    """
    bins = np.array(bins)
    actions = []
    
    for i, b in enumerate(bins):
        # 1. Calcola il valore grezzo tra -1 e 1
        raw_val = (b / 255.0) * 2.0 - 1.0
        
        # 2. Correzione Zero Assoluto (Bin 127 è STOP)
        if b == 127:
            actions.append(0.0)
            continue
            
        # 3. Non-Linear Boost
        # Calcoliamo quanto siamo distanti dal centro (127)
        dist = abs(b - 127)
        
        if dist < 5: # Se siamo vicinissimi allo zero (es. 125, 126, 128, 129)
            # Applichiamo un BOOST enorme perché il modello è troppo timido
            final_val = raw_val * SMALL_ACTION_BOOST
        else:
            # Se il modello è deciso (es. 100 o 150), usiamo scale normale
            final_val = raw_val * LARGE_ACTION_SCALE
            
        actions.append(final_val)
        
    return np.array(actions)

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
    print(f">>> Caricamento Modello Tuning (Smart Scale)...")
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
    
    print("\n>>> INIZIO LOOP (Premi 'q' per uscire)...")
    
    step = 0
    try:
        while True:
            render_frame = env.render() 
            if isinstance(render_frame, torch.Tensor): render_frame = render_frame.cpu().numpy()
            if render_frame.ndim == 4: render_frame = render_frame[0]
            if render_frame.max() <= 1.5: render_frame = (render_frame * 255).astype(np.uint8)
            else: render_frame = render_frame.astype(np.uint8)

            img_pil = Image.fromarray(render_frame)
            cv2_img = cv2.cvtColor(render_frame, cv2.COLOR_RGB2BGR)
            
            # Info overlay
            cv2.putText(cv2_img, f"Step: {step}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Robot Eye", cv2_img)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

            # Salva frame di debug per vedere cosa vede il modello
            if step % 20 == 0:
                os.makedirs("debug_vision_check", exist_ok=True)
                img_pil.save(f"debug_vision_check/step_{step}.png")

            # Inferenza
            action_bins, raw_text = get_action_from_model(model, processor, img_pil, INSTRUCTION)
            
            if action_bins:
                # --- QUI LA MAGIA ---
                # Usiamo la nuova funzione smart
                raw_physics = smart_de_discretize(action_bins)
                
                action_history.append(raw_physics)
                avg_action = np.mean(np.array(action_history), axis=0)
                
                final_action = avg_action.copy()
                # Pinza (trattata separatamente)
                if final_action[6] < -0.5: final_action[6] = -1.0
                else: final_action[6] = 1.0
                
                grip_st = "✊" if final_action[6] < 0 else "✋"
                
                # Check se si muove davvero
                move_mag = np.linalg.norm(final_action[:3])
                status = "💤 FERMO" if move_mag == 0 else f"🚀 MOV ({move_mag:.2f})"
                
                print(f"Step {step} | Bins: {action_bins[:3]} | {status} | Act: {np.round(final_action[:3], 3)} | {grip_st}")
                
                for _ in range(ACTION_REPEAT):
                    obs, reward, terminated, truncated, info = env.step(final_action)
                    
                    success = False
                    if isinstance(info, dict) and "success" in info:
                        if isinstance(info["success"], torch.Tensor): success = info["success"].item()
                        else: success = info["success"]
                    
                    if success:
                        print(f"\n🏆🏆🏆 VITTORIA! 🏆🏆🏆")
                        time.sleep(2)
                        obs, _ = env.reset()
                        action_history.clear()
                        break
                    
                    if terminated or truncated:
                        print(f">>> Reset.")
                        obs, _ = env.reset()
                        action_history.clear()
                        break
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