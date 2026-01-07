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
import shutil
import sapien.core as sapien
from collections import deque

# --- CONFIGURAZIONE ---
BASE_MODEL_ID = "openvla/openvla-7b"
ADAPTER_PATH = "checkpoints_franka_diamond/ckpt-4000" 

# NUOVO TASK: Oggetti Reali (YCB)
# Richiede: python -m mani_skill.utils.download_asset ycb
TASK_ID = "PickSingleYCB-v1" 

obs_mode = "rgbd"
control_mode = "pd_ee_delta_pose"

ACTION_MIN = -1.0 
ACTION_MAX = 1.0

# --- PARAMETRI ---
APPROACH_SCALE = 2.0      
SMOOTHING_WINDOW = 2

def hide_goal_marker(env):
    """Nasconde eventuali marker di aiuto (se presenti)."""
    try:
        # In YCB il goal potrebbe essere visualizzato diversamente o non esserci
        for actor in env.unwrapped.scene.get_all_actors():
            if "goal" in actor.name.lower() or "site" in actor.name.lower():
                pose = actor.get_pose()
                # Sposta sotto terra
                new_pose = sapien.Pose(p=[pose.p[0], pose.p[1], -100], q=pose.q)
                actor.set_pose(new_pose)
    except: pass

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
    debug_dir = "debug_run_logs"
    if os.path.exists(debug_dir): shutil.rmtree(debug_dir)
    os.makedirs(debug_dir, exist_ok=True)

    print(f">>> Caricamento Modello...")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(BASE_MODEL_ID, quantization_config=bnb_config, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.to("cuda")
    model.eval()
    
    print(f">>> Avvio Environment Reale ({TASK_ID})...")
    try:
        env = gym.make(TASK_ID, obs_mode=obs_mode, control_mode=control_mode, render_mode="rgb_array", max_episode_steps=1000)
    except Exception as e:
        print("\n❌ ERRORE: Hai scaricato gli asset YCB?")
        print("Esegui nel terminale: python -m mani_skill.utils.download_asset ycb")
        raise e
    
    print(">>> GENERAZIONE SCENA...")
    obs, _ = env.reset()
    hide_goal_marker(env)
    
    # --- FASE INTERATTIVA ---
    # Mostriamo l'immagine all'utente e chiediamo il prompt
    render_frame = env.render()
    if isinstance(render_frame, torch.Tensor): render_frame = render_frame.cpu().numpy()
    if render_frame.ndim == 4: render_frame = render_frame[0]
    if render_frame.max() <= 1.5: render_frame = (render_frame * 255).astype(np.uint8)
    else: render_frame = render_frame.astype(np.uint8)
    
    img_pil = Image.fromarray(render_frame)
    img_pil.save(f"{debug_dir}/PREVIEW_SCENE.png")
    
    cv2_img = cv2.cvtColor(render_frame, cv2.COLOR_RGB2BGR)
    cv2.imshow("COSA DEVO PRENDERE?", cv2_img)
    cv2.waitKey(1) # Aggiorna finestra
    
    print("\n" + "="*50)
    print("👀 GUARDA LA FINESTRA 'COSA DEVO PRENDERE?'")
    print("Identifica l'oggetto (es: barattolo giallo, scatola di cracker, ecc.)")
    print("Scrivi il comando in inglese (es: 'pick up the mustard bottle')")
    print("="*50)
    
    INSTRUCTION = input(">>> SCRIVI IL PROMPT QUI: ")
    print(f"\n>>> Obiettivo acquisito: '{INSTRUCTION}'")
    print(">>> AVVIO ROBOT...")
    
    # Inizia il loop
    action_history = deque(maxlen=SMOOTHING_WINDOW)
    consecutive_close_requests = 0
    step = 0
    
    try:
        while True:
            # Acquisizione
            render_frame = env.render() 
            if isinstance(render_frame, torch.Tensor): render_frame = render_frame.cpu().numpy()
            if render_frame.ndim == 4: render_frame = render_frame[0]
            if render_frame.max() <= 1.5: render_frame = (render_frame * 255).astype(np.uint8)
            else: render_frame = render_frame.astype(np.uint8)

            img_pil = Image.fromarray(render_frame)
            if step % 5 == 0: # Salva ogni 5 frame per non intasare
                img_pil.save(f"{debug_dir}/step_{step:04d}.png")

            cv2_img = cv2.cvtColor(render_frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("Robot Eye", cv2_img)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

            # Inferenza
            action_bins, raw_text = get_action_from_model(model, processor, img_pil, INSTRUCTION)
            
            if action_bins:
                raw_physics = de_discretize_actions(action_bins)
                action_history.append(raw_physics)
                avg_action = np.mean(np.array(action_history), axis=0)
                
                final_action = avg_action.copy()
                
                # --- LOGICA BLIND TRUST ---
                wants_to_close = final_action[6] < -0.2 
                
                if wants_to_close:
                    consecutive_close_requests += 1
                else:
                    consecutive_close_requests = 0
                
                if consecutive_close_requests >= 2: 
                    # ESEGUIAMO LA PRESA
                    status = "🔴 GRASPING"
                    final_action[6] = -1.0
                    final_action[:6] = final_action[:6] * 0.1 # Rallenta per stabilità
                    
                    if consecutive_close_requests > 5:
                         final_action[2] += 0.02 # Lift leggero
                else:
                    status = "🟢 APPROACH"
                    final_action[:6] = final_action[:6] * APPROACH_SCALE
                    final_action[6] = 1.0

                print(f"Step {step:03d} | {status} | Act: {np.round(final_action[:3], 3)}")
                
                obs, reward, terminated, truncated, info = env.step(final_action)
                
                if isinstance(info, dict) and "success" in info:
                    succ = info["success"].item() if isinstance(info["success"], torch.Tensor) else info["success"]
                    if succ:
                        print(f"\n🏆🏆🏆 VITTORIA! OGGETTO PRESO! 🏆🏆🏆")
                        img_pil.save(f"{debug_dir}/VICTORY_step_{step}.png")
                        time.sleep(3)
                        break
                    
                if terminated or truncated:
                    print(f">>> Episodio Finito. Riavvia lo script per un nuovo oggetto.")
                    break
            else:
                obs, _, _, _, _ = env.step(np.zeros(7))
            
            step += 1

    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()