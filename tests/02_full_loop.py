import torch
import cv2
import numpy as np
import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from PIL import Image
import os

def run_robot_loop():
    # --- 1. CONFIGURAZIONE TASK (Il più stabile per Demo) ---
    # Usiamo "eggplant_in_basket" perché l'oggetto è grande e facile da prendere
    task_name = "widowx_put_eggplant_in_basket"
    print(f">>> Inizializzazione Ambiente Realistico: {task_name}...")
    
    # OBS_MODE="rgbd" è mandatorio per SimplerEnv + OpenVLA
    env = simpler_env.make(task_name, render_mode="human")
    
    # --- 2. CARICAMENTO MODELLO VLA ---
    print(">>> Caricamento OpenVLA in 4-bit...")
    # Carichiamo il processore (gestisce le immagini)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    
    # Configurazione per far stare tutto nella 3090 Ti (4-bit quantization)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    # Carichiamo il modello vero e proprio
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", 
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True, 
        trust_remote_code=True,
        quantization_config=quantization_config
    )
    
    # --- 3. ISTRUZIONE SPECIFICA ---
    # Questa istruzione deve essere precisa per attivare il VLA
    instruction = "Put the eggplant in the basket" 
    print(f">>> Istruzione inviata al robot: '{instruction}'")

    # --- 4. LOOP DI CONTROLLO ---
    obs, reset_info = env.reset()
    
    # Setup cartella output
    os.makedirs("test_output", exist_ok=True)
    video_writer = None
    
    print(">>> Avvio simulazione (max 120 steps per dare tempo al robot)...")
    # Aumentiamo gli step a 120 perché il robot può essere lento
    for step in range(120):
        # A. Ottieni immagine (SimplerEnv format -> Numpy)
        image_np = get_image_from_maniskill2_obs_dict(env, obs) 
        image_pil = Image.fromarray(image_np)

        # B. Inizializzazione Video (Al primo frame)
        if video_writer is None:
            h, w = image_np.shape[:2]
            print(f">>> Rilevata risoluzione video: {w}x{h}")
            video_writer = cv2.VideoWriter(
                'test_output/robot_demo.mp4',
                cv2.VideoWriter_fourcc(*'mp4v'),
                10, (w, h)
            )

        # C. Chiedi al VLA cosa fare
        # Prompt standard per OpenVLA
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"
        inputs = processor(prompt, image_pil).to("cuda:0", dtype=torch.bfloat16)
        
        # Predizione azione
        action = model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
        
        # --- POLICY ADAPTER (TRUCCO) ---
        # Aumentiamo l'aggressività del movimento (primi 6 valori: xyz, rpy)
        # Lasciamo inalterata la pinza (ultimo valore)
        action[:-1] *= 2.0 

        # D. Esegui azione nel simulatore
        obs, reward, terminated, truncated, info = env.step(action)
        
        # E. Salva il frame
        frame_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        video_writer.write(frame_bgr) # Video
        
        # Salviamo anche qualche frame statico per debug ogni 10 step
        if step % 10 == 0:
            cv2.imwrite(f"test_output/frame_{step:03d}.png", frame_bgr)
        
        print(f"Step {step}: Azione {action[:3]}... Reward: {reward:.2f}")
        
        # Se reward è 1.0, il task è completato con successo!
        if reward > 0.9 or terminated or truncated:
            print(">>> SUCCESS! Task completato.")
            break

    # Chiusura e salvataggio
    if video_writer:
        video_writer.release()
    env.close()
    print(">>> Finito! Controlla 'test_output/robot_demo.mp4'.")

if __name__ == "__main__":
    run_robot_loop()