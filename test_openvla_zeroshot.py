import gymnasium as gym
import torch
import numpy as np
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
import mani_skill.envs
import os
import shutil
import time

# --- CONFIGURAZIONE V9.0 ---
ENV_ID = "PickClutterYCB-v1" # Ambiente con oggetti multipli
MODEL_PATH = "openvla/openvla-7b"
DEBUG_DIR = "debug_frames"

# ISTRUZIONE PRECISA
# "ball" è ambiguo. "baseball" è meglio per OpenVLA.
INSTRUCTION = "Pick up the baseball"

# SETUP GRAFICO & FISICO
MAX_STEPS = 500 
SAVE_EVERY_STEPS = 5
ACTION_SCALE = 20.0 

# --- CALIBRAZIONE (Manteniamo la V7/V8 che funzionava in discesa) ---
SWAP_XY = False   
INVERT_X = True   
INVERT_Y = True   
INVERT_Z = True   

# Gravità: La pallina è piccola, serve precisione. 
# Riduco leggermente il bias per evitare che si schianti sul tavolo troppo forte.
Z_BIAS = -0.05

# --- STICKY GRIPPER ---
GRIPPER_CLOSE_THRESHOLD = 0.5 
GRIPPER_OPEN_THRESHOLD = 0.95

np.set_printoptions(precision=3, suppress=True, linewidth=100)

def setup_debug_dir():
    if os.path.exists(DEBUG_DIR):
        shutil.rmtree(DEBUG_DIR)
    os.makedirs(DEBUG_DIR)
    print(f"[DEBUG] Cartella '{DEBUG_DIR}' pronta.")

def to_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return data

def get_image_from_obs(obs):
    rgb_data = None
    if 'sensor_data' in obs:
        for sensor_name, sensor_data in obs['sensor_data'].items():
            if 'rgb' in sensor_data:
                rgb_data = sensor_data['rgb']
                break     
    if rgb_data is None and 'rgb' in obs:
        rgb_data = obs['rgb']
    if rgb_data is None:
        raise ValueError("Impossibile trovare immagine RGB.")
    img_np = to_numpy(rgb_data)
    img_np = np.squeeze(img_np)
    return img_np

def get_ee_pose(env):
    try:
        tcp_pose = env.unwrapped.agent.tcp.pose
        return to_numpy(tcp_pose.p)
    except:
        return np.array([0,0,0])

def apply_axis_correction(action_xyz):
    x, y, z = action_xyz[0], action_xyz[1], action_xyz[2]
    
    if SWAP_XY: x, y = y, x
    if INVERT_X: x = -x
    if INVERT_Y: y = -y
    if INVERT_Z: z = -z 
    
    z += Z_BIAS
    return np.array([x, y, z])

# Stato globale gripper
is_gripper_closed = False 

def process_gripper_sticky(raw_action):
    global is_gripper_closed
    if not is_gripper_closed:
        if raw_action < GRIPPER_CLOSE_THRESHOLD:
            is_gripper_closed = True
            print(">>> GRIPPER: LOCK (Chiusura)")
    else:
        if raw_action > GRIPPER_OPEN_THRESHOLD:
            is_gripper_closed = False
            print(">>> GRIPPER: UNLOCK (Apertura)")
    return -1.0 if is_gripper_closed else 1.0

def main():
    print(f"\n[INIT] Avvio script V9.0 (Ray Tracing + Baseball Task)...")
    setup_debug_dir()
    
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    print("[INIT] Caricamento modello...")
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.to("cuda")
    device = model.device

    print(f"[INIT] Avvio Ambiente {ENV_ID} con Ray Tracing...")
    
    # --- ABILITAZIONE SHADER RT-FAST ---
    # Nota: Se crasha, rimuovi 'shader_dir'. Ma con la 3090 Ti dovrebbe volare.
    try:
        env = gym.make(
            ENV_ID, 
            obs_mode="rgb", 
            control_mode="pd_ee_delta_pose", 
            render_mode="human",
            max_episode_steps=MAX_STEPS,
            shader_dir="rt" # Abilita Ray Tracing (luci e ombre migliori)
        )
    except Exception as e:
        print(f"[WARNING] Shader RT non supportato o errore ({e}). Uso standard.")
        env = gym.make(ENV_ID, obs_mode="rgb", control_mode="pd_ee_delta_pose", render_mode="human", max_episode_steps=MAX_STEPS)

    obs, _ = env.reset()
    
    print(f"[TASK] Istruzione: '{INSTRUCTION}'")

    env.render()
    time.sleep(2) # Pausa per ammirare il Ray Tracing

    for step in range(MAX_STEPS):
        print(f"\n--- STEP {step} ---")
        
        # 1. ACQUISIZIONE
        rgb_image = get_image_from_obs(obs)
        if rgb_image.dtype != np.uint8:
             rgb_image = (rgb_image * 255).astype(np.uint8) if rgb_image.max() <= 1.0 else rgb_image.astype(np.uint8)
        image_pil = Image.fromarray(rgb_image)

        if step % SAVE_EVERY_STEPS == 0:
            image_pil.save(os.path.join(DEBUG_DIR, f"step_{step:03d}.png"))
        
        # 2. INFERENZA
        inputs = processor(text=INSTRUCTION, images=image_pil, return_tensors="pt").to(device, torch.float16)
        
        with torch.inference_mode():
            action = model.predict_action(**inputs, unnorm_key="bridge_orig")
        
        if isinstance(action, torch.Tensor):
            action_np = action.cpu().numpy()
        else:
            action_np = action
            
        # 3. CONTROLLO
        xyz_cmd = action_np[:3] * ACTION_SCALE
        xyz_cmd = apply_axis_correction(xyz_cmd)
        
        rpy_cmd = action_np[3:6] * 0.1 
        
        raw_gripper = action_np[6]
        gripper_cmd = process_gripper_sticky(raw_gripper)
        
        final_action = np.concatenate([xyz_cmd, rpy_cmd, [gripper_cmd]])
        
        state_str = "LOCKED" if gripper_cmd < 0 else "OPEN"
        print(f"[AI] Move: {xyz_cmd} | Gripper: {state_str}")
        
        # 4. STEP FISICO
        obs, reward, terminated, truncated, info = env.step(final_action)
        env.render() 
        
        if reward > 0.5:
            print("\n!!! SUCCESSO !!! Oggetto afferrato.")
            time.sleep(3)
            break

        if terminated or truncated:
            print(f"[END] Episodio terminato (Step {step}).")
            break

    env.close()

if __name__ == "__main__":
    main()