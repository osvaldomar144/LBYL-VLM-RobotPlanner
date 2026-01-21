import gymnasium as gym
import torch
import numpy as np
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
import mani_skill.envs
import os
import shutil
import time
import sapien.core as sapien

# --- CONFIGURAZIONE V13.0 ---
ENV_ID = "PickClutterYCB-v1"
MODEL_PATH = "openvla/openvla-7b"
DEBUG_DIR = "debug_frames"
INSTRUCTION = "Pick up the baseball"

MAX_STEPS = 500 
SAVE_EVERY_STEPS = 5
ACTION_SCALE = 20.0 

# --- RICALIBRAZIONE ASSI (PER CAMERA FRONTALE) ---
SWAP_XY = False   
INVERT_X = True    # Avanti/Indietro sembrava ok
INVERT_Y = False   # <--- CAMBIATO! (Era True). Ora seguiamo il modello.
INVERT_Z = True    # Discesa ok
Z_BIAS = -0.05

# Gripper
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

def apply_axis_correction(action_xyz):
    x, y, z = action_xyz[0], action_xyz[1], action_xyz[2]
    if SWAP_XY: x, y = y, x
    if INVERT_X: x = -x
    if INVERT_Y: y = -y
    if INVERT_Z: z = -z 
    z += Z_BIAS
    return np.array([x, y, z])

is_gripper_closed = False 
def process_gripper_sticky(raw_action):
    global is_gripper_closed
    if not is_gripper_closed:
        if raw_action < GRIPPER_CLOSE_THRESHOLD:
            is_gripper_closed = True
            print(">>> GRIPPER: LOCK")
    else:
        if raw_action > GRIPPER_OPEN_THRESHOLD:
            is_gripper_closed = False
            print(">>> GRIPPER: UNLOCK")
    return -1.0 if is_gripper_closed else 1.0

def setup_bridge_camera_view(env):
    """Sposta le camere nella posizione BridgeData (Frontale-Alto)."""
    print("[CAMERA] Tentativo di spostamento camere...")
    found = False
    new_pose = sapien.Pose(p=[0.6, 0.0, 0.6], q=[0.9238, 0.0, 0.3826, 0.0])
    
    try:
        scene = env.unwrapped.scene
        cameras = scene.get_cameras()
        for i, cam in enumerate(cameras):
            cam.set_local_pose(new_pose)
            print(f"[CAMERA] Spostata camera #{i}: {cam.name}")
            found = True
        if not found: print("[WARNING] Nessuna camera trovata.")
    except Exception as e:
        print(f"[WARNING] Errore setup camera: {e}")

def get_render_rgb(env):
    img = env.unwrapped.render_rgb_array()
    img = to_numpy(img) # Fix crash
    img = img.squeeze() 
    if img.dtype != np.uint8:
        if img.max() <= 1.0: img = (img * 255).astype(np.uint8)
        else: img = img.astype(np.uint8)
    return img

def main():
    print(f"\n[INIT] Avvio script V13.0 (Corrected Y-Axis)...")
    setup_debug_dir()
    
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, low_cpu_mem_usage=True, trust_remote_code=True
    ).to("cuda")
    device = model.device

    print(f"[INIT] Avvio Ambiente {ENV_ID} con RT...")
    try:
        env = gym.make(ENV_ID, obs_mode="rgb", control_mode="pd_ee_delta_pose", render_mode="human", max_episode_steps=MAX_STEPS, shader_dir="rt")
    except:
        env = gym.make(ENV_ID, obs_mode="rgb", control_mode="pd_ee_delta_pose", render_mode="human", max_episode_steps=MAX_STEPS)

    obs, _ = env.reset()
    setup_bridge_camera_view(env)
    
    print(f"[TASK] Istruzione: '{INSTRUCTION}'")
    env.render()
    time.sleep(2) 

    for step in range(MAX_STEPS):
        print(f"\n--- STEP {step} ---")
        
        rgb_image = get_render_rgb(env)
        image_pil = Image.fromarray(rgb_image)

        if step % SAVE_EVERY_STEPS == 0:
            image_pil.save(os.path.join(DEBUG_DIR, f"step_{step:03d}.png"))
        
        inputs = processor(text=INSTRUCTION, images=image_pil, return_tensors="pt").to(device, torch.float16)
        with torch.inference_mode():
            action = model.predict_action(**inputs, unnorm_key="bridge_orig")
        
        if isinstance(action, torch.Tensor): action_np = action.cpu().numpy()
        else: action_np = action
            
        # Controllo
        xyz_cmd = apply_axis_correction(action_np[:3] * ACTION_SCALE)
        rpy_cmd = action_np[3:6] * 0.1 
        gripper_cmd = process_gripper_sticky(action_np[6])
        final_action = np.concatenate([xyz_cmd, rpy_cmd, [gripper_cmd]])
        
        state_str = "LOCKED" if gripper_cmd < 0 else "OPEN"
        print(f"[AI] Move: {xyz_cmd} | Gripper: {state_str}")
        
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