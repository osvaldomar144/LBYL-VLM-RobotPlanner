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

# --- CONFIGURAZIONE V16.0 (SCENA ARREDATA FIX) ---
ENV_ID = "PickClutterYCB-v1"
MODEL_PATH = "openvla/openvla-7b"
DEBUG_DIR = "debug_frames"
INSTRUCTION = "Pick up the baseball"

MAX_STEPS = 500 
SAVE_EVERY_STEPS = 5
ACTION_SCALE = 20.0 

# Calibrazione
SWAP_XY = False   
INVERT_X = True   
INVERT_Y = False   
INVERT_Z = True   
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

# --- FIX: SCENE BUILDER CON SAPIEN 3 COMPATIBILITY ---
def create_material(renderer, color):
    """Helper per creare materiali in modo sicuro."""
    mat = renderer.create_material()
    mat.base_color = color
    mat.roughness = 0.8
    mat.metallic = 0.1
    return mat

def decorate_scene(env):
    """
    Costruisce la stanza usando la sintassi corretta per i materiali.
    """
    try:
        # Accesso a scena e renderer
        if hasattr(env.unwrapped, "scene"):
            scene = env.unwrapped.scene
        else:
            scene = env.scene # Fallback
            
        renderer = scene.renderer
        
        # Colori
        floor_color = [0.1, 0.1, 0.1, 1.0] # Grigio scuro
        wall_color = [0.85, 0.82, 0.75, 1.0] # Beige muro
        
        # Creiamo i materiali PRIMA (questo risolve il crash)
        mat_floor = create_material(renderer, floor_color)
        mat_wall = create_material(renderer, wall_color)

        # 1. Pavimento
        builder = scene.create_actor_builder()
        builder.add_box_visual(half_size=[5.0, 5.0, 0.1], material=mat_floor)
        floor = builder.build_static(name="room_floor")
        floor.set_pose(sapien.Pose(p=[0, 0, -0.1]))

        # 2. Muro Posteriore
        builder = scene.create_actor_builder()
        builder.add_box_visual(half_size=[0.1, 5.0, 3.0], material=mat_wall)
        wall_back = builder.build_static(name="wall_back")
        wall_back.set_pose(sapien.Pose(p=[1.5, 0, 1.5]))

        # 3. Muri Laterali
        builder = scene.create_actor_builder()
        builder.add_box_visual(half_size=[5.0, 0.1, 3.0], material=mat_wall)
        wall_left = builder.build_static(name="wall_left")
        wall_left.set_pose(sapien.Pose(p=[0, 2.0, 1.5]))
        
        builder = scene.create_actor_builder()
        builder.add_box_visual(half_size=[5.0, 0.1, 3.0], material=mat_wall)
        wall_right = builder.build_static(name="wall_right")
        wall_right.set_pose(sapien.Pose(p=[0, -2.0, 1.5]))

        print("[SCENE] Stanza costruita con successo!")
        
    except Exception as e:
        print(f"[WARNING] Errore costruzione scena: {e}")

# --- FIX: CAMERA FINDER ---
def setup_bridge_camera_view(env):
    """
    Cerca la camera nelle proprietà interne di ManiSkill se l'API pubblica fallisce.
    """
    print("[CAMERA] Ricerca camera...")
    
    # Posa BridgeData
    new_pose = sapien.Pose(p=[0.6, 0.0, 0.6], q=[0.9238, 0.0, 0.3826, 0.0])
    
    found_cam = None
    
    # Tentativo 1: API pubblica SAPIEN
    try:
        scene = env.unwrapped.scene
        # Prova a cercare nelle camere del renderer
        if hasattr(scene, "get_cameras"):
            cams = scene.get_cameras()
            if len(cams) > 0: found_cam = cams[0]
    except: pass

    # Tentativo 2: Interni di ManiSkill (Spesso le camere sono qui)
    if not found_cam:
        try:
            # Molti env MS3 salvano le camere in un dizionario
            if hasattr(env.unwrapped, "_render_cameras"):
                cameras = env.unwrapped._render_cameras
                if len(cameras) > 0:
                    # Prendi la prima (o quella chiamata 'base_camera' o 'render_camera')
                    found_cam = list(cameras.values())[0]
        except: pass

    if found_cam:
        try:
            found_cam.set_local_pose(new_pose)
            print(f"[CAMERA] Camera trovata e spostata: {found_cam.name}")
        except:
            # Se è una camera montata su entità
            if hasattr(found_cam, "set_pose"):
                found_cam.set_pose(new_pose)
                print(f"[CAMERA] Camera (Entity) spostata.")
    else:
        print("[WARNING] Nessuna camera trovata. Uso default.")

def get_render_rgb(env):
    img = env.unwrapped.render_rgb_array()
    img = to_numpy(img)
    img = img.squeeze() 
    if img.dtype != np.uint8:
        if img.max() <= 1.0: img = (img * 255).astype(np.uint8)
        else: img = img.astype(np.uint8)
    return img

def main():
    print(f"\n[INIT] Avvio script V16.0 (Fixed Scene Builder)...")
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
    
    # 1. Costruisci la stanza (ora col fix per i materiali)
    decorate_scene(env)
    
    # 2. Sposta la camera (con ricerca avanzata)
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