import gymnasium as gym
import mani_skill2.envs
import numpy as np
import cv2
import os

def test_simulation():
    print(">>> Inizializzazione Ambiente ManiSkill2 con Franka Panda...")
    
    # SETUP DELL'AMBIENTE
    # env_id: Il task. PickCube-v1 è lo standard "Hello World"
    # obs_mode="rgbd": Fondamentale! I VLA (e i VLM) hanno bisogno di immagini RGB.
    # control_mode="pd_ee_delta_pose": Questo è CRUCIALE per il Sim-to-Real. 
    #   Stiamo controllando la differenza (delta) di posizione dell'End Effector (la pinza).
    #   È molto più facile trasferire questo comando al robot reale rispetto ai giunti (joints).
    
    env = gym.make(
        "PickCube-v0", 
        obs_mode="rgbd", 
        control_mode="pd_ee_delta_pose",
        render_mode="rgb_array" # Render off-screen per velocità e stabilità server
    )
    
    print(f">>> Ambiente creato. Robot: Franka Panda. Action Space: {env.action_space}")
    
    obs, _ = env.reset()
    
    # Cartella per salvare i test
    os.makedirs("test_output", exist_ok=True)
    
    print(">>> Avvio loop di test (20 step)...")
    
    for i in range(20):
        # 1. Azione casuale
        # L'azione è solitamente [dx, dy, dz, drx, dry, drz, gripper]
        action = env.action_space.sample()
        
        # Facciamo muovere il robot un po' più piano/delicato (opzionale per debug)
        action = action * 0.5 
        
        # 2. Step fisico
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 3. Estrazione Immagini per VLM/VLA
        # ManiSkill restituisce le camere in obs['image']
        # Solitamente c'è 'hand_camera' (sulla pinza) e 'base_camera' (esterna)
        
        # Prendiamo la camera base (quella che userà il VLA per vedere la scena globale)
        cam_data = obs['image']['base_camera']['rgb'] # Shape: (H, W, 3)
        
        # Convertiamo da RGB a BGR per OpenCV
        img_bgr = cv2.cvtColor(cam_data, cv2.COLOR_RGB2BGR)
        
        # Salviamo il frame
        cv2.imwrite(f"test_output/step_{i:03d}.png", img_bgr)
        
    env.close()
    print(">>> Test completato! Controlla la cartella 'test_output' per vedere se il robot si muove.")

if __name__ == "__main__":
    test_simulation()