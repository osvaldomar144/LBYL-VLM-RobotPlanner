import time
import numpy as np
import robosuite as suite
import robocasa  # Registra gli ambienti
import cv2
import os
import sys

# === CONFIGURAZIONE ===
LOG_DIR = "logs/calibration"
os.makedirs(LOG_DIR, exist_ok=True)

def get_osc_pose_config():
    return {
        "type": "OSC_POSE",
        "input_max": 1,
        "input_min": -1,
        "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
        "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
        "kp": 150,
        "damping_ratio": 1,
        "impedance_mode": "fixed",
        "uncouple_pos_ori": True,
        "control_delta": True,
        "interpolation": None,
        "ramp_ratio": 0.2
    }

# === DRIVER CON FIX NOMI ===
class PandaSafeDriver:
    def __init__(self, env):
        self.env = env
        self.sim = env.sim
        
        # 1. Trova ID End Effector (CORRETTO PER TUA CONFIG)
        # Cerchiamo il nome esatto apparso nel tuo log di errore
        target_site = "gripper0_right_grip_site"
        try:
            self.eef_site_id = self.env.sim.model.site_name2id(target_site)
            print(f"✅ EEF Site found: '{target_site}'")
        except:
            # Fallback disperato (prova nomi alternativi se cambia ancora)
            print(f"⚠️ Site '{target_site}' not found. Trying alternatives...")
            possible_names = ["gripper0_grip_site", "right_gripper_collision_site", "gripper0_right_ee_x"]
            found = False
            for name in possible_names:
                try:
                    self.eef_site_id = self.env.sim.model.site_name2id(name)
                    print(f"✅ Fallback success: using '{name}'")
                    found = True
                    break
                except:
                    continue
            if not found:
                raise ValueError("❌ IMPOSSIBILE TROVARE IL GRIPPER SITE. Controlla i nomi in Mujoco.")

            
        # 2. Trova ID Base Robot (Per calibrazione relativa stabile)
        try:
            self.base_body_id = self.env.sim.model.body_name2id("robot0_link0")
            print(f"✅ Robot Base Frame found: 'robot0_link0' (id {self.base_body_id})")
        except:
            print("⚠️ Base link 'robot0_link0' not found. Trying 'mobilebase0_center'...")
            try:
                # Fallback per base mobile
                self.base_body_id = self.env.sim.model.body_name2id("mobilebase0_center") # Nota: questo è un site solitamente, controlliamo body
            except: 
                 self.base_body_id = None
                 print("⚠️ Base Frame not found. Calibration will be in World Frame (Rotation Unstable).")

    def get_tip_pose_in_base_frame(self):
        """
        Restituisce XYZ del gripper RELATIVI alla base del robot.
        """
        # Posizione EEF in World
        eef_pos_world = np.array(self.env.sim.data.site_xpos[self.eef_site_id])
        
        if self.base_body_id is not None:
            # Posizione e Rotazione Base in World
            base_pos_world = np.array(self.env.sim.data.body_xpos[self.base_body_id])
            base_rot_mat = np.array(self.env.sim.data.body_xmat[self.base_body_id]).reshape(3, 3)
            
            # Calcola Delta nel mondo
            delta_world = eef_pos_world - base_pos_world
            
            # Proietta nel frame della base: v_local = R.T @ v_world
            pos_local = base_rot_mat.T @ delta_world
            return pos_local
        else:
            return eef_pos_world

# === HAL CLASS ===
class CalibrationHAL:
    def __init__(self, env_name="PnPCounterToSink", robot_name="PandaMobile", render=False):
        print(f"Initializing HAL for {env_name}...")
        
        self.cam_global = "robot0_agentview_center"
        self.cam_hand = "robot0_eye_in_hand"
        
        self.env = suite.make(
            env_name=env_name,
            robots=robot_name,
            controller_configs=get_osc_pose_config(),
            has_renderer=render,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names=[self.cam_global, self.cam_hand],
            camera_heights=224,
            camera_widths=224,
        )
        
        self.driver = PandaSafeDriver(self.env)
        self.obs = self.env.reset()
        self.check_torso()

    def check_torso(self):
        joint_names = [self.env.sim.model.joint_id2name(i) for i in range(self.env.sim.model.njnt)]
        if any("torso" in str(name) for name in joint_names):
            print("\n⚠️  WARNING: TORSO JOINT RILEVATO! (Sim Only)")
            print("   L'HAL forzerà questo asse a 0 per simulare il Franka Reale.\n")
        else:
            print("\n✅ OK: Nessun torso joint rilevato.")

    def get_obs_images(self):
        return {
            "global": np.flipud(self.obs[self.cam_global + "_image"]),
            "hand": np.flipud(self.obs[self.cam_hand + "_image"])
        }

    def step_manual(self, arm_action_7d):
        full_action = np.zeros(self.env.action_dim)
        full_action[:6] = arm_action_7d[:6] # Arm
        full_action[6] = 1.0 if arm_action_7d[6] > 0.5 else -1.0 # Gripper
        
        # Mode switch / Torso lock
        if self.env.action_dim >= 12:
            full_action[11] = -1.0 
            
        self.obs, _, _, _ = self.env.step(full_action)
        return self.obs

# === CALIBRATION LOOP ===
def run_calibration_axis(hal, axis_idx, axis_val, steps=20, label="test"):
    print(f"\n--- RUN: {label} (Ax {axis_idx} Val {axis_val}) ---")
    
    hal.env.reset()
    # Warmup
    for _ in range(15): hal.step_manual(np.zeros(7))
        
    start_pos = hal.driver.get_tip_pose_in_base_frame()
    
    for i in range(steps):
        action = np.zeros(7)
        action[axis_idx] = axis_val
        hal.step_manual(action)
        
        if i % 10 == 0:
            cv2.imshow("Calib", cv2.cvtColor(hal.get_obs_images()["global"], cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

    end_pos = hal.driver.get_tip_pose_in_base_frame()
    delta = end_pos - start_pos
    
    print(f"-> Delta (Base Frame): {np.round(delta, 4)}")
    return delta

if __name__ == "__main__":
    try:
        hal = CalibrationHAL(render=False)
        
        # Test X, Y, Z
        dx = run_calibration_axis(hal, 0, 1.0, label="X+ (Forward)")
        dy = run_calibration_axis(hal, 1, 1.0, label="Y+ (Left/Right)")
        dz = run_calibration_axis(hal, 2, 1.0, label="Z+ (Up)")
        
        print("\n" + "="*40)
        print("=== MATRICE DI CALIBRAZIONE (ROBOT FRAME) ===")
        print(f"Input X+ (1.0) -> Delta: {np.round(dx, 4)}")
        print(f"Input Y+ (1.0) -> Delta: {np.round(dy, 4)}")
        print(f"Input Z+ (1.0) -> Delta: {np.round(dz, 4)}")
        print("="*40 + "\n")
        
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"\n❌ ERRORE:\n{e}")
        import traceback
        traceback.print_exc()