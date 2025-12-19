import time
import argparse
import numpy as np
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import cv2
import robosuite as suite
import robocasa
import warnings
import os

# Ignora warning HF
warnings.filterwarnings("ignore")

# === 1. CONFIGURAZIONE PARAMETRI ===
DEFAULT_POS_SCALE = 80.0
DEFAULT_ROT_SCALE = 20.0
DEFAULT_EMA = 0.20

# === 2. HAL & DRIVER ===
def get_osc_pose_config():
    return {
        "type": "OSC_POSE",
        "input_max": 1, "input_min": -1,
        "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
        "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
        "kp": 150, "damping_ratio": 1, "impedance_mode": "fixed",
        "uncouple_pos_ori": True, "control_delta": True,
        "interpolation": None, "ramp_ratio": 0.2
    }

class ActionSmoother:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.prev = None
    def update(self, val):
        if self.prev is None: self.prev = val
        self.prev = self.alpha * val + (1 - self.alpha) * self.prev
        return self.prev
    def reset(self):
        self.prev = None

class RobotHAL:
    def __init__(self, env_name="PnPCounterToSink", robot_name="PandaMobile", ema_alpha=0.2):
        print(f"[HAL] Initializing {env_name}...")
        self.cam_g = "robot0_agentview_center"
        self.cam_h = "robot0_eye_in_hand"
        
        self.env = suite.make(
            env_name=env_name, robots=robot_name,
            controller_configs=get_osc_pose_config(),
            has_renderer=False, has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names=[self.cam_g, self.cam_h],
            camera_heights=224, camera_widths=224,
        )
        
        try:
            self.eef_id = self.env.sim.model.site_name2id("gripper0_right_grip_site")
        except:
            self.eef_id = self.env.sim.model.site_name2id("right_gripper_collision_site")
            
        try:
            self.base_id = self.env.sim.model.body_name2id("robot0_link0")
        except:
            self.base_id = None

        self.obs = self.env.reset()
        self.smoother = ActionSmoother(ema_alpha)

    def get_images(self):
        # LOGICA SCRIPT 09: flipud -> Image.fromarray
        # Questo funzionava, quindi non tocchiamolo.
        g_arr = np.flipud(self.obs[self.cam_g + "_image"])
        h_arr = np.flipud(self.obs[self.cam_h + "_image"])
        
        g_pil = Image.fromarray(g_arr)
        h_pil = Image.fromarray(h_arr)
        
        return g_pil, h_pil
    
    def get_eef_pos(self):
        eef_world = np.array(self.env.sim.data.site_xpos[self.eef_id])
        if self.base_id is not None:
            base_pos = np.array(self.env.sim.data.body_xpos[self.base_id])
            base_rot = np.array(self.env.sim.data.body_xmat[self.base_id]).reshape(3,3)
            return base_rot.T @ (eef_world - base_pos)
        return eef_world

    def step(self, raw_action, pos_scale, rot_scale):
        act = raw_action.copy()
        act[:3] *= pos_scale
        act[3:6] *= rot_scale
        act[:6] = self.smoother.update(act[:6])
        act[:6] = np.clip(act[:6], -1.0, 1.0)
        
        full = np.zeros(self.env.action_dim)
        full[:6] = act[:6]
        full[6] = 1.0 if raw_action[6] > 0.5 else -1.0 
        if self.env.action_dim >= 12: full[11] = -1.0 
        
        self.obs, _, _, _ = self.env.step(full)
        return self.obs, act

# === 3. VLA CONTROLLER (NATIVE BFLOAT16) ===
class VLAController:
    def __init__(self, model_id="openvla/openvla-7b"):
        print(f"[VLA] Loading {model_id} in bfloat16...")
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager", 
            trust_remote_code=True,
            low_cpu_mem_usage=True
        ).to("cuda:0")
        
    def predict(self, instruction, img_g, img_h):
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"
        inputs = self.processor(prompt, img_g).to("cuda:0", dtype=torch.bfloat16)
        
        if not torch.all(inputs["input_ids"][:, -1] == 29871):
            bsz = inputs["input_ids"].shape[0]
            ex = torch.full((bsz, 1), 29871, device="cuda:0", dtype=inputs["input_ids"].dtype)
            inputs["input_ids"] = torch.cat([inputs["input_ids"], ex], dim=1)
            inputs["attention_mask"] = torch.cat([inputs["attention_mask"], torch.ones((bsz,1), device="cuda:0")], dim=1)
            
        action = self.model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
        return action

# === 4. MAIN ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instr", type=str, default="pick the red pepper", help="Comando simulato dal VLM")
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--scale", type=float, default=DEFAULT_POS_SCALE)
    parser.add_argument("--ema", type=float, default=DEFAULT_EMA)
    args = parser.parse_args()

    hal = RobotHAL(ema_alpha=args.ema)
    
    try:
        vla = VLAController()
    except Exception as e:
        print(f"\n❌ Errore caricamento modello: {e}")
        return
    
    print(f"\n--- SIMULAZIONE COMANDO VLM: '{args.instr}' ---")
    print(f"Parametri: Scale={args.scale}, EMA={args.ema}")
    print("Premi 'q' nel video per uscire.\n")
    
    start_pos = hal.get_eef_pos()
    
    for i in range(args.steps):
        t0 = time.time()
        
        # 1. Ottieni PIL images (usando la logica script 09)
        img_g_pil, img_h_pil = hal.get_images()
        
        # 2. Think
        raw_action = vla.predict(args.instr, img_g_pil, img_h_pil)
        
        # 3. Act
        _, smooth_act = hal.step(raw_action, args.scale, DEFAULT_ROT_SCALE)
        
        # Stats
        dt = time.time() - t0
        curr_pos = hal.get_eef_pos()
        delta = curr_pos - start_pos
        grip_state = "CLOSED" if raw_action[6] > 0.5 else "OPEN"
        
        print(f"\rStep {i:03d} | FPS {1/dt:.1f} | Grip: {grip_state} | ActionXYZ: {smooth_act[:3].round(2)} | PosDelta: {delta.round(2)}", end="")
        
        # 4. Visualization (Convertendo da PIL come nello script 09)
        # Questo passaggio è cruciale: convertiamo l'oggetto PIL (che è pulito) in numpy per OpenCV
        frame_cv = np.array(img_g_pil) 
        # OpenCV usa BGR, PIL usa RGB -> conversione
        frame_cv = cv2.cvtColor(frame_cv, cv2.COLOR_RGB2BGR)
        
        cv2.putText(frame_cv, f"CMD: {args.instr}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("VLA Simulation", frame_cv)
        if cv2.waitKey(1) == ord('q'): break

    print("\n\nTest completato.")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()