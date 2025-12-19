import time
import argparse
import numpy as np
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image, ImageDraw, ImageFont
import robosuite as suite
import robocasa
import warnings
import os
import shutil

warnings.filterwarnings("ignore")

# === 1. PARAMETRI DI TUNING ===
# Aumentati per compensare i valori raw molto piccoli
DEFAULT_POS_SCALE = 60.0   
DEFAULT_ROT_SCALE = 5.0
DEFAULT_EMA = 0.7  

# === 2. MAPPING PER CAMERA FISSA (AgentView) ===
def remap_fixed_cam_action(vla_xyz):
    """
    Mappa l'output del VLA (Image Frame) al Robot Base Frame
    quando si usa una camera FISSA (AgentView).
    """
    # MAPPING EMPIRICO PER AGENTVIEW:
    # VLA Output: [0]=Depth/Forward, [1]=Right, [2]=Down
    
    # 1. Image Down (VLA[2]) -> Robot Forward (X+)
    #    (Se l'oggetto è in basso nell'immagine, è più vicino al robot)
    robot_x = -vla_xyz[2]  
    
    # 2. Image Right (VLA[1]) -> Robot Right (Y-)
    #    (In Robosuite Y+ è sinistra, Y- è destra)
    robot_y = vla_xyz[1]   
    
    # 3. Image Depth (VLA[0]) -> Robot Down (Z-)
    #    (Il VLA spesso usa la depth per "avvicinarsi/scendere")
    robot_z = vla_xyz[0]   
    
    return np.array([robot_x, robot_y, robot_z])

# === 3. CONFIGURAZIONE CONTROLLER MANUALE (FIX ERRORE) ===
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

# === 4. UTILITY ===
class StickyGripper:
    def __init__(self):
        self.state = -1.0 
        self.counter = 0
    def update(self, raw_val, step_idx):
        if step_idx < 15: return -1.0 # Warmup
        if raw_val > 0.5: self.counter += 1
        elif raw_val < 0.5: self.counter -= 1
        self.counter = max(-3, min(3, self.counter))
        if self.counter >= 2: self.state = 1.0
        elif self.counter <= -2: self.state = -1.0
        return self.state

class ActionSmoother:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.prev = None
    def update(self, val):
        if self.prev is None: self.prev = val
        self.prev = self.alpha * val + (1 - self.alpha) * self.prev
        return self.prev

def draw_debug(img, vla_act, robot_act, grip):
    draw = ImageDraw.Draw(img)
    w, h = img.size
    cx, cy = w//2, h//2
    
    # Vettore Giallo (Intenzione VLA sul piano immagine)
    dx = vla_act[1] * 2000
    dy = vla_act[2] * 2000 
    draw.line([(cx, cy), (cx+dx, cy+dy)], fill="yellow", width=3)
    draw.ellipse([(cx+dx-4, cy+dy-4), (cx+dx+4, cy+dy+4)], fill="yellow")
    
    # Testo Debug
    text = f"ROBOT: [{robot_act[0]:.2f}, {robot_act[1]:.2f}, {robot_act[2]:.2f}]"
    draw.text((10, 10), text, fill="lime")
    
    g_txt = "CLOSE" if grip > 0 else "OPEN"
    draw.text((10, 25), f"GRIP: {g_txt}", fill="red" if grip > 0 else "lime")
    
    return img

class RobotHAL:
    def __init__(self, env_name="PnPCounterToSink", robot_name="PandaMobile"):
        print(f"[HAL] Initializing {env_name}...")
        self.cams = ["robot0_agentview_center", "robot0_eye_in_hand"]
        
        # FIX: Uso configurazione manuale invece di load_controller_config
        self.env = suite.make(
            env_name=env_name, 
            robots=robot_name,
            controller_configs=get_osc_pose_config(), # <--- CORRETTO
            has_renderer=True, 
            has_offscreen_renderer=True, 
            use_camera_obs=True,
            camera_names=self.cams, 
            camera_heights=224, camera_widths=224
        )
        self.obs = self.env.reset()
        self.smoother = ActionSmoother(DEFAULT_EMA)
        self.gripper = StickyGripper()

    def get_images(self):
        return {
            "agent": Image.fromarray(np.flipud(self.obs["robot0_agentview_center_image"])),
            "hand": Image.fromarray(np.flipud(self.obs["robot0_eye_in_hand_image"]))
        }

    def step(self, vla_action, scale, idx):
        # 1. Remap per Camera Fissa
        robot_action_xyz = remap_fixed_cam_action(vla_action[:3])
        
        # 2. Scale & Smooth
        robot_action_xyz = self.smoother.update(robot_action_xyz * scale)
        
        # 3. Grip
        grip = self.gripper.update(vla_action[6], idx)
        
        # 4. Step
        full_action = np.zeros(self.env.action_dim)
        full_action[:3] = robot_action_xyz
        full_action[6] = grip
        if self.env.action_dim >= 12: full_action[7:] = 0 
        
        self.obs, _, _, _ = self.env.step(full_action)
        self.env.render()
        
        return self.obs, robot_action_xyz, grip

class VLA:
    def __init__(self, model_id="openvla/openvla-7b"):
        print(f"[VLA] Loading {model_id}...")
        self.proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, 
            trust_remote_code=True, low_cpu_mem_usage=True
        ).to("cuda:0")
    
    def act(self, text, image):
        inputs = self.proc(f"In: What action should the robot take to {text}?\nOut:", image).to("cuda:0", dtype=torch.bfloat16)
        
        # Padding fix
        if not torch.all(inputs["input_ids"][:, -1] == 29871):
            bsz = inputs["input_ids"].shape[0]
            ex = torch.full((bsz, 1), 29871, device="cuda:0", dtype=inputs["input_ids"].dtype)
            inputs["input_ids"] = torch.cat([inputs["input_ids"], ex], dim=1)
            inputs["attention_mask"] = torch.cat([inputs["attention_mask"], torch.ones((bsz,1), device="cuda:0")], dim=1)

        return self.model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

def main():
    try:
        hal = RobotHAL()
        vla = VLA()
    except Exception as e:
        print(f"❌ Errore Inizializzazione: {e}")
        return
    
    # Debug Dir
    debug_dir = "debug_runs_fixed"
    if os.path.exists(debug_dir): shutil.rmtree(debug_dir)
    os.makedirs(debug_dir)
    
    print("\n" + "="*60)
    print("🎥 FIXED CAMERA MODE (AgentView)")
    print(f"Log salvati in: {debug_dir}")
    print("="*60)

    while True:
        # Reset
        print("\n🔄 Resetting...")
        hal.obs = hal.env.reset()
        hal.smoother.reset()
        hal.gripper.counter = 0
        
        # Warmup (Alza il braccio per non occludere subito)
        for _ in range(20): 
            act = np.zeros(hal.env.action_dim); act[2]=0.2; act[6]=-1
            hal.env.step(act)
            
        # Preview
        imgs = hal.get_images()
        imgs['agent'].save(f"{debug_dir}/PREVIEW.jpg")
        print(f"📸 Preview salvata. Controlla {debug_dir}/PREVIEW.jpg")
        
        task = input("Task (es: 'put the apple in the sink'): ")
        if task.lower() in ['q', 'quit']: break
        
        run_dir = os.path.join(debug_dir, f"{int(time.time())}_{task.replace(' ', '_')}")
        os.makedirs(run_dir)
        
        print(f"🚀 Executing: '{task}' using AgentView...")
        
        for i in range(60):
            imgs = hal.get_images()
            
            # USARE SEMPRE AGENT VIEW (Più stabile)
            vla_input = imgs['agent']
            
            # VLA Inference
            raw = vla.act(task, vla_input)
            
            # Step
            _, rob_act, grip = hal.step(raw, DEFAULT_POS_SCALE, i)
            
            # Save Debug Frame
            debug = draw_debug(vla_input.copy(), raw, rob_act, grip)
            debug.save(f"{run_dir}/step_{i:02d}.jpg")
            
            grip_icon = "✊" if grip > 0 else "✋"
            print(f"Step {i:02d} | {grip_icon} | RobotAct: {rob_act.round(3)}")
            
    hal.env.close()

if __name__ == "__main__":
    main()