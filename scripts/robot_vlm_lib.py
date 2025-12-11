# robot_vlm_lib.py
import numpy as np
import robosuite
import robocasa
from robosuite.utils.transform_utils import quat2mat
from typing import Dict, Tuple, Any, Optional

# =============================================================================
# 1. DRIVER DI BASSO LIVELLO (Hardware Abstraction Layer)
# =============================================================================

class PandaSafeDriver:
    """
    Gestisce la comunicazione diretta con il simulatore/robot.
    """
    def __init__(self, env):
        self.env = env
        self.sim = env.sim
        self.action_dim = 12
        
        # Mappatura
        self.idx_arm = slice(0, 6)
        self.idx_gripper = slice(6, 7)
        self.idx_base = slice(7, 10)
        self.idx_torso = slice(10, 11)
        
        # Polso per FK
        try:
            self.wrist_body_id = self.env.sim.model.body_name2id("robot0_link7")
        except:
            self.wrist_body_id = None

        # Giunti braccio per Safety Check
        self.arm_joint_ids = []
        self.arm_joint_names = []
        for i in range(self.env.sim.model.njnt):
            name = self.env.sim.model.joint_id2name(i)
            if "robot0_joint" in name and "finger" not in name and "wheel" not in name:
                self.arm_joint_ids.append(i)
                self.arm_joint_names.append(name)

    def get_real_tip_pose(self) -> np.ndarray:
        if self.wrist_body_id is None: return np.array([0,0,0])
        self.env.sim.forward()
        wrist_pos = self.env.sim.data.body_xpos[self.wrist_body_id]
        wrist_mat = quat2mat(self.env.sim.data.body_xquat[self.wrist_body_id])
        return wrist_pos + wrist_mat.dot(np.array([0, 0, 0.13]))

    def get_safety_report(self) -> Tuple[bool, list]:
        """Controlla limiti giunti (<5% o >95%)."""
        danger_joints = []
        is_critical = False
        
        for j_id, name in zip(self.arm_joint_ids, self.arm_joint_names):
            addr = self.env.sim.model.jnt_qposadr[j_id]
            val = self.env.sim.data.qpos[addr]
            min_v, max_v = self.env.sim.model.jnt_range[j_id]
            rng = max_v - min_v
            
            if rng > 0:
                pct = (val - min_v) / rng * 100
                if pct < 5 or pct > 95: # Soglia un po' più conservativa per avvisare prima
                    danger_joints.append(f"{name[-2:]}({int(pct)}%)")
                    is_critical = True
        
        return is_critical, danger_joints

    def build_safe_action(self, base_vel, torso_vel, arm_delta, gripper_cmd) -> Tuple[np.ndarray, bool]:
        """
        Assembla l'azione. Se is_critical è True, stampa a video e rallenta.
        """
        is_critical, dangers = self.get_safety_report()
        
        # SAFETY INTERVENTION
        # Se stiamo chiedendo di muovere il braccio E siamo in zona critica
        if is_critical and np.linalg.norm(arm_delta) > 0.01:
            # Stampa log visibile all'utente (con \r per non spammare troppe righe)
            print(f"\r\033[93m[SAFETY CLAMP] Limiti vicini: {dangers}\033[0m   ", end="", flush=True)
            
            # Riduciamo drasticamente la velocità (Clamp)
            arm_delta = [x * 0.1 for x in arm_delta] 
            
        action = np.zeros(self.action_dim)
        action[self.idx_arm] = np.clip(arm_delta, -1, 1)
        action[self.idx_gripper] = gripper_cmd
        action[self.idx_base] = np.clip(base_vel, -1, 1)
        action[self.idx_torso] = np.clip(torso_vel, -1, 1)
        
        return action, is_critical

# =============================================================================
# 2. INTERFACCIA VLM
# =============================================================================

class VLMRobotInterface:
    def __init__(self, env_name="PnPCounterToCab", render=True):
        
        # Configurazione Controller
        controller_config = {
            "type": "OSC_POSE",
            "input_max": 1, "input_min": -1,
            "output_max": [1]*6, "output_min": [-1]*6,
            "kp": 150, "damping": 2,
            "impedance_mode": "fixed",
            "kp_limits": [0, 300], "damping_limits": [0, 10],
            "uncouple_pos_ori": True, "control_delta": True,
            "interpolation": None, "ramp_ratio": 0.2
        }

        self.camera_names = ["robot0_eye_in_hand", "robot0_agentview_center"]
        
        print(f"[VLMLib] Inizializzazione {env_name}...")
        try:
            self.env = robosuite.make(
                env_name=env_name,
                robots="PandaMobile",
                controller_configs=controller_config,
                has_renderer=render,
                has_offscreen_renderer=True,
                use_camera_obs=True,
                camera_names=self.camera_names,
                camera_heights=256,
                camera_widths=256,
                ignore_done=True
            )
        except Exception as e:
            raise RuntimeError(f"Errore caricamento: {e}")

        self.driver = PandaSafeDriver(self.env)
        self.env.reset()
        
        self.gripper_state = -1.0
        self.last_obs = None
        
        # --- PARAMETRI OTTIMIZZATI PER VELOCITÀ ---
        self.ARM_SENSITIVITY = 0.05   # Aumentata (era 0.02) -> Muove di più per ogni step
        self.BASE_SENSITIVITY = 1.0   
        self.STEPS_PER_ACTION = 15    # Diminuita (era 25) -> Ogni comando dura meno tempo (più fluido)

    def get_context(self) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
        tip_pos = self.driver.get_real_tip_pose()
        safety_crit, safety_msg = self.driver.get_safety_report()
        
        text_info = {
            "eef_xyz": [round(x, 3) for x in tip_pos],
            "gripper": "closed" if self.gripper_state > 0 else "open",
            "safety_warning": safety_msg if safety_crit else "nominal"
        }
        
        if self.last_obs is None:
            # Dummy step per inizializzare sensori
            null_action, _ = self.driver.build_safe_action([0,0,0], 0, [0]*6, self.gripper_state)
            self.last_obs, _, _, _ = self.env.step(null_action)

        visual_dict = {}
        for cam in self.camera_names:
            key = cam + "_image"
            if key in self.last_obs:
                visual_dict[cam] = np.flipud(self.last_obs[key])

        return text_info, visual_dict

    def execute_action(self, primitive: str, params: list) -> Dict[str, Any]:
        """
        Esegue primitiva. Ritorna info sullo stato finale.
        """
        base_cmd = [0, 0, 0]
        arm_cmd = [0, 0, 0, 0, 0, 0]
        torso_cmd = 0.0
        
        clamped_count = 0 # Contiamo quante volte la safety è intervenuta
        
        try:
            if primitive == 'base':
                base_cmd = [p * self.BASE_SENSITIVITY for p in params]
            elif primitive == 'arm':
                arm_cmd[0] = params[0] * self.ARM_SENSITIVITY
                arm_cmd[1] = params[1] * self.ARM_SENSITIVITY
                arm_cmd[2] = params[2] * self.ARM_SENSITIVITY
            elif primitive == 'torso':
                torso_cmd = float(params[0])
            elif primitive == 'gripper':
                val = params[0]
                if isinstance(val, str):
                    self.gripper_state = 1.0 if val == 'close' else -1.0
                else:
                    self.gripper_state = 1.0 if val > 0 else -1.0

            # Esecuzione Loop Fisico
            for _ in range(self.STEPS_PER_ACTION):
                action, is_crit = self.driver.build_safe_action(
                    base_cmd, torso_cmd, arm_cmd, self.gripper_state
                )
                if is_crit and primitive == 'arm': # Se critico durante movimento braccio
                    clamped_count += 1
                    
                self.last_obs, reward, done, info = self.env.step(action)
                self.env.render()
            
            # Recuperiamo contesto finale
            final_info, _ = self.get_context()
            
            # Aggiungiamo flag al ritorno per dire se l'azione è stata problematica
            final_info['movement_clamped'] = (clamped_count > self.STEPS_PER_ACTION // 2)
            return final_info

        except Exception as e:
            print(f"[VLMLib] Errore execute_action: {e}")
            return {}

    def close(self):
        if self.env:
            self.env.close()
            print("\n[VLMLib] Ambiente chiuso.")