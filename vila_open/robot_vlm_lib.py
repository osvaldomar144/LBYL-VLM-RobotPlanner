# vila_open/robot_vlm_lib.py
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
    Si occupa di: mappatura indici, cinematica diretta (FK) e controlli di sicurezza.
    """
    def __init__(self, env):
        self.env = env
        self.sim = env.sim
        self.action_dim = 12
        
        # --- Mappatura Indici Azione ---
        self.idx_arm = slice(0, 6)      # x, y, z, ax, ay, az
        self.idx_gripper = slice(6, 7)  # 1 (chiuso) / -1 (aperto)
        self.idx_base = slice(7, 10)    # vx, vy, w
        self.idx_torso = slice(10, 11)  # lift
        
        # --- Identificazione Componenti ---
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
            # Filtra solo i giunti del braccio (esclude dita e ruote)
            if "robot0_joint" in name and "finger" not in name and "wheel" not in name:
                self.arm_joint_ids.append(i)
                self.arm_joint_names.append(name)

    def get_real_tip_pose(self) -> np.ndarray:
        """
        Calcola la posizione XYZ reale dell'end-effector (punta delle dita).
        Include l'offset di 13cm dal link7.
        """
        if self.wrist_body_id is None: return np.array([0,0,0])
        self.env.sim.forward()
        wrist_pos = self.env.sim.data.body_xpos[self.wrist_body_id]
        wrist_mat = quat2mat(self.env.sim.data.body_xquat[self.wrist_body_id])
        # Offset locale [0, 0, 0.13] trasformato in coordinate globali
        return wrist_pos + wrist_mat.dot(np.array([0, 0, 0.13]))

    def get_safety_report(self) -> Tuple[bool, list]:
        """
        Verifica se i giunti sono vicini ai limiti fisici (<3% o >97% del range).
        Return: (is_critical, list_of_danger_joints)
        """
        danger_joints = []
        is_critical = False
        
        for j_id, name in zip(self.arm_joint_ids, self.arm_joint_names):
            addr = self.env.sim.model.jnt_qposadr[j_id]
            val = self.env.sim.data.qpos[addr]
            min_v, max_v = self.env.sim.model.jnt_range[j_id]
            rng = max_v - min_v
            
            if rng > 0:
                pct = (val - min_v) / rng * 100
                if pct < 3 or pct > 97:
                    danger_joints.append(f"{name[-2:]}({int(pct)}%)")
                    is_critical = True
                elif pct < 10 or pct > 90: # Warning non critico
                    danger_joints.append(f"{name[-2:]}({int(pct)}%)")
        
        return is_critical, danger_joints

    def build_safe_action(self, base_vel, torso_vel, arm_delta, gripper_cmd) -> Tuple[np.ndarray, bool]:
        """
        Assembla il vettore azione da 12 float applicando filtri di sicurezza.
        Se i giunti sono al limite, riduce drasticamente la velocità del braccio.
        """
        is_critical, _ = self.get_safety_report()
        
        # SAFETY INTERVENTION: Rallenta se critico
        if is_critical and np.linalg.norm(arm_delta) > 0.01:
            arm_delta = [x * 0.1 for x in arm_delta] 
            
        action = np.zeros(self.action_dim)
        action[self.idx_arm] = np.clip(arm_delta, -1, 1)
        action[self.idx_gripper] = gripper_cmd
        action[self.idx_base] = np.clip(base_vel, -1, 1)
        action[self.idx_torso] = np.clip(torso_vel, -1, 1)
        
        return action, is_critical


# =============================================================================
# 2. INTERFACCIA VLM (High Level API)
# =============================================================================

class VLMRobotInterface:
    """
    Interfaccia pensata per essere chiamata da un modello AI (VLM).
    Mette a disposizione metodi 'atomici' e gestisce il loop di simulazione.
    """
    def __init__(self, env_name="PnPCounterToCab", render=True):
        
        # --- CONFIGURAZIONE CONTROLLER "PRECISION" ---
        # Rigido (KP alto) e Reattivo (Damping basso) per seguire fedelmente la VLM
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

        # --- SETUP AMBIENTE ---
        # NOTA: Corretti i nomi delle camere per PandaMobile
        self.camera_names = ["robot0_eye_in_hand", "robot0_agentview_center"]
        
        print(f"[VLMLib] Inizializzazione ambiente: {env_name}...")
        try:
            self.env = robosuite.make(
                env_name=env_name,
                robots="PandaMobile",
                controller_configs=controller_config,
                has_renderer=render,
                has_offscreen_renderer=True, # Necessario per VLM (cattura immagini)
                use_camera_obs=True,
                camera_names=self.camera_names,
                camera_heights=256,
                camera_widths=256,
                ignore_done=True
            )
        except Exception as e:
            raise RuntimeError(f"Errore caricamento Robosuite/Robocasa: {e}")

        # Inizializzazione Driver
        self.driver = PandaSafeDriver(self.env)
        self.env.reset()
        
        # Stato Persistente
        self.gripper_state = -1.0  # Inizia aperto
        self.last_obs = None       # Cache dell'ultima osservazione
        
        # --- PARAMETRI DI SCALA E TEMPO ---
        # Definiscono quanto 'fisicamente' si muove il robot per ogni comando VLM
        self.ARM_SENSITIVITY = 0.02   # Molto fine per precisione
        self.BASE_SENSITIVITY = 1.0   # Standard
        self.STEPS_PER_ACTION = 25    # Durata (tick) di ogni azione atomica

    def get_context(self) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
        """
        Raccoglie e restituisce il contesto attuale per la VLM.
        
        Returns:
            text_info (dict): Info testuali (posizione XYZ, stato gripper, warning safety).
            visual_dict (dict): Dizionario immagini {'camera_name': np.array(H,W,3)}.
        """
        # 1. Dati Fisici
        tip_pos = self.driver.get_real_tip_pose()
        safety_crit, safety_msg = self.driver.get_safety_report()
        
        text_info = {
            "eef_xyz": [round(x, 3) for x in tip_pos],
            "gripper": "closed" if self.gripper_state > 0 else "open",
            "safety_warning": safety_msg if safety_crit else "nominal"
        }
        
        # 2. Dati Visivi
        # Se non abbiamo un'osservazione precedente (avvio), facciamo uno step nullo
        if self.last_obs is None:
            null_action, _ = self.driver.build_safe_action([0,0,0], 0, [0]*6, self.gripper_state)
            self.last_obs, _, _, _ = self.env.step(null_action)

        visual_dict = {}
        for cam in self.camera_names:
            key = cam + "_image"
            if key in self.last_obs:
                # Robosuite renderizza upside-down, flippiamo per la VLM
                visual_dict[cam] = np.flipud(self.last_obs[key])

        return text_info, visual_dict

    def execute_action(self, primitive: str, params: list) -> None:
        """
        Esegue una primitiva inviata dalla VLM.
        Gestisce internamente il loop di simulazione (STEPS_PER_ACTION).
        
        Args:
            primitive (str): 'arm', 'base', 'torso', 'gripper'.
            params (list): Parametri specifici (es. [dx, dy, dz] o [open/close]).
        """
        base_cmd = [0, 0, 0]
        arm_cmd = [0, 0, 0, 0, 0, 0] # x,y,z, r,p,y
        torso_cmd = 0.0
        
        try:
            # --- Parsing Comando ---
            if primitive == 'base':
                # params: [vx, vy, w]
                # Esempio VLM: "base 1 0 0" (avanti)
                base_cmd = [p * self.BASE_SENSITIVITY for p in params]

            elif primitive == 'arm':
                # params: [dx, dy, dz]
                # Esempio VLM: "arm 0 0 -1" (giù)
                arm_cmd[0] = params[0] * self.ARM_SENSITIVITY
                arm_cmd[1] = params[1] * self.ARM_SENSITIVITY
                arm_cmd[2] = params[2] * self.ARM_SENSITIVITY
                # (Rotazione lasciata a 0 per ora, mantiene orientamento corrente)

            elif primitive == 'torso':
                # params: [velocity]
                torso_cmd = float(params[0])

            elif primitive == 'gripper':
                # params: [1.0] (close) o [-1.0] (open)
                # Accetta anche stringhe 'open'/'close' se parsate prima, ma qui assumiamo float/int
                val = params[0]
                if isinstance(val, str):
                    self.gripper_state = 1.0 if val == 'close' else -1.0
                else:
                    self.gripper_state = 1.0 if val > 0 else -1.0

            else:
                print(f"[VLMLib] Warning: Primitiva sconosciuta '{primitive}'")
                return

            # --- Esecuzione Loop Fisico ---
            for _ in range(self.STEPS_PER_ACTION):
                # 1. Costruisci azione sicura
                action, is_crit = self.driver.build_safe_action(
                    base_cmd, torso_cmd, arm_cmd, self.gripper_state
                )
                
                # 2. Step Simulatore
                self.last_obs, reward, done, info = self.env.step(action)
                
                # 3. Render
                self.env.render()

        except Exception as e:
            print(f"[VLMLib] Errore critico in execute_action: {e}")

    def close(self):
        """Chiude l'ambiente e rilascia risorse."""
        if self.env:
            self.env.close()
            print("[VLMLib] Ambiente chiuso.")