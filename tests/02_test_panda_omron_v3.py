import numpy as np
import robosuite
import robocasa
from robosuite.utils.transform_utils import quat2mat

class PandaOmronAgent:
    def __init__(self, env):
        self.env = env
        self.sim = env.sim
        self.action_dim = 12
        
        # Mappatura
        self.idx_arm = slice(0, 6)
        self.idx_gripper = slice(6, 7)
        self.idx_base = slice(7, 10)
        self.idx_torso = slice(10, 11)
        
        # Tracking Polso
        try:
            self.wrist_body_id = self.env.sim.model.body_name2id("robot0_link7")
        except:
            self.wrist_body_id = None

        # Identificazione Giunti Braccio
        self.arm_joint_ids = []
        self.arm_joint_names = []
        for i in range(self.env.sim.model.njnt):
            name = self.env.sim.model.joint_id2name(i)
            if "robot0_joint" in name and "finger" not in name and "wheel" not in name:
                self.arm_joint_ids.append(i)
                self.arm_joint_names.append(name)

    def get_real_tip_pose(self):
        """Cinematica Diretta Polso + Offset."""
        if self.wrist_body_id is None: return np.array([0,0,0])
        self.env.sim.forward()
        wrist_pos = self.env.sim.data.body_xpos[self.wrist_body_id]
        wrist_mat = quat2mat(self.env.sim.data.body_xquat[self.wrist_body_id])
        return wrist_pos + wrist_mat.dot(np.array([0, 0, 0.13]))

    def get_safety_report(self):
        """Restituisce un dizionario con lo stato di pericolo dei giunti."""
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
                elif pct < 10 or pct > 90:
                    danger_joints.append(f"{name[-2:]}({int(pct)}%)")
        
        return is_critical, danger_joints

    def build_safe_action(self, base_vel, torso_vel, arm_delta, gripper_cmd):
        """Costruisce l'azione con filtro di sicurezza attivo."""
        is_critical, dangers = self.get_safety_report()
        
        # Se siamo al limite, riduciamo drasticamente la spinta del braccio
        if is_critical:
            # Se l'utente sta cercando di muoversi, rallentiamo
            if np.linalg.norm(arm_delta) > 0.01:
                print(f"\r\033[91m[SAFETY STOP] Limiti raggiunti: {dangers}\033[0m", end="")
                arm_delta = [x * 0.1 for x in arm_delta] 
            
        action = np.zeros(self.action_dim)
        action[self.idx_arm] = np.clip(arm_delta, -1, 1)
        action[self.idx_gripper] = gripper_cmd
        action[self.idx_base] = np.clip(base_vel, -1, 1)
        action[self.idx_torso] = np.clip(torso_vel, -1, 1)
        
        return action, is_critical

def main():
    env_name = "PnPCounterToCab"
    
    # --- CONFIGURAZIONE "PRECISION" ---
    # KP alto = robot rigido (segue fedelmente il target, meno lag).
    # Damping basso = ferma il movimento residuo più in fretta.
    controller_config = {
        "type": "OSC_POSE",
        "input_max": 1, 
        "input_min": -1,
        "output_max": [1]*6, 
        "output_min": [-1]*6,
        "kp": 150,       # RIGIDO: Il robot sta incollato al target
        "damping": 2,    # REATTIVO: Meno effetto "trascinamento"
        "impedance_mode": "fixed",
        "kp_limits": [0, 300], 
        "damping_limits": [0, 10],
        "uncouple_pos_ori": True, 
        "control_delta": True,
        "interpolation": None, 
        "ramp_ratio": 0.2
    }

    print(f"Caricamento {env_name}...")
    try:
        env = robosuite.make(
            env_name=env_name,
            robots="PandaMobile",
            controller_configs=controller_config,
            has_renderer=True,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            render_camera=None,
            ignore_done=True
        )
    except Exception as e:
        print(f"Errore: {e}")
        return

    agent = PandaOmronAgent(env)
    env.reset()
    
    current_state = {"gripper": -1.0}
    
    # --- SENSIBILITÀ CHIRURGICA ---
    # Poiché KP è alto, dobbiamo muovere il target PIANISSIMO
    # altrimenti il robot scatta.
    ARM_SENS = 0.02  # Molto lento e preciso
    BASE_SENS = 1.0  # La base resta uguale
    
    print("\n--- CONTROLLER DI PRECISIONE (NO LAG) ---")
    print("Comandi: 'base x y w', 'arm x y z', 'torso v', 'info', 'q'")
    
    while True:
        env.render()
        try:
            pos = agent.get_real_tip_pose()
            is_crit, _ = agent.get_safety_report()
            status_char = "!" if is_crit else ">"
            # Mostra coordinate
            user_input = input(f"\nEEF:[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}] {status_char} ").strip().lower()
        except EOFError: break
        
        if user_input == 'q': break
        
        parts = user_input.split()
        if not parts: continue
        cmd = parts[0]
        
        base = [0,0,0]
        arm = [0,0,0,0,0,0]
        torso = 0.0
        steps = 0 

        try:
            if cmd == 'info':
                print("--- Analisi Giunti ---")
                crit, report = agent.get_safety_report()
                if crit: print(f"CRITICITÀ: {report}")
                else: print("Tutti i giunti in range nominale.")
                
            elif cmd == 'base': 
                if len(parts) >= 4:
                    base = [float(parts[1])*BASE_SENS, float(parts[2])*BASE_SENS, float(parts[3])*BASE_SENS]
                    steps = 40
                else: print("Usa: base vx vy w")
                    
            elif cmd == 'torso': 
                if len(parts) >= 2:
                    torso = float(parts[1])
                    steps = 30
                else: print("Usa: torso vel")

            elif cmd == 'arm': 
                if len(parts) >= 4:
                    # Sensibilità bassissima per precisione
                    arm[0] = float(parts[1]) * ARM_SENS
                    arm[1] = float(parts[2]) * ARM_SENS
                    arm[2] = float(parts[3]) * ARM_SENS
                    steps = 20
                else: print("Usa: arm x y z")
            
            elif cmd == 'open':
                current_state["gripper"] = -1.0
                steps = 30 
                
            elif cmd == 'close':
                current_state["gripper"] = 1.0
                steps = 30
            
            if steps > 0:
                for i in range(steps):
                    action, danger = agent.build_safe_action(base, torso, arm, current_state["gripper"])
                    env.step(action)
                    env.render()
                    
        except ValueError: print("Errore numerico.")
        except Exception as e: print(f"Errore: {e}")

    env.close()

if __name__ == "__main__":
    main()