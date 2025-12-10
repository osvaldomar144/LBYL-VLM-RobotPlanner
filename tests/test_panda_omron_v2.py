import numpy as np
import robosuite
import robocasa
import json

class SceneIntrospector:
    """Helper invariato."""
    def __init__(self, env):
        self.env = env
        self.sim = env.sim

    def inspect(self):
        data = {"cameras": [], "objects": []}
        for i in range(self.sim.model.ncam):
            data["cameras"].append(self.sim.model.camera_id2name(i))
        
        for i in range(self.sim.model.nbody):
            name = self.sim.model.body_id2name(i)
            if name and any(k in name for k in ['obj', 'door', 'handle', 'knob', 'cab', 'counter']):
                if "robot0" not in name and "gripper" not in name:
                    pos = self.sim.data.body_xpos[i].tolist()
                    data["objects"].append({"name": name, "pos": [round(x, 3) for x in pos]})
        return data

    def print_report(self, data):
        print("\n" + "="*40)
        print(" REPORT SCENA ")
        print("="*40)
        sorted_objs = sorted(data["objects"], key=lambda x: x['name'])
        for obj in sorted_objs:
            print(f"  - {obj['name']:<45}: {obj['pos']}")
        print("="*40 + "\n")

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
        
        # Tracking EEF (Site Standard)
        target_site = "gripper0_right_grip_site" 
        try:
            self.eef_site_id = self.sim.model.site_name2id(target_site)
            print(f"[INIT] Tracking EEF su SITE: '{target_site}' (ID {self.eef_site_id})")
        except:
            print(f"[WARN] Site '{target_site}' non trovato! Cerco fallback...")
            self.eef_site_id = None

        # --- FIX: TROVARE NOMI GIUNTI DAL SIMULATORE ---
        # Invece di usare robot.joint_names, scansioniamo il modello MuJoCo
        self.arm_joint_names = []
        for i in range(self.sim.model.njnt):
            name = self.sim.model.joint_id2name(i)
            # I giunti del braccio Panda iniziano solitamente con "robot0_joint"
            # e non sono finger (pinza) o wheel (ruote)
            if "robot0_joint" in name and "finger" not in name and "wheel" not in name:
                self.arm_joint_names.append(name)
        
        print(f"[INIT] Giunti braccio rilevati per monitoraggio: {len(self.arm_joint_names)}")

    def get_robot_eef_pose(self):
        self.sim.forward()
        if self.eef_site_id is not None:
            return self.sim.data.site_xpos[self.eef_site_id].copy()
        try:
            bid = self.sim.model.body_name2id("robot0_link7")
            return self.sim.data.body_xpos[bid].copy()
        except:
            return np.array([0, 0, 0])

    def check_limits_auto(self):
        """Controlla i limiti usando i nomi trovati nel simulatore."""
        limit_reached = False
        msg = "!!! ATTENZIONE LIMITI: "
        
        for j_name in self.arm_joint_names:
            try:
                j_id = self.sim.model.joint_name2id(j_name)
                q_idx = self.sim.model.jnt_qposadr[j_id]
                curr_val = self.sim.data.qpos[q_idx]
                min_val, max_val = self.sim.model.jnt_range[j_id]
                
                range_span = max_val - min_val
                if range_span > 0:
                    pct = (curr_val - min_val) / range_span * 100
                    if pct < 5 or pct > 95:
                        msg += f"[{j_name[-2:]}: {pct:.0f}%] "
                        limit_reached = True
            except:
                continue # Salta se c'è un problema con un giunto specifico
        
        if limit_reached:
            # Stampa evidenziata
            print(f" >>> {msg} <<<") 

    def get_joint_status_full(self):
        """Report completo su richiesta."""
        status_msg = []
        for j_name in self.arm_joint_names:
            try:
                j_id = self.sim.model.joint_name2id(j_name)
                q_idx = self.sim.model.jnt_qposadr[j_id]
                curr_val = self.sim.data.qpos[q_idx]
                min_val, max_val = self.sim.model.jnt_range[j_id]
                pct = (curr_val - min_val) / (max_val - min_val) * 100
                status_msg.append(f"{j_name}: {pct:.1f}%")
            except:
                pass
        return status_msg

    def build_action(self, base_vel=(0,0,0), torso_vel=0.0, arm_delta=(0,0,0,0,0,0), gripper_cmd=1.0):
        action = np.zeros(self.action_dim)
        action[self.idx_arm] = np.clip(arm_delta, -1, 1)
        action[self.idx_gripper] = gripper_cmd
        action[self.idx_base] = np.clip(base_vel, -1, 1)
        action[self.idx_torso] = np.clip(torso_vel, -1, 1)
        return action

def main():
    env_name = "PnPCounterToCab"
    
    # --- CONFIGURAZIONE "SOFT" ---
    controller_config = {
        "type": "OSC_POSE",
        "input_max": 1, "input_min": -1,
        "output_max": [1]*6, "output_min": [-1]*6,
        "kp": 80, "damping": 2, "impedance_mode": "fixed",
        "kp_limits": [0, 300], "damping_limits": [0, 10],
        "uncouple_pos_ori": True, "control_delta": True,
        "interpolation": None, "ramp_ratio": 0.2
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
        print(f"\nERRORE CRITICO: {e}")
        return

    agent = PandaOmronAgent(env)
    env.reset()
    
    current_state = {"gripper": -1.0}
    
    # --- SENSIBILITÀ COMANDI ---
    ARM_SENSITIVITY = 0.05 
    BASE_SENSITIVITY = 1.0
    
    print("\n--- CONTROLLER FLUIDO (FIXED) ---")
    print("Modifiche: Loop Giunti MuJoCo diretto (No crash)")
    print("Comandi: 'base x y w', 'arm x y z', 'torso v', 'info', 'q'")
    
    while True:
        env.render()
        try:
            user_input = input("Cmd > ").strip().lower()
        except EOFError: break
        
        if user_input == 'q': break
        
        parts = user_input.split()
        if not parts: continue
        cmd = parts[0]
        
        base_cmd = [0,0,0]
        arm_cmd = [0,0,0,0,0,0]
        torso_cmd = 0.0
        steps = 0 

        try:
            if cmd == 'info':
                pos = agent.get_robot_eef_pose()
                print(f"EEF: {np.round(pos, 3)}")
                print("--- Stato Giunti ---")
                for j in agent.get_joint_status_full(): print(j)
                
            elif cmd == 'base': 
                if len(parts) >= 4:
                    base_cmd = [float(parts[1])*BASE_SENSITIVITY, float(parts[2])*BASE_SENSITIVITY, float(parts[3])*BASE_SENSITIVITY]
                    steps = 40
                else: print("Usa: base vx vy w")
                    
            elif cmd == 'torso': 
                if len(parts) >= 2:
                    torso_cmd = float(parts[1])
                    steps = 30
                else: print("Usa: torso vel")

            elif cmd == 'arm': 
                if len(parts) >= 4:
                    raw_x = float(parts[1])
                    raw_y = float(parts[2])
                    raw_z = float(parts[3])
                    # Applica sensibilità
                    arm_cmd[0] = raw_x * ARM_SENSITIVITY
                    arm_cmd[1] = raw_y * ARM_SENSITIVITY
                    arm_cmd[2] = raw_z * ARM_SENSITIVITY
                    steps = 20
                else: print("Usa: arm x y z")
            
            elif cmd == 'neutral':
                print("Rilassamento braccio...")
                arm_cmd = [0,0,0,0,0,0] 
                steps = 50 
                    
            elif cmd == 'open':
                current_state["gripper"] = -1.0
                steps = 20
                
            elif cmd == 'close':
                current_state["gripper"] = 1.0
                steps = 20
            
            if steps > 0:
                print(f"Eseguo {cmd}...", end=" ", flush=True)
                for _ in range(steps):
                    action = agent.build_action(
                        base_vel=base_cmd,
                        torso_vel=torso_cmd, 
                        arm_delta=arm_cmd, 
                        gripper_cmd=current_state["gripper"]
                    )
                    env.step(action)
                    env.render()
                    agent.check_limits_auto() # Ora è sicuro chiamarlo
                print("Fatto.")
                    
        except ValueError:
            print("Errore numerico.")
        except Exception as e:
            print(f"Errore: {e}")

    env.close()

if __name__ == "__main__":
    main()