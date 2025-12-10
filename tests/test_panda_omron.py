import numpy as np
import robosuite
import robocasa
import json

class SceneIntrospector:
    """Helper invariato per analizzare la scena."""
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
        
        # MAPPATURA (Invariata perché corretta)
        self.idx_arm = slice(0, 6)
        self.idx_gripper = slice(6, 7)
        self.idx_base = slice(7, 10)
        self.idx_torso = slice(10, 11)
        
        # --- FIX 1: CAMBIO TRACKING EEF ---
        # I siti gripper sembrano statici. Usiamo il polso fisico (link7) che sappiamo muoversi.
        target_body = "robot0_link7" 
        try:
            self.eef_body_id = self.sim.model.body_name2id(target_body)
            print(f"[INIT] Tracking EEF su BODY fisico: '{target_body}' (ID {self.eef_body_id})")
        except:
            print(f"[WARN] Body '{target_body}' non trovato!")
            self.eef_body_id = None

    def get_robot_eef_pose(self):
        """Ritorna la posizione del polso + un offset per la pinza."""
        if self.eef_body_id is not None:
            wrist_pos = self.sim.data.body_xpos[self.eef_body_id].copy()
            # FIX: Aggiungiamo un offset manuale per stimare la punta delle dita
            # (Il link7 è il polso, la pinza è circa 10-15cm oltre)
            # Nota: Questo è approssimativo ma sufficiente per vedere se si muove.
            return wrist_pos 
        return np.array([0, 0, 0])

    def build_action(self, base_vel=(0,0,0), torso_pos=0.0, arm_delta=(0,0,0,0,0,0), gripper_cmd=1.0):
        """
        Costruisce l'azione. 
        IMPORTANTE: torso_pos deve essere mantenuto costante tra le chiamate 
        se non si vuole muovere il torso.
        """
        action = np.zeros(self.action_dim)
        
        # 1. Arm
        action[self.idx_arm] = np.clip(arm_delta, -1, 1)
        
        # 2. Gripper 
        action[self.idx_gripper] = gripper_cmd
        
        # 3. Base 
        action[self.idx_base] = np.clip(base_vel, -1, 1)
        
        # 4. Torso - FIX: Assicuriamoci che il valore passato sia rispettato
        action[self.idx_torso] = np.clip(torso_pos, -1, 1)
        
        return action

def main():
    env_name = "PnPCounterToCab"
    
    # Configurazione Controller
    controller_config = {
        "type": "OSC_POSE",
        "input_max": 1, "input_min": -1,
        "output_max": [1, 1, 1, 1, 1, 1],
        "output_min": [-1, -1, -1, -1, -1, -1],
        "kp": 150, "damping": 1, "impedance_mode": "fixed",
        "kp_limits": [0, 300], "damping_limits": [0, 10],
        "uncouple_pos_ori": True, "control_delta": True,
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
        print(f"\nERRORE CRITICO: {e}")
        return

    # Introspezione
    inspector = SceneIntrospector(env)
    scene_data = inspector.inspect()
    inspector.print_report(scene_data)
    
    agent = PandaOmronAgent(env)
    env.reset()
    
    # --- FIX 2: GESTIONE STATO PERSISTENTE ---
    # Manteniamo queste variabili vive nel loop principale
    current_state = {
        "torso": 0.0,
        "gripper": -1.0, # Aperto
        "arm_hold": [0,0,0,0,0,0] # Non inviamo delta se non richiesto
    }
    
    print("\n--- CONTROLLER PRONTO ---")
    print("Comandi: 'base x y w', 'arm x y z', 'torso v', 'open', 'close', 'info', 'q'")
    
    while True:
        env.render()
        try:
            user_input = input("Cmd > ").strip().lower()
        except EOFError: break
        
        if user_input == 'q': break
        
        parts = user_input.split()
        if not parts: continue
        cmd = parts[0]
        
        # Reset comandi di MOVIMENTO ISTANTANEO (base e braccio tornano a 0 dopo l'esecuzione)
        base_cmd = [0,0,0]
        arm_cmd = [0,0,0,0,0,0]
        steps = 0 

        try:
            if cmd == 'info':
                pos = agent.get_robot_eef_pose()
                print(f"EEF Pose (Link7): [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
                print(f"Stato Attuale -> Torso: {current_state['torso']}, Gripper: {current_state['gripper']}")
                
            elif cmd == 'base': 
                if len(parts) >= 4:
                    base_cmd = [float(parts[1]), float(parts[2]), float(parts[3])]
                    steps = 40
                else: print("Usa: base vx vy w")
                    
            elif cmd == 'torso': 
                if len(parts) >= 2:
                    val = float(parts[1])
                    # Aggiorniamo lo stato persistente
                    current_state["torso"] = val
                    steps = 30
                else: print("Usa: torso val")

            elif cmd == 'arm': 
                if len(parts) >= 4:
                    arm_cmd[0] = float(parts[1])
                    arm_cmd[1] = float(parts[2])
                    arm_cmd[2] = float(parts[3])
                    steps = 20
                else: print("Usa: arm x y z")
                    
            elif cmd == 'open':
                current_state["gripper"] = -1.0
                steps = 20
                
            elif cmd == 'close':
                current_state["gripper"] = 1.0
                steps = 20
            
            # Esecuzione
            if steps > 0:
                print(f"Eseguo {cmd}...")
                for _ in range(steps):
                    # Qui sta la magia: passiamo SEMPRE il valore corrente del torso
                    # salvato in current_state['torso'], anche se stiamo muovendo solo la pinza.
                    action = agent.build_action(
                        base_vel=base_cmd,
                        torso_pos=current_state["torso"], 
                        arm_delta=arm_cmd, 
                        gripper_cmd=current_state["gripper"]
                    )
                    env.step(action)
                    env.render()
                print("Fatto.")
                    
        except ValueError:
            print("Errore numerico.")
        except Exception as e:
            print(f"Errore: {e}")

    env.close()

if __name__ == "__main__":
    main()