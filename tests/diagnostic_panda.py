import numpy as np
import robosuite
import robocasa
import time

class DiagnosticAgent:
    def __init__(self, env):
        self.env = env
        self.sim = env.sim
        # Totale azioni per PandaOmron
        self.action_dim = 12

    def find_real_eef(self):
        """
        Cerca il vero corpo della mano che si muove.
        Invece di cercare un 'site', cerchiamo il 'body' fisico della mano.
        """
        possible_names = ["right_hand", "link7", "gripper0_eef"]
        print("\n--- RICERCA BODY END-EFFECTOR ---")
        found_id = -1
        found_name = ""
        
        for i in range(self.sim.model.nbody):
            name = self.sim.model.body_id2name(i)
            # Stampa i body che sembrano parti finali del braccio
            if any(x in name for x in ["hand", "gripper", "link7"]):
                print(f"Trovato candidate body: '{name}' (ID: {i})")
                # Preferiamo 'right_hand' o 'link7' perché sono fisici
                if "right_hand" in name or "hand" in name:
                    found_id = i
                    found_name = name

        if found_id != -1:
            print(f">>> SELEZIONATO: {found_name} (ID {found_id})")
            return found_id
        return None

    def get_body_pose(self, body_id):
        if body_id is not None:
            return self.sim.data.body_xpos[body_id]
        return np.zeros(3)

def main():
    print("Avvio Diagnostica PandaOmron...")
    
    # Configurazione Controller
    controller_config = {
        "type": "OSC_POSE",
        "input_max": 1, "input_min": -1,
        "output_max": [1]*6, "output_min": [-1]*6,
        "kp": 150, "damping": 1, "impedance_mode": "fixed",
        "kp_limits": [0, 300], "damping_limits": [0, 10],
        "uncouple_pos_ori": True, "control_delta": True, 
        "interpolation": None, "ramp_ratio": 0.2
    }

    env = robosuite.make(
        env_name="PnPCounterToCab",
        robots="PandaMobile",
        controller_configs=controller_config,
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        render_camera=None, # Usa default
        ignore_done=True
    )
    
    agent = DiagnosticAgent(env)
    eef_body_id = agent.find_real_eef()
    env.reset()
    
    print("\n" + "="*50)
    print("TEST MOTORI: Guarda la finestra e leggi qui sotto!")
    print("="*50)
    
    # TEST 1: Identificazione Indici
    # Proviamo a muovere gruppi di indici per capire chi fa cosa
    
    # Gruppo A: Indici 0-5 (Solitamente Braccio)
    print("\n>>> TEST GRUPPO A [0:6] (Dovrebbe essere il BRACCIO)")
    input("Premi ENTER per muovere indici 0-6...")
    for _ in range(30):
        action = np.zeros(12)
        action[2] = 0.5 # Prova a muovere asse Z del braccio
        env.step(action)
        env.render()
    print("Fatto. Il braccio si è mosso?")

    env.reset()
    
    # Gruppo B: Indice 6 (Dovrebbe essere TORSO... ma forse è GRIPPER?)
    print("\n>>> TEST GRUPPO B [Indice 6]")
    input("Premi ENTER per inviare 1.0 all'indice 6...")
    for _ in range(40):
        action = np.zeros(12)
        action[6] = 1.0 
        # Mantieni il torso fermo sugli altri canali se possibile
        env.step(action)
        env.render()
        # Debug posizione
        if eef_body_id:
            pos = agent.get_body_pose(eef_body_id)
            # print(f"Pos EEF: {pos}") 
    print("Fatto. Cosa si è mosso? (Se si è aperta la pinza -> Indice 6 è Gripper)")

    env.reset()

    # Gruppo C: Indici 7-9 (Solitamente BASE)
    print("\n>>> TEST GRUPPO C [7:10]")
    input("Premi ENTER per muovere la BASE (avanti)...")
    for _ in range(30):
        action = np.zeros(12)
        action[7] = 0.5 # Velocità X
        env.step(action)
        env.render()
    print("Fatto. La base si è mossa?")
    
    env.reset()

    # Gruppo D: Indici 10-11 (Dovrebbe essere GRIPPER... ma forse è TORSO?)
    print("\n>>> TEST GRUPPO D [10:12]")
    input("Premi ENTER per inviare 1.0 agli indici 10 e 11...")
    for _ in range(40):
        action = np.zeros(12)
        action[10] = 1.0
        action[11] = 1.0
        env.step(action)
        env.render()
    print("Fatto. Cosa si è mosso? (Se il robot è salito -> 10 è il Torso)")

    print("\n" + "="*50)
    print("DIAGNOSTICA COMPLETATA")
    print("="*50)
    env.close()

if __name__ == "__main__":
    main()