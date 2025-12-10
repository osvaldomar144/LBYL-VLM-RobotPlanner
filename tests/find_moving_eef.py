import numpy as np
import robosuite
import robocasa   # <--- QUESTA ERA LA RIGA MANCANTE
import time

def main():
    print("--- CACCIA ALL'END EFFECTOR MOBILE ---")
    
    # Configurazione controller
    controller_config = {
        "type": "OSC_POSE",
        "input_max": 1, "input_min": -1,
        "output_max": [1]*6, "output_min": [-1]*6,
        "kp": 150, "damping": 1, "impedance_mode": "fixed",
        "kp_limits": [0, 300], "damping_limits": [0, 10],
        "uncouple_pos_ori": True, "control_delta": True, 
        "interpolation": None, "ramp_ratio": 0.2
    }

    print("Caricamento environment...")
    try:
        env = robosuite.make(
            env_name="PnPCounterToCab",
            robots="PandaMobile",
            controller_configs=controller_config,
            has_renderer=True,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            render_camera=None,
            ignore_done=True
        )
    except Exception as e:
        print(f"Errore caricamento: {e}")
        return
    
    env.reset()
    sim = env.sim
    
    # 1. Raccogliamo tutti i candidati (SITES e BODIES)
    candidates = {}
    
    # Cerca nei SITES (di solito i più precisi per il tracking)
    print("\n[1] Scansione Sites...")
    for i in range(sim.model.nsite):
        name = sim.model.site_id2name(i)
        # Cerchiamo parole chiave come grip, ee, hand
        if any(k in name for k in ["grip", "ee", "hand", "tip"]):
            candidates[f"SITE::{name}"] = {
                "type": "site", "id": i, 
                "start_pos": sim.data.site_xpos[i].copy()
            }
            
    # Cerca nei BODIES (alternativa se i sites non vanno)
    print("[2] Scansione Bodies...")
    for i in range(sim.model.nbody):
        name = sim.model.body_id2name(i)
        if any(k in name for k in ["grip", "ee", "hand", "link7"]):
            candidates[f"BODY::{name}"] = {
                "type": "body", "id": i, 
                "start_pos": sim.data.body_xpos[i].copy()
            }

    print(f"Monitoraggio di {len(candidates)} candidati potenziali.")
    
    # 2. Muoviamo il robot (Braccio + Torso)
    print("\n[3] Esecuzione movimento (Arm + Torso)...")
    
    # Creiamo un'azione che muove sicuramente il braccio e il torso
    # Basandoci sui tuoi test: Indice 2 (Arm Z), Indice 10 (Torso)
    action = np.zeros(12)
    action[2] = 0.5   # Arm Z+
    action[10] = 0.5  # Torso Up
    
    # Eseguiamo per 50 step per dare tempo al movimento di avvenire
    for _ in range(50):
        env.step(action)
        # env.render() # Decommenta se vuoi vederlo, ma rallenta
        
    # 3. Controllo Delta (Cosa si è mosso?)
    print("\n[4] Analisi Spostamenti:")
    best_candidate = None
    max_dist = 0.0
    
    print(f"{'NOME':<50} | {'SPOSTAMENTO (m)':<15}")
    print("-" * 70)
    
    for name, data in candidates.items():
        if data["type"] == "site":
            curr_pos = sim.data.site_xpos[data["id"]]
        else:
            curr_pos = sim.data.body_xpos[data["id"]]
            
        # Calcola distanza Euclidea
        dist = np.linalg.norm(curr_pos - data["start_pos"])
        
        # Filtriamo il rumore (dev'essersi mosso almeno di 1cm)
        if dist > 0.01:
            print(f"{name:<50} | {dist:.4f}")
            # Cerchiamo quello che si è mosso di più (è probabile sia l'EEF)
            if dist > max_dist:
                max_dist = dist
                best_candidate = (name, data["id"], data["type"])
    
    print("-" * 70)
    
    if best_candidate:
        full_name, id_val, type_val = best_candidate
        clean_name = full_name.split("::")[1]
        print(f"\n>>> VINCITORE ASSOLUTO: {full_name}")
        print(f">>> ID MuJoCo: {id_val}")
    else:
        print("NESSUN movimento rilevato sopra la soglia. Assicurati che il robot si muova.")

    env.close()

if __name__ == "__main__":
    main()