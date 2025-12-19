import sys
import os
import numpy as np
import robocasa
import robosuite
import time

# Usa un environment "vuoto" o semplice per evitare collisioni
ENV_NAME = "PnPCounterToCab" 
ROBOT_NAME = "PandaMobile"
CAMERA_NAME = "robot0_agentview_center" # Puoi provare anche 'robot0_eye_in_hand' qui se vuoi

def main():
    print("=== CALIBRAZIONE ASSI ROBOT ===")
    print("Questo script muoverà il robot su un asse alla volta.")
    print("Tu dovrai guardare il simulatore e dire dove si è mosso.")
    
    # Configurazione Controller (Stessa del main)
    controller_config = {
        "type": "OSC_POSE",
        "input_max": 1,
        "input_min": -1,
        "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
        "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
        "kp": 150,
        "damping_ratio": 1,
        "impedance_mode": "fixed",
        "kp_limits": [0, 300],
        "damping_ratio_limits": [0, 10],
        "uncouple_pos_ori": True,
        "control_delta": True,
        "interpolation": None,
        "ramp_ratio": 0.2
    }

    try:
        env = robosuite.make(
            env_name=ENV_NAME,
            robots=[ROBOT_NAME],
            controller_configs=controller_config,
            has_renderer=True,
            has_offscreen_renderer=False,
            use_camera_obs=False, # Non ci serve la camera per calibrare, guardi tu
            render_camera=CAMERA_NAME
        )
    except Exception as e:
        print(f"[ERRORE] {e}")
        return

    obs = env.reset()
    action_dim = env.action_dim
    print(f"Robot inizializzato. Action Dim: {action_dim}")
    print("Premi INVIO per iniziare la calibrazione...")
    input()

    # Mappatura assi da testare (Solo posizione X, Y, Z per ora)
    axes_labels = ["X (0)", "Y (1)", "Z (2)"]
    calibration_result = [1.0, 1.0, 1.0] # [sign_x, sign_y, sign_z]
    
    # Test Loop
    for axis_idx, label in enumerate(axes_labels):
        print(f"\n--- TEST ASSE {label} ---")
        print("Sto inviando comando POSITIVO (+1.0)...")
        
        # Reset Env per pulire
        # env.reset() # Opzionale, meglio di no per mantenere continuità visiva
        
        # Crea azione: tutto 0 tranne l'asse corrente
        action = np.zeros(action_dim)
        action[axis_idx] = 1.0 * 80.0 # Uso un gain alto per vederlo bene
        
        # Esegui per 20 step per renderlo visibile
        for _ in range(30):
            env.step(action)
            env.render()
            time.sleep(0.02)
            
        print("\nDOVE SI E' MOSSO IL BRACCIO (rispetto alla TUA vista della camera)?")
        print("  f = Forward (Avanti/Lontano da te)")
        print("  b = Backward (Indietro/Verso di te)")
        print("  r = Right (Destra)")
        print("  l = Left (Sinistra)")
        print("  u = Up (Su)")
        print("  d = Down (Giù)")
        user_input = input("Risposta: ").strip().lower()
        
        # Logica di inversione basata su OpenVLA Standard
        # OpenVLA standard: 
        #   X = Avanti (Forward)
        #   Y = Destra (Right)
        #   Z = Su (Up)
        
        correction = 1.0
        if axis_idx == 0: # Stiamo testando X (Dovrebbe essere Forward)
            if user_input == 'b': correction = -1.0 # Se è andato indietro, invertiamo
            print(f"  -> Asse X mappato come: {'Normale' if correction==1 else 'INVERTITO'}")
            calibration_result[0] = correction
            
        elif axis_idx == 1: # Stiamo testando Y (Dovrebbe essere Right)
            if user_input == 'l': correction = -1.0 # Se è andato a sinistra, invertiamo
            print(f"  -> Asse Y mappato come: {'Normale' if correction==1 else 'INVERTITO'}")
            calibration_result[1] = correction

        elif axis_idx == 2: # Stiamo testando Z (Dovrebbe essere Up)
            if user_input == 'd': correction = -1.0
            print(f"  -> Asse Z mappato come: {'Normale' if correction==1 else 'INVERTITO'}")
            calibration_result[2] = correction
            
        # Torna indietro per resettare posizione (circa)
        undo_action = action * -1.0
        for _ in range(10):
            env.step(undo_action)
            env.render()

    print("\n\n=== RISULTATO CALIBRAZIONE ===")
    print("Copia e incolla questa riga nel tuo script `run_simulation.py` dentro `adapt_action`:")
    print(f"AXIS_SIGNS = np.array({calibration_result} + [1.0, 1.0, 1.0]) # [x,y,z, rx,ry,rz]")
    
    env.close()

if __name__ == "__main__":
    main()