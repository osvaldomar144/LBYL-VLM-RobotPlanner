# interactive_vlm.py
import time
import cv2
import sys
import numpy as np
from robot_vlm_lib import VLMRobotInterface

def save_vlm_views(visual_dict, step_count):
    """Salva le immagini su disco sovrascrivendo le precedenti."""
    for cam_name, img in visual_dict.items():
        # Robosuite da RGB, OpenCV vuole BGR
        bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        filename = f"view_{cam_name}.png"
        cv2.imwrite(filename, bgr_img)
    print(f"   [FOTO] Immagini salvate: view_*.png (Step {step_count})")

def print_help():
    print("\n--- COMANDI DISPONIBILI (SINTASSI VLM) ---")
    print(" 1. Navigazione Base:")
    print("    base vx vy w      -> Es: 'base -1 0 0' (Indietro), 'base 0 0 1' (Ruota SX)")
    print(" 2. Movimento Braccio (Delta):")
    print("    arm x y z         -> Es: 'arm 0.5 0 0' (Avanti), 'arm 0 0 0.5' (Su)")
    print(" 3. Torso:")
    print("    torso val         -> Es: 'torso 0.5' (Alza), 'torso -0.5' (Abbassa)")
    print(" 4. Pinza:")
    print("    gripper open      -> Apre")
    print("    gripper close     -> Chiude")
    print(" 5. Altro:")
    print("    q                 -> Esci")
    print("------------------------------------------")

def main():
    print("=== INTERFACCIA VLM INTERATTIVA ===")
    print("Inizializzazione ambiente (può richiedere qualche secondo)...")
    
    try:
        # Inizializza la libreria
        robot = VLMRobotInterface(env_name="PnPCounterToCab", render=True)
    except Exception as e:
        print(f"Errore critico inizializzazione: {e}")
        return

    print_help()
    print("\nconsiglio: Se sei incastrato, prova subito 'base -0.5 0 0' per indietreggiare.")

    step_counter = 0

    while True:
        step_counter += 1
        
        # 1. ACQUISIZIONE DATI (LOOK)
        # Recuperiamo info testuali e immagini
        text_info, visual_dict = robot.get_context()
        
        # 2. SALVATAGGIO FOTO
        # Salviamo subito le foto così puoi guardarle mentre decidi il comando
        save_vlm_views(visual_dict, step_counter)

        # 3. MOSTRA STATO
        pos = text_info['eef_xyz']
        grip = text_info['gripper']
        safe = text_info['safety_warning']
        
        # Coloriamo l'output di safety se non è nominale
        safe_str = f"\033[92m{safe}\033[0m" if safe == "nominal" else f"\033[91m{safe}\033[0m"
        
        print(f"\n[STATO] PosEEF: {pos} | Grip: {grip} | Safety: {safe_str}")

        # 4. INPUT COMANDO (PLAN)
        try:
            user_input = input("VLM Action > ").strip().lower()
        except EOFError: break
        
        if user_input == 'q' or user_input == 'quit': 
            break
        
        if not user_input: continue

        # Parsing del comando
        parts = user_input.split()
        primitive = parts[0]
        params = []

        try:
            valid_cmd = False
            
            if primitive == 'base' and len(parts) >= 4:
                # base vx vy w
                params = [float(parts[1]), float(parts[2]), float(parts[3])]
                valid_cmd = True
                
            elif primitive == 'arm' and len(parts) >= 4:
                # arm x y z
                params = [float(parts[1]), float(parts[2]), float(parts[3])]
                valid_cmd = True
                
            elif primitive == 'torso' and len(parts) >= 2:
                # torso val
                params = [float(parts[1])]
                valid_cmd = True
                
            elif primitive == 'gripper' and len(parts) >= 2:
                # gripper open/close
                params = [parts[1]] # Passiamo la stringa, la lib la gestisce
                valid_cmd = True
            
            else:
                print(" -> Comando non riconosciuto o parametri mancanti. Riprova.")

            # 5. ESECUZIONE (LEAP)
            if valid_cmd:
                print(f" -> Eseguo: {primitive} {params} ...")
                result = robot.execute_action(primitive, params)
                
                # Feedback post-azione
                if result.get('movement_clamped', False):
                    print(" -> \033[93m[WARNING] L'azione è stata rallentata dalla Safety!\033[0m")
                
                # Se abbiamo toccato un limite critico
                new_safe = result.get('safety_warning', 'nominal')
                if new_safe != 'nominal':
                    print(f" -> \033[91m[ALERT] Attenzione limiti giunti: {new_safe}\033[0m")

        except ValueError:
            print(" -> Errore: Assicurati di usare numeri validi (es. 0.5, -1.0).")
        except Exception as e:
            print(f" -> Errore imprevisto: {e}")

    robot.close()
    print("Simulazione terminata.")

if __name__ == "__main__":
    main()