import argparse
import numpy as np
import robocasa
import time
import sys
import traceback

# Importa il controller
from vila_open.controller import OracleController

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="PnPCounterToSink")
    parser.add_argument("--robot", type=str, default="PandaMobile")
    args = parser.parse_args()

    print(f"\n--- TEST INTERATTIVO ROBUSTO ---")
    print(f"Env: {args.env_name}, Robot: {args.robot}")
    
    # Setup ambiente
    env = robocasa.make(
        env_name=args.env_name,
        robots=args.robot,
        seed=0,
        has_renderer=True,           
        has_offscreen_renderer=False,
        use_camera_obs=False,        
        render_camera=None
    )

    controller = OracleController(env)
    
    obs = env.reset()
    env.render()

    print("\n--- ISTRUZIONI ---")
    print(" 1. 'pick bell_pepper' (esegue nav -> untuck -> pick)")
    print(" 2. 'navigate counter' (solo navigazione)")
    print(" 3. 'look' (movimento testa - disabilitato/ignorato)")
    print(" 4. 'exit'")
    print("------------------\n")

    try:
        while True:
            user_input = input("Comando > ").strip().lower()
            
            if user_input in ["exit", "quit", "q"]:
                break
            
            if not user_input:
                continue

            # Parsing comando
            parts = user_input.split(" ", 1)
            primitive = parts[0]
            obj_name = parts[1] if len(parts) > 1 else "obj_main"

            print(f"Esecuzione: {primitive} -> {obj_name} ...")

            try:
                # Esecuzione Primitiva
                action_gen = controller.execute_primitive(primitive, obj_name)
                
                step_count = 0
                if action_gen:
                    for action in action_gen:
                        obs, reward, done, info = env.step(action.astype(np.float32))
                        env.render()
                        step_count += 1
                    print(f"Completato in {step_count} step.\n")
                else:
                    print("Nessuna azione generata.")

            except Exception as e:
                print(f"Errore durante l'esecuzione: {e}")
                traceback.print_exc()
                print("--- Puoi provare un altro comando ---")

    except KeyboardInterrupt:
        print("\nInterrotto dall'utente.")
    finally:
        env.close()
        print("Ambiente chiuso.")

if __name__ == "__main__":
    main()