# scripts/robocasa_native_replay.py

import argparse
import time
import numpy as np
import robocasa  # Usiamo robocasa.make

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episode_path",
        type=str,
        default="recorded_episode.npz",
        help="Percorso del file .npz con l'episodio registrato.",
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default=None,
        help="Nome dell'env (se vuoi sovrascrivere quello nel file).",
    )
    parser.add_argument(
        "--step_sleep",
        type=float,
        default=0.02,
        help="Sleep (secondi) tra uno step e l'altro.",
    )
    args = parser.parse_args()

    print(f"[Replay] Carico episodio da {args.episode_path}")
    try:
        data = np.load(args.episode_path, allow_pickle=True)
    except FileNotFoundError:
        print(f"[Error] File non trovato: {args.episode_path}")
        return

    # === 1. RECUPERO DATI ===
    if "env" in data:
        file_env_name = str(data["env"])
    elif "env_name" in data:
        file_env_name = str(data["env_name"])
    else:
        file_env_name = "PnPCounterToSink"

    if "actions" in data:
        low_actions = data["actions"]
    elif "low_actions" in data:
        low_actions = data["low_actions"]
    else:
        raise KeyError("Nessuna chiave 'actions' o 'low_actions' trovata nel file .npz!")

    if "seed" in data:
        file_seed = int(data["seed"])
    else:
        file_seed = 0

    env_name = args.env_name or file_env_name
    
    print(f"[Replay] env_name: {env_name}")
    print(f"[Replay] seed:     {file_seed}")
    print(f"[Replay] steps:    {low_actions.shape[0]}")
    print(f"[Replay] action_dim: {low_actions.shape[1] if len(low_actions.shape) > 1 else 'Unknown'}")

    # === 2. CREAZIONE ENV (CORRETTA) ===
    print(f"[Replay] Creo env con viewer nativo: {env_name}")
    
    # FIX: Dobbiamo usare PandaMobile perché le azioni registrate sono a 12 dimensioni!
    env = robocasa.make(
        env_name=env_name,
        robots="PandaMobile",        # <--- MODIFICA FONDAMENTALE
        seed=file_seed,
        has_renderer=True,           # ABILITA FINESTRA MUJOCO
        has_offscreen_renderer=False,
        use_camera_obs=False,
        render_camera=None,
    )

    try:
        obs = env.reset()
        print("[Replay] Env reset completato. Premi SPACE nella finestra MuJoCo per avviare.")
        
        # Piccola pausa per lasciare caricare la finestra
        time.sleep(1.0)

        for idx, action in enumerate(low_actions):
            action_to_apply = np.asarray(action, dtype=np.float32)
            
            obs, reward, done, info = env.step(action_to_apply)
            
            env.render()

            if args.step_sleep > 0.0:
                time.sleep(args.step_sleep)

            if done:
                print(f"[Replay] Episodio terminato (done=True) al passo {idx}.")
                break

        print("[Replay] Replay completato.")
        time.sleep(2.0)

    except KeyboardInterrupt:
        print("\n[Replay] Interrotto dall'utente.")
    finally:
        env.close()
        print("[Replay] Env chiuso.")

if __name__ == "__main__":
    main()