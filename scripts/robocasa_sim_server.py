# scripts/robocasa_sim_server.py

import argparse
from multiprocessing.connection import Listener
import subprocess

import numpy as np

from robocasa.utils.env_utils import create_env
from vila_open.robocasa_utils import obs_to_pil_image

# ============================================
# Monitor GPU: NVML (pynvml) o nvidia-smi
# ============================================

_HAS_NVML = False
try:
    import pynvml

    pynvml.nvmlInit()
    _HAS_NVML = True
    print("[GPU server] NVML inizializzato.")
except Exception as e:
    print(f"[GPU server] NVML non disponibile ({e}). Proverò con nvidia-smi se presente.")


def _get_gpu_status():
    """
    Ritorna un dizionario con stato GPU.
    Prova prima con NVML, poi con nvidia-smi, altrimenti errore.
    """
    if _HAS_NVML:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode()

            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)

            return {
                "backend": "nvml",
                "name": name,
                "memory_used_MB": mem.used / (1024**2),
                "memory_total_MB": mem.total / (1024**2),
                "util_gpu_pct": util.gpu,
                "util_mem_pct": util.memory,
            }
        except Exception as e:
            return {"backend": "nvml", "error": str(e)}

    # Fallback: nvidia-smi
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total,utilization.gpu,utilization.memory",
            "--format=csv,noheader,nounits",
        ]
        out = subprocess.check_output(cmd, encoding="utf-8").strip()
        line = out.splitlines()[0]
        parts = [p.strip() for p in line.split(",")]

        mem_used = float(parts[0])
        mem_total = float(parts[1])
        util_gpu = float(parts[2])
        util_mem = float(parts[3])

        return {
            "backend": "nvidia-smi",
            "memory_used_MB": mem_used,
            "memory_total_MB": mem_total,
            "util_gpu_pct": util_gpu,
            "util_mem_pct": util_mem,
        }
    except Exception as e:
        return {"backend": "none", "error": str(e)}


def log_gpu_status(prefix: str = "[GPU server]"):
    status = _get_gpu_status()
    if "error" in status:
        print(
            f"{prefix} Info GPU non disponibili (backend={status.get('backend')}): {status['error']}"
        )
    else:
        name = status.get("name", "")
        mem_used = status["memory_used_MB"]
        mem_total = status["memory_total_MB"]
        util_gpu = status["util_gpu_pct"]
        util_mem = status["util_mem_pct"]
        backend = status["backend"]

        if name:
            print(
                f"{prefix} {name} | mem {mem_used:.0f}/{mem_total:.0f} MB | "
                f"util {util_gpu:.0f}% gpu, {util_mem:.0f}% mem (via {backend})"
            )
        else:
            print(
                f"{prefix} mem {mem_used:.0f}/{mem_total:.0f} MB | "
                f"util {util_gpu:.0f}% gpu, {util_mem:.0f}% mem (via {backend})"
            )


# ============================================
# Logica SimServer
# ============================================


def execute_high_level_action(
    ctrl_env,
    action_dict: dict,
    steps_per_action: int = 40,
    frame_width: int = 512,
    frame_height: int = 512,
):
    """
    Esegue un'azione high-level come sequenza di azioni low-level (placeholder random).

    ctrl_env : env OFFSCREEN (con immagini in obs) usato per lo stato "vero".

    Ritorna:
      - obs      : ultima osservazione
      - done     : episodic done
      - success  : flag di successo
      - info     : dict info
      - frames   : lista di frame RGB (np.ndarray HxWx3 uint8) per visualizzazione
    """
    print(f"[SimServer] EXEC high-level action: {action_dict}")
    log_gpu_status(prefix="[GPU server][prima azione]")

    obs = None
    done = False
    info = {}
    frames: list[np.ndarray] = []

    for t in range(steps_per_action):
        # Azione low-level random un po' più "energica" per vedere movimento
        low_action = np.random.randn(*ctrl_env.action_spec[0].shape) * 0.3

        # Step sull'env di controllo (che contiene anche le immagini in obs)
        obs, reward, done, info = ctrl_env.step(low_action)

        # Salva il frame corrente per il "video"
        try:
            frame_pil = obs_to_pil_image(
                env=ctrl_env,
                obs=obs,
                width=frame_width,
                height=frame_height,
            )
            frame_np = np.array(frame_pil, dtype=np.uint8)  # RGB HxWx3
            frames.append(frame_np)
        except Exception as e:
            print(f"[SimServer] Warning: impossibile convertire obs in frame: {e}")

        if done:
            break

    success = bool(info.get("success", False)) if isinstance(info, dict) else False
    log_gpu_status(prefix="[GPU server][dopo azione]")
    return obs, done, success, info, frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_name",
        type=str,
        default="PnPCounterToSink",
        help="Nome dell'environment RoboCasa",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=6000,
        help="Porta TCP locale su cui il server ascolta",
    )
    parser.add_argument(
        "--steps_per_action",
        type=int,
        default=40,
        help="Passi low-level per ogni azione di alto livello (placeholder).",
    )
    parser.add_argument(
        "--render_onscreen",
        action="store_true",
        help="(IGNORATO) La visualizzazione è fatta lato client.",
    )
    parser.add_argument(
        "--frame_width",
        type=int,
        default=512,
        help="Larghezza dei frame generati per VLM e viewer client.",
    )
    parser.add_argument(
        "--frame_height",
        type=int,
        default=512,
        help="Altezza dei frame generati per VLM e viewer client.",
    )
    args = parser.parse_args()

    # === Env di controllo OFFSCREEN (per immagini + logica) ===
    print(f"[SimServer] Creo ctrl_env (offscreen): {args.env_name}")
    ctrl_env = create_env(
        env_name=args.env_name,
        render_onscreen=False,   # <-- SEMPRE offscreen, niente viewer MuJoCo
        seed=0,
    )

    # Reset iniziale del ctrl_env e metadati episodio
    obs = ctrl_env.reset()
    ep_meta = ctrl_env.get_ep_meta()
    lang_instr = ep_meta.get("lang", "")
    print(f"[SimServer] Istruzione del task: {lang_instr}")
    log_gpu_status(prefix="[GPU server][dopo reset]")

    address = ("localhost", args.port)
    listener = Listener(address, authkey=b"robocasa")
    print(f"[SimServer] In ascolto su {address} ...")

    conn = listener.accept()
    print(f"[SimServer] Client connesso da {listener.last_accepted}")
    log_gpu_status(prefix="[GPU server][dopo connessione client]")

    done = False
    success = False

    try:
        while True:
            try:
                msg = conn.recv()  # dizionario dal client
            except EOFError:
                print("[SimServer] Connessione chiusa dal client.")
                break

            cmd = msg.get("cmd")
            frames = []  # default: nessun frame extra

            if cmd == "reset":
                print("[SimServer] Comando RESET")
                obs = ctrl_env.reset()
                ep_meta = ctrl_env.get_ep_meta()
                lang_instr = ep_meta.get("lang", "")
                done = False
                success = False
                log_gpu_status(prefix="[GPU server][reset richiesto]")

            elif cmd == "execute_action":
                action_dict = msg.get("action", {})
                steps = msg.get("steps_per_action", args.steps_per_action)
                obs, done, success, info, frames = execute_high_level_action(
                    ctrl_env,
                    action_dict,
                    steps_per_action=steps,
                    frame_width=args.frame_width,
                    frame_height=args.frame_height,
                )

            elif cmd == "close":
                print("[SimServer] Comando CLOSE")
                conn.send({"ok": True})
                break

            else:
                print(f"[SimServer] Comando sconosciuto: {cmd}")
                conn.send({"ok": False, "error": f"Unknown cmd {cmd}"})
                continue

            # Converte l'ultima obs in immagine per il planner (frame singolo)
            try:
                pil_image = obs_to_pil_image(
                    env=ctrl_env,
                    obs=obs,
                    width=args.frame_width,
                    height=args.frame_height,
                )
                obs_image = np.array(pil_image, dtype=np.uint8)
            except Exception as e:
                print(f"[SimServer] Errore nel convertire obs in immagine: {e}")
                obs_image = None

            reply = {
                "ok": True,
                "done": bool(done),
                "success": bool(success),
                "obs_image": obs_image,  # ultimo frame (per il VLM)
                "frames": frames,        # tutta la sequenza di frame per il video
                "lang_instr": lang_instr,
            }
            conn.send(reply)

    finally:
        ctrl_env.close()
        conn.close()
        listener.close()
        if _HAS_NVML:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
        print("[SimServer] Terminato.")


if __name__ == "__main__":
    main()
