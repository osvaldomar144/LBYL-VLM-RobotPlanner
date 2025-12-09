# scripts/demo_planner_robocasa_client.py

import argparse
from multiprocessing.connection import Client
import subprocess

import numpy as np
from PIL import Image

from vila_open.vlm_client import VLMClient, VLMConfig
from vila_open.planning_loop import plan_once
from vila_open.schema import Action, Plan

# ============================================
# Monitor GPU lato client
# ============================================

_HAS_NVML = False
try:
    import pynvml

    pynvml.nvmlInit()
    _HAS_NVML = True
    print("[GPU client] NVML inizializzato.")
except Exception as e:
    print(f"[GPU client] NVML non disponibile ({e}). Proverò con nvidia-smi se presente.")


def _get_gpu_status():
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


def log_gpu_status(prefix: str = "[GPU client]"):
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
# OpenCV per visualizzazione
# ============================================

try:
    import cv2

    _HAS_CV2 = True
except ImportError:
    cv2 = None
    _HAS_CV2 = False
    print("[PlannerClient] OpenCV non disponibile, nessuna finestra video.")


def np_image_to_pil(arr: np.ndarray) -> Image.Image:
    if arr is None:
        raise RuntimeError("obs_image è None (server non ha mandato un frame valido).")
    return Image.fromarray(arr.astype("uint8"), mode="RGB")


def show_frame_array(
    arr: np.ndarray,
    window_name: str = "RoboCasa Client View",
    flip_vertical: bool = True,
    display_size: int | None = 768,
    delay_ms: int = 33,
):
    """
    Mostra un frame numpy RGB usando OpenCV.
    - flip_vertical: se True gira l'immagine (per correggere il "sotto-sopra")
    - display_size : dimensione massima (in pixel) del lato dell'immagine;
                     il rapporto d'aspetto viene preservato.
                     Se None, mostra a risoluzione nativa.
    - delay_ms     : tempo di attesa in millisecondi (~ 33ms ≈ 30 FPS)
    """
    if not _HAS_CV2:
        return

    frame = arr.astype("uint8")

    if flip_vertical:
        frame = np.flipud(frame)  # ribalta verticalmente

    h, w = frame.shape[:2]

    if display_size is not None and display_size > 0:
        # Scala mantenendo il rapporto d'aspetto
        scale = min(display_size / h, display_size / w)
        if scale != 1.0:
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            # Cubic per upscaling (più nitido), Area per downscaling
            if scale > 1.0:
                interp = cv2.INTER_CUBIC
            else:
                interp = cv2.INTER_AREA
            frame = cv2.resize(frame, (new_w, new_h), interpolation=interp)

    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, frame_bgr)
    cv2.waitKey(delay_ms)


def show_pil_frame(
    pil_img: Image.Image,
    window_name: str,
    flip_vertical: bool,
    display_size: int | None,
    delay_ms: int,
):
    frame = np.array(pil_img, copy=True)
    show_frame_array(
        frame,
        window_name=window_name,
        flip_vertical=flip_vertical,
        display_size=display_size,
        delay_ms=delay_ms,
    )


# ============================================
# Main client
# ============================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port",
        type=int,
        default=6000,
        help="Porta su cui è in ascolto robocasa_sim_server.py",
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=5,
        help="Numero massimo di iterazioni planner-esecuzione.",
    )
    parser.add_argument(
        "--steps_per_action",
        type=int,
        default=40,
        help="Passi low-level per ogni azione di alto livello (placeholder).",
    )
    parser.add_argument(
        "--show_view",
        action="store_true",
        help="Se attivo, mostra una finestra video con OpenCV (lato client).",
    )
    parser.add_argument(
        "--display_fps",
        type=float,
        default=20.0,
        help="FPS del video mostrato lato client (es. 20.0, 30.0).",
    )
    parser.add_argument(
        "--display_size",
        type=int,
        default=768,
        help="Dimensione massima (in pixel) del lato della finestra di visualizzazione.",
    )
    parser.add_argument(
        "--no_flip",
        action="store_true",
        help="Se specificato, NON ribalta verticalmente l'immagine.",
    )
    args = parser.parse_args()

    delay_ms = max(1, int(1000.0 / max(args.display_fps, 1.0)))
    flip_vertical = not args.no_flip

    # === Connessione al server (env RoboCasa) ===
    address = ("localhost", args.port)
    print(f"[PlannerClient] Mi collego al SimServer su {address} ...")
    conn = Client(address, authkey=b"robocasa")
    print("[PlannerClient] Connesso.")
    log_gpu_status(prefix="[GPU client][dopo connessione]")

    # RESET iniziale → otteniamo primo frame + istruzione
    conn.send({"cmd": "reset"})
    reply = conn.recv()
    obs_image = reply["obs_image"]
    pil_image = np_image_to_pil(obs_image)
    goal_instruction = reply.get("lang_instr") or "Complete the task in this environment."
    print(f"[PlannerClient] Istruzione del task (dal server): {goal_instruction}")

    # Salva il primo frame per debug
    pil_image.save("debug_frame_it1.png")
    print("[PlannerClient] Salvato debug_frame_it1.png")

    # Mostra il primo frame (se richiesto)
    if args.show_view:
        show_pil_frame(
            pil_image,
            window_name="RoboCasa Client View",
            flip_vertical=flip_vertical,
            display_size=args.display_size,
            delay_ms=delay_ms,
        )

    # === Config VLM ===
    print("[PlannerClient] Carico il VLM...")
    log_gpu_status(prefix="[GPU client][prima del VLM]")
    vlm_config = VLMConfig(
        device="cuda",
        device_map="auto",
        max_new_tokens=512,
        verbose=True,
    )
    vlm_client = VLMClient(vlm_config)
    log_gpu_status(prefix="[GPU client][dopo caricamento VLM]")

    history = []

    available_primitives = [
        "pick",
        "place",
        "open",
        "close",
        "press_button",
        "turn_knob",
        "navigate",
    ]

    try:
        for it in range(1, args.max_iters + 1):
            print("\n==============================")
            print(f"[PlannerClient] Iterazione di planning {it}")
            print("==============================")
            log_gpu_status(prefix=f"[GPU client][inizio iter {it}]")

            # 1) Chiedi un piano al VLM usando l'ultima immagine
            plan: Plan = plan_once(
                image=pil_image,
                goal_instruction=goal_instruction,
                history=history,
                available_primitives=available_primitives,
                vlm_client=vlm_client,
            )

            print("\n[PlannerClient] Piano proposto:")
            print(plan.to_json())
            log_gpu_status(prefix=f"[GPU client][dopo planning iter {it}]")

            if not plan.plan:
                print("[PlannerClient] Piano vuoto. Mi fermo.")
                break

            # 2) Esegui SOLO la prima azione (Look-Before-You-Leap)
            action: Action = plan.plan[0]
            print(f"\n[PlannerClient] Eseguo solo la prima azione: {action.to_dict()}")

            conn.send(
                {
                    "cmd": "execute_action",
                    "action": action.to_dict(),
                    "steps_per_action": args.steps_per_action,
                }
            )
            reply = conn.recv()
            done = bool(reply.get("done", False))
            success = bool(reply.get("success", False))

            # Sequenza dei frame low-level (video)
            frames = reply.get("frames") or []

            # Nuova immagine "finale" dopo l'azione
            if frames:
                last_frame = frames[-1]
                pil_image = np_image_to_pil(last_frame)
            else:
                pil_image = np_image_to_pil(reply["obs_image"])

            # Mostra la sequenza di frame come video (se richiesto)
            if args.show_view and _HAS_CV2:
                if frames:
                    for f in frames:
                        show_frame_array(
                            f,
                            window_name="RoboCasa Client View",
                            flip_vertical=flip_vertical,
                            display_size=args.display_size,
                            delay_ms=delay_ms,
                        )
                else:
                    show_pil_frame(
                        pil_image,
                        window_name="RoboCasa Client View",
                        flip_vertical=flip_vertical,
                        display_size=args.display_size,
                        delay_ms=delay_ms,
                    )

            # 3) Aggiorna history per il prossimo giro di planning
            result_str = "success" if success else ("done" if done else "in_progress")
            history.append(
                {
                    "step_id": action.step_id,
                    "primitive": action.primitive,
                    "object": action.object,
                    "result": result_str,
                }
            )
            print(f"[PlannerClient] Risultato azione: {result_str}")
            log_gpu_status(prefix=f"[GPU client][fine iter {it}]")

            if done:
                print("[PlannerClient] Episodio terminato (done=True). Esco dal loop.")
                break

    finally:
        # Chiudi in modo pulito
        try:
            conn.send({"cmd": "close"})
            _ = conn.recv()
        except Exception:
            pass
        conn.close()
        if _HAS_CV2:
            cv2.destroyAllWindows()
        if _HAS_NVML:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
        print("\n[PlannerClient] Fine demo planner + RoboCasa (client).")


if __name__ == "__main__":
    main()
