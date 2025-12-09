# scripts/demo_planner_robocasa.py

import argparse
import numpy as np

from robocasa.utils.env_utils import create_env

from vila_open.vlm_client import VLMClient, VLMConfig
from vila_open.planning_loop import plan_once
from vila_open.robocasa_utils import obs_to_pil_image
from vila_open.schema import Action, Plan

# ==========================
# OpenCV per visualizzazione
# ==========================
try:
    import cv2

    _HAS_CV2 = True
except ImportError:
    cv2 = None
    _HAS_CV2 = False
    print("[demo_planner_robocasa] OpenCV (cv2) non disponibile. "
          "La visualizzazione video sarà disabilitata.")


def execute_high_level_action(
    env,
    action: Action,
    steps_per_action: int = 40,
    show_rgb: bool = False,
    display_scale: int = 4,
    display_fps: int = 30,
    flip_vertical: bool = True,
):
    """
    Esegue un'azione high-level come una sequenza di azioni low-level.

    - env: environment RoboCasa (OFFSCREEN, render_onscreen=False)
    - action: azione di alto livello (Plan → Action)
    - steps_per_action: numero di step low-level per ogni azione high-level
    - show_rgb: se True, mostra un video continuo via OpenCV
    - display_scale: fattore di scala della finestra (es. 4 → 128x→512x)
    - display_fps: FPS target per cv2.imshow (es. 30 → ~33 ms per frame)
    - flip_vertical: se True, ruota l'immagine verticalmente (fix sottosopra)
    """
    print(f"[EXEC] High-level action: {action.to_dict()}")

    obs = None
    done = False
    info = {}

    # delay in ms per avere circa display_fps
    delay_ms = max(1, int(1000 / display_fps))

    for t in range(steps_per_action):
        # Placeholder: azione random (come negli esempi RoboCasa)
        low_action = np.random.randn(*env.action_spec[0].shape) * 0.3
        obs, reward, done, info = env.step(low_action)

        # ==============
        # Visualizzazione
        # ==============
        if show_rgb and _HAS_CV2:
            try:
                # Usa l'osservazione visiva dall'env (obs_to_pil_image va a
                # prendere robot0_agentview_*_image dentro obs)
                frame_pil = obs_to_pil_image(env=env, obs=obs)

                # PIL → numpy RGB
                frame = np.array(frame_pil, copy=True)  # (H, W, 3), uint8

                # Flip verticale se necessario (molti renderer danno l'immagine capovolta)
                if flip_vertical:
                    frame = np.flipud(frame)

                # Upscaling per avere una finestra più grande
                if display_scale != 1:
                    h, w, _ = frame.shape
                    new_w = int(w * display_scale)
                    new_h = int(h * display_scale)
                    frame = cv2.resize(
                        frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR
                    )

                # RGB -> BGR per OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                cv2.imshow("RoboCasa VLM run (OpenCV)", frame_bgr)
                cv2.waitKey(delay_ms)
            except Exception as e:
                print(f"[EXEC] Warning: impossibile mostrare il frame con OpenCV: {e}")

        if done:
            break

    success = bool(info.get("success", False)) if isinstance(info, dict) else False
    return obs, done, success, info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_name",
        type=str,
        default="PnPCounterToSink",
        help="Nome dell'environment RoboCasa (es. PnPCounterToSink, HeatMug, ...)",
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
        "--render_onscreen",
        action="store_true",
        help="Se specificato, mostra video continuo con OpenCV.",
    )
    parser.add_argument(
        "--display_scale",
        type=int,
        default=4,
        help="Fattore di scala per la finestra video (es. 4 → immagine 4x più grande).",
    )
    parser.add_argument(
        "--display_fps",
        type=int,
        default=30,
        help="FPS target per il video (es. 30).",
    )
    args = parser.parse_args()

    if args.render_onscreen and not _HAS_CV2:
        print("[demo_planner_robocasa] ATTENZIONE: --render_onscreen richiesto ma "
              "OpenCV non è installato. Esegui: pip install opencv-python")

    # ==========================================================
    #  Env di controllo (UNICO env, SOLO offscreen, niente viewer)
    # ==========================================================
    print(f"[RoboCasa] Creo env di controllo (offscreen): {args.env_name}")

    # Proviamo a chiedere una risoluzione più alta (camera_widths / camera_heights).
    # Se la versione di robocasa non supporta i parametri, facciamo fallback.
    create_env_kwargs = dict(
        env_name=args.env_name,
        render_onscreen=False,  # SEMPRE offscreen: niente env.render(), niente segfault
        seed=0,
    )
    try:
        # ad es. 128 * 2 = 256; se già 256, rimane tale
        create_env_kwargs.update(dict(camera_widths=256, camera_heights=256))
        env = create_env(**create_env_kwargs)
    except TypeError:
        # Versione vecchia di robocasa: niente camera_widths / heights
        print("[RoboCasa] create_env non supporta camera_widths/camera_heights, "
              "uso risoluzione di default.")
        create_env_kwargs.pop("camera_widths", None)
        create_env_kwargs.pop("camera_heights", None)
        env = create_env(**create_env_kwargs)

    obs = env.reset()
    ep_meta = env.get_ep_meta()
    lang_instr = ep_meta.get("lang", "")
    print(f"[RoboCasa] Istruzione del task: {lang_instr}")

    goal_instruction = lang_instr or "Complete the task in this environment."

    # =================
    # Configurazione VLM
    # =================
    vlm_config = VLMConfig(
        device="cuda",
        device_map="auto",
        max_new_tokens=512,
        verbose=True,
    )
    vlm_client = VLMClient(vlm_config)

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

    for it in range(1, args.max_iters + 1):
        print("\n==============================")
        print(f"[LOOP] Iterazione di planning {it}")
        print("==============================")

        # 1) Immagine per il VLM: dall'env offscreen (obs contiene *_image)
        pil_image = obs_to_pil_image(env=env, obs=obs)

        if it == 1:
            pil_image.save("debug_frame_it1.png")
            print("[DEBUG] Salvato debug_frame_it1.png")

        # 2) Chiedi un piano al VLM
        plan: Plan = plan_once(
            image=pil_image,
            goal_instruction=goal_instruction,
            history=history,
            available_primitives=available_primitives,
            vlm_client=vlm_client,
        )

        print("\n[Planner] Piano proposto:")
        print(plan.to_json())

        if not plan.plan:
            print("[Planner] Piano vuoto. Mi fermo.")
            break

        # 3) Esegui SOLO la prima azione (Look-Before-You-Leap)
        action: Action = plan.plan[0]
        print(f"\n[Planner] Eseguo solo la prima azione: {action.to_dict()}")

        # 4) Esegui l'azione nel simulatore + viewer OpenCV continuo
        obs, done, success, info = execute_high_level_action(
            env,
            action,
            steps_per_action=args.steps_per_action,
            show_rgb=args.render_onscreen,
            display_scale=args.display_scale,
            display_fps=args.display_fps,
            flip_vertical=True,  # se vedi di nuovo sottosopra, puoi metterlo a False
        )

        # 5) Aggiorna history
        result_str = "success" if success else ("done" if done else "in_progress")
        history.append(
            {
                "step_id": action.step_id,
                "primitive": action.primitive,
                "object": action.object,
                "result": result_str,
            }
        )

        print(f"[EXEC] Risultato azione: {result_str}")
        if done:
            print("[RoboCasa] Episodio terminato (done=True). Esco dal loop.")
            break

    env.close()

    if _HAS_CV2:
        cv2.destroyAllWindows()

    print("\n[DEMO] Fine demo planner + RoboCasa.")
    

if __name__ == "__main__":
    main()