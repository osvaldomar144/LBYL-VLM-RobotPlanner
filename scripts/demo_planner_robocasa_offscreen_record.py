# scripts/demo_planner_robocasa_offscreen_record.py

import argparse
import os
import json
import numpy as np
import robocasa 
from PIL import Image

from vila_open.vlm_client import VLMClient, VLMConfig
from vila_open.planning_loop import plan_once
from vila_open.robocasa_utils import obs_to_pil_image
from vila_open.schema import Action, Plan
from vila_open.controller import OracleController

def main():
    print("--- LBYL RECORDER: FINAL VERSION ---")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="PnPCounterToSink")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_iters", type=int, default=5)
    parser.add_argument("--episode_path", type=str, default="recorded_episode.npz")
    parser.add_argument("--frame_width", type=int, default=256)
    parser.add_argument("--frame_height", type=int, default=256)
    parser.add_argument("--camera_name", type=str, default="robot0_eye_in_hand")
    parser.add_argument("--log_dir", type=str, default="logs_final_thesis")
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.log_dir, exist_ok=True)
    vlm_img_dir = os.path.join(args.log_dir, "vlm_images")
    plans_dir = os.path.join(args.log_dir, "plans")
    os.makedirs(vlm_img_dir, exist_ok=True)
    os.makedirs(plans_dir, exist_ok=True)

    # === CREAZIONE AMBIENTE ===
    print(f"[Recorder] Init env: {args.env_name}")
    env = robocasa.make(
        env_name=args.env_name,
        robots="PandaMobile", # CRUCIALE: Robot mobile
        seed=args.seed,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera=args.camera_name,
        use_camera_obs=True,
        camera_names=[args.camera_name],
        camera_heights=args.frame_height,
        camera_widths=args.frame_width,
    )

    controller = OracleController(env)
    obs = env.reset()

    # Goal
    try:
        ep_meta = env.get_ep_meta()
        lang_instr = ep_meta.get("lang", "Complete the task.")
    except Exception:
        lang_instr = "Complete the task."
    print(f"[Recorder] Goal: \"{lang_instr}\"")

    print("[Recorder] Loading VLM...")
    vlm_config = VLMConfig(device="cuda", max_new_tokens=512, verbose=False)
    vlm_client = VLMClient(vlm_config)

    history = []
    available_primitives = ["pick", "place", "navigate"]
    low_actions_list = [] 

    try:
        for it in range(1, args.max_iters + 1):
            print(f"\n--- Iterazione {it}/{args.max_iters} ---")

            # === 0. GUARDA LA SCENA (STEP AGGIUNTO) ===
            # Eseguiamo un movimento fisico per alzare la camera e inquadrare il tavolo
            # Altrimenti il VLM vede solo il pavimento/lavandino
            print("[Recorder] Posizionamento per scatto foto...")
            for action in controller.look_at_scene():
                low_actions_list.append(action.astype(np.float32))
                obs, _, _, _ = env.step(action)
            
            # 1. Vision (Ora dovrebbe vedere il peperone!)
            pil_image = obs_to_pil_image(obs, camera_name=args.camera_name)
            img_save_path = os.path.join(vlm_img_dir, f"step_{it:02d}_input.png")
            pil_image.save(img_save_path)
            
            # 2. Plan
            print("[Planner] Thinking...")
            plan: Plan = plan_once(
                image=pil_image,
                goal_instruction=lang_instr,
                history=history,
                available_primitives=available_primitives,
                vlm_client=vlm_client,
            )

            with open(os.path.join(plans_dir, f"step_{it:02d}_plan.json"), "w") as f:
                f.write(plan.to_json())

            if not plan.plan:
                print("[Planner] No plan generated. Stopping.")
                break

            # 3. Execute
            next_action: Action = plan.plan[0]
            
            # ESECUZIONE ORACLE
            executed_steps = 0
            action_generator = controller.execute_primitive(
                next_action.primitive, 
                next_action.object
            )
            
            for low_level_action in action_generator:
                low_actions_list.append(low_level_action.astype(np.float32))
                obs, reward, step_done, step_info = env.step(low_level_action)
                executed_steps += 1
                if step_done: break
            
            print(f"[Env] Azione '{next_action.primitive}' completata in {executed_steps} step.")

            # 4. History
            success = False
            if isinstance(step_info, dict):
                success = step_info.get("success", False) or step_info.get("task_success", False)
            
            result_str = "success" if success else ("done" if step_done else "executed")
            history.append({
                "step_id": it,
                "primitive": next_action.primitive,
                "object": next_action.object,
                "result": result_str
            })

            if step_done or success:
                print(f"[Env] Episodio terminato. Successo: {success}")
                break

    except Exception as e:
        print(f"[Error] {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        env.close()

    if low_actions_list:
        actions_arr = np.stack(low_actions_list)
        np.savez(args.episode_path, actions=actions_arr, env=args.env_name, seed=args.seed)
        print(f"[Recorder] Saved to {args.episode_path}")

if __name__ == "__main__":
    main()