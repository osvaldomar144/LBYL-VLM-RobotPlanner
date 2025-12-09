# webots_project/controllers/panda_lbyl/panda_lbyl.py

"""
Controller Webots per pipeline Look-Before-You-Leap con Franka Panda.

- Cattura immagini da una Camera Webots
- Chiama il VLM (VLMClient / plan_once) come nel codice RoboCasa
- Esegue SOLO la prima azione high-level del piano con un controller low-level placeholder
- Mostra tutto live nella GUI di Webots
- Salva immagini e piani in webots_project/controllers/panda_lbyl/logs_webots/
"""

import os
import sys
import json
from typing import List, Dict, Optional

import numpy as np
from PIL import Image

# --------------------------------------------------------------------
# Aggiungi il root del repo LBYL-VLM-RobotPlanner al PYTHONPATH
# (assumo che questo file sia in: <repo>/webots_project/controllers/panda_lbyl/panda_lbyl.py)
# --------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
WEBOTS_PROJECT_DIR = os.path.dirname(os.path.dirname(THIS_DIR))   # .../webots_project
REPO_ROOT = os.path.dirname(WEBOTS_PROJECT_DIR)                   # .../LBYL-VLM-RobotPlanner

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Ora gli import del tuo repo dovrebbero funzionare
from vila_open.vlm_client import VLMClient, VLMConfig
from vila_open.planning_loop import plan_once
from vila_open.schema import Action, Plan

# API Webots (modulo fornito da Webots, NON dal tuo conda)
from controller import Robot  # type: ignore


# ============================
# PARAMETRI CONFIGURABILI
# ============================

# Nome del device Camera nel modello Webots (controlla nel pannello Devices)
CAMERA_NAME = "camera"

# Joints del Panda in Webots
ARM_JOINT_NAMES = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
]

GRIPPER_JOINT_NAMES = [
    "panda_finger_joint1",
    "panda_finger_joint2",
]

ARM_JOINT_LIMIT = 2.8  # rad, clip simmetrico [-limit, limit]

GRIPPER_MIN = 0.0
GRIPPER_MAX = 0.04

MAX_ITERS = 5
STEPS_PER_ACTION = 40
FRAME_WIDTH = 256
FRAME_HEIGHT = 256

GOAL_INSTRUCTION = "pick the bell pepper from the counter and place it in the sink"

LOG_DIR_NAME = "logs_webots"


class PandaLBYLController:
    def __init__(self) -> None:
        # ---- Webots Robot & timestep ----
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())

        # ---- Logging ----
        self._setup_logging()

        # ---- Dispositivi (camera + motori) ----
        self._init_devices()

        # ---- VLM ----
        self._init_vlm()

        # ---- Stato planner LBYL ----
        self.history: List[Dict] = []
        self.available_primitives = [
            "pick",
            "place",
            "open",
            "close",
            "press_button",
            "turn_knob",
            "navigate",
        ]

        print("[LBYL] Controller inizializzato.")

    # ----------------------------
    # Setup logging
    # ----------------------------
    def _setup_logging(self) -> None:
        base_dir = os.getcwd()
        self.log_dir = os.path.join(base_dir, LOG_DIR_NAME)
        self.vlm_img_dir = os.path.join(self.log_dir, "vlm_images")
        self.plans_dir = os.path.join(self.log_dir, "plans")

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.vlm_img_dir, exist_ok=True)
        os.makedirs(self.plans_dir, exist_ok=True)

        print(f"[LBYL] Log directory: {self.log_dir}")

    # ----------------------------
    # Inizializzazione dispositivi Webots
    # ----------------------------
    def _init_devices(self) -> None:
        # --- Camera ---
        cam = self.robot.getDevice(CAMERA_NAME)
        if cam is None:
            raise RuntimeError(
                f"[LBYL] Camera '{CAMERA_NAME}' non trovata. "
                "Controlla il nome del device nel modello Webots."
            )
        self.camera = cam
        self.camera.enable(self.timestep)
        print(f"[LBYL] Camera '{CAMERA_NAME}' abilitata.")

        # --- Motori del braccio ---
        self.arm_motors = []
        self.arm_sensors = []
        self.arm_targets = []

        for name in ARM_JOINT_NAMES:
            m = self.robot.getDevice(name)
            if m is None:
                print(f"[LBYL] WARNING: Motor '{name}' non trovato.")
                continue
            s = m.getPositionSensor()
            if s is not None:
                s.enable(self.timestep)
            m.setVelocity(1.5)

            self.arm_motors.append(m)
            self.arm_sensors.append(s)
            self.arm_targets.append(0.0)

        print(f"[LBYL] Trovati {len(self.arm_motors)} motori braccio.")

        # --- Motori del gripper ---
        self.gripper_motors = []
        self.gripper_sensors = []
        self.gripper_target = (GRIPPER_MIN + GRIPPER_MAX) / 2.0

        for name in GRIPPER_JOINT_NAMES:
            m = self.robot.getDevice(name)
            if m is None:
                print(f"[LBYL] WARNING: Motor gripper '{name}' non trovato.")
                continue
            s = m.getPositionSensor()
            if s is not None:
                s.enable(self.timestep)
            m.setVelocity(0.2)
            self.gripper_motors.append(m)
            self.gripper_sensors.append(s)

        print(f"[LBYL] Trovati {len(self.gripper_motors)} motori gripper.")

        # --- Warm-up: qualche step per inizializzare sensori/camera ---
        print("[LBYL] Warm-up simulation steps per inizializzare sensori...")
        for _ in range(10):
            if self.robot.step(self.timestep) == -1:
                return

        # Leggi posizioni iniziali come target
        for i, s in enumerate(self.arm_sensors):
            if s is not None:
                v = s.getValue()
                self.arm_targets[i] = float(
                    np.clip(v, -ARM_JOINT_LIMIT, ARM_JOINT_LIMIT)
                )
                self.arm_motors[i].setPosition(self.arm_targets[i])

        if self.gripper_sensors:
            for m, s in zip(self.gripper_motors, self.gripper_sensors):
                if s is not None:
                    v = float(np.clip(s.getValue(), GRIPPER_MIN, GRIPPER_MAX))
                    m.setPosition(v)
            self.gripper_target = v

        print("[LBYL] Dispositivi inizializzati.")

    # ----------------------------
    # Inizializzazione VLM
    # ----------------------------
    def _init_vlm(self) -> None:
        print("[LBYL] Inizializzo VLM (LLaVA / OneVision ecc.)...")
        vlm_config = VLMConfig(
            device="cuda",
            device_map="auto",
            max_new_tokens=512,
            verbose=True,
        )
        self.vlm_client = VLMClient(vlm_config)
        print("[LBYL] VLM caricato.")

    # ----------------------------
    # Utility: step simulazione
    # ----------------------------
    def _step_sim(self) -> bool:
        return self.robot.step(self.timestep) != -1

    # ----------------------------
    # Utility: cattura immagine camera come PIL.Image
    # ----------------------------
    def get_camera_pil(self) -> Optional[Image.Image]:
        image_array = self.camera.getImageArray()
        if not image_array:
            print("[LBYL] WARNING: camera.getImageArray() ha restituito lista vuota.")
            return None

        np_img = np.array(image_array, dtype=np.uint8)  # (W, H, 4)
        np_img = np.transpose(np_img, (1, 0, 2))        # -> (H, W, 4)
        rgb = np_img[:, :, :3]

        pil = Image.fromarray(rgb, mode="RGB")
        if pil.size != (FRAME_WIDTH, FRAME_HEIGHT):
            pil = pil.resize((FRAME_WIDTH, FRAME_HEIGHT), Image.BILINEAR)
        return pil

    def _save_pil(self, pil: Image.Image, filename: str) -> str:
        path = os.path.join(self.vlm_img_dir, filename)
        pil.save(path)
        print(f"[LBYL] Salvata immagine: {path}")
        return path

    # ----------------------------
    # Esecuzione azione high-level (placeholder)
    # ----------------------------
    def execute_high_level_action(self, action: Action) -> None:
        primitive = getattr(action, "primitive", "")
        print(f"[LBYL] Eseguo azione high-level placeholder: {action.to_dict()}")

        if primitive in ("pick", "close", "grasp"):
            self.gripper_target = GRIPPER_MIN
        elif primitive in ("place", "open"):
            self.gripper_target = GRIPPER_MAX

        for step in range(STEPS_PER_ACTION):
            for i, motor in enumerate(self.arm_motors):
                delta = float(np.random.randn() * 0.05)
                self.arm_targets[i] = float(
                    np.clip(self.arm_targets[i] + delta, -ARM_JOINT_LIMIT, ARM_JOINT_LIMIT)
                )
                motor.setPosition(self.arm_targets[i])

            for gm in self.gripper_motors:
                gm.setPosition(self.gripper_target)

            if not self._step_sim():
                print("[LBYL] Simulazione terminata durante l'azione.")
                return

        print("[LBYL] Azione high-level placeholder completata.")

    # ----------------------------
    # Loop principale LBYL
    # ----------------------------
    def run(self) -> None:
        init_img = self.get_camera_pil()
        if init_img is not None:
            self._save_pil(init_img, "vlm_input_it_000_init.png")
        else:
            print("[LBYL] WARNING: impossibile salvare il frame iniziale.")

        for it in range(1, MAX_ITERS + 1):
            print("\n==============================")
            print(f"[LBYL] Iterazione di planning {it}")
            print("==============================")

            if not self._step_sim():
                print("[LBYL] Simulazione fermata dall'utente.")
                break

            pil_image = self.get_camera_pil()
            if pil_image is None:
                print("[LBYL] Nessuna immagine dalla camera; interrompo.")
                break

            img_name = f"vlm_input_it_{it:03d}.png"
            self._save_pil(pil_image, img_name)

            print(f"[LBYL] Iter {it}: goal_instruction = {GOAL_INSTRUCTION}")
            print(f"[LBYL] Iter {it}: history_len = {len(self.history)}")

            plan: Plan = plan_once(
                image=pil_image,
                goal_instruction=GOAL_INSTRUCTION,
                history=self.history,
                available_primitives=self.available_primitives,
                vlm_client=self.vlm_client,
            )

            print("\n[LBYL] Piano proposto (JSON):")
            plan_json = plan.to_json()
            print(plan_json)

            plan_file = os.path.join(self.plans_dir, f"plan_it_{it:03d}.json")
            with open(plan_file, "w", encoding="utf-8") as f:
                f.write(plan_json)
            print(f"[LBYL] Iter {it}: piano salvato in {plan_file}")

            if not plan.plan:
                print("[LBYL] Piano vuoto. Mi fermo.")
                break

            action: Action = plan.plan[0]
            self.execute_high_level_action(action)

            obj_or_target = getattr(action, "object", None) or getattr(action, "target", None)
            result_str = "in_progress"

            self.history.append(
                {
                    "step_id": action.step_id,
                    "primitive": action.primitive,
                    "object": obj_or_target,
                    "result": result_str,
                }
            )
            print(f"[LBYL] Iter {it}: risultato azione = {result_str}")

        print("[LBYL] Loop di planning terminato. Controller concluso.")


if __name__ == "__main__":
    controller = PandaLBYLController()
    controller.run()
