# robot/robocasa_env.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import numpy as np
import robosuite

from .safety import PandaSafeDriver

@dataclass
class RobotObs:
    state: Dict[str, Any]
    images: Dict[str, np.ndarray]   # RGB uint8 HxWx3

class RoboCasaPandaMobileEnv:
    def __init__(
        self,
        env_name: str = "PnPCounterToCab",
        render: bool = False,
        offscreen: bool = True,
        camera_names: Optional[List[str]] = None,
        camera_hw: int = 256,
    ):
        controller_config = {
            "type": "OSC_POSE",
            "input_max": 1, "input_min": -1,
            "output_max": [1] * 6, "output_min": [-1] * 6,
            "kp": 150, "damping": 2,
            "impedance_mode": "fixed",
            "kp_limits": [0, 300], "damping_limits": [0, 10],
            "uncouple_pos_ori": True, "control_delta": True,
            "interpolation": None, "ramp_ratio": 0.2
        }

        self.camera_names = camera_names or ["robot0_eye_in_hand", "robot0_agentview_center"]

        self.env = robosuite.make(
            env_name=env_name,
            robots="PandaMobile",
            controller_configs=controller_config,
            has_renderer=render,
            has_offscreen_renderer=offscreen,
            use_camera_obs=True,
            camera_names=self.camera_names,
            camera_heights=camera_hw,
            camera_widths=camera_hw,
            ignore_done=True,
        )

        # action dim: per PandaMobile spesso è 12, ma non hardcodiamo
        self.action_dim = getattr(self.env, "action_dim", 12)
        self.driver = PandaSafeDriver(self.env, action_dim=self.action_dim)

        self.gripper_state: float = -1.0
        self.last_obs = None
        self.render_enabled = render

        # scaling “tuo”
        self.arm_sensitivity = 0.05
        self.base_sensitivity = 1.0
        self.default_repeat = 15

        self.reset()

    def reset(self):
        self.last_obs = self.env.reset()
        self.gripper_state = -1.0
        return self.get_obs()

    def get_obs(self) -> RobotObs:
        tip_pos = self.driver.get_real_tip_pose()
        safety_crit, safety_msg = self.driver.get_safety_report()

        state = {
            "eef_xyz": [float(x) for x in tip_pos],
            "gripper": "closed" if self.gripper_state > 0 else "open",
            "safety_warning": safety_msg if safety_crit else "nominal"
        }

        if self.last_obs is None:
            self.last_obs = self.env.reset()

        images: Dict[str, np.ndarray] = {}
        for cam in self.camera_names:
            key = cam + "_image"
            if key in self.last_obs:
                # IMPORTANT: qui non flippiamo più di default.
                # Se poi scopri che la camera è invertita, lo gestiamo in un punto solo.
                img = self.last_obs[key]
                images[cam] = img

        return RobotObs(state=state, images=images)

    def build_action_from_components(
        self,
        arm_xyz: Optional[List[float]] = None,
        base: Optional[List[float]] = None,
        torso: Optional[float] = None,
        gripper: Optional[float] = None,
    ) -> np.ndarray:
        base_cmd = base or [0.0, 0.0, 0.0]
        arm_cmd6 = [0.0] * 6
        if arm_xyz is not None:
            arm_cmd6[0] = float(arm_xyz[0]) * self.arm_sensitivity
            arm_cmd6[1] = float(arm_xyz[1]) * self.arm_sensitivity
            arm_cmd6[2] = float(arm_xyz[2]) * self.arm_sensitivity

        torso_cmd = float(torso) if torso is not None else 0.0
        if gripper is not None:
            self.gripper_state = 1.0 if float(gripper) > 0 else -1.0

        action, _ = self.driver.build_safe_action(
            base_vel=[float(x) * self.base_sensitivity for x in base_cmd],
            torso_vel=torso_cmd,
            arm_delta=arm_cmd6,
            gripper_cmd=self.gripper_state,
        )
        return action

    def step_action(self, action_vec: np.ndarray, repeat: Optional[int] = None):
        repeat = int(repeat or self.default_repeat)
        info_accum = {"movement_clamped": False}

        for _ in range(repeat):
            # safety clamp già dentro build_safe_action se passi components;
            # qui assumiamo action_vec già pronto [-1,1]
            self.last_obs, reward, done, info = self.env.step(action_vec)
            if self.render_enabled:
                self.env.render()
        return reward, done, info_accum

    def close(self):
        self.env.close()
