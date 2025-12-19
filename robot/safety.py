# robot/safety.py
from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional
from robosuite.utils.transform_utils import quat2mat

class PandaSafeDriver:
    def __init__(self, env, action_dim: int = 12):
        self.env = env
        self.sim = env.sim
        self.action_dim = action_dim

        self.idx_arm = slice(0, 6)
        self.idx_gripper = slice(6, 7)
        self.idx_base = slice(7, 10)
        self.idx_torso = slice(10, 11)

        try:
            self.wrist_body_id = self.env.sim.model.body_name2id("robot0_link7")
        except Exception:
            self.wrist_body_id = None

        self.arm_joint_ids: List[int] = []
        self.arm_joint_names: List[str] = []
        for i in range(self.env.sim.model.njnt):
            name = self.env.sim.model.joint_id2name(i)
            if name and ("robot0_joint" in name) and ("finger" not in name) and ("wheel" not in name):
                self.arm_joint_ids.append(i)
                self.arm_joint_names.append(name)

    def get_real_tip_pose(self) -> np.ndarray:
        if self.wrist_body_id is None:
            return np.zeros(3, dtype=np.float32)
        self.env.sim.forward()
        wrist_pos = self.env.sim.data.body_xpos[self.wrist_body_id]
        wrist_mat = quat2mat(self.env.sim.data.body_xquat[self.wrist_body_id])
        tip = wrist_pos + wrist_mat.dot(np.array([0.0, 0.0, 0.13]))
        return tip.astype(np.float32)

    def get_safety_report(self) -> Tuple[bool, List[str]]:
        danger_joints: List[str] = []
        is_critical = False

        for j_id, name in zip(self.arm_joint_ids, self.arm_joint_names):
            addr = self.env.sim.model.jnt_qposadr[j_id]
            val = self.env.sim.data.qpos[addr]
            min_v, max_v = self.env.sim.model.jnt_range[j_id]
            rng = max_v - min_v
            if rng <= 0:
                continue
            pct = (val - min_v) / rng * 100.0
            if pct < 5 or pct > 95:
                danger_joints.append(f"{name[-2:]}({int(pct)}%)")
                is_critical = True

        return is_critical, danger_joints

    def build_safe_action(
        self,
        base_vel, torso_vel, arm_delta, gripper_cmd,
        clamp_scale: float = 0.1,
        clamp_threshold_norm: float = 0.01,
    ) -> Tuple[np.ndarray, bool]:
        is_critical, dangers = self.get_safety_report()

        if is_critical and np.linalg.norm(arm_delta) > clamp_threshold_norm:
            print(f"\r\033[93m[SAFETY CLAMP] Limiti vicini: {dangers}\033[0m   ", end="", flush=True)
            arm_delta = [x * clamp_scale for x in arm_delta]

        action = np.zeros(self.action_dim, dtype=np.float32)
        action[self.idx_arm] = np.clip(arm_delta, -1, 1)
        action[self.idx_gripper] = float(gripper_cmd)
        action[self.idx_base] = np.clip(base_vel, -1, 1)
        action[self.idx_torso] = np.clip(torso_vel, -1, 1)
        return action, is_critical
