# scripts/04_rollout_bc_7d.py
import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import robosuite
import robocasa  # registra env RoboCasa


# -------------------------
# Model
# -------------------------
class SmallConvEncoder(nn.Module):
    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.ReLU(inplace=True),   # 128->64
            nn.Conv2d(32, 64, 5, stride=2, padding=2), nn.ReLU(inplace=True),  # 64->32
            nn.Conv2d(64, 128, 5, stride=2, padding=2), nn.ReLU(inplace=True), # 32->16
            nn.Conv2d(128, 256, 5, stride=2, padding=2), nn.ReLU(inplace=True) # 16->8
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.head(self.net(x))

class DualViewPolicy7D(nn.Module):
    def __init__(self, out_dim: int = 7):
        super().__init__()
        self.enc = SmallConvEncoder(out_dim=256)
        self.mlp = nn.Sequential(
            nn.Linear(256 + 256, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Linear(256, out_dim),
        )

    def forward(self, img_global, img_hand):
        fg = self.enc(img_global)
        fh = self.enc(img_hand)
        x = torch.cat([fg, fh], dim=1)
        a = self.mlp(x)
        return torch.tanh(a)


def img_to_tensor(img_uint8: np.ndarray, device: str) -> torch.Tensor:
    x = torch.from_numpy(img_uint8).to(torch.float32) / 255.0
    x = x.permute(2, 0, 1).unsqueeze(0).contiguous()
    return x.to(device)


def compose_dual(img_global, img_hand):
    dual = np.hstack((img_global, img_hand)).copy()
    h, w, _ = dual.shape
    mid = w // 2
    cv2.line(dual, (mid, 0), (mid, h - 1), (255, 255, 255), 2)
    cv2.putText(dual, "GLOBAL (LEFT)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(dual, "HAND (RIGHT)", (mid + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return dual


def apply_slew_limit(x_new: np.ndarray, x_prev: np.ndarray, max_delta: float) -> np.ndarray:
    if max_delta <= 0:
        return x_new
    delta = np.clip(x_new - x_prev, -max_delta, +max_delta)
    return x_prev + delta


def transform_image(img: np.ndarray, flip_ud: bool, flip_lr: bool) -> np.ndarray:
    out = img
    if flip_ud:
        out = np.flipud(out)
    if flip_lr:
        out = np.fliplr(out)
    return out.copy()


def build_joint_monitor(env):
    joint_ids = []
    for i in range(env.sim.model.njnt):
        name = env.sim.model.joint_id2name(i)
        if name and ("robot0_joint" in name) and ("finger" not in name) and ("wheel" not in name):
            joint_ids.append(i)
    return joint_ids


def near_joint_limits(env, joint_ids, pct_low=0.05, pct_high=0.95):
    dangers = []
    is_near = False
    for j_id in joint_ids:
        addr = env.sim.model.jnt_qposadr[j_id]
        q = float(env.sim.data.qpos[addr])
        mn, mx = env.sim.model.jnt_range[j_id]
        rng = float(mx - mn)
        if rng <= 1e-9:
            continue
        pct = (q - mn) / rng
        if pct < pct_low or pct > pct_high:
            dangers.append(f"{env.sim.model.joint_id2name(j_id)}({pct*100:.0f}%)")
            is_near = True
    return is_near, dangers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_name", type=str, default="PnPCounterToCab")
    ap.add_argument("--ckpt", type=str, required=True)

    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--action_repeat", type=int, default=1)

    ap.add_argument("--render", action="store_true")
    ap.add_argument("--show_cv", action="store_true")
    ap.add_argument("--save_video", type=str, default=None)
    ap.add_argument("--ignore_done", action="store_true")

    # Stabilizers (ARM)
    ap.add_argument("--arm_scale", type=float, default=0.20)
    ap.add_argument("--alpha", type=float, default=0.75)
    ap.add_argument("--slew", type=float, default=0.05)
    ap.add_argument("--warmup_steps", type=int, default=30)
    ap.add_argument("--zero_rot", action="store_true", default=True)

    # Image flip
    ap.add_argument("--flip_ud", action="store_true", default=True)
    ap.add_argument("--flip_lr", action="store_true", default=False)

    # Soft safety scaling (arm)
    ap.add_argument("--soft_safety", action="store_true", default=True)
    ap.add_argument("--safety_scale", type=float, default=0.3)

    # Joint-limit override (arm)
    ap.add_argument("--limit_override_after", type=int, default=6)
    ap.add_argument("--limit_override_steps", type=int, default=12)
    ap.add_argument("--retract_z", type=float, default=0.08)
    ap.add_argument("--override_alpha", type=float, default=0.25)
    ap.add_argument("--override_slew", type=float, default=0.10)

    # NEW: Gripper state machine (NO smoothing on gripper)
    ap.add_argument("--grip_close_th", type=float, default=0.15,
                    help="If raw gripper > this -> close")
    ap.add_argument("--grip_open_th", type=float, default=-0.15,
                    help="If raw gripper < this -> open")
    ap.add_argument("--grip_hold_steps", type=int, default=25,
                    help="After closing, hold closed for N steps (prevents reopen jitter)")
    ap.add_argument("--debug_grip", action="store_true",
                    help="Print raw gripper output sometimes")

    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    camera_names = ["robot0_agentview_right", "robot0_eye_in_hand"]

    print(f"[Rollout] loading ckpt: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = DualViewPolicy7D(out_dim=7).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    env = robosuite.make(
        env_name=args.env_name,
        robots="PandaMobile",
        controller_configs=controller_config,
        has_renderer=args.render,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=camera_names,
        camera_heights=128,
        camera_widths=128,
        ignore_done=bool(args.ignore_done),
    )

    joint_ids = build_joint_monitor(env)
    obs = env.reset()

    writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save_video, fourcc, 20.0, (256, 128))

    # IMPORTANT: manteniamo memoria separata ARM vs GRIPPER
    last_arm = np.zeros(6, dtype=np.float32)
    grip_state = -1.0
    grip_hold = 0

    near_count = 0
    override_left = 0

    try:
        for t in range(args.steps):
            g_raw = obs.get("robot0_agentview_right_image", None)
            h_raw = obs.get("robot0_eye_in_hand_image", None)
            if g_raw is None or h_raw is None:
                print("[Rollout] Missing camera frames in obs keys:", list(obs.keys()))
                break

            g = transform_image(g_raw, args.flip_ud, args.flip_lr)
            h = transform_image(h_raw, args.flip_ud, args.flip_lr)

            with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device == "cuda")):
                tg = img_to_tensor(g, device)
                th = img_to_tensor(h, device)
                a7_raw = model(tg, th).squeeze(0).float().cpu().numpy()  # (7,)

            # --- ARM SCALE ---
            ramp = 1.0
            if args.warmup_steps > 0:
                ramp = min(1.0, (t + 1) / float(args.warmup_steps))
            arm_scale_eff = args.arm_scale * ramp

            # Joint limit logic
            near, dangers = near_joint_limits(env, joint_ids)
            near_count = near_count + 1 if near else 0

            if override_left <= 0 and near_count >= args.limit_override_after:
                override_left = int(args.limit_override_steps)
                print(f"\033[93m[Override] near joint limits too long -> RETRACT for {override_left} steps. {dangers[:3]}\033[0m")

            if args.soft_safety and near and override_left <= 0:
                arm_scale_eff *= float(args.safety_scale)

            # --- Build ARM command from policy ---
            arm = a7_raw[:6].copy()

            # scale
            arm *= arm_scale_eff

            # remove rotations if desired
            if args.zero_rot:
                arm[3:6] = 0.0

            # override retract if needed
            if override_left > 0:
                override_left -= 1
                arm[:] = 0.0
                arm[2] = float(args.retract_z)
                alpha_eff = float(np.clip(args.override_alpha, 0.0, 0.999))
                slew_eff = float(max(args.slew, args.override_slew))
            else:
                alpha_eff = float(np.clip(args.alpha, 0.0, 0.999))
                slew_eff = float(args.slew)

            # ARM: slew + smoothing
            arm = apply_slew_limit(arm, last_arm, max_delta=slew_eff)
            arm = (1.0 - alpha_eff) * arm + alpha_eff * last_arm
            last_arm = arm

            # --- GRIPPER state machine (NO smoothing) ---
            raw_g = float(a7_raw[6])

            if args.debug_grip and (t % 25 == 0):
                print(f"[GripDebug] t={t} raw_g={raw_g:+.3f} state={'CLOSE' if grip_state>0 else 'OPEN'} hold={grip_hold}")

            if grip_hold > 0:
                grip_hold -= 1
            else:
                if raw_g > args.grip_close_th:
                    grip_state = 1.0
                    grip_hold = int(args.grip_hold_steps)
                elif raw_g < args.grip_open_th:
                    grip_state = -1.0

            # --- Build full 12D action ---
            a12 = np.zeros(12, dtype=np.float32)
            a12[:6] = arm
            a12[6] = grip_state          # <- diretto, NO smoothing
            a12[7:11] = 0.0
            a12[11] = -1.0

            done = False
            for _ in range(int(args.action_repeat)):
                obs, reward, done, info = env.step(a12)
                if args.render:
                    env.render()
                if done and (not args.ignore_done):
                    break

            if args.show_cv or writer is not None:
                dual = compose_dual(g, h)
                dual_bgr = cv2.cvtColor(dual, cv2.COLOR_RGB2BGR)
                if args.show_cv:
                    cv2.imshow("BC Rollout (GLOBAL|HAND)", dual_bgr)
                    cv2.waitKey(1)
                if writer is not None:
                    writer.write(dual_bgr)

            if done and (not args.ignore_done):
                print(f"[Rollout] DONE at step {t}")
                break

            if (t + 1) % 25 == 0:
                print(f"[Rollout] step {t+1}/{args.steps}  arm_scale_eff={arm_scale_eff:.3f}  near_count={near_count}  override_left={override_left}")

    finally:
        if writer is not None:
            writer.release()
            print(f"[Rollout] saved video: {args.save_video}")
        if args.show_cv:
            cv2.destroyAllWindows()
        env.close()


if __name__ == "__main__":
    main()
