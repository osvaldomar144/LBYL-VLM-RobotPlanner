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
import robocasa


class SmallConvEncoder(nn.Module):
    def __init__(self, in_ch: int = 6, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 5, stride=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 5, stride=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 5, stride=2, padding=2), nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.head(self.net(x))


class VLA6DGripState2F(nn.Module):
    def __init__(self, state_dim: int = 9):
        super().__init__()
        self.enc = SmallConvEncoder(in_ch=6, out_dim=256)
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 64), nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(
            nn.Linear(256 + 256 + 64, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 256), nn.ReLU(inplace=True),
        )
        self.arm_head = nn.Linear(256, 6)
        self.grip_head = nn.Linear(256, 1)

    def forward(self, g, h, s):
        fg = self.enc(g)
        fh = self.enc(h)
        fs = self.state_mlp(s)
        z = self.trunk(torch.cat([fg, fh, fs], dim=1))
        arm = torch.tanh(self.arm_head(z))
        grip_logit = self.grip_head(z).squeeze(1)
        return arm, grip_logit


def transform_image(img: np.ndarray, flip_ud: bool, flip_lr: bool) -> np.ndarray:
    out = img
    if flip_ud:
        out = np.flipud(out)
    if flip_lr:
        out = np.fliplr(out)
    return out.copy()


def img6_to_tensor(img6_uint8: np.ndarray, device: str) -> torch.Tensor:
    t = torch.from_numpy(img6_uint8).float() / 255.0
    t = t.permute(2, 0, 1).unsqueeze(0).contiguous()
    return t.to(device)


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


def build_joint_monitor(env):
    joint_ids = []
    for i in range(env.sim.model.njnt):
        name = env.sim.model.joint_id2name(i)
        if name and ("robot0_joint" in name) and ("finger" not in name) and ("wheel" not in name):
            joint_ids.append(i)
    return joint_ids


def near_joint_limits(env, joint_ids, pct_low=0.03, pct_high=0.97):
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


def gripper_width_from_obs(obs) -> float:
    """
    Misura robusta della "apertura" gripper.
    In robosuite/robocasa tipicamente robot0_gripper_qpos è un vettore di 2 valori.
    """
    gq = obs.get("robot0_gripper_qpos", None)
    if gq is None:
        return 0.0
    gq = np.asarray(gq, dtype=np.float32).reshape(-1)
    return float(np.sum(np.abs(gq)))


def step_with_action(env, a12, render=False, n=1):
    obs = None
    for _ in range(int(n)):
        obs, _, _, _ = env.step(a12)
        if render:
            env.render()
    return obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_name", type=str, default="PnPCounterToCab")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--show_cv", action="store_true")

    # Python 3.10: permette --flip-ud / --no-flip-ud
    ap.add_argument("--flip_ud", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--flip_lr", action=argparse.BooleanOptionalAction, default=False)

    # stabilità
    ap.add_argument("--arm_scale", type=float, default=0.20)
    ap.add_argument("--alpha", type=float, default=0.80)
    ap.add_argument("--slew", type=float, default=0.05)
    ap.add_argument("--warmup_steps", type=int, default=40)
    ap.add_argument("--zero_rot", action=argparse.BooleanOptionalAction, default=True)

    # bootstrap / reset
    ap.add_argument("--reset_tries", type=int, default=12,
                    help="ripeti reset finché non parti NON vicino ai limiti")
    ap.add_argument("--settle_steps", type=int, default=2,
                    help="passi iniziali a zero per costruire prev/curr frame")
    ap.add_argument("--init_freeze_steps", type=int, default=8,
                    help="per i primi N step forza arm=0 (evita jerk/backward)")

    # soft safety SOLO scaling
    ap.add_argument("--soft_safety", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--safety_scale", type=float, default=0.55)
    ap.add_argument("--near_low", type=float, default=0.03)
    ap.add_argument("--near_high", type=float, default=0.97)

    # gripper gating
    ap.add_argument("--grip_min_open_steps", type=int, default=35)
    ap.add_argument("--close_when_arm_small", type=float, default=0.035)
    ap.add_argument("--grip_close_prob", type=float, default=0.80)
    ap.add_argument("--grip_open_prob", type=float, default=0.20)
    ap.add_argument("--grip_hold_steps", type=int, default=25)

    # bias per evitare “scende + va indietro”
    ap.add_argument("--no_back_on_descent", action=argparse.BooleanOptionalAction, default=True)

    # precision mode quando p_close indica pre-grasp
    ap.add_argument("--precision_pclose", type=float, default=0.60,
                    help="quando p_close supera questa soglia (pre-grasp), riduci i passi del braccio")
    ap.add_argument("--precision_scale", type=float, default=0.08,
                    help="arm_scale effettivo target in precision mode (pre-grasp)")
    ap.add_argument("--precision_slew", type=float, default=0.02,
                    help="slew max in precision mode")

    # stop-on-grasp (skill corto) + calibrazione per evitare falsi positivi
    ap.add_argument("--stop_on_grasp", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--calib_grip", action=argparse.BooleanOptionalAction, default=True,
                    help="calibra la width di chiusura a vuoto (senza oggetto)")
    ap.add_argument("--calib_close_steps", type=int, default=12)
    ap.add_argument("--calib_open_steps", type=int, default=8)
    ap.add_argument("--grasp_margin", type=float, default=0.004,
                    help="serve width > closed_empty + margin per dire 'oggetto in mezzo alle dita'")
    ap.add_argument("--grasp_stable_steps", type=int, default=10,
                    help="quanti step consecutivi per confermare grasp e fermare l'episodio")

    ap.add_argument("--debug_grip", action="store_true")
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

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = VLA6DGripState2F(state_dim=9).to(device)
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
        ignore_done=True,
    )

    joint_ids = build_joint_monitor(env)

    # --- reset “buono” ---
    obs = None
    for k in range(max(1, int(args.reset_tries))):
        obs = env.reset()
        env.sim.forward()
        near, dangers = near_joint_limits(env, joint_ids, pct_low=float(args.near_low), pct_high=float(args.near_high))
        if not near:
            break
        if k == int(args.reset_tries) - 1:
            print(f"[Reset] WARNING: ancora near-limit dopo {args.reset_tries} reset. {dangers}")
        else:
            print(f"[Reset] near-limit at start -> reset again. {dangers}")

    # action helper
    a_open = np.zeros(12, dtype=np.float32)
    a_open[6] = -1.0
    a_open[11] = -1.0

    a_close = np.zeros(12, dtype=np.float32)
    a_close[6] = 1.0
    a_close[11] = -1.0

    # --- NEW: calibrazione width closed "a vuoto" per evitare falsi grasp ---
    closed_empty = None
    if args.calib_grip:
        obs = step_with_action(env, a_close, render=args.render, n=int(args.calib_close_steps))
        w_close = gripper_width_from_obs(obs)
        obs = step_with_action(env, a_open, render=args.render, n=int(args.calib_open_steps))
        w_open = gripper_width_from_obs(obs)
        closed_empty = float(w_close)
        if args.debug_grip:
            print(f"[Calib] closed_empty={closed_empty:.4f}  open_width={w_open:.4f}  margin={float(args.grasp_margin):.4f}")
    else:
        closed_empty = 0.0
        if args.debug_grip:
            print("[Calib] disabled -> closed_empty=0.0 (attenzione: stop_on_grasp può fare falsi positivi)")

    # --- bootstrap prev/curr frame ---
    obs_prev = obs
    for _ in range(int(args.settle_steps)):
        obs, _, _, _ = env.step(a_open)
        if args.render:
            env.render()

    g_prev_raw = obs_prev["robot0_agentview_right_image"]
    h_prev_raw = obs_prev["robot0_eye_in_hand_image"]

    last_arm = np.zeros(6, dtype=np.float32)
    grip_state = -1.0
    grip_hold = 0

    grasp_stable = 0

    for t in range(args.steps):
        g_raw = obs.get("robot0_agentview_right_image")
        h_raw = obs.get("robot0_eye_in_hand_image")

        g_prev = transform_image(g_prev_raw, args.flip_ud, args.flip_lr)
        h_prev = transform_image(h_prev_raw, args.flip_ud, args.flip_lr)
        g = transform_image(g_raw, args.flip_ud, args.flip_lr)
        h = transform_image(h_raw, args.flip_ud, args.flip_lr)

        g6 = np.concatenate([g_prev, g], axis=2).astype(np.uint8)
        h6 = np.concatenate([h_prev, h], axis=2).astype(np.uint8)

        joint = obs["robot0_joint_pos"].astype(np.float32) / np.pi
        joint = np.clip(joint, -1.0, 1.0)
        gripq = obs["robot0_gripper_qpos"].astype(np.float32)
        state = np.concatenate([joint, gripq], axis=0).astype(np.float32)
        s = torch.from_numpy(state).unsqueeze(0).to(device)

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device == "cuda")):
            tg = img6_to_tensor(g6, device)
            th = img6_to_tensor(h6, device)
            arm_pred, grip_logit = model(tg, th, s)
            arm = arm_pred.squeeze(0).float().cpu().numpy()
            p_close = float(torch.sigmoid(grip_logit).item())

        # warmup + scale
        ramp = 1.0
        if args.warmup_steps > 0:
            ramp = min(1.0, (t + 1) / float(args.warmup_steps))
        arm_scale_eff = float(args.arm_scale) * ramp

        env.sim.forward()
        near, dangers = near_joint_limits(env, joint_ids, pct_low=float(args.near_low), pct_high=float(args.near_high))
        if args.soft_safety and near:
            arm_scale_eff *= float(args.safety_scale)

        arm *= arm_scale_eff
        if args.zero_rot:
            arm[3:6] = 0.0

        # freeze iniziale
        if t < int(args.init_freeze_steps):
            arm[:] = 0.0

        # NO BACK ON DESCENT
        if args.no_back_on_descent and (grip_state < 0) and (arm[2] < 0.0):
            arm[0] = max(arm[0], 0.0)

        # PRECISION MODE vicino al grasp (quando p_close sale)
        precision_mode = (grip_state < 0) and (p_close >= float(args.precision_pclose))
        if precision_mode:
            target_scale = float(args.precision_scale)
            if arm_scale_eff > 1e-9:
                ratio = min(1.0, target_scale / arm_scale_eff)
                arm *= ratio

        # slew + smoothing
        slew_eff = float(args.precision_slew) if precision_mode else float(args.slew)
        arm = apply_slew_limit(arm, last_arm, max_delta=slew_eff)
        alpha = float(np.clip(args.alpha, 0.0, 0.999))
        arm = (1.0 - alpha) * arm + alpha * last_arm
        last_arm = arm

        # gripper gating
        arm_xyz_norm = float(np.linalg.norm(arm[:3]))
        if t < int(args.grip_min_open_steps):
            grip_state = -1.0
            grip_hold = 0
        else:
            if grip_hold > 0:
                grip_hold -= 1
            else:
                if (p_close >= float(args.grip_close_prob)) and (arm_xyz_norm <= float(args.close_when_arm_small)):
                    grip_state = 1.0
                    grip_hold = int(args.grip_hold_steps)
                elif p_close <= float(args.grip_open_prob):
                    grip_state = -1.0

        # debug
        if args.debug_grip and (t % 25 == 0):
            extra = f"  [NEAR_LIMIT {dangers[:2]}]" if near else ""
            pm = "PREC" if precision_mode else "----"
            print(f"[Grip] t={t} p_close={p_close:.3f} arm||={arm_xyz_norm:.4f} mode={pm} "
                  f"state={'CLOSE' if grip_state>0 else 'OPEN'} hold={grip_hold}{extra}")

        a12 = np.zeros(12, dtype=np.float32)
        a12[:6] = arm
        a12[6] = grip_state
        a12[7:11] = 0.0
        a12[11] = -1.0

        obs_next, _, _, _ = env.step(a12)
        if args.render:
            env.render()

        # STOP ON GRASP (robusto: usa baseline closed_empty)
        if args.stop_on_grasp and (grip_state > 0):
            width = gripper_width_from_obs(obs_next)
            # oggetto tra le dita => non riesce a chiudere fino al "vuoto"
            grasp_like = (width > float(closed_empty) + float(args.grasp_margin))

            if grasp_like:
                grasp_stable += 1
            else:
                grasp_stable = 0

            if args.debug_grip and (t % 25 == 0):
                print(f"[GraspDebug] width={width:.4f} closed_empty={float(closed_empty):.4f} "
                      f"margin={float(args.grasp_margin):.4f} stable={grasp_stable}/{int(args.grasp_stable_steps)}")

            if grasp_stable >= int(args.grasp_stable_steps):
                print(f"[Grasp] DETECTED (width={width:.4f} > {float(closed_empty)+float(args.grasp_margin):.4f}) -> stopping at t={t}")
                break
        else:
            grasp_stable = 0

        if args.show_cv:
            dual = compose_dual(g, h)
            cv2.imshow("VLA Rollout (2F+State) - bootstrap", cv2.cvtColor(dual, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        # shift frames
        g_prev_raw = g_raw
        h_prev_raw = h_raw
        obs = obs_next

    if args.show_cv:
        cv2.destroyAllWindows()
    env.close()


if __name__ == "__main__":
    main()
