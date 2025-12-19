#!/usr/bin/env python3
import argparse
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image, ImageOps, ImageDraw

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

from robocasa.utils.env_utils import create_env

# Optional viewer (frame display)
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    cv2 = None
    _HAS_CV2 = False


# ----------------------------
# Utilities: images
# ----------------------------
def obs_img_to_pil(obs_img: np.ndarray) -> Image.Image:
    """RoboCasa image obs are typically uint8 HxWx3 (RGB)."""
    if obs_img is None:
        raise ValueError("obs_img is None")
    if obs_img.dtype != np.uint8:
        obs_img = np.clip(obs_img, 0, 255).astype(np.uint8)
    if obs_img.ndim != 3 or obs_img.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3 uint8 image, got {obs_img.shape} {obs_img.dtype}")
    return Image.fromarray(obs_img, mode="RGB")


def maybe_flip(img: Image.Image, flip_ud: bool, flip_lr: bool) -> Image.Image:
    if flip_ud:
        img = ImageOps.flip(img)
    if flip_lr:
        img = ImageOps.mirror(img)
    return img


def make_composite(
    global_img: Image.Image,
    hand_img: Image.Image,
    mode: str = "h",  # "h" or "v"
    pad: int = 6,
    label: bool = True,
) -> Image.Image:
    """Create a single RGB image containing both views, e.g. [GLOBAL | HAND]."""
    g = global_img
    h = hand_img

    if mode == "h":
        target_h = min(g.height, h.height)
        g2 = g.resize((int(g.width * target_h / g.height), target_h), Image.BICUBIC)
        h2 = h.resize((int(h.width * target_h / h.height), target_h), Image.BICUBIC)
        out_w = g2.width + pad + h2.width
        out_h = target_h
        out = Image.new("RGB", (out_w, out_h), (0, 0, 0))
        out.paste(g2, (0, 0))
        out.paste(h2, (g2.width + pad, 0))
    elif mode == "v":
        target_w = min(g.width, h.width)
        g2 = g.resize((target_w, int(g.height * target_w / g.width)), Image.BICUBIC)
        h2 = h.resize((target_w, int(h.height * target_w / h.width)), Image.BICUBIC)
        out_w = target_w
        out_h = g2.height + pad + h2.height
        out = Image.new("RGB", (out_w, out_h), (0, 0, 0))
        out.paste(g2, (0, 0))
        out.paste(h2, (0, g2.height + pad))
    else:
        raise ValueError("mode must be 'h' or 'v'")

    if label:
        draw = ImageDraw.Draw(out)
        draw.text((8, 8), "GLOBAL", fill=(255, 255, 255))
        if mode == "h":
            draw.text((out.width // 2 + 8, 8), "HAND", fill=(255, 255, 255))
        else:
            draw.text((8, out.height // 2 + 8), "HAND", fill=(255, 255, 255))
    return out


def list_image_like_keys(obs: dict) -> list:
    keys = []
    for k, v in obs.items():
        if not isinstance(v, np.ndarray):
            continue
        if k.endswith("_image"):
            keys.append(k)
            continue
        if v.ndim == 3 and v.shape[-1] == 3 and v.dtype in (np.uint8, np.uint16, np.float32, np.float64):
            keys.append(k)
    return sorted(keys)


def resolve_image_key(obs: dict, preferred: str) -> str:
    if preferred in obs:
        return preferred

    candidates = []

    if preferred.startswith("robot0_"):
        candidates.append(preferred[len("robot0_"):])
    else:
        candidates.append("robot0_" + preferred)

    if preferred.endswith("_image"):
        candidates.append(preferred[:-6])
    else:
        candidates.append(preferred + "_image")

    for c in list(candidates):
        if c.startswith("robot0_") and c.endswith("_image"):
            candidates.append(c[len("robot0_"):])
            candidates.append(c[:-6])
        if (not c.startswith("robot0_")) and c.endswith("_image"):
            candidates.append("robot0_" + c)
        if c.startswith("robot0_") and (not c.endswith("_image")):
            candidates.append(c + "_image")

    for c in candidates:
        if c in obs:
            return c

    img_keys = list_image_like_keys(obs)
    base = preferred.replace("robot0_", "").replace("_image", "")
    matches = [k for k in img_keys if base in k]

    if len(matches) == 1:
        return matches[0]

    raise KeyError(
        f"Missing obs key '{preferred}'.\n"
        f"Image-like keys in obs: {img_keys}\n"
        f"Tip: if you ran with --render_onscreen and see no *_image keys, "
        f"the env may be running without offscreen rendering; this script will try to fallback to offscreen."
    )


def maybe_show_cv2(window_name: str, pil_img: Image.Image):
    if not _HAS_CV2:
        return
    frame = np.array(pil_img)[:, :, ::-1]  # RGB -> BGR
    cv2.imshow(window_name, frame)


# ----------------------------
# Filters
# ----------------------------
@dataclass
class ActionFilter:
    alpha: float = 0.0
    slew: float = 0.25
    prev: Optional[np.ndarray] = None

    def reset(self):
        self.prev = None

    def __call__(self, arm6: np.ndarray) -> np.ndarray:
        assert arm6.shape == (6,)
        x = arm6.copy()

        if self.alpha > 0.0:
            if self.prev is None:
                self.prev = x
            else:
                x = self.alpha * self.prev + (1.0 - self.alpha) * x

        if self.prev is not None and self.slew > 0.0:
            delta = x - self.prev
            delta = np.clip(delta, -self.slew, self.slew)
            x = self.prev + delta

        self.prev = x
        return x


@dataclass
class GripperHysteresis:
    close_th: float = -0.2
    open_th: float = 0.2
    hold_steps: int = 2
    state: float = 1.0
    hold: int = 0

    def reset(self):
        self.state = 1.0
        self.hold = 0

    def __call__(self, g: float) -> float:
        if self.hold > 0:
            self.hold -= 1
            return self.state

        if self.state > 0:
            if g <= self.close_th:
                self.state = -1.0
                self.hold = self.hold_steps
        else:
            if g >= self.open_th:
                self.state = 1.0
                self.hold = self.hold_steps
        return self.state


def build_prompt(instruction: str, style: str = "openvla") -> str:
    if style == "openvla":
        return f"In: What action should the robot take to {instruction}?\nOut:"
    return f"Instruction: {instruction}\nAction:"


def ensure_openvla_suffix_token(batch: dict, empty_token_id: int = 29871) -> dict:
    
    if "input_ids" not in batch:
        return batch
    input_ids = batch["input_ids"]
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        return batch

    if not torch.all(input_ids[:, -1] == empty_token_id):
        bsz = input_ids.shape[0]
        extra = torch.full((bsz, 1), empty_token_id, dtype=input_ids.dtype, device=input_ids.device)
        batch["input_ids"] = torch.cat([input_ids, extra], dim=1)

        if "attention_mask" in batch and isinstance(batch["attention_mask"], torch.Tensor):
            am = batch["attention_mask"]
            ones = torch.ones((bsz, 1), dtype=am.dtype, device=am.device)
            batch["attention_mask"] = torch.cat([am, ones], dim=1)
        else:
            batch["attention_mask"] = torch.ones((bsz, batch["input_ids"].shape[1]), dtype=torch.long, device=input_ids.device)

    return batch


def build_env_action(action_dim: int, arm6: np.ndarray, grip_cmd: float, clip: float) -> np.ndarray:
    """
    Maps to the *actual* env action space length.
    Common cases:
      7  = [arm6, gripper]
      11 = [arm6, gripper, base3, torso1]
      12 = [arm6, gripper, base3, torso1, dummy]  (some datasets keep a trailing -1)
    """
    a = np.zeros((action_dim,), dtype=np.float32)

    # arm6 always first if present
    n_arm = min(6, action_dim)
    a[:n_arm] = arm6[:n_arm]

    # gripper at index 6 if present
    if action_dim >= 7:
        a[6] = float(np.clip(grip_cmd, -clip, clip))

    # If action_dim==12, many logs/datasets expect last dim dummy = -1
    if action_dim == 12:
        a[11] = -1.0

    return a


def main():
    ap = argparse.ArgumentParser()

    # Env
    ap.add_argument("--env_name", type=str, default="PnPCounterToSink")
    ap.add_argument("--robot", type=str, default="PandaMobile")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--render_onscreen", action="store_true",
                    help="Try to open the sim viewer. If camera obs disappear, script auto-falls back to offscreen and uses cv2 display if available.")
    ap.add_argument("--camera_w", type=int, default=256)
    ap.add_argument("--camera_h", type=int, default=256)

    # Keys
    ap.add_argument("--global_key", type=str, default="robot0_agentview_right_image")
    ap.add_argument("--hand_key", type=str, default="robot0_eye_in_hand_image")

    # OpenVLA
    ap.add_argument("--model_id", type=str, default="openvla/openvla-7b")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--attn_impl", type=str, default="flash_attention_2")
    ap.add_argument("--unnorm_key", type=str, default="bridge_orig")
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--prompt_style", type=str, default="openvla", choices=["openvla", "generic"])

    # Robust generation
    ap.add_argument("--no_cache", action="store_true", help="Disable KV-cache in generation (slower but robust).")

    # OpenVLA suffix token fix
    ap.add_argument("--openvla_empty_token_id", type=int, default=29871)
    ap.add_argument("--disable_openvla_suffix_fix", action="store_true")

    # Instruction
    ap.add_argument("--instruction", type=str, default="pick the object and place it in the sink")

    # Multi-view handling
    ap.add_argument("--use_composite", action="store_true")
    ap.add_argument("--composite_mode", type=str, default="h", choices=["h", "v"])
    ap.add_argument("--composite_label", action="store_true")

    # Image flips
    ap.add_argument("--flip_ud", action="store_true")
    ap.add_argument("--flip_lr", action="store_true")

    # Axis inversions
    ap.add_argument("--inv_x", action="store_true")
    ap.add_argument("--inv_y", action="store_true")
    ap.add_argument("--inv_z", action="store_true")
    ap.add_argument("--inv_roll", action="store_true")
    ap.add_argument("--inv_pitch", action="store_true")
    ap.add_argument("--inv_yaw", action="store_true")

    # Scaling + safety
    ap.add_argument("--pos_step", type=float, default=50.0)
    ap.add_argument("--rot_step", type=float, default=10.0)
    ap.add_argument("--arm_scale", type=float, default=1.0)
    ap.add_argument("--clip", type=float, default=1.0)

    # Filters
    ap.add_argument("--smooth_alpha", type=float, default=0.2)
    ap.add_argument("--slew", type=float, default=0.25)

    # Gripper
    ap.add_argument("--gripper_0_1", action="store_true")
    ap.add_argument("--grip_close_th", type=float, default=-0.2)
    ap.add_argument("--grip_open_th", type=float, default=0.2)
    ap.add_argument("--grip_hold", type=int, default=2)

    # Rollout
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--sleep", type=float, default=0.0)

    args = ap.parse_args()

    # dtype
    if args.dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "float16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    use_cuda = torch.cuda.is_available() and ("cuda" in args.device)
    device = torch.device(args.device if use_cuda else "cpu")

    # Load model / processor
    print(f"[OpenVLA] Loading processor: {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)

    print(f"[OpenVLA] Loading model: {args.model_id} (attn_impl={args.attn_impl})")
    try:
        model = AutoModelForVision2Seq.from_pretrained(
            args.model_id,
            attn_implementation=args.attn_impl,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"[OpenVLA] Warning: failed to load with attn_implementation={args.attn_impl} ({e}) -> fallback.")
        model = AutoModelForVision2Seq.from_pretrained(
            args.model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

    model.to(device)
    model.eval()

    if args.no_cache:
        if hasattr(model, "config"):
            model.config.use_cache = False
        if hasattr(model, "generation_config"):
            model.generation_config.use_cache = False

    def make_env(render_onscreen: bool):
        return create_env(
            env_name=args.env_name,
            robots=args.robot,
            camera_names=["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"],
            camera_widths=args.camera_w,
            camera_heights=args.camera_h,
            seed=args.seed,
            render_onscreen=render_onscreen,
        )

    env = make_env(render_onscreen=args.render_onscreen)
    obs = env.reset()

    img_keys = list_image_like_keys(obs)
    if len(img_keys) == 0 and args.render_onscreen:
        print("[Env] No image observations found with render_onscreen=True.")
        print("[Env] Falling back to render_onscreen=False (offscreen) to recover camera obs.")
        try:
            env.close()
        except Exception:
            pass
        env = make_env(render_onscreen=False)
        obs = env.reset()
        img_keys = list_image_like_keys(obs)

    if len(img_keys) == 0:
        raise RuntimeError(
            f"[Env] No image observations available in obs. Keys: {list(obs.keys())}\n"
            f"Expected camera obs keys ending with '_image'."
        )

    g_key = resolve_image_key(obs, args.global_key)
    h_key = resolve_image_key(obs, args.hand_key)
    print(f"[Obs] using global_key={g_key} hand_key={h_key}")

    # IMPORTANT: match action dim of current env
    action_dim = int(getattr(env, "action_dim", 0))
    if action_dim <= 0:
        raise RuntimeError(f"[Env] Could not read env.action_dim, got {action_dim}")
    print(f"[Env] action_dim={action_dim} (will build actions accordingly)")

    filt = ActionFilter(alpha=args.smooth_alpha, slew=args.slew)
    grip = GripperHysteresis(
        close_th=args.grip_close_th,
        open_th=args.grip_open_th,
        hold_steps=args.grip_hold,
    )

    def inv(v: float, flag: bool) -> float:
        return -v if flag else v

    print("[Rollout] Starting.")
    for t in range(args.max_steps):
        if args.render_onscreen:
            try:
                env.render()
            except Exception:
                pass

        g_img = obs_img_to_pil(obs[g_key])
        h_img = obs_img_to_pil(obs[h_key])

        g_img = maybe_flip(g_img, args.flip_ud, args.flip_lr)
        h_img = maybe_flip(h_img, args.flip_ud, args.flip_lr)

        if args.use_composite:
            img_in = make_composite(g_img, h_img, mode=args.composite_mode, label=args.composite_label)
        else:
            img_in = g_img

        prompt = build_prompt(args.instruction, style=args.prompt_style)

        with torch.inference_mode():
            inputs = processor(prompt, img_in, return_tensors="pt")

            if not args.disable_openvla_suffix_fix:
                inputs = ensure_openvla_suffix_token(inputs, empty_token_id=args.openvla_empty_token_id)

            # Move tensors safely (do NOT cast input_ids / attention_mask)
            for k, v in list(inputs.items()):
                if not isinstance(v, torch.Tensor):
                    continue
                if v.is_floating_point():
                    inputs[k] = v.to(device=device, dtype=torch_dtype)
                else:
                    inputs[k] = v.to(device=device)

            a7 = model.predict_action(
                **inputs,
                unnorm_key=args.unnorm_key,
                do_sample=args.do_sample,
                use_cache=(not args.no_cache),
            )

        a7 = np.array(a7).astype(np.float32).reshape(-1)
        if a7.size < 7:
            print(f"[Warn] action dim={a7.size}, expected 7; padding with zeros.")
            a7 = np.pad(a7, (0, 7 - a7.size))
        a7 = a7[:7]

        dx, dy, dz, droll, dpitch, dyaw, g = a7.tolist()

        if args.gripper_0_1:
            g = float(g) * 2.0 - 1.0

        arm6 = np.array([
            inv(dx, args.inv_x) * args.pos_step,
            inv(dy, args.inv_y) * args.pos_step,
            inv(dz, args.inv_z) * args.pos_step,
            inv(droll, args.inv_roll) * args.rot_step,
            inv(dpitch, args.inv_pitch) * args.rot_step,
            inv(dyaw, args.inv_yaw) * args.rot_step,
        ], dtype=np.float32)

        arm6 *= float(args.arm_scale)
        arm6 = filt(arm6)
        arm6 = np.clip(arm6, -args.clip, args.clip)

        g_cmd = grip(g)
        g_cmd = float(np.clip(g_cmd, -args.clip, args.clip))

        # Build action with correct dimension for current env
        a_env = build_env_action(action_dim, arm6, g_cmd, clip=args.clip)

        obs, rew, done, info = env.step(a_env)

        if args.render_onscreen and _HAS_CV2:
            maybe_show_cv2("GLOBAL", g_img)
            maybe_show_cv2("HAND", h_img)
            if args.use_composite:
                maybe_show_cv2("COMPOSITE", img_in)
            cv2.waitKey(1)

        if t % 10 == 0:
            print(f"[t={t:04d}] a7={a7.round(4)}  arm6={arm6.round(3)}  grip={g_cmd:+.0f}  r={rew:.3f}")

        if args.sleep > 0:
            time.sleep(args.sleep)

        if done:
            print(f"[Rollout] done at step {t}")
            break

    try:
        env.close()
    except Exception:
        pass
    if _HAS_CV2:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
    print("[Rollout] Finished.")


if __name__ == "__main__":
    main()
