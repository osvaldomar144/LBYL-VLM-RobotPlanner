import os, sys, json, random
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import h5py


# -------------------------
# Robust index reader
# -------------------------
def load_index_files(paths: List[str]) -> List[Dict[str, Any]]:
    entries = []
    for p in paths:
        with open(p, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                entries.append(d)
    if len(entries) == 0:
        raise RuntimeError("Index files are empty.")
    return entries


def get_any(d: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default


# -------------------------
# Dataset: sample random (demo, t) but return 2 frames (t-1, t) + proprio + action
# -------------------------
class RoboCasaState2FrameDataset(Dataset):
    def __init__(self, index_entries: List[Dict[str, Any]], samples_per_epoch: int):
        self.entries = index_entries
        self.samples_per_epoch = int(samples_per_epoch)

        # per-process cache (works with dataloader workers)
        self._h5 = {}

    def __len__(self):
        return self.samples_per_epoch

    def _open(self, path: str):
        if path not in self._h5:
            self._h5[path] = h5py.File(path, "r")
        return self._h5[path]

    def __getitem__(self, idx):
        e = random.choice(self.entries)

        h5_path = get_any(e, ["hdf5", "hdf5_path", "dataset_path", "dataset", "path"])
        demo_name = get_any(e, ["demo", "demo_name", "episode"])
        T = int(get_any(e, ["T", "len", "length", "n_steps"], 0))

        if h5_path is None or demo_name is None:
            raise RuntimeError(f"Bad index entry (missing hdf5/demo): {e}")

        # keys (fallback to known names)
        global_key = get_any(e, ["global_key"], "robot0_agentview_right_image")
        hand_key   = get_any(e, ["hand_key"],   "robot0_eye_in_hand_image")
        action_key = get_any(e, ["action_key"], "actions")

        # proprio keys (known from your inspect)
        joint_key  = get_any(e, ["joint_key"],  "robot0_joint_pos")
        gripq_key  = get_any(e, ["gripq_key"],  "robot0_gripper_qpos")

        f = self._open(h5_path)
        grp = f["data"][demo_name]

        # infer T if not in index
        if T <= 0:
            T = grp[action_key].shape[0]

        # choose t >= 1 so we have t-1
        t = random.randint(1, T - 1)
        t0 = t - 1

        # images: (H,W,3) uint8
        g0 = grp["obs"][global_key][t0]
        g1 = grp["obs"][global_key][t]
        h0 = grp["obs"][hand_key][t0]
        h1 = grp["obs"][hand_key][t]

        # stack along channel => (H,W,6)
        g = np.concatenate([g0, g1], axis=2).astype(np.uint8)
        h = np.concatenate([h0, h1], axis=2).astype(np.uint8)

        # proprio
        joint = grp["obs"][joint_key][t].astype(np.float32)          # (7,)
        gripq = grp["obs"][gripq_key][t].astype(np.float32)          # (2,)
        # simple normalization: joints in [-pi,pi] approx
        joint = np.clip(joint / np.pi, -1.0, 1.0)
        state = np.concatenate([joint, gripq], axis=0).astype(np.float32)  # (9,)

        # actions: (12,)
        a12 = grp[action_key][t].astype(np.float32)
        arm6 = a12[:6]
        grip = 1.0 if float(a12[6]) > 0.0 else 0.0   # label for BCE

        return g, h, state, arm6, np.float32(grip)


# -------------------------
# Model: 2-view conv + state MLP, heads: arm6 regression + gripper logit
# -------------------------
class SmallConvEncoder(nn.Module):
    def __init__(self, in_ch: int = 6, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 5, stride=2, padding=2), nn.ReLU(inplace=True),   # 128->64
            nn.Conv2d(32, 64, 5, stride=2, padding=2), nn.ReLU(inplace=True),      # 64->32
            nn.Conv2d(64, 128, 5, stride=2, padding=2), nn.ReLU(inplace=True),     # 32->16
            nn.Conv2d(128, 256, 5, stride=2, padding=2), nn.ReLU(inplace=True)     # 16->8
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
        self.grip_head = nn.Linear(256, 1)  # logit

    def forward(self, g, h, s):
        fg = self.enc(g)
        fh = self.enc(h)
        fs = self.state_mlp(s)
        x = torch.cat([fg, fh, fs], dim=1)
        z = self.trunk(x)
        arm = torch.tanh(self.arm_head(z))        # [-1,1]
        grip_logit = self.grip_head(z).squeeze(1) # (B,)
        return arm, grip_logit


def img_uint8_to_tensor(x: np.ndarray) -> torch.Tensor:
    # (H,W,6) uint8 -> (6,H,W) float [0,1]
    t = torch.from_numpy(x).float() / 255.0
    t = t.permute(2, 0, 1).contiguous()
    return t


def estimate_grip_pos_weight(entries, n=20000) -> float:
    ds = RoboCasaState2FrameDataset(entries, samples_per_epoch=n)
    pos = 0
    for i in range(n):
        _, _, _, _, grip = ds[i]
        pos += int(grip > 0.5)
    neg = n - pos
    # pos_weight = neg/pos
    if pos <= 0:
        return 1.0
    return float(neg / max(pos, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_files", nargs="+", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--samples_per_epoch", type=int, default=150000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--grip_loss_w", type=float, default=5.0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    torch.backends.cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    entries = load_index_files(args.index_files)

    pos_weight = estimate_grip_pos_weight(entries, n=10000)
    print(f"[Train] estimated gripper pos_weight ~ {pos_weight:.3f}")

    ds = RoboCasaState2FrameDataset(entries, samples_per_epoch=args.samples_per_epoch)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = VLA6DGripState2F(state_dim=9).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    arm_criterion = nn.MSELoss()
    grip_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    global_step = 0
    for ep in range(args.epochs):
        model.train()
        running = 0.0

        for it, batch in enumerate(dl):
            g_u8, h_u8, state, arm_tgt, grip_tgt = batch

            # convert images on CPU -> tensors
            g = torch.stack([img_uint8_to_tensor(x.numpy()) for x in g_u8], dim=0)
            h = torch.stack([img_uint8_to_tensor(x.numpy()) for x in h_u8], dim=0)

            g = g.to(device, non_blocking=True)
            h = h.to(device, non_blocking=True)
            state = state.to(device, non_blocking=True).float()
            arm_tgt = arm_tgt.to(device, non_blocking=True).float()
            grip_tgt = grip_tgt.to(device, non_blocking=True).float()

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                arm_pred, grip_logit = model(g, h, state)
                arm_loss = arm_criterion(arm_pred, arm_tgt)
                grip_loss = grip_criterion(grip_logit, grip_tgt)
                loss = arm_loss + args.grip_loss_w * grip_loss

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running += float(loss.item())
            global_step += 1

            if (global_step % 200) == 0:
                print(f"[ep {ep+1}/{args.epochs}] step={global_step} loss={running/200:.6f} arm={arm_loss.item():.4f} grip={grip_loss.item():.4f}")
                running = 0.0

        ckpt_path = os.path.join(args.out_dir, f"ckpt_ep{ep:02d}.pt")
        torch.save({"model": model.state_dict(), "pos_weight": pos_weight}, ckpt_path)
        print(f"[ep {ep+1}] saved {ckpt_path}")

    last_path = os.path.join(args.out_dir, "ckpt_last.pt")
    torch.save({"model": model.state_dict(), "pos_weight": pos_weight}, last_path)
    print(f"[Done] saved: {last_path}")


if __name__ == "__main__":
    main()
