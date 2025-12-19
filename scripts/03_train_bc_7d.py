# scripts/03_train_bc_7d.py
import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from vla_skills.datasets.robocasa_hdf5_dataset import RoboCasaRandomFrameDataset


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
    """
    Predice SOLO action dims [0..6] (6 arm + 1 gripper).
    """
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
        return torch.tanh(a)  # [-1,1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_files", nargs="+", required=True)
    ap.add_argument("--out_dir", type=str, default="runs/bc_7d")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--samples_per_epoch", type=int, default=200_000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    ds = RoboCasaRandomFrameDataset(
        index_files=args.index_files,
        samples_per_epoch=args.samples_per_epoch,
        seed=args.seed,
        return_instruction=False,
        clip_actions=True,
    )

    dl = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=False,  # randomizzazione nel dataset
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    model = DualViewPolicy7D(out_dim=7).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.SmoothL1Loss()  # più robusta di MSE con outlier
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    step = 0
    for ep in range(args.epochs):
        ds.set_epoch(ep)
        model.train()
        t0 = time.time()
        running = 0.0

        for batch in dl:
            img_g = batch["img_global"].to(device, non_blocking=True)
            img_h = batch["img_hand"].to(device, non_blocking=True)

            # target: dims 0..6
            act_t = batch["action"][:, :7].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                act_p = model(img_g, img_h)
                loss = loss_fn(act_p, act_t)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            running += float(loss.item())
            step += 1
            if step % 200 == 0:
                avg = running / 200.0
                running = 0.0
                print(f"[ep {ep+1}/{args.epochs}] step={step} loss={avg:.6f}")

        ckpt_path = os.path.join(args.out_dir, f"ckpt_ep{ep:02d}.pt")
        torch.save({"model": model.state_dict(), "args": vars(args), "epoch": ep}, ckpt_path)
        print(f"[ep {ep+1}] saved {ckpt_path}  epoch_time={(time.time()-t0):.1f}s")

    last_path = os.path.join(args.out_dir, "ckpt_last.pt")
    torch.save({"model": model.state_dict(), "args": vars(args)}, last_path)
    print(f"[Done] saved: {last_path}")


if __name__ == "__main__":
    main()