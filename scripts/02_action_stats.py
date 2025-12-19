# scripts/02_action_stats.py

import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

import argparse
import numpy as np
import torch
from vla_skills.datasets.robocasa_hdf5_dataset import RoboCasaRandomFrameDataset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_files", nargs="+", required=True)
    ap.add_argument("--samples", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ds = RoboCasaRandomFrameDataset(
        index_files=args.index_files,
        samples_per_epoch=args.samples,
        seed=args.seed,
        return_instruction=False,
        clip_actions=False,
    )

    acts = []
    for i in range(len(ds)):
        a = ds[i]["action"].numpy()
        acts.append(a)
    acts = np.stack(acts, axis=0)  # (N,A)

    mn = acts.min(axis=0)
    mx = acts.max(axis=0)
    mean = acts.mean(axis=0)
    std = acts.std(axis=0)

    print(f"[Stats] N={acts.shape[0]}  A={acts.shape[1]}")
    for j in range(acts.shape[1]):
        print(f"  dim{j:02d}: min={mn[j]:+.4f} max={mx[j]:+.4f} mean={mean[j]:+.4f} std={std[j]:+.4f}")

    print("\nTip: se vedi una dimensione sempre ~0, probabilmente è un canale non usato.")

if __name__ == "__main__":
    main()
