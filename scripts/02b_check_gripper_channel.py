import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

import argparse
import numpy as np
from vla_skills.datasets.robocasa_hdf5_dataset import RoboCasaRandomFrameDataset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_files", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args()

    ds = RoboCasaRandomFrameDataset(args.index_files, samples_per_epoch=args.n, clip_actions=False)

    g = []
    dummy11 = []
    for i in range(args.n):
        a = ds[i]["action"].numpy()
        g.append(a[6])
        dummy11.append(a[11])

    g = np.array(g)
    dummy11 = np.array(dummy11)
    print("[Check] a[6] stats:", float(g.min()), float(g.max()), float(g.mean()))
    print("[Check] a[11] unique:", np.unique(dummy11))

if __name__ == "__main__":
    main()
