# scripts/01_make_hdf5_index.py
import argparse
import json
import os
import h5py

GLOBAL_KEY_DEFAULT = "robot0_agentview_right_image"
HAND_KEY_DEFAULT   = "robot0_eye_in_hand_image"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Path to demo_*.hdf5")
    ap.add_argument("--out", required=True, help="Output .jsonl path")
    ap.add_argument("--max_demos", type=int, default=None, help="Optional cap (e.g., 500)")
    ap.add_argument("--global_key", type=str, default=GLOBAL_KEY_DEFAULT)
    ap.add_argument("--hand_key", type=str, default=HAND_KEY_DEFAULT)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with h5py.File(args.dataset, "r") as f, open(args.out, "w") as out:
        data = f["data"]
        demos = sorted([k for k in data.keys() if k.startswith("demo_")],
                       key=lambda x: int(x.split("_")[1]))
        if args.max_demos is not None:
            demos = demos[:args.max_demos]

        # verifica chiavi una volta
        first = data[demos[0]]["obs"]
        if args.global_key not in first:
            raise KeyError(f"global_key '{args.global_key}' not found. Available: {list(first.keys())}")
        if args.hand_key not in first:
            raise KeyError(f"hand_key '{args.hand_key}' not found. Available: {list(first.keys())}")

        print(f"[Index] demos: {len(demos)}")
        print(f"[Index] global_key: {args.global_key}")
        print(f"[Index] hand_key  : {args.hand_key}")

        for dn in demos:
            demo = data[dn]
            T = int(demo["actions"].shape[0])

            ep_meta_raw = demo.attrs.get("ep_meta", "{}")
            try:
                ep_meta = json.loads(ep_meta_raw)
            except Exception:
                ep_meta = {}

            rec = {
                "dataset_path": args.dataset,
                "demo": dn,
                "T": T,
                "instruction": ep_meta.get("lang", ""),
                "layout_id": ep_meta.get("layout_id", None),
                "style_id": ep_meta.get("style_id", None),
                "global_key": args.global_key,
                "hand_key": args.hand_key,
            }
            out.write(json.dumps(rec) + "\n")

    print(f"[Index] wrote: {args.out}")

if __name__ == "__main__":
    main()
