# scripts/00_inspect_robocasa_hdf5.py
import argparse
import json
import h5py

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Path to demo_*.hdf5")
    ap.add_argument("--demo", type=str, default=None, help="Demo name (e.g., demo_0). Default: first demo")
    args = ap.parse_args()

    with h5py.File(args.dataset, "r") as f:
        data = f["data"]
        demos = sorted([k for k in data.keys() if k.startswith("demo_")],
                       key=lambda x: int(x.split("_")[1]))
        print(f"[Inspect] demos found: {len(demos)}")

        dn = args.demo or demos[0]
        demo = data[dn]
        print(f"[Inspect] using demo: {dn}")

        # actions
        actions = demo["actions"]
        print(f"[Inspect] actions shape: {actions.shape}, dtype={actions.dtype}")

        # obs keys
        obs = demo["obs"]
        obs_keys = list(obs.keys())
        print(f"[Inspect] obs keys ({len(obs_keys)}):")
        for k in obs_keys:
            ds = obs[k]
            shape = getattr(ds, "shape", None)
            print(f"  - {k}: shape={shape}, dtype={getattr(ds, 'dtype', None)}")

        # ep_meta
        ep_meta_raw = demo.attrs.get("ep_meta", None)
        if ep_meta_raw is not None:
            try:
                ep_meta = json.loads(ep_meta_raw)
                print("[Inspect] ep_meta keys:", list(ep_meta.keys()))
                if "lang" in ep_meta:
                    print("[Inspect] lang:", ep_meta["lang"])
            except Exception:
                print("[Inspect] ep_meta present but could not parse JSON.")

if __name__ == "__main__":
    main()
