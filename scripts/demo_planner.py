# scripts/demo_planner.py

import argparse
from pathlib import Path

from PIL import Image

from vila_open.vlm_client import VLMClient, VLMConfig
from vila_open.planning_loop import plan_once


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--goal", type=str, required=True)
    args = parser.parse_args()

    image_path = Path(args.image_path)
    image = Image.open(image_path).convert("RGB")

    config = VLMConfig(
        device="cuda",
        device_map="auto",
        max_new_tokens=512,
        verbose=True,
    )
    client = VLMClient(config)

    # per ora nessuna azione eseguita
    history = []

    # primitives disponibili (per ora generiche, dopo le mapperemo a RoboCasa)
    available_primitives = [
        "pick",
        "place",
        "open",
        "close",
        "press_button",
        "turn_knob",
        "navigate",
    ]

    plan = plan_once(
        image=image,
        goal_instruction=args.goal,
        history=history,
        available_primitives=available_primitives,
        vlm_client=client,
    )

    print("\n=== Generated plan ===")
    print(plan.to_json())


if __name__ == "__main__":
    main()
