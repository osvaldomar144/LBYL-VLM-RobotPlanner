# test/test_vlm_client.py

from pathlib import Path
from PIL import Image

from vila_open.vlm_client import VLMClient, VLMConfig


def main():
    img_path = Path("kitchen_test.jpg")  # metti qui un'immagine di prova nella root
    image = Image.open(img_path).convert("RGB")

    config = VLMConfig(
        device="cuda",
        device_map="auto",
        max_new_tokens=128,
        verbose=True,
    )
    client = VLMClient(config)

    system_prompt = "You are a helpful robot planner. Answer ONLY with JSON."
    user_prompt = "Look at the image and output a JSON with a key 'plan' describing the scene."

    out = client.generate_plan_text(
        image=image,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    print("\n=== TEST OUTPUT ===")
    print(out)


if __name__ == "__main__":
    main()
