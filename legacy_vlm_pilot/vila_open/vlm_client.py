# vila_open/vlm_client.py

from dataclasses import dataclass
from typing import List, Sequence, Union, Optional

from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

try:
    from qwen_vl_utils import process_vision_info
except ImportError as e:
    raise ImportError(
        "qwen-vl-utils non è installato. "
        "Installa con: pip install qwen-vl-utils"
    ) from e


@dataclass
class VLMConfig:
    """
    Configurazione per il Vision-Language Model client.
    """
    model_name: str = "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct"
    device: str = "cuda"
    device_map: str = "auto"
    max_new_tokens: int = 256        # <-- abbassato di default (JSON non richiede 1024)
    verbose: bool = True

    # Riduce memoria (molto utile con prompt lunghi + multi-image)
    use_cache: bool = False

    # Fallback automatico in caso di OOM
    oom_retry: bool = True
    oom_retry_tokens: int = 128      # retry con meno token
    oom_retry_use_cache: bool = False


VLMImageInput = Union[Image.Image, Sequence[Image.Image]]


class VLMClient:
    """
    Wrapper per un Vision-Language Model in stile chat.

    Supporta:
      - 1 immagine: image = PIL.Image
      - multi-image: image = [PIL.Image, PIL.Image, ...]
    """

    def __init__(self, config: VLMConfig):
        self.config = config

        if self.config.verbose:
            print(f"[VLMClient] Carico modello: {self.config.model_name}")
            print(f"[VLMClient] device     = {self.config.device}")
            print(f"[VLMClient] device_map = {self.config.device_map}")
            print(f"[VLMClient] max_new_tokens = {self.config.max_new_tokens}")
            print(f"[VLMClient] use_cache      = {self.config.use_cache}")

        # dtype="auto" è il nuovo nome; su versioni HF vecchie può non esistere -> fallback
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                dtype="auto",
                device_map=self.config.device_map,
                trust_remote_code=True,
            )
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype="auto",
                device_map=self.config.device_map,
                trust_remote_code=True,
            )

        self.processor = AutoProcessor.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )

        self.model.eval()

        if self.config.verbose:
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"[VLMClient] Modello caricato ({total_params/1e9:.2f}B parametri circa).")

    def _normalize_images(self, image: VLMImageInput) -> List[Image.Image]:
        if isinstance(image, Image.Image):
            return [image]
        if isinstance(image, (list, tuple)):
            if len(image) == 0:
                raise ValueError("Input images list is empty.")
            if not all(isinstance(im, Image.Image) for im in image):
                raise TypeError("All elements in image list must be PIL.Image.Image.")
            return list(image)
        raise TypeError("image must be a PIL.Image.Image or a list/tuple of PIL.Image.Image.")

    def _build_messages(
        self,
        image: VLMImageInput,
        system_prompt: str,
        user_prompt: str
    ):
        """
        Inserisce label testuali per ancorare Image 1/2.
        """
        images = self._normalize_images(image)

        user_content = []

        if len(images) == 2:
            user_content += [
                {"type": "text", "text": "Image 1 (GLOBAL view, robot from behind)."},
                {"type": "image", "image": images[0]},
                {"type": "text", "text": "Image 2 (HAND view, camera on the gripper)."},
                {"type": "image", "image": images[1]},
            ]
        else:
            for i, im in enumerate(images, start=1):
                user_content.append({"type": "text", "text": f"Image {i}."})
                user_content.append({"type": "image", "image": im})

        user_content.append({"type": "text", "text": user_prompt})

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": user_content},
        ]
        return messages

    def _generate_once(
        self,
        inputs,
        max_new_tokens: int,
        use_cache: bool,
    ):
        with torch.no_grad():
            return self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=use_cache,
            )

    def generate_plan_text(
        self,
        image: VLMImageInput,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        messages = self._build_messages(image, system_prompt, user_prompt)

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # metti input su device principale
        inputs = inputs.to(self.config.device)

        try:
            generated_ids = self._generate_once(
                inputs,
                max_new_tokens=self.config.max_new_tokens,
                use_cache=self.config.use_cache,
            )
        except torch.cuda.OutOfMemoryError as e:
            if not self.config.oom_retry:
                raise

            # Fallback: libera cache e ritenta più leggero
            if self.config.verbose:
                print("[VLMClient] CUDA OOM -> retry con impostazioni più leggere...")
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

            generated_ids = self._generate_once(
                inputs,
                max_new_tokens=min(self.config.max_new_tokens, self.config.oom_retry_tokens),
                use_cache=self.config.oom_retry_use_cache,
            )

        # trim prompt tokens
        input_ids = inputs["input_ids"]
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
        ]

        output_texts = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        output_text = output_texts[0]

        if self.config.verbose:
            print("[VLMClient] Output grezzo dal modello:")
            print(output_text)

        return output_text
