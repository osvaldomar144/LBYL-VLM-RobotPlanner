# vila_open/vlm_client.py

from dataclasses import dataclass
from typing import List

from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

try:
    # qwen-vl-utils è usato anche dall'esempio ufficiale di LLaVA-OneVision
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
    # Modello default: LLaVA-OneVision-1.5-8B-Instruct (open-source, SOTA):contentReference[oaicite:3]{index=3}
    model_name: str = "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct"

    # "cuda" se hai una GPU (RTX 3090), "cpu" solo per test (molto più lento)
    device: str = "cuda"

    # Numero massimo di token di output (il piano JSON sta tranquillamente sotto 512)
    max_new_tokens: int = 512

    # Se vuoi far decidere automaticamente la mappatura sui device
    device_map: str = "auto"

    # Se True, stampa info aggiuntive
    verbose: bool = True


class VLMClient:
    """
    Wrapper per un Vision-Language Model in stile chat (LLaVA-OneVision / Qwen-VL-like).

    Fornisce:
      - caricamento modello + processor
      - metodo generate_plan_text(image, system_prompt, user_prompt)
        che ritorna il testo grezzo generato (idealmente solo JSON).
    """

    def __init__(self, config: VLMConfig):
        self.config = config

        if self.config.verbose:
            print(f"[VLMClient] Carico modello: {self.config.model_name}")
            print(f"[VLMClient] device     = {self.config.device}")
            print(f"[VLMClient] device_map = {self.config.device_map}")

        # Carichiamo il modello in modalità causal LM (come da snippet ufficiale HF):contentReference[oaicite:4]{index=4}
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype="auto",          # sceglie BF16/FP16 automaticamente se possibile
            device_map=self.config.device_map,
            trust_remote_code=True,      # necessario per il codice custom del modello
        )

        # Processor per gestire tokenizzazione + preprocessing immagini/video
        self.processor = AutoProcessor.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )

        # Mettiamo il modello in eval mode
        self.model.eval()

        if self.config.verbose:
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"[VLMClient] Modello caricato ({total_params/1e9:.2f}B parametri circa).")

    def _build_messages(self, image: Image.Image, system_prompt: str, user_prompt: str):
        """
        Costruisce la lista 'messages' nel formato atteso da apply_chat_template.

        - system: contiene il prompt di sistema (regole, schema JSON, ecc.)
        - user: contiene sia l'immagine che il prompt utente (istruzione + history)
        """
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system_prompt,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        # qwen-vl-utils/processor accettano anche direttamente PIL.Image.Image:contentReference[oaicite:5]{index=5}
                        "image": image,
                    },
                    {
                        "type": "text",
                        "text": user_prompt,
                    },
                ],
            },
        ]
        return messages

    def generate_plan_text(
        self,
        image: Image.Image,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """
        Esegue una chiamata al VLM:

        - image: immagine PIL RGB della scena
        - system_prompt: testo di sistema (definizione primitive + schema JSON)
        - user_prompt: testo utente (goal + history)

        Ritorna una stringa con il testo generato (ci aspettiamo solo JSON).
        """
        # 1) Costruisci la struttura chat-style
        messages = self._build_messages(image, system_prompt, user_prompt)

        # 2) Applica il template di chat fornito dal modello (inserisce token speciali, ecc.)
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # 3) Estrai info visive (immagini, video) con qwen_vl_utils
        image_inputs, video_inputs = process_vision_info(messages)

        # 4) Costruisci tensori di input
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # 5) Manda gli input sul device principale (GPU "cuda" nel tuo caso)
        #    BatchEncoding ha già il metodo .to(...)
        inputs = inputs.to(self.config.device)

        # 6) Generazione (no gradiente)
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,   # deterministico (greedy) per piani più stabili
            )

        # 7) Ritaglia i token del prompt per ottenere solo l'output nuovo
        #    (pattern uguale all'esempio ufficiale HF):contentReference[oaicite:6]{index=6}
        input_ids = inputs["input_ids"]
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
        ]

        # 8) Decodifica in testo
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        # Per ora supportiamo batch_size=1 → prendiamo il primo
        output_text = output_texts[0]

        if self.config.verbose:
            print("[VLMClient] Output grezzo dal modello:")
            print(output_text)

        return output_text
