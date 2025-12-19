# src/planner.py
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import json
import re

class VLMPlanner:
    def __init__(self, model_id="Qwen/Qwen2-VL-7B-Instruct"):
        print(f"[VLM] Caricamento Planner: {model_id} (4-bit)...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        
        # Caricamento ottimizzato per 3090 Ti
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
            quantization_config={"load_in_4bit": True} 
        )
        print("[VLM] Planner caricato e pronto.")

    def plan(self, image, instruction):
        """
        Input: PIL Image, String instruction
        Output: List of strings (steps)
        """
        # Prompting strutturato per forzare output JSON
        prompt = f"""
        User: Look at this scene. The goal is: "{instruction}".
        Break this task down into short, low-level robot manipulation steps (e.g., "Move to handle", "Grasp handle").
        Output ONLY a Python list of strings. Do not add markdown or explanations.
        Example: ["Move gripper to cup", "Close gripper", "Lift cup"]
        Assistant:
        """

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Preparazione input
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        # Generazione
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)
            output_text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

        # Post-processing per estrarre la risposta dell'assistente
        # Qwen a volte ripete il prompt, prendiamo solo la parte generata se necessario
        # (Qui assumiamo che il decode gestisca bene, ma puliamo l'output)
        response = output_text.split("Assistant:")[-1].strip()
        return self._clean_output(response)

    def _clean_output(self, text):
        # Cerca una lista tra parentesi quadre
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            list_str = match.group(0)
            try:
                # Usa eval in modo controllato o json.loads se le quote sono corrette
                # Spesso i modelli usano single quotes che json non piace, eval è più permissivo per liste python
                return eval(list_str)
            except:
                print(f"[VLM Error] Parsing fallito su: {list_str}")
                return [text] # Fallback
        return [text]