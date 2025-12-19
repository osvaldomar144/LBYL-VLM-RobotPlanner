# src/policy.py
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
import numpy as np

class VLAPolicy:
    def __init__(self, model_id="openvla/openvla-7b"):
        print(f"[VLA] Caricamento Policy: {model_id} (4-bit)...")
        
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_id, 
            attn_implementation="flash_attention_2", 
            torch_dtype=torch.bfloat16, 
            load_in_4bit=True, 
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        print("[VLA] Policy caricata e pronta.")

    def get_action(self, image, instruction):
        """
        Input: PIL Image, String instruction (sub-task)
        Output: Numpy array (7-DOF action: x,y,z,roll,pitch,yaw,gripper)
        """
        # Prompt specifico per OpenVLA
        prompt = f"In: {instruction}\nOut:"
        
        inputs = self.processor(prompt, image).to("cuda", dtype=torch.bfloat16)
        
        with torch.no_grad():
            # predict_action restituisce l'azione un-normalized (ready to execute)
            action = self.model.predict_action(**inputs, unnorm_key="bridge_orig")
            
        return action