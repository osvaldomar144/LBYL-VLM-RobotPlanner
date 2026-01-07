import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from PIL import Image

def test_vla_loading():
    print(">>> 1. Caricamento del Processor...")
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    
    print(">>> 2. Configurazione Quantizzazione 4-bit...")
    # Configurazione esplicita per evitare i warning di deprecazione
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    print(">>> 3. Caricamento del Modello OpenVLA (Safe Mode)...")
    # Abbiamo rimosso 'attn_implementation="flash_attention_2"' per evitare errori
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", 
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True, 
        trust_remote_code=True,
        quantization_config=quantization_config
    )
    
    # Controlla se la GPU è usata
    if torch.cuda.is_available():
        print(f">>> Modello caricato su GPU: {torch.cuda.get_device_name(0)}")
        print(f">>> Memoria GPU allocata: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    else:
        print(">>> ATTENZIONE: CUDA non rilevato! Il modello è sulla CPU (sarà lentissimo).")

    # Test veloce di inferenza
    print(">>> 4. Test di inferenza simulata...")
    prompt = "In: What action should the robot take to {INSTRUCTION}?\nOut:"
    instruction = "pick up the red cube"
    
    # Immagine dummy rossa
    dummy_image = Image.new('RGB', (224, 224), color='red')
    
    inputs = processor(prompt.format(INSTRUCTION=instruction), dummy_image).to("cuda:0", dtype=torch.bfloat16)
    
    # Generazione azione
    action = model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
    
    print(f">>> Azione generata: {action}")
    print(">>> TEST SUPERATO! Il sistema è pronto.")

if __name__ == "__main__":
    test_vla_loading()