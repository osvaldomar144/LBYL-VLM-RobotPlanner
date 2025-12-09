# vila_open/robocasa_utils.py

from __future__ import annotations
import numpy as np
from PIL import Image

def _to_uint8_image(arr: np.ndarray) -> np.ndarray:
    """Converte un array numpy in un'immagine uint8 (H, W, 3)."""
    arr = np.asarray(arr)
    
    # Gestione canali (C, H, W) -> (H, W, C)
    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))

    # Gestione float [0, 1] o [0, 255]
    if np.issubdtype(arr.dtype, np.floating):
        if arr.max() <= 1.05 and arr.min() >= -0.05:
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    # Gestione Grayscale -> RGB
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)

    return arr

def obs_to_pil_image(obs: dict, camera_name: str) -> Image.Image:
    """Estrae l'immagine dall'osservazione con fallback intelligente."""
    if obs is None:
        raise ValueError("L'osservazione (obs) è None.")

    target_key = f"{camera_name}_image"
    
    # 1. Caso ideale: la chiave esiste
    if target_key in obs:
        return Image.fromarray(_to_uint8_image(obs[target_key]))

    # 2. Fallback: Cerca qualsiasi immagine RGB valida (ignora depth)
    valid_keys = [k for k in obs.keys() if k.endswith("_image") and "depth" not in k]
    
    if not valid_keys:
        raise KeyError(
            f"Nessuna immagine trovata in obs. Chiavi disponibili: {list(obs.keys())}. "
            "Verifica di aver impostato use_camera_obs=True."
        )
    
    # Prendi la prima disponibile e avvisa
    fallback_key = valid_keys[0]
    print(f"[robocasa_utils] WARNING: '{target_key}' non trovata. Uso fallback: '{fallback_key}'")
    return Image.fromarray(_to_uint8_image(obs[fallback_key]))