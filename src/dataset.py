import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoProcessor

class RoboCasaDataset(Dataset):
    def __init__(self, hdf5_paths, processor, instruction="do the task"):
        self.processor = processor
        self.instruction = instruction
        self.episodes = []
        
        print(f"Caricamento {len(hdf5_paths)} file HDF5...")
        for path in hdf5_paths:
            self._load_episode(path)
        print(f"Totale campioni caricati: {len(self.episodes)}")

    def _load_episode(self, path):
        try:
            with h5py.File(path, 'r') as f:
                # Percorsi tipici RoboSuite/RoboCasa
                # Verifica con un print(list(f['obs'].keys())) se cambiano
                demos = list(f['data'].keys())
                for demo_key in demos:
                    demo = f['data'][demo_key]
                    
                    # Carichiamo immagini e azioni
                    # Usa la camera che userai nel test (es. robot0_agentview_right_image)
                    # RoboCasa spesso salva le immagini come (T, H, W, 3) invertite
                    imgs = demo['obs']['robot0_agentview_right_image'][:] 
                    actions = demo['actions'][:] 

                    length = len(imgs)
                    for i in range(length):
                        self.episodes.append({
                            'image': imgs[i], # Numpy array
                            'action': actions[i] # Full action (12 dim)
                        })
        except Exception as e:
            print(f"Errore caricamento {path}: {e}")

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        sample = self.episodes[idx]
        
        # 1. Process Image
        # Robosuite image è spesso capovolta, controlla se serve np.flipud
        image_np = np.flipud(sample['image']) 
        image = Image.fromarray(image_np)
        
        # 2. Process Action (MAPPING CRUCIALE)
        # RoboCasa (12 dim): [Arm(0-5), Gripper(6), Base(7-9), Torso(10-11)]
        full_action = sample['action']
        
        # Prendiamo solo Braccio (6) + Gripper (1) = 7-DoF
        # Normalizziamo? OpenVLA preferisce azioni normalizzate, ma per ora
        # proviamo a far imparare i valori grezzi o applichiamo un clamp.
        arm_action = full_action[:6]
        gripper_action = full_action[6:7] # Mantieni dimensione
        
        # Uniamo in 7-DoF [x,y,z, r,p,y, grip]
        target_action = np.concatenate([arm_action, gripper_action], axis=0)
        target_action = torch.tensor(target_action, dtype=torch.float32)

        # 3. Prompt
        prompt = f"In: {self.instruction}\nOut:"

        # 4. Tokenization (Gestita dal collator o qui se semplice)
        # Per semplicità restituiamo i dati grezzi, il trainer gestirà il forward
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        
        # Rimuovi la batch dimension aggiunta dal processor
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs['labels'] = target_action # Usiamo l'azione come label per la regressione
        
        return inputs