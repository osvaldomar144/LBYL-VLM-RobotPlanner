import json
import numpy as np
import matplotlib.pyplot as plt

DATASET_PATH = "dataset_finetuning_v2/dataset.json"

print(f">>> Caricamento {DATASET_PATH}...")
with open(DATASET_PATH, 'r') as f:
    data = json.load(f)

print(f"Totale Frame: {len(data)}")

velocities = []
grippers = []

for item in data:
    action = np.array(item['action'])
    # Calcola la velocità (magnitudo del vettore movimento x,y,z)
    vel = np.linalg.norm(action[:3])
    velocities.append(vel)
    grippers.append(action[6])

velocities = np.array(velocities)
grippers = np.array(grippers)

# STATISTICHE
print("\n--- AUTOPSIA DEL MOVIMENTO ---")
print(f"Velocità Media: {np.mean(velocities):.4f}")
print(f"Velocità Max:   {np.max(velocities):.4f}")
print(f"Frame 'FERMI' (< 0.01): {np.sum(velocities < 0.01)} ({np.sum(velocities < 0.01)/len(data)*100:.1f}%)")
print(f"Frame 'LENTI' (< 0.02): {np.sum(velocities < 0.02)} ({np.sum(velocities < 0.02)/len(data)*100:.1f}%)")
print(f"Frame 'BUONI' (> 0.05): {np.sum(velocities > 0.05)} ({np.sum(velocities > 0.05)/len(data)*100:.1f}%)")

print("\n--- AUTOPSIA DEL GRIPPER ---")
print(f"Pinza APERTA (> 0.5): {np.sum(grippers > 0.5)}")
print(f"Pinza CHIUSA (< -0.5): {np.sum(grippers < -0.5)}")
print(f"Pinza INCERTA (-0.5 a 0.5): {np.sum((grippers >= -0.5) & (grippers <= 0.5))}")