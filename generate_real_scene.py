import gymnasium as gym
import sapien.core as sapien
import numpy as np
from mani_skill.utils.registration import register_env
from mani_skill.envs.tasks.tabletop.pick_clutter_ycb import PickClutterYCBEnv
from mani_skill.utils.scene_builder.replicacad import ReplicaCADSceneBuilder
from mani_skill.utils.wrappers import RecordEpisode

# --- DEFINIZIONE DEL CUSTOM TASK ---
@register_env("RealisticPickClutter-v1", max_episode_steps=100)
class RealisticPickClutter(PickClutterYCBEnv):
    """
    Task custom che unisce PickClutterYCB con ReplicaCAD.
    """
    
    # 1. FIX DEL CRASH: Intercettiamo scene_builder qui!
    def __init__(self, *args, scene_builder=None, **kwargs):
        # Inizializziamo il builder ReplicaCAD internamente
        self.scene_builder_impl = ReplicaCADSceneBuilder(self)
        
        # Rimuoviamo scene_builder da kwargs se presente per non far crashare il padre
        if 'scene_builder' in kwargs:
            kwargs.pop('scene_builder')
            
        super().__init__(*args, **kwargs)

    def _load_scene(self, options: dict):
        # 2. Costruiamo la stanza
        # Questo comando carica automaticamente frl_apartment_stage.glb e i mobili
        with self.scene_builder_impl.build(self.scene):
            pass # Usa la configurazione di default o quella passata in options

        # 3. POSIZIONAMENTO STRATEGICO (Salotto)
        # Queste coordinate (-1.0, -3.0) sono tipiche del salotto in apt_1.
        # Ruotiamo di 0 o 90 gradi per orientare il tavolo verso il centro stanza.
        # Modifica p=[x, y, z] se vedi ancora muri.
        self.robot_table_pose = sapien.Pose(p=[-1.0, -3.0, 0], q=[1, 0, 0, 0])
        
        # 4. Carichiamo la logica del tavolo e oggetti
        super()._load_scene(options)

    def _load_lighting(self, options: dict):
        # Usiamo le luci della scena ReplicaCAD, non quelle da studio fotografico
        pass

# --- ESECUZIONE ---
def run_custom_task():
    print("=== Avvio Custom Task: RealisticPickClutter-v1 (Fixed) ===")
    
    sensor_configs = {"width": 512, "height": 512}

    # Creiamo l'env
    # NOTA: Non passiamo più scene_builder qui dentro se crea problemi,
    # lo abbiamo gestito nel __init__ ma per sicurezza lo passiamo pulito.
    env = gym.make(
        "RealisticPickClutter-v1",
        obs_mode="rgbd",
        control_mode="pd_ee_delta_pose",
        robot_uids="panda",
        render_mode="rgb_array",
        sim_backend="cpu",
        sensor_configs=sensor_configs,
        # Passiamo l'argomento che ora la nostra classe sa gestire
        scene_builder="replicacad", 
        options={
            "scene_id": "apt_1",  # Specifichiamo l'appartamento
            "reconfigure": True
        } 
    )

    env = RecordEpisode(
        env,
        output_dir="datasets/thesis_custom_task_fixed",
        save_trajectory=True,
        save_video=True,
        info_on_video=True
    )

    print("Generazione 5 Episodi di Test...")
    for i in range(5):
        try:
            obs, _ = env.reset(seed=i)
            
            terminated, truncated = False, False
            step = 0
            while not (terminated or truncated) and step < 60:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                step += 1
            
            env.flush_video()
            print(f"Episodio {i+1} salvato.")
            
        except Exception as e:
            print(f"Errore durante l'episodio {i}: {e}")

    env.close()
    print("Finito! Controlla datasets/thesis_custom_task_fixed")

if __name__ == "__main__":
    run_custom_task()