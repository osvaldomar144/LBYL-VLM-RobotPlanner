import torch
import numpy as np
import sapien.core as sapien
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.agents.robots.panda import Panda # Importa il robot predefinito se disponibile

@register_env("MyFrankaPush-v0", max_episode_steps=100)
class MyFrankaPushEnv(BaseEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _configure_agent(self):
        # Usa la configurazione standard del Panda in ManiSkill
        # Se usi MS3 beta, potresti dover usare self.agent = ...
        self._agent_cfg = dict(
            model_urdf_path="panda", # 'panda' è spesso un alias interno
            control_mode="pd_ee_delta_pose", 
        )

    def _load_scene(self, options: dict):
        # Luci
        self.scene.set_ambient_light([0.3, 0.3, 0.3])
        self.scene.add_directional_light([1, 1, -1], [1, 1, 1])

        # Pavimento
        self.ground = sapien_utils.create_ground(self.scene, altitude=0)

        # Tavolo
        self.table = actors.build_cube(
            self.scene, 
            half_size=[0.4, 0.4, 0.02], 
            color=[0.8, 0.6, 0.4, 1], 
            name="table", 
            body_type="static",
            pose=sapien.Pose(p=[0.5, 0, 0.02]) 
        )

        # Cubo Target
        self.obj = actors.build_cube(
            self.scene,
            half_size=[0.02, 0.02, 0.02],
            color=[1, 0, 0, 1],
            name="target_cube",
            body_type="dynamic",
            pose=sapien.Pose(p=[0.5, 0, 0.1])
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # Reset semplice
        pass
        
    def _get_obs_extra(self, info: dict):
        return dict()