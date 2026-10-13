"""DexJoCo configuration for local and Hugging Face EnvHub loading."""

from dataclasses import dataclass
from typing import Literal

import gymnasium as gym
from lerobot.envs.configs import EnvConfig, HubEnvConfig


@EnvConfig.register_subclass("dexjoco")
@dataclass
class DexJoCoEnvConfig(HubEnvConfig):
    """Use local code when hub_path is None, otherwise load the Hub entry point."""

    render_mode: Literal["rgb_array", "human", "none"] = "rgb_array"
    randomize: bool = False
    randomize_dynamics: bool = False

    @property
    def gym_kwargs(self) -> dict:
        return {
            "render_mode": self.render_mode,
            "randomize": self.randomize,
            "randomize_dynamics": self.randomize_dynamics,
        }

    def create_envs(
        self,
        n_envs: int,
        use_async_envs: bool = False,
    ) -> dict[str, dict[int, gym.vector.VectorEnv]]:
        from dexjoco.envhub import make_env

        return make_env(n_envs=n_envs, use_async_envs=use_async_envs, cfg=self)
