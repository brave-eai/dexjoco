"""DexJoCo configuration for local and Hugging Face EnvHub loading."""

from dataclasses import dataclass
from typing import Literal

import gymnasium as gym
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.envs.configs import EnvConfig, HubEnvConfig


@EnvConfig.register_subclass("dexjoco")
@dataclass
class DexJoCoEnvConfig(HubEnvConfig):
    """Use local code when hub_path is None, otherwise load the Hub entry point."""

    render_mode: Literal["rgb_array", "human", "none"] = "rgb_array"
    randomize: bool = False
    randomize_dynamics: bool = False
    # offsamples > 0 may cause pixels unreproducible
    offsamples: int | None = None  # None uses the XML setting.

    render_width: int = 640
    render_height: int = 640

    def __post_init__(self) -> None:
        self.features = {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(22,)),
            "agent_pos": PolicyFeature(type=FeatureType.STATE, shape=(23,)),
        }
        self.features_map = {
            "action": "action",
            "agent_pos": "observation.state",
        }
        if self.render_mode != "none":
            for camera in ("front", "wrist"):
                key = f"pixels/{camera}"
                self.features[key] = PolicyFeature(
                    type=FeatureType.VISUAL,
                    shape=(self.render_height, self.render_width, 3),
                )
                self.features_map[key] = f"observation.images.{camera}"

    @property
    def gym_kwargs(self) -> dict:
        return {
            "render_mode": self.render_mode,
            "randomize": self.randomize,
            "randomize_dynamics": self.randomize_dynamics,
            "offsamples": self.offsamples,
            "render_width": self.render_width,
            "render_height": self.render_height,
        }

    def create_envs(
        self,
        n_envs: int,
        use_async_envs: bool = False,
    ) -> dict[str, dict[int, gym.vector.VectorEnv]]:
        from dexjoco.envhub import make_env

        return make_env(n_envs=n_envs, use_async_envs=use_async_envs, cfg=self)
