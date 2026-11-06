"""DexJoCo configuration for local and Hugging Face EnvHub loading."""

from dataclasses import dataclass, field
from typing import Literal

import gymnasium as gym
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.envs.configs import EnvConfig, HubEnvConfig


@EnvConfig.register_subclass("dexjoco")
@dataclass
class DexJoCoEnvConfig(HubEnvConfig):
    """Use local code when hub_path is None, otherwise load the Hub entry point."""

    # This plugin requires a non-null task, while EnvConfig also allows None.
    task: str = field(kw_only=True)  # pyright: ignore[reportIncompatibleVariableOverride]
    bimanual: bool | None = None
    action_format: Literal["quaternion", "rotvec"] = "quaternion"
    render_mode: Literal["rgb_array", "human", "none"] = "rgb_array"
    randomize: bool = False
    randomize_dynamics: bool = False
    # offsamples > 0 may cause pixels unreproducible
    offsamples: int | None = None  # None uses the XML setting.

    render_width: int = 640
    render_height: int = 640

    def __post_init__(self) -> None:
        from dexjoco.envhub import TASK_IDS, TASKS

        constructor = TASKS[TASK_IDS[self.task]]
        task_bimanual = self.task.startswith("bimanual_")
        if self.bimanual is not None and self.bimanual != task_bimanual:
            raise ValueError(f"bimanual={self.bimanual} conflicts with task {self.task!r}")
        self.bimanual = task_bimanual
        arm_action_dim = 23 if self.action_format == "quaternion" else 22
        self.features = {
            "action": PolicyFeature(
                type=FeatureType.ACTION, shape=(arm_action_dim * (2 if self.bimanual else 1),)
            ),
            "agent_pos": PolicyFeature(type=FeatureType.STATE, shape=(46 if self.bimanual else 23,)),
        }
        self.features_map = {
            "action": "action",
            "agent_pos": "observation.state",
        }
        if self.render_mode != "none":
            cameras = (
                "ego",
                *getattr(constructor, "side_cameras", ()),
                *(("wrist_left", "wrist_right") if self.bimanual else ("wrist",)),
            )
            for camera in cameras:
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
            "action_format": self.action_format,
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
