"""EnvHub entry point. Install this project's package before loading it."""

from typing import Any

from dexjoco.configs import DexJoCoEnvConfig


def make_env(
    n_envs: int = 1,
    use_async_envs: bool = False,
    *,
    task: str = "water_plant",
    **options: Any,
):
    """Build the requested task, defaulting to water_plant with fixed visuals."""
    cfg = DexJoCoEnvConfig(task=task, **options)
    return cfg.create_envs(n_envs=n_envs, use_async_envs=use_async_envs)

__all__ = ["make_env"]
