"""Create the DexJoCo task suite using the LeRobot EnvHub interface."""

from collections.abc import Callable
from functools import partial
from typing import Any

import gymnasium as gym

from dexjoco.configs import DexJoCoEnvConfig
from dexjoco.envs.water_plant import WaterPlantEnv

# Stable task IDs mapped to constructors of fresh environments.
TASKS: dict[int, Callable[..., gym.Env]] = {0: WaterPlantEnv}


def make_env(
    n_envs: int = 1,
    use_async_envs: bool = False,
    cfg: DexJoCoEnvConfig | None = None,
) -> dict[str, dict[int, gym.vector.VectorEnv]]:
    """Create n_envs independent instances of each implemented task.

    The caller owns the returned vector environments and must close each one.
    Task constructors must defer graphics initialization until rendering.
    Async callers must use an ``if __name__ == "__main__":`` entry point.
    ``cfg.randomize`` enables paper rand-full; ``cfg.randomize_dynamics`` enables
    spray joint friction, stiffness, and body mass randomization.
    ``cfg.render_mode="none"`` disables image observations and graphics resources.
    Omitting ``cfg`` uses the default DexJoCo configuration.
    """
    if cfg is None:
        cfg = DexJoCoEnvConfig()

    if n_envs < 1:
        raise ValueError("`n_envs` must be at least 1")

    vector_cls = gym.vector.AsyncVectorEnv if use_async_envs else gym.vector.SyncVectorEnv
    # Spawn workers with fresh EGL state, even if the parent has already rendered.
    vector_kwargs: dict[str, Any] = {"context": "spawn"} if use_async_envs else {}
    tasks: dict[int, gym.vector.VectorEnv] = {}
    try:
        for task_id, constructor in TASKS.items():
            tasks[task_id] = vector_cls(
                [partial(
                    constructor,
                    randomize=cfg.randomize,
                    randomize_dynamics=cfg.randomize_dynamics,
                    render_mode=cfg.render_mode,
                ) for _ in range(n_envs)],
                **vector_kwargs,
            )
    except Exception:
        for env in tasks.values():
            env.close()
        raise

    return {"dexjoco": tasks}
