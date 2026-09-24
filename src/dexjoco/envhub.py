"""Create the DexJoCo task suite using the LeRobot EnvHub interface."""

from collections.abc import Callable

import gymnasium as gym

from dexjoco.envs.water_plant import WaterPlantEnv

# Stable task IDs mapped to constructors of fresh environments.
TASKS: dict[int, Callable[[], gym.Env]] = {0: WaterPlantEnv}


def make_env(
    n_envs: int = 1,
    use_async_envs: bool = False,
) -> dict[str, dict[int, gym.vector.VectorEnv]]:
    """Create n_envs independent instances of each implemented task.

    The caller owns the returned vector environments and must close each one.
    Task constructors must defer graphics initialization until rendering.
    """
    if n_envs < 1:
        raise ValueError("`n_envs` must be at least 1")

    vector_cls = gym.vector.AsyncVectorEnv if use_async_envs else gym.vector.SyncVectorEnv
    tasks = {}
    try:
        for task_id, constructor in TASKS.items():
            tasks[task_id] = vector_cls(
                [constructor for _ in range(n_envs)],
                autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
            )
    except Exception:
        for env in tasks.values():
            env.close()
        raise

    return {"dexjoco": tasks}
