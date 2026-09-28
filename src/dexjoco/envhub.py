"""Create the DexJoCo task suite using the LeRobot EnvHub interface."""

from collections.abc import Callable
from functools import partial

import gymnasium as gym

from dexjoco.envs.water_plant import WaterPlantEnv

# Stable task IDs mapped to constructors of fresh environments.
TASKS: dict[int, Callable[..., gym.Env]] = {0: WaterPlantEnv}


def make_env(
    n_envs: int = 1,
    use_async_envs: bool = False,
    randomize: bool = False,
    randomize_dynamics: bool = False,
) -> dict[str, dict[int, gym.vector.VectorEnv]]:
    """Create n_envs independent instances of each implemented task.

    The caller owns the returned vector environments and must close each one.
    Task constructors must defer graphics initialization until rendering.
    Async callers must use an ``if __name__ == "__main__":`` entry point.
    ``randomize`` enables paper rand-full; ``randomize_dynamics`` enables
    spray joint friction, stiffness, and body mass randomization.
    """
    if n_envs < 1:
        raise ValueError("`n_envs` must be at least 1")

    vector_cls = gym.vector.AsyncVectorEnv if use_async_envs else gym.vector.SyncVectorEnv
    # Spawn workers with fresh EGL state, even if the parent has already rendered.
    vector_kwargs = {"context": "spawn"} if use_async_envs else {}
    tasks = {}
    try:
        for task_id, constructor in TASKS.items():
            tasks[task_id] = vector_cls(
                [partial(constructor, randomize=randomize, randomize_dynamics=randomize_dynamics)
                 for _ in range(n_envs)],
                autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
                **vector_kwargs,
            )
    except Exception:
        for env in tasks.values():
            env.close()
        raise

    return {"dexjoco": tasks}
