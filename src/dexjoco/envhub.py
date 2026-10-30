"""Create the DexJoCo task suite using the LeRobot EnvHub interface."""

from collections.abc import Callable
from functools import partial
from typing import Any

import gymnasium as gym

from dexjoco.configs import DexJoCoEnvConfig
from dexjoco.envs.bimanual_assembly import BimanualAssemblyEnv
from dexjoco.envs.bimanual_hanoi import BimanualHanoiEnv
from dexjoco.envs.bimanual_microwave_cook import BimanualMicrowaveCookEnv
from dexjoco.envs.click_mouse import ClickMouseEnv
from dexjoco.envs.fold_glasses import FoldGlassesEnv
from dexjoco.envs.hammer_nail import HammerNailEnv
from dexjoco.envs.pick_bucket import PickBucketEnv
from dexjoco.envs.pinch_tongs import PinchTongsEnv
from dexjoco.envs.water_plant import WaterPlantEnv

# Stable task IDs mapped to constructors of fresh environments.
# IDs follow the task overview figure in DexJoCo.pdf, reading each row left to right.
TASKS: dict[int, Callable[..., gym.Env]] = {
    0: WaterPlantEnv,
    1: FoldGlassesEnv,
    2: ClickMouseEnv,
    3: PinchTongsEnv,
    4: PickBucketEnv,
    5: HammerNailEnv,
    6: BimanualMicrowaveCookEnv,
    8: BimanualHanoiEnv,
    9: BimanualAssemblyEnv,
}
TASK_IDS = {
    "water_plant": 0,
    "fold_glasses": 1,
    "click_mouse": 2,
    "pinch_tongs": 3,
    "pick_bucket": 4,
    "hammer_nail": 5,
    "bimanual_microwave_cook": 6,
    "bimanual_hanoi": 8,
    "bimanual_assembly": 9,
}


def make_env(
    cfg: DexJoCoEnvConfig,
    n_envs: int = 1,
    use_async_envs: bool = False,
) -> dict[str, dict[int, gym.vector.VectorEnv]]:
    """Create n_envs independent instances of the selected task.

    The caller owns the returned vector environments and must close each one.
    Task constructors must defer graphics initialization until rendering.
    Async callers must use an ``if __name__ == "__main__":`` entry point.
    ``cfg.randomize`` enables paper rand-full; ``cfg.randomize_dynamics`` enables
    task-specific dynamics randomization.
    ``cfg.render_mode="none"`` disables image observations and graphics resources.
    The configuration must explicitly select a task.
    """
    if n_envs < 1:
        raise ValueError("`n_envs` must be at least 1")

    vector_cls = gym.vector.AsyncVectorEnv if use_async_envs else gym.vector.SyncVectorEnv
    # Spawn workers with fresh EGL state, even if the parent has already rendered.
    vector_kwargs: dict[str, Any] = {"context": "spawn"} if use_async_envs else {}
    tasks: dict[int, gym.vector.VectorEnv] = {}
    try:
        task_id = TASK_IDS[cfg.task]
        constructor = TASKS[task_id]
        tasks[task_id] = vector_cls(
            [partial(
                constructor,
                randomize=cfg.randomize,
                randomize_dynamics=cfg.randomize_dynamics,
                render_mode=cfg.render_mode,
                offsamples=cfg.offsamples,
                render_width=cfg.render_width,
                render_height=cfg.render_height,
            ) for _ in range(n_envs)],
            **vector_kwargs,
        )
    except Exception:
        for env in tasks.values():
            env.close()
        raise

    return {"dexjoco": tasks}
