"""Check seeded restoration and episode-based batching without stepping physics."""

from typing import Any, cast

import gymnasium as gym
import numpy as np
import pytest

from dexjoco.env_factory import TASK_IDS, TASKS
from dexjoco.envs.water_plant import WaterPlantEnv
from dexjoco.replay_dataset import batch_env_reset


@pytest.mark.parametrize("task", list(TASK_IDS))
def test_seed_preserves_restored_scene(task: str) -> None:
    snapshots = []
    for seed in (0, 37, 0):
        env: Any = TASKS[TASK_IDS[task]](
            render_mode="none", randomize=True
        )
        try:
            _, info = env.reset(seed=12)
            initial = info["initial_state"]
            obs, restored = env.reset(
                seed=seed, options={"initial_state": initial}
            )
            for key, value in initial.items():
                np.testing.assert_array_equal(restored["initial_state"][key], value)
            snapshots.append(
                {
                    "state": obs["agent_pos"].copy(),
                    "qpos": env.data.qpos.copy(),
                    "body_pos": env.model.body_pos.copy(),
                    "body_quat": env.model.body_quat.copy(),
                    "body_mass": env.model.body_mass.copy(),
                    "jnt_stiffness": env.model.jnt_stiffness.copy(),
                    "friction": env.model.dof_frictionloss.copy(),
                    "camera": env.model.cam_pos.copy(),
                    "light": env.model.light_pos.copy(),
                }
            )
        finally:
            env.close()
    for key in snapshots[0]:
        np.testing.assert_array_equal(snapshots[0][key], snapshots[2][key])
        if key not in ("camera", "light"):
            np.testing.assert_array_equal(snapshots[0][key], snapshots[1][key])
    assert not np.array_equal(snapshots[0]["light"], snapshots[1]["light"])


def test_default_seed_is_independent_of_batch_size() -> None:
    reference = WaterPlantEnv(render_mode="none")
    try:
        initials = [reference.reset(seed=seed)[1]["initial_state"] for seed in (1, 2, 3)]
    finally:
        reference.close()
    results = []
    for n_envs in (1, 2):
        env = gym.vector.SyncVectorEnv(
            [lambda: WaterPlantEnv(render_mode="none", randomize=True) for _ in range(n_envs)]
        )
        try:
            visuals = []
            for start in range(0, len(initials), n_envs):
                batch = initials[start : start + n_envs]
                batch_env_reset(env, batch + [None] * (n_envs - len(batch)), episode_offset=start)
                for slot in range(len(batch)):
                    model = cast(WaterPlantEnv, env.envs[slot].unwrapped).model
                    visuals.append((model.cam_pos.copy(), model.light_pos.copy()))
            results.append(visuals)
        finally:
            env.close()
    for single, batched in zip(*results, strict=True):
        for a, b in zip(single, batched, strict=True):
            np.testing.assert_array_equal(a, b)
