"""Exercise non-default initial hand poses saved with the scene state."""

import numpy as np
import pytest

from dexjoco.envs.bimanual_microwave_cook import BimanualMicrowaveCookEnv
from dexjoco.envs.bimanual_photograph import BimanualPhotographEnv


@pytest.mark.parametrize("env_class", [BimanualMicrowaveCookEnv, BimanualPhotographEnv])
def test_scene_hand_pose_roundtrip(env_class):
    env = env_class(render_mode="none")
    try:
        _, info = env.reset(seed=4)
        initial = info["initial_state"]
        pose = np.linspace(0.0, 0.3, 32, dtype=np.float64)
        initial["initial_gripper_pose"] = pose
        _, restored = env.reset(options={"initial_state": initial})
        np.testing.assert_array_equal(env.data.qpos[env.finger_qpos_ids], pose)
        np.testing.assert_array_equal(restored["initial_state"]["initial_gripper_pose"], pose)
        pose[:] = -1
        assert np.all(env.data.qpos[env.finger_qpos_ids] >= 0)
    finally:
        env.close()
