"""Exercise bucket restoration, action formats, and lifting outcomes."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dexjoco.envs.pick_bucket import (
    LIFT_HEIGHT,
    MAX_EPISODE_STEPS,
    PickBucketEnv,
    PickBucketStateMachine,
)


@pytest.mark.parametrize("randomize", [False, True])
def test_reset_and_action_formats(randomize: bool) -> None:
    rotvec = PickBucketEnv(render_mode="none", randomize=randomize, randomize_dynamics=True)
    quaternion = PickBucketEnv(
        render_mode="none", action_format="quaternion", randomize=randomize, randomize_dynamics=True
    )
    try:
        obs, info = quaternion.reset(seed=12)
        restored, restored_info = rotvec.reset(
            seed=12, options={"initial_state": info["initial_state"], "unrelated_option": True}
        )
        np.testing.assert_array_equal(restored["agent_pos"], obs["agent_pos"])
        np.testing.assert_array_equal(rotvec.data.qpos, quaternion.data.qpos)
        np.testing.assert_array_equal(rotvec.model.body_mass, quaternion.model.body_mass)
        np.testing.assert_array_equal(rotvec.model.dof_frictionloss, quaternion.model.dof_frictionloss)
        np.testing.assert_array_equal(rotvec.model.cam_pos, quaternion.model.cam_pos)
        np.testing.assert_array_equal(rotvec.initial_bottom_heights, quaternion.initial_bottom_heights)
        np.testing.assert_array_equal(
            rotvec.initial_bottom_heights, rotvec.data.site_xpos[rotvec.bucket_bottom_site_ids, 2]
        )
        assert set(restored_info["initial_state"]) == {
            "orig_bucket_pose", "orig_boxed_food_pose", "table_delta_height"
        }
        assert set(obs) == {"agent_pos"}
        assert quaternion.renderer is None and rotvec.renderer is None
        action_quat = obs["agent_pos"].astype(np.float64)
        action_rotvec = np.concatenate((
            action_quat[:3],
            Rotation.from_quat(action_quat[3:7], scalar_first=True).as_rotvec(),
            action_quat[7:],
        ))
        assert rotvec.action_space.contains(action_rotvec.astype(np.float32))
        assert quaternion.action_space.contains(action_quat.astype(np.float32))
        a = rotvec.step(action_rotvec)
        b = quaternion.step(action_quat)
        np.testing.assert_allclose(a[0]["agent_pos"], b[0]["agent_pos"], atol=1e-7)
        assert a[1:] == b[1:]
        again, again_info = quaternion.reset(seed=12)
        np.testing.assert_array_equal(again["agent_pos"], obs["agent_pos"])
        for key in info["initial_state"]:
            np.testing.assert_array_equal(again_info["initial_state"][key], info["initial_state"][key])
        assert quaternion.state_machine.step_count == 0
    finally:
        rotvec.close()
        quaternion.close()


@pytest.mark.parametrize("outcome", ["success", "food_outside", "one_corner_low", "timeout"])
def test_lifting_outcomes(outcome: str) -> None:
    machine = PickBucketStateMachine()
    corners = np.array([
        [x, y, z] for x in (-0.1, 0.1) for y in (-0.1, 0.1)
        for z in (LIFT_HEIGHT, LIFT_HEIGHT + 0.2)
    ])
    initial_bottom_heights = np.zeros(4)
    food_position = np.array([0.0, 0.0, LIFT_HEIGHT + 0.1])
    if outcome == "food_outside":
        food_position[0] = 0.2
    elif outcome == "one_corner_low":
        corners[0, 2] = LIFT_HEIGHT - 0.01
    elif outcome == "timeout":
        initial_bottom_heights[:] = LIFT_HEIGHT
        for _ in range(MAX_EPISODE_STEPS - 1):
            assert machine.update(food_position, corners, initial_bottom_heights) == (False, False, False)
    success, terminated, truncated = machine.update(food_position, corners, initial_bottom_heights)
    assert success == terminated == (outcome == "success")
    assert truncated == (outcome == "timeout")
    machine.reset()
    assert machine.step_count == 0
