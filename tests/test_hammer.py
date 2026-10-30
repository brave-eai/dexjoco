"""Exercise hammer restoration, action formats, and insertion outcomes."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dexjoco.envs.hammer_nail import (
    IMPACT_VELOCITY,
    INSERT_STEP,
    MAX_DEPTH,
    MAX_EPISODE_STEPS,
    HammerNailEnv,
    HammerNailStateMachine,
)


@pytest.mark.parametrize("randomize", [False, True])
def test_reset_and_action_formats(randomize: bool) -> None:
    rotvec = HammerNailEnv(render_mode="none", randomize=randomize, randomize_dynamics=True)
    quaternion = HammerNailEnv(
        render_mode="none", action_format="quaternion", randomize=randomize, randomize_dynamics=True
    )
    try:
        obs, info = quaternion.reset(seed=12)
        restored, restored_info = rotvec.reset(
            seed=12, options={"initial_state": info["initial_state"], "unrelated_option": True}
        )
        np.testing.assert_array_equal(restored["agent_pos"], obs["agent_pos"])
        np.testing.assert_array_equal(rotvec.data.qpos, quaternion.data.qpos)
        np.testing.assert_array_equal(rotvec.model.body_pos, quaternion.model.body_pos)
        np.testing.assert_array_equal(rotvec.model.body_quat, quaternion.model.body_quat)
        np.testing.assert_array_equal(rotvec.model.body_mass, quaternion.model.body_mass)
        np.testing.assert_array_equal(rotvec.model.dof_frictionloss, quaternion.model.dof_frictionloss)
        np.testing.assert_array_equal(rotvec.model.cam_pos, quaternion.model.cam_pos)
        assert set(restored_info["initial_state"]) == {
            "orig_hammer_pose", "orig_nail_pose", "table_delta_height"
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
        quaternion.state_machine.update(True, -0.05)
        again, again_info = quaternion.reset(seed=12)
        np.testing.assert_array_equal(again["agent_pos"], obs["agent_pos"])
        for key in info["initial_state"]:
            np.testing.assert_array_equal(again_info["initial_state"][key], info["initial_state"][key])
        assert quaternion.state_machine.depth == 0.0
    finally:
        rotvec.close()
        quaternion.close()


def test_contact_insertion_and_timeout() -> None:
    machine = HammerNailStateMachine()
    assert machine.update(False, -0.1) == (False, False, False)
    assert machine.depth == 0.0
    assert machine.update(True, -IMPACT_VELOCITY) == (False, False, False)
    assert machine.depth == 0.0
    assert machine.update(True, -IMPACT_VELOCITY * 2) == (False, False, False)
    assert machine.depth == INSERT_STEP * 2
    assert machine.update(True, -IMPACT_VELOCITY * 3) == (True, True, False)
    for _ in range(10):
        machine.update(True, -1.0)
    assert machine.depth == MAX_DEPTH
    machine.reset()
    for _ in range(MAX_EPISODE_STEPS - 1):
        assert machine.update(False, 0.0) == (False, False, False)
    assert machine.update(False, 0.0) == (False, False, True)
