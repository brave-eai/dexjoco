"""Exercise tongs restoration, action formats, and task outcomes."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dexjoco.envs.pinch_tongs import (
    LIFT_HEIGHT,
    MAX_EPISODE_STEPS,
    REQUIRED_PINCHES,
    SUCCESS_STEPS,
    PinchTongsEnv,
    PinchTongsStateMachine,
)


@pytest.mark.parametrize("randomize", [False, True])
def test_reset_and_action_formats(randomize: bool) -> None:
    rotvec = PinchTongsEnv(render_mode="none", randomize=randomize, randomize_dynamics=True)
    quaternion = PinchTongsEnv(
        render_mode="none", action_format="quaternion", randomize=randomize, randomize_dynamics=True
    )
    try:
        obs, info = quaternion.reset(seed=12)
        original_qpos = quaternion.data.qpos.copy()
        restored, restored_info = rotvec.reset(
            seed=12, options={"initial_state": info["initial_state"], "unrelated_option": True}
        )
        np.testing.assert_array_equal(restored["agent_pos"], obs["agent_pos"])
        np.testing.assert_array_equal(rotvec.data.qpos, original_qpos)
        np.testing.assert_array_equal(rotvec.model.body_mass, quaternion.model.body_mass)
        np.testing.assert_array_equal(rotvec.model.cam_pos, quaternion.model.cam_pos)
        assert set(restored_info["initial_state"]) == {"orig_tongs_pose", "table_delta_height"}
        assert rotvec.lift_z == quaternion.lift_z
        assert rotvec.lift_z == 0.92 + info["initial_state"]["table_delta_height"] + LIFT_HEIGHT
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
        again, _ = quaternion.reset(seed=12)
        np.testing.assert_array_equal(again["agent_pos"], obs["agent_pos"])
        assert quaternion.state_machine.pinch_count == quaternion.state_machine.success_steps == 0
    finally:
        rotvec.close()
        quaternion.close()


def test_pinch_cycles_and_sustained_lift() -> None:
    machine = PinchTongsStateMachine()
    # A close alone, or repeated closed readings, is not a new cycle.
    machine.update(-0.1, True)
    assert machine.pinch_count == 0
    for i in range(REQUIRED_PINCHES):
        machine.update(0.15, False)
        machine.update(0.0, False)
        machine.update(-0.1, False)
        machine.update(-0.1, False)
        assert machine.pinch_count == i + 1
    for _ in range(SUCCESS_STEPS - 1):
        assert machine.update(0.0, True) == (False, False, False)
    machine.update(0.0, False)
    assert machine.success_steps == 0
    for _ in range(SUCCESS_STEPS - 1):
        assert machine.update(0.0, True) == (False, False, False)
    assert machine.update(0.0, True) == (True, True, False)
    machine.reset()
    for _ in range(MAX_EPISODE_STEPS - 1):
        assert machine.update(0.0, False) == (False, False, False)
    assert machine.update(0.0, False) == (False, False, True)
