"""Exercise Hanoi scene restoration, action layouts, and stable outcomes."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dexjoco.envs.bimanual_hanoi import (
    ILLEGAL_STEPS,
    MAX_EPISODE_STEPS,
    SUCCESS_STEPS,
    BimanualHanoiEnv,
    BimanualHanoiStateMachine,
)


def test_reset_and_action_formats() -> None:
    rotvec = BimanualHanoiEnv(render_mode="none")
    quaternion = BimanualHanoiEnv(render_mode="none", action_format="quaternion")
    try:
        obs, info = quaternion.reset(seed=12)
        original_qpos = quaternion.data.qpos.copy()
        restored, restored_info = rotvec.reset(
            seed=99, options={"initial_state": info["initial_state"], "unrelated_option": True}
        )
        np.testing.assert_array_equal(restored["agent_pos"], obs["agent_pos"])
        np.testing.assert_array_equal(rotvec.data.qpos, original_qpos)
        assert set(restored_info["initial_state"]) == {"orig_hanoi_base_pos", "table_delta_height"}
        assert set(obs) == {"agent_pos"}
        assert quaternion.renderer is None and rotvec.renderer is None
        action_quat = obs["agent_pos"].astype(np.float64)
        action_rotvec = np.concatenate((
            action_quat[:3],
            Rotation.from_quat(action_quat[3:7], scalar_first=True).as_rotvec(),
            action_quat[14:30],
            action_quat[7:10],
            Rotation.from_quat(action_quat[10:14], scalar_first=True).as_rotvec(),
            action_quat[30:46],
        ))
        assert rotvec.action_space.contains(action_rotvec.astype(np.float32))
        assert quaternion.action_space.contains(action_quat.astype(np.float32))
        a = rotvec.step(action_rotvec)
        b = quaternion.step(action_quat)
        np.testing.assert_allclose(a[0]["agent_pos"], b[0]["agent_pos"], atol=1e-7)
        assert a[1:] == b[1:]
    finally:
        rotvec.close()
        quaternion.close()


@pytest.mark.parametrize("outcome", ["success", "illegal_stack", "disk_on_table", "timeout"])
def test_stable_outcomes(outcome: str) -> None:
    machine = BimanualHanoiStateMachine()
    posts = np.array([[0.0, 0.0, 1.0], [0.0, 0.2, 1.0], [0.0, 0.4, 1.0]])
    heights = np.full(3, 0.025)
    positions = np.array([[0.0, 0.4, 1.025], [0.0, 0.4, 1.075], [0.0, 0.4, 1.125]])
    velocities = np.zeros((3, 6))
    steps = SUCCESS_STEPS
    if outcome == "illegal_stack":
        positions[[0, 2]] = positions[[2, 0]]
        steps = ILLEGAL_STEPS
    elif outcome == "disk_on_table":
        positions[0] = [0.2, 0.4, 1.025]
        steps = ILLEGAL_STEPS
    elif outcome == "timeout":
        velocities[:] = 1.0
        steps = MAX_EPISODE_STEPS
    for _ in range(steps - 1):
        pending = machine.update(positions, velocities, posts, 1.0, heights)
        assert not any(pending[:3])
    success, terminated, truncated, info = machine.update(positions, velocities, posts, 1.0, heights)
    assert success == (outcome == "success")
    assert truncated == (outcome == "timeout")
    assert terminated == (outcome != "timeout")
    assert info["illegal_state"] == (outcome in ("illegal_stack", "disk_on_table"))
    machine.reset()
    assert machine.step_count == machine.success_steps == machine.illegal_steps == 0
