"""Exercise tongs restoration and action formats."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dexjoco.envs.pinch_tongs import (
    LIFT_HEIGHT,
    PinchTongsEnv,
)


@pytest.mark.parametrize("randomize", [False, True])
def test_reset_and_action_formats(randomize: bool) -> None:
    rotvec = PinchTongsEnv(render_mode="none", action_format="rotvec", randomize=randomize, randomize_dynamics=True)
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
