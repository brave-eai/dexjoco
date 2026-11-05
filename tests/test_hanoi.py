"""Exercise Hanoi restoration and action formats."""

import numpy as np
from scipy.spatial.transform import Rotation

from dexjoco.envs.bimanual_hanoi import BimanualHanoiEnv


def test_reset_and_action_formats() -> None:
    rotvec = BimanualHanoiEnv(render_mode="none", action_format="rotvec")
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
        action_quat = np.concatenate((
            action_quat[:7], action_quat[14:30], action_quat[7:14], action_quat[30:46],
        ))
        assert quaternion.action_space.contains(action_quat.astype(np.float32))
        a = rotvec.step(action_rotvec)
        b = quaternion.step(action_quat)
        np.testing.assert_allclose(a[0]["agent_pos"], b[0]["agent_pos"], atol=1e-7)
        assert a[1:] == b[1:]
    finally:
        rotvec.close()
        quaternion.close()
