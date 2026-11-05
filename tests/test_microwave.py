"""Exercise microwave restoration and action formats."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dexjoco.envs.bimanual_microwave_cook import BimanualMicrowaveCookEnv


@pytest.mark.parametrize("randomize", [False, True])
def test_reset_and_action_formats(randomize: bool) -> None:
    rotvec = BimanualMicrowaveCookEnv(
        render_mode="none", action_format="rotvec", randomize=randomize, randomize_dynamics=True
    )
    quaternion = BimanualMicrowaveCookEnv(
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
        assert set(restored_info["initial_state"]) == {
            "orig_hot_dog_pose",
            "orig_microwave_pose",
            "table_delta_height",
            "initial_hand_joint",
        }
        np.testing.assert_array_equal(rotvec.model.body_pos, quaternion.model.body_pos)
        np.testing.assert_array_equal(rotvec.model.body_quat, quaternion.model.body_quat)
        np.testing.assert_array_equal(rotvec.model.body_mass, quaternion.model.body_mass)
        np.testing.assert_array_equal(
            rotvec.model.dof_frictionloss, quaternion.model.dof_frictionloss
        )
        np.testing.assert_array_equal(rotvec.model.cam_pos, quaternion.model.cam_pos)
        np.testing.assert_array_equal(rotvec.model.geom_pos, quaternion.model.geom_pos)
        np.testing.assert_array_equal(rotvec.model.geom_size, quaternion.model.geom_size)
        assert set(obs) == {"agent_pos"}
        assert quaternion.renderer is None and rotvec.renderer is None
        assert quaternion.observation_space.contains(obs)
        for i, side in enumerate(("right", "left")):
            offset = i * 23
            np.testing.assert_array_equal(
                obs["agent_pos"][offset : offset + 3],
                quaternion.data.sensor(f"franka/flange_pos_{side}").data.astype(np.float32),
            )
            np.testing.assert_array_equal(
                obs["agent_pos"][offset + 3 : offset + 7],
                quaternion.data.sensor(f"franka/flange_quat_{side}").data.astype(np.float32),
            )
            np.testing.assert_array_equal(
                obs["agent_pos"][offset + 7 : offset + 23],
                np.array(
                    [
                        quaternion.data.sensor(sensor).data[0]
                        for sensor in quaternion.finger_sensor_ids[i * 16 : (i + 1) * 16]
                    ],
                    dtype=np.float32,
                ),
            )
        action_quat = obs["agent_pos"].astype(np.float64)
        action_rotvec = np.concatenate(
            (
                action_quat[:3],
                Rotation.from_quat(action_quat[3:7], scalar_first=True).as_rotvec(),
                action_quat[7:23],
                action_quat[23:26],
                Rotation.from_quat(action_quat[26:30], scalar_first=True).as_rotvec(),
                action_quat[30:46],
            )
        )
        assert rotvec.action_space.contains(action_rotvec.astype(np.float32))
        assert quaternion.action_space.contains(action_quat.astype(np.float32))
        a = rotvec.step(action_rotvec)
        b = quaternion.step(action_quat)
        np.testing.assert_allclose(a[0]["agent_pos"], b[0]["agent_pos"], atol=1e-7)
        assert a[1:] == b[1:]
    finally:
        rotvec.close()
        quaternion.close()
