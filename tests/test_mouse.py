"""Exercise mouse restoration, action formats, and click outcomes."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dexjoco.envs.click_mouse import (
    CLICK_THRESHOLD,
    MAX_EPISODE_STEPS,
    MOUSE_FINGER_HOME,
    MOUSEPAD_OFFSET,
    SUCCESS_STEPS,
    ClickMouseEnv,
    ClickMouseStateMachine,
)


@pytest.mark.parametrize("randomize", [False, True])
def test_reset_and_action_formats(randomize: bool) -> None:
    rotvec = ClickMouseEnv(render_mode="none", randomize=randomize, randomize_dynamics=True)
    quaternion = ClickMouseEnv(
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
        np.testing.assert_array_equal(rotvec.model.cam_pos, quaternion.model.cam_pos)
        np.testing.assert_array_equal(rotvec.data.qpos[rotvec.finger_qpos_ids], MOUSE_FINGER_HOME)
        np.testing.assert_array_equal(rotvec.data.ctrl[rotvec.finger_actuator_ids], MOUSE_FINGER_HOME)
        assert set(restored_info["initial_state"]) == {"orig_mouse_pose", "table_delta_height"}
        table_z = rotvec.data.site_xpos[rotvec.table_site_id, 2]
        mouse_pose = info["initial_state"]["orig_mouse_pose"]
        np.testing.assert_array_equal(
            rotvec.model.body_pos[rotvec.mousepad_body_id],
            np.array([mouse_pose[0], mouse_pose[1], table_z]) + MOUSEPAD_OFFSET,
        )
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
        quaternion.model.mat_rgba[quaternion.display_material_id] = [0.0, 0.0, 1.0, 1.0]
        quaternion.state_machine.update(0.002, True)
        again, _ = quaternion.reset(seed=12)
        np.testing.assert_array_equal(again["agent_pos"], obs["agent_pos"])
        np.testing.assert_array_equal(
            quaternion.model.mat_rgba[quaternion.display_material_id], quaternion.orig_display_rgba
        )
        assert not quaternion.state_machine.display_blue
        assert quaternion.state_machine.success_steps == 0
    finally:
        rotvec.close()
        quaternion.close()


def test_click_latch_and_sustained_occupancy() -> None:
    machine = ClickMouseStateMachine()
    machine.update(0.002, False)
    machine.update(CLICK_THRESHOLD, True)
    assert not machine.display_blue
    assert machine.update(0.002, True) == (False, False, False)
    assert machine.display_blue
    machine.update(0.0, False)
    assert machine.success_steps == 0
    assert machine.display_blue
    for _ in range(SUCCESS_STEPS - 1):
        assert machine.update(0.0, True) == (False, False, False)
    assert machine.update(0.0, True) == (True, True, False)
    machine.reset()
    for _ in range(MAX_EPISODE_STEPS - 1):
        assert machine.update(0.0, False) == (False, False, False)
    assert machine.update(0.0, False) == (False, False, True)
