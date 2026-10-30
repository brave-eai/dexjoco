"""Exercise glasses restoration, action formats, and folding outcomes."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dexjoco.envs.fold_glasses import (
    FOLD_THRESHOLD,
    MAX_EPISODE_STEPS,
    SUCCESS_STEPS,
    FoldGlassesEnv,
    FoldGlassesStateMachine,
)


@pytest.mark.parametrize("randomize", [False, True])
def test_reset_and_action_formats(randomize: bool) -> None:
    rotvec = FoldGlassesEnv(render_mode="none", randomize=randomize, randomize_dynamics=True)
    quaternion = FoldGlassesEnv(
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
            "orig_glass_pose", "orig_box_pose", "table_delta_height"
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
        quaternion.state_machine.update(1.2, 1.2, True)
        again, again_info = quaternion.reset(seed=12)
        np.testing.assert_array_equal(again["agent_pos"], obs["agent_pos"])
        for key in info["initial_state"]:
            np.testing.assert_array_equal(again_info["initial_state"][key], info["initial_state"][key])
        assert quaternion.state_machine.success_steps == 0
    finally:
        rotvec.close()
        quaternion.close()


def test_folded_glasses_and_sustained_placement() -> None:
    machine = FoldGlassesStateMachine()
    assert machine.update(FOLD_THRESHOLD, 1.2, True) == (False, False, False)
    assert machine.update(1.2, FOLD_THRESHOLD, True) == (False, False, False)
    assert machine.update(1.2, 1.2, False) == (False, False, False)
    for _ in range(SUCCESS_STEPS - 1):
        assert machine.update(1.2, 1.2, True) == (False, False, False)
    machine.update(1.2, 0.5, True)
    assert machine.success_steps == 0
    for _ in range(SUCCESS_STEPS - 1):
        assert machine.update(1.2, 1.2, True) == (False, False, False)
    assert machine.update(1.2, 1.2, True) == (True, True, False)
    machine.reset()
    for _ in range(MAX_EPISODE_STEPS - 1):
        assert machine.update(0.0, 0.0, False) == (False, False, False)
    assert machine.update(0.0, 0.0, False) == (False, False, True)
