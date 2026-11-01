"""Exercise iPad restoration, password input, and unlock timing."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dexjoco.envs.bimanual_unlock_ipad import (
    MAX_EPISODE_STEPS,
    SUCCESS_STEPS,
    UNLOCK_DELAY_STEPS,
    UNLOCK_TRANSITION_STEPS,
    BimanualUnlockIpadEnv,
    BimanualUnlockIpadStateMachine,
    UnlockState,
)


@pytest.mark.parametrize("randomize", [False, True])
def test_reset_and_action_formats(randomize: bool) -> None:
    rotvec = BimanualUnlockIpadEnv(render_mode="none", randomize=randomize, randomize_dynamics=True)
    quaternion = BimanualUnlockIpadEnv(render_mode="none", action_format="quaternion", randomize=randomize, randomize_dynamics=True)
    try:
        obs, info = quaternion.reset(seed=12)
        original_qpos = quaternion.data.qpos.copy()
        restored, restored_info = rotvec.reset(
            seed=12, options={"initial_state": info["initial_state"], "unrelated_option": True}
        )
        np.testing.assert_array_equal(restored["agent_pos"], obs["agent_pos"])
        np.testing.assert_array_equal(rotvec.data.qpos, original_qpos)
        assert set(restored_info["initial_state"]) == {"orig_stand_pose", "orig_ipad_pose", "table_delta_height"}
        np.testing.assert_array_equal(rotvec.model.body_pos, quaternion.model.body_pos)
        np.testing.assert_array_equal(rotvec.model.body_quat, quaternion.model.body_quat)
        np.testing.assert_array_equal(rotvec.model.body_mass, quaternion.model.body_mass)
        np.testing.assert_array_equal(rotvec.model.dof_frictionloss, quaternion.model.dof_frictionloss)
        np.testing.assert_array_equal(rotvec.model.cam_pos, quaternion.model.cam_pos)
        np.testing.assert_array_equal(rotvec.model.geom_pos, quaternion.model.geom_pos)
        np.testing.assert_array_equal(rotvec.model.geom_size, quaternion.model.geom_size)
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


@pytest.mark.parametrize("screen_effect", [False, True])
def test_unlock_timing(screen_effect: bool) -> None:
    machine = BimanualUnlockIpadStateMachine(screen_effect=screen_effect)
    assert machine.update([1]) == (False, False, False)
    assert machine.password_index == 1
    assert machine.update([2]) == (False, False, False)
    assert machine.update([3, 4]) == (False, False, False)
    assert machine.password_index == 3
    if screen_effect:
        assert machine.state is UnlockState.ANIMATING
        assert machine.animation_progress == 0.0
        for _ in range(UNLOCK_DELAY_STEPS - 1):
            machine.update([])
            assert machine.animation_progress == 0.0
        for _ in range(UNLOCK_TRANSITION_STEPS - 1):
            machine.update([])
            assert machine.state is UnlockState.ANIMATING
            assert 0.0 < machine.animation_progress < 1.0
        machine.update([])
    assert machine.state is UnlockState.CONFIRMING
    assert machine.state_steps == 1
    assert machine.animation_progress == 1.0
    for _ in range(SUCCESS_STEPS - 2):
        assert machine.update([9]) == (False, False, False)
    assert machine.update([]) == (True, True, False)
    assert machine.state is UnlockState.SUCCEEDED
    steps = machine.step_count
    assert machine.update([9]) == (True, True, False)
    assert machine.step_count == steps


def test_failure_and_timeout() -> None:
    machine = BimanualUnlockIpadStateMachine()
    assert machine.update([1, 9, 1, 2, 3]) == (False, True, False)
    assert machine.state is UnlockState.FAILED
    assert machine.failure_reason == "wrong_digit"
    assert machine.password_index == 0
    assert machine.update([1, 2, 3]) == (False, True, False)
    machine.reset()
    assert machine.state is UnlockState.ENTERING
    assert machine.failure_reason is None
    for _ in range(MAX_EPISODE_STEPS - 1):
        assert machine.update([]) == (False, False, False)
    assert machine.update([]) == (False, False, True)
    assert machine.state is UnlockState.FAILED
    assert machine.failure_reason == "timeout"
    assert machine.update([1, 2, 3]) == (False, False, True)


def test_repeated_digits_and_status_queries(monkeypatch: pytest.MonkeyPatch) -> None:
    env = BimanualUnlockIpadEnv(render_mode="none", password=[1, 1, 0])
    try:
        env.reset(seed=7)
        pressed: set[int] = set()
        monkeypatch.setattr(env, "pressed_buttons", lambda: pressed.copy())
        action = np.zeros(44)
        pressed.add(env.button_ids[1])
        env.step(action)
        env.step(action)
        assert env.get_unlock_status()["unlock_index"] == 1
        pressed.clear()
        env.step(action)
        pressed.add(env.button_ids[1])
        env.step(action)
        assert env.get_unlock_status()["unlock_index"] == 2
        pressed.clear()
        env.step(action)
        pressed.add(env.button_ids[0])
        result = env.step(action)
        assert result[4]["unlock_state"] == "confirming"
        assert env.state_machine.state_steps == 1
        snapshot = env.get_unlock_status()
        for _ in range(20):
            assert env.get_unlock_status() == snapshot
            assert not env.state_machine.is_success
        assert env.state_machine.state_steps == 1
        assert env.renderer is None
        assert "110" in env.task_description
    finally:
        env.close()


def test_screen_animation_without_renderer(monkeypatch: pytest.MonkeyPatch) -> None:
    env = BimanualUnlockIpadEnv(render_mode="none", password=[2], ipad_screen_effect=True)
    try:
        env.reset(seed=4)
        monkeypatch.setattr(env, "pressed_buttons", lambda: {env.button_ids[2]})
        action = np.zeros(44)
        env.step(action)
        assert env.model.geom_matid[env.screen_id] == env.locked_material
        for _ in range(UNLOCK_DELAY_STEPS - 1):
            env.step(action)
        assert env.model.geom_matid[env.screen_id] == env.locked_material
        env.step(action)
        assert env.model.geom_matid[env.screen_id] == env.unlocked_material
        assert np.all(env.model.geom_rgba[env.button_ids, 3] > 0)
        assert np.all(env.model.geom_rgba[env.button_ids, 3] < env.button_rgba[:, 3])
        for _ in range(UNLOCK_TRANSITION_STEPS - 1):
            env.step(action)
        assert env.state_machine.state is UnlockState.CONFIRMING
        assert np.all(env.model.geom_rgba[env.button_ids, 3] == 0)
        assert env.renderer is None
        env.reset(seed=4)
        assert env.state_machine.state is UnlockState.ENTERING
        assert env.previous_pressed == set()
        np.testing.assert_array_equal(env.model.geom_rgba[env.button_ids], env.button_rgba)
        np.testing.assert_array_equal(env.model.mat_rgba[env.unlocked_material], env.unlocked_rgba)
        assert env.model.geom_matid[env.screen_id] == env.locked_material
    finally:
        env.close()
