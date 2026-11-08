"""Exercise photograph restoration, action layouts, screen capture, and viewer reuse."""

from unittest.mock import Mock

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dexjoco.envs.bimanual_photograph import (
    FLASH_FRAMES,
    BimanualPhotographEnv,
)


@pytest.mark.parametrize("randomize", [False, True])
def test_reset_and_action_formats(randomize: bool) -> None:
    rotvec = BimanualPhotographEnv(
        render_mode="none",
        action_format="rotvec",
        randomize=randomize,
        randomize_dynamics=True,
        camera_screen_effect=True,
    )
    quaternion = BimanualPhotographEnv(
        render_mode="none",
        action_format="quaternion",
        randomize=randomize,
        randomize_dynamics=True,
        camera_screen_effect=True,
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
            "orig_camera_pose",
            "orig_logo_pose",
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
        assert a[1:4] == b[1:4]
        assert a[4].keys() == b[4].keys()
        for key in a[4]:
            np.testing.assert_allclose(a[4][key], b[4][key], atol=1e-7)
    finally:
        rotvec.close()
        quaternion.close()


def test_camera_screen_capture_and_reset() -> None:
    env = BimanualPhotographEnv(render_width=64, render_height=48)
    try:
        env.reset(seed=12)
        assert env.captured_image is None
        env.latest_shutter_pressed = True
        env.update_screen_texture()
        assert env.captured_image is not None
        captured = env.captured_image.copy()
        size = env.screen_width * env.screen_height * env.screen_channels
        texture = env.model.tex_data[env.screen_address : env.screen_address + size]
        np.testing.assert_array_equal(texture, np.full(size, 255, dtype=np.uint8))
        env.latest_shutter_pressed = False
        for _ in range(FLASH_FRAMES):
            env.update_screen_texture()
        np.testing.assert_array_equal(env.captured_image, captured)
        np.testing.assert_array_equal(texture, captured.reshape(-1))
        assert env.flash_frame == FLASH_FRAMES
        env.reset(seed=13)
        assert env.captured_image is None and env.flash_frame == 0
    finally:
        env.close()


def test_human_viewer_creation_and_reuse(monkeypatch: pytest.MonkeyPatch) -> None:
    from mujoco import viewer

    handle = Mock()
    handle.opt.geomgroup = np.zeros(6, dtype=np.uint8)
    handle.cam.type = mujoco.mjtCamera.mjCAMERA_FREE  # pyright: ignore[reportAttributeAccessIssue]
    launch = Mock(return_value=handle)
    monkeypatch.setattr(viewer, "launch_passive", launch)
    env = BimanualPhotographEnv(render_mode="human", camera_screen_effect=False)
    try:
        env.sync_human_viewer()
        assert handle.opt.geomgroup[5] == 1
        handle.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED  # pyright: ignore[reportAttributeAccessIssue]
        env.sync_human_viewer()
        assert handle.opt.geomgroup[5] == 0
        launch.assert_called_once_with(env.model, env.data)
        assert handle.sync.call_count == 2
        assert env.renderer is None
    finally:
        env.close()
