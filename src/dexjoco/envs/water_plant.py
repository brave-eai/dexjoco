"""A compact MuJoCo implementation of the Water Plant task."""

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from scipy.spatial.transform import Rotation

_ASSET_ROOT = Path(__file__).resolve().parents[1] / "assets"
_XML_PATH = _ASSET_ROOT / "tasks" / "water_plant" / "water_plant.xml"
_MAX_EPISODE_STEPS = 1000
_CONTROL_TIMESTEP = 0.02
_PHYSICS_TIMESTEP = 0.002
_SUCCESS_STEPS = 30
_TRIGGER_RELEASE = 0.25
_TRIGGER_PULL = 0.34

_ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 8))
_FINGER_JOINTS = (
    "ffj0", "ffj1", "ffj2", "ffj3",
    "mfj0", "mfj1", "mfj2", "mfj3",
    "rfj0", "rfj1", "rfj2", "rfj3",
    "thj0", "thj1", "thj2", "thj3",
)
_FINGER_ACTUATORS = (
    "ffa0", "ffa1", "ffa2", "ffa3",
    "mfa0", "mfa1", "mfa2", "mfa3",
    "rfa0", "rfa1", "rfa2", "rfa3",
    "tha0", "tha1", "tha2", "tha3",
)


class WaterPlantEnv(gym.Env):
    """Control a Panda-Allegro hand to water a plant."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}
    task = "water_plant"
    task_description = "Pick up the spray bottle and water the plant."
    _max_episode_steps = _MAX_EPISODE_STEPS

    def __init__(self, render_mode: str | None = "rgb_array", image_observations: bool = True):
        if render_mode not in (None, "rgb_array"):
            raise ValueError(f"Unsupported render mode: {render_mode}")
        self.render_mode = render_mode
        self.image_observations = image_observations
        self.model = mujoco.MjModel.from_xml_path(str(_XML_PATH))
        self.model.opt.timestep = _PHYSICS_TIMESTEP
        self.data = mujoco.MjData(self.model)
        self._renderer = None
        self._step_count = 0
        self._success_steps = 0
        self._trigger_pulled = False

        self._site_id = self.model.site("attachment_site").id
        self._plant_body_id = self.model.body("plant").id
        self._ref_site_id = self.model.site("ref_point").id
        self._cone_geom_id = self.model.geom("cone_visual").id
        self._cone_steps = 0
        self._trigger_qpos_id = self.model.joint("joint_0").qposadr[0]
        self._arm_joint_ids = np.array([self.model.joint(name).id for name in _ARM_JOINTS])
        self._arm_dof_ids = self.model.jnt_dofadr[self._arm_joint_ids]
        self._arm_actuator_ids = np.array([
            self.model.actuator(name).id
            for name in (f"actuator{i}" for i in range(1, 8))
        ])
        self._finger_joint_ids = np.array([
            self.model.joint(name).id for name in _FINGER_JOINTS
        ])
        self._finger_qpos_ids = self.model.jnt_qposadr[self._finger_joint_ids]
        self._finger_actuator_ids = np.array([
            self.model.actuator(name).id for name in _FINGER_ACTUATORS
        ])
        self._arm_home = np.array([0.0, -0.785, 0.0, -2.35, 0.0, 1.57, np.pi / 4])

        hand_limits = self.model.actuator_ctrlrange[self._finger_actuator_ids]
        self.action_space = spaces.Box(
            low=np.concatenate((np.full(6, -np.inf), hand_limits[:, 0])).astype(np.float32),
            high=np.concatenate((np.full(6, np.inf), hand_limits[:, 1])).astype(np.float32),
            dtype=np.float32,
        )
        observation_spaces = {
            "agent_pos": spaces.Box(-np.inf, np.inf, shape=(22,), dtype=np.float32),
        }
        if image_observations:
            observation_spaces["pixels"] = spaces.Dict({
                name: spaces.Box(0, 255, shape=(480, 640, 3), dtype=np.uint8)
                for name in ("front", "wrist")
            })
        self.observation_space = spaces.Dict(observation_spaces)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.model.jnt_qposadr[self._arm_joint_ids]] = self._arm_home
        self.data.qpos[self._finger_qpos_ids] = self._finger_home()
        self.data.ctrl[self._finger_actuator_ids] = self._finger_home()
        spray_qpos = self.model.joint("spray_root").qposadr[0]
        self.data.qpos[spray_qpos:spray_qpos + 2] = self.np_random.uniform(
            [-0.35, -0.25], [-0.30, -0.20]
        )
        self.model.body_pos[self._plant_body_id, :2] = self.np_random.uniform(
            [-0.10, 0.15], [-0.05, 0.20]
        )
        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0
        self._success_steps = 0
        self._trigger_pulled = False
        self._cone_steps = 0
        self.model.geom_rgba[self._cone_geom_id, 3] = 0
        return self._observation(), {"is_success": False}

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (22,) or not np.isfinite(action).all():
            raise ValueError("Expected 22 finite action values")
        target_position = action[:3]
        target_rotation = Rotation.from_rotvec(action[3:6])
        finger_targets = np.clip(
            action[6:], self.model.jnt_range[self._finger_joint_ids, 0],
            self.model.jnt_range[self._finger_joint_ids, 1],
        )

        for _ in range(round(_CONTROL_TIMESTEP / _PHYSICS_TIMESTEP)):
            mujoco.mj_forward(self.model, self.data)
            self._apply_arm_control(target_position, target_rotation)
            self.data.ctrl[self._finger_actuator_ids] = finger_targets
            mujoco.mj_step(self.model, self.data)

        mujoco.mj_forward(self.model, self.data)
        trigger = self.data.qpos[self._trigger_qpos_id]
        was_pulled = self._trigger_pulled
        if trigger < _TRIGGER_RELEASE:
            self._trigger_pulled = False
        elif trigger > _TRIGGER_PULL:
            self._trigger_pulled = True
        if self._trigger_pulled and not was_pulled:
            self._cone_steps = 30
        self.model.geom_rgba[self._cone_geom_id, 3] = 0.5 if self._cone_steps else 0
        self._cone_steps = max(0, self._cone_steps - 1)
        inside = self._spray_hits_plant()
        self._success_steps = self._success_steps + 1 if inside and self._trigger_pulled else 0
        success = self._success_steps >= _SUCCESS_STEPS
        failed = self._trigger_pulled and not inside
        self._step_count += 1
        truncated = self._step_count >= _MAX_EPISODE_STEPS
        terminated = success or failed
        reward = float(success)
        return self._observation(), reward, terminated, truncated, {"is_success": success}

    def _finger_home(self):
        home = np.zeros(16)
        home[12] = 0.263
        return home

    def _apply_arm_control(self, target_position, target_rotation):
        current = self.data.site_xpos[self._site_id]
        position_error = target_position - current
        jac_pos = np.zeros((3, self.model.nv))
        jac_rot = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jac_pos, jac_rot, self._site_id)
        jac = np.vstack((jac_pos[:, self._arm_dof_ids], jac_rot[:, self._arm_dof_ids]))
        current_rotation = Rotation.from_matrix(
            self.data.site_xmat[self._site_id].reshape(3, 3)
        )
        error = np.concatenate((
            position_error, (target_rotation * current_rotation.inv()).as_rotvec(),
        ))
        gains = np.array([300.0, 300.0, 300.0, 40.0, 40.0, 40.0])
        velocity = self.data.qvel[self._arm_dof_ids]
        wrench = gains * error - 2 * np.sqrt(gains) * (jac @ velocity)
        torque = jac.T @ wrench - 2.0 * velocity
        torque += self.data.qfrc_bias[self._arm_dof_ids]
        limits = self.model.actuator_ctrlrange[self._arm_actuator_ids]
        self.data.ctrl[self._arm_actuator_ids] = np.clip(torque, limits[:, 0], limits[:, 1])

    def _spray_hits_plant(self):
        # Preserve the benchmark's reference-point cylinder test, not fluid simulation.
        spray_point = self.data.site_xpos[self._ref_site_id]
        plant = self.data.xpos[self._plant_body_id]
        delta = spray_point - plant
        return bool(float(delta[:2] @ delta[:2]) <= 0.2**2 and abs(delta[2]) <= 0.2)

    def _observation(self):
        position = self.data.site_xpos[self._site_id]
        rotation = Rotation.from_matrix(self.data.site_xmat[self._site_id].reshape(3, 3)).as_rotvec()
        fingers = self.data.qpos[self._finger_qpos_ids]
        observation = {
            "agent_pos": np.concatenate((position, rotation, fingers)).astype(np.float32)
        }
        if self.image_observations:
            observation["pixels"] = {
                "front": self._render_camera("front"),
                "wrist": self._render_camera("handcam_rgb"),
            }
        return observation

    def render(self):
        return self._render_camera("front")

    def _render_camera(self, camera):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render().copy()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
