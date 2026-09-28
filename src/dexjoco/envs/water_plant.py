"""A compact MuJoCo implementation of the Water Plant task."""

import random
from typing import ClassVar, Literal

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from scipy.spatial.transform import Rotation

from dexjoco.common import (
    _ARM_ACTUATORS,
    _ARM_JOINTS,
    _ASSET_ROOT,
    _CONTROL_TIMESTEP,
    _FINGER_ACTUATORS,
    _FINGER_JOINTS,
    _PHYSICS_TIMESTEP,
)
from dexjoco.controllers import opspace

# Render modes: "rgb_array" = offscreen RGB frames, "human" = live GUI window,
# "none" = no rendering (pixel observations off by default).
RenderMode = Literal["rgb_array", "human", "none"]

_XML_PATH = _ASSET_ROOT / "tasks" / "water_plant" / "water_plant.xml"
_MAX_EPISODE_STEPS = 1000
_SUCCESS_STEPS = 30
_TRIGGER_RELEASE = 0.25
_TRIGGER_PULL = 0.34


class WaterPlantEnv(gym.Env):
    """Control a Panda-Allegro hand to water a plant."""

    metadata: ClassVar[dict] = {"render_modes": ["rgb_array", "human", "none"], "render_fps": 50}
    task = "water_plant"
    task_description = "Pick up the spray bottle and water the plant."
    _max_episode_steps = _MAX_EPISODE_STEPS

    def __init__(
        self, render_mode: RenderMode = "rgb_array", image_observations: bool | None = None,
        action_format: str = "rotvec",
        randomize: bool = False, randomize_dynamics: bool = False,
    ):
        if render_mode not in ("rgb_array", "human", "none"):
            raise ValueError(f"Unsupported render mode: {render_mode}")
        if action_format not in ("rotvec", "quaternion"):
            raise ValueError(f"Unsupported action format: {action_format}")
        self.action_format = action_format
        self.randomize = randomize
        self.randomize_dynamics = randomize_dynamics
        self.render_mode = render_mode
        # Legacy behavior: pixel observations are off only when rendering is off.
        self.image_observations = render_mode != "none" if image_observations is None else image_observations
        self.model = mujoco.MjModel.from_xml_path(str(_XML_PATH))  # pyright: ignore[reportAttributeAccessIssue]
        self.model.opt.timestep = _PHYSICS_TIMESTEP
        self.data = mujoco.MjData(self.model)  # pyright: ignore[reportAttributeAccessIssue]
        self.renderer = None
        self.human_viewer = None
        self.step_count = 0
        self.success_steps = 0
        self.trigger_pulled = False

        self.site_id = self.model.site("attachment_site").id
        self.plant_body_id = self.model.body("plant").id
        self.ref_site_id = self.model.site("ref_point").id
        self.cone_geom_id = self.model.geom("cone_visual").id
        self.cone_steps = 0
        self.arm_joint_ids = np.array([self.model.joint(name).id for name in _ARM_JOINTS])
        self.arm_dof_ids = self.model.jnt_dofadr[self.arm_joint_ids]
        self.arm_actuator_ids = np.array([
            self.model.actuator(name).id for name in _ARM_ACTUATORS
        ])
        self.finger_joint_ids = np.array([
            self.model.joint(name).id for name in _FINGER_JOINTS
        ])
        self.finger_qpos_ids = self.model.jnt_qposadr[self.finger_joint_ids]
        self.finger_actuator_ids = np.array([
            self.model.actuator(name).id for name in _FINGER_ACTUATORS
        ])
        self.arm_home = np.array([0.0, -0.785, 0.0, -2.35, 0.0, 1.57, np.pi / 4])

        # Match legacy np.random.seed/uniform without sharing global RNG state.
        self.reset_rng = np.random.RandomState(0)
        self.visual_rng = random.Random(0)
        self.camera_params = np.load(_ASSET_ROOT / "common" / "replay_cameras.npy")
        self.front_camera_id = self.model.camera("front").id
        self.orig_light_pos = self.model.light_pos.copy()
        self.orig_light_dir = self.model.light_dir.copy()
        self.texture_names = (
            "table_bamboo", "table_blue-wood", "table_brass-ambra", "table_ceramic",
            "table_cream-plaster", "table_dark_wood_planks_2", "table_dark-wood",
            "table_gray-plaster", "table_gray_wood_planks", "table_light-wood",
            "table_metal", "table_pink-plaster", "table_red-wood", "table_legs_metal",
            "table_steel-scratched", "table_walnut_wood_grain", "table_warm_wood_grain_2",
            "table_white-plaster", "table_wood_grain_1", "table_yellow-plaster",
        )
        self.spray_joint_id = self.model.joint("joint_0").id
        self.spray_dof_id = self.model.jnt_dofadr[self.spray_joint_id]
        self.spray_body_id = self.model.body("link_2").id
        self.spray_stiffness0 = float(self.model.jnt_stiffness[self.spray_joint_id])
        self.spray_mass0 = float(self.model.body_mass[self.spray_body_id])
        self.table_z0 = self.model.body("table").pos[2].copy()
        self.plant_z0 = self.model.body("plant").pos[2].copy()
        self.spray_z0 = self.model.body("link_2").pos[2].copy()
        self.leg_ids = np.array([
            self.model.geom(f"table_leg_{i}").id for i in range(1, 5)
        ])
        self.leg_half_lengths = self.model.geom_size[self.leg_ids, 1].copy()
        pose_size = 6 if action_format == "rotvec" else 7
        hand_limits = self.model.actuator_ctrlrange[self.finger_actuator_ids]
        self.action_space = spaces.Box(
            low=np.concatenate((np.full(pose_size, -np.inf), hand_limits[:, 0])).astype(np.float32),
            high=np.concatenate((np.full(pose_size, np.inf), hand_limits[:, 1])).astype(np.float32),
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
        if seed is not None:
            self.reset_rng.seed(seed)
            self.visual_rng.seed(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.delta_h = self.reset_rng.uniform(0.0, 0.05)
        self.model.body("table").pos[2] = self.table_z0 + self.delta_h
        self.model.geom_size[self.leg_ids, 1] = self.leg_half_lengths + self.delta_h
        spray_pos = self.model.body("link_2").pos
        spray_pos[:2] = self.reset_rng.uniform([-0.35, -0.25], [-0.30, -0.20])
        spray_pos[2] = self.spray_z0 + self.delta_h
        self.data.jnt("spray_root").qpos[:3] = spray_pos
        plant_pos = self.model.body("plant").pos
        plant_pos[:2] = self.reset_rng.uniform([-0.10, 0.15], [-0.05, 0.20])
        plant_pos[2] = self.plant_z0 + self.delta_h
        if options:
            # Same scene restoration as the legacy water_plant replay script.
            self.delta_h = float(np.asarray(options["table_delta_height"]).item())
            self.model.body("table").pos[2] = self.table_z0 + self.delta_h
            self.model.geom_size[self.leg_ids, 1] = self.leg_half_lengths + self.delta_h
            self.data.jnt("spray_root").qpos[:] = options["spray_ori_pose"]
            self.model.body("plant").pos[:] = options["plant_ori_pose"][:3]
        self.data.qpos[self.model.jnt_qposadr[self.arm_joint_ids]] = self.arm_home
        self.data.qpos[self.finger_qpos_ids] = self.finger_home()
        mujoco.mj_forward(self.model, self.data)
        self.data.mocap_pos[0] = self.data.sensor("franka/flange_pos").data
        self.data.mocap_quat[0] = self.data.sensor("franka/flange_quat").data
        if self.randomize:
            self.randomize_lighting()
            self.randomize_camera()
            texture = self.visual_rng.choice(self.texture_names)
            self.model.geom("table_visual").matid[0] = self.model.material(texture).id
        if self.randomize_dynamics:
            self.randomize_spray_dynamics()
        mujoco.mj_forward(self.model, self.data)
        self.step_count = 0
        self.success_steps = 0
        self.trigger_pulled = False
        self.cone_steps = 0
        self.model.geom_rgba[self.cone_geom_id, 3] = 0
        return self.observation(), {"is_success": False}

    def randomize_lighting(self):
        model = self.model
        rng = self.visual_rng
        model.light_pos[:] = self.orig_light_pos
        model.light_dir[:] = self.orig_light_dir
        for i in range(model.nlight):
            model.light_pos[i, 0] += rng.uniform(-0.3, 0.3)
            model.light_pos[i, 1] += rng.uniform(-0.3, 0.3)
            model.light_dir[i, 0] += rng.uniform(-0.4, 0.4)
            model.light_dir[i, 1] += rng.uniform(-0.4, 0.4)
            model.light_diffuse[i] = [rng.uniform(0.3, 0.8) for _ in range(3)]
        model.vis.headlight.ambient[:] = [rng.uniform(0.3, 0.7) for _ in range(3)]
        model.vis.headlight.diffuse[:] = [rng.uniform(0.2, 0.6) for _ in range(3)]

    def randomize_camera(self):
        camera = self.camera_params[self.visual_rng.randint(0, len(self.camera_params) - 1)]
        azimuth = np.deg2rad(float(camera[0]))
        elevation = np.deg2rad(float(-camera[1]))
        distance = float(camera[2])
        offset = np.array([
            -distance * np.cos(elevation) * np.cos(azimuth),
            distance * np.cos(elevation) * np.sin(azimuth),
            distance * np.sin(elevation),
        ], dtype=np.float64)
        forward = -offset
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, np.array([0.0, 0.0, 1.0], dtype=np.float64))
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        rotation = np.column_stack([right, up, -forward])
        self.model.cam_pos[self.front_camera_id] = np.array([0.0, 0.0, 1.0]) + offset
        self.model.cam_quat[self.front_camera_id] = Rotation.from_matrix(rotation).as_quat(scalar_first=True)

    def randomize_spray_dynamics(self):
        # Paper Table VI: joint friction loss, stiffness, and spray body mass.
        # Scale the original values each reset, never the previous sample.
        self.model.dof_frictionloss[self.spray_dof_id] = float(self.reset_rng.uniform(0.0, 0.05))
        self.model.jnt_stiffness[self.spray_joint_id] = self.spray_stiffness0 * float(self.reset_rng.uniform(0.75, 1.25))
        self.model.body_mass[self.spray_body_id] = self.spray_mass0 * float(self.reset_rng.uniform(0.75, 1.25))

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != self.action_space.shape or not np.isfinite(action).all():
            raise ValueError(f"Expected {self.action_space.shape[0]} finite action values")
        if self.action_format == "rotvec":
            target_quat = Rotation.from_rotvec(action[3:6]).as_quat(scalar_first=True)
            finger_targets = action[6:]
        else:
            target_quat = action[3:7]
            finger_targets = action[7:]
        # Legacy all-zero quaternion pose means keep the previous mocap target.
        if not (np.allclose(action[:3], 0.0) and np.allclose(target_quat, 0.0)):
            self.data.mocap_pos[0] = action[:3]
            self.data.mocap_quat[0] = target_quat

        for _ in range(int(_CONTROL_TIMESTEP // _PHYSICS_TIMESTEP)):
            self.apply_arm_control()
            self.data.ctrl[self.finger_actuator_ids] = finger_targets
            mujoco.mj_step(self.model, self.data)

        # Preserve the legacy sensor/kinematics sampling time after mj_step.
        trigger = self.data.sensor("spray_joint_0_pos").data[0]
        was_pulled = self.trigger_pulled
        if trigger < _TRIGGER_RELEASE:
            self.trigger_pulled = False
        elif trigger > _TRIGGER_PULL:
            self.trigger_pulled = True
        if self.trigger_pulled and not was_pulled:
            self.cone_steps = 30
        self.model.geom_rgba[self.cone_geom_id, 3] = 0.5 if self.cone_steps else 0
        self.cone_steps = max(0, self.cone_steps - 1)
        inside = self.spray_hits_plant()
        self.success_steps = self.success_steps + 1 if inside and self.trigger_pulled else 0
        success = self.success_steps >= _SUCCESS_STEPS
        failed = self.trigger_pulled and not inside
        self.step_count += 1
        truncated = self.step_count >= _MAX_EPISODE_STEPS
        terminated = success or failed
        reward = float(success)
        if self.render_mode == "human":
            self.sync_human_viewer()
        return self.observation(), reward, terminated, truncated, {"is_success": success}

    def finger_home(self):
        # Legacy home is float32 before assignment into MuJoCo's float64 qpos.
        home = np.zeros(16, dtype=np.float32)
        home[12] = 0.263
        return home

    def apply_arm_control(self):
        self.data.ctrl[self.arm_actuator_ids] = opspace(
            model=self.model,
            data=self.data,
            site_id=self.site_id,
            dof_ids=self.arm_dof_ids,
            pos=self.data.mocap_pos[0],
            ori=self.data.mocap_quat[0],
            joint=self.arm_home,
            gravity_comp=True,
            pos_gains=(400.0, 400.0, 400.0),
            damping_ratio=4,
        )

    def spray_hits_plant(self):
        # Preserve the benchmark's reference-point cylinder test, not fluid simulation.
        spray_point = self.data.site_xpos[self.ref_site_id]
        plant = self.data.xpos[self.plant_body_id]
        dx, dy, dz = spray_point - plant
        return bool(dx * dx + dy * dy <= 0.2 * 0.2 and -0.2 <= dz <= 0.2)

    def observation(self):
        position = self.data.site_xpos[self.site_id]
        rotation = Rotation.from_matrix(self.data.site_xmat[self.site_id].reshape(3, 3)).as_rotvec()
        fingers = np.array([
            self.data.sensor(f"allegro_right/{name}_pos").data[0]
            for name in _FINGER_JOINTS
        ], dtype=np.float32)
        observation = {
            "agent_pos": np.concatenate((position, rotation, fingers)).astype(np.float32)
        }
        if self.image_observations:
            observation["pixels"] = {
                "front": self.render_camera("front"),
                "wrist": self.render_camera("handcam_rgb"),
            }
        return observation

    def render(self):
        if self.render_mode == "none":
            raise RuntimeError("Rendering is disabled because render_mode='none'.")
        if self.render_mode == "human":
            self.sync_human_viewer()
            return None
        return self.render_camera("front")

    def sync_human_viewer(self):
        # Open the GUI window lazily so rgb_array/none never touch a windowing system.
        if self.human_viewer is None:
            import mujoco.viewer

            self.human_viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.human_viewer.sync()

    def render_camera(self, camera):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render().copy()

    def close(self):
        if self.human_viewer is not None:
            self.human_viewer.close()
            self.human_viewer = None
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
