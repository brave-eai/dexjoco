from typing import Any, Literal

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from scipy.spatial.transform import Rotation

from dexjoco.common import (
    ARM_ACTUATORS,
    ARM_HOME,
    ARM_JOINTS,
    ASSET_ROOT,
    CONTROL_TIMESTEP,
    FINGER_ACTUATORS,
    FINGER_HOME,
    FINGER_JOINTS,
    FINGER_QPOS_MARGIN,
    PHYSICS_TIMESTEP,
    TABLE_TEXTURES,
    TCP_HIGH,
    TCP_LOW,
)
from dexjoco.controllers import opspace

XML_PATH = ASSET_ROOT / "tasks" / "water_plant" / "water_plant.xml"
MAX_EPISODE_STEPS = 1000
SUCCESS_STEPS = 30
TRIGGER_RELEASE = 0.25
TRIGGER_PULL = 0.34


class WaterPlantStateMachine:
    """Track trigger hysteresis, spray timing, and episode outcomes."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.step_count = 0
        self.success_steps = 0
        self.trigger_pulled = False
        self.cone_steps = 0
        self.cone_alpha = 0.0

    def update(self, trigger: float, inside: bool) -> tuple[bool, bool, bool]:
        was_pulled = self.trigger_pulled
        if trigger < TRIGGER_RELEASE:
            self.trigger_pulled = False
        elif trigger > TRIGGER_PULL:
            self.trigger_pulled = True
        if self.trigger_pulled and not was_pulled:
            self.cone_steps = 30
        # Display this step before decrementing the remaining spray duration.
        self.cone_alpha = 0.5 if self.cone_steps else 0.0
        self.cone_steps = max(0, self.cone_steps - 1)
        self.success_steps = (
            self.success_steps + 1 if inside and self.trigger_pulled else 0
        )
        success = self.success_steps >= SUCCESS_STEPS
        failed = self.trigger_pulled and not inside
        self.step_count += 1
        truncated = self.step_count >= MAX_EPISODE_STEPS
        return success, success or failed, truncated


class WaterPlantEnv(gym.Env):
    """Control a Panda-Allegro hand to water a plant."""

    # render fps is 30 Hz, but control frequency is 50 Hz
    # can not align due to backward compatibility
    # Not a ClassVar on purpose: gymnasium.Env types metadata as instance-assignable.
    metadata: dict[str, Any] = {  # noqa: RUF012
        "render_modes": ["rgb_array", "human", "none"],
        "render_fps": 30,
    }
    task = "water_plant"
    task_description = "Pick up the spray bottle and water the plant."
    _max_episode_steps = MAX_EPISODE_STEPS

    def __init__(
        self,
        render_mode: Literal["rgb_array", "human", "none"] = "rgb_array",
        action_format: Literal["rotvec", "quaternion"] = "rotvec",
        randomize: bool = False,
        randomize_dynamics: bool = False,
        render_width: int = 640,
        render_height: int = 480,
    ):
        assert render_mode in ("rgb_array", "human", "none"), (
            f"Unsupported render mode: {render_mode}"
        )
        assert action_format in ("rotvec", "quaternion"), (
            f"Unsupported action format: {action_format}"
        )
        # Environment configuration.
        self.action_format = action_format
        self.randomize = randomize
        self.randomize_dynamics = randomize_dynamics
        self.render_mode: Literal["rgb_array", "human", "none"] = render_mode
        # MuJoCo simulation.
        self.model = mujoco.MjModel.from_xml_path(str(XML_PATH))  # pyright: ignore[reportAttributeAccessIssue]
        assert self.model.opt.timestep == PHYSICS_TIMESTEP, (
            "XML timestep must match PHYSICS_TIMESTEP"
        )
        self.data = mujoco.MjData(self.model)  # pyright: ignore[reportAttributeAccessIssue]
        # Offscreen renderer for pixel observations; not needed when rendering is off.
        if render_mode != "none":
            self.renderer = mujoco.Renderer(
                self.model, height=render_height, width=render_width
            )
        else:
            self.renderer = None
        self.human_viewer = None
        self.state_machine = WaterPlantStateMachine()

        # Arm and finger control indices.
        self.site_id = self.model.site("attachment_site").id
        self.arm_joint_ids = np.array([
            self.model.joint(name).id for name in ARM_JOINTS
        ])
        self.arm_dof_ids = self.model.jnt_dofadr[self.arm_joint_ids]
        self.arm_actuator_ids = np.array([
            self.model.actuator(name).id for name in ARM_ACTUATORS
        ])
        self.finger_joint_ids = np.array([
            self.model.joint(name).id for name in FINGER_JOINTS
        ])
        self.finger_qpos_ids = self.model.jnt_qposadr[self.finger_joint_ids]
        self.finger_actuator_ids = np.array([
            self.model.actuator(name).id for name in FINGER_ACTUATORS
        ])

        # Scene placement and task geometry.
        self.plant_body_id = self.model.body("plant").id
        self.ref_site_id = self.model.site("ref_point").id
        self.cone_geom_id = self.model.geom("cone_visual").id
        self.orig_table_z = self.model.body("table").pos[2].copy()
        self.orig_plant_z = self.model.body("plant").pos[2].copy()
        self.orig_spray_z = self.model.body("link_2").pos[2].copy()
        self.leg_ids = np.array([
            self.model.geom(f"table_leg_{i}").id for i in range(1, 5)
        ])
        self.leg_half_lengths = self.model.geom_size[self.leg_ids, 1].copy()

        # Independent RNGs per randomization domain.
        self.reset_rng = np.random.RandomState()
        self.dynamics_rng = np.random.RandomState()
        self.visual_rng = np.random.RandomState()

        # Visual randomization baselines and camera samples.
        self.camera_params = np.load(ASSET_ROOT / "common" / "replay_cameras.npy")
        self.front_camera_id = self.model.camera("front").id
        # light_pos: [n_lights, 3], x, y, z
        # light_dir: [n_lights, 3], x, y, z (the direction)
        self.orig_light_pos = self.model.light_pos.copy()
        self.orig_light_dir = self.model.light_dir.copy()

        # Spray dynamics indices and baselines.
        self.spray_joint_id = self.model.joint("joint_0").id
        self.spray_dof_id = self.model.jnt_dofadr[self.spray_joint_id]
        self.spray_body_id = self.model.body("link_2").id
        self.orig_spray_stiffness = float(self.model.jnt_stiffness[self.spray_joint_id])
        self.orig_spray_mass = float(self.model.body_mass[self.spray_body_id])

        # Action and observation spaces.
        hand_limits = self.model.actuator_ctrlrange[self.finger_actuator_ids]
        # Action layout: 6-dim pose (xyz + rotvec) or 7-dim pose (xyz + quat),
        # then 16 finger targets bounded by actuator ctrlrange. TCP position uses
        # the workspace envelope; rotvec is within +-pi; quat components within
        # [-1, 1].
        if action_format == "rotvec":
            pose_low = np.concatenate((TCP_LOW, np.full(3, -np.pi)))
            pose_high = np.concatenate((TCP_HIGH, np.full(3, np.pi)))
        else:
            pose_low = np.concatenate((TCP_LOW, np.full(4, -1.0)))
            pose_high = np.concatenate((TCP_HIGH, np.full(4, 1.0)))
        self.action_space = spaces.Box(
            low=np.concatenate((pose_low, hand_limits[:, 0])).astype(np.float32),
            high=np.concatenate((pose_high, hand_limits[:, 1])).astype(np.float32),
            dtype=np.float32,
        )
        self.action_dim = len(pose_low) + len(hand_limits)
        observation_spaces: dict[str, spaces.Space] = {
            # proprio layout (23 dims):
            #   [0:3]   TCP position xyz from the franka/flange_pos sensor,
            #           bounded by the workspace envelope
            #   [3:7]   TCP rotation from the franka/flange_quat sensor,
            #           scalar-first [qw, qx, qy, qz], components within [-1, 1]
            #   [7:11]  index finger joints ffj0-ffj3, jnt_range +- margin
            #   [11:15] middle finger joints mfj0-mfj3
            #   [15:19] ring finger joints rfj0-rfj3
            #   [19:23] thumb joints thj0-thj3
            "proprio": spaces.Box(
                np.concatenate((
                    TCP_LOW,
                    np.full(4, -1.0),
                    self.model.jnt_range[self.finger_joint_ids, 0] - FINGER_QPOS_MARGIN,
                )).astype(np.float32),
                np.concatenate((
                    TCP_HIGH,
                    np.full(4, 1.0),
                    self.model.jnt_range[self.finger_joint_ids, 1] + FINGER_QPOS_MARGIN,
                )).astype(np.float32),
                dtype=np.float32,
            ),
        }
        if render_mode != "none":
            observation_spaces["images"] = spaces.Dict({
                name: spaces.Box(
                    0, 255, shape=(render_height, render_width, 3), dtype=np.uint8
                )
                for name in ("front", "wrist")
            })
        self.observation_space = spaces.Dict(observation_spaces)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.reset_rng.seed(seed)
            self.dynamics_rng.seed(seed)
            self.visual_rng.seed(seed)
        assert not options or "initial_state" in options, (
            "Unsupported reset options; expected 'initial_state'."
        )
        initial_state = options.get("initial_state") if options is not None else None
        mujoco.mj_resetData(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]

        # Sample or restore the scene layout.
        if initial_state is None:
            delta_h = self.reset_rng.uniform(0.0, 0.05)
            spray_pos = self.data.jnt("spray_root").qpos[:3]
            spray_pos[:2] = self.reset_rng.uniform([-0.35, -0.25], [-0.30, -0.20])
            spray_pos[2] = self.orig_spray_z + delta_h
            plant_pos = self.model.body("plant").pos
            plant_pos[:2] = self.reset_rng.uniform([-0.10, 0.15], [-0.05, 0.20])
            plant_pos[2] = self.orig_plant_z + delta_h
        else:
            delta_h = float(initial_state["table_delta_height"])
            self.data.jnt("spray_root").qpos[:] = initial_state["orig_spray_pose"]
            plant_pose = initial_state["orig_plant_pose"]
            self.model.body("plant").pos[:] = plant_pose[:3]
            self.model.body("plant").quat[:] = plant_pose[3:]
        self.model.body("table").pos[2] = self.orig_table_z + delta_h
        self.model.geom_size[self.leg_ids, 1] = self.leg_half_lengths + delta_h

        # Restore robot posture.
        self.data.qpos[self.model.jnt_qposadr[self.arm_joint_ids]] = ARM_HOME
        self.data.qpos[self.finger_qpos_ids] = FINGER_HOME

        # Apply visual and dynamics randomization before computing derived data.
        if self.randomize:
            self.randomize_lighting()
            self.randomize_camera()
            texture = self.visual_rng.choice(TABLE_TEXTURES)
            self.model.geom("table_visual").matid[0] = self.model.material(texture).id
        if self.randomize_dynamics:
            self.randomize_spray_dynamics()

        self.state_machine.reset()
        self.model.geom_rgba[self.cone_geom_id, 3] = self.state_machine.cone_alpha
        mujoco.mj_forward(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]
        # Set the mocap target from the final initial robot pose.
        self.data.mocap_pos[0] = self.data.sensor("franka/flange_pos").data
        self.data.mocap_quat[0] = self.data.sensor("franka/flange_quat").data

        # Capture the scene layout for replay.
        initial_state = {
            "table_delta_height": float(delta_h),
            # [x, y, z, qw, qx, qy, qz] (MuJoCo scalar-first).
            "orig_spray_pose": self.data.jnt("spray_root").qpos.copy(),
            # [x, y, z, qw, qx, qy, qz] (MuJoCo scalar-first).
            "orig_plant_pose": np.concatenate((
                self.model.body("plant").pos.copy(),
                self.model.body("plant").quat.copy(),
            )),
        }
        return self.observation(), {
            "is_success": False,
            "initial_state": initial_state,
        }

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        assert action.shape == (self.action_dim,), (
            f"Expected {self.action_dim} action values"
        )
        if self.action_format == "rotvec":
            target_quat = Rotation.from_rotvec(action[3:6]).as_quat(scalar_first=True)
            finger_targets = action[6:]
        else:
            target_quat = action[3:7]
            finger_targets = action[7:]
        self.data.mocap_pos[0] = action[:3]
        self.data.mocap_quat[0] = target_quat

        for _ in range(int(CONTROL_TIMESTEP // PHYSICS_TIMESTEP)):
            self.apply_arm_control()
            self.data.ctrl[self.finger_actuator_ids] = finger_targets
            mujoco.mj_step(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]

        # Sample sensors and kinematics after mj_step.
        trigger = self.data.sensor("spray_joint_0_pos").data[0]
        success, terminated, truncated = self.state_machine.update(
            float(trigger), self.spray_hits_plant()
        )
        self.model.geom_rgba[self.cone_geom_id, 3] = self.state_machine.cone_alpha
        reward = float(success)
        if self.render_mode == "human":
            self.sync_human_viewer()
        return (
            self.observation(),
            reward,
            terminated,
            truncated,
            {"is_success": success},
        )

    def apply_arm_control(self):
        self.data.ctrl[self.arm_actuator_ids] = opspace(
            model=self.model,
            data=self.data,
            site_id=self.site_id,
            dof_ids=self.arm_dof_ids,
            pos=self.data.mocap_pos[0],
            ori=self.data.mocap_quat[0],
            joint=ARM_HOME,
            gravity_comp=True,
            pos_gains=(400.0, 400.0, 400.0),
            damping_ratio=4,
        )

    def spray_hits_plant(self):
        spray_point = self.data.site_xpos[self.ref_site_id]
        plant = self.data.xpos[self.plant_body_id]
        dx, dy, dz = spray_point - plant
        return bool(dx * dx + dy * dy <= 0.2 * 0.2 and -0.2 <= dz <= 0.2)

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
            model.light_diffuse[i] = rng.uniform(0.3, 0.8, size=3)
        model.vis.headlight.ambient[:] = rng.uniform(0.3, 0.7, size=3)
        model.vis.headlight.diffuse[:] = rng.uniform(0.2, 0.6, size=3)

    def randomize_camera(self):
        camera = self.camera_params[self.visual_rng.randint(len(self.camera_params))]
        azimuth = np.deg2rad(float(camera[0]))
        elevation = np.deg2rad(float(-camera[1]))
        distance = float(camera[2])
        offset = np.array(
            [
                -distance * np.cos(elevation) * np.cos(azimuth),
                distance * np.cos(elevation) * np.sin(azimuth),
                distance * np.sin(elevation),
            ],
            dtype=np.float64,
        )
        forward = -offset
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, np.array([0.0, 0.0, 1.0], dtype=np.float64))
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        rotation = np.column_stack([right, up, -forward])
        self.model.cam_pos[self.front_camera_id] = np.array([0.0, 0.0, 1.0]) + offset
        self.model.cam_quat[self.front_camera_id] = Rotation.from_matrix(
            rotation
        ).as_quat(scalar_first=True)

    def randomize_spray_dynamics(self):
        # Scale the original values each reset
        self.model.dof_frictionloss[self.spray_dof_id] = float(
            self.dynamics_rng.uniform(0.0, 0.05)
        )
        self.model.jnt_stiffness[self.spray_joint_id] = (
            self.orig_spray_stiffness * float(self.dynamics_rng.uniform(0.75, 1.25))
        )
        self.model.body_mass[self.spray_body_id] = self.orig_spray_mass * float(
            self.dynamics_rng.uniform(0.75, 1.25)
        )

    def observation(self) -> dict[str, Any]:
        position = self.data.sensor("franka/flange_pos").data
        rotation = self.data.sensor("franka/flange_quat").data
        fingers = np.array(
            [
                self.data.sensor(f"allegro_right/{name}_pos").data[0]
                for name in FINGER_JOINTS
            ],
            dtype=np.float32,
        )
        observation: dict[str, Any] = {
            "proprio": np.concatenate((position, rotation, fingers)).astype(np.float32)
        }
        if self.render_mode != "none":
            observation["images"] = {
                "front": self.render_camera("front"),
                "wrist": self.render_camera("handcam_rgb"),
            }
        return observation

    def render(self):
        assert self.render_mode != "none", (
            "Rendering is disabled because render_mode='none'."
        )
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
        assert self.renderer is not None
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render().copy()

    def close(self):
        if self.human_viewer is not None:
            self.human_viewer.close()
            self.human_viewer = None
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
