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

XML_PATH = ASSET_ROOT / "tasks" / "hammer_nail" / "hammer_nail.xml"
MAX_EPISODE_STEPS = 1000
SUCCESS_DEPTH = 0.04
INSERT_STEP = 0.008
IMPACT_VELOCITY = 0.02
MAX_DEPTH = 0.0726


class HammerNailStateMachine:
    """Track contact-driven insertion depth and episode outcomes."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.step_count = 0
        self.depth = 0.0

    def update(self, hit: bool, velocity: float) -> tuple[bool, bool, bool]:
        if hit and velocity < -IMPACT_VELOCITY:
            scale = min(3.0, abs(velocity) / IMPACT_VELOCITY)
            self.depth = min(MAX_DEPTH, self.depth + INSERT_STEP * scale)
        self.step_count += 1
        success = self.depth >= SUCCESS_DEPTH
        return success, success, self.step_count >= MAX_EPISODE_STEPS


class HammerNailEnv(gym.Env):
    """Control a Panda-Allegro hand to hammer a nail into a wooden board."""

    # render fps is 30 Hz, but control frequency is 50 Hz
    # can not align due to backward compatibility
    # Not a ClassVar on purpose: gymnasium.Env types metadata as instance-assignable.
    metadata: dict[str, Any] = {  # noqa: RUF012
        "render_modes": ["rgb_array", "human", "none"],
        "render_fps": 30,
    }
    task = "hammer_nail"
    task_description = "Use the hammer to drive the nail into the wooden board."
    side_cameras = ("ego_left", "ego_right")
    _max_episode_steps = MAX_EPISODE_STEPS

    def __init__(
        self,
        render_mode: Literal["rgb_array", "human", "none"] = "rgb_array",
        action_format: Literal["rotvec", "quaternion"] = "quaternion",
        randomize: bool = False,
        randomize_dynamics: bool = False,
        render_width: int = 640,
        render_height: int = 640,
        seed: int | None = None,
        offsamples: int | None = None,
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
        self.render_mode = render_mode
        # MuJoCo simulation.
        self.model = mujoco.MjModel.from_xml_path(str(XML_PATH))  # pyright: ignore[reportAttributeAccessIssue]
        if offsamples is not None:
            self.model.vis.quality.offsamples = offsamples
        assert self.model.opt.timestep == PHYSICS_TIMESTEP, (
            "XML timestep must match PHYSICS_TIMESTEP"
        )
        self.data = mujoco.MjData(self.model)  # pyright: ignore[reportAttributeAccessIssue]
        # Create graphics resources only when pixel observations are requested.
        self.render_width = render_width
        self.render_height = render_height
        self.renderer: mujoco.Renderer | None = None
        self.human_viewer = None
        self.state_machine = HammerNailStateMachine()

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
        self.hammer_body_id = self.model.body("hammer_body").id
        self.nail_body_id = self.model.body("nail").id
        self.nail_mocap_id = int(self.model.body_mocapid[self.nail_body_id])
        self.robot_mocap_id = int(self.model.body_mocapid[self.model.body("target").id])
        self.face_geom_id = self.model.geom("face").id
        self.hammer_geom_ids = {self.model.geom(name).id for name in ("face", "head", "neck", "claw")}
        self.nail_geom_ids = {self.model.geom(name).id for name in ("nail_head", "nail_shaft")}
        self.orig_table_z = self.model.body("table").pos[2].copy()
        self.orig_wood_z = self.model.body("wood").pos[2].copy()
        self.orig_hammer_z = self.model.body_pos[self.hammer_body_id, 2].copy()
        self.orig_nail_pos = self.model.body_pos[self.nail_body_id].copy()
        self.orig_nail_quat = self.model.body_quat[self.nail_body_id].copy()
        self.leg_ids = np.array([
            self.model.geom(f"table_leg_{i}").id for i in range(1, 5)
        ])
        self.leg_half_lengths = self.model.geom_size[self.leg_ids, 1].copy()

        # Independent RNGs per randomization domain.
        self.reset_rng = np.random.RandomState(seed)
        self.dynamics_rng = np.random.RandomState(seed)
        self.visual_rng = np.random.RandomState(seed)

        # Visual randomization baselines and camera samples.
        self.camera_params = np.load(ASSET_ROOT / "common" / "replay_cameras.npy")
        self.front_camera_id = self.model.camera("front").id
        # light_pos: [n_lights, 3], x, y, z
        # light_dir: [n_lights, 3], x, y, z (the direction)
        self.orig_light_pos = self.model.light_pos.copy()
        self.orig_light_dir = self.model.light_dir.copy()

        # Hammer mass baseline for dynamics randomization.
        self.orig_hammer_mass = float(self.model.body_mass[self.hammer_body_id])

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
            "agent_pos": spaces.Box(
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
            observation_spaces["pixels"] = spaces.Dict({
                name: spaces.Box(
                    0, 255, shape=(render_height, render_width, 3), dtype=np.uint8
                )
                for name in ("ego", "ego_left", "ego_right", "wrist")
            })
        self.observation_space = spaces.Dict(observation_spaces)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.reset_rng.seed(seed)
            self.dynamics_rng.seed(seed)
            self.visual_rng.seed(seed)
        initial_state = options.get("initial_state") if options is not None else None
        mujoco.mj_resetData(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]

        # Sample or restore the scene layout.
        if initial_state is None:
            delta_h = self.reset_rng.uniform(0.0, 0.05)
            hammer_pose = self.data.jnt("hammer_joint").qpos
            hammer_pose[:2] = self.reset_rng.uniform([-0.25, -0.35], [-0.40, -0.50])
            hammer_pose[2] = self.orig_hammer_z + delta_h
            hammer_pose[3:7] = self.randomize_yaw(hammer_pose[3:7].copy())
            nail_pose = np.concatenate((self.orig_nail_pos.copy(), self.orig_nail_quat))
            nail_pose[:2] = self.reset_rng.uniform([-0.10, 0.00], [0.00, 0.10])
        else:
            delta_h = float(initial_state["table_delta_height"])
            self.data.jnt("hammer_joint").qpos[:] = initial_state["orig_hammer_pose"]
            nail_pose = np.array(initial_state["orig_nail_pose"], dtype=np.float64)
        nail_pose[2] = self.orig_nail_pos[2] + delta_h
        self.nail_initial_position = nail_pose[:3].copy()
        self.data.mocap_pos[self.nail_mocap_id] = nail_pose[:3]
        self.data.mocap_quat[self.nail_mocap_id] = nail_pose[3:7]
        self.model.body("wood").pos[2] = self.orig_wood_z + delta_h
        self.model.body_pos[self.hammer_body_id, 2] = self.orig_hammer_z + delta_h
        self.model.body("table").pos[2] = self.orig_table_z + delta_h
        self.model.geom_size[self.leg_ids, 1] = self.leg_half_lengths + delta_h

        # Restore robot posture.
        self.data.qpos[self.model.jnt_qposadr[self.arm_joint_ids]] = ARM_HOME
        self.data.qpos[self.finger_qpos_ids] = FINGER_HOME

        # Apply visual and dynamics randomization before computing derived data.
        if self.randomize:
            self.randomize_lighting()
            self.randomize_camera()
            texture = self.visual_rng.choice(tuple(
                "dark_wood_mat" if name == "table_dark-wood" else name
                for name in TABLE_TEXTURES
            ))
            self.model.geom("table_visual").matid[0] = self.model.material(texture).id
        if self.randomize_dynamics:
            self.randomize_hammer_dynamics()

        self.state_machine.reset()
        mujoco.mj_forward(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]
        self.previous_face_z = float(self.data.geom_xpos[self.face_geom_id, 2])
        self.velocity_history: list[float] = []
        # Set the mocap target from the final initial robot pose.
        self.data.mocap_pos[self.robot_mocap_id] = self.data.sensor("franka/flange_pos").data
        self.data.mocap_quat[self.robot_mocap_id] = self.data.sensor("franka/flange_quat").data

        # Capture the scene layout for replay.
        initial_state = {
            "table_delta_height": float(delta_h),
            # [x, y, z, qw, qx, qy, qz] (MuJoCo scalar-first).
            "orig_hammer_pose": self.data.jnt("hammer_joint").qpos.copy(),
            "orig_nail_pose": nail_pose.copy(),
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
        pose_dim = 6 if self.action_format == "rotvec" else 7
        finger_targets = action[pose_dim:]
        if self.action_format == "rotvec":
            target_quat = Rotation.from_rotvec(action[3:6]).as_quat(
                scalar_first=True
            )
        else:
            target_quat = action[3:7]
        self.data.mocap_pos[self.robot_mocap_id] = action[:3]
        self.data.mocap_quat[self.robot_mocap_id] = target_quat

        for _ in range(int(CONTROL_TIMESTEP // PHYSICS_TIMESTEP)):
            self.apply_arm_control()
            self.data.ctrl[self.finger_actuator_ids] = finger_targets
            mujoco.mj_step(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]

            # Preserve upstream's control-time divisor for each physics-step sample.
            face_z = float(self.data.geom_xpos[self.face_geom_id, 2])
            self.velocity_history.append((face_z - self.previous_face_z) / CONTROL_TIMESTEP)
            self.velocity_history = self.velocity_history[-12:]
            self.previous_face_z = face_z

        hit = self.hammer_contacts_nail()
        previous_depth = self.state_machine.depth
        velocity = min(self.velocity_history)
        success, terminated, truncated = self.state_machine.update(hit, velocity)
        if self.state_machine.depth > previous_depth:
            position = self.nail_initial_position.copy()
            position[2] -= self.state_machine.depth
            self.data.mocap_pos[self.nail_mocap_id] = position
            self.data.mocap_quat[self.nail_mocap_id] = self.orig_nail_quat
            mujoco.mj_forward(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]
        reward = float(success)
        if self.render_mode == "human":
            self.sync_human_viewer()
        return (
            self.observation(),
            reward,
            terminated,
            truncated,
            {"is_success": success, "nail_depth": self.state_machine.depth, "hammer_hit": hit},
        )

    def apply_arm_control(self):
        self.data.ctrl[self.arm_actuator_ids] = opspace(
            model=self.model,
            data=self.data,
            site_id=self.site_id,
            dof_ids=self.arm_dof_ids,
            pos=self.data.mocap_pos[self.robot_mocap_id],
            ori=self.data.mocap_quat[self.robot_mocap_id],
            joint=ARM_HOME,
            gravity_comp=True,
            pos_gains=(400.0, 400.0, 400.0),
            damping_ratio=4,
        )

    def randomize_yaw(self, quaternion):
        yaw = np.deg2rad(self.reset_rng.uniform(-10, 10))
        w1, z1 = np.cos(yaw / 2), np.sin(yaw / 2)
        w2, x2, y2, z2 = quaternion
        rotated = np.array([
            w1 * w2 - z1 * z2,
            w1 * x2 - z1 * y2,
            w1 * y2 + z1 * x2,
            w1 * z2 + z1 * w2,
        ])
        return rotated / np.linalg.norm(rotated)

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

    def hammer_contacts_nail(self) -> bool:
        for contact in self.data.contact[:self.data.ncon]:
            first, second = int(contact.geom1), int(contact.geom2)
            if (first in self.hammer_geom_ids and second in self.nail_geom_ids) or (
                second in self.hammer_geom_ids and first in self.nail_geom_ids
            ):
                return True
        return False

    def randomize_hammer_dynamics(self):
        self.model.body_mass[self.hammer_body_id] = self.orig_hammer_mass * float(
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
            "agent_pos": np.concatenate((position, rotation, fingers)).astype(np.float32)
        }
        if self.render_mode != "none":
            observation["pixels"] = {
                "ego": self.render_camera("front"),
                "ego_left": self.render_camera("left"),
                "ego_right": self.render_camera("right"),
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
        assert self.render_mode != "none", (
            "Rendering is disabled because render_mode='none'."
        )
        if self.renderer is None:
            self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, self.render_width)
            self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, self.render_height)
            self.renderer = mujoco.Renderer(
                self.model, height=self.render_height, width=self.render_width
            )
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render().copy()

    def close(self):
        if self.human_viewer is not None:
            self.human_viewer.close()
            self.human_viewer = None
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
