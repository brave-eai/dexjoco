from itertools import pairwise
from typing import Any, Literal

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from dexjoco.common import (
    ARM_HOME,
    ASSET_ROOT,
    CONTROL_TIMESTEP,
    FINGER_HOME,
    FINGER_QPOS_MARGIN,
    LEFT_ARM_ACTUATORS,
    LEFT_ARM_JOINTS,
    LEFT_FINGER_ACTUATORS,
    LEFT_FINGER_JOINTS,
    PHYSICS_TIMESTEP,
    RIGHT_ARM_ACTUATORS,
    RIGHT_ARM_JOINTS,
    RIGHT_FINGER_ACTUATORS,
    RIGHT_FINGER_JOINTS,
    TABLE_TEXTURES,
)
from dexjoco.controllers import opspace

XML_PATH = ASSET_ROOT / "tasks" / "bimanual_hanoi" / "bimanual_hanoi.xml"
MAX_EPISODE_STEPS = 1500
POST_ASSIGNMENT_TOL = 0.040
PLACEMENT_HEIGHT_TOL = 0.012
STATIC_QVEL_TOL = 0.35
SUCCESS_STEPS = 20
ILLEGAL_STEPS = 10
DISK_NAMES = ("hanoi_disk_large", "hanoi_disk_medium", "hanoi_disk_small")
POST_NAMES = ("A", "B", "C")


class BimanualHanoiStateMachine:
    """Track stable legal placements and the completed tower on post C."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.step_count = 0
        self.success_steps = 0
        self.illegal_steps = 0
        self.illegal_reason = ""

    def update(
        self,
        disk_positions,
        disk_velocities,
        post_positions,
        base_top_z,
        half_heights,
    ) -> tuple[bool, bool, bool, dict[str, Any]]:
        disk_static = np.linalg.norm(disk_velocities, axis=1) <= STATIC_QVEL_TOL
        all_static = bool(disk_static.all())
        towers: dict[str, list[int]] = {post: [] for post in POST_NAMES}
        assignments = {}
        for i, name in enumerate(DISK_NAMES):
            distances = np.linalg.norm(post_positions[:, :2] - disk_positions[i, :2], axis=1)
            closest = int(np.argmin(distances))
            if distances[closest] <= POST_ASSIGNMENT_TOL:
                post = POST_NAMES[closest]
                towers[post].append(i)
                assignments[name] = post
            else:
                assignments[name] = (
                    "table" if disk_positions[i, 2] <= base_top_z + 0.06 else "floating"
                )
        for disks in towers.values():
            disks.sort(key=lambda i: disk_positions[i, 2])

        def geometry_errors(disks, post):
            running_height = base_top_z
            xy_error_max = 0.0
            height_error_max = 0.0
            for i in disks:
                expected_z = running_height + half_heights[i]
                running_height += 2.0 * half_heights[i]
                xy_error_max = max(
                    xy_error_max,
                    float(
                        np.linalg.norm(
                            disk_positions[i, :2] - post_positions[POST_NAMES.index(post), :2]
                        )
                    ),
                )
                height_error_max = max(
                    height_error_max, float(abs(disk_positions[i, 2] - expected_z))
                )
            return xy_error_max, height_error_max

        illegal_reason = ""
        for post, disks in towers.items():
            # Disk indices increase as disk sizes decrease.
            if any(a > b for a, b in pairwise(disks)) and all(disk_static[disks]):
                xy_error, height_error = geometry_errors(disks, post)
                if xy_error <= POST_ASSIGNMENT_TOL and height_error <= PLACEMENT_HEIGHT_TOL:
                    illegal_reason = f"illegal_stack_{post}"
                    break
        if not illegal_reason and all_static and "table" in assignments.values():
            illegal_reason = "disk_on_table"
        if illegal_reason:
            self.illegal_steps = (
                self.illegal_steps + 1 if illegal_reason == self.illegal_reason else 1
            )
        else:
            self.illegal_steps = 0
        self.illegal_reason = illegal_reason
        illegal = self.illegal_steps >= ILLEGAL_STEPS

        xy_error = height_error = 0.0
        geom_success = False
        if towers["C"] == [0, 1, 2] and not towers["A"] and not towers["B"]:
            xy_error, height_error = geometry_errors(towers["C"], "C")
            geom_success = (
                xy_error <= POST_ASSIGNMENT_TOL
                and height_error <= PLACEMENT_HEIGHT_TOL
                and all_static
            )
        self.success_steps = self.success_steps + 1 if geom_success and not illegal else 0
        success = self.success_steps >= SUCCESS_STEPS
        self.step_count += 1
        info = {
            "is_success": success,
            "tower_state": {post: [DISK_NAMES[i] for i in disks] for post, disks in towers.items()},
            "disk_assignment": assignments,
            "all_disks_static": all_static,
            "illegal_state": illegal,
            "illegal_reason": illegal_reason,
            "xy_error_max": xy_error,
            "height_error_max": height_error,
            "success_stable_count": self.success_steps,
            "illegal_stable_count": self.illegal_steps,
        }
        return success, success or illegal, self.step_count >= MAX_EPISODE_STEPS, info


class BimanualHanoiEnv(gym.Env):
    """Control two Panda-Allegro hands for the final two Hanoi moves."""

    # Recorded video is 30 Hz; simulation control remains 50 Hz.
    metadata: dict[str, Any] = {  # noqa: RUF012
        "render_modes": ["rgb_array", "human", "none"],
        "render_fps": 30,
    }
    task = "bimanual_hanoi"
    task_description = (
        "Execute the final two moves of the three-level Tower of Hanoi: "
        "move the medium disk from the middle peg to the right peg with the right hand, "
        "then move the small disk from the left peg to the right peg with the left hand."
    )
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
        self.action_format = action_format
        self.randomize = randomize
        self.randomize_dynamics = randomize_dynamics
        self.render_mode = render_mode
        self.model = mujoco.MjModel.from_xml_path(str(XML_PATH))  # pyright: ignore[reportAttributeAccessIssue]
        if offsamples is not None:
            self.model.vis.quality.offsamples = offsamples
        assert self.model.opt.timestep == PHYSICS_TIMESTEP
        self.data = mujoco.MjData(self.model)  # pyright: ignore[reportAttributeAccessIssue]
        self.render_width = render_width
        self.render_height = render_height
        self.renderer: mujoco.Renderer | None = None
        self.human_viewer = None
        self.state_machine = BimanualHanoiStateMachine()

        # Right then left, matching the recorded pose and finger blocks.
        sides = ("right", "left")
        self.site_ids = [self.model.site(f"attachment_site_{side}").id for side in sides]
        self.arm_joint_ids = np.array(
            [
                [self.model.joint(name).id for name in names]
                for names in (RIGHT_ARM_JOINTS, LEFT_ARM_JOINTS)
            ]
        )
        self.arm_dof_ids = self.model.jnt_dofadr[self.arm_joint_ids]
        self.arm_actuator_ids = np.array(
            [
                [self.model.actuator(name).id for name in names]
                for names in (RIGHT_ARM_ACTUATORS, LEFT_ARM_ACTUATORS)
            ]
        )
        finger_names = (RIGHT_FINGER_JOINTS, LEFT_FINGER_JOINTS)
        self.finger_joint_ids = np.array(
            [self.model.joint(name).id for names in finger_names for name in names]
        )
        self.finger_qpos_ids = self.model.jnt_qposadr[self.finger_joint_ids]
        self.finger_actuator_ids = np.array(
            [
                self.model.actuator(name).id
                for names in (RIGHT_FINGER_ACTUATORS, LEFT_FINGER_ACTUATORS)
                for name in names
            ]
        )
        self.finger_sensor_ids = [
            self.model.sensor(f"allegro_{side}/{name.removesuffix('_' + side)}_pos").id
            for side, names in zip(sides, finger_names, strict=True)
            for name in names
        ]
        self.mocap_ids = [int(self.model.body(f"target_{side}").mocapid[0]) for side in sides]
        self.base_body_id = self.model.body("hanoi_base").id
        self.orig_base_pos = self.model.body_pos[self.base_body_id].copy()
        self.post_site_ids = [
            self.model.site(f"hanoi_post_{post.lower()}_site").id for post in POST_NAMES
        ]
        self.disk_body_ids = [self.model.body(name).id for name in DISK_NAMES]
        disk_joint_ids = [self.model.joint(f"{name}_joint").id for name in DISK_NAMES]
        self.disk_dof_ids = self.model.jnt_dofadr[disk_joint_ids]
        self.orig_disk_masses = self.model.body_mass[self.disk_body_ids].copy()
        self.disk_half_heights = np.array(
            [
                0.5
                * (
                    self.model.site(f"{name}_top_site").pos[2]
                    - self.model.site(f"{name}_bottom_site").pos[2]
                )
                for name in DISK_NAMES
            ]
        )
        geom = self.model.geom("hanoi_base_upper_collision")
        self.base_upper_top_z = float(geom.pos[2] + geom.size[2])
        self.orig_table_z = self.model.body("table").pos[2].copy()
        self.leg_ids = [self.model.geom(f"table_leg_{i}").id for i in range(1, 5)]
        self.leg_half_lengths = self.model.geom_size[self.leg_ids, 1].copy()

        self.reset_rng = np.random.RandomState(seed)
        self.dynamics_rng = np.random.RandomState(seed)
        self.visual_rng = np.random.RandomState(seed)
        self.camera_params = np.load(ASSET_ROOT / "common" / "replay_cameras_2.npy")
        self.front_camera_id = self.model.camera("back").id
        self.orig_light_pos = self.model.light_pos.copy()
        self.orig_light_dir = self.model.light_dir.copy()

        # A sphere of radius 1.37 m around each arm base encloses its TCP.
        mujoco.mj_forward(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]
        bases = [self.data.body(f"link0_{side}").xpos.copy() for side in sides]
        tcp_low = [base - 1.37 for base in bases]
        tcp_high = [base + 1.37 for base in bases]
        rot_dim = 3 if action_format == "rotvec" else 4
        rot_limit = np.pi if action_format == "rotvec" else 1.0
        hand_limits = self.model.actuator_ctrlrange[self.finger_actuator_ids]
        # Group each arm pose with its hand, matching the original recordings.
        action_order = (
            np.r_[0:6, 12:28, 6:12, 28:44]
            if action_format == "rotvec"
            else np.r_[0:7, 14:30, 7:14, 30:46]
        )
        self.action_space = spaces.Box(
            low=np.concatenate(
                (
                    *[np.concatenate((low, np.full(rot_dim, -rot_limit))) for low in tcp_low],
                    hand_limits[:, 0],
                )
            )[action_order].astype(np.float32),
            high=np.concatenate(
                (
                    *[np.concatenate((high, np.full(rot_dim, rot_limit))) for high in tcp_high],
                    hand_limits[:, 1],
                )
            )[action_order].astype(np.float32),
            dtype=np.float32,
        )
        self.action_dim = 2 * (3 + rot_dim) + 32
        observation_spaces: dict[str, spaces.Space] = {
            # Right pose, left pose, right fingers, left fingers; quaternions are wxyz.
            "agent_pos": spaces.Box(
                np.concatenate(
                    (
                        tcp_low[0],
                        np.full(4, -1.0),
                        tcp_low[1],
                        np.full(4, -1.0),
                        self.model.jnt_range[self.finger_joint_ids, 0] - FINGER_QPOS_MARGIN,
                    )
                ).astype(np.float32),
                np.concatenate(
                    (
                        tcp_high[0],
                        np.full(4, 1.0),
                        tcp_high[1],
                        np.full(4, 1.0),
                        self.model.jnt_range[self.finger_joint_ids, 1] + FINGER_QPOS_MARGIN,
                    )
                ).astype(np.float32),
                dtype=np.float32,
            ),
        }
        if render_mode != "none":
            observation_spaces["pixels"] = spaces.Dict(
                {
                    camera: spaces.Box(
                        0, 255, shape=(render_height, render_width, 3), dtype=np.uint8
                    )
                    for camera in ("ego", "wrist_left", "wrist_right")
                }
            )
        self.observation_space = spaces.Dict(observation_spaces)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.reset_rng.seed(seed)
            self.dynamics_rng.seed(seed)
            self.visual_rng.seed(seed)
        initial_state = options.get("initial_state") if options is not None else None
        mujoco.mj_resetData(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]
        if initial_state is None:
            delta_h = self.reset_rng.uniform(0.0, 0.05)
            base_pos = self.orig_base_pos.copy()
            base_pos[:2] = self.reset_rng.uniform([-0.25, 0.0], [-0.20, 0.0])
            base_pos[2] += delta_h
        else:
            delta_h = float(initial_state["table_delta_height"])
            base_pos = np.asarray(initial_state["orig_hanoi_base_pos"], dtype=np.float64)
        self.model.body_pos[self.base_body_id] = base_pos
        self.model.body("table").pos[2] = self.orig_table_z + delta_h
        self.model.geom_size[self.leg_ids, 1] = self.leg_half_lengths + delta_h
        # The recorded preset starts with small on A, medium on B, large on C.
        for i, name in enumerate(DISK_NAMES):
            post = self.post_site_ids[2 - i]
            disk_pos = self.data.jnt(f"{name}_joint").qpos[:3]
            disk_pos[:2] = base_pos[:2] + self.model.site_pos[post, :2]
            disk_pos[2] = (
                self.orig_base_pos[2] + delta_h + self.base_upper_top_z + self.disk_half_heights[i]
            )
        self.data.qpos[self.model.jnt_qposadr[self.arm_joint_ids]] = ARM_HOME
        self.data.qpos[self.finger_qpos_ids] = np.tile(FINGER_HOME, 2)
        if self.randomize:
            self.randomize_lighting()
            self.randomize_camera()
            texture = self.visual_rng.choice(TABLE_TEXTURES)
            self.model.geom("table_visual").matid[0] = self.model.material(texture).id
        if self.randomize_dynamics:
            self.model.body_mass[self.disk_body_ids] = (
                self.orig_disk_masses * self.dynamics_rng.uniform(0.75, 1.25, size=3)
            )
        self.state_machine.reset()
        mujoco.mj_forward(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]
        for side, mocap in zip(("right", "left"), self.mocap_ids, strict=True):
            self.data.mocap_pos[mocap] = self.data.sensor(f"franka/flange_pos_{side}").data
            self.data.mocap_quat[mocap] = self.data.sensor(f"franka/flange_quat_{side}").data
        return self.observation(), {
            "is_success": False,
            "initial_state": {
                "orig_hanoi_base_pos": base_pos.copy(),
                "table_delta_height": float(delta_h),
            },
        }

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        assert action.shape == (self.action_dim,), f"Expected {self.action_dim} action values"
        pose_dim = 6 if self.action_format == "rotvec" else 7
        arm_dim = pose_dim + 16
        pose_offsets = (0, arm_dim)
        finger_targets = np.concatenate((action[pose_dim:arm_dim], action[arm_dim + pose_dim:]))
        for offset, mocap in zip(pose_offsets, self.mocap_ids, strict=True):
            pose = action[offset : offset + pose_dim]
            if np.allclose(pose, 0.0):
                # Keep the legacy zero-pose hold behavior used by the recordings.
                continue
            self.data.mocap_pos[mocap] = pose[:3]
            self.data.mocap_quat[mocap] = (
                Rotation.from_rotvec(pose[3:]).as_quat(scalar_first=True)
                if self.action_format == "rotvec"
                else pose[3:]
            )
        for _ in range(int(CONTROL_TIMESTEP // PHYSICS_TIMESTEP)):
            self.apply_arm_control()
            self.data.ctrl[self.finger_actuator_ids] = finger_targets
            mujoco.mj_step(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]
        success, terminated, truncated, info = self.state_machine.update(
            self.data.xpos[self.disk_body_ids],
            np.array([self.data.qvel[dof : dof + 6] for dof in self.disk_dof_ids]),
            self.data.site_xpos[self.post_site_ids],
            self.data.xpos[self.base_body_id, 2] + self.base_upper_top_z,
            self.disk_half_heights,
        )
        if self.render_mode == "human":
            self.sync_human_viewer()
        return self.observation(), float(success), terminated, truncated, info

    def apply_arm_control(self):
        for i, mocap in enumerate(self.mocap_ids):
            self.data.ctrl[self.arm_actuator_ids[i]] = opspace(
                model=self.model,
                data=self.data,
                site_id=self.site_ids[i],
                dof_ids=self.arm_dof_ids[i],
                pos=self.data.mocap_pos[mocap],
                ori=self.data.mocap_quat[mocap],
                joint=ARM_HOME,
                gravity_comp=True,
                pos_gains=(400.0, 400.0, 400.0),
                damping_ratio=4,
            )

    def observation(self) -> dict[str, Any]:
        poses: list[NDArray] = []
        for side in ("right", "left"):
            poses.extend(
                (
                    self.data.sensor(f"franka/flange_pos_{side}").data,
                    self.data.sensor(f"franka/flange_quat_{side}").data,
                )
            )
        fingers = [self.data.sensor(sensor).data[0] for sensor in self.finger_sensor_ids]
        observation: dict[str, Any] = {
            "agent_pos": np.concatenate((*poses, fingers)).astype(np.float32)
        }
        if self.render_mode != "none":
            observation["pixels"] = {
                "ego": self.render_camera("back"),
                "wrist_left": self.render_camera("handcam_rgb_left"),
                "wrist_right": self.render_camera("handcam_rgb_right"),
            }
        return observation

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
        self.model.cam_quat[self.front_camera_id] = Rotation.from_matrix(rotation).as_quat(
            scalar_first=True
        )

    def render(self):
        assert self.render_mode != "none", "Rendering is disabled because render_mode='none'."
        if self.render_mode == "human":
            self.sync_human_viewer()
            return None
        return self.render_camera("back")

    def sync_human_viewer(self):
        # Open the GUI window lazily so rgb_array/none never touch a windowing system.
        if self.human_viewer is None:
            import mujoco.viewer

            self.human_viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.human_viewer.sync()

    def render_camera(self, camera):
        assert self.render_mode != "none", "Rendering is disabled because render_mode='none'."
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
