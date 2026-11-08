from enum import Enum, auto
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

XML_PATH = ASSET_ROOT / "tasks" / "bimanual_unlock_ipad" / "bimanual_unlock_ipad.xml"
MAX_EPISODE_STEPS = 1200
SUCCESS_STEPS = 10
UNLOCK_DELAY_STEPS = 12
UNLOCK_TRANSITION_STEPS = 18


class UnlockState(Enum):
    ENTERING = auto()
    ANIMATING = auto()
    CONFIRMING = auto()
    SUCCEEDED = auto()
    FAILED = auto()


class BimanualUnlockIpadStateMachine:
    """Advance password entry and unlock timing once per control step."""

    def __init__(self, password: list[int] | None = None, screen_effect: bool = False):
        self.password = tuple([1, 2, 3] if password is None else password)
        assert self.password and all(isinstance(d, int) and 0 <= d <= 9 for d in self.password)
        self.screen_effect = screen_effect
        self.reset()

    def reset(self):
        self.state = UnlockState.ENTERING
        self.password_index = 0
        self.state_steps = 0
        self.step_count = 0
        self.failure_reason: str | None = None

    def transition(self, state: UnlockState):
        self.state = state
        self.state_steps = 0

    @property
    def is_success(self) -> bool:
        return self.state is UnlockState.SUCCEEDED

    @property
    def animation_progress(self) -> float:
        if self.state in (UnlockState.CONFIRMING, UnlockState.SUCCEEDED):
            return 1.0
        if self.state is not UnlockState.ANIMATING:
            return 0.0
        t = max(0.0, (self.state_steps - UNLOCK_DELAY_STEPS) / UNLOCK_TRANSITION_STEPS)
        return t * t * (3.0 - 2.0 * t)

    def outcome(self) -> tuple[bool, bool, bool]:
        timeout = self.failure_reason == "timeout"
        terminated = self.is_success or (self.state is UnlockState.FAILED and not timeout)
        return self.is_success, terminated, timeout

    def update(self, digits: list[int]) -> tuple[bool, bool, bool]:
        if self.state in (UnlockState.SUCCEEDED, UnlockState.FAILED):
            return self.outcome()
        self.step_count += 1
        if self.state is UnlockState.ENTERING:
            for digit in digits:
                if digit != self.password[self.password_index]:
                    self.password_index = 0
                    self.failure_reason = "wrong_digit"
                    self.transition(UnlockState.FAILED)
                    break
                self.password_index += 1
                if self.password_index == len(self.password):
                    self.transition(
                        UnlockState.ANIMATING if self.screen_effect else UnlockState.CONFIRMING
                    )
                    break
        # The matching step counts as the first animation or confirmation update,
        # and the last animation step also counts as the first confirmation step.
        if self.state is UnlockState.ANIMATING:
            self.state_steps += 1
            if self.state_steps == UNLOCK_DELAY_STEPS + UNLOCK_TRANSITION_STEPS:
                self.transition(UnlockState.CONFIRMING)
        if self.state is UnlockState.CONFIRMING:
            self.state_steps += 1
            if self.state_steps == SUCCESS_STEPS:
                self.transition(UnlockState.SUCCEEDED)
        if (
            self.step_count >= MAX_EPISODE_STEPS
            and self.state not in (UnlockState.SUCCEEDED, UnlockState.FAILED)
        ):
            self.failure_reason = "timeout"
            self.transition(UnlockState.FAILED)
        return self.outcome()


class BimanualUnlockIpadEnv(gym.Env):
    """Control two Panda-Allegro hands to hold an iPad and enter its password."""

    # Recorded video is 30 Hz; simulation control remains 50 Hz.
    metadata: dict[str, Any] = {  # noqa: RUF012
        "render_modes": ["rgb_array", "human", "none"],
        "render_fps": 30,
    }
    task = "bimanual_unlock_ipad"
    task_description = "Grasp the iPad and enter the password 123 to unlock the device."
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
        password: list[int] | None = None,
        ipad_screen_effect: bool = True,
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
        self.state_machine = BimanualUnlockIpadStateMachine(password, ipad_screen_effect)
        self.task_description = (
            "Grasp the iPad and enter the password "
            + "".join(str(digit) for digit in self.state_machine.password)
            + " to unlock the device."
        )

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
        self.ipad_body_id = self.model.body("ipad_body").id
        self.stand_body_id = self.model.body("ipad_stand").id
        self.orig_ipad_pose = np.concatenate((
            self.model.body_pos[self.ipad_body_id], self.model.body_quat[self.ipad_body_id],
        ))
        self.orig_stand_pose = np.concatenate((
            self.model.body_pos[self.stand_body_id], self.model.body_quat[self.stand_body_id],
        ))
        self.orig_ipad_mass = float(self.model.body_mass[self.ipad_body_id])
        self.orig_table_z = self.model.body("table").pos[2].copy()
        self.leg_ids = [self.model.geom(f"table_leg_{i}").id for i in range(1, 5)]
        self.leg_half_lengths = self.model.geom_size[self.leg_ids, 1].copy()
        self.leg_center_z = self.model.geom_pos[self.leg_ids, 2].copy()

        self.button_ids = [self.model.geom(f"btn{digit}_cyl").id for digit in range(10)]
        self.button_digits = {gid: digit for digit, gid in enumerate(self.button_ids)}
        self.fingertip_ids = {
            self.model.geom(name).id for name in (
                "fingertip_ff_right", "fingertip_mf_right",
                "fingertip_rf_right", "thumbtip_right",
            )
        }
        self.button_off_materials = [self.model.material(f"mat_digit_{d}").id for d in range(10)]
        self.button_on_materials = [self.model.material(f"mat_digit_{d}_light").id for d in range(10)]
        self.button_rgba = self.model.geom_rgba[self.button_ids].copy()
        self.screen_id = self.model.geom("ipad_screen").id
        self.locked_material = int(self.model.geom_matid[self.screen_id])
        self.unlocked_material = self.model.material("mat_screen_unlocked").id
        self.locked_rgba = self.model.mat_rgba[self.locked_material].copy()
        self.unlocked_rgba = self.model.mat_rgba[self.unlocked_material].copy()
        self.previous_pressed: set[int] = set()

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
        # Right arm and hand, then left arm and hand.
        self.action_space = spaces.Box(
            low=np.concatenate(
                [
                    np.concatenate(
                        (
                            tcp_low[i],
                            np.full(rot_dim, -rot_limit),
                            hand_limits[i * 16 : (i + 1) * 16, 0],
                        )
                    )
                    for i in range(2)
                ]
            ).astype(np.float32),
            high=np.concatenate(
                [
                    np.concatenate(
                        (
                            tcp_high[i],
                            np.full(rot_dim, rot_limit),
                            hand_limits[i * 16 : (i + 1) * 16, 1],
                        )
                    )
                    for i in range(2)
                ]
            ).astype(np.float32),
            dtype=np.float32,
        )
        self.action_dim = 2 * (3 + rot_dim) + 32
        observation_spaces: dict[str, spaces.Space] = {
            # Right pose, right fingers, left pose, left fingers; quaternions are wxyz.
            "agent_pos": spaces.Box(
                np.concatenate(
                    (
                        tcp_low[0],
                        np.full(4, -1.0),
                        self.model.jnt_range[self.finger_joint_ids[:16], 0] - FINGER_QPOS_MARGIN,
                        tcp_low[1],
                        np.full(4, -1.0),
                        self.model.jnt_range[self.finger_joint_ids[16:], 0] - FINGER_QPOS_MARGIN,
                    )
                ).astype(np.float32),
                np.concatenate(
                    (
                        tcp_high[0],
                        np.full(4, 1.0),
                        self.model.jnt_range[self.finger_joint_ids[:16], 1] + FINGER_QPOS_MARGIN,
                        tcp_high[1],
                        np.full(4, 1.0),
                        self.model.jnt_range[self.finger_joint_ids[16:], 1] + FINGER_QPOS_MARGIN,
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
            stand_pose = self.orig_stand_pose.copy()
            stand_pose[:2] = self.reset_rng.uniform([-0.35, 0.05], [-0.30, 0.10])
            stand_pose[2] += delta_h
            ipad_pose = self.orig_ipad_pose.copy()
            ipad_pose[:2] += stand_pose[:2] - self.orig_stand_pose[:2]
            ipad_pose[2] += delta_h
        else:
            delta_h = float(initial_state["table_delta_height"])
            stand_pose = np.asarray(initial_state["orig_stand_pose"], dtype=np.float64)
            ipad_pose = np.asarray(initial_state["orig_ipad_pose"], dtype=np.float64)
        self.model.body_pos[self.stand_body_id] = stand_pose[:3]
        self.model.body_quat[self.stand_body_id] = stand_pose[3:]
        self.model.body_pos[self.ipad_body_id] = ipad_pose[:3]
        self.model.body_quat[self.ipad_body_id] = ipad_pose[3:]
        self.data.jnt("ipad_freejoint").qpos[:] = ipad_pose
        self.model.body("table").pos[2] = self.orig_table_z + delta_h
        self.model.geom_size[self.leg_ids, 1] = self.leg_half_lengths + 0.5 * delta_h
        self.model.geom_pos[self.leg_ids, 2] = self.leg_center_z - 0.5 * delta_h
        self.data.qpos[self.model.jnt_qposadr[self.arm_joint_ids]] = ARM_HOME
        self.data.qpos[self.finger_qpos_ids] = np.tile(FINGER_HOME, 2)
        if self.randomize:
            self.randomize_lighting()
            self.randomize_camera()
            texture = self.visual_rng.choice(TABLE_TEXTURES)
            self.model.geom("table_visual").matid[0] = self.model.material(texture).id
        if self.randomize_dynamics:
            self.model.body_mass[self.ipad_body_id] = self.orig_ipad_mass * float(
                self.dynamics_rng.uniform(0.75, 1.25)
            )
        self.state_machine.reset()
        self.previous_pressed = set()
        self.set_button_materials(set())
        self.model.geom_rgba[self.button_ids] = self.button_rgba
        self.model.geom_matid[self.screen_id] = self.locked_material
        self.model.mat_rgba[self.unlocked_material] = self.unlocked_rgba
        mujoco.mj_forward(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]
        for side, mocap in zip(("right", "left"), self.mocap_ids, strict=True):
            self.data.mocap_pos[mocap] = self.data.sensor(f"franka/flange_pos_{side}").data
            self.data.mocap_quat[mocap] = self.data.sensor(f"franka/flange_quat_{side}").data
        return self.observation(), {
            "is_success": False,
            "initial_state": {
                "orig_stand_pose": stand_pose.copy(),
                "orig_ipad_pose": ipad_pose.copy(),
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
        pressed = self.pressed_buttons()
        new_buttons = sorted(pressed - self.previous_pressed)
        self.previous_pressed = pressed
        digits = [self.button_digits[gid] for gid in new_buttons]
        if self.state_machine.state is UnlockState.ENTERING:
            self.set_button_materials(pressed)
        success, terminated, truncated = self.state_machine.update(digits)
        self.update_unlock_visuals()
        info = {"is_success": success, **self.get_unlock_status()}
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

    def pressed_buttons(self) -> set[int]:
        pressed = set()
        for contact in self.data.contact[:self.data.ncon]:
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if g1 in self.button_digits and g2 in self.fingertip_ids:
                pressed.add(g1)
            elif g2 in self.button_digits and g1 in self.fingertip_ids:
                pressed.add(g2)
        return pressed

    def set_button_materials(self, pressed: set[int]):
        for digit, gid in enumerate(self.button_ids):
            materials = self.button_on_materials if gid in pressed else self.button_off_materials
            self.model.geom_matid[gid] = materials[digit]

    def update_unlock_visuals(self):
        machine = self.state_machine
        if machine.state in (UnlockState.ENTERING, UnlockState.FAILED):
            return
        if machine.state is UnlockState.ANIMATING:
            self.set_button_materials({self.button_ids[machine.password[-1]]})
            if machine.state_steps <= UNLOCK_DELAY_STEPS:
                return
        progress = machine.animation_progress
        self.model.geom_matid[self.screen_id] = self.unlocked_material
        rgba = self.locked_rgba * (1.0 - progress) + self.unlocked_rgba * progress
        rgba[3] = self.unlocked_rgba[3]
        self.model.mat_rgba[self.unlocked_material] = rgba
        self.model.geom_rgba[self.button_ids, 3] = self.button_rgba[:, 3] * (1.0 - progress)

    def get_unlock_status(self) -> dict[str, Any]:
        machine = self.state_machine
        return {
            "unlock_state": machine.state.name.lower(),
            "unlock_index": machine.password_index,
            "failure_reason": machine.failure_reason,
            "is_success": machine.is_success,
        }

    def observation(self) -> dict[str, Any]:
        poses: list[NDArray] = []
        for i, side in enumerate(("right", "left")):
            poses.extend(
                (
                    self.data.sensor(f"franka/flange_pos_{side}").data,
                    self.data.sensor(f"franka/flange_quat_{side}").data,
                    np.array(
                        [
                            self.data.sensor(sensor).data[0]
                            for sensor in self.finger_sensor_ids[i * 16 : (i + 1) * 16]
                        ]
                    ),
                )
            )
        observation: dict[str, Any] = {"agent_pos": np.concatenate(poses).astype(np.float32)}
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
