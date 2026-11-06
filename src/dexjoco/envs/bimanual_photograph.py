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

XML_PATH = ASSET_ROOT / "tasks" / "bimanual_photograph" / "bimanual_photograph.xml"
MAX_EPISODE_STEPS = 1000
TARGET_RADIUS = 0.16
TARGET_HALF_HEIGHT = 0.10
TARGET_ANGLE = 10.0
TARGET_OFFSET = np.array([-0.40, 0.0, 0.0])
FLASH_FRAMES = 5


class BimanualPhotographStateMachine:
    """Require the camera position, viewing direction, and shutter contact together."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.step_count = 0

    def update(self, local_position, axis, shutter_count: int) -> tuple[bool, bool, bool, dict]:
        radial_dist = float(np.linalg.norm(local_position[:2]))
        height_dist = float(abs(local_position[2]))
        region_pass = radial_dist <= TARGET_RADIUS and height_dist <= TARGET_HALF_HEIGHT
        axis = axis / np.linalg.norm(axis)
        angle = float(np.degrees(np.arccos(np.clip(axis[0], -1.0, 1.0))))
        angle_pass = angle <= TARGET_ANGLE
        shutter_pressed = shutter_count > 0
        success = region_pass and angle_pass and shutter_pressed
        self.step_count += 1
        return success, success, self.step_count >= MAX_EPISODE_STEPS, {
            "is_success": success,
            "region_pass": region_pass,
            "angle_pass": angle_pass,
            "shutter_pressed": shutter_pressed,
            "radial_dist": radial_dist,
            "region_height_dist": height_dist,
            "view_angle_deg": angle,
            "shutter_contact_count": shutter_count,
        }


class BimanualPhotographEnv(gym.Env):
    """Control two Panda-Allegro hands to aim a camera and press its shutter."""

    # Recorded video is 30 Hz; simulation control remains 50 Hz.
    metadata: dict[str, Any] = {  # noqa: RUF012
        "render_modes": ["rgb_array", "human", "none"],
        "render_fps": 30,
    }
    task = "bimanual_photograph"
    task_description = (
        "Grasp the camera with the left hand, align it with the logo, "
        "and press the shutter button with the right hand."
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
        camera_screen_effect: bool = False,
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
        self.camera_screen_effect = camera_screen_effect
        self.model = mujoco.MjModel.from_xml_path(str(XML_PATH))  # pyright: ignore[reportAttributeAccessIssue]
        if offsamples is not None:
            self.model.vis.quality.offsamples = offsamples
        assert self.model.opt.timestep == PHYSICS_TIMESTEP
        self.data = mujoco.MjData(self.model)  # pyright: ignore[reportAttributeAccessIssue]
        self.render_width = render_width
        self.render_height = render_height
        self.renderer: mujoco.Renderer | None = None
        self.human_viewer = None
        self.state_machine = BimanualPhotographStateMachine()

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
        self.camera_body_id = self.model.body("link_19").id
        self.orig_camera_pose = np.concatenate((
            self.model.body_pos[self.camera_body_id], self.model.body_quat[self.camera_body_id],
        ))
        self.orig_camera_mass = float(self.model.body_mass[self.camera_body_id])
        self.logo_geom_id = self.model.geom("photograph_logo_disk").id
        self.orig_logo_pose = np.concatenate((
            self.model.geom_pos[self.logo_geom_id], self.model.geom_quat[self.logo_geom_id],
        ))
        self.region_geom_id = self.model.geom("photograph_target_region_visual").id
        self.axis_geom_id = self.model.geom("camera_axis_z").id
        self.view_site_id = self.model.site("view_direction_site").id
        self.shutter_geom_id = self.model.geom("camera_body_top_cap").id
        camera_bodies = {self.camera_body_id}
        for body in range(self.camera_body_id + 1, self.model.nbody):
            if int(self.model.body_parentid[body]) in camera_bodies:
                camera_bodies.add(body)
        self.camera_geom_ids = {
            geom for geom in range(self.model.ngeom)
            if int(self.model.geom_bodyid[geom]) in camera_bodies
        }
        self.render_options = mujoco.MjvOption()  # pyright: ignore[reportAttributeAccessIssue]
        self.render_options.geomgroup[5] = 0
        self.screen_texture_id = self.model.texture("camera_screen_tex").id
        self.screen_width = int(self.model.tex_width[self.screen_texture_id])
        self.screen_height = int(self.model.tex_height[self.screen_texture_id])
        self.screen_address = int(self.model.tex_adr[self.screen_texture_id])
        self.screen_channels = int(self.model.tex_nchannel[self.screen_texture_id])
        self.captured_image: NDArray[np.uint8] | None = None
        self.flash_frame = 0
        self.latest_shutter_pressed = False
        if not camera_screen_effect:
            screen = self.model.geom("camera_live_screen")
            screen.matid[0] = -1
            screen.rgba[:] = [0.0, 0.0, 0.0, 0.0]
            screen.size[:] = [1e-9, 1e-9, 1e-9]
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
            logo_pose = self.orig_logo_pose.copy()
            logo_pose[1:3] = self.reset_rng.uniform([-0.10, 1.22], [0.10, 1.38])
            camera_pose = self.orig_camera_pose.copy()
            camera_pose[:2] = self.reset_rng.uniform([-0.30, 0.1], [-0.20, 0.2])
            camera_pose[2] += delta_h + 0.002
            initial_hand_joint = np.tile(FINGER_HOME, 2)
        else:
            delta_h = float(initial_state["table_delta_height"])
            logo_pose = np.asarray(initial_state["orig_logo_pose"], dtype=np.float64)
            camera_pose = np.asarray(initial_state["orig_camera_pose"], dtype=np.float64)
            initial_hand_joint = np.asarray(
                initial_state["initial_hand_joint"], dtype=np.float64
            )
        self.model.geom_pos[self.logo_geom_id] = logo_pose[:3]
        self.model.geom_quat[self.logo_geom_id] = logo_pose[3:7]
        self.model.geom_pos[self.region_geom_id] = logo_pose[:3] + TARGET_OFFSET
        self.model.geom_size[self.region_geom_id, :2] = [TARGET_RADIUS, TARGET_HALF_HEIGHT]
        self.data.jnt("camera_root").qpos[:] = camera_pose
        self.captured_image = None
        self.flash_frame = 0
        self.latest_shutter_pressed = False
        self.model.body("table").pos[2] = self.orig_table_z + delta_h
        self.model.geom_size[self.leg_ids, 1] = self.leg_half_lengths + delta_h
        self.data.qpos[self.model.jnt_qposadr[self.arm_joint_ids]] = ARM_HOME
        self.data.qpos[self.finger_qpos_ids] = initial_hand_joint
        if self.randomize:
            self.randomize_lighting()
            self.randomize_camera()
            texture = self.visual_rng.choice(TABLE_TEXTURES)
            self.model.geom("table_visual").matid[0] = self.model.material(texture).id
        if self.randomize_dynamics:
            self.model.body_mass[self.camera_body_id] = self.orig_camera_mass * float(
                self.dynamics_rng.uniform(0.75, 1.25)
            )
        self.state_machine.reset()
        mujoco.mj_forward(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]
        for side, mocap in zip(("right", "left"), self.mocap_ids, strict=True):
            self.data.mocap_pos[mocap] = self.data.sensor(f"franka/flange_pos_{side}").data
            self.data.mocap_quat[mocap] = self.data.sensor(f"franka/flange_quat_{side}").data
        return self.observation(), {
            "is_success": False,
            "initial_state": {
                "orig_camera_pose": camera_pose.copy(),
                "orig_logo_pose": logo_pose.copy(),
                "table_delta_height": float(delta_h),
                "initial_hand_joint": initial_hand_joint.copy(),
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
        region = self.data.geom_xpos[self.region_geom_id]
        rotation = self.data.geom_xmat[self.region_geom_id].reshape(3, 3)
        local_position = rotation.T @ (self.data.site_xpos[self.view_site_id] - region)
        axis = self.data.geom_xmat[self.axis_geom_id].reshape(3, 3)[:, 2].copy()
        shutter_count = int(sum(
            (contact.geom1 == self.shutter_geom_id and contact.geom2 not in self.camera_geom_ids)
            or (contact.geom2 == self.shutter_geom_id and contact.geom1 not in self.camera_geom_ids)
            for contact in self.data.contact[:self.data.ncon]
        ))
        success, terminated, truncated, info = self.state_machine.update(
            local_position, axis, shutter_count,
        )
        self.latest_shutter_pressed = info["shutter_pressed"]
        observation = self.observation()
        if self.render_mode == "human":
            self.sync_human_viewer()
        return observation, float(success), terminated, truncated, info

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
            self.update_screen_texture()
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
        self.update_screen_texture()
        return self.render_camera("back")

    def sync_human_viewer(self):
        # Open the GUI window lazily so rgb_array/none never touch a windowing system.
        if self.human_viewer is None:
            import mujoco.viewer

            self.human_viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.update_human_viewer_visuals()
        self.human_viewer.sync()

    def update_human_viewer_visuals(self):
        assert self.human_viewer is not None
        self.update_screen_texture()
        self.human_viewer.opt.geomgroup[5] = int(
            self.human_viewer.cam.type == mujoco.mjtCamera.mjCAMERA_FREE  # pyright: ignore[reportAttributeAccessIssue]
        )

    def render_camera(self, camera):
        assert self.render_mode != "none", "Rendering is disabled because render_mode='none'."
        if self.renderer is None:
            self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, self.render_width)
            self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, self.render_height)
            self.renderer = mujoco.Renderer(
                self.model, height=self.render_height, width=self.render_width
            )
        self.renderer.update_scene(self.data, camera=camera, scene_option=self.render_options)
        return self.renderer.render().copy()

    def fit_screen_image(self, image):
        height, width = image.shape[:2]
        aspect = self.screen_width / self.screen_height
        if abs(width / height - aspect) > 1e-3:
            if width / height > aspect:
                cropped_width = round(height * aspect)
                left = (width - cropped_width) // 2
                image = image[:, left:left + cropped_width]
            else:
                cropped_height = round(width / aspect)
                top = (height - cropped_height) // 2
                image = image[top:top + cropped_height]
            height, width = image.shape[:2]
        rows = np.linspace(0, height - 1, self.screen_height).astype(np.int64)
        columns = np.linspace(0, width - 1, self.screen_width).astype(np.int64)
        return image[np.ix_(rows, columns)]

    def update_screen_texture(self):
        if not self.camera_screen_effect or self.render_mode == "none":
            return
        if self.captured_image is None:
            image = self.fit_screen_image(self.render_camera("cam_pos_z"))
            if self.latest_shutter_pressed:
                self.captured_image = image.copy()
                self.flash_frame = 0
        else:
            image = self.captured_image
        if self.captured_image is not None and self.flash_frame < FLASH_FRAMES:
            alpha = 1.0 - self.flash_frame / FLASH_FRAMES
            image = np.clip(
                alpha * 255.0 + (1.0 - alpha) * image.astype(np.float32), 0.0, 255.0,
            ).astype(np.uint8)
            self.flash_frame += 1
        pixels = np.ascontiguousarray(image[:, :, :self.screen_channels]).reshape(-1)
        self.model.tex_data[self.screen_address:self.screen_address + pixels.size] = pixels
        assert self.renderer is not None
        if self.renderer._gl_context is not None:
            self.renderer._gl_context.make_current()
        context = self.renderer._mjr_context
        assert context is not None
        mujoco.mjr_uploadTexture(self.model, context, self.screen_texture_id)  # pyright: ignore[reportAttributeAccessIssue]
        if self.human_viewer is not None:
            self.human_viewer.update_texture(self.screen_texture_id)

    def close(self):
        if self.human_viewer is not None:
            self.human_viewer.close()
            self.human_viewer = None
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
