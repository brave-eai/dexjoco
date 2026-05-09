"""MuJoCo water_plant task used by the simulated teleoperation collector."""

import random
import time
from pathlib import Path
from typing import Literal

import mujoco
import numpy as np
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R

from ..controllers import opspace
from ..mujoco_gym_env import MujocoGymEnv
from ..rendering import MujocoRenderer

_HERE = Path(__file__).parent
_XML_PATH = _HERE / "xmls" / "arena_arm_hand_plant.xml"
_PANDA_HOME = np.asarray((0, -0.785, 0, -2.35, 0, 1.57, np.pi / 4))  # Origin
_ALLEGRO_HOME = np.asarray(
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.263, 0, 0, 0), dtype=np.float32
)
_SPRAY_SAMPLE_LOW = np.array([-0.35, -0.25], dtype=np.float64)
_SPRAY_SAMPLE_HIGH = np.array([-0.30, -0.20], dtype=np.float64)
_PLANT_SAMPLE_LOW = np.array([-0.10, 0.15], dtype=np.float64)
_PLANT_SAMPLE_HIGH = np.array([-0.05, 0.20], dtype=np.float64)
_MAX_EPISODE_STEPS = 1000
_CONE_VISIBLE_STEPS = 30
_TRIGGER_RELEASE_THRESHOLD = 0.25
_TRIGGER_PULL_THRESHOLD = 0.34
_SUCCESS_STEPS_REQUIRED = 30

_ALLEGRO_JOINT_NAMES = (
    "ffj0",
    "ffj1",
    "ffj2",
    "ffj3",
    "mfj0",
    "mfj1",
    "mfj2",
    "mfj3",
    "rfj0",
    "rfj1",
    "rfj2",
    "rfj3",
    "thj0",
    "thj1",
    "thj2",
    "thj3",
)

_ALLEGRO_ACTUATOR_NAMES = (
    "ffa0",
    "ffa1",
    "ffa2",
    "ffa3",
    "mfa0",
    "mfa1",
    "mfa2",
    "mfa3",
    "rfa0",
    "rfa1",
    "rfa2",
    "rfa3",
    "tha0",
    "tha1",
    "tha2",
    "tha3",
)

_ALLEGRO_SENSOR_NAMES = (
    "allegro_right/ffj0_pos",
    "allegro_right/ffj1_pos",
    "allegro_right/ffj2_pos",
    "allegro_right/ffj3_pos",
    "allegro_right/mfj0_pos",
    "allegro_right/mfj1_pos",
    "allegro_right/mfj2_pos",
    "allegro_right/mfj3_pos",
    "allegro_right/rfj0_pos",
    "allegro_right/rfj1_pos",
    "allegro_right/rfj2_pos",
    "allegro_right/rfj3_pos",
    "allegro_right/thj0_pos",
    "allegro_right/thj1_pos",
    "allegro_right/thj2_pos",
    "allegro_right/thj3_pos",
)

_TABLE_LEG_NAMES = (
    "table_leg_1",
    "table_leg_2",
    "table_leg_3",
    "table_leg_4",
)

# Number of Allegro joints expected
_N_ALLEGRO = len(_ALLEGRO_JOINT_NAMES)


class PandaWaterPlantGymEnv(MujocoGymEnv):
    def __init__(
        self,
        render_mode: Literal["rgb_array", "human", "none"],
        randomize: bool,
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        hz=30,
        randomize_dynamics: bool = False,
    ):
        self.hz = hz
        self.randomize = randomize
        self.randomize_dynamics = randomize_dynamics
        self.image_obs = render_mode != "none"

        super().__init__(
            xml_path=_XML_PATH, seed=seed, control_dt=control_dt, physics_dt=physics_dt
        )

        # Seed the RNGs used by environment randomization.
        random.seed(seed)
        np.random.seed(seed)

        self.render_mode = render_mode
        self.env_step = 0

        # Panda caches
        self._panda_dof_ids = np.asarray(
            [self._model.joint(f"joint{i}").id for i in range(1, 8)]
        )
        self._panda_ctrl_ids = np.asarray(
            [self._model.actuator(f"actuator{i}").id for i in range(1, 8)]
        )

        self._site_id = self._model.site("attachment_site").id

        self._allegro_dof_ids = np.asarray(
            [int(self._model.joint(n).qposadr) for n in _ALLEGRO_JOINT_NAMES],
            dtype=int,
        )

        # Get actuator ids
        self._allegro_ctrl_ids = np.asarray(
            [self._model.actuator(name).id for name in _ALLEGRO_ACTUATOR_NAMES],
            dtype=int,
        )

        # find qpos addresses for Allegro joints (None if missing)
        for name in _ALLEGRO_JOINT_NAMES:
            jid = int(self._model.joint(name).id)
            self._model.jnt_qposadr[jid]

        self._mj_viewer = None
        if self.image_obs:
            self._mj_viewer = MujocoRenderer(self.model, self.data)
            self._mj_viewer.render(self.render_mode)

        # --- GROUP-5 handling: record which geoms are in group 5 ---
        # model.geom_group should be an array-like
        geom_groups = np.asarray(self._model.geom_group)
        self._group5_geom_ids = np.where(geom_groups == 5)[0].tolist()

        self._cone_alpha_counter = 0
        self._trigger_pulled = False
        self._success_counter = 0

        # Fixed outputs: wrist camera + one random preset free camera.
        self._front_camera_id = int(self._model.camera("front").id)
        self._wrist_camera_id = int(self._model.camera("handcam_rgb").id)
        self._camera_params = np.load(_HERE / "replay_cameras.npy")
        self._num_preset_cameras = int(self._camera_params.shape[0])

        self._scene_center = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        self._table_body_z0 = float(self._model.body("table").pos[2])
        self._table_leg_half_len0 = {
            name: float(self._model.geom(name).size[1]) for name in _TABLE_LEG_NAMES
        }

        self._plant_body_z0 = float(self._model.body("plant").pos[2])
        self._spray_body_z0 = float(self._model.body("link_2").pos[2])

        # Keep pristine lighting values so reset randomization never drifts.
        self._orig_light_pos = self._model.light_pos.copy()
        self._orig_light_dir = self._model.light_dir.copy()

        self._texture_names = [
            "table_bamboo",
            "table_blue-wood",
            "table_brass-ambra",
            "table_ceramic",
            "table_cream-plaster",
            "table_dark_wood_planks_2",
            "table_dark-wood",
            "table_gray-plaster",
            "table_gray_wood_planks",
            "table_light-wood",
            "table_metal",
            "table_pink-plaster",
            "table_red-wood",
            "table_legs_metal",
            "table_steel-scratched",
            "table_walnut_wood_grain",
            "table_warm_wood_grain_2",
            "table_white-plaster",
            "table_wood_grain_1",
            "table_yellow-plaster",
        ]

        image_h = int(self._model.vis.global_.offheight)
        image_w = int(self._model.vis.global_.offwidth)

        self.observation_space = spaces.Dict(  # type: ignore[assignment]
            {
                "state": spaces.Dict(
                    {
                        "tcp_pose": spaces.Box(
                            -np.inf, np.inf, shape=(7,), dtype=np.float64
                        ),
                        "gripper_pose": spaces.Box(
                            -np.inf, np.inf, shape=(16,), dtype=np.float64
                        ),
                        "spray_ori_pose": spaces.Box(
                            -np.inf, np.inf, shape=(7,), dtype=np.float64
                        ),
                        "plant_ori_pose": spaces.Box(
                            -np.inf, np.inf, shape=(7,), dtype=np.float64
                        ),
                        "table_delta_height": spaces.Box(
                            -np.inf, np.inf, shape=(1,), dtype=np.float64
                        ),
                    }
                ),
            }
        )
        if self.image_obs:
            self.observation_space["images"] = spaces.Dict(
                {
                    "wrist": spaces.Box(
                        0, 255, shape=(image_h, image_w, 3), dtype=np.uint8
                    ),
                    "random_camera" if self.randomize else "front": spaces.Box(
                        0, 255, shape=(image_h, image_w, 3), dtype=np.uint8
                    ),
                }
            )

        self.action_space = spaces.Box(  # type: ignore[assignment]
            low=np.full(7 + _N_ALLEGRO, -1.0, dtype=np.float32),
            high=np.full(7 + _N_ALLEGRO, 1.0, dtype=np.float32),
            dtype=np.float32,
        )

        self._table_geom_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_GEOM, "table_visual"
        )

        # for spray dynamics randomization
        self._init_spray_dynamics_cache()

    # --------------------------
    # Helper methods for group manipulation
    # --------------------------
    def _set_group_for_geom_ids(self, geom_ids, new_group: int):
        """Set model.geom_group[geom_id] = new_group for given ids (in-place)."""
        for gid in geom_ids:
            self._model.geom_group[gid] = new_group

    def _temporarily_show_group5_in_gui(self):
        """Temporarily move group5 geoms to group0 (so GUI shows them)."""
        # set to 0 for GUI visibility
        self._set_group_for_geom_ids(self._group5_geom_ids, 0)

    def _restore_group5(self):
        """Restore group5 geoms to original group (5)."""
        # restore to 5
        self._set_group_for_geom_ids(self._group5_geom_ids, 5)

    def randomize_lighting(self):
        model = self._model

        # Start from baseline each reset to avoid cumulative offsets.
        model.light_pos[:] = self._orig_light_pos
        model.light_dir[:] = self._orig_light_dir

        for i in range(model.nlight):
            model.light_pos[i, 0] += random.uniform(-0.3, 0.3)
            model.light_pos[i, 1] += random.uniform(-0.3, 0.3)
            model.light_dir[i, 0] += random.uniform(-0.4, 0.4)
            model.light_dir[i, 1] += random.uniform(-0.4, 0.4)
            model.light_diffuse[i] = [random.uniform(0.3, 0.8) for _ in range(3)]

        model.vis.headlight.ambient[:] = [random.uniform(0.3, 0.7) for _ in range(3)]
        model.vis.headlight.diffuse[:] = [random.uniform(0.2, 0.6) for _ in range(3)]

    def randomize_desktop_texture(self):
        chosen_texture = random.choice(self._texture_names)
        # print(f"Chosen desktop texture: {chosen_texture}")
        mat_id = self._model.material(chosen_texture).id
        # print(f"Randomized desktop texture to {chosen_texture} (mat_id={mat_id})")
        self._model.geom_matid[self._table_geom_id] = mat_id

    def randomize_camera(self):
        """Randomly choose one preset camera from random_cameras.npy."""
        random_camera_idx = random.randint(0, self._num_preset_cameras - 1)
        self._apply_random_camera_to_front(random_camera_idx)

    def _init_spray_dynamics_cache(self) -> None:
        self._spray_joint_id = int(self._model.joint("joint_0").id)
        self._spray_dof_id = int(self._model.jnt_dofadr[self._spray_joint_id])
        self._spray_body_id = int(self._model.body("link_2").id)

        self._spray_joint_frictionloss0 = float(
            self._model.dof_frictionloss[self._spray_dof_id]
        )
        self._spray_joint_stiffness0 = float(
            self._model.jnt_stiffness[self._spray_joint_id]
        )
        self._spray_body_mass0 = float(self._model.body_mass[self._spray_body_id])

        # Randomization ranges.
        self._spray_dyn_friction_range = (0.0, 0.05)
        self._spray_dyn_stiffness_mul = (0.75, 1.25)
        self._spray_dyn_mass_mul = (0.75, 1.25)

    def _randomize_spray_dynamics(self) -> None:
        def _mul(rng):
            return float(np.random.uniform(rng[0], rng[1]))

        frictionloss = float(
            np.random.uniform(
                self._spray_dyn_friction_range[0], self._spray_dyn_friction_range[1]
            )
        )
        stiffness = self._spray_joint_stiffness0 * _mul(self._spray_dyn_stiffness_mul)
        mass = self._spray_body_mass0 * _mul(self._spray_dyn_mass_mul)

        self._model.dof_frictionloss[self._spray_dof_id] = frictionloss
        self._model.jnt_stiffness[self._spray_joint_id] = stiffness

        self._model.body_mass[self._spray_body_id] = mass

    def _prime_rgb_array_renderer(self):
        """Discard one offscreen frame per camera to avoid stale first-reset images."""
        if self._mj_viewer is None:
            return
        self._mj_viewer.render(render_mode="rgb_array", camera_id=self._wrist_camera_id)
        self._mj_viewer.render(render_mode="rgb_array", camera_id=self._front_camera_id)

    def _apply_random_camera_to_front(self, camera_idx):
        camera = self._camera_params[camera_idx]
        azimuth = float(camera[0])
        elevation = float(
            -camera[1]
        )  # CAMERA_PARAMS are designed for mujoco where elevation is negative
        distance = float(camera[2])

        azim_rad = np.deg2rad(azimuth)
        elev_rad = np.deg2rad(elevation)

        cam_offset = np.array(
            [
                -distance * np.cos(elev_rad) * np.cos(azim_rad),
                distance * np.cos(elev_rad) * np.sin(azim_rad),
                distance * np.sin(elev_rad),
            ],
            dtype=np.float64,
        )
        cam_pos = self._scene_center + cam_offset

        # Forward direction: camera points to scene center.
        forward = -cam_offset
        forward /= np.linalg.norm(forward)

        # World-space up direction.
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        # Camera right = forward x up.
        right = np.cross(forward, world_up)
        right /= np.linalg.norm(right)

        # Camera up = right x forward (keep basis orthonormal).
        up = np.cross(right, forward)

        # Build rotation matrix with columns [right, up, -forward].
        # MuJoCo camera convention: X=right, Y=up, Z=back (-forward).
        rot_matrix = np.column_stack([right, up, -forward])

        cam_quat_wxyz = R.from_matrix(rot_matrix).as_quat(scalar_first=True)

        self._model.cam_pos[self._front_camera_id] = cam_pos
        self._model.cam_quat[self._front_camera_id] = cam_quat_wxyz

    def reset(self):
        """Reset the environment."""

        mujoco.mj_resetData(self._model, self._data)  # type: ignore
        # reset table height
        self.delta_h = np.random.uniform(0.0, 0.05)

        # Move the whole table body (absolute)
        table_body_id = self._model.body("table").id
        self._model.body_pos[table_body_id, 2] = self._table_body_z0 + self.delta_h

        # Adjust legs (absolute): extend so feet stay on floor
        for lname in _TABLE_LEG_NAMES:
            lgid = self._model.geom(lname).id
            self._model.geom_size[lgid, 1] = (
                self._table_leg_half_len0[lname] + self.delta_h
            )

        # reset cone_visual alpha to 0 (as before)
        geom_id = self._model.geom("cone_visual").id
        self._model.geom_rgba[geom_id][3] = 0

        # Sample a new spray position.
        spray_xy = np.random.uniform(_SPRAY_SAMPLE_LOW, _SPRAY_SAMPLE_HIGH)
        spray_ori_pos = self._model.body("link_2").pos
        spray_ori_quat = self._model.body("link_2").quat
        spray_ori_pos[:2] = spray_xy
        spray_ori_pos[2] = self._spray_body_z0 + self.delta_h
        self._data.jnt("spray_root").qpos[:3] = spray_ori_pos
        self._spray_ori_pose = np.concatenate(
            [spray_ori_pos.copy(), spray_ori_quat.copy()]
        )

        # Sample a new plant position.
        body_id = self._model.body("plant").id
        plant_xy = np.random.uniform(_PLANT_SAMPLE_LOW, _PLANT_SAMPLE_HIGH)
        plant_ori_pos = self._model.body("plant").pos
        plant_ori_quat = self._model.body("plant").quat
        plant_ori_pos[:2] = plant_xy
        plant_ori_pos[2] = self._plant_body_z0 + self.delta_h
        self._model.body_pos[body_id] = plant_ori_pos
        self._plant_ori_pose = np.concatenate(
            [plant_ori_pos.copy(), plant_ori_quat.copy()]
        )

        # Reset arm to home position.
        self._data.qpos[self._panda_dof_ids] = _PANDA_HOME
        self._data.qpos[self._allegro_dof_ids] = _ALLEGRO_HOME

        mujoco.mj_forward(self._model, self._data)  # type: ignore

        # Reset mocap body to home position and orientation.
        tcp_pos = self._data.sensor("franka/flange_pos").data
        tcp_quat = self._data.sensor("franka/flange_quat").data
        self._data.mocap_pos[0] = tcp_pos
        self._data.mocap_quat[0] = tcp_quat

        self.env_step = 0
        self._trigger_pulled = False
        self._cone_alpha_counter = 0
        self._success_counter = 0

        if self.randomize:
            self.randomize_lighting()
            self.randomize_camera()
            self.randomize_desktop_texture()

        if self.randomize_dynamics:
            self._randomize_spray_dynamics()

        # forward after rerandomization to correct firsst frame
        mujoco.mj_forward(self._model, self._data)  # type: ignore

        # Ensure renderer and scene are synchronized before producing reset observation.
        self._prime_rgb_array_renderer()

        obs = self._compute_observation()

        return obs, {"succeed": False}

    def step(self, action: np.ndarray):
        start_time = time.time()

        xyz = action[:3]
        wxyz_quat = action[3:7]

        if action.shape[0] >= 7 + _N_ALLEGRO:
            allegro_angles = np.asarray(action[7 : 7 + _N_ALLEGRO], dtype=np.float64)
        else:
            allegro_angles = np.zeros(_N_ALLEGRO, dtype=np.float64)

        if not (np.allclose(xyz, 0.0) and np.allclose(wxyz_quat, 0.0)):
            self._data.mocap_pos[0] = xyz
            self._data.mocap_quat[0] = wxyz_quat

        for _ in range(self._n_substeps):
            tau = opspace(
                model=self._model,
                data=self._data,
                site_id=self._site_id,
                dof_ids=self._panda_dof_ids,
                pos=self._data.mocap_pos[0],
                ori=self._data.mocap_quat[0],
                joint=_PANDA_HOME,
                gravity_comp=True,
                pos_gains=(400.0, 400.0, 400.0),
                damping_ratio=4,
            )
            self._data.ctrl[self._panda_ctrl_ids] = tau

            self._data.ctrl[self._allegro_ctrl_ids] = allegro_angles

            mujoco.mj_step(self._model, self._data)  # type: ignore

        # cone alpha logic
        watering_joint_0_pos = float(self._data.sensor("spray_joint_0_pos").data)

        was_trigger_pulled = self._trigger_pulled
        if watering_joint_0_pos < _TRIGGER_RELEASE_THRESHOLD:
            self._trigger_pulled = False
        elif watering_joint_0_pos > _TRIGGER_PULL_THRESHOLD:
            self._trigger_pulled = True

        # only show cone for a short burst on the rising edge of the trigger
        if self._trigger_pulled and not was_trigger_pulled:
            self._cone_alpha_counter = _CONE_VISIBLE_STEPS

        if self._cone_alpha_counter > 0:
            alpha = 0.5
            self._cone_alpha_counter -= 1
        else:
            alpha = 0.0

        geom_id = self._model.geom("cone_visual").id
        rgba = self._model.geom_rgba[geom_id].copy()
        rgba[3] = alpha
        self._model.geom_rgba[geom_id] = rgba

        obs = self._compute_observation()

        self.env_step += 1
        terminated = self.env_step >= _MAX_EPISODE_STEPS

        # ---- human rendering: temporarily show group5 geoms in GUI only ----
        if self.render_mode == "human":
            # Temporarily map group5 -> group0 so GUI shows them
            self._temporarily_show_group5_in_gui()
            try:
                # Render human via MujocoRenderer (gym viewer)
                self._mj_viewer.render("human")
            finally:
                # Restore original groups even if rendering raises
                self._restore_group5()

        dt = time.time() - start_time
        time.sleep(max(0, (1.0 / self.hz) - dt))
        success, failed_trigger_outside = self._compute_success()
        rew = 1.0 if success else 0.0
        terminated = terminated or success or failed_trigger_outside

        return obs, rew, terminated, False, {"succeed": success, "grasp_penalty": 0.0}

    def _compute_success(self):
        site_id = self._model.site("ref_point").id
        p_world = self._data.site_xpos[site_id]
        obj_world_pos = self._data.body("plant").xpos

        R = 0.2
        HALF_H = 0.2
        dx, dy, dz = p_world - obj_world_pos
        inside = (dx * dx + dy * dy <= R * R) and (-HALF_H <= dz <= HALF_H)

        failed_trigger_outside = (not inside) and self._trigger_pulled

        if inside and self._trigger_pulled:
            self._success_counter += 1
        else:
            self._success_counter = 0

        success = self._success_counter >= _SUCCESS_STEPS_REQUIRED
        return success, failed_trigger_outside

    def render(self):
        if self._mj_viewer is None:
            raise RuntimeError("Rendering is disabled because render_mode='none'.")
        wrist_frame = self._mj_viewer.render(
            render_mode="rgb_array", camera_id=self._wrist_camera_id
        )
        third_person_frame = self._mj_viewer.render(
            render_mode="rgb_array", camera_id=self._front_camera_id
        )
        return [wrist_frame, third_person_frame]

    # Helper methods.
    def _compute_observation(self) -> dict:
        obs = {}
        obs["state"] = {}

        tcp_pos = self._data.sensor("franka/flange_pos").data
        tcp_quat = self._data.sensor("franka/flange_quat").data
        tcp_pose = np.concatenate([tcp_pos, tcp_quat])

        allegro_qpos = np.array(
            [self._data.sensor(name).data for name in _ALLEGRO_SENSOR_NAMES],
            dtype=np.float32,
        )

        if self.image_obs:
            obs["images"] = {}
            (
                obs["images"]["wrist"],
                obs["images"]["random_camera" if self.randomize else "front"],
            ) = self.render()

        obs["state"] = {
            "tcp_pose": tcp_pose,
            "gripper_pose": allegro_qpos,
            "spray_ori_pose": self._spray_ori_pose,
            "plant_ori_pose": self._plant_ori_pose,
            "table_delta_height": np.array([self.delta_h], dtype=np.float32),
        }

        return obs

    def close(self):
        viewer = getattr(self, "_mj_viewer", None)
        if viewer is not None:
            try:
                viewer.close()
            except Exception:
                # Cleanup must be best-effort at shutdown.
                pass

        super().close()

    def get_end_effector_pose_matrix(self) -> np.ndarray:
        pos = self._data.mocap_pos[0]
        quat = self._data.mocap_quat[0]
        quat = np.array([quat[1], quat[2], quat[3], quat[0]])
        rot_mat = R.from_quat(quat).as_matrix()
        T = np.eye(4)
        T[:3, :3] = rot_mat
        T[:3, 3] = pos
        return T
