import random
import time
from pathlib import Path
from typing import Any, Dict, Literal, Tuple

import mujoco
import numpy as np
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R

from ..controllers import opspace
from ..mujoco_gym_env import GymRenderingSpec, MujocoGymEnv
from ..rendering import MujocoRenderer

_HERE = Path(__file__).parent
_XML_PATH = _HERE / "xmls" / "arena_arm_hand_ipad.xml"
_PANDA_HOME = np.asarray((0, -0.785, 0, -2.35, 0, 1.57, np.pi / 4))  # Origin
_ALLEGRO_HOME = np.asarray(
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.263, 0, 0, 0), dtype=np.float32
)
_CARTESIAN_BOUNDS = np.asarray([[-0.8, -0.8, -0.8], [0.8, 0.8, 0.8]])
_STAND_SAMPLING_BOUNDS = np.array([[-0.35, 0.05], [-0.30, 0.1]])
_N_ALLEGRO = 16


class PandaBimanualUnlockIpadGymEnv(MujocoGymEnv):
    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        action_scale: np.ndarray = np.asarray([0.1, 1]),
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        time_limit: float = 10.0,
        render_spec: GymRenderingSpec = GymRenderingSpec(),
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = True,
        randomize: bool = False,
        randomize_dynamics: bool = False,
        config=None,
        hz=30,
        password: list[int] | None = None,
    ):
        self.hz = 30
        self._action_scale = action_scale
        self.randomize = randomize
        self._randomize_dynamics = randomize_dynamics

        super().__init__(
            xml_path=_XML_PATH,
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            time_limit=time_limit,
            render_spec=render_spec,
        )

        # Seed the RNGs used by environment randomization.
        random.seed(seed)
        np.random.seed(seed)

        self.metadata = {
            "render_modes": ["human", "rgb_array"],
            "render_fps": int(np.round(1.0 / self.control_dt)),
        }

        self.render_mode = render_mode
        self.image_obs = image_obs
        self.env_step = 0
        self.intervened = False

        # Panda caches
        self._panda_right_dof_ids = np.asarray(
            [self._model.joint(f"joint{i}_right").id for i in range(1, 8)]
        )
        self._panda_right_ctrl_ids = np.asarray(
            [self._model.actuator(f"actuator{i}_right").id for i in range(1, 8)]
        )
        self._site_right_id = self._model.site("attachment_site_right").id

        self._panda_left_dof_ids = np.asarray(
            [self._model.joint(f"joint{i}_left").id for i in range(1, 8)]
        )
        self._panda_left_ctrl_ids = np.asarray(
            [self._model.actuator(f"actuator{i}_left").id for i in range(1, 8)]
        )
        self._site_left_id = self._model.site("attachment_site_left").id

        self._panda_dof_ids = np.concatenate(
            [self._panda_right_dof_ids, self._panda_left_dof_ids]
        )
        self._panda_ctrl_ids = np.concatenate(
            [self._panda_right_ctrl_ids, self._panda_left_ctrl_ids]
        )

        # fmt: off
        allegro_actuator_right_names = [
            "ffa0_right", "ffa1_right", "ffa2_right", "ffa3_right",
            "mfa0_right", "mfa1_right", "mfa2_right", "mfa3_right",
            "rfa0_right", "rfa1_right", "rfa2_right", "rfa3_right",
            "tha0_right", "tha1_right", "tha2_right", "tha3_right",
        ]
        allegro_actuator_left_names = [
            "rfa0_left", "rfa1_left", "rfa2_left", "rfa3_left",
            "mfa0_left", "mfa1_left", "mfa2_left", "mfa3_left",
            "ffa0_left", "ffa1_left", "ffa2_left", "ffa3_left",
            "tha0_left", "tha1_left", "tha2_left", "tha3_left",
        ]

        self._allegro_joint_right_names = [
            "ffj0_right", "ffj1_right", "ffj2_right", "ffj3_right",
            "mfj0_right", "mfj1_right", "mfj2_right", "mfj3_right",
            "rfj0_right", "rfj1_right", "rfj2_right", "rfj3_right",
            "thj0_right", "thj1_right", "thj2_right", "thj3_right",
        ]
        self._allegro_joint_left_names = [
            "rfj0_left", "rfj1_left", "rfj2_left", "rfj3_left",
            "mfj0_left", "mfj1_left", "mfj2_left", "mfj3_left",
            "ffj0_left", "ffj1_left", "ffj2_left", "ffj3_left",
            "thj0_left", "thj1_left", "thj2_left", "thj3_left",
        ]
        # fmt: on
        # Get actuator ids (fall back to mj_name2id if necessary)
        allegro_ids = []
        for name in allegro_actuator_right_names:
            aid = self._model.actuator(name).id
            allegro_ids.append(aid)

        for name in allegro_actuator_left_names:
            aid = self._model.actuator(name).id
            allegro_ids.append(aid)

        # print("Allegro actuator IDs:", allegro_ids)

        self._allegro_ctrl_ids = np.asarray(allegro_ids, dtype=int)

        self._allegro_dof_right_ids = np.asarray(
            [
                int(self._model.joint(n).qposadr)
                for n in self._allegro_joint_right_names
            ],
            dtype=int,
        )
        self._allegro_dof_left_ids = np.asarray(
            [int(self._model.joint(n).qposadr) for n in self._allegro_joint_left_names],
            dtype=int,
        )

        self._table_body_id = self._model.body("table").id
        self._table_leg_geom_ids = [
            gid
            for gid in range(self._model.ngeom)
            if self._model.geom_bodyid[gid] == self._table_body_id
            and self._model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_CYLINDER
        ]
        self._table_leg_half_len0 = {
            gid: float(self._model.geom_size[gid, 1])
            for gid in self._table_leg_geom_ids
        }

        self._front_camera_id = int(self._model.camera("back").id)
        self._wrist_left_camera_id = self._get_cam_id_by_name("handcam_rgb_left")
        self._wrist_right_camera_id = self._get_cam_id_by_name("handcam_rgb_right")
        self.camera_id = (
            self._front_camera_id,
            self._wrist_left_camera_id,
            self._wrist_right_camera_id,
        )

        self._camera_params = np.load(_HERE / "replay_cameras_2.npy")
        self._num_preset_cameras = int(self._camera_params.shape[0])
        self._scene_center = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        self._orig_light_pos = self._model.light_pos.copy()
        self._orig_light_dir = self._model.light_dir.copy()
        self._table_geom_id = self._model.geom("table_visual").id
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

        state_space = spaces.Dict(
            {
                "tcp_pose": spaces.Box(-np.inf, np.inf, shape=(14,)),
                "gripper_pose": spaces.Box(-np.inf, np.inf, shape=(32,)),
                "stand_ori_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
                "ipad_ori_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
                "table_delta_height": spaces.Box(-np.inf, np.inf, shape=(1,)),
            }
        )

        observation_space_dict = {"state": state_space}
        if self.image_obs:
            image_h = int(self._model.vis.global_.offheight)
            image_w = int(self._model.vis.global_.offwidth)

            observation_space_dict["images"] = spaces.Dict(
                {
                    "wrist_left": spaces.Box(
                        0, 255, shape=(image_h, image_w, 3), dtype=np.uint8
                    ),
                    "wrist_right": spaces.Box(
                        0, 255, shape=(image_h, image_w, 3), dtype=np.uint8
                    ),
                    "random_camera" if self.randomize else "ego": spaces.Box(
                        0, 255, shape=(image_h, image_w, 3), dtype=np.uint8
                    ),
                }
            )

        self.observation_space = spaces.Dict(observation_space_dict)

        # Action space
        self.action_space = spaces.Box(
            low=np.full(7 + _N_ALLEGRO, -1.0, dtype=np.float32),
            high=np.full(7 + _N_ALLEGRO, 1.0, dtype=np.float32),
            dtype=np.float32,
        )

        self._viewer = MujocoRenderer(self.model, self.data)
        try:
            self._viewer.render(self.render_mode)
        except Exception:
            pass

        # --- Button interaction setup ---
        self._button_geom_ids = []
        self._button_geom_orig_matid = {}
        self._button_geom_orig_rgba = {}
        self._button_geom_to_digit = {}
        self._button_visual_geom_ids = []
        self._button_visual_orig_rgba = {}
        self._digit_geom_ids_by_digit = {}
        try:
            for gid in range(self._model.ngeom):
                gname = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_GEOM, gid)
                if gname and gname.startswith("btn") and gname.endswith("_cyl"):
                    self._button_geom_ids.append(gid)
                    self._button_geom_orig_matid[gid] = int(self._model.geom_matid[gid])
                    self._button_geom_orig_rgba[gid] = self._model.geom_rgba[gid].copy()
                    # Map btnX_cyl -> digit X
                    try:
                        num = int(gname.replace("btn", "").replace("_cyl", ""))
                        self._button_geom_to_digit[gid] = num
                    except Exception:
                        pass
                if gname and gname.startswith("btn") and gname.endswith("_digit"):
                    try:
                        num = int(gname.replace("btn", "").replace("_digit", ""))
                        self._digit_geom_ids_by_digit[num] = gid
                    except Exception:
                        pass
                # Track all button/digit visuals by material name
                try:
                    matid = int(self._model.geom_matid[gid])
                    mname = mujoco.mj_id2name(
                        self._model, mujoco.mjtObj.mjOBJ_MATERIAL, matid
                    )
                    if mname in ("mat_digit",) or (
                        mname and mname.startswith("mat_digit_")
                    ):
                        self._button_visual_geom_ids.append(gid)
                        self._button_visual_orig_rgba[gid] = self._model.geom_rgba[
                            gid
                        ].copy()
                except Exception:
                    pass
        except Exception:
            pass
            # print("[Warning] failed to collect button geoms:", e)

        self._mat_digit_ids = {}
        self._mat_digit_light_ids = {}
        for d in range(10):
            try:
                self._mat_digit_ids[d] = int(self._model.material(f"mat_digit_{d}").id)
            except Exception:
                self._mat_digit_ids[d] = -1
            try:
                self._mat_digit_light_ids[d] = int(
                    self._model.material(f"mat_digit_{d}_light").id
                )
            except Exception:
                self._mat_digit_light_ids[d] = -1

        # --- Screen unlock interaction ---
        if password is None:
            self._unlock_sequence = [1, 2, 3]
        else:
            assert all(0 <= p and p <= 9 for p in password), "Password digits must be between 0 and 9."
            self._unlock_sequence = list(password)

        self._unlock_index = 0
        self._unlock_failed = False
        self._pressed_last = set()
        self._buttons_hidden = False
        self._screen_unlocked = False
        self._screen_geom_id = -1
        self._mat_screen_locked_id = -1
        self._mat_screen_unlocked_id = -1
        try:
            self._screen_geom_id = int(self._model.geom("ipad_screen").id)
        except Exception:
            pass
        try:
            self._mat_screen_locked_id = int(self._model.material("Material_black").id)
        except Exception:
            pass
        try:
            self._mat_screen_unlocked_id = int(
                self._model.material("mat_screen_unlocked").id
            )
        except Exception:
            pass

        # Default unlock parameters and pose.
        self._unlock_params = {
            "sequence": list(self._unlock_sequence),
            "settle_steps": 10,
            "above_steps": 90,
            "press_steps": 50,
            "post_press_steps": 15,
            "release_steps": 40,
            "above_offset": 0.09,
            "press_offset": 0.004,
            "dwell_s": 0.2,
        }

        self._unlock_action_template = np.zeros(7 + _N_ALLEGRO, dtype=np.float32)
        # fmt: off
        self._unlock_action_template[7 : 7 + _N_ALLEGRO] = np.array(
            [
                0.1, 0.1, 0.1, 0.0,  # ff (index extended)
                1.2, 1.2, 1.2, 1.2,  # mf curled
                1.2, 1.2, 1.2, 1.2,  # rf curled
                1.0, 1.0, 1.0, 1.0,  # thumb curled
            ],
            dtype=np.float32,
        )
        # fmt: on
        self._success_trigger_count = 0
        self._success_trigger_target = 10

        self._model_geom_pos0 = self._model.geom_pos.copy()
        self._model_geom_size0 = self._model.geom_size.copy()
        self._model_site_pos0 = self._model.site_pos.copy()

        self._table_z = self._model.body("table").pos[2].copy()

        self._ipad_body_z0 = self._model.body("ipad_body").pos[2].copy()
        self._stand_body_z0 = self._model.body("ipad_stand").pos[2].copy()

        self._finger_geom_ids = {
            self._model.geom("fingertip_ff_right").id,
            self._model.geom("fingertip_mf_right").id,
            self._model.geom("fingertip_rf_right").id,
            self._model.geom("thumbtip_right").id,
        }
        self._ipad_body_id = self._model.body("ipad_body").id
        self._ipad_mass0 = float(self._model.body_mass[self._ipad_body_id])
        self._ipad_mass_mul = (0.75, 1.25)

    def _get_cam_id_by_name(self, name: str) -> int:
        """Return camera id from name; return -1 if not found."""
        try:
            cam = self._model.camera(name)
            return int(cam.id)
        except Exception:
            try:
                return int(
                    mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, name)
                )
            except Exception:
                return -1

    def randomize_lighting(self):
        model = self._model

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

        # print("Randomized light positions:", model.light_pos)

    def randomize_desktop_texture(self):
        chosen_texture = random.choice(self._texture_names)
        mat_id = self._model.material(chosen_texture).id
        self._model.geom_matid[self._table_geom_id] = mat_id

    def randomize_camera(self):
        """Randomly choose one preset camera from replay_cameras.npy."""
        random_camera_idx = random.randint(0, self._num_preset_cameras - 1)
        self._apply_random_camera_to_front(random_camera_idx)

    def _prime_rgb_array_renderer(self):
        """Discard one offscreen frame per camera to avoid stale first-reset images."""
        self._viewer.render(
            render_mode="rgb_array", camera_id=self._wrist_left_camera_id
        )
        self._viewer.render(
            render_mode="rgb_array", camera_id=self._wrist_right_camera_id
        )
        self._viewer.render(render_mode="rgb_array", camera_id=self._front_camera_id)

    def _apply_random_camera_to_front(self, camera_idx):
        camera = self._camera_params[camera_idx]
        azimuth = float(camera[0])
        elevation = float(-camera[1])
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

        forward = -cam_offset
        forward /= np.linalg.norm(forward)

        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        right = np.cross(forward, world_up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)

        rot_matrix = np.column_stack([right, up, -forward])
        cam_quat_wxyz = R.from_matrix(rot_matrix).as_quat(scalar_first=True)

        self._model.cam_pos[self._front_camera_id] = cam_pos
        self._model.cam_quat[self._front_camera_id] = cam_quat_wxyz

    # --------------------------
    def reset(
        self, seed=None, **kwargs
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment."""
        mujoco.mj_resetData(self._model, self._data)

        # reset table height
        self.delta_h = np.float64(np.random.uniform(0.0, 0.05))

        table_ori_pos = self._model.body("table").pos
        table_ori_pos[2] = self.delta_h + self._table_z
        self._model.body("table").pos = table_ori_pos

        dh = float(self.delta_h)

        for lgid in self._table_leg_geom_ids:
            base_center_z = float(self._model_geom_pos0[lgid, 2])
            base_half_len = self._table_leg_half_len0[lgid]

            new_center_z = base_center_z - 0.5 * dh
            new_half_len = base_half_len + 0.5 * dh

            self._model.geom_pos[lgid, 2] = new_center_z
            self._model.geom_size[lgid, 1] = new_half_len

        # Sample a new ipad position.
        stand_xy = np.random.uniform(*_STAND_SAMPLING_BOUNDS)
        stand_body_id = self._model.body("ipad_stand").id
        stand_pos = self._model.body("ipad_stand").pos
        stand_quat = self._model.body("ipad_stand").quat
        ori_stand_xy = np.array(self._model.body_pos[stand_body_id][:2])
        delta_xy = stand_xy - ori_stand_xy

        # print("delta_xy:", delta_xy)
        ipad_body_id = self._model.body("ipad_body").id
        ipad_pos = self._model.body_pos[ipad_body_id]
        ipad_quat = self._model.body("ipad_body").quat
        ipad_pos[:2] += delta_xy
        ipad_pos[2] = self._ipad_body_z0 + self.delta_h

        # print("stand_xy:", stand_xy)

        stand_pos[:2] = stand_xy
        stand_pos[2] = self._stand_body_z0 + self.delta_h
        self._ipad_ori_pose = np.array(
            list(ipad_pos) + list(ipad_quat), dtype=np.float64
        )
        self._stand_ori_pose = np.array(
            list(stand_pos) + list(stand_quat), dtype=np.float64
        )

        self._model.body_pos[stand_body_id] = self._stand_ori_pose[:3]
        self._data.jnt("ipad_freejoint").qpos[:3] = self._ipad_ori_pose[:3]

        # Reset arm to home position.
        self._data.qpos[self._panda_right_dof_ids] = _PANDA_HOME
        self._data.qpos[self._panda_left_dof_ids] = _PANDA_HOME
        self._data.qpos[self._allegro_dof_right_ids] = _ALLEGRO_HOME
        self._data.qpos[self._allegro_dof_left_ids] = _ALLEGRO_HOME
        mujoco.mj_forward(self._model, self._data)

        # Reset mocap body to home position.
        tcp_pos_right = self._data.sensor("franka/flange_pos_right").data
        tcp_pos_left = self._data.sensor("franka/flange_pos_left").data
        self._data.mocap_pos[0] = tcp_pos_right
        self._data.mocap_pos[1] = tcp_pos_left

        if self.randomize:
            self.randomize_lighting()
            self.randomize_camera()
            self.randomize_desktop_texture()

        if self._randomize_dynamics:
            mass = float(
                np.random.uniform(self._ipad_mass_mul[0], self._ipad_mass_mul[1])
            )
            self._model.body_mass[self._ipad_body_id] = self._ipad_mass0 * mass

        # print(
        #     "ipad mass="
        #     f"{self._model.body_mass[self._ipad_body_id]}"
        # )

        mujoco.mj_forward(self._model, self._data)

        self.env_step = 0

        # Reset button visuals
        self._set_button_materials(set())
        self._unlock_index = 0
        self._unlock_failed = False
        self._success_trigger_count = 0
        self._pressed_last = set()
        self._buttons_hidden = False
        self._set_screen_locked()
        self._show_buttons()
        self._screen_unlocked = False

        self._prime_rgb_array_renderer()
        obs = self._compute_observation()
        return obs, {"succeed": False}

    def step(
        self, action
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        start_time = time.time()
        # Determine if action is a "zero op" (None / scalar 0 / size-1 array 0).
        is_zero_action = False
        if action is None:
            is_zero_action = True
        else:
            try:
                if np.isscalar(action) and action == 0:
                    is_zero_action = True
                elif (
                    isinstance(action, np.ndarray)
                    and action.size == 1
                    and action.item() == 0
                ):
                    is_zero_action = True
            except Exception:
                is_zero_action = False

        # ----- parse action when not zero -----
        if not is_zero_action:
            # print("Received action:", action)
            # action expected to be dict: {"right": arr23, "left": arr23}
            right = np.asarray(action["right"])
            left = np.asarray(action["left"])

            # Parse franka delta
            x_r, y_r, z_r, w_r, qx_r, qy_r, qz_r = right[0:7]
            x_l, y_l, z_l, w_l, qx_l, qy_l, qz_l = left[0:7]

            # Parse allegro (each side _N_ALLEGRO)
            allegro_r = np.asarray(right[7 : 7 + _N_ALLEGRO], dtype=np.float64)
            allegro_l = np.asarray(left[7 : 7 + _N_ALLEGRO], dtype=np.float64)
            allegro_angles = np.concatenate([allegro_r, allegro_l], axis=0)
        else:
            # zero action -> do not change mocap nor allegro
            x_r = y_r = z_r = w_r = qx_r = qy_r = qz_r = 0.0
            x_l = y_l = z_l = w_l = qx_l = qy_l = qz_l = 0.0
            allegro_angles = None  # Keep the current Allegro pose on zero action.

        # ----- read current mocap (keep backups) -----
        r_pos = self._data.mocap_pos[0].copy()
        l_pos = self._data.mocap_pos[1].copy()
        r_quat = self._data.mocap_quat[0].copy()
        l_quat = self._data.mocap_quat[1].copy()

        # construct target mocap arrays (if action given)
        tpos_r = np.asarray([x_r, y_r, z_r])
        tquat_r = np.array([w_r, qx_r, qy_r, qz_r])
        tpos_l = np.asarray([x_l, y_l, z_l])
        tquat_l = np.array([w_l, qx_l, qy_l, qz_l])

        # ----- apply mocap for right (keep original protection logic) -----
        if not is_zero_action and not (
            np.allclose(tpos_r, 0.0) and np.allclose(tquat_r, 0.0)
        ):
            # print("Applying mocap action - Right TCP pos:", tpos_r, "quat:", tquat_r)
            self._data.mocap_pos[0] = tpos_r
            self._data.mocap_quat[0] = tquat_r
        else:
            # keep original right pose
            self._data.mocap_pos[0] = r_pos
            self._data.mocap_quat[0] = r_quat

        # ----- apply mocap for left (preserve the current pose on zero action) -----
        if not is_zero_action and not (
            np.allclose(tpos_l, 0.0) and np.allclose(tquat_l, 0.0)
        ):
            self._data.mocap_pos[1] = tpos_l
            self._data.mocap_quat[1] = tquat_l
        else:
            # keep original left pose
            self._data.mocap_pos[1] = l_pos
            self._data.mocap_quat[1] = l_quat

        # print("Applied mocap pos right:", self._data.mocap_pos[0])
        # print("Applied mocap quat right:", self._data.mocap_quat[0])
        # print("Applied mocap pos left:", self._data.mocap_pos[1])
        # print("Applied mocap quat left:", self._data.mocap_quat[1])

        # ----- control loop  -----
        for _ in range(self._n_substeps):
            tau_right = opspace(
                model=self._model,
                data=self._data,
                site_id=self._site_right_id,
                dof_ids=self._panda_right_dof_ids,
                pos=self._data.mocap_pos[0],
                ori=self._data.mocap_quat[0],
                joint=_PANDA_HOME,
                gravity_comp=True,
                pos_gains=(400.0, 400.0, 400.0),
                damping_ratio=4,
            )
            self._data.ctrl[self._panda_right_ctrl_ids] = tau_right

            tau_left = opspace(
                model=self._model,
                data=self._data,
                site_id=self._site_left_id,
                dof_ids=self._panda_left_dof_ids,
                pos=self._data.mocap_pos[1],
                ori=self._data.mocap_quat[1],
                joint=_PANDA_HOME,
                gravity_comp=True,
                pos_gains=(400.0, 400.0, 400.0),
                damping_ratio=4,
            )
            self._data.ctrl[self._panda_left_ctrl_ids] = tau_left

            try:
                ctrl_ids = self._allegro_ctrl_ids
                valid_mask = ctrl_ids >= 0

                # Only update Allegro control when new joint targets are provided.
                if allegro_angles is not None and np.any(valid_mask):
                    target_qpos = allegro_angles
                    self._data.ctrl[ctrl_ids[valid_mask].astype(int)] = target_qpos[
                        valid_mask
                    ]

            except Exception:
                pass
                # print("[Warning] failed to write Allegro ctrl:", e)

            mujoco.mj_step(self._model, self._data)

        # Update button colors based on contacts
        try:
            pressed_digits = self._update_button_visuals()
        except Exception:
            pass
            # print("[Warning] failed to update button visuals:", e)

        obs = self._compute_observation()

        self.env_step += 1
        terminated = False
        if self.env_step >= 1200 or self._unlock_failed:
            terminated = True

        # ---- human rendering ----
        if self.render_mode == "human":
            try:
                self._viewer.render("human")
            except Exception:
                pass
                # print("[Warning] human render failed:", e)

        dt = time.time() - start_time
        time.sleep(max(0, (1.0 / self.hz) - dt))

        success = self._compute_success()
        rew = 1.0 if success else 0.0
        terminated = terminated or success

        return (
            obs,
            rew,
            terminated,
            False,
            {
                "succeed": success,
                "grasp_penalty": 0.0,
                "pressed_digits": pressed_digits,
            },
        )

    def _set_button_materials(self, pressed_ids: set):
        """Set button materials based on pressed state."""
        if not self._button_geom_ids:
            return
        for gid in self._button_geom_ids:
            digit = self._button_geom_to_digit.get(gid)
            digit_gid = self._digit_geom_ids_by_digit.get(digit, None)
            if gid in pressed_ids:
                mat_light = self._mat_digit_light_ids.get(digit, -1)
                if mat_light >= 0:
                    self._model.geom_matid[gid] = mat_light
                    if digit_gid is not None:
                        self._model.geom_matid[digit_gid] = mat_light
                else:
                    rgba = self._model.geom_rgba[gid].copy()
                    rgba[:3] = [0.2, 0.4, 1.0]
                    rgba[3] = 1.0
                    self._model.geom_rgba[gid] = rgba
            else:
                mat_off = self._mat_digit_ids.get(digit, -1)
                if mat_off >= 0:
                    self._model.geom_matid[gid] = mat_off
                    if digit_gid is not None:
                        self._model.geom_matid[digit_gid] = mat_off
                else:
                    self._model.geom_rgba[gid] = self._button_geom_orig_rgba.get(
                        gid, self._model.geom_rgba[gid]
                    )

    def _update_button_visuals(self):
        """Detect contacts and toggle button color while pressed.
        Only count a button as pressed when the contacting geom is a fingertip geom.
        """
        if not self._button_geom_ids:
            return
        if self._buttons_hidden:
            return

        button_set = set(self._button_geom_ids)
        pressed = set()

        ncon = int(self._data.ncon)
        for i in range(ncon):
            c = self._data.contact[i]
            g1 = int(c.geom1)
            g2 = int(c.geom2)

            # Only accept button <-> fingertip contacts.
            if g1 in button_set and g2 in self._finger_geom_ids:
                pressed.add(g1)

            elif g2 in button_set and g1 in self._finger_geom_ids:
                pressed.add(g2)

        self._set_button_materials(pressed)
        pressed_digits = self._update_unlock_sequence(pressed)
        return pressed_digits

    def _update_unlock_sequence(self, pressed: set):
        """Track rising-edge button presses for unlock sequence."""
        if not self._button_geom_to_digit:
            self._pressed_last = pressed
            return
        newly_pressed = pressed - self._pressed_last
        pressed_digits = []
        # TODO: only consider the first newly pressed button
        for gid in sorted(newly_pressed):
            digit = self._button_geom_to_digit.get(gid)
            if digit is None:
                continue
            pressed_digits.append(digit)
            expected = self._unlock_sequence[self._unlock_index]
            if digit == expected:
                self._unlock_index += 1
                if self._unlock_index >= len(self._unlock_sequence):
                    self._set_screen_unlocked()
                    self._hide_buttons()
                    self._screen_unlocked = True
            else:
                # print("----!!!wrong input!!!----")
                # self.reset()
                # reset on wrong digit
                self._unlock_failed = True
                self._unlock_index = 0
                self._set_screen_locked()
                self._screen_unlocked = False
        self._pressed_last = pressed
        return pressed_digits

    def _set_screen_locked(self):
        if self._screen_geom_id >= 0 and self._mat_screen_locked_id >= 0:
            self._model.geom_matid[self._screen_geom_id] = self._mat_screen_locked_id

    def _set_screen_unlocked(self):
        if self._screen_geom_id >= 0 and self._mat_screen_unlocked_id >= 0:
            self._model.geom_matid[self._screen_geom_id] = self._mat_screen_unlocked_id
        self._screen_unlocked = True

    def _hide_buttons(self):
        """Hide all button and digit visuals (alpha=0)."""
        if not self._button_visual_geom_ids:
            return
        for gid in self._button_visual_geom_ids:
            rgba = self._model.geom_rgba[gid].copy()
            rgba[3] = 0.0
            self._model.geom_rgba[gid] = rgba
        self._buttons_hidden = True

    def _show_buttons(self):
        """Restore button and digit visuals."""
        if not self._button_visual_geom_ids:
            return
        for gid in self._button_visual_geom_ids:
            self._model.geom_rgba[gid] = self._button_visual_orig_rgba.get(
                gid, self._model.geom_rgba[gid]
            )

    def _compute_success(self):
        if self._unlock_failed:
            return False
        
        if self._screen_unlocked:
            self._success_trigger_count += 1
        else:
            self._success_trigger_count = 0

        return self._success_trigger_count >= self._success_trigger_target

        # if self._screen_geom_id >= 0 and self._mat_screen_unlocked_id >= 0:
        #     return int(self._model.geom_matid[self._screen_geom_id]) == self._mat_screen_unlocked_id
        # return False

    def get_unlock_params(self) -> dict:
        params = dict(self._unlock_params)
        params["action_template"] = self._unlock_action_template.copy()
        params["sequence"] = list(self._unlock_sequence)
        params["other_arm_home"] = self._other_arm_home.copy()
        return params

    def get_unlock_status(self) -> dict:
        screen_mat = (
            int(self._model.geom_matid[self._screen_geom_id])
            if self._screen_geom_id >= 0
            else -1
        )
        return {
            "unlocked": bool(self._compute_success()),
            "unlock_index": int(self._unlock_index),
            "screen_mat_id": screen_mat,
            "screen_mat_unlocked_id": int(self._mat_screen_unlocked_id),
        }

    def _find_button_geom_id(self, preferred: str = "btn1_cyl") -> tuple[int, str]:
        try:
            gid = self._model.geom(preferred).id
            return gid, preferred
        except Exception:
            pass
        for gid in range(self._model.ngeom):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_GEOM, gid)
            if name and name.startswith("btn") and name.endswith("_cyl"):
                return gid, name
        raise RuntimeError("No button geom found (expected names like btn*_cyl).")

    def _get_button_geom_id_by_digit(self, digit: int) -> int:
        name = f"btn{digit}_cyl"
        try:
            return self._model.geom(name).id
        except Exception as e:
            raise RuntimeError(f"Button geom not found: {name}") from e

    def _get_screen_normal_and_center(self) -> tuple[np.ndarray, np.ndarray]:
        """Return screen normal (world) and screen center using ipad_screen geom."""
        try:
            gid = self._model.geom("ipad_screen").id
            xmat = self._data.geom_xmat[gid].reshape(3, 3)
            normal = xmat[:, 2]
            center = self._data.geom_xpos[gid].copy()
            return normal, center
        except Exception:
            gid = self._model.geom("btn1_cyl").id
            xmat = self._data.geom_xmat[gid].reshape(3, 3)
            normal = xmat[:, 2]
            center = self._data.geom_xpos[gid].copy()
            return normal, center

    def _compute_press_targets(
        self,
        fingertip_id: int,
        button_id: int,
        normal: np.ndarray,
        above_offset: float,
        press_offset: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute above/press targets along a consistent normal direction."""
        tip = self._data.geom_xpos[fingertip_id].copy()
        center = self._data.geom_xpos[button_id].copy()
        n = normal / (np.linalg.norm(normal) + 1e-9)
        if np.dot(tip - center, n) < 0:
            n = -n
        target_above = center + n * float(above_offset)
        target_press = center - n * float(press_offset)
        return target_above, target_press

    # ==========================
    def render(self):
        rendered_frames = []
        for cam_id in self.camera_id:
            rendered_frames.append(
                self._viewer.render(render_mode="rgb_array", camera_id=cam_id)
            )
        return rendered_frames

    # Helper methods.
    def _compute_observation(self) -> dict:
        obs = {}
        obs["state"] = {}

        tcp_pos_right = self._data.sensor("franka/flange_pos_right").data
        tcp_quat_right = self._data.sensor("franka/flange_quat_right").data
        tcp_pose_right = np.concatenate([tcp_pos_right, tcp_quat_right])

        tcp_pos_left = self._data.sensor("franka/flange_pos_left").data
        tcp_quat_left = self._data.sensor("franka/flange_quat_left").data
        tcp_pose_left = np.concatenate([tcp_pos_left, tcp_quat_left])

        tcp_pose = np.concatenate([tcp_pose_right, tcp_pose_left])

        # allegro_qpos = self._data.qpos[self._allegro_dof_ids].astype(np.float32)
        # fmt: off
        joint_names_right = [
            "allegro_right/ffj0_pos", "allegro_right/ffj1_pos", "allegro_right/ffj2_pos", "allegro_right/ffj3_pos",
            "allegro_right/mfj0_pos", "allegro_right/mfj1_pos", "allegro_right/mfj2_pos", "allegro_right/mfj3_pos",
            "allegro_right/rfj0_pos", "allegro_right/rfj1_pos", "allegro_right/rfj2_pos", "allegro_right/rfj3_pos",
            "allegro_right/thj0_pos", "allegro_right/thj1_pos", "allegro_right/thj2_pos", "allegro_right/thj3_pos",
        ]
        # fmt: on
        allegro_right_qpos = np.array(
            [self._data.sensor(name).data for name in joint_names_right],
            dtype=np.float32,
        )

        # fmt: off
        joint_names_left = [
            "allegro_left/rfj0_pos", "allegro_left/rfj1_pos", "allegro_left/rfj2_pos", "allegro_left/rfj3_pos",
            "allegro_left/mfj0_pos", "allegro_left/mfj1_pos", "allegro_left/mfj2_pos", "allegro_left/mfj3_pos",
            "allegro_left/ffj0_pos", "allegro_left/ffj1_pos", "allegro_left/ffj2_pos", "allegro_left/ffj3_pos",
            "allegro_left/thj0_pos", "allegro_left/thj1_pos", "allegro_left/thj2_pos", "allegro_left/thj3_pos",
        ]
        # fmt: on
        allegro_left_qpos = np.array(
            [self._data.sensor(name).data for name in joint_names_left],
            dtype=np.float32,
        )
        allegro_qpos = np.concatenate([allegro_right_qpos, allegro_left_qpos])

        if self.image_obs:
            obs["images"] = {}
            (
                obs["images"]["random_camera" if self.randomize else "ego"],
                obs["images"]["wrist_left"],
                obs["images"]["wrist_right"],
            ) = self.render()

        obs["state"] = {
            "tcp_pose": tcp_pose,
            "gripper_pose": allegro_qpos,
            "stand_ori_pose": self._stand_ori_pose,
            "ipad_ori_pose": self._ipad_ori_pose,
            "table_delta_height": self.delta_h,
        }

        return obs

    def get_end_effector_pose_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        # ---------- Right ----------
        pos_r = self._data.mocap_pos[0]
        quat_r = self._data.mocap_quat[0]
        quat_r = np.array([quat_r[1], quat_r[2], quat_r[3], quat_r[0]])
        rot_r = R.from_quat(quat_r).as_matrix()

        T_right = np.eye(4)
        T_right[:3, :3] = rot_r
        T_right[:3, 3] = pos_r

        # ---------- Left ----------

        pos_l = self._data.mocap_pos[1]
        quat_l = self._data.mocap_quat[1]
        quat_l = np.array([quat_l[1], quat_l[2], quat_l[3], quat_l[0]])
        rot_l = R.from_quat(quat_l).as_matrix()

        T_left = np.eye(4)
        T_left[:3, :3] = rot_l
        T_left[:3, 3] = pos_l

        return T_right, T_left


if __name__ == "__main__":
    # quick manual test
    env = PandaIpadGymEnv(render_mode="human")
    obs, info = env.reset()
    for i in range(200):
        # random actions: 7 franka + 16 allegro = 23
        action = np.random.uniform(-1, 1, 23)
        obs, rew, done, trunc, info = env.step(action)
    env.close()
