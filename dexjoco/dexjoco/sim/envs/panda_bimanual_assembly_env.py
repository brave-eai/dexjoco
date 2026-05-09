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
_XML_PATH = _HERE / "xmls" / "arena_arm_hand_bimanual_assembly.xml"
_PANDA_HOME = np.asarray((0, -0.785, 0, -2.35, 0, 1.57, np.pi / 4), dtype=np.float64)
_ALLEGRO_HOME = np.asarray((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.263, 0, 0, 0), dtype=np.float32)

_PEG_SAMPLING_BOUNDS = np.asarray([[-0.30, -0.25], [-0.25, -0.20]], dtype=np.float64)
_SOCKET_SAMPLING_BOUNDS = np.asarray([[-0.30, 0.15], [-0.20, 0.25]], dtype=np.float64)
_PEG_YAW_BOUNDS_DEG = np.asarray([-10.0, 10.0], dtype=np.float64)
_SOCKET_YAW_BOUNDS_DEG = np.asarray([-20.0, 20.0], dtype=np.float64)

_N_ALLEGRO = 16


class PandaBimanualAssemblyGymEnv(MujocoGymEnv):
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
        self._panda_right_dof_ids = np.asarray([self._model.joint(f"joint{i}_right").id for i in range(1, 8)])
        self._panda_right_ctrl_ids = np.asarray([self._model.actuator(f"actuator{i}_right").id for i in range(1, 8)])
        self._site_right_id = self._model.site("attachment_site_right").id

        self._panda_left_dof_ids = np.asarray([self._model.joint(f"joint{i}_left").id for i in range(1, 8)])
        self._panda_left_ctrl_ids = np.asarray([self._model.actuator(f"actuator{i}_left").id for i in range(1, 8)])
        self._site_left_id = self._model.site("attachment_site_left").id

        self._panda_dof_ids = np.concatenate([self._panda_right_dof_ids, self._panda_left_dof_ids])
        self._panda_ctrl_ids = np.concatenate([self._panda_right_ctrl_ids, self._panda_left_ctrl_ids])

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

        allegro_ids = []
        for name in allegro_actuator_right_names:
            allegro_ids.append(self._model.actuator(name).id)
        for name in allegro_actuator_left_names:
            allegro_ids.append(self._model.actuator(name).id)
        self._allegro_ctrl_ids = np.asarray(allegro_ids, dtype=int)

        self._allegro_dof_right_ids = np.asarray(
            [int(self._model.joint(n).qposadr) for n in self._allegro_joint_right_names],
            dtype=int,
        )
        self._allegro_dof_left_ids = np.asarray(
            [int(self._model.joint(n).qposadr) for n in self._allegro_joint_left_names],
            dtype=int,
        )

        self._mocap_right_id = int(self._model.body("target_right").mocapid)
        self._mocap_left_id = int(self._model.body("target_left").mocapid)

        # Object handles
        self._peg_joint_id = self._model.joint("industreal_round_peg_8mm_joint").id
        self._socket_joint_id = self._model.joint("industreal_tray_insert_round_peg_8mm_joint").id
        self._peg_qpos_adr = int(self._model.jnt_qposadr[self._peg_joint_id])
        self._peg_qvel_adr = int(self._model.jnt_dofadr[self._peg_joint_id])
        self._socket_qpos_adr = int(self._model.jnt_qposadr[self._socket_joint_id])
        self._socket_qvel_adr = int(self._model.jnt_dofadr[self._socket_joint_id])

        self._peg_body_id = self._model.body("industreal_round_peg_8mm").id
        self._socket_body_id = self._model.body("industreal_tray_insert_round_peg_8mm").id

        self._peg_init_pos = self._model.body_pos[self._peg_body_id].copy()
        self._peg_init_quat = self._model.body_quat[self._peg_body_id].copy()
        self._socket_init_pos = self._model.body_pos[self._socket_body_id].copy()
        self._socket_init_quat = self._model.body_quat[self._socket_body_id].copy()

        self._peg_body_z0 = float(self._peg_init_pos[2])
        self._socket_body_z0 = float(self._socket_init_pos[2])

        self._peg_geom_id = self._model.geom("industreal_round_peg_8mm_collision").id
        self._socket_bottom_geom_id = self._model.geom("industreal_tray_insert_round_peg_8mm_bottom_contact").id
        # Table height randomization base values.
        self._table_body_id = self._model.body("table").id
        self._table_z = float(self._model.body("table").pos[2])
        self._table_leg_geom_ids = [
            gid
            for gid in range(self._model.ngeom)
            if self._model.geom_bodyid[gid] == self._table_body_id
            and self._model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_CYLINDER
        ]
        self._table_leg_half_len0 = {
            gid: float(self._model.geom_size[gid, 1]) for gid in self._table_leg_geom_ids
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
                "socket_ori_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
                "peg_ori_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
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


        self.action_space = spaces.Box(
            low=np.full(7 + _N_ALLEGRO, -1.0, dtype=np.float32),
            high=np.full(7 + _N_ALLEGRO, 1.0, dtype=np.float32),
            dtype=np.float32,
        )

        self._viewer = MujocoRenderer(self.model, self.data)

        #for dynamics randomization
        self._peg_mass0 = float(self._model.body_mass[self._peg_body_id])
        self._socket_mass0 = float(self._model.body_mass[self._socket_body_id])
        self._mass_mul = (0.75, 1.25)

        try:
            self._viewer.render(self.render_mode)
        except Exception:
            pass

    def _get_cam_id_by_name(self, name: str) -> int:
        """Return camera id from name; return -1 if not found."""
        try:
            cam = self._model.camera(name)
            return int(cam.id)
        except Exception:
            try:
                return int(mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, name))
            except Exception:
                return -1

    @staticmethod
    def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        out = np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=np.float64,
        )
        out /= max(np.linalg.norm(out), 1e-8)
        return out

    def _set_free_joint_pose(
        self,
        qpos_adr: int,
        qvel_adr: int,
        pos: np.ndarray,
        quat: np.ndarray,
    ) -> None:
        self._data.qpos[qpos_adr : qpos_adr + 3] = np.asarray(pos, dtype=np.float64)
        self._data.qpos[qpos_adr + 3 : qpos_adr + 7] = np.asarray(quat, dtype=np.float64)
        self._data.qvel[qvel_adr : qvel_adr + 6] = 0.0

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
        """Randomly choose one preset camera from random_cameras.npy."""
        random_camera_idx = random.randint(0, self._num_preset_cameras - 1)
        self._apply_random_camera_to_front(random_camera_idx)

    def _prime_rgb_array_renderer(self):
        """Discard one offscreen frame per camera to avoid stale first-reset images."""
        self._viewer.render(render_mode="rgb_array", camera_id=self._wrist_left_camera_id)
        self._viewer.render(render_mode="rgb_array", camera_id=self._wrist_right_camera_id)
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

    def reset(self, seed=None, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment."""
        mujoco.mj_resetData(self._model, self._data)

        self.delta_h = np.float64(np.random.uniform(0.0, 0.05))
        table_pos = self._model.body("table").pos
        table_pos[2] = self.delta_h + self._table_z
        self._model.body("table").pos = table_pos

        for gid in self._table_leg_geom_ids:
            self._model.geom_size[gid, 1] = self._table_leg_half_len0[gid] + self.delta_h

        peg_xy = np.random.uniform(_PEG_SAMPLING_BOUNDS[0], _PEG_SAMPLING_BOUNDS[1])
        peg_yaw = np.deg2rad(np.random.uniform(*_PEG_YAW_BOUNDS_DEG))
        peg_yaw_quat = np.array([np.cos(peg_yaw / 2.0), 0.0, 0.0, np.sin(peg_yaw / 2.0)], dtype=np.float64)
        peg_quat = self._quat_mul(peg_yaw_quat, self._peg_init_quat)
        peg_pos = np.array([peg_xy[0], peg_xy[1], self._peg_body_z0 + self.delta_h], dtype=np.float64)
        self._set_free_joint_pose(self._peg_qpos_adr, self._peg_qvel_adr, peg_pos, peg_quat)

        socket_xy = np.random.uniform(_SOCKET_SAMPLING_BOUNDS[0], _SOCKET_SAMPLING_BOUNDS[1])
        socket_yaw = np.deg2rad(np.random.uniform(*_SOCKET_YAW_BOUNDS_DEG))
        socket_yaw_quat = np.array([np.cos(socket_yaw / 2.0), 0.0, 0.0, np.sin(socket_yaw / 2.0)], dtype=np.float64)
        socket_quat = self._quat_mul(socket_yaw_quat, self._socket_init_quat)
        socket_pos = np.array([socket_xy[0], socket_xy[1], self._socket_body_z0 + self.delta_h], dtype=np.float64)
        self._set_free_joint_pose(self._socket_qpos_adr, self._socket_qvel_adr, socket_pos, socket_quat)

        self._peg_ori_pose = np.concatenate([peg_pos, peg_quat])
        self._socket_ori_pose = np.concatenate([socket_pos, socket_quat])

        self._data.qpos[self._panda_right_dof_ids] = _PANDA_HOME
        self._data.qpos[self._panda_left_dof_ids] = _PANDA_HOME
        self._data.qpos[self._allegro_dof_right_ids] = _ALLEGRO_HOME
        self._data.qpos[self._allegro_dof_left_ids] = _ALLEGRO_HOME
        mujoco.mj_forward(self._model, self._data)

        tcp_pos_right = self._data.sensor("franka/flange_pos_right").data
        tcp_pos_left = self._data.sensor("franka/flange_pos_left").data
        tcp_quat_right = self._data.sensor("franka/flange_quat_right").data
        tcp_quat_left = self._data.sensor("franka/flange_quat_left").data
        self._data.mocap_pos[self._mocap_right_id] = tcp_pos_right
        self._data.mocap_pos[self._mocap_left_id] = tcp_pos_left
        self._data.mocap_quat[self._mocap_right_id] = tcp_quat_right
        self._data.mocap_quat[self._mocap_left_id] = tcp_quat_left

        if self.randomize:
            self.randomize_lighting()
            self.randomize_camera()
            self.randomize_desktop_texture()

        if self._randomize_dynamics:
            mass = float(np.random.uniform(self._mass_mul[0], self._mass_mul[1]))
            self._model.body_mass[self._peg_body_id] = self._peg_mass0 * mass
            mass = float(np.random.uniform(self._mass_mul[0], self._mass_mul[1]))
            self._model.body_mass[self._socket_body_id] = self._socket_mass0 * mass

        # print(
        #     "peg mass="
        #     f"{self._model.body_mass[self._peg_body_id]}, "
        #     "tray mass="
        #     f"{self._model.body_mass[self._socket_body_id]}"
        # )

        mujoco.mj_forward(self._model, self._data)

        self.env_step = 0
        self._prime_rgb_array_renderer()
        obs = self._compute_observation()
        return obs, {
            "succeed": False,
            "bottom_contact": False,
            "contact_count": 0,
        }

    def step(self, action) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        start_time = time.time()

        is_zero_action = False
        if action is None:
            is_zero_action = True
        else:
            try:
                if np.isscalar(action) and action == 0:
                    is_zero_action = True
                elif isinstance(action, np.ndarray) and action.size == 1 and action.item() == 0:
                    is_zero_action = True
            except Exception:
                is_zero_action = False

        if not is_zero_action:
            right = np.asarray(action["right"])
            left = np.asarray(action["left"])

            x_r, y_r, z_r, w_r, qx_r, qy_r, qz_r = right[0:7]
            x_l, y_l, z_l, w_l, qx_l, qy_l, qz_l = left[0:7]

            allegro_r = np.asarray(right[7:7 + _N_ALLEGRO], dtype=np.float64)
            allegro_l = np.asarray(left[7:7 + _N_ALLEGRO], dtype=np.float64)
            allegro_angles = np.concatenate([allegro_r, allegro_l], axis=0)
        else:
            x_r = y_r = z_r = w_r = qx_r = qy_r = qz_r = 0.0
            x_l = y_l = z_l = w_l = qx_l = qy_l = qz_l = 0.0
            allegro_angles = None

        r_pos = self._data.mocap_pos[self._mocap_right_id].copy()
        l_pos = self._data.mocap_pos[self._mocap_left_id].copy()
        r_quat = self._data.mocap_quat[self._mocap_right_id].copy()
        l_quat = self._data.mocap_quat[self._mocap_left_id].copy()

        tpos_r = np.asarray([x_r, y_r, z_r], dtype=np.float64)
        tquat_r = np.array([w_r, qx_r, qy_r, qz_r], dtype=np.float64)
        tpos_l = np.asarray([x_l, y_l, z_l], dtype=np.float64)
        tquat_l = np.array([w_l, qx_l, qy_l, qz_l], dtype=np.float64)

        if not is_zero_action and not (np.allclose(tpos_r, 0.0) and np.allclose(tquat_r, 0.0)):
            self._data.mocap_pos[self._mocap_right_id] = tpos_r
            self._data.mocap_quat[self._mocap_right_id] = tquat_r
        else:
            self._data.mocap_pos[self._mocap_right_id] = r_pos
            self._data.mocap_quat[self._mocap_right_id] = r_quat

        if not is_zero_action and not (np.allclose(tpos_l, 0.0) and np.allclose(tquat_l, 0.0)):
            self._data.mocap_pos[self._mocap_left_id] = tpos_l
            self._data.mocap_quat[self._mocap_left_id] = tquat_l
        else:
            self._data.mocap_pos[self._mocap_left_id] = l_pos
            self._data.mocap_quat[self._mocap_left_id] = l_quat

        # print("Applied mocap pos right:", self._data.mocap_pos[0])
        # print("Applied mocap quat right:", self._data.mocap_quat[0])
        # print("Applied mocap pos left:", self._data.mocap_pos[1])
        # print("Applied mocap quat left:", self._data.mocap_quat[1])

        for _ in range(self._n_substeps):
            tau_right = opspace(
                model=self._model,
                data=self._data,
                site_id=self._site_right_id,
                dof_ids=self._panda_right_dof_ids,
                pos=self._data.mocap_pos[self._mocap_right_id],
                ori=self._data.mocap_quat[self._mocap_right_id],
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
                pos=self._data.mocap_pos[self._mocap_left_id],
                ori=self._data.mocap_quat[self._mocap_left_id],
                joint=_PANDA_HOME,
                gravity_comp=True,
                pos_gains=(400.0, 400.0, 400.0),
                damping_ratio=4,
            )
            self._data.ctrl[self._panda_left_ctrl_ids] = tau_left

            if allegro_angles is not None:
                self._data.ctrl[self._allegro_ctrl_ids] = allegro_angles

            mujoco.mj_step(self._model, self._data)

        success = self._compute_success_metrics()

        obs = self._compute_observation()
        self.env_step += 1

        terminated = False
        if self.env_step >= 1500:
            terminated = True

        if self.render_mode == "human":
            try:
                self._viewer.render("human")
            except Exception:
                pass

        dt = time.time() - start_time
        time.sleep(max(0.0, (1.0 / self.hz) - dt))

        reward = 1.0 if success else 0.0
        terminated = terminated or success

        return obs, reward, terminated, False, {
            "succeed": success,
        }

    def _compute_success_metrics(self) -> Tuple[bool, bool, int]:
        contact_count = 0
        for i in range(int(self._data.ncon)):
            c = self._data.contact[i]
            g1 = int(c.geom1)
            g2 = int(c.geom2)
            if (
                (g1 == self._peg_geom_id and g2 == self._socket_bottom_geom_id)
                or (g2 == self._peg_geom_id and g1 == self._socket_bottom_geom_id)
            ):
                contact_count += 1
        bottom_contact = contact_count > 0
        triggered = bottom_contact

        if triggered:
            if not getattr(self, "_success_started", False):
                self._success_started = True
                self._success_counter = 1
            else:
                self._success_counter += 1
        else:
            self._success_started = False
            self._success_counter = 0
        # print(f"Triggered: {triggered}")
        if getattr(self, "_success_counter", 0) >= 30:
            return True

        return False
        # return bool(bottom_contact), bool(bottom_contact), int(contact_count)

    def _compute_success(self) -> bool:
        success, _, _ = self._compute_success_metrics()
        return bool(success)

    def render(self):
        rendered_frames = []
        for cam_id in self.camera_id:
            rendered_frames.append(self._viewer.render(render_mode="rgb_array", camera_id=cam_id))
        return rendered_frames

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

        joint_names_right = [
            "allegro_right/ffj0_pos", "allegro_right/ffj1_pos", "allegro_right/ffj2_pos", "allegro_right/ffj3_pos",
            "allegro_right/mfj0_pos", "allegro_right/mfj1_pos", "allegro_right/mfj2_pos", "allegro_right/mfj3_pos",
            "allegro_right/rfj0_pos", "allegro_right/rfj1_pos", "allegro_right/rfj2_pos", "allegro_right/rfj3_pos",
            "allegro_right/thj0_pos", "allegro_right/thj1_pos", "allegro_right/thj2_pos", "allegro_right/thj3_pos",
        ]
        allegro_right_qpos = np.array([self._data.sensor(name).data for name in joint_names_right], dtype=np.float32)

        joint_names_left = [
            "allegro_left/rfj0_pos", "allegro_left/rfj1_pos", "allegro_left/rfj2_pos", "allegro_left/rfj3_pos",
            "allegro_left/mfj0_pos", "allegro_left/mfj1_pos", "allegro_left/mfj2_pos", "allegro_left/mfj3_pos",
            "allegro_left/ffj0_pos", "allegro_left/ffj1_pos", "allegro_left/ffj2_pos", "allegro_left/ffj3_pos",
            "allegro_left/thj0_pos", "allegro_left/thj1_pos", "allegro_left/thj2_pos", "allegro_left/thj3_pos",
        ]
        allegro_left_qpos = np.array([self._data.sensor(name).data for name in joint_names_left], dtype=np.float32)
        allegro_qpos = np.concatenate([allegro_right_qpos, allegro_left_qpos])

        peg_pose = np.concatenate([
            self._data.sensor("assembly_peg_pos").data,
            self._data.body("industreal_round_peg_8mm").xquat,
        ])
        socket_pose = np.concatenate([
            self._data.sensor("assembly_socket_pos").data,
            self._data.body("industreal_tray_insert_round_peg_8mm").xquat,
        ])

        if self.image_obs:
            obs["images"] = {}
            obs["images"]["random_camera" if self.randomize else "ego"], obs["images"]["wrist_left"], obs["images"]["wrist_right"] = self.render()

        obs["state"] = {
            "tcp_pose": tcp_pose,
            "gripper_pose": allegro_qpos,
            "socket_ori_pose": self._socket_ori_pose,
            "peg_ori_pose": self._peg_ori_pose,
            "table_delta_height": self.delta_h,
        }

        return obs

    def get_end_effector_pose_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        pos_r = self._data.mocap_pos[self._mocap_right_id]
        quat_r = self._data.mocap_quat[self._mocap_right_id]
        quat_r = np.array([quat_r[1], quat_r[2], quat_r[3], quat_r[0]])
        rot_r = R.from_quat(quat_r).as_matrix()

        T_right = np.eye(4)
        T_right[:3, :3] = rot_r
        T_right[:3, 3] = pos_r

        pos_l = self._data.mocap_pos[self._mocap_left_id]
        quat_l = self._data.mocap_quat[self._mocap_left_id]
        quat_l = np.array([quat_l[1], quat_l[2], quat_l[3], quat_l[0]])
        rot_l = R.from_quat(quat_l).as_matrix()

        T_left = np.eye(4)
        T_left[:3, :3] = rot_l
        T_left[:3, 3] = pos_l

        return T_right, T_left


if __name__ == "__main__":
    env = PandaBimanualAssemblyGymEnv(render_mode="human")
    obs, info = env.reset()
    for _ in range(200):
        action = {
            "right": np.zeros(23, dtype=np.float32),
            "left": np.zeros(23, dtype=np.float32),
        }
        obs, rew, done, trunc, info = env.step(action)
        if done or trunc:
            break
    env.close()
