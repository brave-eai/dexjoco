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
_XML_PATH = _HERE / "xmls" / "arena_arm_hand_bimanual_hanoi.xml"
_PANDA_HOME = np.asarray((0, -0.785, 0, -2.35, 0, 1.57, np.pi / 4), dtype=np.float64)
_ALLEGRO_HOME = np.asarray((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.263, 0, 0, 0), dtype=np.float32)

_HANOI_BASE_SAMPLING_BOUNDS = np.asarray([[-0.25, 0], [-0.2, 0]], dtype=np.float64)
_N_ALLEGRO = 16
_HANOI_RESET_STATES = (
    (
        4,
        {
            "A": ["hanoi_disk_large"],
            "B": ["hanoi_disk_medium", "hanoi_disk_small"],
            "C": [],
        },
    ),
    (
        3,
        {
            "A": [],
            "B": ["hanoi_disk_medium", "hanoi_disk_small"],
            "C": ["hanoi_disk_large"],
        },
    ),
    (
        2,
        {
            "A": ["hanoi_disk_small"],
            "B": ["hanoi_disk_medium"],
            "C": ["hanoi_disk_large"],
        },
    ),
)


class PandaBimanualHanoiGymEnv(MujocoGymEnv):
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
        post_assignment_tol: float = 0.040,
        placement_height_tol: float = 0.012,
        static_qvel_tol: float = 0.35,
        success_trigger_target: int = 20,
        illegal_trigger_target: int = 10,
        illegal_reset_delay: float = 3.0,
    ):
        self.hz = hz
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

        self._post_assignment_tol = float(post_assignment_tol)
        self._placement_height_tol = float(placement_height_tol)
        self._static_qvel_tol = float(static_qvel_tol)
        self._success_trigger_target = int(success_trigger_target)
        self._illegal_trigger_target = int(illegal_trigger_target)
        self._illegal_reset_delay = float(illegal_reset_delay)
        self._success_trigger_count = 0
        self._illegal_trigger_count = 0
        self._illegal_trigger_reason = ""
        self._pending_illegal_reset = False
        self._pending_illegal_deadline = 0.0
        self._pending_illegal_info: Dict[str, Any] | None = None

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

        # Scene objects
        self._base_body_id = self._model.body("hanoi_base").id
        self._base_init_pos = self._model.body_pos[self._base_body_id].copy()
        self._base_init_quat = self._model.body_quat[self._base_body_id].copy()
        self._base_body_z0 = float(self._base_init_pos[2])

        self._disk_names = ["hanoi_disk_large", "hanoi_disk_medium", "hanoi_disk_small"]
        self._disk_size_rank = {
            "hanoi_disk_large": 3,
            "hanoi_disk_medium": 2,
            "hanoi_disk_small": 1,
        }
        self._disk_qpos_adr = {}
        self._disk_qvel_adr = {}
        self._disk_body_id = {}
        self._disk_init_pos = {}
        self._disk_init_quat = {}
        self._disk_body_z0 = {}
        self._disk_half_height = {}
        self._disk_mass0 = {}
        self._disk_mass_mul = (0.75, 1.25)

        for disk_name in self._disk_names:
            joint_name = f"{disk_name}_joint"
            joint_id = self._model.joint(joint_name).id
            body_id = self._model.body(disk_name).id
            self._disk_qpos_adr[disk_name] = int(self._model.jnt_qposadr[joint_id])
            self._disk_qvel_adr[disk_name] = int(self._model.jnt_dofadr[joint_id])
            self._disk_body_id[disk_name] = body_id
            self._disk_init_pos[disk_name] = self._model.body_pos[body_id].copy()
            self._disk_init_quat[disk_name] = self._model.body_quat[body_id].copy()
            self._disk_body_z0[disk_name] = float(self._disk_init_pos[disk_name][2])
            self._disk_mass0[disk_name] = float(self._model.body_mass[body_id])

            top_site = self._model.site(f"{disk_name}_top_site").id
            bottom_site = self._model.site(f"{disk_name}_bottom_site").id
            half_height = 0.5 * (
                float(self._model.site_pos[top_site][2]) - float(self._model.site_pos[bottom_site][2])
            )
            self._disk_half_height[disk_name] = half_height

        self._post_site_names = {
            "A": "hanoi_post_a_site",
            "B": "hanoi_post_b_site",
            "C": "hanoi_post_c_site",
        }
        self._post_site_ids = {name: self._model.site(site_name).id for name, site_name in self._post_site_names.items()}

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

        self._base_upper_geom_id = self._model.geom("hanoi_base_upper_collision").id
        self._base_upper_top_local_z = float(
            self._model.geom_pos[self._base_upper_geom_id, 2] + self._model.geom_size[self._base_upper_geom_id, 2]
        )

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
                "hanoi_base_ori_pos": spaces.Box(-np.inf, np.inf, shape=(3,)),
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
        try:
            self._viewer.render(self.render_mode)
        except Exception:
            pass

    def _get_cam_id_by_name(self, name: str) -> int:
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

    def _body_pose(self, body_name: str) -> np.ndarray:
        body = self._data.body(body_name)
        return np.concatenate([np.array(body.xpos, dtype=np.float64), np.array(body.xquat, dtype=np.float64)])

    def _sample_reset_tower_state(self) -> Tuple[int, Dict[str, list[str]]]:
        # idx = int(np.random.randint(len(_HANOI_RESET_STATES)))
        # steps_remaining, tower_state = _HANOI_RESET_STATES[idx]
        steps_remaining, tower_state = _HANOI_RESET_STATES[2]
        return steps_remaining, {post: list(disks) for post, disks in tower_state.items()}

    def _apply_reset_tower_state(
        self,
        tower_state: Dict[str, list[str]],
        base_delta_xy: np.ndarray,
    ) -> None:
        base_top_z = self._base_body_z0 + self.delta_h + self._base_upper_top_local_z
        post_xy = {
            post_name: self._base_init_pos[:2] + self._model.site_pos[site_id][:2]
            for post_name, site_id in self._post_site_ids.items()
        }

        for post_name, disk_names in tower_state.items():
            running_height = base_top_z
            for disk_name in disk_names:
                disk_pos = self._disk_init_pos[disk_name].copy()
                disk_pos[:2] = post_xy[post_name] + base_delta_xy
                disk_pos[2] = running_height + self._disk_half_height[disk_name]
                running_height += 2.0 * self._disk_half_height[disk_name]
                self._set_free_joint_pose(
                    self._disk_qpos_adr[disk_name],
                    self._disk_qvel_adr[disk_name],
                    disk_pos,
                    self._disk_init_quat[disk_name],
                )

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
        mujoco.mj_resetData(self._model, self._data)

        self.delta_h = np.float64(np.random.uniform(0.0, 0.05))
        table_pos = self._model.body("table").pos
        table_pos[2] = self.delta_h + self._table_z
        self._model.body("table").pos = table_pos

        for gid in self._table_leg_geom_ids:
            self._model.geom_size[gid, 1] = self._table_leg_half_len0[gid] + self.delta_h

        base_xy = np.random.uniform(_HANOI_BASE_SAMPLING_BOUNDS[0], _HANOI_BASE_SAMPLING_BOUNDS[1])
        base_pos = np.array([base_xy[0], base_xy[1], self._base_body_z0 + self.delta_h], dtype=np.float64)
        base_delta_xy = base_pos[:2] - self._base_init_pos[:2]
        self.base_ori_pos = base_pos
        self._model.body_pos[self._base_body_id] = self.base_ori_pos

        reset_steps_remaining, sampled_tower_state = self._sample_reset_tower_state()
        self._apply_reset_tower_state(sampled_tower_state, base_delta_xy)

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

        mujoco.mj_forward(self._model, self._data)

        self.env_step = 0
        self._prime_rgb_array_renderer()
        self._success_trigger_count = 0
        self._illegal_trigger_count = 0
        self._illegal_trigger_reason = ""
        self._pending_illegal_reset = False
        self._pending_illegal_deadline = 0.0
        self._pending_illegal_info = None

        success, metrics = self._compute_success_metrics()
        obs = self._compute_observation()

        if self._randomize_dynamics:
            for disk_name in self._disk_names:
                body_id = self._disk_body_id[disk_name]
                mass_mul = float(
                    np.random.uniform(self._disk_mass_mul[0], self._disk_mass_mul[1])
                )
                self._model.body_mass[body_id] = self._disk_mass0[disk_name] * mass_mul

        # print(
        #     "hanoi disks mass: large="
        #     f"{self._model.body_mass[self._disk_body_id['hanoi_disk_large']]}, "
        #     "medium="
        #     f"{self._model.body_mass[self._disk_body_id['hanoi_disk_medium']]}, "
        #     "small="
        #     f"{self._model.body_mass[self._disk_body_id['hanoi_disk_small']]}"
        # )

        return obs, {
            "succeed": False,
            "reset_steps_remaining": int(reset_steps_remaining),
            "reset_tower_state": sampled_tower_state,
            "tower_state": metrics["tower_state"],
            "disk_assignment": metrics["disk_assignment"],
            "all_disks_static": metrics["all_disks_static"],
            "illegal_state": metrics["illegal_state"],
            "success_stable_count": 0,
            "illegal_stable_count": 0,
        }

    def step(self, action) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        if self._pending_illegal_reset:
            if time.time() >= self._pending_illegal_deadline:
                pending_info = dict(self._pending_illegal_info or {})
                obs, _ = self.reset()
                if self.render_mode == "human":
                    try:
                        self._viewer.render("human")
                    except Exception:
                        pass
                pending_info["auto_reset"] = True
                return obs, 0.0, True, False, pending_info

            obs = self._compute_observation()
            pending_info = dict(self._pending_illegal_info or {})
            pending_info["auto_reset"] = False
            return obs, 0.0, True, False, pending_info

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

        success, metrics = self._compute_success_metrics()
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
        terminated = terminated or success or metrics["illegal_state"]
        success_stable_count = int(self._success_trigger_count)
        illegal_stable_count = int(self._illegal_trigger_count)
        info = {
            "succeed": success,
            "tower_state": metrics["tower_state"],
            "disk_assignment": metrics["disk_assignment"],
            "all_disks_static": metrics["all_disks_static"],
            "illegal_state": metrics["illegal_state"],
            "illegal_reason": metrics["illegal_reason"],
            "xy_error_max": float(metrics["xy_error_max"]),
            "height_error_max": float(metrics["height_error_max"]),
            "success_stable_count": success_stable_count,
            "illegal_stable_count": illegal_stable_count,
            "auto_reset": False,
        }
        if metrics["illegal_state"]:
            self._pending_illegal_reset = True
            self._pending_illegal_deadline = time.time() + max(0.0, self._illegal_reset_delay)
            self._pending_illegal_info = dict(info)

        return obs, reward, terminated, False, info

    def _disk_is_static(self, disk_name: str) -> bool:
        qvel_adr = self._disk_qvel_adr[disk_name]
        return float(np.linalg.norm(self._data.qvel[qvel_adr : qvel_adr + 6])) <= self._static_qvel_tol

    def _get_post_world_positions(self) -> Dict[str, np.ndarray]:
        return {
            post_name: np.array(self._data.site_xpos[site_id], dtype=np.float64)
            for post_name, site_id in self._post_site_ids.items()
        }

    def _base_top_world_z(self) -> float:
        base_body = self._data.body("hanoi_base")
        return float(base_body.xpos[2] + self._base_upper_top_local_z)

    def _reconstruct_tower_state(self) -> Dict[str, Any]:
        post_positions = self._get_post_world_positions()
        base_top_z = self._base_top_world_z()

        disk_assignment = {}
        disk_positions = {}
        disk_static = {}
        towers = {"A": [], "B": [], "C": []}

        for disk_name in self._disk_names:
            pos = np.array(self._data.body(disk_name).xpos, dtype=np.float64)
            disk_positions[disk_name] = pos
            disk_static[disk_name] = self._disk_is_static(disk_name)

            closest_post = min(
                post_positions.keys(),
                key=lambda k: np.linalg.norm(pos[:2] - post_positions[k][:2]),
            )
            closest_dist = float(np.linalg.norm(pos[:2] - post_positions[closest_post][:2]))

            if closest_dist <= self._post_assignment_tol:
                disk_assignment[disk_name] = closest_post
                towers[closest_post].append(disk_name)
            else:
                if pos[2] <= base_top_z + 0.06:
                    disk_assignment[disk_name] = "table"
                else:
                    disk_assignment[disk_name] = "floating"

        for post_name in towers:
            towers[post_name].sort(key=lambda name: disk_positions[name][2])

        return {
            "post_positions": post_positions,
            "base_top_z": float(base_top_z),
            "disk_assignment": disk_assignment,
            "disk_positions": disk_positions,
            "disk_static": disk_static,
            "towers": towers,
        }

    def _tower_state_repr(self, towers: Dict[str, list[str]]) -> Dict[str, list[str]]:
        return {post: list(names) for post, names in towers.items()}

    def _tower_geometry_errors(
        self,
        disks: list[str],
        post_name: str,
        disk_positions: Dict[str, np.ndarray],
        post_positions: Dict[str, np.ndarray],
        base_top_z: float,
    ) -> Tuple[float, float]:
        running_height = base_top_z
        xy_error_max = 0.0
        height_error_max = 0.0
        for disk_name in disks:
            expected_center_z = running_height + self._disk_half_height[disk_name]
            running_height += 2.0 * self._disk_half_height[disk_name]
            pos = disk_positions[disk_name]
            xy_error = float(np.linalg.norm(pos[:2] - post_positions[post_name][:2]))
            z_error = float(abs(pos[2] - expected_center_z))
            xy_error_max = max(xy_error_max, xy_error)
            height_error_max = max(height_error_max, z_error)
        return xy_error_max, height_error_max

    def _compute_success_metrics(self) -> Tuple[bool, Dict[str, Any]]:
        state = self._reconstruct_tower_state()
        towers = state["towers"]
        disk_assignment = state["disk_assignment"]
        disk_positions = state["disk_positions"]
        disk_static = state["disk_static"]
        post_positions = state["post_positions"]
        base_top_z = state["base_top_z"]

        tower_state = self._tower_state_repr(towers)
        all_disks_static = all(bool(v) for v in disk_static.values())

        illegal_candidate = False
        illegal_reason = ""

        for post_name, disks in towers.items():
            ranks = [self._disk_size_rank[disk] for disk in disks]
            if any(ranks[i] < ranks[i + 1] for i in range(len(ranks) - 1)):
                if all(disk_static[disk] for disk in disks):
                    xy_error, height_error = self._tower_geometry_errors(
                        disks, post_name, disk_positions, post_positions, base_top_z
                    )
                    if (
                        xy_error <= self._post_assignment_tol
                        and height_error <= self._placement_height_tol
                    ):
                        illegal_candidate = True
                        illegal_reason = f"illegal_stack_{post_name}"
                        break

        if not illegal_candidate and all_disks_static:
            on_table = [disk for disk, assign in disk_assignment.items() if assign == "table"]
            if on_table:
                illegal_candidate = True
                illegal_reason = "disk_on_table"

        if illegal_candidate:
            if illegal_reason == self._illegal_trigger_reason:
                self._illegal_trigger_count += 1
            else:
                self._illegal_trigger_reason = illegal_reason
                self._illegal_trigger_count = 1
        else:
            self._illegal_trigger_reason = ""
            self._illegal_trigger_count = 0

        illegal_state = self._illegal_trigger_count >= self._illegal_trigger_target

        xy_error_max = 0.0
        height_error_max = 0.0
        target_stack = ["hanoi_disk_large", "hanoi_disk_medium", "hanoi_disk_small"]

        if towers["C"] == target_stack and len(towers["A"]) == 0 and len(towers["B"]) == 0:
            xy_error_max, height_error_max = self._tower_geometry_errors(
                target_stack, "C", disk_positions, post_positions, base_top_z
            )
            geom_success = (
                xy_error_max <= self._post_assignment_tol
                and height_error_max <= self._placement_height_tol
                and all_disks_static
            )
        else:
            geom_success = False

        if geom_success and not illegal_state:
            self._success_trigger_count += 1
        else:
            self._success_trigger_count = 0

        success = self._success_trigger_count >= self._success_trigger_target

        return bool(success), {
            "tower_state": tower_state,
            "disk_assignment": dict(disk_assignment),
            "all_disks_static": bool(all_disks_static),
            "illegal_state": bool(illegal_state),
            "illegal_reason": illegal_reason,
            "xy_error_max": float(xy_error_max),
            "height_error_max": float(height_error_max),
        }

    def _compute_success(self) -> bool:
        success, _ = self._compute_success_metrics()
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

        if self.image_obs:
            obs["images"] = {}
            obs["images"]["random_camera" if self.randomize else "ego"], obs["images"]["wrist_left"], obs["images"]["wrist_right"] = self.render()

        obs["state"] = {
            "tcp_pose": tcp_pose,
            "gripper_pose": allegro_qpos,
            "hanoi_base_ori_pos": self.base_ori_pos,
            "table_delta_height": self.delta_h
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
    env = PandaBimanualHanoiGymEnv(render_mode="human")
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
