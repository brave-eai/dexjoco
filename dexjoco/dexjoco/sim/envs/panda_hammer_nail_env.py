import random
import time
from pathlib import Path
from typing import Any, Dict, Literal, Tuple

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R

from ..controllers import opspace
from ..mujoco_gym_env import GymRenderingSpec, MujocoGymEnv
from ..rendering import MujocoRenderer

_HERE = Path(__file__).parent
_XML_PATH = _HERE / "xmls" / "arena_arm_hand_hammer_nail.xml"

_PANDA_HOME = np.asarray((0, -0.785, 0, -2.35, 0, 1.57, np.pi / 4))
_ALLEGRO_HOME = np.asarray(
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.263, 0, 0, 0), dtype=np.float32
)

_HAMMER_SAMPLING_BOUNDS = np.asarray([[-0.25, -0.35], [-0.40, -0.50]])
_NAIL_SAMPLING_BOUNDS = np.asarray([[-0.10, 0.00], [0.00, 0.10]])
_HAMMER_YAW_PERTURB_BOUNDS = np.array([-10, 10])

_N_ALLEGRO = 16


class PandaHammerNailGymEnv(MujocoGymEnv):
    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        action_scale: np.ndarray = np.asarray([0.1, 1]),
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        config=None,
        hz=30,
        success_depth: float = 0.04,
        impact_insert_step: float = 0.008,
        impact_vel_threshold: float = 0.02,
        max_insert_depth: float = 0.0726,
        randomize: bool = False,
        randomize_dynamics: bool = False,
    ):
        self.hz = 30
        self.randomize = randomize
        self._randomize_dynamics = randomize_dynamics

        super().__init__(
            xml_path=_XML_PATH,
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
        )

        # Seed the RNGs used by environment randomization.
        random.seed(seed)
        np.random.seed(seed)

        self.metadata = {
            "render_modes": ["human", "rgb_array"],
            "render_fps": int(np.round(1.0 / self.control_dt)),
        }

        self.render_mode = render_mode
        self.env_step = 0
        self._success_depth = float(success_depth)
        self._impact_insert_step = float(impact_insert_step)
        self._impact_vel_threshold = float(impact_vel_threshold)
        self._nail_max_depth = float(max_insert_depth)

        # Panda caches
        self._panda_dof_ids = np.asarray(
            [self._model.joint(f"joint{i}").id for i in range(1, 8)]
        )
        self._panda_ctrl_ids = np.asarray(
            [self._model.actuator(f"actuator{i}").id for i in range(1, 8)]
        )
        self._panda_mocap_id = int(self._model.body("target").mocapid)

        # Allegro
        self._site_id = self._model.site("attachment_site").id
        self._allegro_joint_names = [
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
        ]
        self._allegro_dof_ids = np.asarray(
            [int(self._model.joint(n).qposadr) for n in self._allegro_joint_names],
            dtype=int,
        )

        allegro_actuator_names = [
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
        ]
        allegro_ids = []
        for name in allegro_actuator_names:
            try:
                aid = self._model.actuator(name).id
            except Exception:
                try:
                    aid = mujoco.mj_name2id(
                        self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
                    )
                except Exception:
                    aid = -1
            allegro_ids.append(aid)
        self._allegro_ctrl_ids = np.asarray(allegro_ids, dtype=int)

        # Nail body (mocap) for insertion depth.
        nail_body_id = self._model.body("nail").id
        self._nail_mocap_id = int(self._model.body("nail").mocapid)
        if self._nail_mocap_id < 0:
            raise RuntimeError("Nail body must be mocap-enabled (mocap='true') in XML.")
        self._nail_init_pos = self._model.body_pos[nail_body_id].copy()
        self._nail_init_quat = self._model.body_quat[nail_body_id].copy()
        self._nail_depth = 0.0

        # Contact-driven interaction between hammer and nail.
        self._hammer_geom_ids = []
        for name in ("face", "head", "neck", "claw"):
            try:
                self._hammer_geom_ids.append(self._model.geom(name).id)
            except Exception:
                continue

        self._nail_geom_ids = []
        for name in ("nail_head", "nail_shaft"):
            try:
                self._nail_geom_ids.append(self._model.geom(name).id)
            except Exception:
                continue

        try:
            self._model.body("hammer_body").id
        except Exception:
            pass

        try:
            self._face_gid = self._model.geom("face").id
        except Exception:
            self._face_gid = -1
        self._prev_face_z = None
        self._vz_buf: list[float] = []

        image_h = int(self._model.vis.global_.offheight)
        image_w = int(self._model.vis.global_.offwidth)

        # Observation space
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose": gym.spaces.Box(-np.inf, np.inf, shape=(7,)),
                        "gripper_pose": gym.spaces.Box(-1, 1, shape=(1,)),
                        "hammer_ori_pose": gym.spaces.Box(
                            -np.inf, np.inf, shape=(7,), dtype=np.float64
                        ),
                        "nail_ori_pose": gym.spaces.Box(
                            -np.inf, np.inf, shape=(7,), dtype=np.float64
                        ),
                        "table_delta_height": gym.spaces.Box(
                            -np.inf, np.inf, shape=(1,), dtype=np.float64
                        ),
                    }
                ),
                "images": spaces.Dict(
                    {
                        "wrist": spaces.Box(
                            0, 255, shape=(image_h, image_w, 3), dtype=np.uint8
                        ),
                        "random_camera" if self.randomize else "front": spaces.Box(
                            0, 255, shape=(image_h, image_w, 3), dtype=np.uint8
                        ),
                    }
                ),
            }
        )

        self.action_space = gym.spaces.Box(
            low=np.full(7 + _N_ALLEGRO, -1.0, dtype=np.float32),
            high=np.full(7 + _N_ALLEGRO, 1.0, dtype=np.float32),
            dtype=np.float32,
        )

        self._viewer = MujocoRenderer(self.model, self.data)
        try:
            self._viewer.render(self.render_mode)
        except Exception:
            pass

        def _get_cam_id_by_name(name: str) -> int:
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

        front_id = _get_cam_id_by_name("front")
        ego_left_id = _get_cam_id_by_name("left")
        ego_right_id = _get_cam_id_by_name("right")
        handcam_rgb_id = _get_cam_id_by_name("handcam_rgb")

        missing = []
        if front_id < 0:
            missing.append("front")
        if handcam_rgb_id < 0:
            missing.append("handcam_rgb")
        if len(missing) > 0:
            raise RuntimeError(
                f"Required camera(s) not found in MuJoCo model: {missing}. "
                "Please ensure these cameras exist in your XML (names: 'front', 'handcam_rgb')."
            )
        self.camera_id = (front_id, ego_left_id, ego_right_id, handcam_rgb_id)
        self._front_camera_id = front_id
        self._wrist_camera_id = handcam_rgb_id

        self._table_body_id = self._model.body("table").id
        self._table_body_z0 = float(self._model.body("table").pos[2])
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

        self._wood_body_z0 = float(self._model.body("wood").pos[2])
        self._hammer_body_z0 = float(self._model.body("hammer_body").pos[2])
        self._nail_body_z0 = float(self._model.body("nail").pos[2])

        # --- randomize support copied from code2 ---
        self._orig_light_pos = self._model.light_pos.copy()
        self._orig_light_dir = self._model.light_dir.copy()

        camera_param_path = _HERE / "replay_cameras.npy"
        if camera_param_path.exists():
            self._camera_params = np.load(camera_param_path)
        else:
            self._camera_params = np.zeros((0, 3), dtype=np.float64)
        self._num_preset_cameras = int(self._camera_params.shape[0])

        self._scene_center = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        self._table_geom_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_GEOM, "table_visual"
        )

        self._texture_names = [
            "table_bamboo",
            "table_blue-wood",
            "table_brass-ambra",
            "table_ceramic",
            "table_cream-plaster",
            "table_dark_wood_planks_2",
            "dark_wood_mat",
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

        try:
            self._hammer_body_id = self._model.body("hammer_body").id
        except Exception:
            self._hammer_body_id = -1
        self._hammer_mass0 = (
            float(self._model.body_mass[self._hammer_body_id])
            if self._hammer_body_id >= 0
            else 0.0
        )
        self._hammer_mass_mul = (0.75, 1.25)

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

    def randomize_desktop_texture(self):
        chosen_texture = random.choice(self._texture_names)
        # print(f"Chosen desktop texture: {chosen_texture}")
        mat_id = self._model.material(chosen_texture).id
        # print(f"Randomized desktop texture to {chosen_texture} (mat_id={mat_id})")
        self._model.geom_matid[self._table_geom_id] = mat_id

    def randomize_camera(self):
        random_camera_idx = random.randint(0, self._num_preset_cameras - 1)
        self._apply_random_camera_to_front(random_camera_idx)

    def _prime_rgb_array_renderer(self):
        self._viewer.render(render_mode="rgb_array", camera_id=self._wrist_camera_id)
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

    def reset(
        self, seed=None, **kwargs
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        mujoco.mj_resetData(self._model, self._data)

        # reset table height
        self.delta_h = np.float64(np.random.uniform(0.0, 0.05))

        self._model.body_pos[self._table_body_id, 2] = (
            self._table_body_z0 + self.delta_h
        )

        for gid in self._table_leg_geom_ids:
            self._model.geom_size[gid, 1] = (
                self._table_leg_half_len0[gid] + self.delta_h
            )

        self._model.body_pos[self._model.body("wood").id, 2] = (
            self._wood_body_z0 + self.delta_h
        )
        self._model.body_pos[self._model.body("hammer_body").id, 2] = (
            self._hammer_body_z0 + self.delta_h
        )
        self._data.mocap_pos[self._nail_mocap_id][2] = self._nail_body_z0 + self.delta_h
        self._nail_init_pos[2] = self._nail_body_z0 + self.delta_h

        # Randomize hammer position (x, y) and yaw
        hammer_xy = np.random.uniform(*_HAMMER_SAMPLING_BOUNDS)
        hammer_orig = np.array(
            self._data.jnt("hammer_joint").qpos[3:7], dtype=np.float64
        )
        hammer_yaw = np.deg2rad(np.random.uniform(*_HAMMER_YAW_PERTURB_BOUNDS))
        hqw, hqz = np.cos(hammer_yaw / 2), np.sin(hammer_yaw / 2)
        hw1, hx1, hy1, hz1 = hqw, 0, 0, hqz
        hw2, hx2, hy2, hz2 = hammer_orig
        hammer_q_new = np.array(
            [
                hw1 * hw2 - hx1 * hx2 - hy1 * hy2 - hz1 * hz2,
                hw1 * hx2 + hx1 * hw2 + hy1 * hz2 - hz1 * hy2,
                hw1 * hy2 - hx1 * hz2 + hy1 * hw2 + hz1 * hx2,
                hw1 * hz2 + hx1 * hy2 - hy1 * hx2 + hz1 * hw2,
            ]
        )
        hammer_q_new /= np.linalg.norm(hammer_q_new)
        self.hammer_ori_pose = np.concatenate(
            [hammer_xy, [self._hammer_body_z0 + self.delta_h], hammer_q_new]
        ).astype(np.float64)
        self._data.jnt("hammer_joint").qpos = self.hammer_ori_pose

        # Randomize nail position (x, y)
        nail_xy = np.random.uniform(*_NAIL_SAMPLING_BOUNDS)
        self._nail_init_pos[:2] = nail_xy
        self._data.mocap_pos[self._nail_mocap_id][:2] = nail_xy

        self._data.qpos[self._panda_dof_ids] = _PANDA_HOME
        self._data.qpos[self._allegro_dof_ids] = _ALLEGRO_HOME
        self._nail_depth = 0.0
        self.nail_ori_pose = np.concatenate(
            [self._nail_init_pos, self._nail_init_quat]
        ).astype(np.float64)
        self._data.mocap_pos[self._nail_mocap_id] = self._nail_init_pos
        self._data.mocap_quat[self._nail_mocap_id] = self._nail_init_quat

        mujoco.mj_forward(self._model, self._data)

        tcp_pos = self._data.sensor("franka/flange_pos").data
        self._data.mocap_pos[self._panda_mocap_id] = tcp_pos

        self.env_step = 0
        if self._face_gid >= 0:
            self._prev_face_z = float(self._data.geom_xpos[self._face_gid][2])
            self._vz_buf.clear()

        if self.randomize:
            self.randomize_lighting()
            self.randomize_camera()
            self.randomize_desktop_texture()

        if self._randomize_dynamics:
            mass = float(
                np.random.uniform(self._hammer_mass_mul[0], self._hammer_mass_mul[1])
            )
            self._model.body_mass[self._hammer_body_id] = self._hammer_mass0 * mass
        # print(
        #     "hammer mass="
        #     f"{self._model.body_mass[self._hammer_body_id]}"
        # )

        mujoco.mj_forward(self._model, self._data)

        if self.randomize:
            self._prime_rgb_array_renderer()

        obs = self._compute_observation()

        return obs, {"succeed": False, "nail_depth": float(self._nail_depth)}

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        start_time = time.time()

        if action is None or action.shape[0] < 7:
            raise ValueError("Action must have at least 7 elements (franka delta).")

        x, y, z, w, qx, qy, qz = (
            action[0],
            action[1],
            action[2],
            action[3],
            action[4],
            action[5],
            action[6],
        )

        if action.shape[0] >= 7 + _N_ALLEGRO:
            allegro_angles = np.asarray(action[7 : 7 + _N_ALLEGRO], dtype=np.float64)
        else:
            allegro_angles = np.zeros(_N_ALLEGRO, dtype=np.float64)

        pos = self._data.mocap_pos[self._panda_mocap_id].copy()
        quat = self._data.mocap_quat[self._panda_mocap_id].copy()

        tpos = np.asarray([x, y, z])
        tquat = np.array([w, qx, qy, qz])

        if np.allclose(tpos, 0.0) and np.allclose(tquat, 0.0):
            self._data.mocap_pos[self._panda_mocap_id] = pos
            self._data.mocap_quat[self._panda_mocap_id] = quat
        else:
            self._data.mocap_pos[self._panda_mocap_id] = tpos
            self._data.mocap_quat[self._panda_mocap_id] = tquat

        nail_depth = float(self._nail_depth)

        for _ in range(self._n_substeps):
            tau = opspace(
                model=self._model,
                data=self._data,
                site_id=self._site_id,
                dof_ids=self._panda_dof_ids,
                pos=self._data.mocap_pos[self._panda_mocap_id],
                ori=self._data.mocap_quat[self._panda_mocap_id],
                joint=_PANDA_HOME,
                gravity_comp=True,
                pos_gains=(400.0, 400.0, 400.0),
                damping_ratio=4,
            )
            self._data.ctrl[self._panda_ctrl_ids] = tau

            try:
                ctrl_ids = self._allegro_ctrl_ids
                valid_mask = ctrl_ids >= 0

                if np.any(valid_mask):
                    target_qpos = allegro_angles
                    self._data.ctrl[ctrl_ids[valid_mask].astype(int)] = target_qpos[
                        valid_mask
                    ]

            except Exception:
                pass
                # print("[Warning] failed to write Allegro ctrl:", e)

            mujoco.mj_step(self._model, self._data)
            if self._face_gid >= 0:
                face_z = float(self._data.geom_xpos[self._face_gid][2])
                if self._prev_face_z is None:
                    self._prev_face_z = face_z
                vz = (face_z - self._prev_face_z) / self.control_dt
                self._prev_face_z = face_z
                self._vz_buf.append(vz)
                if len(self._vz_buf) > 12:
                    self._vz_buf.pop(0)
            self._nail_depth = nail_depth

        hit = self._apply_hammer_nail_interaction()

        obs = self._compute_observation()

        self.env_step += 1
        terminated = self.env_step >= 1000

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
                "nail_depth": float(self._nail_depth),
                "hammer_hit": hit,
            },
        )

    def _compute_success(self):
        return float(self._nail_depth) >= self._success_depth

    def get_nail_depth(self) -> float:
        return float(self._nail_depth)

    def set_nail_depth(self, depth: float) -> None:
        depth = float(np.clip(depth, 0.0, self._nail_max_depth))
        self._nail_depth = depth
        new_pos = self._nail_init_pos.copy()
        new_pos[2] = self._nail_init_pos[2] - depth
        self._data.mocap_pos[self._nail_mocap_id] = new_pos
        self._data.mocap_quat[self._nail_mocap_id] = self._nail_init_quat
        mujoco.mj_forward(self._model, self._data)

    def _apply_hammer_nail_interaction(self) -> bool:
        if not self._hammer_geom_ids or not self._nail_geom_ids:
            return False

        hammer_set = set(self._hammer_geom_ids)
        nail_set = set(self._nail_geom_ids)
        hit = False
        ncon = int(self._data.ncon)
        for i in range(ncon):
            c = self._data.contact[i]
            g1 = int(c.geom1)
            g2 = int(c.geom2)
            if (g1 in hammer_set and g2 in nail_set) or (
                g2 in hammer_set and g1 in nail_set
            ):
                hit = True
                break

        if not hit:
            return False

        preimpact_vz = min(self._vz_buf) if self._vz_buf else 0.0
        if preimpact_vz >= -self._impact_vel_threshold:
            return True

        depth = float(self._nail_depth)
        max_depth = float(self._nail_max_depth)
        scale = min(3.0, abs(preimpact_vz) / max(self._impact_vel_threshold, 1e-6))
        delta = self._impact_insert_step * scale
        new_depth = min(max_depth, depth + delta)
        if new_depth > depth:
            self._nail_depth = new_depth
            new_pos = self._nail_init_pos.copy()
            new_pos[2] = self._nail_init_pos[2] - new_depth
            self._data.mocap_pos[self._nail_mocap_id] = new_pos
            self._data.mocap_quat[self._nail_mocap_id] = self._nail_init_quat
            mujoco.mj_forward(self._model, self._data)

        return True

    # ==========================
    def render(self):
        rendered_frames = []
        for cam_id in self.camera_id:
            rendered_frames.append(
                self._viewer.render(render_mode="rgb_array", camera_id=cam_id)
            )
        return rendered_frames

    def _compute_observation(self) -> dict:
        obs = {}
        obs["state"] = {}

        tcp_pos = self._data.sensor("franka/flange_pos").data
        tcp_quat = self._data.sensor("franka/flange_quat").data
        tcp_pose = np.concatenate([tcp_pos, tcp_quat])

        allegro_qpos = self._data.qpos[self._allegro_dof_ids].astype(np.float32)

        obs["images"] = {}
        (
            obs["images"]["random_camera" if self.randomize else "front"],
            obs["images"]["ego_left"],
            obs["images"]["ego_right"],
            obs["images"]["wrist"],
        ) = self.render()

        obs["state"] = {
            "tcp_pose": tcp_pose,
            "gripper_pose": allegro_qpos,
            "hammer_ori_pose": self.hammer_ori_pose,
            "nail_ori_pose": self.nail_ori_pose,
            "table_delta_height": self.delta_h,
        }

        return obs

    def get_end_effector_pose_matrix(self) -> np.ndarray:
        pos = self._data.mocap_pos[self._panda_mocap_id]
        quat = self._data.mocap_quat[self._panda_mocap_id]
        quat = np.array([quat[1], quat[2], quat[3], quat[0]])
        rot_mat = R.from_quat(quat).as_matrix()
        T = np.eye(4)
        T[:3, :3] = rot_mat
        T[:3, 3] = pos
        return T


if __name__ == "__main__":
    env = PandaHammerNailGymEnv(render_mode="human", randomize=True)
    obs, info = env.reset()
    for _ in range(200):
        action = np.random.uniform(-1, 1, 23)
        obs, rew, done, trunc, info = env.step(action)
    env.close()
