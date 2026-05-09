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
from ..mujoco_gym_env import MujocoGymEnv
from ..rendering import MujocoRenderer

_HERE = Path(__file__).parent
_XML_PATH = _HERE / "xmls" / "arena_arm_hand_table_tongs.xml"
_PANDA_HOME = np.asarray((0, -0.785, 0, -2.35, 0, 1.57, np.pi / 4))
_ALLEGRO_HOME = np.asarray(
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.263, 0, 0, 0),
    dtype=np.float32,
)
_TONGS_XY_BOUNDS = np.asarray([[-0.35, -0.25], [-0.3, -0.2]])
_MATCHED_Z = 0.955
_LIFT_HEIGHT = 0.1
_TONGS_CLOSE_THRESHOLD = -0.07
_TONGS_OPEN_THRESHOLD = 0.1
_REQUIRED_PINCHES = 3
_N_ALLEGRO = 16


class PandaPinchTongsGymEnv(MujocoGymEnv):
    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        randomize: bool,
        randomize_dynamics: bool = False,
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        hz: int = 30,
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
        self.intervened = False
        self._pinch_count = 0
        self._pinch_in_progress = True

        self._panda_dof_ids = np.asarray(
            [self._model.joint(f"joint{i}").id for i in range(1, 8)]
        )
        self._panda_ctrl_ids = np.asarray(
            [self._model.actuator(f"actuator{i}").id for i in range(1, 8)]
        )
        self._site_id = self._model.site("attachment_site").id
        self._tongs_z = _MATCHED_Z
        self._table_site_id = None
        try:
            self._table_site_id = self._model.site("table_top").id
        except Exception:
            self._table_site_id = None
        self._table_z = 0.92
        self._lift_z = self._table_z + _LIFT_HEIGHT

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

        image_h = int(self._model.vis.global_.offheight)
        image_w = int(self._model.vis.global_.offwidth)

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose": gym.spaces.Box(-np.inf, np.inf, shape=(7,)),
                        "gripper_pose": gym.spaces.Box(-1, 1, shape=(1,)),
                        "tongs_ori_pose": gym.spaces.Box(
                            -np.inf, np.inf, shape=(7,), dtype=np.float64
                        ),
                        "table_delta_height": gym.spaces.Box(
                            -np.inf, np.inf, shape=(1,), dtype=np.float64
                        ),
                    }
                ),
                "images": gym.spaces.Dict(
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
        left_id = _get_cam_id_by_name("left")
        right_id = _get_cam_id_by_name("right")
        handcam_rgb_id = _get_cam_id_by_name("handcam_rgb")

        missing = []
        if front_id < 0:
            missing.append("front")
        if handcam_rgb_id < 0:
            missing.append("handcam_rgb")
        if missing:
            raise RuntimeError(
                f"Required camera(s) not found in MuJoCo model: {missing}. "
                "Please ensure these cameras exist in your XML (names: 'front', 'handcam_rgb')."
            )
        self.camera_id = (front_id, left_id, right_id, handcam_rgb_id)

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
        self._tongs_z0 = _MATCHED_Z

        self._orig_light_pos = self._model.light_pos.copy()
        self._orig_light_dir = self._model.light_dir.copy()

        self._front_camera_id = int(self._model.camera("front").id)
        self._camera_params = np.load(_HERE / "replay_cameras.npy")
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

        self._tongs_body_id = self._model.body("link_1").id
        self._tongs_mass0 = float(self._model.body_mass[self._tongs_body_id])
        self._tongs_mass_mul = (0.75, 1.25)
        self._tongs_joint_id = self._model.joint("joint_0").id
        self._tongs_dof_id = int(self._model.jnt_dofadr[self._tongs_joint_id])
        self._tongs_friction_range = (0.0, 0.05)
        self._tongs_stiffness0 = float(self._model.jnt_stiffness[self._tongs_joint_id])
        self._tongs_stiffness_mul = (0.75, 1.25)

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

        self.delta_h = np.float64(np.random.uniform(0.0, 0.05))

        self._model.body_pos[self._table_body_id, 2] = (
            self._table_body_z0 + self.delta_h
        )

        for gid in self._table_leg_geom_ids:
            self._model.geom_size[gid, 1] = (
                self._table_leg_half_len0[gid] + self.delta_h
            )

        self._data.qpos[self._panda_dof_ids] = _PANDA_HOME
        self._data.qpos[self._allegro_dof_ids] = _ALLEGRO_HOME
        mujoco.mj_forward(self._model, self._data)

        tcp_pos = self._data.sensor("franka/flange_pos").data
        self._data.mocap_pos[0] = tcp_pos

        if self._table_site_id is not None:
            self._table_z = float(self._data.site_xpos[self._table_site_id][2])
        self._lift_z = self._table_z + _LIFT_HEIGHT
        self._tongs_z = self._tongs_z0 + self.delta_h

        tongs_xy = np.random.uniform(*_TONGS_XY_BOUNDS)
        tongs_ori_quat = self._data.jnt("tongs_root").qpos[3:7]
        self.tongs_ori_pose = np.concatenate(
            [tongs_xy, [self._tongs_z], tongs_ori_quat]
        ).astype(np.float64)
        self._data.jnt("tongs_root").qpos = self.tongs_ori_pose

        try:
            tongs_joint_0_id = mujoco.mj_name2id(
                self._model, mujoco.mjtObj.mjOBJ_JOINT, "joint_0"
            )
            if tongs_joint_0_id >= 0:
                tongs_joint_0_addr = self._model.jnt_qposadr[tongs_joint_0_id]
                self._data.qpos[tongs_joint_0_addr] = 0.3
        except Exception:
            pass
            # print("[Warning] failed to set joint_0 initial position:", e)

        self._pinch_count = 0
        self._pinch_in_progress = False

        if self.randomize:
            self.randomize_lighting()
            self.randomize_camera()
            self.randomize_desktop_texture()

        if self._randomize_dynamics:
            frictionloss = float(
                np.random.uniform(
                    self._tongs_friction_range[0], self._tongs_friction_range[1]
                )
            )
            stiffness = self._tongs_stiffness0 * float(
                np.random.uniform(
                    self._tongs_stiffness_mul[0], self._tongs_stiffness_mul[1]
                )
            )
            self._model.dof_frictionloss[self._tongs_dof_id] = frictionloss
            self._model.jnt_stiffness[self._tongs_joint_id] = stiffness

            mass_mul = float(
                np.random.uniform(self._tongs_mass_mul[0], self._tongs_mass_mul[1])
            )
            self._model.body_mass[self._tongs_body_id] = self._tongs_mass0 * mass_mul

        # print(
        #     "tongs joint_0: frictionloss="
        #     f"{self._model.dof_frictionloss[self._tongs_dof_id]}, "
        #     "stiffness="
        #     f"{self._model.jnt_stiffness[self._tongs_joint_id]}"
        # )
        # print(
        #     "tongs mass="
        #     f"{self._model.body_mass[self._tongs_body_id]}"
        # )

        mujoco.mj_forward(self._model, self._data)

        self.env_step = 0
        obs = self._compute_observation()
        return obs, {"succeed": False}

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

        pos = self._data.mocap_pos[0].copy()
        quat = self._data.mocap_quat[0].copy()

        tpos = np.asarray([x, y, z])
        tquat = np.array([w, qx, qy, qz])

        if np.allclose(tpos, 0.0) and np.allclose(tquat, 0.0):
            self._data.mocap_pos[0] = pos
            self._data.mocap_quat[0] = quat
        else:
            self._data.mocap_pos[0] = tpos
            self._data.mocap_quat[0] = tquat

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

            try:
                ctrl_ids = self._allegro_ctrl_ids
                valid_mask = ctrl_ids >= 0
                if np.any(valid_mask):
                    self._data.ctrl[ctrl_ids[valid_mask].astype(int)] = allegro_angles[
                        valid_mask
                    ]
            except Exception:
                pass
                # print("[Warning] failed to write Allegro ctrl:", e)

            mujoco.mj_step(self._model, self._data)

        self._update_pinch_count()
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
        reward = 1.0 if success else 0.0
        terminated = terminated or success

        return (
            obs,
            reward,
            terminated,
            False,
            {
                "succeed": success,
                "pinch_count": self._pinch_count,
            },
        )

    def render(self):
        rendered_frames = []
        for cam_id in self.camera_id:
            rendered_frames.append(
                self._viewer.render(render_mode="rgb_array", camera_id=cam_id)
            )
        return rendered_frames

    def _compute_observation(self) -> dict:
        tcp_pos = self._data.sensor("franka/flange_pos").data
        tcp_quat = self._data.sensor("franka/flange_quat").data
        tcp_pose = np.concatenate([tcp_pos, tcp_quat])

        joint_names = [
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
        ]
        allegro_qpos = np.array(
            [self._data.sensor(name).data for name in joint_names], dtype=np.float32
        )

        obs = {}
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
            "tongs_ori_pose": self.tongs_ori_pose,
            "table_delta_height": self.delta_h,
        }

        return obs

    def _update_pinch_count(self) -> None:
        try:
            joint_pos = float(self._data.sensor("tongs_joint_0_pos").data)
        except Exception:
            return

        if joint_pos <= _TONGS_CLOSE_THRESHOLD and self._pinch_in_progress:
            self._pinch_in_progress = False
            self._pinch_count += 1
        elif joint_pos >= _TONGS_OPEN_THRESHOLD and not self._pinch_in_progress:
            self._pinch_in_progress = True

        # print(f"Pinch count: {self._pinch_count}, Joint pos: {joint_pos:.3f}")

    def _compute_success(self) -> bool:
        try:
            tongs_pos = self._data.sensor("tongs_pos").data
        except Exception:
            return False

        lifted = tongs_pos[2] >= self._lift_z
        triggered = lifted and self._pinch_count >= _REQUIRED_PINCHES

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

    def get_end_effector_pose_matrix(self) -> np.ndarray:
        pos = self._data.mocap_pos[0]
        quat = self._data.mocap_quat[0]
        quat = np.array([quat[1], quat[2], quat[3], quat[0]])
        rot_mat = R.from_quat(quat).as_matrix()
        T = np.eye(4)
        T[:3, :3] = rot_mat
        T[:3, 3] = pos
        return T


if __name__ == "__main__":
    env = PandaPinchTongsGymEnv(render_mode="human")
    env.reset()
    for _ in range(200):
        action = np.random.uniform(-1, 1, 7 + _N_ALLEGRO)
        env.step(action)
    env.close()
