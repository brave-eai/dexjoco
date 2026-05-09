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
_XML_PATH = _HERE / "xmls" / "arena_arm_hand_microwave_cook.xml"
_PANDA_HOME = np.asarray((0, -0.785, 0, -2.35, 0, 1.57, np.pi / 4))  # Origin
_ALLEGRO_HOME = np.asarray((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.263, 0, 0, 0))
_CARTESIAN_BOUNDS = np.asarray([[0.2, -0.3, 0], [0.6, 0.3, 0.5]])
_SAMPLING_BOUNDS = np.asarray([[-0.35, -0.3], [-0.25, -0.4]])
_HOT_DOG_YAW_BOUNDS = (-20.0, 20.0)
_N_ALLEGRO = 16


class PandaBimanualMicrowaveCookGymEnv(MujocoGymEnv):
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

        allegro_joint_right_names = [
            "ffj0_right", "ffj1_right", "ffj2_right", "ffj3_right",
            "mfj0_right", "mfj1_right", "mfj2_right", "mfj3_right",
            "rfj0_right", "rfj1_right", "rfj2_right", "rfj3_right",
            "thj0_right", "thj1_right", "thj2_right", "thj3_right",
        ]
        allegro_joint_left_names = [
            "rfj0_left", "rfj1_left", "rfj2_left", "rfj3_left",
            "mfj0_left", "mfj1_left", "mfj2_left", "mfj3_left",
            "ffj0_left", "ffj1_left", "ffj2_left", "ffj3_left",
            "thj0_left", "thj1_left", "thj2_left", "thj3_left",
        ]

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
            [int(self._model.joint(n).qposadr) for n in allegro_joint_right_names],
            dtype=int
        )
        self._allegro_dof_left_ids = np.asarray(
            [int(self._model.joint(n).qposadr) for n in allegro_joint_left_names],
            dtype=int
        )

        state_space = spaces.Dict(
            {
                "tcp_pose": spaces.Box(-np.inf, np.inf, shape=(14,)),
                "gripper_pose": spaces.Box(-np.inf, np.inf, shape=(32,)),
                "hot_dog_ori_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
                "microwave_ori_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
                "table_delta_height": spaces.Box(-np.inf, np.inf, shape=(1,)),
            }
        )
        observation_space_dict = {"state": state_space}

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

        self._front_camera_id = int(self._model.camera("back").id)
        self._wrist_left_camera_id = self._get_cam_id_by_name("handcam_rgb_left")
        self._wrist_right_camera_id = self._get_cam_id_by_name("handcam_rgb_right")
        # print("ego_id:", self._front_camera_id)

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

        self._table_z = self._model.body("table").pos[2].copy()


        self._model_geom_pos0 = self._model.geom_pos.copy()
        self._model_geom_size0 = self._model.geom_size.copy()
        self._model_site_pos0 = self._model.site_pos.copy()
        self._hot_dog_body_z0 = self._model.body("hot_dog").pos[2].copy()
        self._microwave_body_z0 = self._model.body("microwave_object").pos[2].copy()

        #for dynamics randomization
        self._micro_joint_id = self._model.joint("microjoint").id
        self._micro_dof_id = int(self._model.jnt_dofadr[self._micro_joint_id])
        self._micro_friction0 = float(self._model.dof_frictionloss[self._micro_dof_id])
        self._micro_friction_mul = (0.75, 1.25)

        self._microwave_body_id = self._model.body("microwave_object").id
        self._hot_dog_body_id = self._model.body("hot_dog").id
        self._microwave_mass0 = float(self._model.body_mass[self._microwave_body_id])
        self._hot_dog_mass0 = float(self._model.body_mass[self._hot_dog_body_id])
        self._mass_mul = (0.75, 1.25)



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

        # reset table height
        self.delta_h = np.float64(np.random.uniform(0.0, 0.05))

        table_ori_pos = self._model.body("table").pos
        table_ori_pos[2] = self.delta_h + self._table_z
        self._model.body("table").pos = table_ori_pos

        leg_names = ("table_leg_1", "table_leg_2", "table_leg_3", "table_leg_4")
        dh = self.delta_h

        for lname in leg_names:
            lgid = self._model.geom(lname).id

            base_center_z = float(self._model_geom_pos0[lgid, 2])   # c0
            base_half_len = float(self._model_geom_size0[lgid, 1])   # h0 (cylinder: [radius, half_length])

            new_center_z = base_center_z - 0.5 * dh
            new_half_len = base_half_len + 0.5 * dh

            self._model.geom_pos[lgid, 2] = new_center_z
            self._model.geom_size[lgid, 1] = new_half_len

        # Sample a new hot_dog pose.
        hot_dog_base_quat = self._model.body("hot_dog").quat
        hot_dog_base_quat = np.array([hot_dog_base_quat[1], hot_dog_base_quat[2], hot_dog_base_quat[3], hot_dog_base_quat[0]])
        r_base = R.from_quat(hot_dog_base_quat)
        yaw_deg = np.random.uniform(*_HOT_DOG_YAW_BOUNDS)
        r_yaw = R.from_euler('z', yaw_deg, degrees=True)  # rotation representing yaw offset
        r_new = r_yaw * r_base
        hot_dog_ori_quat = r_new.as_quat()
        hot_dog_ori_quat = np.array([hot_dog_ori_quat[3], hot_dog_ori_quat[0], hot_dog_ori_quat[1], hot_dog_ori_quat[2]])
        hot_dog_xy = np.random.uniform(*_SAMPLING_BOUNDS)
        hot_dog_z = self._hot_dog_body_z0 + self.delta_h
        hot_dog_ori_pos = (*hot_dog_xy, hot_dog_z)

        self._hot_dog_ori_pose = np.concatenate([hot_dog_ori_pos, hot_dog_ori_quat]).astype(np.float64)
        self._data.jnt("hot_dog_free").qpos = self._hot_dog_ori_pose

        microwave_ori_pos = self._model.body("microwave_object").pos
        microwave_ori_quat = self._model.body("microwave_object").quat
        microwave_ori_pos[2] = self._microwave_body_z0 + self.delta_h
        self._microwave_ori_pose = np.concatenate([microwave_ori_pos, microwave_ori_quat]).astype(np.float64)
        self._model.body("microwave_object").pos = self._microwave_ori_pose[:3]
        self._model.body("microwave_object").quat = self._microwave_ori_pose[3:]

        # Reset arm to home position.
        self._data.qpos[self._panda_right_dof_ids] = _PANDA_HOME
        self._data.qpos[self._panda_left_dof_ids] = _PANDA_HOME
        self._data.qpos[self._allegro_dof_right_ids] = _ALLEGRO_HOME
        self._data.qpos[self._allegro_dof_left_ids] = _ALLEGRO_HOME
        mujoco.mj_forward(self._model, self._data)

        # Reset mocap body to home position.
        tcp_pos_right = self._data.sensor("franka/flange_pos_right").data
        # print("Reset TCP right pos:", tcp_pos_right)
        tcp_pos_left = self._data.sensor("franka/flange_pos_left").data
        # print("Reset TCP left pos:", tcp_pos_left)
        self._data.mocap_pos[0] = tcp_pos_right
        self._data.mocap_pos[1] = tcp_pos_left

        mujoco.mj_forward(self._model, self._data)

        if self.randomize:
            self.randomize_lighting()
            self.randomize_camera()
            self.randomize_desktop_texture()

        if self._randomize_dynamics:
            frictionloss = self._micro_friction0 * float(
                np.random.uniform(self._micro_friction_mul[0], self._micro_friction_mul[1])
            )
            self._model.dof_frictionloss[self._micro_dof_id] = frictionloss

            mass_mul = float(np.random.uniform(self._mass_mul[0], self._mass_mul[1]))
            self._model.body_mass[self._microwave_body_id] = self._microwave_mass0 * mass_mul
            mass_mul = float(np.random.uniform(self._mass_mul[0], self._mass_mul[1]))
            self._model.body_mass[self._hot_dog_body_id] = self._hot_dog_mass0 * mass_mul

        # print(
        #     "microwave door: frictionloss="
        #     f"{self._model.dof_frictionloss[self._micro_dof_id]}"
        # )
        # print(
        #     "microwave mass="
        #     f"{self._model.body_mass[self._microwave_body_id]}, "
        #     "hot_dog mass="
        #     f"{self._model.body_mass[self._hot_dog_body_id]}"
        # )

        mujoco.mj_forward(self._model, self._data)

        self.env_step = 0
        self._prime_rgb_array_renderer()

        obs = self._compute_observation()
        return obs, {"succeed": False}

    def _geom_in_contact(self, target_geom_name: str, dist_threshold: float = 0.0) -> bool:
        """
        Return True if `target_geom_name` is in contact this step (dist <= dist_threshold).
        Must be called after a physics step (mujoco.mj_step / sim.step).
        """
        ncon = int(self.data.ncon)
        # debug
        # print("DEBUG: ncon =", ncon)

        for i in range(ncon):
            c = self.data.contact[i]

            g1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(c.geom1))
            g2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(c.geom2))

            # robust name lookup
            # try:
            #     g1 = self.model.geom_id2name(int(c.geom1))
            #     g2 = self.model.geom_id2name(int(c.geom2))
            # except Exception:
            #     # fallback for some bindings that return tuples
            #     if hasattr(c, "__len__") and len(c) > 0:
            #         g1 = self.model.geom_id2name(int(c[0].geom1))
            #         g2 = self.model.geom_id2name(int(c[0].geom2))
            #     else:
            #         g1 = g2 = None

            # debug prints (uncomment for interactive debugging)
            # print(f"Contact {i}: {g1} - {g2}, dist={getattr(c,'dist', None)}")

            dist = float(getattr(c, "dist", 0.0))
            if (g1 == target_geom_name or g2 == target_geom_name) and (dist <= dist_threshold):
                return True
        return False

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:

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

        # ----- parse action when not zero -----
        if not is_zero_action:
            # action expected to be dict: {"right": arr23, "left": arr23}
            right = np.asarray(action["right"])
            left = np.asarray(action["left"])
            # Parse franka delta
            x_r, y_r, z_r, w_r, qx_r, qy_r, qz_r = right[0:7]
            x_l, y_l, z_l, w_l, qx_l, qy_l, qz_l = left[0:7]

            # Parse allegro (each side _N_ALLEGRO)
            allegro_r = np.asarray(right[7:7 + _N_ALLEGRO], dtype=np.float64)
            allegro_l = np.asarray(left[7:7 + _N_ALLEGRO], dtype=np.float64)
            allegro_angles = np.concatenate([allegro_r, allegro_l], axis=0)
            # print("Parsed allegro angles:", allegro_angles)
        else:
            # zero action -> do not change mocap nor allegro
            x_r = y_r = z_r = w_r = qx_r = qy_r = qz_r = 0.0
            x_l = y_l = z_l = w_l = qx_l = qy_l = qz_l = 0.0
            allegro_angles = None

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
        if not is_zero_action and not (np.allclose(tpos_r, 0.0) and np.allclose(tquat_r, 0.0)):
            self._data.mocap_pos[0] = tpos_r
            self._data.mocap_quat[0] = tquat_r
        else:
            # keep original right pose
            self._data.mocap_pos[0] = r_pos
            self._data.mocap_quat[0] = r_quat

        # ----- apply mocap for left  -----
        if not is_zero_action and not (np.allclose(tpos_l, 0.0) and np.allclose(tquat_l, 0.0)):
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
                if allegro_angles is not None and np.any(valid_mask):
                    target_qpos = allegro_angles
                    self._data.ctrl[ctrl_ids[valid_mask].astype(int)] = target_qpos[valid_mask]

            except Exception:
                pass
                # print("[Warning] failed to write Allegro ctrl:", e)

            mujoco.mj_step(self._model, self._data)

        obs = self._compute_observation()

        self.env_step += 1
        terminated = False
        if self.env_step >= 1100:
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

        return obs, rew, terminated, False, {"succeed": success, "grasp_penalty": 0.0}

    def _compute_success(self):
        # ---- door must be closed ----
        micro_qpos = float(self._data.joint("microjoint").qpos)
        microwave_closed = abs(micro_qpos) < 1e-2

        # ---- hot_dog must be inside interior bbox ----
        inside = self.hot_dog_inside_microwave(margin=0.01)

        contact = self._geom_in_contact("start_button")

        # Debug helpers kept nearby for local inspection when needed.
        # print("microjoint qpos:", micro_qpos)
        # print("hot_dog inside:", inside)
        # print("start button pressed:", contact)

        return inside and microwave_closed and contact

    # ==========================
    def render(self):
        rendered_frames = []
        for cam_id in self.camera_id:
            rendered_frames.append(self._viewer.render(render_mode="rgb_array", camera_id=cam_id))
        return rendered_frames

    # Helper methods.

    def _compute_observation(self) -> dict:
        obs = {}
        obs["state"] = {}

        tcp_pos_right = self._data.sensor("franka/flange_pos_right").data
        tcp_quat_right = self._data.sensor("franka/flange_quat_right").data
        tcp_pose_right = np.concatenate([tcp_pos_right, tcp_quat_right])
        tcp_vel_right = self._data.sensor("franka/flange_vel_right").data

        tcp_pos_left = self._data.sensor("franka/flange_pos_left").data
        tcp_quat_left = self._data.sensor("franka/flange_quat_left").data
        tcp_pose_left = np.concatenate([tcp_pos_left, tcp_quat_left])
        tcp_vel_left = self._data.sensor("franka/flange_vel_left").data

        tcp_pose = np.concatenate([tcp_pose_right, tcp_pose_left])
        tcp_vel = np.concatenate([tcp_vel_right, tcp_vel_left])

        # allegro_qpos = self._data.qpos[self._allegro_dof_ids].astype(np.float32)
        joint_names_right = [
            "allegro_right/ffj0_pos", "allegro_right/ffj1_pos", "allegro_right/ffj2_pos", "allegro_right/ffj3_pos",
            "allegro_right/mfj0_pos", "allegro_right/mfj1_pos", "allegro_right/mfj2_pos", "allegro_right/mfj3_pos",
            "allegro_right/rfj0_pos", "allegro_right/rfj1_pos", "allegro_right/rfj2_pos", "allegro_right/rfj3_pos",
            "allegro_right/thj0_pos", "allegro_right/thj1_pos", "allegro_right/thj2_pos", "allegro_right/thj3_pos"
        ]
        allegro_right_qpos = np.array([self._data.sensor(name).data for name in joint_names_right], dtype=np.float32)

        joint_names_left = [
            "allegro_left/rfj0_pos", "allegro_left/rfj1_pos", "allegro_left/rfj2_pos", "allegro_left/rfj3_pos",
            "allegro_left/mfj0_pos", "allegro_left/mfj1_pos", "allegro_left/mfj2_pos", "allegro_left/mfj3_pos",
            "allegro_left/ffj0_pos", "allegro_left/ffj1_pos", "allegro_left/ffj2_pos", "allegro_left/ffj3_pos",
            "allegro_left/thj0_pos", "allegro_left/thj1_pos", "allegro_left/thj2_pos", "allegro_left/thj3_pos"
        ]
        allegro_left_qpos = np.array([self._data.sensor(name).data for name in joint_names_left], dtype=np.float32)
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
            "hot_dog_ori_pose": self._hot_dog_ori_pose,
            "microwave_ori_pose": self._microwave_ori_pose,
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


    def world_to_body(self, body_name, p_world):
        bid = self._model.body(body_name).id
        # body_pos = self._data.body_xpos[bid]               # world
        # body_mat = self._data.body_xmat[bid].reshape(3, 3) # world_R_body

        body_pos = self._data.body("microwave_object").xpos
        body_mat = self._data.body("microwave_object").xmat.reshape(3, 3)
        # p_local = R^T (p_world - body_pos)
        return body_mat.T @ (p_world - body_pos)

    def hot_dog_inside_microwave(self, margin=0.0):
        # ---- hot_dog world position ----
        p_world = self._data.sensor("hot_dog_pos").data

        # ---- transform to microwave local frame ----
        p_local = self.world_to_body("microwave_object", p_world)

        # ---- read interior box sites (local frame) ----
        p0 = self._model.site("int_p0").pos
        px = self._model.site("int_px").pos
        py = self._model.site("int_py").pos
        pz = self._model.site("int_pz").pos

        inside_x = (p0[0] - margin) <= p_local[0] <= (px[0] + margin)
        inside_y = (p0[1] - margin) <= p_local[1] <= (py[1] + margin)
        inside_z = (p0[2] - margin) <= p_local[2] <= (pz[2] + margin)

        return inside_x and inside_y and inside_z

if __name__ == "__main__":
    env = PandaBimanualMicrowaveCookGymEnv(render_mode="human")
    env.reset()
    for i in range(100):
        env.step(np.random.uniform(-1, 1, 4))
        env.render()
    env.close()
