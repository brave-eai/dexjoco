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
_XML_PATH = _HERE / "xmls" / "arena_arm_hand_bimanual_photograph.xml"
_PANDA_HOME = np.asarray((0, -0.785, 0, -2.35, 0, 1.57, np.pi / 4), dtype=np.float64)
_ALLEGRO_HOME = np.asarray((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.263, 0, 0, 0), dtype=np.float32)
_N_ALLEGRO = 16

_TABLE_HEIGHT_BOUNDS = np.asarray([0.0, 0.05], dtype=np.float64)
_LOGO_SAMPLING_BOUNDS_YZ = np.asarray([[-0.10, 1.22], [0.10, 1.38]], dtype=np.float64)
_CAMERA_SAMPLING_BOUNDS = np.asarray([[-0.30, 0.1], [-0.20, 0.2]], dtype=np.float64)
_TARGET_REGION_OFFSET = np.asarray([-0.40, 0.0, 0.0], dtype=np.float64)
_TARGET_REGION_RADIUS = 0.16
_TARGET_REGION_HALF_HEIGHT = 0.10
_TARGET_ANGLE_DEG = 10.0
_WORLD_POS_X = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
_PASSIVE_VIEWER_PATCHED = False
_PASSIVE_MODEL_TARGET_REGION = {}


def _register_passive_target_region(model, target_region_geom_id: int) -> None:
    _PASSIVE_MODEL_TARGET_REGION[id(model)] = int(target_region_geom_id)


def _patch_mujoco_passive_viewer_sync() -> None:
    global _PASSIVE_VIEWER_PATCHED
    if _PASSIVE_VIEWER_PATCHED:
        return
    try:
        viewer_mod = getattr(mujoco, "viewer", None)
        if viewer_mod is None:
            return
        launch_passive_orig = getattr(viewer_mod, "launch_passive", None)
        if launch_passive_orig is None:
            return
        if getattr(launch_passive_orig, "_photo_group5_patch", False):
            _PASSIVE_VIEWER_PATCHED = True
            return

        def launch_passive_patched(*args, **kwargs):
            ctx = launch_passive_orig(*args, **kwargs)
            model = args[0] if len(args) > 0 else kwargs.get("model", None)

            class _CtxWrapper:
                def __enter__(self):
                    v = ctx.__enter__()
                    orig_sync = getattr(v, "sync", None)
                    if callable(orig_sync):
                        def sync_patched(*s_args, **s_kwargs):
                            try:
                                gid = _PASSIVE_MODEL_TARGET_REGION.get(id(model), None)
                                if gid is not None:
                                    cam = getattr(v, "cam", None)
                                    is_free = False
                                    if cam is not None:
                                        try:
                                            is_free = int(cam.type) == int(mujoco.mjtCamera.mjCAMERA_FREE)
                                        except Exception:
                                            is_free = False
                                    model.geom_group[gid] = 0 if is_free else 5
                            except Exception:
                                pass
                            return orig_sync(*s_args, **s_kwargs)
                        v.sync = sync_patched
                    return v

                def __exit__(self, exc_type, exc, tb):
                    return ctx.__exit__(exc_type, exc, tb)

            return _CtxWrapper()

        launch_passive_patched._photo_group5_patch = True
        viewer_mod.launch_passive = launch_passive_patched
        _PASSIVE_VIEWER_PATCHED = True
    except Exception:
        pass


class PandaBimanualPhotographGymEnv(MujocoGymEnv):
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
        hz: int = 30,
        camera_screen_effect: bool = True,
    ):
        self.hz = hz
        self._action_scale = action_scale
        self.randomize = randomize
        self._randomize_dynamics = randomize_dynamics
        self._camera_screen_effect = bool(camera_screen_effect)

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

        self._logo_geom_id = int(self._model.geom("photograph_logo_disk").id)
        self._target_region_geom_id = int(self._model.geom("photograph_target_region_visual").id)
        self._camera_axis_z_geom_id = int(self._model.geom("camera_axis_z").id)
        self._view_site_id = int(self._model.site("view_direction_site").id)
        self._camera_body_id = int(self._model.body("link_19").id)
        self._shutter_geom_id = int(self._model.geom("camera_body_top_cap").id)
        self._camera_geom_ids = self._collect_object_geom_ids(self._camera_body_id)
        self._logo_local_pos0 = self._model.geom_pos[self._logo_geom_id].copy()
        self._logo_local_quat0 = self._model.geom_quat[self._logo_geom_id].copy()
        self._logo_size0 = self._model.geom_size[self._logo_geom_id].copy()
        self._target_region_local_pos0 = self._model.geom_pos[self._target_region_geom_id].copy()
        self._target_region_size0 = self._model.geom_size[self._target_region_geom_id].copy()
        self._camera_body_pos0 = self._model.body_pos[self._camera_body_id].copy()
        self._camera_body_quat0 = self._model.body_quat[self._camera_body_id].copy()
        self._camera_body_z0 = float(self._camera_body_pos0[2])
        self._camera_mass0 = float(self._model.body_mass[self._camera_body_id])
        self._camera_mass_mul = (0.75, 1.25)

        _register_passive_target_region(self._model, self._target_region_geom_id)
        _patch_mujoco_passive_viewer_sync()

        self._group5_geom_ids = []
        try:
            geom_groups = np.asarray(self._model.geom_group)
            self._group5_geom_ids = np.where(geom_groups == 5)[0].tolist()
        except Exception:
            self._group5_geom_ids = []

        self._viewer = MujocoRenderer(self.model, self.data)
        try:
            self._viewer.render(self.render_mode)
        except Exception:
            pass

        self._screen_tex_id = -1
        self._screen_cam_id = -1
        self._screen_tex_w = 0
        self._screen_tex_h = 0
        self._screen_tex_adr = 0
        self._screen_tex_nchan = 3
        self._init_camera_screen_streaming()

        if not self._camera_screen_effect:
            self._hide_camera_screen_geom()

        # Photo capture / flash state.
        self._capture_flash_total = 5  # ~0.17 s of flash at hz=30
        self._captured_image = None
        self._capture_flash_frame = 0
        self._latest_shutter_pressed = False

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
                "camera_ori_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
                "logo_ori_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
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

    def _get_cam_id_by_name(self, name: str) -> int:
        try:
            return int(self._model.camera(name).id)
        except Exception:
            try:
                return int(mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, name))
            except Exception:
                return -1

    def _collect_object_geom_ids(self, root_body_id: int) -> set:
        body_ids = {root_body_id}
        changed = True
        while changed:
            changed = False
            for bid in range(self._model.nbody):
                pid = int(self._model.body_parentid[bid])
                if pid in body_ids and bid not in body_ids:
                    body_ids.add(bid)
                    changed = True
        return {gid for gid in range(self._model.ngeom) if int(self._model.geom_bodyid[gid]) in body_ids}

    def _parse_action(self, action):
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

        if is_zero_action:
            return is_zero_action, np.zeros(7), np.zeros(7), None

        right = np.asarray(action["right"], dtype=np.float64)
        left = np.asarray(action["left"], dtype=np.float64)
        pose_r = right[0:7]
        pose_l = left[0:7]
        allegro_r = np.asarray(right[7:7 + _N_ALLEGRO], dtype=np.float64)
        allegro_l = np.asarray(left[7:7 + _N_ALLEGRO], dtype=np.float64)
        allegro_angles = np.concatenate([allegro_r, allegro_l], axis=0)
        return is_zero_action, pose_r, pose_l, allegro_angles

    def _set_group_for_geom_ids(self, geom_ids, new_group: int):
        if len(geom_ids) == 0:
            return
        try:
            for gid in geom_ids:
                self._model.geom_group[gid] = new_group
        except Exception:
            try:
                tmp = np.asarray(self._model.geom_group).copy()
                for gid in geom_ids:
                    tmp[gid] = new_group
                self._model.geom_group[:] = tmp
            except Exception:
                pass

    def _temporarily_show_group5_in_gui(self):
        if not self._group5_geom_ids:
            return
        self._set_group_for_geom_ids(self._group5_geom_ids, 0)

    def _restore_group5(self):
        if not self._group5_geom_ids:
            return
        self._set_group_for_geom_ids(self._group5_geom_ids, 5)

    def _is_human_camera_free_pose(self) -> bool:
        free_type = int(mujoco.mjtCamera.mjCAMERA_FREE)

        viewer_candidates = []
        try:
            viewers = getattr(self._viewer, "_viewers", None)
            if isinstance(viewers, dict):
                human_viewer = viewers.get("human", None)
                if human_viewer is not None:
                    viewer_candidates.append(human_viewer)
        except Exception:
            pass

        viewer_candidates.extend(
            [
                getattr(self._viewer, "viewer", None),
                getattr(self._viewer, "_viewer", None),
                self._viewer,
            ]
        )

        for viewer_obj in viewer_candidates:
            if viewer_obj is None:
                continue
            cam = getattr(viewer_obj, "cam", None)
            if cam is None:
                cam = getattr(viewer_obj, "_cam", None)
            if cam is None:
                continue
            cam_type = getattr(cam, "type", None)
            if cam_type is None:
                continue
            try:
                if int(cam_type) == free_type:
                    return True
                fixedcamid = int(getattr(cam, "fixedcamid", -1))
                trackbodyid = int(getattr(cam, "trackbodyid", -1))
                # Free camera usually has no fixed camera id and no tracking body.
                return fixedcamid < 0 and trackbodyid < 0
            except Exception:
                continue

        return False

    def _set_human_geom_group_visibility(self, group_idx: int, visible: bool) -> bool:
        viewer_candidates = []
        try:
            viewers = getattr(self._viewer, "_viewers", None)
            if isinstance(viewers, dict):
                human_viewer = viewers.get("human", None)
                if human_viewer is not None:
                    viewer_candidates.append(human_viewer)
        except Exception:
            pass

        viewer_candidates.extend(
            [
                getattr(self._viewer, "viewer", None),
                getattr(self._viewer, "_viewer", None),
                self._viewer,
            ]
        )

        for viewer_obj in viewer_candidates:
            if viewer_obj is None:
                continue
            for opt_name in ("opt", "vopt", "_opt", "_vopt"):
                opt = getattr(viewer_obj, opt_name, None)
                if opt is None:
                    continue
                geomgroup = getattr(opt, "geomgroup", None)
                if geomgroup is None:
                    continue
                try:
                    geomgroup[group_idx] = 1 if visible else 0
                    return True
                except Exception:
                    continue
        return False

    def _set_offscreen_geom_group_visibility(self, group_idx: int, visible: bool) -> bool:
        try:
            viewers = getattr(self._viewer, "_viewers", None)
            if not isinstance(viewers, dict):
                return False
        except Exception:
            return False

        viewer_candidates = []
        for key, viewer_obj in viewers.items():
            if key == "human":
                continue
            if viewer_obj is not None:
                viewer_candidates.append(viewer_obj)

        for viewer_obj in viewer_candidates:
            for opt_name in ("opt", "vopt", "_opt", "_vopt"):
                opt = getattr(viewer_obj, opt_name, None)
                if opt is None:
                    continue
                geomgroup = getattr(opt, "geomgroup", None)
                if geomgroup is None:
                    continue
                try:
                    geomgroup[group_idx] = 1 if visible else 0
                except Exception:
                    continue

        return len(viewer_candidates) > 0

    def _init_camera_screen_streaming(self):
        """Look up texture and camera ids used to stream cam_pos_z onto the camera screen."""
        try:
            tex_id = int(
                mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_TEXTURE, "camera_screen_tex")
            )
        except Exception:
            tex_id = -1
        try:
            cam_id = int(
                mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_pos_z")
            )
        except Exception:
            cam_id = -1
        if tex_id < 0 or cam_id < 0:
            return
        self._screen_tex_id = tex_id
        self._screen_cam_id = cam_id
        self._screen_tex_w = int(self._model.tex_width[tex_id])
        self._screen_tex_h = int(self._model.tex_height[tex_id])
        self._screen_tex_adr = int(self._model.tex_adr[tex_id])
        self._screen_tex_nchan = int(self._model.tex_nchannel[tex_id])

    def _update_camera_screen_texture(self):
        """Drive the camera screen: live preview, capture flash, or frozen photo."""
        if self._screen_tex_id < 0 or not self._camera_screen_effect:
            return

        if self._captured_image is None:
            try:
                self._set_offscreen_geom_group_visibility(5, False)
                img = self._viewer.render(
                    render_mode="rgb_array", camera_id=self._screen_cam_id
                )
            except Exception:
                return
            if img is None or img.ndim != 3 or img.shape[2] < self._screen_tex_nchan:
                return
            img = self._fit_image_to_screen(img)

            if self._latest_shutter_pressed:
                # First shutter press: cache this frame, start the flash.
                self._captured_image = img.copy()
                self._capture_flash_frame = 0
                final = self._compose_flash_over(img)
                self._capture_flash_frame += 1
            else:
                final = img
        else:
            if self._capture_flash_frame < self._capture_flash_total:
                final = self._compose_flash_over(self._captured_image)
                self._capture_flash_frame += 1
            else:
                final = self._captured_image

        self._write_screen_texture(final)

    def _fit_image_to_screen(self, img: np.ndarray) -> np.ndarray:
        """Center-crop to the screen aspect ratio and resample to texture size."""
        th, tw = self._screen_tex_h, self._screen_tex_w
        h_in, w_in = img.shape[:2]

        tex_aspect = tw / max(th, 1)
        img_aspect = w_in / max(h_in, 1)
        if abs(img_aspect - tex_aspect) > 1e-3:
            if img_aspect > tex_aspect:
                new_w = max(1, min(int(round(h_in * tex_aspect)), w_in))
                x0 = (w_in - new_w) // 2
                img = img[:, x0:x0 + new_w, :]
            else:
                new_h = max(1, min(int(round(w_in / tex_aspect)), h_in))
                y0 = (h_in - new_h) // 2
                img = img[y0:y0 + new_h, :, :]
            h_in, w_in = img.shape[:2]

        if h_in != th or w_in != tw:
            ys = np.linspace(0, h_in - 1, th).astype(np.int64)
            xs = np.linspace(0, w_in - 1, tw).astype(np.int64)
            img = img[np.ix_(ys, xs)]
        return img

    def _compose_flash_over(self, base_img: np.ndarray) -> np.ndarray:
        """Blend a white sheet over the captured image, alpha fading 1 -> 0."""
        total = max(self._capture_flash_total, 1)
        progress = min(1.0, self._capture_flash_frame / total)
        alpha = max(0.0, 1.0 - progress)
        if alpha <= 0.0:
            return base_img
        blended = alpha * 255.0 + (1.0 - alpha) * base_img.astype(np.float32)
        return np.clip(blended, 0.0, 255.0).astype(np.uint8)

    def _write_screen_texture(self, img: np.ndarray):
        nchan = self._screen_tex_nchan
        flat = np.ascontiguousarray(img[:, :, :nchan], dtype=np.uint8).reshape(-1)
        adr = self._screen_tex_adr
        self._model.tex_data[adr:adr + flat.size] = flat
        self._upload_screen_texture_to_all_viewers()

    def _hide_camera_screen_geom(self):
        """Make the camera_live_screen geom visually disappear (used when the
        camera-screen effect is disabled)."""
        try:
            gid = int(self._model.geom("camera_live_screen").id)
        except Exception:
            return
        try:
            self._model.geom_matid[gid] = -1
            self._model.geom_rgba[gid] = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            self._model.geom_size[gid] = np.array([1e-9, 1e-9, 1e-9], dtype=np.float64)
        except Exception:
            pass

    def _upload_screen_texture_to_all_viewers(self):
        if self._screen_tex_id < 0:
            return
        viewers = getattr(self._viewer, "_viewers", None)
        if not isinstance(viewers, dict):
            return
        for v in viewers.values():
            if v is None:
                continue
            con = getattr(v, "con", None)
            if con is None:
                continue
            try:
                mkc = getattr(v, "make_context_current", None)
                if callable(mkc):
                    mkc()
                mujoco.mjr_uploadTexture(self._model, con, self._screen_tex_id)
            except Exception:
                pass

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
        self._set_offscreen_geom_group_visibility(5, False)
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

        # Clear any prior capture so the screen returns to live preview.
        self._captured_image = None
        self._capture_flash_frame = 0
        self._latest_shutter_pressed = False

        self.delta_h = np.float64(np.random.uniform(*_TABLE_HEIGHT_BOUNDS))
        table_pos = self._model.body("table").pos
        table_pos[2] = self.delta_h + self._table_z
        self._model.body("table").pos = table_pos
        for gid in self._table_leg_geom_ids:
            self._model.geom_size[gid, 1] = self._table_leg_half_len0[gid] + self.delta_h

        logo_yz = np.random.uniform(_LOGO_SAMPLING_BOUNDS_YZ[0], _LOGO_SAMPLING_BOUNDS_YZ[1])
        logo_local_pos = self._logo_local_pos0.copy()
        logo_local_pos[1:] = logo_yz
        logpo_local_quat = self._logo_local_quat0.copy()
        self.logo_ori_pose = np.concatenate([logo_local_pos, logpo_local_quat]).astype(np.float64)
        self._model.geom_pos[self._logo_geom_id] = self.logo_ori_pose[0:3]
        self._model.geom_quat[self._logo_geom_id] = self.logo_ori_pose[3:7]
        self._model.geom_size[self._logo_geom_id] = self._logo_size0

        target_region_local_pos = logo_local_pos + _TARGET_REGION_OFFSET
        self._model.geom_pos[self._target_region_geom_id] = target_region_local_pos
        self._model.geom_size[self._target_region_geom_id, 0] = _TARGET_REGION_RADIUS
        self._model.geom_size[self._target_region_geom_id, 1] = _TARGET_REGION_HALF_HEIGHT
        self._model.geom_size[self._target_region_geom_id, 2] = self._target_region_size0[2]

        camera_xy = np.random.uniform(_CAMERA_SAMPLING_BOUNDS[0], _CAMERA_SAMPLING_BOUNDS[1])
        camera_body_pos = self._camera_body_pos0.copy()
        camera_body_pos[:2] = camera_xy
        camera_body_pos[2] = self._camera_body_z0 + self.delta_h + 0.002
        self.camera_ori_pose = np.concatenate([camera_body_pos, self._camera_body_quat0]).astype(np.float64)
        self._data.jnt("camera_root").qpos = self.camera_ori_pose


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
            mass_mul = float(np.random.uniform(self._camera_mass_mul[0], self._camera_mass_mul[1]))
            self._model.body_mass[self._camera_body_id] = self._camera_mass0 * mass_mul

        # print(
        #     "camera mass="
        #     f"{self._model.body_mass[self._camera_body_id]}"
        # )

        mujoco.mj_forward(self._model, self._data)

        self.env_step = 0
        self._prime_rgb_array_renderer()
        obs = self._compute_observation()
        return obs, {
            "succeed": False,
            "region_pass": False,
            "angle_pass": False,
            "shutter_pressed": False,
            "view_angle_deg": float(180.0),
        }

    def step(self, action) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        start_time = time.time()
        is_zero_action, pose_r, pose_l, allegro_angles = self._parse_action(action)

        r_pos = self._data.mocap_pos[self._mocap_right_id].copy()
        l_pos = self._data.mocap_pos[self._mocap_left_id].copy()
        r_quat = self._data.mocap_quat[self._mocap_right_id].copy()
        l_quat = self._data.mocap_quat[self._mocap_left_id].copy()

        tpos_r = np.asarray(pose_r[0:3], dtype=np.float64)
        tquat_r = np.asarray([pose_r[3], pose_r[4], pose_r[5], pose_r[6]], dtype=np.float64)
        tpos_l = np.asarray(pose_l[0:3], dtype=np.float64)
        tquat_l = np.asarray([pose_l[3], pose_l[4], pose_l[5], pose_l[6]], dtype=np.float64)

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
        self._latest_shutter_pressed = bool(metrics.get("shutter_pressed", False))
        obs = self._compute_observation(metrics=metrics)
        self.env_step += 1

        terminated = self.env_step >= 1000 or success

        if self.render_mode == "human":
            try:
                # Refresh the camera screen texture for the human viewer in case
                # render() was not called this step (e.g. image_obs is False).
                self._update_camera_screen_texture()
                # Keep target-region geom (group 5) visible in MuJoCo human window only.
                self._set_human_geom_group_visibility(5, True)
                self._viewer.render("human")
            except Exception:
                pass

        dt = time.time() - start_time
        time.sleep(max(0.0, (1.0 / self.hz) - dt))

        reward = 1.0 if success else 0.0
        info = {"succeed": bool(success)}
        info.update(metrics)
        return obs, reward, terminated, False, info

    def _compute_success_metrics(self) -> Tuple[bool, Dict[str, Any]]:
        region_center = self._data.geom_xpos[self._target_region_geom_id].copy()
        region_rot = self._data.geom_xmat[self._target_region_geom_id].reshape(3, 3)
        view_pos = self._data.site_xpos[self._view_site_id].copy()
        delta_local = region_rot.T @ (view_pos - region_center)
        radial_dist = float(np.linalg.norm(delta_local[:2]))
        region_height_dist = float(abs(delta_local[2]))
        region_pass = radial_dist <= _TARGET_REGION_RADIUS and region_height_dist <= _TARGET_REGION_HALF_HEIGHT

        axis_z_world = self._data.geom_xmat[self._camera_axis_z_geom_id].reshape(3, 3)[:, 2].copy()
        axis_z_norm = float(np.linalg.norm(axis_z_world))
        if axis_z_norm < 1e-8:
            axis_z_world = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            axis_z_world = axis_z_world / axis_z_norm

        cos_val = float(np.clip(np.dot(axis_z_world, _WORLD_POS_X), -1.0, 1.0))
        angle_deg = float(np.degrees(np.arccos(cos_val)))
        angle_pass = angle_deg <= _TARGET_ANGLE_DEG

        shutter_pressed = False
        shutter_contact_count = 0
        ncon = int(self._data.ncon)
        for i in range(ncon):
            c = self._data.contact[i]
            g1 = int(c.geom1)
            g2 = int(c.geom2)
            if g1 == self._shutter_geom_id:
                if g2 not in self._camera_geom_ids:
                    shutter_pressed = True
                    shutter_contact_count += 1
            elif g2 == self._shutter_geom_id:
                if g1 not in self._camera_geom_ids:
                    shutter_pressed = True
                    shutter_contact_count += 1

        success = region_pass and angle_pass and shutter_pressed
        return bool(success), {
            "region_pass": bool(region_pass),
            "angle_pass": bool(angle_pass),
            "shutter_pressed": bool(shutter_pressed),
            "radial_dist": radial_dist,
            "region_height_dist": region_height_dist,
            "view_angle_deg": angle_deg,
            "shutter_contact_count": int(shutter_contact_count),
        }

    def _compute_success(self) -> bool:
        success, _ = self._compute_success_metrics()
        return bool(success)

    def render(self):
        # Refresh the camera screen texture with the latest cam_pos_z view.
        self._update_camera_screen_texture()
        # Hide target-region geom (group 5) in offscreen rgb_array cameras.
        self._set_offscreen_geom_group_visibility(5, False)
        rendered_frames = []
        for cam_id in self.camera_id:
            rendered_frames.append(self._viewer.render(render_mode="rgb_array", camera_id=cam_id))
        return rendered_frames

    def _compute_observation(self, metrics: Dict[str, Any] = None) -> dict:
        if metrics is None:
            _, metrics = self._compute_success_metrics()

        obs = {"state": {}}

        tcp_pos_right = self._data.sensor("franka/flange_pos_right").data
        tcp_quat_right = self._data.sensor("franka/flange_quat_right").data
        tcp_pos_left = self._data.sensor("franka/flange_pos_left").data
        tcp_quat_left = self._data.sensor("franka/flange_quat_left").data
        tcp_pose = np.concatenate(
            [np.concatenate([tcp_pos_right, tcp_quat_right]), np.concatenate([tcp_pos_left, tcp_quat_left])]
        )

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
            "camera_ori_pose": self.camera_ori_pose,
            "logo_ori_pose": self.logo_ori_pose,
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
    env = PandaBimanualPhotographGymEnv(render_mode="human")
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
