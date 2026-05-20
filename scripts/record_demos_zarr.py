#!/usr/bin/env python3
"""
Record successful simulated teleoperation demos as Zarr episodes plus MP4 videos.

For each successful episode, the script writes:
- `replay.zarr/` with low-dimensional observations and actions
- `videos/<camera_key>.mp4` for every RGB camera stream present in the observations
"""

import copy
import datetime
from pathlib import Path

import cv2
import numpy as np
import zarr
from absl import app, flags

# Project-specific utilities for replay storage and video writing.
from dexjoco.data.depth_capture import collect_depth_frames, write_depth_outputs
from dexjoco.data.episode_store import ZarrEpisodeStore
from dexjoco.data.video_writer import Mp4VideoWriter
from dexjoco.tasks.mappings import CONFIG_MAPPING
from scipy.spatial.transform import Rotation
from tqdm import tqdm

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "exp_name",
    "water_plant",
    "Task name, such as water_plant.",
)
flags.DEFINE_integer("successes_needed", 2, "Number of successful demos to collect.")
flags.DEFINE_bool("show_sim_cameras", True, "Show simulation cameras")
flags.DEFINE_integer(
    "max_steps",
    0,
    "Stop after this many environment steps; 0 means run until successes_needed",
)
flags.DEFINE_string("out_dir", "./", "Output base directory for zarr and videos")
flags.DEFINE_integer("video_fps", 30, "FPS for saved MP4 videos")
flags.DEFINE_float(
    "data_fps",
    30,
    "Sampling frequency of recorded low-dim data in Hz (used to write timestamps)",
)
flags.DEFINE_enum(
    "render_mode",
    "human",
    ["human", "rgb_array"],
    "MuJoCo render mode for the task environment",
)
flags.DEFINE_bool(
    "randomize",
    False,
    "Enable environment randomization when the selected task supports it",
)
flags.DEFINE_bool(
    "save_depth",
    False,
    "Also render a depth_array per RGB camera and save <cam>_depth.npz + .mp4 alongside each video",
)
flags.DEFINE_bool(
    "camera_screen_effect",
    False,
    "(bimanual_photograph only) Enable the camera live-screen effect "
    "(live preview + shutter flash + frozen photo). When False, the screen "
    "geom is hidden entirely.",
)


def _ensure_base_outdir(base: str) -> Path:
    p = Path(base)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _safe_squeeze_image(img: np.ndarray) -> np.ndarray:
    """Squeeze common single-batch dimensions and ensure HWC uint8 RGB.

    Accepts image formats: (1,H,W,3), (H,W,3), (H,W) (grayscale).
    Returns (H,W,3) uint8 RGB.
    """
    if img is None:
        return None
    arr = np.asarray(img)
    # squeeze batch dim if present
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    # grayscale to 3-channel
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    # if single-channel last dim, expand
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.concatenate([arr, arr, arr], axis=2)
    # ensure dtype is uint8
    if arr.dtype != np.uint8:
        # try to scale floats in [0,1] to [0,255]
        if np.issubdtype(arr.dtype, np.floating):
            if np.nanmax(arr) <= 1.0:
                arr = np.clip(arr, 0.0, 1.0) * 255.0
            else:
                arr = np.clip(arr, 0.0, 255.0)
            arr = arr.astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    # At this point arr is uint8 but might be BGR or RGB depending on source.
    # We assume observations are RGB (common in sim). If you know they are
    # BGR, flip channels where needed before writing with Mp4VideoWriter.
    return arr


def _to_bgr_image(image: np.ndarray) -> np.ndarray:
    """Convert a top-level observation image into HWC uint8 BGR for OpenCV."""
    if image is None:
        return None

    frame = np.asarray(image)
    if frame.ndim == 4 and frame.shape[0] == 1:
        frame = frame[0]
    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=2)
    if frame.ndim == 3 and frame.shape[2] == 1:
        frame = np.repeat(frame, 3, axis=2)
    if frame.ndim != 3 or frame.shape[2] != 3:
        return None

    if frame.dtype != np.uint8:
        if np.issubdtype(frame.dtype, np.floating):
            if np.nanmax(frame) <= 1.0:
                frame = np.clip(frame, 0.0, 1.0) * 255.0
            else:
                frame = np.clip(frame, 0.0, 255.0)
            frame = frame.astype(np.uint8)
        else:
            frame = frame.astype(np.uint8)

    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


class WristCameraViewer:
    """Display only wrist cameras during teleoperation data collection."""

    WINDOW_TITLES = {
        "wrist": "Wrist",
        "wrist_left": "Wrist Left",
        "wrist_right": "Wrist Right",
    }

    def __init__(self, display_scale=0.75):
        self.display_scale = display_scale
        self.active_windows = {}

    def _get_visible_keys(self, observation):
        if not isinstance(observation, dict):
            return []

        if _to_bgr_image(observation.get("wrist")) is not None:
            return ["wrist"]

        visible_keys = []
        for camera_key in ("wrist_left", "wrist_right"):
            if _to_bgr_image(observation.get(camera_key)) is not None:
                visible_keys.append(camera_key)
        return visible_keys

    def _ensure_windows(self, visible_keys):
        active_keys = set(self.active_windows)
        visible_key_set = set(visible_keys)

        for camera_key in active_keys - visible_key_set:
            try:
                cv2.destroyWindow(self.active_windows[camera_key])
            except cv2.error:
                pass
            del self.active_windows[camera_key]

        for camera_key in visible_keys:
            if camera_key not in self.active_windows:
                window_name = self.WINDOW_TITLES[camera_key]
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                self.active_windows[camera_key] = window_name

    def update_and_show(self, observation):
        """Show wrist-only windows and return the last pressed key."""
        visible_keys = self._get_visible_keys(observation)
        self._ensure_windows(visible_keys)

        if not visible_keys:
            return cv2.waitKey(1) & 0xFF

        for camera_key in visible_keys:
            frame = _to_bgr_image(observation.get(camera_key))
            if frame is None:
                continue
            if self.display_scale != 1.0:
                new_width = max(1, int(frame.shape[1] * self.display_scale))
                new_height = max(1, int(frame.shape[0] * self.display_scale))
                frame = cv2.resize(
                    frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR
                )
            window_name = self.active_windows[camera_key]
            cv2.imshow(window_name, frame)
            cv2.resizeWindow(window_name, frame.shape[1], frame.shape[0])

        return cv2.waitKey(1) & 0xFF

    def close(self):
        for window_name in list(self.active_windows.values()):
            try:
                cv2.destroyWindow(window_name)
            except cv2.error:
                pass
        self.active_windows.clear()


def convert_action_quat_to_rotvec(action: np.ndarray) -> np.ndarray:
    """
    Convert action quaternions from wxyz to rotation vectors.

    Supported layouts:
      - single-arm 23-d: [pos3, quat4, 16 joints] -> 22-d
      - bimanual 46-d: [right23, left23] -> 44-d

    If quaternion has zero norm, treat it as identity rotation.
    """
    action = np.asarray(action)
    if action.ndim != 1:
        return None

    def _convert_single_arm(single_arm_action: np.ndarray) -> np.ndarray:
        pos = single_arm_action[0:3]
        quat = single_arm_action[3:7]
        hand_joints = single_arm_action[7:23]

        quat = quat.astype(np.float64)
        norm = np.linalg.norm(quat)

        if norm < 1e-8:
            rotvec = np.zeros(3, dtype=np.float64)
        else:
            quat = quat / norm
            quat_xyzw = [quat[1], quat[2], quat[3], quat[0]]
            rot = Rotation.from_quat(quat_xyzw)
            rotvec = rot.as_rotvec()

        return np.concatenate([pos, rotvec, hand_joints])

    if action.shape[0] == 23:
        return _convert_single_arm(action)

    if action.shape[0] == 46:
        return np.concatenate(
            [
                _convert_single_arm(action[:23]),
                _convert_single_arm(action[23:46]),
            ],
            axis=0,
        )

    return None


def _flatten_action_for_storage(action) -> np.ndarray:
    if isinstance(action, dict):
        if "right" not in action or "left" not in action:
            raise ValueError(
                f"Dict action must contain right and left keys, got {sorted(action)}."
            )
        return np.concatenate(
            [
                np.asarray(action["right"], dtype=np.float64).reshape(-1),
                np.asarray(action["left"], dtype=np.float64).reshape(-1),
            ],
            axis=0,
        )
    return np.asarray(action, dtype=np.float64).reshape(-1)


def _write_demo_zarr_and_videos(
    trajectory, exp_name: str, success_index: int, base_out: Path, video_fps: int
):
    """Save a single demo trajectory to a zarr group and MP4 camera captures.

    Structure created:
      <base_out>/<exp_name>_demo_<index>_<timestamp>/
         replay.zarr/
         videos/<cam_key>.mp4

    The episode store will contain arrays for:
      - action: (T, N)
      - action_rotvec: (T,22) for single-arm or (T,44) for bimanual if convertible
      - state: (T, ... ) if present under observations['state']
      - timestamp: (T,) in seconds
      - any small consistent low-dim fields
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    demo_dir = base_out / f"{exp_name}_demo_{success_index}_{timestamp}"
    demo_dir.mkdir(parents=True, exist_ok=True)

    # Prepare episode-level low-dim arrays and cameras
    actions = []
    states = []
    timestamps = []
    extra_lowdim = {}
    camera_keys = []

    for step_idx, transition in enumerate(trajectory):
        obs = transition.get("observations", {})
        act = transition.get("actions")
        info = transition.get("infos") or transition.get("info") or {}

        actions.append(np.asarray(act))

        if isinstance(obs, dict) and ("state" in obs or "low_dim" in obs):
            s = obs.get("state", obs.get("low_dim"))
            states.append(np.asarray(s))

        # collect any timestamp if environment provides one
        t_val = None
        if isinstance(info, dict):
            if "timestamp" in info:
                t_val = info["timestamp"]
            elif "time" in info:
                t_val = info["time"]
        timestamps.append(t_val)

        if isinstance(obs, dict):
            for k, v in obs.items():
                if k in ("state", "low_dim"):
                    continue
                if isinstance(v, np.ndarray) and v.ndim >= 3 and v.shape[-1] == 3:
                    if k not in camera_keys:
                        camera_keys.append(k)
                if isinstance(v, np.ndarray) and v.ndim == 1 and v.size <= 128:
                    if k not in extra_lowdim:
                        extra_lowdim[k] = []
                    extra_lowdim[k].append(v)

    # Build episode dict for the Zarr episode store.
    actions_arr = np.stack(actions, axis=0)
    T = actions_arr.shape[0]
    timestamps_arr = np.arange(T) / float(FLAGS.data_fps)

    episode_data = {
        "action": actions_arr,
        "timestamp": timestamps_arr,
    }

    if len(states) > 0:
        try:
            episode_data["state"] = np.stack(states, axis=0)
        except Exception:
            episode_data["state"] = np.array(states, dtype=object)

    # add extra low-dim fields if consistent
    for k, lst in list(extra_lowdim.items()):
        if len(lst) == T:
            try:
                episode_data[k] = np.stack(lst, axis=0)
            except Exception:
                episode_data[k] = np.array(lst, dtype=object)

    # action_rotvec when convertible
    converted = [convert_action_quat_to_rotvec(a) for a in episode_data["action"]]
    if all(c is not None for c in converted):
        episode_data["action_rotvec"] = np.stack(converted, axis=0)

    # Create the Zarr episode store.
    zarr_path = demo_dir / "replay.zarr"
    store = zarr.DirectoryStore(str(zarr_path))
    episode_store = ZarrEpisodeStore.create_empty(storage=store)

    # Append the current episode to disk.
    episode_store.append_episode(episode_data, compressors="disk")

    print(f"[Saved replay.zarr] {zarr_path} (steps: {episode_data['action'].shape[0]})")

    # --- write videos per discovered camera key ---
    videos_dir = demo_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    for cam_key in camera_keys:
        video_writer = Mp4VideoWriter.create_h264(
            fps=video_fps,
            codec="h264",
            input_pix_fmt="rgb24",
            crf=21,
            thread_type="FRAME",
            thread_count=2,
        )

        out_path = str(videos_dir / f"{cam_key}.mp4")
        video_writer.start(out_path)

        for step in trajectory:
            obs = step.get("observations", {})
            img = obs.get(cam_key)
            if img is None:
                # write a black frame with same dimensions as first discovered
                # (fallback: skip writing this frame)
                continue
            frame = _safe_squeeze_image(img)
            # _safe_squeeze_image returns RGB uint8. Mp4VideoWriter expects RGB.
            video_writer.write_frame(frame)

        video_writer.stop()
        print(f"[Saved video] {out_path}")

    if FLAGS.save_depth:
        depth_per_camera = {k: [] for k in camera_keys}
        for step in trajectory:
            for k, frame in step.get("depth", {}).items():
                if k in depth_per_camera and frame is not None:
                    depth_per_camera[k].append(frame)
        for path in write_depth_outputs(depth_per_camera, videos_dir, video_fps):
            print(f"[Saved depth]  {path}")

    return str(demo_dir)


def _has_displayable_wrist_images(obs) -> bool:
    if not isinstance(obs, dict):
        return False
    for key in ("wrist", "wrist_left", "wrist_right"):
        value = obs.get(key)
        if value is None:
            continue
        arr = np.asarray(value)
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim == 3 and arr.shape[-1] in (1, 3):
            return True
    return False


def main(_argv):
    task_id = FLAGS.exp_name

    config = CONFIG_MAPPING[task_id]()

    env_extra_kwargs = {}
    if task_id == "bimanual_photograph":
        env_extra_kwargs["camera_screen_effect"] = FLAGS.camera_screen_effect

    env = config.get_environment(
        render_mode=FLAGS.render_mode,
        randomize=FLAGS.randomize,
        **env_extra_kwargs,
    )

    obs, info = env.reset()
    print(f"Environment reset complete for {task_id}.")

    base_out = _ensure_base_outdir(FLAGS.out_dir)
    saved_demo_dirs = []
    success_count = 0
    success_needed = FLAGS.successes_needed

    pbar = tqdm(total=success_needed)

    trajectory = []  # temporary storage for current episode transitions
    returns = 0.0
    total_steps = 0

    # Wrist-camera viewer initialization (optional)
    wrist_cam_viewer = None
    if FLAGS.show_sim_cameras and _has_displayable_wrist_images(obs):
        wrist_cam_viewer = WristCameraViewer()

    # Main collection loop
    while success_count < success_needed:
        actions = np.zeros(env.action_space.sample().shape)

        # Capture depth now (env state currently matches `obs`); it'll be
        # stored alongside `obs` below for parity with the RGB cameras.
        depth_now = None
        if FLAGS.save_depth and isinstance(obs, dict):
            obs_image_keys = [
                k for k, v in obs.items()
                if k not in ("state", "low_dim")
                and isinstance(v, np.ndarray) and v.ndim >= 3 and v.shape[-1] == 3
            ]
            depth_now = collect_depth_frames(env, obs_image_keys)

        if (
            FLAGS.show_sim_cameras
            and wrist_cam_viewer is None
            and _has_displayable_wrist_images(obs)
        ):
            wrist_cam_viewer = WristCameraViewer()

        if wrist_cam_viewer is not None:
            key = wrist_cam_viewer.update_and_show(obs)
            if key == ord("r"):
                print("[Keyboard] Reset triggered")
                trajectory = []
                returns = 0.0
                obs, info = env.reset()
                continue

        next_obs, rew, done, truncated, info = env.step(actions)
        total_steps += 1
        returns += rew

        if "intervene_action" in info:
            actions = info["intervene_action"]

        actions = _flatten_action_for_storage(actions)
        transition = copy.deepcopy(
            dict(observations=obs, actions=actions, dones=done, infos=info)
        )
        if depth_now is not None:
            transition["depth"] = depth_now
        trajectory.append(transition)

        pbar.set_description(f"Return: {returns}")
        obs = next_obs

        if info.get("manual_reset", False):
            print("[Teleop] Manual reset triggered; dropping current trajectory.")
            trajectory = []
            returns = 0.0
            obs, info = env.reset()
            continue

        if done:
            if info.get("succeed", False):
                success_count += 1
                demo_dir = _write_demo_zarr_and_videos(
                    trajectory, task_id, success_count, base_out, FLAGS.video_fps
                )
                saved_demo_dirs.append(demo_dir)
                print(f"[Saved] success #{success_count} -> {demo_dir}")
                pbar.update(1)

                # release trajectory
                trajectory = []
                returns = 0.0
            else:
                trajectory = []
                returns = 0.0

            obs, info = env.reset()

        if FLAGS.max_steps > 0 and total_steps >= FLAGS.max_steps:
            print(f"Reached max_steps={FLAGS.max_steps}; stopping run.")
            break

    print(f"Collected {success_count} successful demos. Individual demo directories:")
    for p in saved_demo_dirs:
        print("  ", p)

    # Attempt to close viewer and env
    if wrist_cam_viewer is not None:
        try:
            wrist_cam_viewer.close()
        except Exception:
            pass

    try:
        env.close()
    except Exception:
        pass

    cv2.destroyAllWindows()


if __name__ == "__main__":
    app.run(main)
