#!/usr/bin/env python3
"""
Replay previously recorded Zarr demos under the policy interface and save the
result as a fresh Zarr episode plus MP4 videos.

For each input demo directory containing a `replay.zarr`, the script writes:
- `<exp_name>_demo_<index>_replay_<timestamp>/replay.zarr/`
- `<exp_name>_demo_<index>_replay_<timestamp>/videos/<camera_key>.mp4`

The script plays the recorded action sequence on a freshly reset environment
constructed via `TaskConfig.get_environment(policy_mode=True, ...)`. With
`--randomize=True`, the underlying task draws a new preset camera from
`dexjoco/sim/envs/replay_cameras.npy` plus randomized lighting/texture, which
makes this useful for visual augmentation of an existing dataset.

When `--restore_state=True` (default), `state[0]` from the input zarr is
sliced by the task's `proprio_keys` and the env's initial object poses +
table height are patched back to match the recording. The per-task restorers
live in `dexjoco.tasks.state_restorers`.
"""

import copy
import datetime
from pathlib import Path

import numpy as np
import zarr
from absl import app, flags
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from dexjoco.data.depth_capture import collect_depth_frames, write_depth_outputs
from dexjoco.data.episode_store import ZarrEpisodeStore
from dexjoco.data.video_writer import Mp4VideoWriter
from dexjoco.tasks.mappings import CONFIG_MAPPING
from dexjoco.tasks.state_restorers import has_restorer, restore_initial_state

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "exp_name",
    "water_plant",
    "Task name, such as water_plant.",
)
flags.DEFINE_string(
    "input_dir",
    "./",
    "Directory containing recorded demo folders (each holds a replay.zarr).",
)
flags.DEFINE_string(
    "out_dir",
    "./replay_output",
    "Output base directory for the new zarr and videos.",
)
flags.DEFINE_integer("video_fps", 30, "FPS for saved MP4 videos")
flags.DEFINE_float(
    "data_fps",
    30,
    "Sampling frequency of recorded low-dim data in Hz (used to write timestamps)",
)
flags.DEFINE_bool(
    "randomize",
    True,
    "Enable environment randomization (random preset camera, lighting, texture) at reset",
)
flags.DEFINE_integer(
    "seed",
    0,
    "Base seed for the replay environment; the demo index is added per demo",
)
flags.DEFINE_integer(
    "extend_steps",
    0,
    "Extra steps to repeat the last action after the recorded trajectory ends. "
    "Default 0 is fine for demos recorded by record_demos_zarr.py (the saved "
    "trajectory ends on the succeed step). Set >0 only when replaying older "
    "data that was clipped before success.",
)
flags.DEFINE_bool(
    "save_failed",
    False,
    "Save the replayed demo even when the environment does not report success",
)
flags.DEFINE_bool(
    "restore_state",
    True,
    "Restore the recorded initial scene state (table height + object poses) from state[0]",
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
flags.DEFINE_bool(
    "ipad_screen_effect",
    False,
    "(bimanual_unlock_ipad only) Enable the iPad unlock screen fade-in "
    "transition. When False, the screen flips to unlocked instantly on the "
    "final correct press.",
)


def _safe_squeeze_image(img: np.ndarray) -> np.ndarray:
    """Squeeze common single-batch dimensions and ensure HWC uint8 RGB."""
    if img is None:
        return None
    arr = np.asarray(img)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.concatenate([arr, arr, arr], axis=2)
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            if np.nanmax(arr) <= 1.0:
                arr = np.clip(arr, 0.0, 1.0) * 255.0
            else:
                arr = np.clip(arr, 0.0, 255.0)
            arr = arr.astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    return arr


def convert_action_quat_to_rotvec(action: np.ndarray) -> np.ndarray:
    """Convert action quaternions from wxyz to rotation vectors.

    Supported layouts:
      - single-arm 23-d: [pos3, quat4, 16 joints] -> 22-d
      - bimanual 46-d: [right23, left23] -> 44-d
    """
    action = np.asarray(action)
    if action.ndim != 1:
        return None

    def _convert_single_arm(single_arm_action: np.ndarray) -> np.ndarray:
        pos = single_arm_action[0:3]
        quat = single_arm_action[3:7].astype(np.float64)
        hand_joints = single_arm_action[7:23]

        norm = np.linalg.norm(quat)
        if norm < 1e-8:
            rotvec = np.zeros(3, dtype=np.float64)
        else:
            quat = quat / norm
            quat_xyzw = [quat[1], quat[2], quat[3], quat[0]]
            rotvec = Rotation.from_quat(quat_xyzw).as_rotvec()

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


def _load_episode_actions(zarr_path: Path) -> np.ndarray:
    """Read the recorded action array for a single-episode demo zarr."""
    root = zarr.open(str(zarr_path), mode="r")
    return np.asarray(root["data"]["action"])


def _load_episode_initial_state(zarr_path: Path):
    """Read state[0] for a single-episode demo zarr, or None if not stored."""
    root = zarr.open(str(zarr_path), mode="r")
    if "state" not in root["data"]:
        return None
    return np.asarray(root["data"]["state"][0]).ravel()


def _step_with_recorded_action(env, action_flat: np.ndarray):
    """Step the raw env with a recorded action and rewrap the observation.

    Recorded actions are stored in the same layout that the raw bimanual env consumes
    (``[right(23), left(23)]``), whereas ``DualArmPolicyWrapper`` expects
    ``[r_pose, l_pose, r_hand, l_hand]``. We bypass the policy wrapper and feed the
    raw env directly to keep replay faithful to the recording.
    """
    raw_env = env.unwrapped
    action_flat = np.asarray(action_flat, dtype=np.float64)
    if action_flat.shape == (46,):
        raw_action = {
            "right": action_flat[:23],
            "left": action_flat[23:46],
        }
    else:
        raw_action = action_flat
    raw_obs, rew, done, trunc, info = raw_env.step(raw_action)
    return env.observation(raw_obs), rew, done, trunc, info


def _collect_camera_keys(observation: dict) -> list:
    keys = []
    for k, v in observation.items():
        if isinstance(v, np.ndarray) and v.ndim >= 3 and v.shape[-1] == 3:
            keys.append(k)
    return keys


def _write_demo_zarr_and_videos(
    trajectory, exp_name: str, success_index: int, base_out: Path, video_fps: int
):
    """Save one replayed trajectory to a zarr group plus MP4 camera captures."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    demo_dir = base_out / f"{exp_name}_demo_{success_index}_replay_{timestamp}"
    demo_dir.mkdir(parents=True, exist_ok=True)

    actions = np.stack([t["actions"] for t in trajectory], axis=0)
    T = actions.shape[0]
    timestamps_arr = np.arange(T) / float(FLAGS.data_fps)

    episode_data = {
        "action": actions,
        "timestamp": timestamps_arr,
    }

    states = [t["observations"].get("state") for t in trajectory]
    if all(isinstance(s, np.ndarray) for s in states):
        try:
            episode_data["state"] = np.stack(states, axis=0)
        except ValueError:
            pass

    converted = [convert_action_quat_to_rotvec(a) for a in actions]
    if all(c is not None for c in converted):
        episode_data["action_rotvec"] = np.stack(converted, axis=0)

    zarr_path = demo_dir / "replay.zarr"
    store = zarr.DirectoryStore(str(zarr_path))
    episode_store = ZarrEpisodeStore.create_empty(storage=store)
    episode_store.append_episode(episode_data, compressors="disk")
    print(f"[Saved replay.zarr] {zarr_path} (steps: {T})")

    videos_dir = demo_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    camera_keys = []
    for step in trajectory:
        for k in _collect_camera_keys(step["observations"]):
            if k not in camera_keys:
                camera_keys.append(k)

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
            img = step["observations"].get(cam_key)
            if img is None:
                continue
            video_writer.write_frame(_safe_squeeze_image(img))

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


def _replay_single_demo(
    actions: np.ndarray,
    initial_state: np.ndarray,
    task_id: str,
    config,
    env_seed: int,
    desc: str,
):
    env_extra_kwargs = {}
    if task_id == "bimanual_photograph":
        env_extra_kwargs["camera_screen_effect"] = FLAGS.camera_screen_effect
    if task_id == "bimanual_unlock_ipad":
        env_extra_kwargs["ipad_screen_effect"] = FLAGS.ipad_screen_effect

    env = config.get_environment(
        policy_mode=True,
        render_mode="rgb_array",
        randomize=FLAGS.randomize,
        randomize_dynamics=False,
        seed=env_seed,
        **env_extra_kwargs,
    )
    try:
        obs, _info = env.reset()

        if FLAGS.restore_state and initial_state is not None and has_restorer(task_id):
            obs = restore_initial_state(env, task_id, config, initial_state)

        trajectory = []
        succeed = False
        num_steps = actions.shape[0]
        total_steps = num_steps + max(0, FLAGS.extend_steps)

        for step_idx in tqdm(range(total_steps), desc=desc):
            action = actions[step_idx if step_idx < num_steps else -1]
            # Capture depth BEFORE step so it lines up with `obs` (which
            # is what gets stored alongside this transition's action).
            depth_now = (
                collect_depth_frames(env, _collect_camera_keys(obs))
                if FLAGS.save_depth else None
            )
            next_obs, _rew, done, _trunc, info = _step_with_recorded_action(env, action)
            transition = dict(observations=obs, actions=action, dones=done, infos=info)
            if depth_now is not None:
                transition["depth"] = depth_now
            trajectory.append(copy.deepcopy(transition))
            obs = next_obs
            if info.get("succeed", False):
                succeed = True
    finally:
        try:
            env.close()
        except Exception:
            pass

    return succeed, trajectory


def main(_argv):
    task_id = FLAGS.exp_name
    config = CONFIG_MAPPING[task_id]()

    base_out = Path(FLAGS.out_dir)
    base_out.mkdir(parents=True, exist_ok=True)

    input_root = Path(FLAGS.input_dir)
    demo_dirs = sorted(p.parent for p in input_root.glob("*/replay.zarr"))
    if not demo_dirs:
        print(f"No replay.zarr found under {input_root}")
        return

    saved_demo_dirs = []
    for index, demo_dir in enumerate(demo_dirs, start=1):
        print(f"\n[{index}/{len(demo_dirs)}] {demo_dir.name}")
        zarr_path = demo_dir / "replay.zarr"
        actions = _load_episode_actions(zarr_path)
        initial_state = _load_episode_initial_state(zarr_path)
        if FLAGS.restore_state and initial_state is None:
            print("[Warning] state[0] not found in input zarr; replay will use the reset scene as-is.")

        succeed, trajectory = _replay_single_demo(
            actions, initial_state, task_id, config,
            env_seed=FLAGS.seed + index, desc=demo_dir.name,
        )
        print(f"Replay finished: succeed={succeed}, steps={len(trajectory)}")

        if not trajectory:
            continue
        if not (succeed or FLAGS.save_failed):
            print("[Skipped] env did not report success; pass --save_failed to keep.")
            continue

        saved = _write_demo_zarr_and_videos(
            trajectory, task_id, index, base_out, FLAGS.video_fps
        )
        saved_demo_dirs.append(saved)

    print(f"\nReplayed {len(saved_demo_dirs)}/{len(demo_dirs)} demos:")
    for p in saved_demo_dirs:
        print("  ", p)


if __name__ == "__main__":
    app.run(main)
