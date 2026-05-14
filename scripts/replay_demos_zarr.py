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

When `--restore_state=True` (default), `state[0]` from the input zarr is sliced
by the task's `proprio_keys` and the env's initial object poses + table height
are patched back to match the recording.
"""

import copy
import datetime
from pathlib import Path

import mujoco
import numpy as np
import zarr
from absl import app, flags
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from dexjoco.data.episode_store import ZarrEpisodeStore
from dexjoco.data.video_writer import Mp4VideoWriter
from dexjoco.tasks.mappings import CONFIG_MAPPING

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
    60,
    "Extra steps to repeat the last action after the recorded trajectory ends",
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


def _split_state_by_proprio(state_vec: np.ndarray, proprio_keys, ref_state: dict) -> dict:
    """Split the flat recorded state vector back into a {proprio_key: array} dict.

    Uses sizes from a fresh `_compute_observation()` sample because some envs declare
    `gripper_pose` as shape `(1,)` in `observation_space` while actually returning a
    16-d sensor vector; `DexjocoObsAdapter` flattens the real array, not the spec.
    """
    parts = {}
    offset = 0
    for key in proprio_keys:
        size = max(1, np.asarray(ref_state[key]).ravel().size)
        parts[key] = np.asarray(state_vec[offset : offset + size], dtype=np.float64)
        offset += size
    return parts


def _apply_delta_h(raw_env, delta_h: float) -> None:
    """Mimic each env's table-height adjustment based on what attributes it caches."""
    raw_env.delta_h = np.float64(delta_h)
    z0 = getattr(raw_env, "_table_body_z0", None)
    if z0 is None:
        z0 = raw_env._table_z
    table_body_id = raw_env._model.body("table").id
    raw_env._model.body_pos[table_body_id, 2] = z0 + delta_h

    # Resolve which leg geoms to adjust:
    #   - most envs cache `_table_leg_geom_ids` (either hardcoded or discovered)
    #   - water_plant caches `_table_leg_half_len0` keyed by leg name instead
    #   - bimanual_microwave_cook caches neither and iterates 4 hardcoded names
    if hasattr(raw_env, "_table_leg_geom_ids"):
        leg_gids = list(raw_env._table_leg_geom_ids)
    else:
        h0_map = getattr(raw_env, "_table_leg_half_len0", None)
        if isinstance(h0_map, dict) and h0_map and isinstance(next(iter(h0_map)), str):
            leg_gids = [raw_env._model.geom(n).id for n in h0_map]
        else:
            leg_gids = [
                raw_env._model.geom(n).id
                for n in ("table_leg_1", "table_leg_2", "table_leg_3", "table_leg_4")
            ]

    if hasattr(raw_env, "_model_geom_pos0"):
        # bimanual_microwave_cook / bimanual_unlock_ipad: shift center down and extend size by half.
        for gid in leg_gids:
            raw_env._model.geom_pos[gid, 2] = raw_env._model_geom_pos0[gid, 2] - 0.5 * delta_h
            raw_env._model.geom_size[gid, 1] = raw_env._model_geom_size0[gid, 1] + 0.5 * delta_h
        return

    # Standard style: extend each leg by the full delta.
    h0_map = raw_env._table_leg_half_len0
    for gid in leg_gids:
        if gid in h0_map:
            h0 = h0_map[gid]
        else:
            name = mujoco.mj_id2name(raw_env._model, mujoco.mjtObj.mjOBJ_GEOM, gid)
            h0 = h0_map[name]
        raw_env._model.geom_size[gid, 1] = h0 + delta_h


def _restore_water_plant(raw_env, parts):
    _apply_delta_h(raw_env, float(parts["table_delta_height"][0]))
    spray = parts["spray_ori_pose"]
    raw_env._data.jnt("spray_root").qpos[:3] = spray[:3]
    raw_env._data.jnt("spray_root").qpos[3:7] = spray[3:7]
    raw_env._spray_ori_pose = spray.copy()
    plant = parts["plant_ori_pose"]
    raw_env._model.body("plant").pos = plant[:3]
    raw_env._plant_ori_pose = plant.copy()


def _restore_click_mouse(raw_env, parts):
    _apply_delta_h(raw_env, float(parts["table_delta_height"][0]))
    mouse = parts["mouse_ori_pose"]
    raw_env._data.jnt("mouse_root").qpos = mouse
    raw_env.mouse_ori_pose = mouse.copy()


def _restore_pinch_tongs(raw_env, parts):
    _apply_delta_h(raw_env, float(parts["table_delta_height"][0]))
    tongs = parts["tongs_ori_pose"]
    raw_env._data.jnt("tongs_root").qpos = tongs
    raw_env.tongs_ori_pose = tongs.copy()


def _restore_fold_glasses(raw_env, parts):
    _apply_delta_h(raw_env, float(parts["table_delta_height"][0]))
    glass = parts["glass_ori_pose"]
    raw_env._data.jnt("glass_root").qpos = glass
    raw_env.glass_ori_pose = glass.copy()
    box = parts["box_ori_pose"]
    box_body_id = raw_env._model.body("open_box").id
    raw_env._model.body_pos[box_body_id] = box[:3]
    raw_env._model.body_quat[box_body_id] = box[3:7]
    raw_env.open_box_ori_pose = box.copy()


def _restore_hammer_nail(raw_env, parts):
    _apply_delta_h(raw_env, float(parts["table_delta_height"][0]))
    raw_env._model.body_pos[raw_env._model.body("wood").id, 2] = (
        raw_env._wood_body_z0 + raw_env.delta_h
    )
    hammer = parts["hammer_ori_pose"]
    raw_env._data.jnt("hammer_joint").qpos = hammer
    raw_env.hammer_ori_pose = hammer.copy()
    nail = parts["nail_ori_pose"]
    nail_pos = nail[:3].copy()
    nail_pos[2] = raw_env._nail_body_z0 + raw_env.delta_h
    raw_env._nail_init_pos[:] = nail_pos
    raw_env._data.mocap_pos[raw_env._nail_mocap_id] = nail_pos
    raw_env._data.mocap_quat[raw_env._nail_mocap_id] = nail[3:7]
    raw_env.nail_ori_pose = np.concatenate([nail_pos, nail[3:7]])


def _restore_pick_bucket(raw_env, parts):
    _apply_delta_h(raw_env, float(parts["table_delta_height"][0]))
    bucket = parts["bucket_ori_pose"]
    raw_env._data.jnt("bucket_root").qpos = bucket
    raw_env.bucket_ori_pose = bucket.copy()
    raw_env._bucket_z = float(bucket[2])
    boxed_food = parts["boxed_food_ori_pose"]
    raw_env._data.jnt("boxed_food_0_freejoint").qpos = boxed_food
    raw_env.box_food_ori_pose = boxed_food.copy()
    # Re-baseline `_bucket_bottom_z0`: env.reset() captures it at the
    # randomly-sampled bucket pose, but success uses (bottom_z - z0 >= 0.15)
    # so the baseline must reflect the restored bucket pose.
    mujoco.mj_forward(raw_env._model, raw_env._data)
    raw_env._bucket_bottom_z0 = raw_env._data.site_xpos[
        raw_env._bucket_bottom_site_ids, 2
    ].copy()


def _restore_bimanual_assembly(raw_env, parts):
    _apply_delta_h(raw_env, float(parts["table_delta_height"][0]))
    socket = parts["socket_ori_pose"]
    raw_env._set_free_joint_pose(
        raw_env._socket_qpos_adr, raw_env._socket_qvel_adr, socket[:3], socket[3:7]
    )
    raw_env._socket_ori_pose = socket.copy()
    peg = parts["peg_ori_pose"]
    raw_env._set_free_joint_pose(
        raw_env._peg_qpos_adr, raw_env._peg_qvel_adr, peg[:3], peg[3:7]
    )
    raw_env._peg_ori_pose = peg.copy()


def _restore_bimanual_hanoi(raw_env, parts):
    _apply_delta_h(raw_env, float(parts["table_delta_height"][0]))
    base = np.asarray(parts["hanoi_base_ori_pos"], dtype=np.float64)
    raw_env._model.body_pos[raw_env._base_body_id] = base
    raw_env.base_ori_pos = base.copy()
    # Re-stack disks at the recorded base position; the tower preset matches
    # what reset() would pick, but we need to recompute it relative to the
    # restored base position (otherwise disks float above the prior location).
    base_delta_xy = base[:2] - raw_env._base_init_pos[:2]
    _, tower_state = raw_env._sample_reset_tower_state()
    raw_env._apply_reset_tower_state(tower_state, base_delta_xy)


def _restore_bimanual_microwave_cook(raw_env, parts):
    _apply_delta_h(raw_env, float(parts["table_delta_height"][0]))
    hot_dog = parts["hot_dog_ori_pose"]
    raw_env._data.jnt("hot_dog_free").qpos = hot_dog
    raw_env._hot_dog_ori_pose = hot_dog.copy()
    microwave = parts["microwave_ori_pose"]
    raw_env._model.body("microwave_object").pos = microwave[:3]
    raw_env._model.body("microwave_object").quat = microwave[3:7]
    raw_env._microwave_ori_pose = microwave.copy()


def _restore_bimanual_photograph(raw_env, parts):
    _apply_delta_h(raw_env, float(parts["table_delta_height"][0]))
    # The target_region geom is reset to (logo_pos + constant offset); preserve
    # that constant offset when we move the logo to the recorded pose.
    target_offset = (
        raw_env._model.geom_pos[raw_env._target_region_geom_id]
        - raw_env._model.geom_pos[raw_env._logo_geom_id]
    ).copy()
    logo = parts["logo_ori_pose"]
    raw_env._model.geom_pos[raw_env._logo_geom_id] = logo[:3]
    raw_env._model.geom_quat[raw_env._logo_geom_id] = logo[3:7]
    raw_env._model.geom_pos[raw_env._target_region_geom_id] = logo[:3] + target_offset
    raw_env.logo_ori_pose = logo.copy()
    camera = parts["camera_ori_pose"]
    raw_env._data.jnt("camera_root").qpos = camera
    raw_env.camera_ori_pose = camera.copy()


def _restore_bimanual_unlock_ipad(raw_env, parts):
    _apply_delta_h(raw_env, float(parts["table_delta_height"][0]))
    stand = parts["stand_ori_pose"]
    stand_body_id = raw_env._model.body("ipad_stand").id
    raw_env._model.body_pos[stand_body_id] = stand[:3]
    raw_env._stand_ori_pose = stand.copy()
    ipad = parts["ipad_ori_pose"]
    raw_env._data.jnt("ipad_freejoint").qpos[:3] = ipad[:3]
    raw_env._ipad_ori_pose = ipad.copy()


_STATE_RESTORERS = {
    "water_plant": _restore_water_plant,
    "click_mouse": _restore_click_mouse,
    "pinch_tongs": _restore_pinch_tongs,
    "fold_glasses": _restore_fold_glasses,
    "hammer_nail": _restore_hammer_nail,
    "pick_bucket": _restore_pick_bucket,
    "bimanual_assembly": _restore_bimanual_assembly,
    "bimanual_hanoi": _restore_bimanual_hanoi,
    "bimanual_microwave_cook": _restore_bimanual_microwave_cook,
    "bimanual_photograph": _restore_bimanual_photograph,
    "bimanual_unlock_ipad": _restore_bimanual_unlock_ipad,
}


def _restore_initial_state(env, task_id: str, config, state_vec: np.ndarray):
    """Patch the freshly reset env to match the recorded state[0] and return refreshed obs."""
    raw_env = env.unwrapped
    ref_state = raw_env._compute_observation()["state"]
    parts = _split_state_by_proprio(state_vec, config.proprio_keys, ref_state)
    _STATE_RESTORERS[task_id](raw_env, parts)
    mujoco.mj_forward(raw_env._model, raw_env._data)
    return env.observation(raw_env._compute_observation())


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

    return str(demo_dir)


def _replay_single_demo(
    actions: np.ndarray,
    initial_state: np.ndarray,
    task_id: str,
    config,
    env_seed: int,
    desc: str,
):
    env = config.get_environment(
        policy_mode=True,
        render_mode="rgb_array",
        randomize=FLAGS.randomize,
        randomize_dynamics=False,
        seed=env_seed,
    )
    try:
        obs, _info = env.reset()

        if FLAGS.restore_state and initial_state is not None and task_id in _STATE_RESTORERS:
            obs = _restore_initial_state(env, task_id, config, initial_state)

        trajectory = []
        succeed = False
        num_steps = actions.shape[0]
        total_steps = num_steps + max(0, FLAGS.extend_steps)

        for step_idx in tqdm(range(total_steps), desc=desc):
            action = actions[step_idx if step_idx < num_steps else -1]
            next_obs, _rew, done, _trunc, info = _step_with_recorded_action(env, action)
            trajectory.append(
                copy.deepcopy(
                    dict(observations=obs, actions=action, dones=done, infos=info)
                )
            )
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
