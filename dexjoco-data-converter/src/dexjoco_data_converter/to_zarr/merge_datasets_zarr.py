import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from queue import Empty

import numpy as np
import yaml
import zarr
from diffusion_policy.common.replay_buffer import ReplayBuffer  # type: ignore

from ..episode_common import find_first_non_static_frame
from ..process_protocol import DoneMsg, ErrorMsg, InitMsg, ProgressBar, ProgressMsg
from ..utils import dict_to_slice, normalize_array_shape, terminate_process
from .merge_episode_zarr import _process_video_task


def _pad_to_dim(arr: np.ndarray, target_dim: int) -> np.ndarray:
    """Right-pad the feature dimension of an array with zeros.

    Args:
        arr: Array whose second dimension is the feature dimension.
        target_dim: Required feature dimension.

    Returns:
        Array with the same leading and trailing dimensions and a second
        dimension equal to ``target_dim``.

    Raises:
        AssertionError: If the array has fewer than two dimensions.
        Exception: If the current feature dimension is larger than ``target_dim``.
    """
    assert arr.ndim >= 2, f"arr must have at least 2 dims, got {arr.shape}"
    current_dim = arr.shape[1]
    if current_dim > target_dim:
        raise Exception(f"arr dim {current_dim} exceeds target dim {target_dim}")
    if current_dim == target_dim:
        return arr

    pad_width = [(0, 0)] * arr.ndim
    pad_width[1] = (0, target_dim - current_dim)
    return np.pad(arr, pad_width, mode="constant", constant_values=0)


def _save_video_name_map(
    output_dir: Path, dataset_specs: list[dict], output_video_map: dict[str, str]
) -> None:
    """Write camera-name mapping metadata for a merged Zarr dataset.

    Args:
        output_dir: Directory where ``video_name_map.yaml`` is written.
        dataset_specs: Normalized dataset specifications containing camera maps.
        output_video_map: Mapping from standard camera names to output video names.

    Returns:
        None.
    """
    # dataset_raw_to_std_map: dataset_id -> raw_cam_name -> [std_cam_name, ...]
    dataset_raw_to_std_map = {}
    for spec in dataset_specs:
        raw_to_std: dict[str, list[str]] = {}
        for std_cam, raw_cam in spec["cam_map"].items():
            raw_to_std.setdefault(raw_cam, []).append(std_cam)
        dataset_raw_to_std_map[spec["dataset_id"]] = raw_to_std

    with open(output_dir / "video_name_map.yaml", "w") as f:
        yaml.safe_dump(
            {
                "dataset_raw_to_std_map": dataset_raw_to_std_map,
                "std_to_video_name_map": output_video_map,
            },
            f,
            sort_keys=False,
        )
    return


def merge_episode_worker(
    dataset_dirs: list[Path],
    output_dir: Path,
    dataset_name: str,
    message_queue: mp.Queue,
    slice_config: dict[str, dict[str, slice]],
    selected_data_rename_config: dict[str, dict],
    target_action_dim: int,
    target_state_dim: int,
    bad_episodes_config: dict[str, list[str]] | None = None,
    skip_static_frames: bool = True,
):
    """Merge several datasets into one unified Zarr replay buffer.

    Args:
        dataset_dirs: Dataset roots to merge in order.
        output_dir: Destination directory for the merged replay buffer and videos.
        dataset_name: Name used in progress messages.
        message_queue: Multiprocessing queue used to emit worker status messages.
        slice_config: Per-dataset slices keyed by unified data key.
        selected_data_rename_config: Per-dataset mapping for action, state, and
            camera key normalization.
        target_action_dim: Required action feature dimension after padding.
        target_state_dim: Required state feature dimension after padding.
        bad_episodes_config: Per-dataset episode names excluded from the merge.
        skip_static_frames: Whether to remove static leading frames from each episode.

    Returns:
        None.

    Raises:
        AssertionError: If dataset specs are inconsistent or invalid.
        Exception: If replay buffer loading, padding, video processing, or output
            writing fails.
    """
    try:
        # dataset_specs: one normalized processing spec per input dataset
        dataset_specs: list[dict] = []
        # total_episodes: number of episodes across all datasets (for one global progress bar)
        total_episodes = 0
        # standard_cam_names: canonical camera key order shared by all datasets (e.g., base/wrist1/wrist2)
        standard_cam_names: list[str] | None = None

        # Build and validate per-dataset specs first.
        for dataset_dir in dataset_dirs:
            dataset_id = dataset_dir.name
            selected_data = selected_data_rename_config[dataset_id]
            # selected_action_key / selected_state_key: raw key names in replay.zarr/data
            selected_action_key: str = selected_data["action"]
            selected_state_key: str = selected_data["state"]
            # selected_cam_map: std_cam_name -> raw_cam_name
            selected_cam_map: dict[str, str] = selected_data["cameras"]
            selected_cam_keys = list(selected_cam_map.keys())
            selected_raw_cams = list(set(selected_cam_map.values()))

            episode_dirs = sorted(
                [d for d in dataset_dir.iterdir()], key=lambda x: x.name
            )

            bad_episodes = (
                None
                if bad_episodes_config is None
                else bad_episodes_config.get(dataset_id)
            )
            if bad_episodes is not None:
                assert set(bad_episodes).issubset(set(d.name for d in episode_dirs)), (
                    "bad_episodes must be a subset of episode directories"
                )
                episode_dirs = [d for d in episode_dirs if d.name not in bad_episodes]
            assert len(episode_dirs) > 0, f"No episodes found in {dataset_dir}"

            # video_name_map: raw_cam_name -> video file name (with extension)
            first_episode_videos = sorted(
                (episode_dirs[0] / "videos").iterdir(), key=lambda x: x.name
            )
            available_video_stems = set(v.stem for v in first_episode_videos)
            assert set(selected_raw_cams).issubset(available_video_stems), (
                f"Selected cameras must be a subset of available videos: {dataset_id}"
            )
            video_name_map = {
                video.stem: video.name
                for video in first_episode_videos
                if video.stem in selected_raw_cams
            }

            standard_cam_names = (
                selected_cam_keys if standard_cam_names is None else standard_cam_names
            )
            assert standard_cam_names == selected_cam_keys, (
                f"Camera key mismatch in dataset {dataset_id}: {selected_cam_keys} vs {standard_cam_names}"
            )

            dataset_specs.append(
                {
                    "dataset_id": dataset_id,
                    "episode_dirs": episode_dirs,
                    "action_key": selected_action_key,
                    "state_key": selected_state_key,
                    "cam_map": selected_cam_map,
                    "video_name_map": video_name_map,
                    "slice_cfg": slice_config.get(dataset_id, {}),
                }
            )
            total_episodes += len(episode_dirs)

        assert standard_cam_names is not None
        message_queue.put(
            InitMsg(dataset_name=dataset_name, total_episodes=total_episodes)
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        # output_video_map: std_cam_name -> output video file name
        output_video_map = {
            cam_name: f"{idx}.mp4" for idx, cam_name in enumerate(standard_cam_names)
        }
        
        _save_video_name_map(
            output_dir=output_dir,
            dataset_specs=dataset_specs,
            output_video_map=output_video_map,
        )

        zarr_path = output_dir / "replay_buffer.zarr"
        merged_buffer = ReplayBuffer.create_empty_zarr(
            storage=zarr.DirectoryStore(str(zarr_path))
        )

        total_steps = 0
        total_truncated_frames = 0
    except Exception as e:
        message_queue.put(ErrorMsg(dataset_name=dataset_name, error=str(e)))
        raise

    # global_ep_idx: continuous episode index in merged output (across all datasets)
    global_ep_idx = 0
    with ProcessPoolExecutor(max_workers=4) as executor:
        for spec in dataset_specs:
            selected_action_key = spec["action_key"]
            selected_state_key = spec["state_key"]
            selected_cam_map: dict[str, str] = spec["cam_map"]
            video_name_map: dict[str, str] = spec["video_name_map"]
            slice_cfg: dict[str, slice] = spec["slice_cfg"]

            for ep_dir in spec["episode_dirs"]:
                try:
                    zarr_file = ep_dir / "replay.zarr"
                    videos_dir = ep_dir / "videos"
                    ep_buffer = ReplayBuffer.create_from_path(str(zarr_file), mode="r")
                    episode_data = ep_buffer.get_episode(0, copy=True)
                    # Normalize array shapes before slicing/padding (keep behavior aligned with existing pipeline).
                    for key, value in episode_data.items():
                        episode_data[key] = normalize_array_shape(value)

                    if "action_rotvec" in episode_data:
                        episode_data["action"] = episode_data.pop("action_rotvec")

                    n_steps = episode_data["action"].shape[0]

                    if skip_static_frames:
                        start_idx = find_first_non_static_frame(episode_data["action"])
                        if start_idx < n_steps and np.all(
                            episode_data["action"][start_idx] == 0
                        ):
                            start_idx += 1
                        total_truncated_frames += start_idx
                        for key in episode_data:
                            episode_data[key] = episode_data[key][start_idx:]
                        n_steps -= start_idx
                    else:
                        start_idx = 0

                    for key, value in slice_cfg.items():
                        # slice_cfg keys must use unified keys {"action", "state"}.
                        episode_data[key] = episode_data[key][:, value]

                    episode_data["action"] = _pad_to_dim(
                        episode_data["action"], target_action_dim
                    )
                    episode_data["state"] = _pad_to_dim(
                        episode_data["state"], target_state_dim
                    )
                    total_steps += n_steps
                    merged_buffer.add_episode(episode_data, compressors="disk")

                    output_video_dir = output_dir / "videos" / str(global_ep_idx)
                    output_video_dir.mkdir(parents=True, exist_ok=True)
                    # video_jobs item: (src_video_file_name, dst_video_file_name)
                    video_jobs = []
                    for std_cam, raw_cam in selected_cam_map.items():
                        src_video_name = video_name_map[raw_cam]
                        dst_video_name = output_video_map[std_cam]
                        video_jobs.append((src_video_name, dst_video_name))
                    process_func = partial(
                        _process_video_task,
                        videos_dir=videos_dir,
                        output_video_dir=output_video_dir,
                        start_idx=start_idx,
                    )
                    list(executor.map(process_func, video_jobs))

                    message_queue.put(
                        ProgressMsg(
                            dataset_name=dataset_name,
                            episode_name=f"{spec['dataset_id']}/{ep_dir.name}",
                        )
                    )
                    global_ep_idx += 1
                except Exception as e:
                    message_queue.put(
                        ErrorMsg(
                            dataset_name=dataset_name,
                            error=str(e),
                            episode_name=f"{spec['dataset_id']}/{ep_dir.name}",
                        )
                    )
                    raise

    statistics_msg = (
        f"Output directory: {output_dir}\n"
        f"Total steps: {total_steps}\n"
        f"Total frames truncated: {total_truncated_frames}"
    )
    message_queue.put(DoneMsg(dataset_name=dataset_name, statistics_msg=statistics_msg))


def merge_datasets(
    output: Path,
    dataset_paths_cfg_path: Path,
    selected_data_rename_cfg_path: Path,
    slice_cfg_path: Path,
    target_action_dim: int,
    target_state_dim: int,
    bad_episodes_cfg_path: Path | None = None,
    skip_static_frames: bool = True,
) -> None:
    """Run a multi-dataset Zarr merge from YAML configuration files.

    Args:
        output: Destination directory for the merged dataset.
        dataset_paths_cfg_path: YAML file containing dataset root paths.
        selected_data_rename_cfg_path: YAML file mapping each dataset to selected
            action, state, and camera keys.
        slice_cfg_path: YAML file containing per-dataset slice specifications.
        target_action_dim: Required action feature dimension after padding.
        target_state_dim: Required state feature dimension after padding.
        bad_episodes_cfg_path: Optional YAML file containing per-dataset excluded
            episodes.
        skip_static_frames: Whether to remove static leading frames from each episode.

    Returns:
        None.

    Raises:
        RuntimeError: If the output directory is not empty, the worker reports a
            failure, or the worker exits without completion.
        AssertionError: If configured dataset paths do not exist.
        Exception: If the worker process exits with a non-zero code.
    """
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {output}")

    with open(dataset_paths_cfg_path, "r") as f:
        dataset_paths = yaml.safe_load(f)
    # dataset_dirs: absolute dataset roots to be merged in order
    dataset_dirs = [Path(p) for p in dataset_paths]
    assert all(p.exists() for p in dataset_dirs), "Some datasets path do not exist"

    with open(selected_data_rename_cfg_path, "r") as f:
        # selected_data_rename_config: dataset_id -> {"action","state","cameras"}
        selected_data_rename_config = yaml.safe_load(f)

    with open(slice_cfg_path, "r") as f:
        slice_config = yaml.safe_load(f)
    # slice_config: dataset_id -> {unified_key: slice}, unified_key in {"action", "state"}
    slice_config = {k: dict_to_slice(v) for k, v in slice_config.items()}

    if bad_episodes_cfg_path is not None:
        with open(bad_episodes_cfg_path, "r") as f:
            bad_episodes_config = yaml.safe_load(f)
    else:
        bad_episodes_config = None

    dataset_name = output.stem
    # queue: cross-process message channel for progress/errors/statistics
    queue: mp.Queue = mp.Queue()
    worker = mp.Process(
        target=merge_episode_worker,
        kwargs={
            "dataset_dirs": dataset_dirs,
            "output_dir": output,
            "dataset_name": dataset_name,
            "message_queue": queue,
            "slice_config": slice_config,
            "selected_data_rename_config": selected_data_rename_config,
            "target_action_dim": target_action_dim,
            "target_state_dim": target_state_dim,
            "bad_episodes_config": bad_episodes_config,
            "skip_static_frames": skip_static_frames,
        },
    )
    worker.start()

    pbar = ProgressBar(idx=0, dataset_name=dataset_name)
    try:
        while True:
            try:
                msg = queue.get(timeout=1.0)
            except Empty:
                if not worker.is_alive() and queue.empty():
                    break
                continue

            pbar.update(msg)
    finally:
        terminate_process(worker, process_name=dataset_name)

        queue.cancel_join_thread()
        queue.close()
        pbar.close()

    state = pbar.state
    match state.status:
        case "failed":
            raise RuntimeError(state.last_error)
        case "done":
            if state.done_msg is not None:
                print(state.done_msg)
        case "running":
            raise RuntimeError(f"{dataset_name} exited without sending done message")


if __name__ == "__main__":
    import tyro

    tyro.cli(merge_datasets)
