import logging
import multiprocessing as mp
import os
import sys
from collections import OrderedDict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from typing import Any, Literal, cast

import imageio
import imageio.v3 as iio
import numpy as np
import yaml
import zarr
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from zarr import Array, Group

from ..episode_common import find_first_non_static_frame
from ..process_protocol import DoneMsg, ErrorMsg, InitMsg, ProgressBar, ProgressMsg
from ..utils import dict_to_slice, normalize_array_shape, pad_to_dim, terminate_process


def _load_episode_arrays(
    zarr_path: Path, selected_keys: list[str]
) -> dict[str, np.ndarray]:
    """Load selected episode arrays from a replay Zarr store.

    Args:
        zarr_path: Path to an episode `replay.zarr` directory.
        selected_keys: Data keys to load from the Zarr `data` group.

    Returns:
        Mapping from selected key to a normalized `float32` NumPy array.
    """
    # Load all arrays from replay.zarr/data group as numpy arrays.
    root = cast(Group, zarr.open(str(zarr_path), mode="r"))
    data_group = cast(Group, root["data"])

    episode_data: dict[str, np.ndarray] = {}
    for key in selected_keys:
        raw = np.asarray(cast(Array, data_group[key])[:], dtype=np.float32)
        episode_data[key] = normalize_array_shape(raw)
    return episode_data


@dataclass
class ProcessVideoMessage:
    type: Literal["frame", "end"]
    step: int
    frame: np.ndarray | None


@dataclass(frozen=True)
class DatasetSpec:
    """Processing configuration for one input dataset."""

    dataset_id: str
    episode_dirs: list[Path]
    instruction: str
    action_key: str
    state_key: str
    cam_map: OrderedDict[str, str]  # std cam -> raw cam
    slice_cfg: dict[str, slice]


def _process_video_task(
    video_name: str,  # video_name(include ".mp4")
    videos_dir: Path,
    start_idx: int,
    que: mp.Queue,
):
    """Decode one video and stream frames through a multiprocessing queue.

    Args:
        video_name: Source video file name, including extension.
        videos_dir: Directory containing source videos.
        start_idx: Number of leading frames to skip.
        que: Queue receiving ``ProcessVideoMessage`` frame and end messages.

    Returns:
        None.
    """
    video_path = videos_dir / video_name

    reader = imageio.get_reader(video_path)
    try:
        for i, frame in enumerate(reader.iter_data()):
            if i < start_idx:
                continue
            que.put(ProcessVideoMessage(type="frame", step=i - start_idx, frame=frame))
        que.put(ProcessVideoMessage(type="end", step=i + 1 - start_idx, frame=None))
    finally:
        reader.close()
    return


def _build_features(
    camera_names: list[str],
    image_size: tuple,
    action_shape: tuple,
    state_shape: tuple,
) -> dict[str, Any]:
    """Build a LeRobot feature schema for images, action, and state.

    Args:
        camera_names: Camera names exposed as observation image keys.
        image_size: Frame size as ``(height, width)``.
        action_shape: Per-frame action tensor shape.
        state_shape: Per-frame state tensor shape.

    Returns:
        Feature specification accepted by ``LeRobotDataset.create``.
    """
    features: dict[str, Any] = {}

    for cam_name in camera_names:
        features[f"{OBS_IMAGES}.{cam_name}"] = {
            "dtype": "video",
            "shape": (image_size[0], image_size[1], 3),
            "names": ["height", "width", "channel"],
        }

    features.update({
        ACTION: {"dtype": "float32", "shape": action_shape},
        OBS_STATE: {"dtype": "float32", "shape": state_shape},
    })

    return features


def _get_episode_dirs(
    dataset_dir: Path, bad_episodes: list[str] | None = None
) -> list[Path]:
    """Return sorted episode directories after excluding configured episodes."""
    episode_dirs = list(filter(lambda d: d.is_dir(), dataset_dir.iterdir()))
    episode_dirs.sort(key=lambda x: x.name)

    if bad_episodes is not None:
        assert set(bad_episodes).issubset(set(d.name for d in episode_dirs)), (
            "bad_episodes must be a subset of episode directories"
        )
        episode_dirs = [d for d in episode_dirs if d.name not in bad_episodes]

    assert len(episode_dirs) > 0, f"No episodes found in {dataset_dir}"
    return episode_dirs


def _build_dataset_specs(
    dataset_dirs: list[Path],
    language_instruction_config: dict[str, str],
    selected_data_config: dict[str, dict],
    slice_config: dict[str, dict[str, slice]],
    bad_episodes_config: dict[str, list[str]] | None = None,
) -> tuple[list[DatasetSpec], list[str], tuple, int]:
    """Build per-dataset processing specs and validate shared LeRobot schema."""
    dataset_specs: list[DatasetSpec] = []
    total_episodes = 0
    cam_key_lists = [
        list(selected_data["cameras"].keys())
        for selected_data in selected_data_config.values()
    ]
    assert len(cam_key_lists) > 0, "No camera config found"
    assert all(cam_keys == cam_key_lists[0] for cam_keys in cam_key_lists), (
        f"Camera key mismatch: {cam_key_lists}"
    )
    standard_cam_names = cam_key_lists[0]

    # confirm constant image size
    image_sizes = []
    dataset_episode_dirs: dict[str, list[Path]] = {}
    for dataset_dir in dataset_dirs:
        dataset_id = dataset_dir.name
        bad_episodes = (
            None if bad_episodes_config is None else bad_episodes_config[dataset_id]
        )
        episode_dirs = _get_episode_dirs(dataset_dir, bad_episodes=bad_episodes)
        dataset_episode_dirs[dataset_id] = episode_dirs

        first_episode_videos = sorted(
            (episode_dirs[0] / "videos").iterdir(), key=lambda x: x.name
        )
        for video in first_episode_videos:
            props = iio.improps(video, index=0)
            current_image_size = tuple(props.shape[:2])
            if current_image_size[0] != current_image_size[1]:
                logging.warning(
                    f"Non-square frames detected in {video}: {current_image_size}"
                )
            image_sizes.append(current_image_size)

    assert len(image_sizes) > 0, "No video found"
    assert all(size == image_sizes[0] for size in image_sizes), (
        f"Image size mismatch: {image_sizes}"
    )
    image_size = image_sizes[0]

    for dataset_dir in dataset_dirs:
        dataset_id = dataset_dir.name
        selected_data = selected_data_config[dataset_id]
        selected_action_key: str = selected_data["action"]
        selected_state_key: str = selected_data["state"]
        selected_cam_map: OrderedDict[str, str] = OrderedDict(selected_data["cameras"])
        selected_raw_cams = list(set(selected_cam_map.values()))

        episode_dirs = dataset_episode_dirs[dataset_id]

        first_episode_videos = sorted(
            (episode_dirs[0] / "videos").iterdir(), key=lambda x: x.name
        )
        available_video_stems = set(v.stem for v in first_episode_videos)
        assert set(selected_raw_cams).issubset(available_video_stems), (
            f"Selected cameras must be a subset of available videos: {dataset_id}"
        )

        dataset_specs.append(
            DatasetSpec(
                dataset_id=dataset_id,
                episode_dirs=episode_dirs,
                instruction=language_instruction_config[dataset_id],
                action_key=selected_action_key,
                state_key=selected_state_key,
                cam_map=selected_cam_map,
                slice_cfg=slice_config[dataset_id],
            )
        )
        total_episodes += len(episode_dirs)

    return dataset_specs, standard_cam_names, image_size, total_episodes


@contextmanager
def _suppress_process_output():
    """Temporarily redirect process stdout and stderr to ``os.devnull``.

    Args:
        None.

    Yields:
        None while stdout and stderr are suppressed.
    """
    stdout = sys.stdout
    stderr = sys.stderr
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)
    devnull = open(os.devnull, "w")
    try:
        sys.stdout = devnull
        sys.stderr = devnull
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
        yield
    finally:
        os.dup2(saved_stdout_fd, 1)
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
        sys.stdout = stdout
        sys.stderr = stderr
        devnull.close()


def merge_datasets_worker(
    dataset_dirs: list[Path],
    output_dir: Path,
    dataset_name: str,
    message_queue: mp.Queue,
    language_instruction_config: dict[str, str],
    slice_config: dict[str, dict[str, slice]],
    selected_data_config: dict[str, dict],
    target_action_dim: int,
    target_state_dim: int,
    bad_episodes_config: dict[str, list[str]] | None = None,
    skip_static_frames: bool = True,
    silent: bool = True,
) -> None:
    """Run a multi-dataset LeRobot merge with optional output suppression.

    Args:
        dataset_dirs: Dataset roots to merge in order.
        output_dir: Destination LeRobot dataset root.
        dataset_name: Name used in progress messages.
        message_queue: Multiprocessing queue used to emit worker status messages.
        language_instruction_config: Per-dataset task instructions.
        slice_config: Per-dataset slices keyed by selected data key.
        selected_data_config: Per-dataset mapping for action, state, and cameras.
        target_action_dim: Required action feature dimension after padding.
        target_state_dim: Required state feature dimension after padding.
        bad_episodes_config: Per-dataset episode names excluded from the merge.
        skip_static_frames: Whether to remove static leading frames from each episode.
        silent: Whether to suppress stdout and stderr inside the worker.

    Returns:
        None.
    """
    output_ctx = _suppress_process_output() if silent else nullcontext()
    with output_ctx:
        _merge_datasets_worker_impl(
            dataset_dirs=dataset_dirs,
            output_dir=output_dir,
            dataset_name=dataset_name,
            message_queue=message_queue,
            language_instruction_config=language_instruction_config,
            slice_config=slice_config,
            selected_data_config=selected_data_config,
            target_action_dim=target_action_dim,
            target_state_dim=target_state_dim,
            bad_episodes_config=bad_episodes_config,
            skip_static_frames=skip_static_frames,
        )


def _merge_datasets_worker_impl(
    dataset_dirs: list[Path],
    output_dir: Path,
    dataset_name: str,
    message_queue: mp.Queue,
    language_instruction_config: dict[str, str],
    slice_config: dict[str, dict[str, slice]],
    selected_data_config: dict[str, dict],
    target_action_dim: int,
    target_state_dim: int,
    bad_episodes_config: dict[str, list[str]] | None = None,
    skip_static_frames: bool = True,
) -> None:
    """Merge several datasets into one unified LeRobot dataset.

    Args:
        dataset_dirs: Dataset roots to merge in order.
        output_dir: Destination LeRobot dataset root.
        dataset_name: Name used in progress messages.
        message_queue: Multiprocessing queue used to emit worker status messages.
        language_instruction_config: Per-dataset task instructions.
        slice_config: Per-dataset slices keyed by selected data key.
        selected_data_config: Per-dataset mapping for action, state, and cameras.
        target_action_dim: Required action feature dimension after padding.
        target_state_dim: Required state feature dimension after padding.
        bad_episodes_config: Per-dataset episode names excluded from the merge.
        skip_static_frames: Whether to remove static leading frames from each episode.
    """
    try:
        dataset_specs, standard_cam_names, image_size, total_episodes = (
            _build_dataset_specs(
                dataset_dirs=dataset_dirs,
                language_instruction_config=language_instruction_config,
                selected_data_config=selected_data_config,
                slice_config=slice_config,
                bad_episodes_config=bad_episodes_config,
            )
        )

        message_queue.put(
            InitMsg(dataset_name=dataset_name, total_episodes=total_episodes)
        )
        features = _build_features(
            standard_cam_names,
            image_size=image_size,
            action_shape=(target_action_dim,),
            state_shape=(target_state_dim,),
        )

        # lerobot will automatically create data folder
        dataset = LeRobotDataset.create(
            repo_id="local_repo",
            fps=30,
            features=features,
            root=output_dir,
            image_writer_threads=4,
            streaming_encoding=True,
            # ! 0 represents infinite queue size
            # ! lerobot streaming video encoder drops frames when the queue is full
            encoder_queue_maxsize=0,
        )
    except Exception as e:
        message_queue.put(ErrorMsg(dataset_name=dataset_name, error=str(e)))
        raise

    video_queues = None
    video_processes = None
    total_steps = 0
    total_truncated_frames = 0
    for spec in dataset_specs:
        selected_action_key = spec.action_key
        selected_state_key = spec.state_key
        selected_cam_map = spec.cam_map
        # 1 raw cam may be mapped to multiple std cams
        selected_raw_cams = set(selected_cam_map.values())
        slice_cfg = spec.slice_cfg

        for ep_dir in spec.episode_dirs:
            try:
                zarr_file = ep_dir / "replay.zarr"
                videos_dir = ep_dir / "videos"

                episode_data = _load_episode_arrays(
                    zarr_file,
                    selected_keys=[selected_action_key, selected_state_key],
                )
                n_steps = episode_data[selected_action_key].shape[0]
                if skip_static_frames:
                    start_idx = find_first_non_static_frame(
                        episode_data[selected_action_key]
                    )
                    if start_idx < n_steps and np.all(
                        episode_data[selected_action_key][start_idx] == 0
                    ):
                        start_idx += 1
                    total_truncated_frames += start_idx
                    for key in episode_data:
                        episode_data[key] = episode_data[key][start_idx:]
                    n_steps -= start_idx
                else:
                    start_idx = 0

                for k, v in slice_cfg.items():
                    episode_data[k] = episode_data[k][:, v]
                # Enforce a fixed action/state schema across mixed datasets by right-padding with zeros.
                episode_data[selected_action_key] = pad_to_dim(
                    episode_data[selected_action_key], target_action_dim
                )
                episode_data[selected_state_key] = pad_to_dim(
                    episode_data[selected_state_key], target_state_dim
                )

                video_queues = {raw_cam: mp.Queue() for raw_cam in selected_raw_cams}
                video_processes = {
                    raw_cam: mp.Process(
                        target=_process_video_task,
                        args=(
                            f"{raw_cam}.mp4",
                            videos_dir,
                            start_idx,
                            video_queues[raw_cam],
                        ),
                    )
                    for raw_cam in selected_raw_cams
                }
                for p in video_processes.values():
                    p.start()

                for t in range(n_steps):
                    frame_data: dict[str, Any] = {"task": spec.instruction}
                    frame_data[ACTION] = episode_data[selected_action_key][t]
                    frame_data[OBS_STATE] = episode_data[selected_state_key][t]
                    raw_video_frame: dict = {}
                    for raw_cam in selected_raw_cams:
                        try:
                            video_msg: ProcessVideoMessage = video_queues[raw_cam].get(
                                timeout=30.0
                            )
                        except Empty:
                            if not video_processes[raw_cam].is_alive():
                                raise Exception(f"decoder process died: {raw_cam}")
                            raise Exception(
                                f"waiting frame timeout: {raw_cam}, step={t}"
                            )
                        assert video_msg.type == "frame" and video_msg.step == t, (
                            "Video processing out of sync"
                        )
                        raw_video_frame[raw_cam] = video_msg.frame
                    for std_cam, raw_cam in selected_cam_map.items():
                        frame_data[f"{OBS_IMAGES}.{std_cam}"] = raw_video_frame[raw_cam]
                    dataset.add_frame(frame_data)
                dataset.save_episode()
                total_steps += n_steps

                for p in video_processes.values():
                    p.join(timeout=10.0)
                    assert not p.is_alive(), "Video processing did not finish properly"
                for raw_cam in selected_raw_cams:
                    video_msg = video_queues[raw_cam].get(timeout=10.0)
                    assert video_msg.type == "end" and video_msg.step == n_steps, (
                        "Video processing did not end properly"
                    )
                    video_queues[raw_cam].close()

                message_queue.put(
                    ProgressMsg(
                        dataset_name=dataset_name,
                        episode_name=f"{spec.dataset_id}/{ep_dir.name}",
                    )
                )
            except Exception as e:
                message_queue.put(
                    ErrorMsg(
                        dataset_name=dataset_name,
                        error=str(e),
                        episode_name=f"{spec.dataset_id}/{ep_dir.name}",
                    )
                )
                if video_processes is not None:
                    for p in video_processes.values():
                        if p.is_alive():
                            p.terminate()
                        p.join(timeout=1.0)
                if video_queues is not None:
                    for q in video_queues.values():
                        q.cancel_join_thread()
                        q.close()
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
    language_instruction_cfg_path: Path,
    selected_data_cfg_path: Path,
    slice_cfg_path: Path,
    target_action_dim: int,
    target_state_dim: int,
    bad_episodes_cfg_path: Path | None = None,
    skip_static_frames: bool = True,
    silent: bool = True,
) -> None:
    """Run a multi-dataset LeRobot merge from YAML configuration files.

    Args:
        output: Destination LeRobot dataset root.
        dataset_paths_cfg_path: YAML file containing dataset root paths.
        language_instruction_cfg_path: YAML file containing per-dataset task
            instructions.
        selected_data_cfg_path: YAML file mapping each dataset to selected action,
            state, and camera keys.
        slice_cfg_path: YAML file containing per-dataset slice specifications.
        target_action_dim: Required action feature dimension after padding.
        target_state_dim: Required state feature dimension after padding.
        bad_episodes_cfg_path: Optional YAML file containing per-dataset excluded
            episodes.
        skip_static_frames: Whether to remove static leading frames from each episode.
        silent: Whether to suppress stdout and stderr inside the worker.

    Returns:
        None.

    Raises:
        RuntimeError: If the output directory is not empty or the worker reports a
            failure.
        AssertionError: If configured dataset paths do not exist.
        Exception: If the worker process exits with a non-zero code.
    """
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {output}")

    # Load all merge controls from config files so one worker can process all datasets sequentially.
    with open(dataset_paths_cfg_path, "r") as f:
        dataset_paths = yaml.safe_load(f)
    dataset_dirs = [Path(p) for p in dataset_paths]
    assert all(p.exists() for p in dataset_dirs), "Some datasets path do not exist"
    with open(language_instruction_cfg_path, "r") as f:
        language_instruction_config = yaml.safe_load(f)
    with open(selected_data_cfg_path, "r") as f:
        selected_data_config = yaml.safe_load(f)
    with open(slice_cfg_path, "r") as f:
        raw_slice_config = yaml.safe_load(f)
    slice_config = {k: dict_to_slice(v) for k, v in raw_slice_config.items()}
    if bad_episodes_cfg_path is not None:
        with open(bad_episodes_cfg_path, "r") as f:
            bad_episodes_config = yaml.safe_load(f)
    else:
        bad_episodes_config = None

    dataset_name = output.stem
    msg_que: mp.Queue = mp.Queue()
    worker = mp.Process(
        target=merge_datasets_worker,
        kwargs={
            "dataset_dirs": dataset_dirs,
            "output_dir": output,
            "dataset_name": dataset_name,
            "message_queue": msg_que,
            "language_instruction_config": language_instruction_config,
            "slice_config": slice_config,
            "selected_data_config": selected_data_config,
            "target_action_dim": target_action_dim,
            "target_state_dim": target_state_dim,
            "bad_episodes_config": bad_episodes_config,
            "skip_static_frames": skip_static_frames,
            "silent": silent,
        },
    )
    worker.start()

    pbar = ProgressBar(idx=0, dataset_name=dataset_name)
    try:
        while True:
            try:
                msg = msg_que.get(timeout=1.0)
            except Empty:
                if not worker.is_alive() and msg_que.empty():
                    break
                continue

            pbar.update(msg)
    finally:
        terminate_process(worker, process_name=dataset_name)
        msg_que.cancel_join_thread()
        msg_que.close()
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
