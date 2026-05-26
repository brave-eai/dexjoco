import json
import logging
import multiprocessing as mp
import os
import re
import sys
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from typing import Any, Literal, cast

import imageio
import imageio.v3 as iio
import numpy as np
import zarr
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from zarr import Array, Group

from ..episode_common import find_first_non_static_frame
from ..process_protocol import DoneMsg, ErrorMsg, InitMsg, ProgressBar, ProgressMsg
from ..utils import dict_to_slice, normalize_array_shape, terminate_process

IPAD_PASSWORD_FOLDERS = [f"ipad_pwd_{i}" for i in range(1, 6)]
EPISODES_PER_FOLDER = 20
TOTAL_SELECTED_EPISODES = len(IPAD_PASSWORD_FOLDERS) * EPISODES_PER_FOLDER


def _build_instruction_from_folder(folder_name: str) -> str:
    """Build the language instruction for an iPad password folder.

    Args:
        folder_name: Folder name in the ``ipad_pwd_<number>`` format.

    Returns:
        Task instruction containing the password number.

    Raises:
        AssertionError: If the folder name does not match the expected format.
    """
    match = re.fullmatch(r"ipad_pwd_(\d+)", folder_name)
    assert match is not None, f"Unexpected folder name format: {folder_name}"
    password = match.group(1)
    return f"Grasp the iPad and enter the password {password} to unlock the device."


def _collect_selected_episodes(input_root: Path) -> list[dict[str, str]]:
    """Collect the fixed iPad password episode selection plan.

    Args:
        input_root: Root directory containing ``ipad_pwd_1`` through
            ``ipad_pwd_5`` folders.

    Returns:
        Ordered episode metadata records with folder name, episode name,
        resolved episode path, and language instruction.

    Raises:
        AssertionError: If required folders are missing, a folder has too few
            episodes, or the resulting selection contains duplicates.
    """
    # Hard-coded dataset contract: only consume ipad_pwd_1..ipad_pwd_5.
    assert input_root.exists() and input_root.is_dir(), (
        f"Input root does not exist or is not a directory: {input_root}"
    )

    selected_episodes: list[dict[str, str]] = []
    for folder_name in IPAD_PASSWORD_FOLDERS:
        folder_dir = input_root / folder_name
        assert folder_dir.exists() and folder_dir.is_dir(), (
            f"Required folder is missing: {folder_dir}"
        )

        # Select first 20 episodes by lexicographic order per folder.
        episode_dirs = sorted(
            [d for d in folder_dir.iterdir() if d.is_dir()], key=lambda x: x.name
        )
        assert len(episode_dirs) >= EPISODES_PER_FOLDER, (
            f"{folder_name} has only {len(episode_dirs)} episodes, "
            f"expected >= {EPISODES_PER_FOLDER}"
        )

        instruction = _build_instruction_from_folder(folder_name)
        for ep_dir in episode_dirs[:EPISODES_PER_FOLDER]:
            selected_episodes.append(
                {
                    "folder_name": folder_name,
                    "episode_name": ep_dir.name,
                    "episode_path": str(ep_dir.resolve()),
                    "language_instruction": instruction,
                }
            )

    assert len(selected_episodes) == TOTAL_SELECTED_EPISODES, (
        f"Selected {len(selected_episodes)} episodes, expected {TOTAL_SELECTED_EPISODES}"
    )
    assert len({item["episode_path"] for item in selected_episodes}) == len(
        selected_episodes
    ), "Found duplicated episode paths in selected episodes"
    return selected_episodes


def _write_selected_episode_manifest(
    output_dir: Path, selected_episodes: list[dict[str, str]]
) -> None:
    """Write selected episode metadata to a JSON manifest.

    Args:
        output_dir: Destination LeRobot dataset root.
        selected_episodes: Episode metadata records produced by
            ``_collect_selected_episodes``.

    Returns:
        None.
    """
    # Save selection trace for reproducibility and later auditing.
    manifest_path = output_dir / "selected_episodes.json"
    with open(manifest_path, "w") as f:
        json.dump(selected_episodes, f, indent=2)


def _load_episode_arrays(
    zarr_path: Path, selected_keys: list[str]
) -> dict[str, np.ndarray]:
    """Load selected episode arrays from a replay Zarr store.

    Args:
        zarr_path: Path to an episode ``replay.zarr`` directory.
        selected_keys: Data keys to load from the Zarr ``data`` group.

    Returns:
        Mapping from selected key to a normalized ``float32`` NumPy array.
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

    features.update(
        {
            ACTION: {"dtype": "float32", "shape": action_shape},
            OBS_STATE: {"dtype": "float32", "shape": state_shape},
        }
    )

    return features


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


def merge_episode_worker(
    output_dir: Path,
    dataset_name: str,
    message_queue: mp.Queue,
    slice_cfg: dict[str, slice],
    selected_data: dict,
    selected_episodes: list[dict[str, str]],
    skip_static_frames: bool = True,
    silent: bool = True,
) -> None:
    """Run the iPad reasoning LeRobot merge with optional output suppression.

    Args:
        output_dir: Destination LeRobot dataset root.
        dataset_name: Dataset name used in progress messages.
        message_queue: Multiprocessing queue used to emit worker status messages.
        slice_cfg: Mapping from selected data key to slice applied after truncation.
        selected_data: Mapping with selected action, state, and camera keys.
        selected_episodes: Ordered episode metadata records to convert.
        skip_static_frames: Whether to remove static leading frames from each episode.
        silent: Whether to suppress stdout and stderr inside the worker.

    Returns:
        None.
    """
    output_ctx = _suppress_process_output() if silent else nullcontext()
    with output_ctx:
        _merge_episode_worker_impl(
            output_dir=output_dir,
            dataset_name=dataset_name,
            message_queue=message_queue,
            slice_cfg=slice_cfg,
            selected_data=selected_data,
            selected_episodes=selected_episodes,
            skip_static_frames=skip_static_frames,
        )


def _merge_episode_worker_impl(
    output_dir: Path,
    dataset_name: str,
    message_queue: mp.Queue,
    slice_cfg: dict[str, slice],
    selected_data: dict,
    selected_episodes: list[dict[str, str]],
    skip_static_frames: bool = True,
) -> None:
    """Merge the selected iPad password episodes into a LeRobot dataset.

    Args:
        output_dir: Destination LeRobot dataset root.
        dataset_name: Dataset name used in progress messages.
        message_queue: Multiprocessing queue used to emit worker status messages.
        slice_cfg: Mapping from selected data key to slice applied after truncation.
        selected_data: Mapping with selected action, state, and camera keys.
        selected_episodes: Ordered episode metadata records to convert.
        skip_static_frames: Whether to remove static leading frames from each episode.

    Returns:
        None.

    Raises:
        AssertionError: If camera selection or video decoding synchronization is
            invalid.
        Exception: If Zarr loading, LeRobot writing, or worker processing fails.
    """
    try:
        selected_action_key: str = selected_data["action"]
        selected_state_key: str = selected_data["state"]
        selected_cam_keys: list[str] = selected_data["cameras"]
        message_queue.put(
            InitMsg(dataset_name=dataset_name, total_episodes=len(selected_episodes))
        )

        # * Map between video key and file name
        first_episode_dir = Path(selected_episodes[0]["episode_path"])
        first_episode_videos = (first_episode_dir / "videos").iterdir()
        first_episode_videos = sorted(first_episode_videos, key=lambda x: x.name)
        assert set(selected_cam_keys).issubset(
            set(v.stem for v in first_episode_videos)
        ), "Selected cameras must be a subset of available videos in the first episode"
        video_name_map = {
            video.name: Path(video.name).stem
            for video in first_episode_videos
            if video.stem in selected_cam_keys
        }

        # * Get image size
        props = iio.improps(first_episode_videos[0], index=0)
        image_size = tuple(props.shape[:2])
        if image_size[0] != image_size[1]:
            logging.warning(
                f"Non-square frames detected in {first_episode_videos[0]}: {image_size}"
            )

        # * Get shape and buile features
        first_episode_data = _load_episode_arrays(
            first_episode_dir / "replay.zarr",
            selected_keys=[selected_action_key, selected_state_key],
        )
        for k, v in slice_cfg.items():
            first_episode_data[k] = first_episode_data[k][:, v]

        features = _build_features(
            selected_cam_keys,
            image_size=image_size,
            action_shape=tuple(first_episode_data[selected_action_key].shape[1:]),
            state_shape=tuple(first_episode_data[selected_state_key].shape[1:]),
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

        total_steps = 0
        total_truncated_frames = 0
    except Exception as e:
        message_queue.put(ErrorMsg(dataset_name=dataset_name, error=str(e)))
        raise

    video_queues = None
    video_processes = None
    for item in selected_episodes:
        ep_dir = Path(item["episode_path"])
        language_instruction = item["language_instruction"]
        progress_name = f"{item['folder_name']}/{item['episode_name']}"
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

            video_queues = {cam_name: mp.Queue() for cam_name in selected_cam_keys}
            video_processes = {
                cam_name: mp.Process(
                    target=_process_video_task,
                    args=(video_name, videos_dir, start_idx, video_queues[cam_name]),
                )
                for video_name, cam_name in video_name_map.items()
            }

            for p in video_processes.values():
                p.start()

            for t in range(n_steps):
                frame_data: dict[str, Any] = {"task": language_instruction}
                frame_data[ACTION] = episode_data[selected_action_key][t]
                frame_data[OBS_STATE] = episode_data[selected_state_key][t]
                for cam_name in selected_cam_keys:
                    try:
                        video_msg = video_queues[cam_name].get(timeout=30.0)
                    except Empty:
                        if not video_processes[cam_name].is_alive():
                            raise Exception(f"decoder process died: {cam_name}")
                        raise Exception(f"waiting frame timeout: {cam_name}, step={t}")
                    assert video_msg.type == "frame" and video_msg.step == t, (
                        "Video processing out of sync"
                    )
                    frame_data[f"{OBS_IMAGES}.{cam_name}"] = video_msg.frame
                dataset.add_frame(frame_data)
            dataset.save_episode()
            total_steps += n_steps

            for p in video_processes.values():
                p.join(timeout=10.0)
                assert not p.is_alive(), "Video processing did not finish properly"

            for cam_name in selected_cam_keys:
                video_msg = video_queues[cam_name].get(timeout=10.0)

                assert video_msg.type == "end" and video_msg.step == n_steps, (
                    "Video processing did not end properly"
                )
                video_queues[cam_name].close()

            message_queue.put(
                ProgressMsg(dataset_name=dataset_name, episode_name=progress_name)
            )
        except Exception as e:
            message_queue.put(
                ErrorMsg(
                    dataset_name=dataset_name,
                    error=str(e),
                    episode_name=progress_name,
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


def merge_episode(
    input: Path,
    output: Path,
    selected_data: dict,
    slice_spec: dict[str, list],
    skip_static_frames: bool = True,
    silent: bool = True,
) -> None:
    """Run the iPad reasoning LeRobot conversion in a worker process.

    Args:
        input: Root directory containing the iPad password folders.
        output: Destination LeRobot dataset root.
        selected_data: Mapping with selected action, state, and camera keys.
        slice_spec: Serializable slice configuration keyed by selected data key.
        skip_static_frames: Whether to remove static leading frames from each episode.
        silent: Whether to suppress stdout and stderr inside the worker.

    Returns:
        None.

    Raises:
        RuntimeError: If the output directory is not empty or the worker reports a
            failure.
        AssertionError: If the fixed episode selection contract is not satisfied.
        Exception: If the worker process exits with a non-zero code.
    """
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {output}")

    # Build fixed 5x20 selection plan before starting worker process.
    selected_episodes = _collect_selected_episodes(input)
    dataset_name = output.name

    queue: mp.Queue = mp.Queue()
    worker = mp.Process(
        target=merge_episode_worker,
        kwargs={
            "output_dir": output,
            "dataset_name": dataset_name,
            "message_queue": queue,
            "slice_cfg": dict_to_slice(slice_spec),
            "selected_data": selected_data,
            "selected_episodes": selected_episodes,
            "skip_static_frames": skip_static_frames,
            "silent": silent,
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
            _write_selected_episode_manifest(output, selected_episodes)
            if state.done_msg is not None:
                print(state.done_msg)
        case "running":
            pass
