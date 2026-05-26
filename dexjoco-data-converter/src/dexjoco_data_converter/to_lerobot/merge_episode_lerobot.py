import os
import logging
import multiprocessing as mp
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from queue import Empty
from typing import Any, cast, Literal

import imageio
import imageio.v3 as iio
import numpy as np
import zarr
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from dataclasses import dataclass
from zarr import Array, Group

from ..episode_common import find_first_non_static_frame
from ..process_protocol import DoneMsg, ErrorMsg, InitMsg, ProgressBar, ProgressMsg
from ..utils import dict_to_slice, normalize_array_shape, terminate_process


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

    features.update({
        ACTION: {"dtype": "float32", "shape": action_shape},
        OBS_STATE: {"dtype": "float32", "shape": state_shape},
    })

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
    input_dir: Path,
    output_dir: Path,
    dataset_name: str,
    message_queue: Any,
    language_instruction: str,
    slice_cfg: dict[str, slice],
    selected_data: dict,
    bad_episodes: list[str] | None = None,
    skip_static_frames: bool = True,
    silent: bool = True,
) -> None:
    """Run a single-dataset LeRobot merge with optional worker output suppression.

    Args:
        input_dir: Dataset directory containing episode subdirectories.
        output_dir: Destination LeRobot dataset root.
        dataset_name: Dataset name used in progress messages.
        message_queue: Multiprocessing queue used to emit worker status messages.
        language_instruction: Task instruction written to each output frame.
        slice_cfg: Mapping from selected data key to slice applied after truncation.
        selected_data: Mapping with selected action, state, and camera keys.
        bad_episodes: Episode directory names excluded from the merge.
        skip_static_frames: Whether to remove static leading frames from each episode.
        silent: Whether to suppress stdout and stderr inside the worker.
    """
    output_ctx = _suppress_process_output() if silent else nullcontext()
    with output_ctx:
        _merge_episode_worker_impl(
            input_dir=input_dir,
            output_dir=output_dir,
            dataset_name=dataset_name,
            message_queue=message_queue,
            language_instruction=language_instruction,
            slice_cfg=slice_cfg,
            selected_data=selected_data,
            bad_episodes=bad_episodes,
            skip_static_frames=skip_static_frames,
        )


def _merge_episode_worker_impl(
    input_dir: Path,
    output_dir: Path,
    dataset_name: str,
    message_queue: mp.Queue,
    language_instruction: str,
    slice_cfg: dict[str, slice],
    selected_data: dict,
    bad_episodes: list[str] | None = None,
    skip_static_frames: bool = True,
) -> None:
    """Merge one dataset of episode folders into a LeRobot dataset.

    Args:
        input_dir: Dataset directory containing episode subdirectories.
        output_dir: Destination LeRobot dataset root.
        dataset_name: Dataset name used in progress messages.
        message_queue: Multiprocessing queue used to emit worker status messages.
        language_instruction: Task instruction written to each output frame.
        slice_cfg: Mapping from selected data key to slice applied after truncation.
        selected_data: Mapping with selected action, state, and camera keys.
        bad_episodes: Episode directory names excluded from the merge.
        skip_static_frames: Whether to remove static leading frames from each episode.
    """
    try:
        selected_action_key: str = selected_data["action"]
        selected_state_key: str = selected_data["state"]
        selected_cam_keys: list[str] = selected_data["cameras"]

        # * Get num episodes
        episode_dirs = [d for d in input_dir.iterdir()]
        episode_dirs.sort(key=lambda x: x.name)

        if bad_episodes is not None:
            assert set(bad_episodes).issubset(set(d.name for d in episode_dirs)), (
                "bad_episodes must be a subset of episode directories"
            )
            episode_dirs = [d for d in episode_dirs if d.name not in bad_episodes]
        message_queue.put(
            InitMsg(dataset_name=dataset_name, total_episodes=len(episode_dirs))
        )

        # * Map between video key and file name
        first_episode_videos = (episode_dirs[0] / "videos").iterdir()
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
            episode_dirs[0] / "replay.zarr",
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
    for ep_dir in episode_dirs:
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
                ProgressMsg(dataset_name=dataset_name, episode_name=ep_dir.name)
            )
        except Exception as e:
            message_queue.put(
                ErrorMsg(
                    dataset_name=dataset_name,
                    error=str(e),
                    episode_name=ep_dir.name,
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
    language_instruction: str,
    selected_data: dict,
    slice_spec: dict[str, list],
    bad_episodes: list[str] | None = None,
    skip_static_frames: bool = True,
    silent: bool = True,
) -> None:
    """Run a single-dataset LeRobot merge in a worker process.

    Args:
        input: Dataset directory containing episode subdirectories.
        output: Destination LeRobot dataset root.
        language_instruction: Task instruction written to each output frame.
        selected_data: Mapping with selected action, state, and camera keys.
        slice_spec: Serializable slice configuration keyed by selected data key.
        bad_episodes: Episode directory names excluded from the merge.
        skip_static_frames: Whether to remove static leading frames from each episode.
        silent: Whether to suppress stdout and stderr inside the worker.

    Returns:
        None.

    Raises:
        RuntimeError: If the output directory is not empty or the worker reports a
            failure.
        Exception: If the worker process exits with a non-zero code.
    """
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {output}")

    dataset_name = input.name

    queue: mp.Queue = mp.Queue()
    worker = mp.Process(
        target=merge_episode_worker,
        kwargs={
            "input_dir": input,
            "output_dir": output,
            "dataset_name": dataset_name,
            "message_queue": queue,
            "language_instruction": language_instruction,
            "slice_cfg": dict_to_slice(slice_spec),
            "selected_data": selected_data,
            "bad_episodes": bad_episodes,
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
            if state.done_msg is not None:
                print(state.done_msg)
        case "running":
            pass
