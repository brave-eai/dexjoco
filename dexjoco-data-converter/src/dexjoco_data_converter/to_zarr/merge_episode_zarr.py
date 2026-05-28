import multiprocessing as mp
import shutil
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from queue import Empty
from typing import Any

import imageio
import numpy as np
import yaml
import zarr
from diffusion_policy.common.replay_buffer import ReplayBuffer  # type: ignore

from ..episode_common import find_first_non_static_frame
from ..process_protocol import DoneMsg, ErrorMsg, InitMsg, ProgressBar, ProgressMsg
from ..utils import dict_to_slice, normalize_array_shape, terminate_process


def trim_video(input_path: Path, output_path: Path, start_frame: int):
    """Trim a video so the output starts at a selected frame.

    Args:
        input_path: Input video file.
        output_path: Output video file.
        start_frame: Zero-based index of the first frame to keep.

    Returns:
        None.
    """
    reader = imageio.get_reader(input_path)
    meta = reader.get_meta_data()
    fps = meta.get("fps", 30)
    codec = meta.get("codec", "libx264")

    with imageio.get_writer(output_path, fps=fps, codec=codec) as writer:
        for i, frame in enumerate(reader.iter_data()):
            if i < start_frame:
                continue
            writer.append_data(frame)  # type: ignore
    reader.close()


def _process_video_task(
    item: tuple[str, str], videos_dir: Path, output_video_dir: Path, start_idx: int
):
    """Copy or trim one episode video into the merged output directory.

    Args:
        item: Tuple of source video file name and destination video file name.
        videos_dir: Directory containing source videos.
        output_video_dir: Directory where processed videos are written.
        start_idx: Frame index to start from. A value of ``0`` copies the file.

    Returns:
        None.
    """
    video_name, camera_name = item
    src_video = videos_dir / video_name
    dst_video = output_video_dir / camera_name
    if start_idx > 0:
        trim_video(src_video, dst_video, start_frame=start_idx)
    else:
        shutil.copy2(src_video, dst_video)


def merge_episode_worker(
    input_dir: Path,
    output_dir: Path,
    dataset_name: str,
    message_queue: Any,
    slice_cfg: dict[str, slice],
    bad_episodes: list[str] | None = None,
    skip_static_frames: bool = True,
):
    """Merge one dataset of episode folders into a single Zarr replay buffer.

    Args:
        input_dir: Dataset directory containing episode subdirectories.
        output_dir: Directory where the merged replay buffer and videos are saved.
        dataset_name: Dataset name used in progress messages.
        message_queue: Multiprocessing queue used to emit worker status messages.
        slice_cfg: Mapping from episode data key to slice applied after truncation.
        bad_episodes: Episode directory names excluded from the merge.
        skip_static_frames: Whether to remove static leading frames from each episode.
    """
    try:
        # Find all episode directories
        episode_dirs = list(filter(lambda d: d.is_dir(), input_dir.iterdir()))
        episode_dirs.sort(key=lambda x: x.name)

        if bad_episodes is not None:
            assert set(bad_episodes).issubset(set(d.name for d in episode_dirs)), (
                "bad_episodes must be a subset of episode directories"
            )
            episode_dirs = [d for d in episode_dirs if d.name not in bad_episodes]

        # create progress bar
        message_queue.put(
            InitMsg(dataset_name=dataset_name, total_episodes=len(episode_dirs))
        )

        # make video_name_map from first episode
        first_episode_videos = (episode_dirs[0] / "videos").iterdir()
        first_episode_videos = sorted(first_episode_videos, key=lambda x: x.name)
        # map video name (xxx.mp4) to idx (0.mp4)
        video_name_map = {
            video.name: f"{idx}.mp4" for idx, video in enumerate(first_episode_videos)
        }

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # save video_name_map to output_dir for reference
        with open(output_dir / "video_name_map.yaml", "w") as f:
            yaml.safe_dump(video_name_map, f, sort_keys=False)

        # Create merged replay buffer
        zarr_path = output_dir / "replay_buffer.zarr"  # hardcoded name
        merged_buffer = ReplayBuffer.create_empty_zarr(
            storage=zarr.DirectoryStore(str(zarr_path))
        )

        # Track statistics
        total_steps = 0  # counter
        total_truncated_frames = 0  # counter
    except Exception as e:
        message_queue.put(ErrorMsg(dataset_name=dataset_name, error=str(e)))
        raise

    with ProcessPoolExecutor(max_workers=4) as executor:
        for ep_idx, ep_dir in enumerate(episode_dirs):
            try:
                # get data from raw dataset
                zarr_file = Path(ep_dir) / "replay.zarr"
                videos_dir = Path(ep_dir) / "videos"
                ep_buffer = ReplayBuffer.create_from_path(str(zarr_file), mode="r")
                episode_data = ep_buffer.get_episode(0, copy=True)
                for k, v in episode_data.items():
                    episode_data[k] = normalize_array_shape(v)
                if "action_rotvec" in episode_data:
                    # we use rotvec to train
                    episode_data["action"] = episode_data.pop("action_rotvec")
                n_steps: int = ep_buffer.n_steps

                if skip_static_frames:
                    start_idx = find_first_non_static_frame(episode_data["action"])
                    if start_idx < n_steps and np.all(
                        episode_data["action"][start_idx] == 0
                    ):
                        start_idx += 1
                    total_truncated_frames += start_idx
                    for key in episode_data:
                        episode_data[key] = episode_data[key][start_idx:]
                    n_steps = n_steps - start_idx
                else:
                    start_idx = 0

                for k, v in slice_cfg.items():
                    episode_data[k] = episode_data[k][:, v]

                total_steps += n_steps
                merged_buffer.add_episode(episode_data, compressors="disk")
                output_video_dir = output_dir / "videos" / str(ep_idx)
                output_video_dir.mkdir(parents=True, exist_ok=True)
                process_func = partial(
                    _process_video_task,
                    videos_dir=videos_dir,
                    output_video_dir=output_video_dir,
                    start_idx=start_idx,
                )
                list(executor.map(process_func, video_name_map.items()))
                # update progress bar
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
    slice_yaml: str,
    bad_episodes_yaml: str | None = None,
    skip_static_frames: bool = True,
) -> None:
    """Run a single-dataset Zarr merge in a worker process.

    Args:
        input: Dataset directory containing episode subdirectories.
        output: Destination directory for the merged Zarr dataset.
        slice_yaml: YAML string containing slice configuration keyed by data field.
        bad_episodes_yaml: YAML string containing episode directory names excluded
            from the merge.
        skip_static_frames: Whether to remove static leading frames from each episode.
    """
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {output}")

    slice_spec = yaml.safe_load(slice_yaml)
    bad_episodes = (
        None if bad_episodes_yaml is None else yaml.safe_load(bad_episodes_yaml)
    )
    assert isinstance(slice_spec, dict), "slice_yaml must be a YAML dict string"
    assert bad_episodes is None or isinstance(bad_episodes, list), (
        "bad_episodes_yaml must be a YAML list string"
    )

    dataset_name = input.name

    que = mp.Queue()
    worker = mp.Process(
        target=merge_episode_worker,
        kwargs={
            "input_dir": input,
            "output_dir": output,
            "dataset_name": dataset_name,
            "message_queue": que,
            "slice_cfg": dict_to_slice(slice_spec),
            "bad_episodes": bad_episodes,
            "skip_static_frames": skip_static_frames,
        },
    )
    worker.start()

    pbar = ProgressBar(idx=0, dataset_name=dataset_name)
    try:
        while True:
            try:
                msg = que.get(timeout=1.0)
            except Empty:
                if not worker.is_alive() and que.empty():
                    break
                continue

            pbar.update(msg)
    finally:
        terminate_process(worker, process_name=dataset_name)
        que.cancel_join_thread()
        que.close()
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
