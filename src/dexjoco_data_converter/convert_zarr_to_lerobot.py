"""Convert raw water_plant demonstrations to LeRobot v3 with replay metadata.

Run from the repository root:
    uv run python src/dexjoco_data_converter/convert_zarr_to_lerobot.py \
        --input-dir /path/to/raw/water_plant --output-dir /path/to/output \
        --repo-id local/water_plant --max-episodes 2

Requires lerobot[dataset] 0.6.1 and zarr 2.x. No Diffusion Policy dependency.
"""

import json
import multiprocessing as mp
import queue
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import Any

import av
import numpy as np
import tyro
import zarr
from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from numpy.typing import NDArray
from tqdm import tqdm

CAMERAS = ("front", "wrist")


# Initialized once in each persistent decoder process.
_VIDEO_QUEUES: tuple[Any, ...] = ()
_STOP: Any = None


def initialize_decoder(queues: tuple[Any, ...], stop: Any) -> None:
    global _VIDEO_QUEUES, _STOP
    _VIDEO_QUEUES = queues
    _STOP = stop


def decode_video(path: Path, camera_index: int) -> None:
    """One video per task; workers and bounded frame queues survive episodes."""
    frames = _VIDEO_QUEUES[camera_index]

    def put(item: tuple[int, NDArray[Any]] | str | None) -> None:
        while not _STOP.is_set():
            try:
                frames.put(item, timeout=0.1)
                return
            except queue.Full:
                continue

    try:
        with av.open(str(path)) as container:
            for index, frame in enumerate(container.decode(video=0)):
                if _STOP.is_set():
                    return
                put((index, frame.to_ndarray(format="rgb24")))
        put(None)
    except Exception as exc:  # noqa: BLE001 -- Forward worker failures to the consumer.
        put(f"{type(exc).__name__}: {exc}")


class VideoDecoder:
    """Consumer for a video task running in the persistent process pool."""

    def __init__(self, path: Path, frames: Any, future: Future[None]):
        self.path = path
        self.frames = frames
        self.future = future

    def next_frame(self, expected_index: int) -> NDArray[Any]:
        item = self.frames.get(timeout=120)
        if isinstance(item, str):
            raise RuntimeError(f"Failed to decode {self.path}: {item}")  # noqa: TRY004 -- Worker error message.
        if item is None:
            raise ValueError(f"Video ended early: {self.path}, frame {expected_index}")
        index, frame = item
        if index != expected_index:
            raise ValueError(f"Video out of sync: {self.path}, {index} != {expected_index}")
        return frame

    def check_end(self) -> None:
        item = self.frames.get(timeout=120)
        if isinstance(item, str):
            raise RuntimeError(f"Failed to decode {self.path}: {item}")  # noqa: TRY004 -- Worker error message.
        if item is not None:
            raise ValueError(f"Video has more frames than actions: {self.path}")
        self.future.result(timeout=120)


def load_demo(demo_dir: Path) -> dict[str, NDArray[Any]]:
    """Read Zarr directly and keep all frames, including leading static frames."""
    root = zarr.open_group(str(demo_dir / "replay.zarr"), mode="r")
    arrays = {
        key: np.asarray(root["data"][key][:])
        for key in ("action", "action_rotvec", "state", "timestamp")
    }
    length = len(arrays["action"])
    if length == 0:
        raise ValueError(f"Empty episode: {demo_dir}")
    for key, array in arrays.items():
        if len(array) != length:
            raise ValueError(f"Mismatched length for {key}: {demo_dir}")
        arrays[key] = array.reshape(length, -1)
    for key, dim in (("action", 23), ("action_rotvec", 22), ("state", 38), ("timestamp", 1)):
        if arrays[key].shape[1] != dim:
            raise ValueError(f"Expected {key} shape ({length}, {dim}): {demo_dir}")
        if not np.isfinite(arrays[key]).all():
            raise ValueError(f"Non-finite values in {key}: {demo_dir}")
    scene = arrays["state"][:, 23:]
    if not np.array_equal(scene, np.broadcast_to(scene[0], scene.shape)):
        raise ValueError(f"Initial scene fields change within episode: {demo_dir}")
    return arrays


def build_features(demo_dir: Path) -> dict[str, Any]:
    features: dict[str, Any] = {
        "action": {"dtype": "float32", "shape": (22,)},
        "action_quat": {"dtype": "float32", "shape": (23,)},
        "observation.state": {"dtype": "float32", "shape": (23,)},
        # LeRobot generates its own timestamp; retain the source separately.
        "source_timestamp": {"dtype": "float32", "shape": (1,)},
    }
    for camera in CAMERAS:
        path = demo_dir / "videos" / f"{camera}.mp4"
        with av.open(str(path)) as container:
            frame = next(container.decode(video=0))
            features[f"observation.images.{camera}"] = {
                "dtype": "video",
                "shape": (frame.height, frame.width, 3),
                "names": ["height", "width", "channel"],
            }
    return features


def convert_episode(
    dataset: LeRobotDataset,
    demo_dir: Path,
    task_description: str,
    executor: ProcessPoolExecutor,
    video_queues: tuple[Any, ...],
) -> dict[str, Any]:
    arrays = load_demo(demo_dir)
    episode_index = dataset.meta.total_episodes
    state0 = arrays["state"][0]
    record = {
        "episode_index": episode_index,
        "source_demo": demo_dir.name,
        "task": "water_plant",
        "length": len(arrays["action"]),
        "initial_state": {
            "orig_spray_pose": state0[23:30].tolist(),
            "orig_plant_pose": state0[30:37].tolist(),
            "table_delta_height": float(state0[37]),
        },
        "action_quaternion_order": "wxyz",
        "zero_pose_behavior": "hold_previous_target",
        "observation_timing": "before_action",
    }
    decoders = {}
    for index, camera in enumerate(CAMERAS):
        path = demo_dir / "videos" / f"{camera}.mp4"
        future = executor.submit(decode_video, path, index)
        decoders[camera] = VideoDecoder(path, video_queues[index], future)
    for t in tqdm(range(record["length"]), desc=demo_dir.name):
        frame_data = {
            "action": arrays["action_rotvec"][t].astype(np.float32),
            "action_quat": arrays["action"][t].astype(np.float32),
            "observation.state": arrays["state"][t, :23].astype(np.float32),
            "source_timestamp": arrays["timestamp"][t].astype(np.float32),
            "task": task_description,
        }
        for camera, decoder in decoders.items():
            frame_data[f"observation.images.{camera}"] = decoder.next_frame(t)
        dataset.add_frame(frame_data)
    for decoder in decoders.values():
        decoder.check_end()
    dataset.save_episode()
    return record


def verify_video_lengths(output_dir: Path, expected_frames: int) -> None:
    """Count decoded output frames across all shards to catch missing frames."""
    for camera in CAMERAS:
        count = 0
        for path in sorted((output_dir / "videos" / f"observation.images.{camera}").rglob("*.mp4")):
            with av.open(str(path)) as container:
                count += sum(1 for _ in container.decode(video=0))
        if count != expected_frames:
            raise ValueError(f"Output {camera} has {count} frames; expected {expected_frames}")


def main(
    input_dir: Path,
    output_dir: Path,
    repo_id: str = "local/water_plant",
    max_episodes: int | None = None,
    task_description: str = "Pick up the spray bottle and water the plant.",
    decoder_queue_size: int = 4,
    vcodec: str = "libsvtav1",
    encoder_threads: int = 2,
) -> None:
    """Convert water_plant only; refuse to overwrite an existing dataset."""
    if decoder_queue_size <= 0:
        raise ValueError("decoder_queue_size must be positive")
    if max_episodes is not None and max_episodes <= 0:
        raise ValueError("max_episodes must be positive")
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}")
    demos = sorted(path.parent for path in input_dir.glob("*/replay.zarr"))
    if max_episodes is not None:
        demos = demos[:max_episodes]
    if not demos:
        raise ValueError(f"No replay.zarr episodes found in {input_dir}")
    load_demo(demos[0])
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=output_dir,
        fps=30,
        features=build_features(demos[0]),
        streaming_encoding=True,
        # Python Queue(maxsize=0) is unbounded: no capacity-induced frame drops.
        # Encoding can accumulate frames in RAM if decoding is faster.
        encoder_queue_maxsize=0,
        encoder_threads=encoder_threads,
        rgb_encoder=RGBEncoderConfig(vcodec=vcodec),
    )
    total_frames = 0
    with (output_dir / "meta" / "replay.jsonl").open("w") as replay_file:
        context = mp.get_context("spawn")
        video_queues = tuple(context.Queue(maxsize=decoder_queue_size) for _ in CAMERAS)
        stop = context.Event()
        try:
            with ProcessPoolExecutor(
                max_workers=len(CAMERAS),
                mp_context=context,
                initializer=initialize_decoder,
                initargs=(video_queues, stop),
            ) as executor:
                try:
                    for demo in demos:
                        record = convert_episode(
                            dataset, demo, task_description, executor, video_queues
                        )
                        replay_file.write(json.dumps(record) + "\n")
                        replay_file.flush()
                        total_frames += record["length"]
                finally:
                    # Unblock producers before the executor waits for workers.
                    stop.set()
        finally:
            for frames in video_queues:
                frames.cancel_join_thread()
                frames.close()
            dataset.finalize()
    verify_video_lengths(output_dir, total_frames)
    print(f"Saved {len(demos)} episodes, {total_frames} frames to {output_dir}")


if __name__ == "__main__":
    tyro.cli(main)
