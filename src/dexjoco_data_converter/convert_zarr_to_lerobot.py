import multiprocessing as mp
import queue
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing.synchronize import Event as EventClass
from pathlib import Path
from typing import Any

import av
import numpy as np
from lerobot.datasets import LeRobotDataset
from numpy.typing import NDArray
from tqdm import tqdm

from .video_loader import decode_video
from .zarr_loader import load_zarr_episode


@dataclass(frozen=True)
class DatasetConfig:
    zarr_dataset_dir: Path
    output_dir: Path
    is_bimanual: bool
    task_description: str

    # Ordered like proprio_keys after tcp_pose and gripper_pose.
    # Each pair contains the field name and its size in the state tail.
    replay_initial_position_fields: tuple[tuple[str, int], ...]

    def split_replay_initial_positions(
        self, positions: NDArray[np.float64]
    ) -> dict[str, list[float]]:
        """Split one episode's initial state into JSON-serializable arrays.

        Offsets start at zero in the tail returned by load_zarr_episode.
        Keep table_delta_height as a one-element list, matching the legacy
        state restorers.
        """
        expected_dim = sum(size for _, size in self.replay_initial_position_fields)
        if positions.shape != (expected_dim,):
            raise ValueError(
                f"Expected replay initial positions shape {(expected_dim,)}, "
                f"got {positions.shape}"
            )

        fields: dict[str, list[float]] = {}
        offset = 0
        for name, size in self.replay_initial_position_fields:
            fields[name] = positions[offset : offset + size].tolist()
            offset += size
        return fields

    @property
    def action_dim(self) -> int:
        return 44 if self.is_bimanual else 22

    @property
    def state_dim(self) -> int:
        return 46 if self.is_bimanual else 23


def build_features(
    cfg: DatasetConfig,
    episode_dir: Path,
    camera_names: list[str],
) -> dict[str, Any]:
    """Build action, robot state, and RGB video features for a dataset."""
    features: dict[str, Any] = {
        "action": {"dtype": "float32", "shape": (cfg.action_dim,)},
        "observation.state": {"dtype": "float32", "shape": (cfg.state_dim,)},
    }
    for camera_name in camera_names:
        video_path = episode_dir / "videos" / f"{camera_name}.mp4"
        with av.open(str(video_path)) as container:
            codec = container.streams.video[0].codec_context
            features[f"observation.images.{camera_name}"] = {
                "dtype": "video",
                "shape": (codec.height, codec.width, 3),
                "names": ["height", "width", "channel"],
            }
    return features


def convert_one_dataset(cfg: DatasetConfig):
    ctx = mp.get_context("spawn")

    zarr_episodes = [p for p in cfg.zarr_dataset_dir.iterdir()]
    zarr_episodes.sort()

    camera_names = sorted(p.stem for p in (zarr_episodes[0] / "videos").glob("*.mp4"))

    dataset = LeRobotDataset.create(
        repo_id="local_repo",
        root=cfg.output_dir,
        fps=30,
        features=build_features(cfg, zarr_episodes[0], camera_names),
        streaming_encoding=True,
        # Python Queue(maxsize=0) is unbounded: no capacity-induced frame drops.
        # Encoding can accumulate frames in RAM if decoding is faster.
        encoder_queue_maxsize=0,
    )

    replay_info = []

    with ProcessPoolExecutor(max_workers=len(camera_names), mp_context=ctx) as executor:
        image_ques = {
            camera_name: mp.Queue(maxsize=100) for camera_name in camera_names
        }
        stop_event = ctx.Event()

        try:
            for epi_path in zarr_episodes:
                # for each episode
                for camera_name in camera_names:
                    # for each camera
                    executor.submit(
                        decode_video,
                        epi_path / "videos" / f"{camera_name}.mp4",
                        stop_event,
                        image_ques[camera_name],
                    )

                state, action, replay_initial_positions = load_zarr_episode(
                    epi_path, cfg.action_dim, cfg.state_dim
                )

                episode_index = dataset.meta.total_episodes
                replay_initial_fields = cfg.split_replay_initial_positions(
                    replay_initial_positions
                )
                replay_info.append({
                    "episode_index": episode_index,
                    "replay_initial_values": replay_initial_fields,
                    "orig_episode_path": str(epi_path)
                })

                for t in range(state.shape[0]):
                    frame = {
                        "action": action[t],
                        "observation.state": state[t],
                        "task": cfg.task_description,
                    }
                    for camera_name in camera_names:
                        frame_msg = image_ques[camera_name].get()
                        if isinstance(frame_msg, Exception):
                            raise frame_msg
                        elif frame_msg is None:
                            raise ValueError(
                                f"episode {epi_path} camera {camera_name}: Frame length mismatch."
                            )
                        else:
                            image, idx = frame_msg
                            assert idx == t, (
                                f"episode {epi_path} camera {camera_name}: Frame index mismatch."
                            )
                            frame[f"observation.images.{camera_name}"] = image
                    dataset.add_frame(frame)
                for camera_name in camera_names:
                    assert image_ques[camera_name].get() is None, (
                        f"episode {epi_path} camera {camera_name}: Frame index mismatch."
                    )
                dataset.save_episode()
            
        finally:
            stop_event.set()
            dataset.finalize()
