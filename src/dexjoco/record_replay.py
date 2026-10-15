"""Replay a local dataset and record fresh LeRobot observations."""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import shutil
from contextlib import closing
from pathlib import Path
from typing import Any

import numpy as np
import tyro
from lerobot.datasets import LeRobotDataset
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from tqdm import tqdm

from dexjoco.replay_dataset import ReplayResult, load_replay_data, replay_batches


def build_features(result: ReplayResult) -> dict[str, Any]:
    """Build features from the replay's states, actions, and RGB frames."""
    assert result.images is not None
    features: dict[str, Any] = {
        "action": {"dtype": "float64", "shape": (result.actions[0].shape[1],)},
        "observation.state": {"dtype": "float64", "shape": (result.states[0].shape[1],)},
    }
    for camera, frames in result.images[0].items():
        features[f"observation.images.{camera}"] = {
            "dtype": "video",
            "shape": frames.shape[1:],
            "names": ["height", "width", "channel"],
        }
    return features


def record_replay(
    dataset_dir: Path,
    output_dir: Path,
    randomize: bool = False,
    render_hw: tuple[int, int] | None = None,
    n_envs: int = 10,
    use_async_envs: bool = True,
    save_failed: bool = False,
) -> None:
    """Record water_plant replays, optionally with fresh visual randomization.

    Args:
        dataset_dir: Source LeRobot dataset containing replay_initial_values.jsonl.
        output_dir: New output dataset directory; must not already exist.
        n_envs: Number of parallel replay environments.
        use_async_envs: Run environments in separate processes.
        randomize: Randomize cameras, lighting, and table textures.
        render_hw: Image resolution as (height, width); None uses the environment default.
        save_failed: Save failed episodes too; otherwise raise on any failed batch.
    """
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    metadata = LeRobotDatasetMetadata(repo_id="local_repo", root=dataset_dir)
    assert metadata.episodes is not None
    actions, initial_states = load_replay_data(dataset_dir)
    dataset: LeRobotDataset | None = None
    source_index = 0

    try:
        with (
            closing(
                replay_batches(
                    actions,
                    initial_states,
                    n_envs=n_envs,
                    use_async_envs=use_async_envs,
                    return_images=True,
                    randomize=randomize,
                    render_hw=render_hw,
                )
            ) as batches,
            tqdm(total=len(actions), desc="Recording replay", unit="episode") as progress,
        ):
            for result in batches:
                assert result.images is not None
                if not save_failed and np.any(~result.success):
                    raise RuntimeError(
                        f"Replay failed for source episodes: "
                        f"{(source_index + np.flatnonzero(~result.success)).tolist()}"
                    )
                if dataset is None:
                    dataset = LeRobotDataset.create(
                        repo_id="local_repo",
                        root=output_dir,
                        fps=metadata.fps,
                        features=build_features(result),
                        streaming_encoding=True,
                        encoder_queue_maxsize=0,
                    )
                for i in range(len(result.success)):
                    episode_tasks = metadata.episodes[source_index]["tasks"]
                    if len(episode_tasks) != 1:
                        raise ValueError(
                            f"Source episode {source_index} must have exactly one task prompt"
                        )
                    task_description = episode_tasks[0]
                    for t in range(len(result.actions[i])):
                        frame = {
                            "action": result.actions[i][t],
                            "observation.state": result.states[i][t],
                            "task": task_description,
                        }
                        for camera, frames in result.images[i].items():
                            frame[f"observation.images.{camera}"] = frames[t]
                        dataset.add_frame(frame)
                    dataset.save_episode()
                    source_index += 1
                    progress.update(1)
                del result
    finally:
        if dataset is not None:
            dataset.finalize()

    shutil.copy(dataset_dir / "replay_initial_values.jsonl", output_dir)
    print(f"Replayed {source_index} episodes")


def cli() -> None:
    tyro.cli(record_replay)


if __name__ == "__main__":
    cli()
