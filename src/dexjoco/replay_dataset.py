"""Replay converted water_plant data twice using the local EnvHub interface."""

import json
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import tyro
from lerobot.datasets import LeRobotDataset
from numpy.typing import NDArray

from dexjoco.envhub import make_env


@dataclass
class ReplayResult:
    states: NDArray[np.float64]
    success: bool
    reason: str


def replay_episode(
    env: gym.vector.VectorEnv,
    actions: NDArray[np.float64],
    initial: dict,
) -> ReplayResult:
    obs, _ = env.reset(options={
        "initial_state": {
            "table_delta_height": initial["table_delta_height"][0],
            "orig_spray_pose": initial["spray_ori_pose"],
            "orig_plant_pose": initial["plant_ori_pose"],
        },
    })
    states = []
    success = False
    reason = "extension exhausted"
    for step in range(len(actions) + 10):
        # Stored state[t] precedes action[t], matching the original recorder.
        states.append(np.asarray(obs["proprio"][0], dtype=np.float64).copy())
        action = actions[min(step, len(actions) - 1)]
        obs, _, terminated, truncated, info = env.step(action[None, :])
        if bool(info["is_success"][0]):
            success = True
            reason = "success"
            break
        if bool(terminated[0] or truncated[0]):
            # NEXT_STEP would reset on the next call, losing the recorded scene.
            reason = "terminated" if terminated[0] else "truncated"
            break
    return ReplayResult(np.asarray(states), success, reason)


def state_errors(
    actual: NDArray[np.float64], expected: NDArray[np.float64]
) -> NDArray[np.float64]:
    count = min(len(actual), len(expected))
    aligned = actual[:count].copy()
    expected = expected[:count]
    signs = np.sum(aligned[:, 3:7] * expected[:, 3:7], axis=1) < 0
    aligned[signs, 3:7] *= -1
    return np.abs(aligned - expected)


def report_errors(label: str, errors: NDArray[np.float64], atol: float) -> None:
    bad = np.count_nonzero(np.any(~np.isfinite(errors) | (errors > atol), axis=1))
    print(
        f"{label}: compared_frames={len(errors)}, bad_frames={bad}, "
        f"mae={errors.mean():.8g}, max_abs={errors.max():.8g}, "
        f"rmse={np.sqrt(np.mean(errors ** 2)):.8g}, "
        f"xyz_max_m={errors[:, :3].max():.8g}, "
        f"quat_max={errors[:, 3:7].max():.8g}, "
        f"finger_max_rad={errors[:, 7:].max():.8g}",
        flush=True,
    )


def main(
    dataset_dir: Path = Path("datasets/water_plant"),
    episodes: list[int] | None = None,
    atol: float = 1e-4,
) -> None:
    """Replay every selected episode twice and compare states in memory.

    Args:
        dataset_dir: Converted water_plant LeRobot dataset directory.
        episodes: Episode indices; defaults to all episodes.
        atol: Absolute per-component tolerance, with no relative tolerance.
    """
    dataset = LeRobotDataset(
        repo_id="local_repo", root=dataset_dir, episodes=episodes, download_videos=False
    )
    rows = dataset.select_columns(["episode_index", "action", "observation.state"])
    episode_indices = np.asarray(rows["episode_index"], dtype=np.int64).reshape(-1)
    with (dataset_dir / "replay_initial_values.jsonl").open() as f:
        initial_values = {
            entry["episode_index"]: entry["replay_initial_values"]
            for entry in (json.loads(line) for line in f)
        }
    trajectories = {}
    for episode_index in np.unique(episode_indices):
        episode_rows = rows.select(np.flatnonzero(episode_indices == episode_index).tolist())
        trajectories[int(episode_index)] = (
            np.asarray(episode_rows["action"], dtype=np.float64),
            np.asarray(episode_rows["observation.state"], dtype=np.float64),
        )

    # All first-pass episodes complete before the second pass begins.
    passes: list[dict[int, ReplayResult]] = []
    for pass_index in range(2):
        env = make_env(n_envs=1, use_async_envs=False, render_mode="none")["dexjoco"][0]
        results = {}
        try:
            for episode_index, (actions, _) in trajectories.items():
                result = replay_episode(env, actions, initial_values[episode_index])
                results[episode_index] = result
                print(
                    f"pass={pass_index + 1} episode={episode_index}: "
                    f"success={result.success}, steps={len(result.states)}, "
                    f"recorded_steps={len(actions)}, "
                    f"extra_steps={max(0, len(result.states) - len(actions))}, "
                    f"reason={result.reason}",
                    flush=True,
                )
        finally:
            env.close()
        passes.append(results)
        print(f"Pass {pass_index + 1}: {sum(r.success for r in results.values())}"
              f"/{len(results)} episodes succeeded.", flush=True)

    original_errors = []
    repeat_errors = []
    for episode_index, (_, expected) in trajectories.items():
        first = passes[0][episode_index]
        second = passes[1][episode_index]
        print(
            f"episode={episode_index}: recorded_frames={len(expected)}, "
            f"pass1_frames={len(first.states)}, pass2_frames={len(second.states)}, "
            f"same_length={len(first.states) == len(second.states)}, "
            f"same_success={first.success == second.success}",
            flush=True,
        )
        original = state_errors(first.states, expected)
        repeat = state_errors(second.states, first.states)
        original_errors.append(original)
        repeat_errors.append(repeat)
        report_errors("  pass1 vs dataset", original, atol)
        report_errors("  pass2 vs pass1", repeat, atol)
    report_errors("Overall pass1 vs dataset", np.concatenate(original_errors), atol)
    report_errors("Overall pass2 vs pass1", np.concatenate(repeat_errors), atol)


if __name__ == "__main__":
    tyro.cli(main)
