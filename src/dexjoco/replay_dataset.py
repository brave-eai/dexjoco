import json
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import tyro
from lerobot.datasets import LeRobotDataset
from lerobot.envs import make_env
from numpy.typing import NDArray

from dexjoco.configs import DexJoCoEnvConfig


@dataclass
class ReplayResult:
    states: list[NDArray]
    success: NDArray[np.bool]


def batch_env_reset(env: gym.vector.VectorEnv, initial_states: list[dict | None]):
    for i, initial_state in enumerate(initial_states):
        mask = np.zeros(env.num_envs, dtype=bool)
        mask[i] = True

        obs, _ = env.reset(
            options={
                "reset_mask": mask,
                "initial_state": initial_state,
            }
        )
    return obs


def batch_replay_episode(
    env: gym.vector.VectorEnv,
    actions: NDArray[np.float64],
    initial_states: list,
    episodes_lengths: NDArray[np.int64],
    extend_steps: int = 30,
) -> ReplayResult:
    """
    shape of actions: (num_episodes, max_episode_length, action_dim)
    """
    NUM_EPISODES = len(episodes_lengths)
    MAX_STEPS = int(episodes_lengths[:NUM_EPISODES].max()) + extend_steps
    assert NUM_EPISODES <= env.num_envs, (
        f"Number of episodes {NUM_EPISODES} exceeds number of vector envs {env.num_envs}"
    )
    if NUM_EPISODES < env.num_envs:
        # Fill the remaining envs with dummy actions and initial states.
        dummy_actions = np.zeros(
            (env.num_envs - NUM_EPISODES, *actions.shape[1:]), dtype=np.float64
        )
        actions = np.concatenate([actions, dummy_actions], axis=0)
        dummy_initials = [None] * (env.num_envs - NUM_EPISODES)
        initial_states = initial_states + dummy_initials
        # Fill the remaining envs with dummy episode lengths
        episodes_lengths = np.concatenate(
            [episodes_lengths, np.ones(env.num_envs - NUM_EPISODES, dtype=np.int64)], axis=0
        )

    obs = batch_env_reset(env, initial_states)
    states = []
    done = np.arange(env.num_envs) >= NUM_EPISODES
    success = np.zeros(env.num_envs, dtype=bool)
    replay_lengths = np.zeros(env.num_envs, dtype=np.int64)
    for step in range(MAX_STEPS):
        # Stored state[t] precedes action[t], matching the original recorder.
        states.append(np.asarray(obs["proprio"], dtype=np.float64).copy())
        action_indices = np.minimum(step, episodes_lengths - 1)
        action = actions[np.arange(env.num_envs), action_indices]
        obs, _, terminated, truncated, info = env.step(action)

        current_success = ~done & np.asarray(info["is_success"], dtype=bool)
        success |= current_success

        # Mark episodes done on success, termination, truncation, or their step limit.
        # Success on the final allowed step has already been recorded above.
        newly_done = ~done & (
            current_success | terminated | truncated | (step + 1 >= episodes_lengths + extend_steps)
        )
        replay_lengths[newly_done] = step + 1
        done |= newly_done

        if done.all():
            break

    states_array = np.asarray(states)  # (steps, envs, state_dim)
    episode_states = [states_array[: replay_lengths[i], i].copy() for i in range(NUM_EPISODES)]
    episode_success = success[:NUM_EPISODES]

    return ReplayResult(episode_states, episode_success)


def state_errors(actual: NDArray[np.float64], expected: NDArray[np.float64]) -> NDArray[np.float64]:
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
        f"rmse={np.sqrt(np.mean(errors**2)):.8g}, "
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
        env = make_env(DexJoCoEnvConfig(render_mode="none"), n_envs=1, use_async_envs=False)[
            "dexjoco"
        ][0]
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
        print(
            f"Pass {pass_index + 1}: {sum(r.success for r in results.values())}"
            f"/{len(results)} episodes succeeded.",
            flush=True,
        )

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
