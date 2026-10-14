import json
from collections.abc import Generator
from contextlib import closing
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
    images: list[dict[str, NDArray[np.uint8]]] | None = None


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
    extend_steps: int,
    return_images: bool = False,
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
    image_frames: list[dict[str, list[NDArray[np.uint8]]]] = (
        [{camera: [] for camera in obs["images"]} for _ in range(NUM_EPISODES)]
        if return_images
        else []
    )
    done = np.arange(env.num_envs) >= NUM_EPISODES
    success = np.zeros(env.num_envs, dtype=bool)
    replay_lengths = np.zeros(env.num_envs, dtype=np.int64)
    for step in range(MAX_STEPS):
        # Stored state[t] precedes action[t], matching the original recorder.
        states.append(np.asarray(obs["proprio"], dtype=np.float64).copy())
        if return_images:
            for i in range(NUM_EPISODES):
                if not done[i]:
                    for camera, frames in image_frames[i].items():
                        frames.append(obs["images"][camera][i].copy())
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

    episode_images = (
        [
            {camera: np.stack(frames) for camera, frames in episode.items()}
            for episode in image_frames
        ]
        if return_images
        else None
    )
    return ReplayResult(episode_states, episode_success, episode_images)


def replay_batches(
    actions_by_episode: list[NDArray[np.float64]],
    initial_states: list[dict],
    n_envs: int,
    use_async_envs: bool,
    extend_steps: int,
    return_images: bool = False,
) -> Generator[ReplayResult, None, None]:
    """Yield each batch in episode order; exhaust or close to release environments."""
    if n_envs < 1 or extend_steps < 0:
        raise ValueError("n_envs must be positive and extend_steps must be nonnegative")
    if not actions_by_episode:
        raise ValueError("No episodes to replay")
    if len(actions_by_episode) != len(initial_states):
        raise ValueError("Each episode must have an initial state")
    if any(len(actions) == 0 for actions in actions_by_episode):
        raise ValueError("Cannot replay an episode with no actions")

    env = make_env(
        DexJoCoEnvConfig(render_mode="rgb_array" if return_images else "none"),
        n_envs=n_envs,
        use_async_envs=use_async_envs,
    )["dexjoco"][0]  # TODO: only support water_plant for now
    try:
        for start in range(0, len(actions_by_episode), n_envs):
            batch_actions = actions_by_episode[start : start + n_envs]
            lengths = np.asarray([len(actions) for actions in batch_actions], dtype=np.int64)
            # pad actions to the max length
            action_dim = batch_actions[0].shape[1]
            actions = np.zeros(
                (len(batch_actions), int(lengths.max()), action_dim), dtype=np.float64
            )
            for slot, episode_actions in enumerate(batch_actions):
                actions[slot, : lengths[slot]] = episode_actions
            result = batch_replay_episode(
                env,
                actions,
                initial_states[start : start + n_envs],
                lengths,
                extend_steps,
                return_images=return_images,
            )
            yield result
            del result
    finally:
        env.close()


def load_replay_data(
    dataset_dir: Path,
) -> tuple[list[NDArray[np.float64]], list[dict]]:
    """Load actions and initial states in episode index order."""
    dataset = LeRobotDataset(repo_id="local_repo", root=dataset_dir, download_videos=False)
    rows = dataset.select_columns(["episode_index", "action"])
    episode_indices = np.asarray(rows["episode_index"], dtype=np.int64).reshape(-1)
    with (dataset_dir / "replay_initial_values.jsonl").open() as f:
        entries = sorted((json.loads(line) for line in f), key=lambda entry: entry["episode_index"])
    if [entry["episode_index"] for entry in entries] != list(range(dataset.meta.total_episodes)):
        raise ValueError("Initial-state records must cover every episode exactly once")
    initial_states = [entry["replay_initial_values"] for entry in entries]
    actions_by_episode = []
    for episode_index in range(dataset.meta.total_episodes):
        episode_rows = rows.select(np.flatnonzero(episode_indices == episode_index))
        actions_by_episode.append(np.asarray(episode_rows["action"], dtype=np.float64))

    return actions_by_episode, initial_states


def check_replay_success(
    dataset_dir: Path,
    n_envs: int = 10,
    use_async_envs: bool = True,
    extend_steps: int = 30,
) -> NDArray[np.bool_]:
    """Replay once without rendering, returning success in episode index order."""
    actions, initial_states = load_replay_data(dataset_dir)
    success = np.zeros(len(actions), dtype=bool)
    start = 0
    with closing(
        replay_batches(actions, initial_states, n_envs, use_async_envs, extend_steps)
    ) as batches:
        for result in batches:
            stop = start + len(result.success)
            success[start:stop] = result.success
            start = stop
            del result
    return success


def check_replay_consistency(
    dataset_dir: Path,
    n_envs: int = 10,
    use_async_envs: bool = True,
    extend_steps: int = 30,
    check_images: bool = False,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Return exact consistency and first-pass success in episode index order.

    Success flags and states must match, along with RGB frames when requested.
    Failed episodes can still be consistent. Only the current batch is retained.
    """
    actions, initial_states = load_replay_data(dataset_dir)
    consistent = np.zeros(len(actions), dtype=bool)
    success = np.zeros(len(actions), dtype=bool)
    start = 0
    with (
        closing(
            replay_batches(
                actions,
                initial_states,
                n_envs,
                use_async_envs,
                extend_steps,
                check_images,
            )
        ) as first_batches,
        closing(
            replay_batches(
                actions,
                initial_states,
                n_envs,
                use_async_envs,
                extend_steps,
                check_images,
            )
        ) as second_batches,
    ):
        # Advance separately so zip does not retain the previous pair of results.
        for first in first_batches:
            second = next(second_batches)
            stop = start + len(first.success)
            success[start:stop] = first.success
            consistent[start:stop] = first.success == second.success
            consistent[start:stop] &= np.asarray(
                [np.array_equal(a, b) for a, b in zip(first.states, second.states, strict=True)],
                dtype=bool,
            )
            if check_images:
                assert first.images is not None and second.images is not None
                for i, (a, b) in enumerate(zip(first.images, second.images, strict=True)):
                    consistent[start + i] &= a.keys() == b.keys() and all(
                        np.array_equal(a[camera], b[camera]) for camera in a
                    )
                del a, b
            start = stop
            del first, second
    return consistent, success


if __name__ == "__main__":
    tyro.cli(check_replay_success)
