"""Exercise LeRobot's real preprocessing, rollout, and plugin discovery paths."""

import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import av
import gymnasium as gym
import numpy as np
import pytest
import torch
from lerobot.envs import make_env
from lerobot.envs.utils import env_to_policy_features, preprocess_observation
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import (
    PolicyProcessorPipeline,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.scripts.lerobot_eval import rollout
from lerobot.utils.io_utils import write_video

from dexjoco.configs import DexJoCoEnvConfig
from dexjoco.envhub import TASK_IDS, TASKS
from dexjoco.replay_dataset import batch_replay_episode


def test_official_cli_discovers_plugin() -> None:
    # A fresh interpreter must discover the distribution without importing dexjoco first.
    result = subprocess.run(
        [sys.executable, "-m", "lerobot.scripts.lerobot_eval", "--env.type=dexjoco", "--help"],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert "DexJoCoEnvConfig" in result.stdout


@pytest.mark.parametrize(
    "task,bimanual",
    [
        ("water_plant", False), ("fold_glasses", False), ("click_mouse", False),
        ("pinch_tongs", False), ("pick_bucket", False), ("hammer_nail", False), ("bimanual_unlock_ipad", True), ("bimanual_hanoi", True), ("bimanual_assembly", True), ("bimanual_photograph", True), ("bimanual_microwave_cook", True),
    ],
)
@pytest.mark.parametrize("use_async_envs", [False, True])
def test_batch_replay_initial_states(task: str, bimanual: bool, use_async_envs: bool) -> None:
    cfg = DexJoCoEnvConfig(task=task, bimanual=bimanual, render_mode="none")
    reference = TASKS[TASK_IDS[task]](render_mode="none")
    try:
        initial_states = [reference.reset(seed=seed)[1]["initial_state"] for seed in (1, 2)]
    finally:
        reference.close()
    env = make_env(cfg, n_envs=2, use_async_envs=use_async_envs)["dexjoco"][TASK_IDS[task]]
    try:
        result = batch_replay_episode(
            env,
            np.zeros((2, 1, *cfg.features["action"].shape), dtype=np.float64),
            initial_states,
            np.ones(2, dtype=np.int64),
            extend_steps=0,
        )
        assert len(result.states) == 2
        assert all(state.shape == (1, *cfg.features["agent_pos"].shape) for state in result.states)
    finally:
        env.close()


@pytest.mark.parametrize(
    "task,bimanual",
    [
        ("water_plant", False), ("fold_glasses", False), ("click_mouse", False),
        ("pinch_tongs", False), ("pick_bucket", False), ("hammer_nail", False), ("bimanual_unlock_ipad", True), ("bimanual_hanoi", True), ("bimanual_assembly", True), ("bimanual_photograph", True), ("bimanual_microwave_cook", True),
    ],
)
@pytest.mark.parametrize("use_async_envs", [False, True])
@pytest.mark.parametrize("with_images", [False, True])
def test_official_rollout_and_video(
    task: str, bimanual: bool, use_async_envs: bool, with_images: bool,
) -> None:
    cfg = DexJoCoEnvConfig(
        task=task,
        bimanual=bimanual,
        render_mode="rgb_array" if with_images else "none",
        render_height=48,
        render_width=64,
    )
    features = env_to_policy_features(cfg)
    expected_inputs = set(features) - {"action"}
    frames = []

    class HoldPolicy(torch.nn.Module):
        def reset(self):
            pass

        def select_action(self, observation):
            assert expected_inputs | {"task"} <= set(observation)
            for key in expected_inputs:
                assert observation[key].shape == (2, *features[key].shape)
                assert observation[key].dtype == torch.float32
            return torch.zeros((2, *features["action"].shape))

    env = make_env(cfg, n_envs=2, use_async_envs=use_async_envs)["dexjoco"][TASK_IDS[task]]
    try:
        assert isinstance(env, (gym.vector.SyncVectorEnv, gym.vector.AsyncVectorEnv))
        obs: dict
        obs, _ = env.reset(seed=[1, 2])
        single_obs = {"agent_pos": obs["agent_pos"][0]}
        if with_images:
            single_obs["pixels"] = {
                camera: image[0] for camera, image in obs["pixels"].items()
            }
        assert env.single_observation_space.contains(single_obs)
        processed = preprocess_observation(obs)
        assert set(processed) == expected_inputs
        if with_images:
            for camera in obs["pixels"]:
                expected = torch.from_numpy(obs["pixels"][camera]).permute(0, 3, 1, 2).float() / 255
                torch.testing.assert_close(processed[f"observation.images.{camera}"], expected)
        env.call("set_wrapper_attr", "_max_episode_steps", 2)
        env_preprocessor, env_postprocessor = cfg.get_env_processors()

        def capture_frame(vector_env):
            frames.append(vector_env.call("render")[0])

        result = rollout(
            env=env,
            # rollout accepts nn.Module policies with reset() and select_action().
            policy=cast(PreTrainedPolicy, HoldPolicy()),
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=PolicyProcessorPipeline(steps=[]),
            postprocessor=PolicyProcessorPipeline(
                steps=[],
                to_transition=policy_action_to_transition,
                to_output=transition_to_policy_action,
            ),
            seeds=[1, 2],
            return_observations=True,
            render_callback=capture_frame if with_images else None,
        )
        assert result["action"].shape == (2, 2, *features["action"].shape)
        assert result["done"][:, -1].all()
        assert result["observation"]["observation.state"].shape == (2, 3, *features["observation.state"].shape)
        if with_images:
            with TemporaryDirectory(prefix="dexjoco-video-test-") as directory:
                path = Path(directory) / "rollout.mp4"
                write_video(path, frames, fps=env.metadata["render_fps"])
                with av.open(str(path)) as container:
                    assert container.streams.video[0].average_rate == 30
                    decoded = list(container.decode(video=0))
                    assert len(decoded) == 3
                    assert (decoded[0].height, decoded[0].width) == (48, 64)
    finally:
        env.close()
