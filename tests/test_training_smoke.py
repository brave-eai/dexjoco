"""Run the official LeRobot training CLI against both DexJoCo entry points."""

import ast
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import torch

from dexjoco.configs import DexJoCoEnvConfig
from dexjoco.envs.click_mouse import ClickMouseEnv


@pytest.fixture(scope="module")
def training_dataset_root(pytestconfig: pytest.Config) -> Path:
    root: Path | None = pytestconfig.getoption("--training-dataset-root")
    if root is None:
        pytest.skip("Pass --training-dataset-root to run ACT training")
    if not (root / "meta/info.json").is_file():
        pytest.fail("Pass --training-dataset-root pointing to the local click_mouse dataset")
    if not torch.cuda.is_available():
        pytest.fail("ACT training smoke tests require a CUDA GPU")
    tasks = pq.read_table(root / "meta/tasks.parquet", columns=["task"])["task"].to_pylist()
    assert tasks == [ClickMouseEnv.task_description]
    return root.resolve()


@pytest.mark.parametrize("env_type", ["dexjoco", "envhub"])
def test_act_training_with_periodic_eval(
    env_type: str, training_dataset_root: Path, tmp_path: Path,
) -> None:
    repo = Path(__file__).resolve().parents[1]
    output = tmp_path / "train"
    cfg = DexJoCoEnvConfig(task="click_mouse")
    env_args = [f"--env.type={env_type}"]
    if env_type == "envhub":
        env_args.extend([
            f"--env.hub_path={repo}",
            "--env.trust_remote_code=true",
            f"--env.make_env_kwargs={json.dumps(cfg.to_make_env_kwargs())}",
        ])
    else:
        env_args.append(f"--env.task={cfg.task}")

    result = subprocess.run(
        [
            sys.executable, "-m", "lerobot.scripts.lerobot_train",
            f"--output_dir={output}",
            f"--dataset.repo_id=local/{cfg.task}",
            f"--dataset.root={training_dataset_root}",
            "--dataset.episodes=[0]", "--dataset.video_backend=pyav",
            "--policy.type=act", "--policy.device=cuda:0", "--policy.push_to_hub=false",
            "--policy.pretrained_backbone_weights=null",
            "--policy.chunk_size=4", "--policy.n_action_steps=2",
            "--policy.dim_model=128", "--policy.dim_feedforward=256",
            "--policy.n_encoder_layers=1", "--policy.n_vae_encoder_layers=1",
            "--steps=2", "--batch_size=1", "--num_workers=0", "--log_freq=1",
            "--env_eval_freq=2", "--eval.n_episodes=1", "--eval.batch_size=1",
            "--eval.use_async_envs=false", "--eval.max_episode_steps=2",
            *env_args,
        ],
        cwd=repo,
        env={**os.environ, "HF_HOME": str(tmp_path / "hf"), "HF_HUB_OFFLINE": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    losses = re.findall(r"\bstep:\s*\d+[^\n]*?\bloss:\s*(\S+)", result.stdout)
    assert len(losses) == 2 and all(math.isfinite(float(loss)) for loss in losses), result.stdout
    checkpoint = output / "checkpoints/last"
    assert (checkpoint / "pretrained_model/model.safetensors").is_file()
    assert (checkpoint / "training_state/optimizer_state.safetensors").is_file()

    assert "Eval policy at step 2" in result.stdout, result.stdout
    match = re.search(r"Suite overall aggregated: (\{[^\n]+\})", result.stdout)
    assert match is not None, result.stdout
    metrics = ast.literal_eval(match[1])
    assert metrics["n_episodes"] == 1
    assert math.isfinite(metrics["pc_success"])
