from pathlib import Path
from pkgutil import iter_modules
from typing import cast

import numpy as np
import pyarrow.parquet as pq
import pytest

import dexjoco.envs
from dexjoco.replay_dataset import check_replay_consistency, load_replay_data

DATASETS_DIR = Path(__file__).resolve().parents[1] / "datasets"
DATASET_GROUPS = ("rand_obj", "rand_full")
ENV_NAMES = sorted(
    module.name
    for module in iter_modules(dexjoco.envs.__path__)
    if not module.name.startswith("_")
)


@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_dataset_replay(env_name: str, pytestconfig: pytest.Config) -> None:
    dataset_dirs = [DATASETS_DIR / group / env_name for group in DATASET_GROUPS]

    for dataset_dir in dataset_dirs:
        assert dataset_dir.is_dir(), f"Missing dataset: {dataset_dir}"

    for dataset_dir in dataset_dirs:
        dataset_id = dataset_dir.relative_to(DATASETS_DIR).as_posix()
        actions, _ = load_replay_data(dataset_dir)
        stored = pq.read_table(
            sorted((dataset_dir / "data").rglob("*.parquet")),
            columns=["episode_index", "action"],
        )
        stored_actions = np.asarray(stored["action"].to_pylist(), dtype=np.float64)
        episode_indices = np.asarray(stored["episode_index"], dtype=np.int64)
        for episode_index, action in enumerate(actions):
            np.testing.assert_array_equal(
                action, stored_actions[episode_indices == episode_index]
            )
        del actions, stored, stored_actions, episode_indices
        consistent, success = check_replay_consistency(
            dataset_dir=dataset_dir,
            task=env_name,
            n_envs=cast(int, pytestconfig.getoption("--replay-n-envs")),
            use_async_envs=True,
            check_images=True,
        )

        assert consistent.all(), (
            f"{dataset_id}: replay differed between passes for episodes: "
            f"{np.flatnonzero(~consistent).tolist()}"
        )
        assert success.all(), (
            f"{dataset_id}: replay failed for episodes: {np.flatnonzero(~success).tolist()}"
        )
