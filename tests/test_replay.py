from pathlib import Path
from pkgutil import iter_modules
from typing import Literal, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import dexjoco.envs
from dexjoco.replay_dataset import check_replay_consistency, load_replay_data

DATASET_GROUPS = ("rand_obj", "rand_full")
ENV_NAMES = sorted(
    module.name for module in iter_modules(dexjoco.envs.__path__) if not module.name.startswith("_")
)


@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_dataset_replay(env_name: str, pytestconfig: pytest.Config) -> None:
    datasets_dir = cast(Path | None, pytestconfig.getoption("--replay-dataset-root"))
    if datasets_dir is None:
        pytest.skip("Pass --replay-dataset-root to enable dataset replay tests")
    dataset_dirs = [datasets_dir / group / env_name for group in DATASET_GROUPS]

    for dataset_dir in dataset_dirs:
        assert dataset_dir.is_dir(), f"Missing dataset: {dataset_dir}"

    for dataset_dir in dataset_dirs:
        dataset_id = dataset_dir.relative_to(datasets_dir).as_posix()
        stored = pq.read_table(
            sorted((dataset_dir / "data").rglob("*.parquet")),
            columns=["episode_index", "action", "action_rotvec"],
        )
        episode_indices = np.asarray(stored["episode_index"], dtype=np.int64)
        formats: tuple[Literal["quaternion", "rotvec"], ...] = ("quaternion", "rotvec")
        for action_format, column in zip(formats, ("action", "action_rotvec"), strict=True):
            assert stored[column].type.value_type == pa.float64()
            actions, _ = load_replay_data(dataset_dir, action_format)
            stored_actions = np.asarray(stored[column].to_pylist(), dtype=np.float64)
            for episode_index, action in enumerate(actions):
                np.testing.assert_array_equal(
                    action, stored_actions[episode_indices == episode_index]
                )
            del actions, stored_actions
        del stored, episode_indices
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
