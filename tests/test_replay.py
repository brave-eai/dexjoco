from pathlib import Path
from pkgutil import iter_modules

import numpy as np
import pytest

import dexjoco.envs
from dexjoco.replay_dataset import check_replay_consistency

DATASETS_DIR = Path(__file__).resolve().parents[1] / "datasets"
DATASET_GROUPS = ("rand_obj", "rand_full")
ENV_NAMES = sorted(
    module.name
    for module in iter_modules(dexjoco.envs.__path__)
    if not module.name.startswith("_")
)


@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_dataset_replay(env_name: str) -> None:
    dataset_dirs = [DATASETS_DIR / group / env_name for group in DATASET_GROUPS]

    for dataset_dir in dataset_dirs:
        assert dataset_dir.is_dir(), f"Missing dataset: {dataset_dir}"

    for dataset_dir in dataset_dirs:
        dataset_id = dataset_dir.relative_to(DATASETS_DIR).as_posix()
        consistent, success = check_replay_consistency(
            dataset_dir=dataset_dir,
            n_envs=10,
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
