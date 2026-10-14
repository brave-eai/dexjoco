from pathlib import Path
from pkgutil import iter_modules

import numpy as np
import pytest

import dexjoco.envs
from dexjoco.replay_dataset import check_replay_consistency

DATASETS_DIR = Path(__file__).resolve().parents[1] / "datasets"


@pytest.mark.parametrize(
    "dataset_dir",
    sorted(path for path in DATASETS_DIR.iterdir() if path.is_dir()),
    ids=lambda path: path.name,
)
def test_dataset_replay(dataset_dir: Path) -> None:
    env_names = {
        module.name
        for module in iter_modules(dexjoco.envs.__path__)
        if not module.name.startswith("_")
    }
    dataset_names = {path.name for path in DATASETS_DIR.iterdir() if path.is_dir()}
    assert dataset_names == env_names, (
        f"Datasets must match environments. Missing: {sorted(env_names - dataset_names)}; "
        f"unexpected: {sorted(dataset_names - env_names)}"
    )
    consistent, success = check_replay_consistency(
        dataset_dir=dataset_dir,
        n_envs=10,
        use_async_envs=True,
        extend_steps=30,
        check_images=True
    )

    assert success.all(), f"Replay failed for episodes: {np.flatnonzero(~success).tolist()}"
    assert consistent.all(), (
        f"Replay differed between passes for episodes: {np.flatnonzero(~consistent).tolist()}"
    )
