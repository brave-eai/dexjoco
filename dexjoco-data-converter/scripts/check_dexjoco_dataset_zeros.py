from pathlib import Path
from collections.abc import Iterable

import tyro
import zarr
import numpy as np


def _iter_dataset_dirs(input_root: Path) -> Iterable[Path]:
    # Iterate datasets in stable order for deterministic output.
    return (d for d in sorted(input_root.iterdir(), key=lambda p: p.name) if d.is_dir())


def _check_dataset_consistency(dataset_dir: Path,) -> tuple[int, int]:
    episodes = sorted(
        (p for p in dataset_dir.iterdir() if p.is_dir()), key=lambda p: p.name
    )
    if not episodes:
        raise ValueError(f"{dataset_dir.name}: no episodes found")

    bad_episodes: list = []
    dataset_action_z_zeros = 0

    for episode_dir in episodes:
        data_path = episode_dir / "replay.zarr/data"
        data = zarr.open(str(data_path), mode="r")
        is_dual_arm = data["action_rotvec"].shape[1] == 44

        action_z_zeros = np.count_nonzero(data["action_rotvec"][:, 2] == 0)
        if is_dual_arm:
            action_z_zeros += np.count_nonzero(data["action_rotvec"][:, 22 + 2] == 0)

        dataset_action_z_zeros += action_z_zeros

        if action_z_zeros > 0:
            first_nonzero_index = np.argmax(data["action_rotvec"][:, 2] != 0)
            bad_episodes.append((episode_dir, action_z_zeros, first_nonzero_index))

    total = len(episodes)
    bad = len(bad_episodes)
    ratio = bad / total if total else 0.0

    print(f"Dataset: {dataset_dir.name} -- {"dual arm" if is_dual_arm else "single arm"}")
    print(f"  episodes={total}, z-zeros={bad}, ratio={ratio:.2%}, total_action_z_zeros={dataset_action_z_zeros}")
    print(f"  bad_episodes: {"\n".join(f'  - {p.name} (z-zeros: {z}, first_nonzero_index: {i})' for p, z, i in bad_episodes)}")
    return total, bad


def main(
    dataset_path: Path | None = None,
    input_root: Path = Path("/data/weizhi_zhao/dexjoco/dexjoco_raw_datasets"),
    verbose: bool = True,
) -> None:
    if dataset_path is not None:
        _check_dataset_consistency(dataset_path)
        return

    dataset_count = 0
    bad_dataset_count = 0
    total_episodes = 0
    bad_episodes = 0

    for dataset_dir in _iter_dataset_dirs(input_root):
        dataset_count += 1
        total, bad = _check_dataset_consistency(dataset_dir)
        total_episodes += total
        bad_episodes += bad
        if bad > 0:
            bad_dataset_count += 1

    print("Global summary:")
    print(f"  datasets={dataset_count}")
    print(f"  datasets_with_z_zeros={bad_dataset_count}")
    print(f"  total_episodes={total_episodes}")
    print(f"  total_episodes_with_z_zeros={bad_episodes}")


if __name__ == "__main__":
    tyro.cli(main)


"""
python scripts/check_dexjoco_dataset_zeros.py --input-root /data/weizhi_zhao/dexjoco/dexjoco_raw_datasets/datasets_v2
python scripts/check_dexjoco_dataset_zeros.py --input-root /data/weizhi_zhao/dexjoco/dexjoco_raw_datasets/randomize_datasets
"""
