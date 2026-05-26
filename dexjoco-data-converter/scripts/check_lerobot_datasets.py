from pathlib import Path

import tyro
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader

from tqdm import tqdm


def _check_dataset(dataset_path: Path, batch_size: int = 128) -> None:
    # Read the whole dataset once to catch broken metadata or shards.
    dataset = LeRobotDataset(repo_id="none", root=dataset_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=16)
    for _ in tqdm(loader, desc=f"Checking dataset: {dataset_path}"):
        pass
    print(f"Dataset {dataset_path} passed.")


def main(
    dataset_path: Path | None = None,
    input_root: Path | None = None,
) -> None:
    if dataset_path is not None:
        _check_dataset(dataset_path)
        return

    assert input_root is not None
    for path in input_root.iterdir():
        _check_dataset(path)


if __name__ == "__main__":
    tyro.cli(main)


"""
python scripts/check_lerobot_datasets.py --dataset-path /data1/weizhi_zhao/dexjoco/lerobot_datasets/randomize_datasets/fold_glasses
python scripts/check_lerobot_datasets.py --dataset-path /data1/weizhi_zhao/dexjoco/lerobot_datasets/randomize_datasets/pick_bucket
python scripts/check_lerobot_datasets.py --input-root /data1/weizhi_zhao/dexjoco/lerobot_datasets/datasets_v2
"""
