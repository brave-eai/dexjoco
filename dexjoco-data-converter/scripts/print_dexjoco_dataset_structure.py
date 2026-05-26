from pathlib import Path

import tyro
import yaml
import zarr
from zarr import Array, Group


def _print_zarr_tree_with_leaf_shapes(group: Group, indent: int = 0) -> None:
    # Recursively print zarr tree; leaf arrays include shape/dtype inline.
    prefix = "  " * indent
    keys = sorted(group.keys())

    for key in keys:
        node = group[key]
        if isinstance(node, Group):
            _print_zarr_tree_with_leaf_shapes(node, indent + 1)
        elif isinstance(node, Array):
            print(f"{prefix}{key} (shape={tuple(node.shape)}, dtype={node.dtype})")
        else:
            raise ValueError(f"{prefix}{key} (unknown type={type(node).__name__})")


def _print_videos(videos_dir: Path) -> None:
    # Enforce strict layout: videos/ must be non-empty and contain only files.
    entries = sorted(videos_dir.iterdir(), key=lambda p: p.name)
    if not entries:
        raise ValueError("videos is empty")

    dirs = [p.name for p in entries if p.is_dir()]
    if dirs:
        raise ValueError(f"videos contains subdirectories: {dirs}")

    files = [p.name for p in entries]
    for filename in files:
        print(f"  - {filename}")


def _print_dataset_structure(dataset_dir: Path, episode_index: int) -> None:
    # Assume every child under dataset dir is an episode candidate.
    episodes = sorted(dataset_dir.iterdir(), key=lambda p: p.name)
    episodes = list(filter(lambda p: p.is_dir(), episodes))
    if not episodes:
        raise ValueError(f"{dataset_dir.name}: no episodes found")
    if episode_index < 0 or episode_index >= len(episodes):
        raise IndexError(
            f"{dataset_dir.name}: episode_index={episode_index} out of range "
            f"(valid: 0..{len(episodes) - 1})"
        )

    episode_dir = episodes[episode_index]
    replay_path = episode_dir / "replay.zarr"
    videos_dir = episode_dir / "videos"

    print(f"Dataset: {dataset_dir.name}")
    print(f"Episode: {episode_dir}")

    print("replay.zarr:")
    # Open zarr in read-only mode and print structure from root.
    root = zarr.open(str(replay_path), mode="r")
    if not isinstance(root, Group):
        raise TypeError("replay.zarr root is not a zarr Group")
    _print_zarr_tree_with_leaf_shapes(root, indent=1)

    print("videos:")
    _print_videos(videos_dir)
    print()


def main(
    dataset_path: Path | None = None,
    datasets_path_cfg: Path = Path("configs/datasets_path.yaml"),
    episode_index: int = 0,
) -> None:
    if dataset_path is not None:
        _print_dataset_structure(dataset_path, episode_index)
        return

    with open(datasets_path_cfg, "r") as f:
        datasets_path = yaml.safe_load(f)
        datasets_path = map(Path, datasets_path)

    for dataset_dir in datasets_path:
        _print_dataset_structure(dataset_dir, episode_index)


if __name__ == "__main__":
    tyro.cli(main)


"""
python scripts/print_dexjoco_dataset_structure.py --dataset-path /data/weizhi_zhao/dexjoco/dexjoco_raw_datasets/assembly_3_22_adjust
python scripts/print_dexjoco_dataset_structure.py --datasets-path_cfg configs/filtered_dataset_path.yaml
python scripts/print_dexjoco_dataset_structure.py --datasets-path_cfg configs/assembly_path.yaml
"""
