from pathlib import Path
from collections.abc import Iterable

import tyro
import zarr
from zarr import Array
from zarr import Group
import imageio.v3 as iio


def _iter_dataset_dirs(input_root: Path) -> Iterable[Path]:
    # Iterate datasets in stable order for deterministic output.
    return (d for d in sorted(input_root.iterdir(), key=lambda p: p.name) if d.is_dir())


def _collect_array_name_shape(
    group: Group, prefix: str = ""
) -> list[tuple[str, tuple[int, ...]]]:
    keys = sorted(group.keys())
    arrays: list[tuple[str, tuple[int, ...]]] = []
    for key in keys:
        node = group[key]
        if isinstance(node, Group):
            arrays.extend(_collect_array_name_shape(node, prefix=prefix + key + "/"))
        elif isinstance(node, Array):
            arrays.append((f"{prefix}{key}", tuple(node.shape)))
        else:
            raise ValueError(f"{prefix}{key} (unknown type={type(node).__name__})")
    return arrays


def _get_array_name_shape(path: Path) -> list[tuple[str, tuple[int, ...]]]:
    group = zarr.open(str(path), mode="r")
    if not isinstance(group, Group):
        raise TypeError(f"{path}: not a zarr Group")
    return _collect_array_name_shape(group, prefix="")


def _get_videos_name_shape(videos_dir: Path) -> list[tuple[str, tuple[int, ...]]]:
    entries = sorted(videos_dir.iterdir(), key=lambda p: p.name)
    if not entries:
        raise ValueError("videos is empty")

    videos: list[tuple[str, tuple[int, ...]]] = []
    for file in entries:
        if file.is_dir():
            raise ValueError(f"videos contains subdirectory: {file.name}")
        props = iio.improps(file, plugin="pyav")
        videos.append((file.name, tuple(props.shape)))
    return videos


def _check_dataset_consistency(dataset_dir: Path, verbose: bool) -> tuple[int, int]:
    episodes = sorted(
        (p for p in dataset_dir.iterdir() if p.is_dir()), key=lambda p: p.name
    )
    if not episodes:
        raise ValueError(f"{dataset_dir.name}: no episodes found")

    first_episode_dir = episodes[0]
    first_array_path = first_episode_dir / "replay.zarr/data"
    first_videos_dir = first_episode_dir / "videos"

    first_array_info = _get_array_name_shape(first_array_path)
    first_videos_info = _get_videos_name_shape(first_videos_dir)

    info_dict = dict(first_array_info + first_videos_info)
    info_keys = set(info_dict.keys())

    bad_episodes: set[Path] = set()
    temporal_mismatch_details: list[str] = []
    name_set_mismatch_details: list[str] = []
    shape_mismatch_details: list[str] = []

    for episode_dir in episodes:
        array_path = episode_dir / "replay.zarr/data"
        videos_dir = episode_dir / "videos"
        array_info = _get_array_name_shape(array_path)
        videos_info = _get_videos_name_shape(videos_dir)
        current_pairs = array_info + videos_info

        current_dict = dict(current_pairs)
        current_keys = set(current_dict.keys())
        missing = sorted(info_keys - current_keys)
        extra = sorted(current_keys - info_keys)
        if missing or extra:
            bad_episodes.add(episode_dir)
            name_set_mismatch_details.append(
                f"{episode_dir}: missing={missing}, extra={extra}"
            )

        for name, shape in current_pairs:
            expected_shape = info_dict.get(name)
            expected_tail = None if expected_shape is None else expected_shape[1:]
            current_tail = shape[1:]
            if expected_shape is None or expected_tail != current_tail:
                bad_episodes.add(episode_dir)
                shape_mismatch_details.append(
                    f"{episode_dir}: {name}, expected_tail={expected_tail}, got_tail={current_tail}"
                )

        first_dims = {shape[0] for _, shape in current_pairs}
        if len(first_dims) != 1:
            bad_episodes.add(episode_dir)
            parts = [f"{name}:{shape[0]}" for name, shape in current_pairs]
            temporal_mismatch_details.append(f"{episode_dir}: " + ", ".join(parts))

    total = len(episodes)
    bad = len(bad_episodes)
    ratio = bad / total if total else 0.0

    print(f"Dataset: {dataset_dir.name}")
    print(f"  episodes={total}, inconsistent={bad}, ratio={ratio:.2%}")
    if bad == 0:
        print("  PASS")
        print()
        return total, bad

    if verbose:
        print("  temporal mismatch:")
        if temporal_mismatch_details:
            for line in temporal_mismatch_details:
                print(f"    - {line}")
        else:
            print("    - none")

        print("  name set mismatch:")
        if name_set_mismatch_details:
            for line in name_set_mismatch_details:
                print(f"    - {line}")
        else:
            print("    - none")

        print("  shape mismatch:")
        if shape_mismatch_details:
            for line in shape_mismatch_details:
                print(f"    - {line}")
        else:
            print("    - none")
    print()
    return total, bad


def main(
    dataset_path: Path | None = None,
    input_root: Path = Path("/data/weizhi_zhao/dexjoco/dexjoco_raw_datasets"),
    verbose: bool = True,
) -> None:
    if dataset_path is not None:
        _check_dataset_consistency(dataset_path, verbose=verbose)
        return

    dataset_count = 0
    bad_dataset_count = 0
    total_episodes = 0
    bad_episodes = 0

    for dataset_dir in _iter_dataset_dirs(input_root):
        dataset_count += 1
        total, bad = _check_dataset_consistency(dataset_dir, verbose=verbose)
        total_episodes += total
        bad_episodes += bad
        if bad > 0:
            bad_dataset_count += 1

    print("Global summary:")
    print(f"  datasets={dataset_count}")
    print(f"  datasets_with_inconsistency={bad_dataset_count}")
    print(f"  total_episodes={total_episodes}")
    print(f"  total_inconsistent_episodes={bad_episodes}")


if __name__ == "__main__":
    tyro.cli(main)


"""
python scripts/check_dexjoco_dataset_consistency.py --dataset-path=/data/weizhi_zhao/dexjoco/dexjoco_raw_datasets/ipad_pswd123_3_22_test
python scripts/check_dexjoco_dataset_consistency.py --dataset-path=/data/weizhi_zhao/dexjoco/dexjoco_raw_datasets/ipad_pswd123_3_22_replay_test 
python scripts/check_dexjoco_dataset_consistency.py --input-root /data/weizhi_zhao/dexjoco/dexjoco_raw_datasets/dataset_filter

python scripts/check_dexjoco_dataset_consistency.py --dataset-path /data/weizhi_zhao/dexjoco/dexjoco_raw_datasets/assembly_3_22_adjust

python scripts/check_dexjoco_dataset_consistency.py --input-root /data1/weizhi_zhao/dexjoco/raw_datasets/randomize_datasets
"""
