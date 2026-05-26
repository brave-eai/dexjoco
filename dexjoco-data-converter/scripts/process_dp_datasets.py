from pathlib import Path

from dexjoco_data_converter.to_zarr.batch_merge_episode_zarr import batch_merge_episode


def main(
    output_path: Path,
    dataset_paths_cfg_path: Path,
    slice_cfg_path: Path,
    num_workers: int = 4,
    bad_episodes_cfg_path: Path | None = None,
    skip_static_frames: bool = True,
) -> None:
    batch_merge_episode(
        output=output_path,
        dataset_paths_cfg_path=dataset_paths_cfg_path,
        slice_cfg_path=slice_cfg_path,
        num_workers=num_workers,
        bad_episodes_cfg_path=bad_episodes_cfg_path,
        skip_static_frames=skip_static_frames,
    )


if __name__ == "__main__":
    import tyro

    tyro.cli(main)


"""
python dexjoco-data-converter/scripts/process_dp_datasets.py \
    --output-path="./test" \
    --dataset_paths_cfg_path dexjoco-data-converter/configs/dexjoco/rand_obj/dataset_paths.yaml \
    --slice_cfg_path dexjoco-data-converter/configs/dexjoco/rand_obj/slice_config.yaml \
    --num-workers=2
"""
