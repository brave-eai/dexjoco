from pathlib import Path

from dexjoco_data_converter.to_lerobot.batch_merge_episode_lerobot import batch_merge_episode


def main(
    output_path: Path,
    dataset_paths_cfg_path: Path,
    language_instruction_cfg_path: Path,
    selected_data_cfg_path: Path,
    slice_cfg_path: Path,
    bad_episodes_cfg_path: Path | None = None,
    skip_static_frames: bool = True,
) -> None:
    batch_merge_episode(
        output=output_path,
        dataset_paths_cfg_path=dataset_paths_cfg_path,
        language_instruction_cfg_path=language_instruction_cfg_path,
        selected_data_cfg_path=selected_data_cfg_path,
        slice_cfg_path=slice_cfg_path,
        bad_episodes_cfg_path=bad_episodes_cfg_path,
        skip_static_frames=skip_static_frames,
    )


if __name__ == "__main__":
    import tyro

    tyro.cli(main)


"""
python dexjoco-data-converter/scripts/process_lerobot_datasets.py \
    --output-path ./test \
    --dataset-paths-cfg-path dexjoco-data-converter/configs/dexjoco/rand_obj/dataset_paths.yaml \
    --language-instruction-cfg-path dexjoco-data-converter/configs/dexjoco/rand_obj/language_instructions.yaml \
    --selected-data-cfg-path dexjoco-data-converter/configs/dexjoco/rand_obj/selected_data.yaml \
    --slice-cfg-path dexjoco-data-converter/configs/dexjoco/rand_obj/slice_config.yaml

python dexjoco-data-converter/scripts/process_dp_datasets.py \
    --output-path="./test" \
    --dataset_paths_cfg_path dexjoco-data-converter/configs/dexjoco/rand_obj/dataset_paths.yaml \
    --slice_cfg_path dexjoco-data-converter/configs/dexjoco/rand_obj/slice_config.yaml \
    --num-workers=2
"""
