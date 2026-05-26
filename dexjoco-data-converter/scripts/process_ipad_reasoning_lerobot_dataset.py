from pathlib import Path

import yaml

from dexjoco_data_converter.to_lerobot.ipad_reasoning_merge_episode_lerobot import merge_episode


def main(
    input_path: Path,
    output_path: Path,
    slice_yaml: str,
    selected_data_yaml: str,
    bad_episodes_yaml: str | None = None,
    skip_static_frames: bool = True,
) -> None:
    slice_spec = yaml.safe_load(slice_yaml)
    selected_data = yaml.safe_load(selected_data_yaml)
    bad_episodes = (
        None if bad_episodes_yaml is None else yaml.safe_load(bad_episodes_yaml)
    )
    assert isinstance(slice_spec, dict), "slice_yaml must be a YAML dict string"
    assert isinstance(selected_data, dict), (
        "selected_data_yaml must be a YAML dict string"
    )
    assert bad_episodes is None or isinstance(bad_episodes, list), (
        "bad_episodes_yaml must be a YAML list string"
    )

    merge_episode(
        input=input_path,
        output=output_path,
        selected_data=selected_data,
        slice_spec=slice_spec,
        skip_static_frames=skip_static_frames,
    )


if __name__ == "__main__":
    import tyro

    tyro.cli(main)


"""
python scripts/process_ipad_reasoning_lerobot_dataset.py \
    --input-path /data/weizhi_zhao/dexjoco/dexjoco_raw_datasets/reasoning_exp \
    --output-path /data/weizhi_zhao/dexjoco/lerobot_datasets/datasets_v2/bimanual_unlock_ipad_reasoning \
    --slice-yaml "{state: [null, 46]}" \
    --selected-data-yaml "{cameras: [ego, wrist_left, wrist_right], action: action_rotvec, state: state}"
"""
