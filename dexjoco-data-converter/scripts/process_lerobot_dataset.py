from pathlib import Path

import yaml

from dexjoco_data_converter.to_lerobot.merge_episode_lerobot import merge_episode


def main(
    input_path: Path,
    output_path: Path,
    language_instruction: str,
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
        language_instruction=language_instruction,
        selected_data=selected_data,
        slice_spec=slice_spec,
        bad_episodes=bad_episodes,
        skip_static_frames=skip_static_frames,
    )


if __name__ == "__main__":
    import tyro

    tyro.cli(main)


"""
python scripts/process_lerobot_dataset.py \
    --input-path /data/weizhi_zhao/dexjoco/dexjoco_raw_datasets/water_plant \
    --output-path /data/weizhi_zhao/dexjoco/lerobot-pi/datasets/water_plant \
    --language-instruction "water the plant" \
    --slice-yaml "{action_rotvec: [0, 6], state: [0, 16]}" \
    --selected-data-yaml "{action: action_rotvec, state: state, cameras: [ego_left, ego_right, front, wrist]}"

python scripts/process_lerobot_dataset.py \
    --input-path /data0/weizhi_zhao/dexjoco/raw_datasets/datasets_v2/bimanual_microwave_cook \
    --output-path /data0/weizhi_zhao/dexjoco/lerobot_datasets/datasets_v2/bimanual_microwave_cook \
    --language-instruction "Open the microwave door, place the food inside the microwave, close the door, and press the start button." \
    --slice-yaml "{state: [null, 46]}" \
    --selected-data-yaml "{cameras: [ego, wrist_left, wrist_right], action: action_rotvec, state: state}"
"""
