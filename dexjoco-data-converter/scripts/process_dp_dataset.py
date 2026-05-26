from pathlib import Path

import yaml

from dexjoco_data_converter.to_zarr.merge_episode_zarr import merge_episode


def main(
    input_path: Path,
    output_path: Path,
    slice_yaml: str,
    bad_episodes_yaml: str | None = None,
    skip_static_frames: bool = True,
) -> None:
    # Parse CLI-provided YAML strings into Python objects.
    slice_spec = yaml.safe_load(slice_yaml)
    bad_episodes = (
        None if bad_episodes_yaml is None else yaml.safe_load(bad_episodes_yaml)
    )
    assert isinstance(slice_spec, dict), "slice_yaml must be a YAML dict string"
    assert bad_episodes is None or isinstance(bad_episodes, list), (
        "bad_episodes_yaml must be a YAML list string"
    )

    merge_episode(
        input=input_path,
        output=output_path,
        slice_spec=slice_spec,
        bad_episodes=bad_episodes,
        skip_static_frames=skip_static_frames,
    )


if __name__ == "__main__":
    import tyro

    tyro.cli(main)


"""
python scripts/process_dp_dataset.py \
    --input-path /data/weizhi_zhao/dexjoco/dexjoco_raw_datasets/water_plant_replay_output \
    --output-path "/data/weizhi_zhao/dexjoco/diffusion_policy/datasets/water_plant_replay_output" \
    --slice-yaml "{obs_pos: [0, 3], action: [0, 6]}" \
    --bad-episodes-yaml "['episode_0003', 'episode_0012']" \
"""
