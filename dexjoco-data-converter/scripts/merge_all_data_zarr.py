from dexjoco_data_converter.to_zarr.merge_datasets_zarr import merge_datasets


if __name__ == "__main__":
    import tyro

    tyro.cli(merge_datasets)

"""
python scripts/merge_all_data_zarr.py \
    --output=/data/weizhi_zhao/dexjoco/diffusion_policy/dataset_v2_merged \
    --dataset-paths-cfg-path=configs/configs_46/datasets_v2/datasets_path.yaml \
    --selected-data-rename-cfg-path=configs/configs_46/datasets_v2/selected_data_rename.yaml \
    --slice-cfg-path=configs/configs_46/datasets_v2/slice_config.yaml \
    --target-action-dim=44 \
    --target-state-dim=46
"""
