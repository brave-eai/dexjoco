#!/usr/bin/env bash
set -euo pipefail


# ---------------------------------------------------------------------------
# CLI: single dataset conversion
# ---------------------------------------------------------------------------

# Convert one Water Plant dataset to LeRobot.
dexjoco-dc-single-lerobot \
    --input ./datasets/raw/dexjoco_raw_datasets/water_plant \
    --output ./converted_datasets/lerobot/single/water_plant \
    --language-instruction "Grasp the watering can and apply water to the plant." \
    --selected-data-yaml "{action: action_rotvec, state: state, cameras: {front: front, wrist: wrist}}" \
    --slice-yaml "{state: [null, 23]}"

# Convert one Water Plant dataset to Zarr.
dexjoco-dc-single-zarr \
    --input ./datasets/raw/dexjoco_raw_datasets/water_plant \
    --output ./converted_datasets/zarr/single/water_plant \
    --slice-yaml "{state: [null, 23]}"


# ---------------------------------------------------------------------------
# CLI: batch conversion
# ---------------------------------------------------------------------------

# Convert each dataset listed in rand_obj/dataset_paths.yaml to a separate
# LeRobot dataset.
dexjoco-dc-batch-lerobot \
    --output ./converted_datasets/lerobot/rand_obj \
    --dataset-paths-cfg-path dexjoco-data-converter/configs/rand_obj/dataset_paths.yaml \
    --language-instruction-cfg-path dexjoco-data-converter/configs/rand_obj/language_instructions.yaml \
    --selected-data-cfg-path dexjoco-data-converter/configs/rand_obj/selected_data.yaml \
    --slice-cfg-path dexjoco-data-converter/configs/rand_obj/slice_config.yaml \
    --num-workers 6

# Convert each dataset listed in rand_obj/dataset_paths.yaml to a separate
# Zarr dataset.
dexjoco-dc-batch-zarr \
    --output ./converted_datasets/zarr/rand_obj \
    --dataset-paths-cfg-path dexjoco-data-converter/configs/rand_obj/dataset_paths.yaml \
    --slice-cfg-path dexjoco-data-converter/configs/rand_obj/slice_config.yaml \
    --num-workers 6

# Convert each dataset listed in rand_full/dataset_paths.yaml to a separate
# LeRobot dataset.
dexjoco-dc-batch-lerobot \
    --output ./converted_datasets/lerobot/rand_full \
    --dataset-paths-cfg-path dexjoco-data-converter/configs/rand_full/dataset_paths.yaml \
    --language-instruction-cfg-path dexjoco-data-converter/configs/rand_full/language_instructions.yaml \
    --selected-data-cfg-path dexjoco-data-converter/configs/rand_full/selected_data.yaml \
    --slice-cfg-path dexjoco-data-converter/configs/rand_full/slice_config.yaml \
    --num-workers 3

# Convert each dataset listed in rand_full/dataset_paths.yaml to a separate
# Zarr dataset.
dexjoco-dc-batch-zarr \
    --output ./converted_datasets/zarr/rand_full \
    --dataset-paths-cfg-path dexjoco-data-converter/configs/rand_full/dataset_paths.yaml \
    --slice-cfg-path dexjoco-data-converter/configs/rand_full/slice_config.yaml \
    --num-workers 3


# ---------------------------------------------------------------------------
# CLI: multi-task dataset merge
# ---------------------------------------------------------------------------

# Merge all datasets listed in multi_task/dataset_paths.yaml into one LeRobot
# dataset with a shared action/state/camera schema.
dexjoco-dc-merge-lerobot \
    --output ./converted_datasets/lerobot/multi_task \
    --dataset-paths-cfg-path dexjoco-data-converter/configs/multi_task/dataset_paths.yaml \
    --language-instruction-cfg-path dexjoco-data-converter/configs/multi_task/language_instructions.yaml \
    --selected-data-cfg-path dexjoco-data-converter/configs/multi_task/selected_data.yaml \
    --slice-cfg-path dexjoco-data-converter/configs/multi_task/slice_config.yaml \
    --target-action-dim 44 \
    --target-state-dim 46

# Merge all datasets listed in multi_task/dataset_paths.yaml into one Zarr
# replay buffer with a shared action/state/camera schema.
dexjoco-dc-merge-zarr \
    --output ./converted_datasets/zarr/multi_task \
    --dataset-paths-cfg-path dexjoco-data-converter/configs/multi_task/dataset_paths.yaml \
    --selected-data-cfg-path dexjoco-data-converter/configs/multi_task/selected_data.yaml \
    --slice-cfg-path dexjoco-data-converter/configs/multi_task/slice_config.yaml \
    --target-action-dim 44 \
    --target-state-dim 46


# ---------------------------------------------------------------------------
# Utility scripts
# ---------------------------------------------------------------------------

# Check raw episode consistency for all datasets in a root directory.
python dexjoco-data-converter/scripts/check_dataset_consistency.py \
    --input-root ./datasets/raw/dexjoco_raw_datasets

# Print one raw episode structure.
python dexjoco-data-converter/scripts/print_episode_structure.py \
    --dataset-path ./datasets/raw/dexjoco_raw_datasets/water_plant \
    --episode-index 0

# Check every LeRobot dataset under a root directory.
python dexjoco-data-converter/scripts/check_lerobot_datasets.py \
    --input-root ./datasets/dexjoco_lerobot_datasets_rand_full
