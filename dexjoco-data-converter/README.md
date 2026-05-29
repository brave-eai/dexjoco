# DexJoCo Data Converter

Convert DexJoCo raw datasets into LeRobot datasets and Zarr replay buffers (for
Diffusion Policy).

## Overview

DexJoCo Data Converter provides command-line tools for converting DexJoCo raw
datasets into training-ready dataset formats. It supports single-dataset
conversion, batch conversion over multiple datasets, and multi-task merging with
a shared action, state, and camera schema.

Supported output formats:

- LeRobot dataset
- Zarr replay buffer

Supported workflows:

- Single dataset conversion
- Parallel batch conversion over multiple datasets
- Multi-task merge into one dataset

## Installation

Install the converter with the provided setup script:

```bash
bash dexjoco-data-converter/install.bash
conda activate dexjoco-dc
```

The setup script is [`dexjoco-data-converter/install.bash`](./install.bash).

## Quick Example

Convert one DexJoCo raw dataset into a LeRobot dataset:

```bash
dexjoco-dc-single-lerobot \
    --input ./datasets/raw/dexjoco_raw_datasets/water_plant \
    --output ./converted_datasets/lerobot/single/water_plant \
    --language-instruction "Grasp the watering can and apply water to the plant." \
    --selected-data-yaml "{action: action_rotvec, state: state, cameras: {front: front, wrist: wrist}}" \
    --slice-yaml "{state: [null, 23]}"
```

## Dataset Layout

### Raw Dataset Layout

The input path passed to `--input` or listed in `dataset_paths.yaml` points to a
DexJoCo raw dataset directory. Each dataset contains episode directories with a
`replay.zarr` store and camera videos:

```text
datasets/raw/dexjoco_raw_datasets/
└── water_plant/
    ├── episode_000000/
    │   ├── replay.zarr/
    │   └── videos/
    ├── episode_000001/
    │   ├── replay.zarr/
    │   └── videos/
    └── ...
```

### Output Dataset Layout

The output directory depends on the selected format and workflow:

```text
converted_datasets/
├── lerobot/
│   ├── single/
│   ├── rand_obj/
│   ├── rand_full/
│   └── multi_task/
└── zarr/
    ├── single/
    ├── rand_obj/
    ├── rand_full/
    └── multi_task/
```

LeRobot outputs follow the LeRobot dataset layout. Zarr outputs contain a merged
`replay_buffer.zarr` store and a `videos/` directory.

## Configuration Files

Configuration files are stored under
[`dexjoco-data-converter/configs/`](./configs/). The repository provides
configuration groups for [`rand_obj`](./configs/rand_obj/),
[`rand_full`](./configs/rand_full/), and [`multi_task`](./configs/multi_task/).

| File                         | Description                                                                            |
| ---------------------------- | -------------------------------------------------------------------------------------- |
| `dataset_paths.yaml`         | Raw dataset paths processed by batch conversion or multi-task merge.                   |
| `language_instructions.yaml` | Per-dataset language instructions for LeRobot outputs.                                 |
| `selected_data.yaml`         | Per-dataset action, state, and camera mappings.                                        |
| `slice_config.yaml`          | Per-dataset array slicing rules, commonly used to remove privileged state information. |

For example, the multi-task configuration files are:

- [`dexjoco-data-converter/configs/multi_task/dataset_paths.yaml`](./configs/multi_task/dataset_paths.yaml)
- [`dexjoco-data-converter/configs/multi_task/language_instructions.yaml`](./configs/multi_task/language_instructions.yaml)
- [`dexjoco-data-converter/configs/multi_task/selected_data.yaml`](./configs/multi_task/selected_data.yaml)
- [`dexjoco-data-converter/configs/multi_task/slice_config.yaml`](./configs/multi_task/slice_config.yaml)

### Selected Data

The selected data configuration maps raw dataset fields to the normalized schema
used by converter commands that accept `--selected-data-yaml` or
`--selected-data-cfg-path`. A typical entry selects one action key, one state
key, and a camera mapping:

```yaml
water_plant:
  action: action_rotvec
  state: state
  cameras:
    front: front
    wrist: wrist
```

For `action` and `state`, the left side is the converter's logical field name
and the right side is the raw array key loaded from `replay.zarr`:

```text
converter logical field name -> raw replay.zarr array key
```

For example, `action: action_rotvec` reads the raw `action_rotvec` array and
uses it as the selected action data. `state: state` reads the raw `state` array
and uses it as the selected state data.

The actual output keys depend on the target format:

- Zarr writes the selected arrays as `action` and `state`.
- LeRobot writes them through the LeRobot constants `ACTION` and `OBS_STATE`,
  whose values are `action` and `observation.state`.

For multi-task conversion, the selected action and state arrays are sliced and
padded to the configured target dimensions.

The `cameras` mapping is:

```text
LeRobot output camera name -> raw video name
```

For example, `front: front` reads `videos/front.mp4` and writes it as
`observation.images.front`. The output camera name is the key used by downstream
LeRobot/OpenPI training code, while the raw video name is the source video stem
under each raw episode's `videos/` directory. One raw video can be mapped to
more than one output camera name when a downstream schema requires duplicated
views.

### Slice Configuration

The slice configuration controls which dimensions are kept from selected arrays.
It is commonly used to remove privileged information from `state` before writing
the training dataset.

Each value is a list of Python slice arguments. The converter unpacks the list
with `slice(*value)`, so the entries represent `start`, `stop`, and `step`:

```text
[start, stop, step] -> slice(start, stop, step)
```

The `step` entry can be omitted. YAML `null` is converted to Python `None`. For
example, `[null, 23]` becomes `slice(None, 23)`, which keeps dimensions before
index `23` and removes the remaining dimensions:

```yaml
water_plant:
  state: [null, 23]
```

## Usage

The converter exposes six CLI commands.

| Command                     | Purpose                                                           |
| --------------------------- | ----------------------------------------------------------------- |
| `dexjoco-dc-single-lerobot` | Convert one raw dataset to one LeRobot dataset.                   |
| `dexjoco-dc-single-zarr`    | Convert one raw dataset to one Zarr replay buffer.                |
| `dexjoco-dc-batch-lerobot`  | Convert multiple raw datasets to LeRobot datasets in parallel.    |
| `dexjoco-dc-batch-zarr`     | Convert multiple raw datasets to Zarr replay buffers in parallel. |
| `dexjoco-dc-merge-lerobot`  | Merge multiple raw datasets into one LeRobot dataset.             |
| `dexjoco-dc-merge-zarr`     | Merge multiple raw datasets into one Zarr replay buffer.          |

### Single Dataset Conversion

Convert one DexJoCo raw dataset into a LeRobot dataset:

```bash
dexjoco-dc-single-lerobot \
    --input ./datasets/raw/dexjoco_raw_datasets/water_plant \
    --output ./converted_datasets/lerobot/single/water_plant \
    --language-instruction "Grasp the watering can and apply water to the plant." \
    --selected-data-yaml "{action: action_rotvec, state: state, cameras: {front: front, wrist: wrist}}" \
    --slice-yaml "{state: [null, 23]}"
```

Convert one DexJoCo raw dataset into a Zarr replay buffer:

```bash
dexjoco-dc-single-zarr \
    --input ./datasets/raw/dexjoco_raw_datasets/water_plant \
    --output ./converted_datasets/zarr/single/water_plant \
    --slice-yaml "{state: [null, 23]}"
```

| Option                   | Applies to    | Description                                                      |
| ------------------------ | ------------- | ---------------------------------------------------------------- |
| `--input`                | LeRobot, Zarr | Raw dataset directory containing episode folders.                |
| `--output`               | LeRobot, Zarr | Output dataset directory. The directory must be empty or absent. |
| `--language-instruction` | LeRobot       | Task instruction written to each output frame.                   |
| `--selected-data-yaml`   | LeRobot       | YAML string that selects action, state, and camera mappings.     |
| `--slice-yaml`           | LeRobot, Zarr | YAML string that defines array slicing rules.                    |
| `--bad-episodes-yaml`    | LeRobot, Zarr | Optional YAML list of episode directory names to exclude.        |
| `--skip-static-frames`   | LeRobot, Zarr | Whether to remove static leading frames from each episode.       |
| `--silent`               | LeRobot       | Whether to suppress worker stdout and stderr.                    |

### Batch Conversion

Batch conversion reads dataset paths and per-dataset settings from YAML
configuration files, then runs independent dataset conversions in parallel. Each
worker processes one dataset and writes its result under the output parent
directory.

Convert multiple datasets into LeRobot datasets in parallel:

```bash
dexjoco-dc-batch-lerobot \
    --output ./converted_datasets/lerobot/rand_obj \
    --dataset-paths-cfg-path dexjoco-data-converter/configs/rand_obj/dataset_paths.yaml \
    --language-instruction-cfg-path dexjoco-data-converter/configs/rand_obj/language_instructions.yaml \
    --selected-data-cfg-path dexjoco-data-converter/configs/rand_obj/selected_data.yaml \
    --slice-cfg-path dexjoco-data-converter/configs/rand_obj/slice_config.yaml \
    --num-workers 6
```

Convert multiple datasets into Zarr replay buffers in parallel:

```bash
dexjoco-dc-batch-zarr \
    --output ./converted_datasets/zarr/rand_obj \
    --dataset-paths-cfg-path dexjoco-data-converter/configs/rand_obj/dataset_paths.yaml \
    --slice-cfg-path dexjoco-data-converter/configs/rand_obj/slice_config.yaml \
    --num-workers 6
```

| Option                            | Applies to    | Description                                                          |
| --------------------------------- | ------------- | -------------------------------------------------------------------- |
| `--output`                        | LeRobot, Zarr | Parent directory where parallel workers write dataset outputs.       |
| `--dataset-paths-cfg-path`        | LeRobot, Zarr | YAML file containing raw dataset paths.                              |
| `--language-instruction-cfg-path` | LeRobot       | YAML file containing per-dataset task instructions.                  |
| `--selected-data-cfg-path`        | LeRobot       | YAML file containing per-dataset action, state, and camera mappings. |
| `--slice-cfg-path`                | LeRobot, Zarr | YAML file containing per-dataset slicing rules.                      |
| `--num-workers`                   | LeRobot, Zarr | Maximum number of dataset workers.                                   |
| `--bad-episodes-cfg-path`         | LeRobot, Zarr | Optional YAML file containing excluded episodes per dataset.         |
| `--skip-static-frames`            | LeRobot, Zarr | Whether to remove static leading frames from each episode.           |
| `--silent`                        | LeRobot       | Whether to suppress worker stdout and stderr.                        |

### Multi-task Merge

Multi-task merge combines multiple datasets into one output dataset with a
shared schema. The target action and state dimensions define the unified feature
size after selection, slicing, and padding.

Merge multiple datasets into one LeRobot dataset:

```bash
dexjoco-dc-merge-lerobot \
    --output ./converted_datasets/lerobot/multi_task \
    --dataset-paths-cfg-path dexjoco-data-converter/configs/multi_task/dataset_paths.yaml \
    --language-instruction-cfg-path dexjoco-data-converter/configs/multi_task/language_instructions.yaml \
    --selected-data-cfg-path dexjoco-data-converter/configs/multi_task/selected_data.yaml \
    --slice-cfg-path dexjoco-data-converter/configs/multi_task/slice_config.yaml \
    --target-action-dim 44 \
    --target-state-dim 46
```

Merge multiple datasets into one Zarr replay buffer:

```bash
dexjoco-dc-merge-zarr \
    --output ./converted_datasets/zarr/multi_task \
    --dataset-paths-cfg-path dexjoco-data-converter/configs/multi_task/dataset_paths.yaml \
    --selected-data-cfg-path dexjoco-data-converter/configs/multi_task/selected_data.yaml \
    --slice-cfg-path dexjoco-data-converter/configs/multi_task/slice_config.yaml \
    --target-action-dim 44 \
    --target-state-dim 46
```

| Option                            | Applies to    | Description                                                          |
| --------------------------------- | ------------- | -------------------------------------------------------------------- |
| `--output`                        | LeRobot, Zarr | Output directory for the merged dataset.                             |
| `--dataset-paths-cfg-path`        | LeRobot, Zarr | YAML file containing raw dataset paths.                              |
| `--language-instruction-cfg-path` | LeRobot       | YAML file containing per-dataset task instructions.                  |
| `--selected-data-cfg-path`        | LeRobot, Zarr | YAML file containing per-dataset action, state, and camera mappings. |
| `--slice-cfg-path`                | LeRobot, Zarr | YAML file containing per-dataset slicing rules.                      |
| `--target-action-dim`             | LeRobot, Zarr | Unified action dimension after padding.                              |
| `--target-state-dim`              | LeRobot, Zarr | Unified state dimension after padding.                               |
| `--bad-episodes-cfg-path`         | LeRobot, Zarr | Optional YAML file containing excluded episodes per dataset.         |
| `--skip-static-frames`            | LeRobot, Zarr | Whether to remove static leading frames from each episode.           |
| `--silent`                        | LeRobot       | Whether to suppress worker stdout and stderr.                        |

### Commands Without Selected Data

The single-dataset and batch Zarr commands do not take `selected_data`:

- `dexjoco-dc-single-zarr`
- `dexjoco-dc-batch-zarr`

These commands preserve the raw episode arrays from `replay.zarr` with minimal
normalization. If an episode contains `action_rotvec`, it is renamed to
`action`; otherwise the existing `action` key is used. The `state` key is used
directly. The `slice_yaml` or `slice_config.yaml` rules are applied to the
resulting array keys, such as `action` or `state`.

For videos, these commands do not select or rename cameras through
`selected_data`. They enumerate the video files from the first episode, copy or
trim all listed videos for each episode, and write numeric output video names
such as `0.mp4`, `1.mp4`, and `2.mp4`. The generated `video_name_map.yaml`
records the mapping from raw video file names to numeric output names in the
output dataset directory.

The multi-task Zarr command, `dexjoco-dc-merge-zarr`, does take `selected_data`.
It uses the selected action/state keys, enforces a shared camera schema across
datasets, and writes selected videos according to the configured camera mapping.

For complete command examples, see
[`dexjoco-data-converter/example.bash`](./example.bash).

## OpenPI Integration

OpenPI reads LeRobot image fields through a repack transform before applying the
single-arm or dual-arm policy transforms. The converter should write camera
names that either match the OpenPI defaults or are explicitly configured in
OpenPI.

### Expected Camera Names

For single-arm tasks, the default LeRobot image fields are:

```text
observation.images.front
observation.images.wrist
```

This corresponds to a selected data mapping such as:

```yaml
water_plant:
  action: action_rotvec
  state: state
  cameras:
    front: front
    wrist: wrist
```

For dual-arm tasks, the default LeRobot image fields are:

```text
observation.images.ego
observation.images.wrist_left
observation.images.wrist_right
```

This corresponds to a selected data mapping such as:

```yaml
bimanual_photograph:
  action: action_rotvec
  state: state
  cameras:
    ego: ego
    wrist_left: wrist_left
    wrist_right: wrist_right
```

The multi-task OpenPI configuration in this repository uses a unified camera
schema:

```text
observation.images.base
observation.images.wrist1
observation.images.wrist2
```

That schema is useful when single-arm and dual-arm datasets are merged into one
dataset. For example, a single-arm task can duplicate the same raw wrist video
for both wrist slots:

```yaml
water_plant:
  action: action_rotvec
  state: state
  cameras:
    base: front
    wrist1: wrist
    wrist2: wrist
```

### Custom Camera Names

OpenPI camera names are customized in
[`openpi/src/openpi/training/dexjoco_configs.py`](../openpi/src/openpi/training/dexjoco_configs.py).
Set `base_img_name`, `wrist_left_img_name`, or `wrist_right_img_name` in the
corresponding `DexJoCoConfig` entry to point OpenPI to a different LeRobot image
field:

```python
DexJoCoConfig(
    name="my_task",
    checkpoint_base_dir=f"{CKPTS_ROOT}",
    data_root=Path(f"{DATASET_ROOT}/my_task"),
    single_arm=True,
    base_img_name="observation.images.random_camera",
)
```

The selected data file should then use `random_camera` as the output camera name
and map it to the raw video stem that should provide that view:

```yaml
my_task:
  action: action_rotvec
  state: state
  cameras:
    random_camera: random_camera
    wrist: wrist
```

## Validation

Check raw episode consistency for all datasets in a root directory:

```bash
python dexjoco-data-converter/scripts/check_dataset_consistency.py \
    --input-root ./datasets/raw/dexjoco_raw_datasets
```

The script is
[`dexjoco-data-converter/scripts/check_dataset_consistency.py`](./scripts/check_dataset_consistency.py).

Print the structure of one raw episode:

```bash
python dexjoco-data-converter/scripts/print_episode_structure.py \
    --dataset-path ./datasets/raw/dexjoco_raw_datasets/water_plant \
    --episode-index 0
```

The script is
[`dexjoco-data-converter/scripts/print_episode_structure.py`](./scripts/print_episode_structure.py).

Check LeRobot datasets under a root directory:

```bash
python dexjoco-data-converter/scripts/check_lerobot_datasets.py \
    --input-root ./converted_datasets/lerobot/rand_full
```

The script is
[`dexjoco-data-converter/scripts/check_lerobot_datasets.py`](./scripts/check_lerobot_datasets.py).

## Implementation Notes

### Processing Pipeline

The converter loads action and state arrays from `replay.zarr`, normalizes array
shapes, removes static leading frames when requested, applies the configured
slices, and writes the result to the selected target format. LeRobot conversion
also writes task instructions into the LeRobot dataset schema.

In multi-task merge, the converter applies per-dataset key selection and
slicing, then pads action and state arrays to `--target-action-dim` and
`--target-state-dim` so all tasks share one output schema.

### Multiprocessing

Batch conversion uses a process pool to run independent dataset conversions in
parallel. The main process builds dataset jobs from configuration files,
receives progress messages from workers, and reports dataset-level completion or
failure.

LeRobot episode conversion also uses separate video decoder processes. Each
decoder reads one raw camera video and streams decoded frames through a
multiprocessing queue. The main conversion loop receives one frame per selected
camera at each timestep, verifies frame synchronization, builds a LeRobot
`frame_data` dictionary, and calls `dataset.add_frame(frame_data)`.

This design allows video decoding and LeRobot video encoding to proceed at the
same time: decoder processes read and decode source videos while the main
conversion process adds frames to the LeRobot dataset, where the output video
writer performs encoding.

### Video Trimming

Zarr conversion preserves the external video files used by the raw dataset. When
static leading frames are skipped,
[`trim_video`](./src/dexjoco_data_converter/to_zarr/merge_episode_zarr.py) opens
an `imageio` reader for the source video and an `imageio` writer for the
destination video. The function iterates through decoded frames, skips frames
before `start_frame`, and appends the remaining frames to the writer.

The reader and writer are active in the same loop, so video decoding and
encoding are performed as a streaming operation instead of loading the full
video into memory.
