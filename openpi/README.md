# [OpenPI](https://github.com/Physical-Intelligence/openpi) π0.5 for DexJoCo

This directory contains the OpenPI π0.5 training and serving setup for DexJoCo
tasks. The configuration follows the DexJoCo evaluation protocol with two data
regimes:

- `rand-obj`: object placement and table height randomization.
- `rand-full`: `rand-obj` plus third-person camera, lighting, and table texture
  randomization.

## Environment

Create the Conda environment and install the local packages:

```bash
cd openpi
bash install.bash
conda activate openpi
```

The installation uses Conda to support non-sudo installation of `git-lfs` and
`ffmpeg`. The `lerobot` package is installed with `--no-deps` because this setup
only requires the LeRobot dataset interface.

## Checkpoints and Datasets

To download the π0.5 base checkpoint, first install the Google Cloud CLI by
following the instructions at
[https://docs.cloud.google.com/sdk/docs/downloads-interactive#linux-mac](https://docs.cloud.google.com/sdk/docs/downloads-interactive#linux-mac).
Then follow the [OpenPI README](https://github.com/Physical-Intelligence/openpi)
to download the checkpoint.

A typical command sequence is:
```bash
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud auth login
mkdir -p ./checkpoints
gcloud storage cp --recursive gs://openpi-assets/checkpoints/pi05_base ./checkpoints
```

To download the DexJoCo LeRobot datasets, use the Hugging Face Hub CLI:
```bash
curl -LsSf https://hf.co/cli/install.sh | bash
hf auth login
hf download DexJoCo/DexJoCo-Datasets-LeRobot --repo-type=dataset --local-dir ./datasets
```

Place the π0.5 base checkpoint and DexJoCo LeRobot datasets according to the
paths in [`config.yaml`](config.yaml). You can modify [`config.yaml`](config.yaml)
to use your own paths:

```yaml
pretrained_model_path: "../checkpoints/pi05_base/params"
pretrained_model_action_dim_44_path: "../checkpoints/pi05_base_action_dim_44/params"
dataset_root: "../datasets/dexjoco_lerobot_datasets"
rand_full_dataset_root: "../datasets/dexjoco_lerobot_datasets_rand_full"
ckpts_root: "../checkpoints/pi05_ckpts"
rand_full_ckpts_root: "../checkpoints/pi05_rand_full_ckpts"
```

The standard dataset root should contain one directory per DexJoCo task:

```text
../datasets/dexjoco_lerobot_datasets/
  bimanual_assembly/
  bimanual_hanoi/
  bimanual_microwave_cook/
  bimanual_photograph/
  bimanual_unlock_ipad/
  click_mouse/
  fold_glasses/
  hammer_nail/
  pick_bucket/
  pinch_tongs/
  water_plant/
```

The `rand-full` dataset root uses the same 11 task directory names:

```text
../datasets/dexjoco_lerobot_datasets_rand_full/
  bimanual_assembly/
  bimanual_hanoi/
  ...
  water_plant/
```

## Custom Datasets

Custom datasets must use the LeRobot dataset format expected by the DexJoCo
OpenPI data configs. Place each dataset under a task-specific directory and set
the corresponding root in [`config.yaml`](config.yaml):

```text
../datasets/my_lerobot_datasets/
  my_task/
```

The dataset must provide `observation.state`, `action`, and `prompt`.

For a single-arm task, the default image fields are:

```text
observation.images.front
observation.images.wrist
```

The `observation.images.front` field is configurable through `base_img_name` in
the corresponding `DexJoCoConfig` entry.

For a bimanual task, the default image fields are:

```text
observation.images.ego
observation.images.wrist_left
observation.images.wrist_right
```

For bimanual datasets that use different image keys, set `base_img_name`,
`wrist_left_img_name`, or `wrist_right_img_name` in the corresponding
`DexJoCoConfig` entry in
[`src/openpi/training/dexjoco_configs.py`](src/openpi/training/dexjoco_configs.py).

The default action dimensions are 22 for single-arm tasks and 44 for bimanual
tasks.

Register a custom task by adding a `DexJoCoConfig` entry:

```python
DexJoCoConfig(
    name="my_task",
    checkpoint_base_dir=f"{CKPTS_ROOT}",
    data_root=Path(f"{DATASET_ROOT}/my_task"),
    single_arm=True,
)
```

Use `single_arm=False` for bimanual datasets. Bimanual tasks use
`pretrained_model_action_dim_44_path`, so the 44-dimensional checkpoint must be
available before training.

After registering the task, compute normalization statistics and start training
with the custom config name:

```bash
cd openpi
conda activate openpi
python scripts/compute_norm_stats.py my_task --batch-size=64 --num-workers=16
python scripts/train.py my_task
```

## 44-Dimensional Checkpoint

Single-arm DexJoCo tasks use 22-dimensional actions. Bimanual tasks use
44-dimensional actions. Convert the π0.5 base checkpoint before training
bimanual tasks:

```bash
cd openpi
python scripts/convert_to_action_dim_44_model.py \
  --input-path ../checkpoints/pi05_base \
  --output-path ../checkpoints/pi05_base_action_dim_44
```

Set `pretrained_model_action_dim_44_path` in [`config.yaml`](config.yaml) to the
converted checkpoint's `params` directory.

## Normalization Statistics

Compute normalization statistics before training. For a single config:

```bash
cd openpi
python scripts/compute_norm_stats.py hammer_nail --batch-size=64 --num-workers=16
python scripts/compute_norm_stats.py bimanual_assembly --batch-size=64 --num-workers=16
python scripts/compute_norm_stats.py hammer_nail_rand_full --batch-size=64 --num-workers=16
```

For all DexJoCo configs:

```bash
cd openpi
bash scripts/compute_norm_stats.bash
```

The script computes statistics for the 11 standard DexJoCo datasets first, then
for the 11 `rand-full` datasets. The statistics are written under
`assets/<config_name>/local_repo`.

## Training

Multiple tasks can be launched in tmux sessions:

```bash
cd openpi
conda activate openpi
python scripts/launch_tmux_train.py \
  --config-names bimanual_assembly bimanual_unlock_ipad bimanual_microwave_cook bimanual_hanoi \
  --gpus 0,1 2,3 4,5 6,7
```

[`scripts/launch_tmux_train.py`](scripts/launch_tmux_train.py) supports the
following arguments:

| Argument | Required | Default | Description |
| --- | --- | --- | --- |
| `--config-names` | Yes | N/A | Space-separated training config names. Each config is launched in a separate tmux session. |
| `--gpus` | Yes | N/A | Space-separated GPU assignments matched one-to-one with `--config-names`. Use comma-separated IDs, such as `0,1`, to assign multiple GPUs to one task. |
| `--wandb-project` | No | `dexjoco-openpi` | Weights & Biases project name passed to [`scripts/train.py`](scripts/train.py). |
| `--wandb-mode` | No | `online` | Weights & Biases mode. Valid values are `online` and `offline`. |
| `--conda-env` | No | `openpi` | Conda environment activated inside each tmux session. |
| `--mem-fraction` | No | `0.9` | Value assigned to `XLA_PYTHON_CLIENT_MEM_FRACTION` for each training process. |
| `--nccl-p2p-disable` | No | `False` | Sets `NCCL_P2P_DISABLE=1` before launching training. |
| `--dry-run` | No | `False` | Prints the tmux session names and training commands without creating sessions. |

Train a single task by passing the config name to
[`scripts/train.py`](scripts/train.py):

```bash
cd openpi
conda activate openpi
python scripts/train.py hammer_nail
```

## Serve Policy

Serve a trained checkpoint through the WebSocket policy server:

```bash
cd openpi
conda activate openpi
python scripts/serve_policy.py policy:checkpoint \
  --policy.config hammer_nail \
  --policy.dir ../checkpoints/pi05_ckpts/hammer_nail/<exp_name>/<step>
```

For a `rand-full` checkpoint, use the corresponding config and checkpoint root:

```bash
cd openpi
conda activate openpi
python scripts/serve_policy.py policy:checkpoint \
  --policy.config hammer_nail_rand_full \
  --policy.dir ../checkpoints/pi05_rand_full_ckpts/hammer_nail_rand_full/<exp_name>/<step>
```

The server listens on port `8000` by default. Use `--port` to select a different
port.

## License and Notices

This repository is derived from OpenPI and is distributed under the Apache
License, Version 2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

Gemma-based model components and checkpoints are subject to the Gemma Terms of
Use. See [`LICENSE_GEMMA.txt`](LICENSE_GEMMA.txt). Checkpoints are not included
in this repository; users are responsible for obtaining π0.5/Gemma checkpoints
and using them under the applicable model license terms.
