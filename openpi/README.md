# [OpenPI](https://github.com/Physical-Intelligence/openpi/tree/main) π0.5 for DexJoCo

This directory contains the OpenPI π0.5 training and serving setup for DexJoCo tasks. The configuration follows the DexJoCo evaluation protocol with two data regimes:

- `rand-obj`: object placement and table height randomization.
- `rand-full`: `rand-obj` plus third-person camera, lighting, and table texture randomization.

## Environment

Create the Conda environment and install the local packages:

```bash
cd openpi
bash install.bash
conda activate openpi
```

The installation uses Conda to support non-sudo installation of `git-lfs` and `ffmpeg`. The `lerobot` package is installed with `--no-deps` because this setup only requires the LeRobot dataset interface.

## Checkpoints and Datasets

Download links for the π0.5 base checkpoint and DexJoCo LeRobot datasets are provided here:

- π0.5 base checkpoint: TODO
- DexJoCo LeRobot datasets: TODO

Place the π0.5 base checkpoint and DexJoCo LeRobot datasets according to the paths in `config.yaml`:

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
  hammer_nail/
  click_mouse/
  pick_bucket/
  pinch_tongs/
  fold_glasses/
  water_plant/
  bimanual_unlock_ipad/
  bimanual_hanoi/
  bimanual_assembly/
  bimanual_microwave_cook/
  bimanual_photograph/
```

The `rand-full` dataset root uses the same 11 task directory names:

```text
../datasets/dexjoco_lerobot_datasets_rand_full/
  hammer_nail/
  click_mouse/
  ...
  bimanual_photograph/
```

## 44-Dimensional Checkpoint

Single-arm DexJoCo tasks use 22-dimensional actions. Bimanual tasks use 44-dimensional actions. Convert the π0.5 base checkpoint before training bimanual tasks:

```bash
cd openpi
python scripts/convert_to_action_dim_44_model.py \
  --input-path ../checkpoints/pi05_base \
  --output-path ../checkpoints/pi05_base_action_dim_44
```

Set `pretrained_model_action_dim_44_path` in `config.yaml` to the converted checkpoint's `params` directory.

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

The script computes statistics for the 11 standard DexJoCo datasets first, then for the 11 `rand-full` datasets. The statistics are written under `assets/<config_name>/local_repo`.

## Training

Multiple tasks can be launched in tmux sessions:

```bash
cd openpi
python scripts/launch_tmux_train.py \
  --config-names bimanual_assembly bimanual_unlock_ipad bimanual_microwave_cook bimanual_hanoi \
  --gpus 0,1 2,3 4,5 6,7 \
  --wandb-project dexjoco-openpi \
  --wandb-mode offline \
  --nccl-p2p-disable
```

Train a single task by passing the config name to `scripts/train.py`:

```bash
cd openpi
python scripts/train.py hammer_nail
python scripts/train.py bimanual_assembly
python scripts/train.py hammer_nail_rand_full
python scripts/train.py bimanual_assembly_rand_full
```

Use `--dry-run` to inspect the generated commands without creating tmux sessions.

## Serve Policy

Serve a trained checkpoint through the websocket policy server:

```bash
cd openpi
python scripts/serve_policy.py policy:checkpoint \
  --policy.config hammer_nail \
  --policy.dir ../checkpoints/pi05_ckpts/hammer_nail/<exp_name>/<step>
```

For a `rand-full` checkpoint, use the corresponding config and checkpoint root:

```bash
cd openpi
python scripts/serve_policy.py policy:checkpoint \
  --policy.config hammer_nail_rand_full \
  --policy.dir ../checkpoints/pi05_rand_full_ckpts/hammer_nail_rand_full/<exp_name>/<step>
```

The server listens on port `8000` by default. Use `--port` to select a different port.

## License and Notices

This repository is derived from OpenPI and is distributed under the Apache License, Version 2.0. See `LICENSE` and `NOTICE`.

Gemma-based model components and checkpoints are subject to the Gemma Terms of Use. See `LICENSE_GEMMA.txt`. Checkpoints are not included in this repository; users are responsible for obtaining π0.5/Gemma checkpoints and using them under the applicable model license terms.
