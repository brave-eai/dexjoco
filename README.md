<p align="center">
  <img src="docs/pics/dexjoco_logo.jpg" alt="dexjoco logo" height="110">
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2605.16257"><img src="https://img.shields.io/badge/arXiv-2605.16257-b31b1b?style=flat-square" alt="arXiv"></a>
  <a href="https://dexjoco.github.io/"><img src="https://img.shields.io/badge/GitHub-Page-FF6B00?style=flat-square&logo=github&logoColor=white" alt="Project Homepage"></a>
  <a href="https://huggingface.co/datasets/DexJoCo/DexJoCo-Datasets-LeRobot"><img src="https://img.shields.io/badge/🤗%20HF-Dataset-FFD21E?style=flat-square" alt="HF Dataset"></a>
  <a href="https://huggingface.co/DexJoCo/DexJoCo-Pi05"><img src="https://img.shields.io/badge/🤗%20HF-Model-FFD21E?style=flat-square" alt="HF Models"></a>
  <a href="https://huggingface.co/papers/2605.16257"><img src="https://img.shields.io/badge/🤗%20HF-Paper-FFD21E?style=flat-square" alt="HF Paper"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License"></a>
</p>

---

DexJoCo is a MuJoCo-based simulation benchmark and toolkit for task-oriented
dexterous manipulation. It provides **11 functionally grounded tasks** covering
🛠️ **tool use**, 🤝 **bimanual coordination**, ⏱️ **long-horizon execution**,
and ⚙️ **reasoning**, together with a low-cost teleoperation data collection
system, replayable demonstrations, domain randomization, and OpenPI π0.5 policy
training/evaluation support.

<table width="100%">
  <tr>
    <td align="center" width="20%">
      <img src="docs/pics/photograph_4.png" alt="Bimanual photograph" width="100%">
      <br><sub>Bimanual photograph</sub>
    </td>
    <td align="center" width="20%">
      <img src="docs/pics/ipad_2.png" alt="Bimanual unlock iPad" width="100%">
      <br><sub>Bimanual unlock iPad</sub>
    </td>
    <td align="center" width="20%">
      <img src="docs/pics/assembly_3.png" alt="Bimanual assembly" width="100%">
      <br><sub>Bimanual assembly</sub>
    </td>
    <td align="center" width="20%">
      <img src="docs/pics/hanoi_3.png" alt="Bimanual hanoi" width="100%">
      <br><sub>Bimanual hanoi</sub>
    </td>
    <td align="center" width="20%">
      <img src="docs/pics/microwave_2.png" alt="Bimanual microwave" width="100%">
      <br><sub>Bimanual microwave</sub>
    </td>
  </tr>
</table>

<table width="100%">
  <tr>
    <td align="center" width="16.66%">
      <img src="docs/pics/water_plant_4.png" alt="Water plant" width="100%">
      <br><sub>Water plant</sub>
    </td>
    <td align="center" width="16.66%">
      <img src="docs/pics/hammer_3.png" alt="Hammer nail" width="100%">
      <br><sub>Hammer nail</sub>
    </td>
    <td align="center" width="16.66%">
      <img src="docs/pics/glass_1.png" alt="Fold glasses" width="100%">
      <br><sub>Fold glasses</sub>
    </td>
    <td align="center" width="16.66%">
      <img src="docs/pics/tongs_3.png" alt="Pinch tongs" width="100%">
      <br><sub>Pinch tongs</sub>
    </td>
    <td align="center" width="16.66%">
      <img src="docs/pics/bucket_4.png" alt="Pick bucket" width="100%">
      <br><sub>Pick bucket</sub>
    </td>
    <td align="center" width="16.66%">
      <img src="docs/pics/mouse_3.png" alt="Click mouse" width="100%">
      <br><sub>Click mouse</sub>
    </td>
  </tr>
</table>

## Table of Contents

- [Installation](#-installation)
- [Policy Evaluation](#-policy-evaluation)
- [Custom Policy Integration](#-custom-policy-integration)
- [Data Collection](#-data-collection)
- [Demonstration Replay](#-demonstration-replay)
- [Data Format](#data-format)
- [Policy Training](#policy-training)
- [Headless Rendering](#headless-rendering)
- [License](#-license)
- [Citation](#-citation)

## 🚀 Installation

Create and activate the DexJoCo environment:

```bash
conda env create -f environment-dexjoco.yaml
conda activate dexjoco
```

OpenPI training and serving environment

```bash
cd openpi
bash install.bash
conda activate openpi
```

## 🤖 Policy Evaluation

Download model checkpoints from
[DexJoCo-Pi05](https://huggingface.co/DexJoCo/DexJoCo-Pi05) or train them
yourself before evaluation.

Start an OpenPI policy server from the `openpi` environment:

```bash
cd openpi
conda activate openpi
python scripts/serve_policy.py --port=8000 policy:checkpoint \
  --policy.config water_plant \
  --policy.dir ../checkpoints/pi05_ckpts/water_plant/<exp_name>/<step>
```

Run evaluation from the repository root in the `dexjoco` environment:

```bash
conda activate dexjoco
dexjoco-openpi-eval \
  --config=./configs/rand_obj/water_plant.yaml \
  --seed=0 \
  --port=8000
```

For `rand_full` evaluation, use a config under `configs/rand_full/` and pass
`--rand-full`:

```bash
dexjoco-openpi-eval \
  --config=./configs/rand_full/water_plant.yaml \
  --seed=0 \
  --port=8000 \
  --rand-full
```

Convenience launch templates are available at
[`scripts/serve_pi05.bash`](scripts/serve_pi05.bash) and
[`scripts/evaluate_pi05.bash`](scripts/evaluate_pi05.bash).

`dexjoco-openpi-eval` options:

| Option                            | Default        | Description                                                        |
| --------------------------------- | -------------- | ------------------------------------------------------------------ |
| `--config PATH`                   | Required       | Evaluation YAML under `configs/rand_obj/` or `configs/rand_full/`. |
| `--seed INT`                      | `0`            | Random seed for NumPy and Python random state.                     |
| `--rand-full`                     | `False`        | Enables the `rand_full` evaluation regime.                         |
| `--randomize-dynamics`            | `False`        | Enables dynamics randomization.                                    |
| `--port INT`                      | `8000`         | OpenPI websocket policy server port.                               |
| `--output PATH`                   | Auto-generated | Output directory for videos and success-rate marker files.         |
| `--render-mode {rgb_array,human}` | `rgb_array`    | DexJoCo rendering mode. `rgb_array` is headless.                   |
| `--episodes INT`                  | `50`           | Number of evaluation episodes to run.                              |

See
[`dexjoco/dexjoco_openpi_client/eval_dexjoco_openpi.py`](dexjoco/dexjoco_openpi_client/eval_dexjoco_openpi.py)
for the complete option set.

## 🔌 Custom Policy Integration

DexJoCo supports custom policy evaluation through the same environment contract
used by the OpenPI client. Observations are collected from the simulator and
passed to a policy for action inference. The resulting actions are executed in
the environment.

Custom integrations should follow the protocol described in
[`docs/custom_policy_integration.md`](docs/custom_policy_integration.md),
including:

- observation fields for camera images, proprioceptive state, and prompts
- action layout conversion from rotation-vector policy actions to quaternion
  environment actions
- chunked action execution and replanning for latency-tolerant inference
- optional multi-frame observation history
- LeRobot-style `async_inference` integration patterns

## 📦 Data Collection

Please refer to the [`teleoperation/`](teleoperation/) directory for the
hardware and software configuration required for teleoperation:

| Component                   | Documentation                                                                                        |
| --------------------------- | ---------------------------------------------------------------------------------------------------- |
| teleoperation overview      | [`teleoperation/README.md`](teleoperation/README.md)                                                 |
| hardware setup              | [`teleoperation/Teleoperation_System_Tutorial.pdf`](teleoperation/Teleoperation_System_Tutorial.pdf) |
| Vive tracker bridge         | [`teleoperation/vive_bridge`](teleoperation/vive_bridge)                                             |
| Rokoko hand-keypoint bridge | [`teleoperation/rokoko`](teleoperation/rokoko)                                                       |
| GeoRT hand retargeting      | [`teleoperation/GeoRT`](teleoperation/GeoRT)                                                         |

Supported tasks:

| Task           | Setup      | Task Name                 |
| -------------- | ---------- | ------------------------- |
| Unlock iPad    | Bimanual   | `bimanual_unlock_ipad`    |
| Hanoi          | Bimanual   | `bimanual_hanoi`          |
| Assembly       | Bimanual   | `bimanual_assembly`       |
| Microwave cook | Bimanual   | `bimanual_microwave_cook` |
| Photograph     | Bimanual   | `bimanual_photograph`     |
| Hammer nail    | Single-arm | `hammer_nail`             |
| Click mouse    | Single-arm | `click_mouse`             |
| Pick bucket    | Single-arm | `pick_bucket`             |
| Pinch tongs    | Single-arm | `pinch_tongs`             |
| Fold glasses   | Single-arm | `fold_glasses`            |
| Water plant    | Single-arm | `water_plant`             |

TODO: how to add new tasks?

Start demonstration recording from the repository root with
[`scripts/record_demos_zarr.py`](scripts/record_demos_zarr.py):

```bash
conda activate dexjoco
python scripts/record_demos_zarr.py \
  --exp_name=water_plant \
  --successes_needed=20 \
  --randomize=True \
  --out_dir=./demos
```

The script opens the selected DexJoCo task, records successful teleoperation
episodes, and writes each success as a replayable Zarr episode with camera
videos.

Common `record_demos_zarr.py` options:

| Flag                 | Purpose                                                                       |
| -------------------- | ----------------------------------------------------------------------------- |
| `--exp_name`         | Selects one of the task names.                                                |
| `--successes_needed` | Stops collection after the requested number of successful demos.              |
| `--randomize`        | Enables the `rand_full` visual randomization regime used for data collection. |
| `--show_sim_cameras` | Displays camera streams during interactive collection.                        |
| `--save_depth`       | Saves depth arrays and depth videos alongside RGB videos.                     |
| `--out_dir`          | Selects the output directory for collected demos.                             |

[`scripts/record_demos_zarr.py`](scripts/record_demos_zarr.py) supports
`--camera_screen_effect` to display a camera viewfinder overlay, defaults to
`False`.

## 🔁 Demonstration Replay

Raw DexJoCo datasets for replay are available from
[`DexJoCo/DexJoCo-Datasets-Raw`](https://huggingface.co/datasets/DexJoCo/DexJoCo-Datasets-Raw).

Replay recorded demonstrations with
[`scripts/replay_demos_zarr.py`](scripts/replay_demos_zarr.py):

```bash
conda activate dexjoco
python scripts/replay_demos_zarr.py \
  --exp_name=water_plant \
  --input_dir=./demos \
  --out_dir=./replay_output \
  --randomize=True \
  --restore_state=True
```

Replay runs through the policy interface, can restore recorded initial object
poses and table height, and can generate `rand_full` visual variants through
camera, lighting, and table texture randomization.

Common `replay_demos_zarr.py` options:

| Flag           | Default           | Purpose                                                                                                 |
| -------------- | ----------------- | ------------------------------------------------------------------------------------------------------- |
| `--exp_name`   | `water_plant`     | Selects the task used to replay the demonstrations.                                                     |
| `--input_dir`  | `./`              | Directory containing recorded demo folders with `replay.zarr`.                                          |
| `--out_dir`    | `./replay_output` | Output directory for replayed Zarr episodes and videos.                                                 |
| `--randomize`  | `True`            | Enables replay-time `rand_full` visual randomization with preset camera, lighting, and texture changes. |
| `--seed`       | `0`               | Base replay seed; the demo index is added for each input demo.                                          |
| `--save_depth` | `False`           | Saves depth arrays and depth videos alongside RGB replay videos.                                        |

See [`scripts/replay_demos_zarr.py`](scripts/replay_demos_zarr.py) for the
complete option set.

[`scripts/replay_demos_zarr.py`](scripts/replay_demos_zarr.py) supports
`--camera_screen_effect` to display a camera viewfinder overlay, defaults to
`False`.

<a id="data-format"></a>

## 🗂️ Data Format

Each successful demonstration is written as:

```text
<out_dir>/<exp_name>_demo_<index>_<timestamp>/
  replay.zarr/
  videos/<camera_key>.mp4
  videos/<camera_key>_depth.npz
  videos/<camera_key>_depth.mp4
```

Depth outputs are present when `--save_depth=True`.

The Zarr replay buffer stores low-dimensional episode data:

| Field           | Description                                                                                     |
| --------------- | ----------------------------------------------------------------------------------------------- |
| `action`        | Recorded policy or teleoperation actions.                                                       |
| `action_rotvec` | Action representation with orientation stored as rotation vectors when conversion is available. |
| `timestamp`     | Per-step timestamps derived from `--data_fps`.                                                  |
| `state`         | Proprioceptive and task state used by replay and state restoration when available.              |

TODO: data converter

### Policy Training Action and State Layout

For bimanual demonstrations, the recorded action layout is:

```text
[r_pose7, r_hand16, l_pose7, l_hand16]
```

The policy-mode DexJoCo environment expects the flat action layout:

```text
[r_pose7, l_pose7, r_hand16, l_hand16]
```

During OpenPI evaluation,
[`dexjoco/dexjoco_openpi_client/dexjoco_openpi_env.py`](dexjoco/dexjoco_openpi_client/dexjoco_openpi_env.py)
handles the action order conversion automatically. It also converts
rotation-vector actions into the quaternion pose representation used by the
DexJoCo environment.

OpenPI training uses `action_rotvec` as the action target. The raw `action`
field stores quaternion poses with 23 dimensions for single-arm tasks and 46
dimensions for bimanual tasks. `action_rotvec` stores the action layout with 22
dimensions for single-arm tasks and 44 dimensions for bimanual tasks.

The recorded `state` field includes privileged environment state for replay,
such as object poses and table height. Policy training should use only robot
proprioception:

| Setup      | Policy State                                                              |
| ---------- | ------------------------------------------------------------------------- |
| Single-arm | First 23 dimensions: TCP pose and hand joints                             |
| Bimanual   | First 46 dimensions: right TCP pose, left TCP pose, right hand, left hand |

Privileged environment fields should be filtered out before training policy
models.

<a id="policy-training"></a>

## ⚙️ Policy Training

DexJoCo LeRobot datasets are available from
[`DexJoCo/DexJoCo-Datasets-LeRobot`](https://huggingface.co/datasets/DexJoCo/DexJoCo-Datasets-LeRobot).

OpenPI π0.5 training support lives under [`openpi/`](openpi). The OpenPI setup
covers two DexJoCo data regimes:

| Regime      | Randomization                                                                  |
| ----------- | ------------------------------------------------------------------------------ |
| `rand_obj`  | Object placement and table height randomization                                |
| `rand_full` | `rand_obj` plus third-person camera, lighting, and table texture randomization |

Training workflow:

1. Install the OpenPI environment with
   [`openpi/install.bash`](openpi/install.bash).
2. Place checkpoints and LeRobot datasets according to
   [`openpi/config.yaml`](openpi/config.yaml).
3. Convert the π0.5 base checkpoint for 44-dimensional bimanual actions with
   [`openpi/scripts/convert_to_action_dim_44_model.py`](openpi/scripts/convert_to_action_dim_44_model.py)
   when training bimanual tasks.
4. Compute normalization statistics with
   [`openpi/scripts/compute_norm_stats.py`](openpi/scripts/compute_norm_stats.py)
   or
   [`openpi/scripts/compute_norm_stats.bash`](openpi/scripts/compute_norm_stats.bash).
5. Launch multiple tmux training jobs with
   [`openpi/scripts/launch_tmux_train.py`](openpi/scripts/launch_tmux_train.py),
   or train a single policy with
   [`openpi/scripts/train.py`](openpi/scripts/train.py).

See [`openpi/README.md`](openpi/README.md) for command examples and checkpoint
layout details.

<a id="headless-rendering"></a>

## 🖥️ Headless Rendering

Headless environments use `policy_mode=True` and `render_mode="rgb_array"`:

```python
TaskConfig.get_environment(
    policy_mode=True,
    render_mode="rgb_array",
    ...
)
```

`policy_mode=True` exposes the policy action interface and disables the
teleoperation wrapper. `render_mode="rgb_array"` selects offscreen rendering for
policy evaluation and automated environment usage.

Interactive teleoperation collection uses `policy_mode=False` and the MuJoCo
viewer, so it does not require the headless configuration.

## 📄 License

DexJoCo-owned code in this repository is released under the
[`MIT License`](LICENSE).

Bundled third-party components and assets retain their separate license terms:

| Component                                                                | License Scope                                                        |
| ------------------------------------------------------------------------ | -------------------------------------------------------------------- |
| [`teleoperation/GeoRT`](teleoperation/GeoRT)                             | Upstream non-commercial GeoRT license                                |
| [`franka_emika_panda`](dexjoco/dexjoco/sim/envs/xmls/franka_emika_panda) | Apache-2.0                                                           |
| [`wonik_allegro`](dexjoco/dexjoco/sim/envs/xmls/wonik_allegro)           | BSD-2-Clause                                                         |
| [`openpi/`](openpi)                                                      | Apache License, Version 2.0, plus Gemma model terms where applicable |

## 📚 Citation

```bibtex
@misc{wang2026dexjocobenchmarktoolkittaskoriented,
      title={DexJoCo: A Benchmark and Toolkit for Task-Oriented Dexterous Manipulation on MuJoCo},
      author={Hanwen Wang and Weizhi Zhao and Xiangyu Wang and Siyuan Huang and He Lin and Boyuan Zheng and Rongtao Xu and Gang Wang and Yao Mu and He Wang and Lue Fan and Hongsheng Li and Zhaoxiang Zhang and Tieniu Tan},
      year={2026},
      eprint={2605.16257},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2605.16257},
}
```
